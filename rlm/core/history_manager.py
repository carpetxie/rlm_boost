"""
Message history management for RLM persistent mode.

Implements pruning strategies to keep message history bounded while preserving
critical information. This is essential for persistent mode where message
history would otherwise grow without bound across turns.

The key insight: for incremental computation, the model doesn't need to see
ALL prior iteration details — it needs:
1. The system prompt (always)
2. A summary of what was computed in prior turns
3. Recent iterations in full detail (for continuity)

This implements a "sliding window + summary" approach:
- Keep the system prompt
- Summarize old iterations into a compact form
- Keep the last N iterations in full
"""

from __future__ import annotations

import ast
import re
from typing import Any

from rlm.utils.parsing import REPL_VARS_PREFIX


class HistoryManager:
    """Manages message history with bounded growth for persistent RLM.

    Strategies:
    - 'sliding_window': Keep last N iterations, drop older ones
    - 'summarize': Keep last N iterations, summarize older ones into a compact message
    - 'token_budget': Keep messages until token budget exhausted, then prune oldest

    The summarize strategy is most novel: it maintains a running summary of
    what the model has computed, enabling true incremental reasoning. The model
    can reference the summary to know what's already been done without
    re-reading all prior messages.
    """

    def __init__(
        self,
        strategy: str = "summarize",
        max_recent_iterations: int = 3,
        max_messages: int = 30,
        estimated_token_budget: int = 50000,
    ):
        self.strategy = strategy
        self.max_recent_iterations = max_recent_iterations
        self.max_messages = max_messages
        self.estimated_token_budget = estimated_token_budget
        self._turn_summaries: list[str] = []

    def prune(
        self,
        message_history: list[dict[str, Any]],
        turn_number: int = 0,
    ) -> list[dict[str, Any]]:
        """Prune message history according to the configured strategy.

        Args:
            message_history: Full message history including system prompt
            turn_number: Current turn number (for multi-turn persistent mode)

        Returns:
            Pruned message history
        """
        if self.strategy == "sliding_window":
            return self._prune_sliding_window(message_history)
        elif self.strategy == "summarize":
            return self._prune_with_summary(message_history, turn_number)
        elif self.strategy == "token_budget":
            return self._prune_token_budget(message_history)
        else:
            return message_history

    def add_turn_summary(self, summary: str) -> None:
        """Record a summary of what was computed in a turn.

        Called at the end of each completion() to record what the model accomplished.
        This feeds into the summarize pruning strategy.
        """
        self._turn_summaries.append(summary)

    def get_turn_summaries(self) -> list[str]:
        """Get all recorded turn summaries."""
        return self._turn_summaries.copy()

    def _prune_sliding_window(self, message_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep system prompt + last N iteration pairs.

        Each RLM iteration produces 2 messages: assistant response + user execution result.
        We keep the first few messages (system prompt + metadata) and the last
        N*2 messages (recent iterations).
        """
        # Find where the system/setup messages end (first assistant response after setup)
        system_end = self._find_system_end(message_history)

        system_messages = message_history[:system_end]
        iteration_messages = message_history[system_end:]

        # Each iteration = ~2 messages (assistant + user/execution)
        max_iter_messages = self.max_recent_iterations * 2
        if len(iteration_messages) > max_iter_messages:
            iteration_messages = iteration_messages[-max_iter_messages:]

        return system_messages + iteration_messages

    def _prune_with_summary(
        self, message_history: list[dict[str, Any]], turn_number: int
    ) -> list[dict[str, Any]]:
        """Keep system prompt + turn summary + last N iterations.

        This is the most novel strategy: instead of dropping old messages,
        we replace them with a compact summary of what was computed.
        The summary tells the model:
        - What variables were created
        - What classifications/computations were done
        - What the intermediate results were

        This enables true incremental computation: the model knows its
        prior work without re-reading the full history.
        """
        system_end = self._find_system_end(message_history)
        system_messages = message_history[:system_end]
        iteration_messages = message_history[system_end:]

        max_iter_messages = self.max_recent_iterations * 2
        if len(iteration_messages) <= max_iter_messages:
            # No pruning needed
            return message_history

        # Split into old (to summarize) and recent (to keep)
        old_messages = iteration_messages[:-max_iter_messages]
        recent_messages = iteration_messages[-max_iter_messages:]

        # Build summary of old iterations
        summary = self._build_iteration_summary(old_messages, turn_number)

        # Insert summary as a prior model output, not user input.
        # This produces proper alternating roles at the seam:
        #   user(first_prompt) → assistant(summary) → user(continue) → assistant(recent) → ...
        # Previously, summary_message had role="user", which produced two consecutive
        # user messages and caused the model to treat its own prior computation as
        # user-supplied input (semantically incorrect for trust/reliability).
        summary_message = {
            "role": "assistant",
            "content": summary,  # model's prior computation — assistant role is correct
        }
        ack_message = {
            "role": "user",
            "content": "Continue with the current task using the state described above.",
        }

        return system_messages + [summary_message, ack_message] + recent_messages

    def _prune_token_budget(self, message_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep messages until estimated token budget is exhausted.

        Simple heuristic: ~4 chars per token. Drop oldest non-system messages first.
        """
        system_end = self._find_system_end(message_history)
        system_messages = message_history[:system_end]
        iteration_messages = message_history[system_end:]

        # Estimate tokens for system messages
        system_tokens = sum(len(m.get("content", "")) // 4 for m in system_messages)
        remaining_budget = self.estimated_token_budget - system_tokens

        # Keep messages from the end until budget exhausted
        kept = []
        for msg in reversed(iteration_messages):
            msg_tokens = len(msg.get("content", "")) // 4
            if remaining_budget - msg_tokens < 0 and kept:
                break
            remaining_budget -= msg_tokens
            kept.append(msg)

        kept.reverse()
        return system_messages + kept

    @staticmethod
    def _find_system_end(message_history: list[dict[str, Any]]) -> int:
        """Find where system/setup messages end.

        System messages are: system prompt, metadata (assistant), first user prompt.
        The iteration messages start after the first user prompt.
        """
        # Look for the pattern: system, assistant (metadata), user (first prompt)
        # Everything after is iteration messages
        user_count = 0
        for i, msg in enumerate(message_history):
            if msg.get("role") == "user":
                user_count += 1
                if user_count == 1:
                    # This is the first user prompt — iteration messages start after it
                    return i + 1
        return len(message_history)

    @staticmethod
    def _extract_assigned_names(code: str) -> set[str]:
        """Extract top-level assigned variable names from code using AST parsing.

        Uses ast.parse() instead of string heuristics to reliably identify
        simple name assignments (e.g. `x = ...`, `def f():`) while correctly
        ignoring augmented assignments, comparisons, f-strings with =, and
        subscript assignments that the old "=" heuristic incorrectly captured.

        Returns only non-underscore names at module scope.
        """
        names: set[str] = set()
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return names

        for node in ast.walk(tree):
            # Simple assignment: x = ..., x, y = ...
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and not target.id.startswith("_"):
                        names.add(target.id)
                    elif isinstance(target, ast.Tuple):
                        for elt in target.elts:
                            if isinstance(elt, ast.Name) and not elt.id.startswith("_"):
                                names.add(elt.id)
            # Annotated assignment: x: int = ...
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and not node.target.id.startswith("_"):
                    names.add(node.target.id)
            # Function definition: def f():
            elif isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                names.add(node.name)
            # Class definition: class C:
            elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                names.add(node.name)

        return names

    @staticmethod
    def _build_iteration_summary(old_messages: list[dict[str, Any]], turn_number: int) -> str:
        """Build a compact summary of old iteration messages.

        Extracts:
        - Code that was executed (abbreviated)
        - Variables created (via AST parsing for correctness)
        - Key outputs
        """
        code_blocks = []
        outputs = []
        variables_mentioned: set[str] = set()

        for msg in old_messages:
            content = msg.get("content", "")
            role = msg.get("role", "")

            if role == "user" and "Code executed:" in content:
                # Extract code block
                code_match = re.search(r"```python\n(.*?)```", content, re.DOTALL)
                if code_match:
                    code = code_match.group(1).strip()
                    # Use AST parsing for reliable variable name extraction
                    variables_mentioned |= HistoryManager._extract_assigned_names(code)
                    code_blocks.append(code[:200])  # Truncate long blocks

                # Extract REPL output (abbreviated)
                if "REPL output:" in content:
                    output_part = content.split("REPL output:")[-1].strip()
                    if output_part and output_part != "No output":
                        outputs.append(output_part[:150])

            elif role == "assistant":
                # Extract key reasoning points (first 150 chars)
                if content and len(content) > 10:
                    outputs.append(f"Model reasoning: {content[:150]}...")

        summary_parts = [
            f"**PRIOR COMPUTATION SUMMARY** (Turn {turn_number}, {len(old_messages)} messages summarized):"
        ]

        if variables_mentioned:
            summary_parts.append(
                f"Variables created/modified: {', '.join(sorted(variables_mentioned))}"
            )

        if code_blocks:
            summary_parts.append(
                f"Code executed ({len(code_blocks)} blocks): "
                + "; ".join(code_blocks[:3])
                + ("..." if len(code_blocks) > 3 else "")
            )

        if outputs:
            summary_parts.append("Key outputs: " + " | ".join(outputs[:3]))

        return "\n".join(summary_parts)

    def generate_turn_summary(
        self, message_history: list[dict[str, Any]], final_answer: str | None
    ) -> str:
        """Generate a summary of the current turn for future reference.

        This is called at the end of each completion() to produce a compact
        description of what was accomplished.
        """
        system_end = self._find_system_end(message_history)
        iteration_messages = message_history[system_end:]

        # Count iterations, code blocks, and extract variable names
        n_iterations = sum(1 for m in iteration_messages if m.get("role") == "assistant")
        variables = set()
        for msg in iteration_messages:
            content = msg.get("content", "")
            if REPL_VARS_PREFIX in content:
                var_match = re.search(rf"{re.escape(REPL_VARS_PREFIX)} \[(.*?)\]", content)
                if var_match:
                    for v in var_match.group(1).split(","):
                        v = v.strip().strip("'\"")
                        if v:
                            variables.add(v)

        summary = f"Turn completed in {n_iterations} iterations."
        if variables:
            summary += f" Variables: {', '.join(sorted(variables))}."
        if final_answer:
            summary += f" Answer: {final_answer[:200]}..."
        return summary
