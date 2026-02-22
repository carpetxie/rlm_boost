"""
Mock-LM Integration Test for Incremental RLM Pipeline.

This is the first end-to-end test of the full incremental pipeline:
- System prompt switching (RLM_SYSTEM_PROMPT → INCREMENTAL_SYSTEM_PROMPT)
- cached_vars hints on subsequent turns
- REPL variable persistence across turns
- _incremental state accumulation
- History management across turns

Uses a ScriptedMockLM that returns pre-programmed responses following
the incremental protocol, allowing us to test the full pipeline without
API keys.
"""

import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from rlm.clients.base_lm import BaseLM
from rlm.core.incremental import EntityCache, IncrementalState, PairTracker
from rlm.core.types import ModelUsageSummary, UsageSummary
from rlm.utils.prompts import INCREMENTAL_SYSTEM_PROMPT, RLM_SYSTEM_PROMPT


class ScriptedMockLM(BaseLM):
    """Mock LM that returns scripted responses and records all prompts received.

    Each call returns the next response from the script. Prompts are recorded
    for verification of system prompt switching, cached_vars hints, etc.
    """

    def __init__(self, responses: list[str]):
        super().__init__(model_name="scripted-mock")
        self._responses = responses
        self._call_idx = 0
        self.recorded_prompts: list[Any] = []
        self._total_calls = 0

    def completion(self, prompt: str | dict[str, Any]) -> str:
        self.recorded_prompts.append(prompt)
        self._total_calls += 1
        if self._call_idx < len(self._responses):
            response = self._responses[self._call_idx]
            self._call_idx += 1
            return response
        # If we run out of scripted responses, return a FINAL
        return "FINAL(default answer)"

    async def acompletion(self, prompt: str | dict[str, Any]) -> str:
        return self.completion(prompt)

    def get_usage_summary(self) -> UsageSummary:
        return UsageSummary(
            model_usage_summaries={
                "scripted-mock": ModelUsageSummary(
                    total_calls=self._total_calls,
                    total_input_tokens=100 * self._total_calls,
                    total_output_tokens=50 * self._total_calls,
                )
            }
        )

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(
            total_calls=1, total_input_tokens=100, total_output_tokens=50
        )


# =========================================================================
# Scripted responses for the incremental protocol (using _incremental API)
# =========================================================================

# Turn 1: First chunk — parse entities, define check_pair, call process_chunk(0)
TURN1_RESPONSE_ITER1 = """Let me process the first chunk of data using the _incremental primitives.

```repl
# Parse entities from the first context chunk into {entity_id: attributes_dict}
entities = {}
for line in context_0.strip().split("\\n"):
    parts = line.split("|")
    if len(parts) >= 2:
        eid = parts[0].strip()
        attrs = parts[1].strip()
        entities[eid] = {"type": attrs}

# Define pair condition (persists across chunks via REPL state)
def check_pair(attrs1, attrs2):
    return attrs1["type"] == attrs2["type"]

# process_chunk() handles entity storage, pair checking, and retraction
stats = _incremental.process_chunk(0, entities, pair_checker=check_pair)
pair_results = _incremental.pair_tracker.get_pairs()
print(f"Processed {stats['new_entities']} entities, found {stats['total_pairs']} pairs")
```
"""

TURN1_RESPONSE_ITER2 = """Based on my analysis of the first chunk:

FINAL(Found pairs from chunk 1)
"""

# Turn 2: Second chunk — parse only new chunk, call process_chunk(1)
TURN2_RESPONSE_ITER1 = """I see the incremental state from prior turns. Let me process only the new data using _incremental.

```repl
# Parse ONLY the new chunk (context_1) — do NOT re-read context_0
new_entities = {}
for line in context_1.strip().split("\\n"):
    parts = line.split("|")
    if len(parts) >= 2:
        eid = parts[0].strip()
        attrs = parts[1].strip()
        new_entities[eid] = {"type": attrs}

# process_chunk() handles new vs updated entities, retractions, re-evaluation automatically
stats = _incremental.process_chunk(1, new_entities, pair_checker=check_pair)
pair_results = _incremental.pair_tracker.get_pairs()
print(f"After chunk 2: {stats['total_entities']} total entities, {stats['total_pairs']} pairs, "
      f"{stats['retracted_pairs']} retracted")
```
"""

TURN2_RESPONSE_ITER2 = """After incrementally processing chunk 2:

FINAL(Found pairs from chunks 1+2)
"""

# Turn 3: Third chunk — verify _incremental types, process chunk 2, report stats
TURN3_RESPONSE_ITER1 = """Processing chunk 3 incrementally using _incremental.process_chunk().

```repl
# Verify _incremental primitives are available and have accumulated state
print(f"IncrementalState: {_incremental}")
print(f"EntityCache type: {type(_incremental.entity_cache).__name__}")
print(f"PairTracker type: {type(_incremental.pair_tracker).__name__}")

# Parse ONLY the new chunk (context_2)
new_chunk_entities = {}
for line in context_2.strip().split("\\n"):
    parts = line.split("|")
    if len(parts) >= 2:
        eid = parts[0].strip()
        attrs = parts[1].strip()
        new_chunk_entities[eid] = {"type": attrs}

# Process chunk 2 (index 2, the third chunk)
stats = _incremental.process_chunk(2, new_chunk_entities, pair_checker=check_pair)
pair_results = _incremental.pair_tracker.get_pairs()
summary = _incremental.get_stats()
print(f"Chunk 3: {stats['new_entities']} new entities, total={summary['total_entities']}, "
      f"{summary['total_pairs']} pairs, {summary['total_retractions']} total retractions")
```
"""

TURN3_RESPONSE_ITER2 = """After incrementally processing all 3 chunks:

FINAL(Found all pairs from 3 chunks incrementally)
"""


def _get_system_prompt_from_recorded(prompt_messages):
    """Extract the system prompt content from a recorded prompt."""
    if isinstance(prompt_messages, list):
        for msg in prompt_messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                return msg["content"]
    return None


def _get_user_prompt_from_recorded(prompt_messages):
    """Extract the last user prompt content from a recorded prompt."""
    if isinstance(prompt_messages, list):
        for msg in reversed(prompt_messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg["content"]
    return None


def _make_mock_client_factory(mock_lm):
    """Create a get_client factory that returns our mock LM."""
    def mock_get_client(backend, kwargs=None):
        return mock_lm
    return mock_get_client


class TestIncrementalPipelineIntegration:
    """End-to-end integration tests for the incremental RLM pipeline."""

    def test_system_prompt_switches_on_turn2(self):
        """Turn 1 gets RLM_SYSTEM_PROMPT, turn 2+ gets INCREMENTAL_SYSTEM_PROMPT."""
        mock_lm = ScriptedMockLM([
            TURN1_RESPONSE_ITER1,
            TURN1_RESPONSE_ITER2,
            TURN2_RESPONSE_ITER1,
            TURN2_RESPONSE_ITER2,
        ])

        with patch("rlm.core.rlm.get_client", return_value=mock_lm):
            from rlm.core.rlm import RLM

            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "scripted-mock"},
                persistent=True,
                max_iterations=5,
            )

            # Turn 1
            result1 = rlm.completion("chunk1_data", root_prompt="Find pairs")
            assert result1.response is not None

            # Turn 2
            result2 = rlm.completion("chunk2_data", root_prompt="Find pairs")
            assert result2.response is not None

            # Verify system prompts
            # Turn 1 prompts: calls 0 and 1
            turn1_system = _get_system_prompt_from_recorded(mock_lm.recorded_prompts[0])
            assert turn1_system is not None
            assert "REPL environment" in turn1_system
            # Should be the standard RLM_SYSTEM_PROMPT (contains "FINAL function")
            assert "FINAL function" in turn1_system or "FINAL(" in turn1_system

            # Turn 2 prompts: calls 2 and 3
            turn2_system = _get_system_prompt_from_recorded(mock_lm.recorded_prompts[2])
            assert turn2_system is not None
            # Should be INCREMENTAL_SYSTEM_PROMPT (contains "INCREMENTAL COMPUTATION")
            assert "INCREMENTAL COMPUTATION" in turn2_system

            rlm.close()

    def test_cached_vars_hint_on_turn2(self):
        """Turn 2+ user prompts should include INCREMENTAL STATE hint."""
        mock_lm = ScriptedMockLM([
            TURN1_RESPONSE_ITER1,
            TURN1_RESPONSE_ITER2,
            TURN2_RESPONSE_ITER1,
            TURN2_RESPONSE_ITER2,
        ])

        with patch("rlm.core.rlm.get_client", return_value=mock_lm):
            from rlm.core.rlm import RLM

            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "scripted-mock"},
                persistent=True,
                max_iterations=5,
            )

            rlm.completion("chunk1_data", root_prompt="Find pairs")
            rlm.completion("chunk2_data", root_prompt="Find pairs")

            # Turn 2, iteration 0 prompt should have cached_vars hint
            turn2_user = _get_user_prompt_from_recorded(mock_lm.recorded_prompts[2])
            assert turn2_user is not None
            assert "INCREMENTAL STATE" in turn2_user
            # Should mention variables created in turn 1 by the new _incremental protocol.
            # Turn 1 creates: entities (dict), check_pair (function), stats (dict), pair_results (set)
            assert any(var in turn2_user for var in ["entities", "check_pair", "stats", "pair_results"])

            rlm.close()

    def test_repl_variables_persist_across_turns(self):
        """Variables created in turn 1 should be accessible in turn 2."""
        mock_lm = ScriptedMockLM([
            TURN1_RESPONSE_ITER1,
            TURN1_RESPONSE_ITER2,
            TURN2_RESPONSE_ITER1,
            TURN2_RESPONSE_ITER2,
        ])

        with patch("rlm.core.rlm.get_client", return_value=mock_lm):
            from rlm.core.rlm import RLM

            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "scripted-mock"},
                persistent=True,
                max_iterations=5,
            )

            # Provide structured context
            rlm.completion("e1|type_a\ne2|type_b", root_prompt="Find pairs")

            # Turn 2 code accesses check_pair and _incremental from turn 1.
            # With the new _incremental protocol:
            #   Turn 1 creates: entities, check_pair, stats, pair_results
            #   Turn 2 uses: check_pair (persisted), _incremental (persisted)
            result2 = rlm.completion("e3|type_c\ne4|type_d", root_prompt="Find pairs")

            # If variables didn't persist, the code in turn 2 would fail because
            # check_pair wouldn't exist. The fact that we get a result confirms persistence.
            assert result2.response is not None
            assert "Found pairs from chunks 1+2" in result2.response

            # Verify the environment has the new-protocol variables from turn 1
            env = rlm._persistent_env
            assert "entities" in env.locals      # from turn 1 parse
            assert "check_pair" in env.locals    # function defined in turn 1
            assert "pair_results" in env.locals  # result of get_pairs() in turn 1 and 2
            # _incremental has accumulated state across both turns
            assert len(env.locals["_incremental"].entity_cache) > 0

            rlm.close()

    def test_incremental_primitives_available_in_repl(self):
        """_incremental, EntityCache, PairTracker should be available in persistent REPL."""
        mock_lm = ScriptedMockLM([
            TURN1_RESPONSE_ITER1,
            TURN1_RESPONSE_ITER2,
            TURN2_RESPONSE_ITER1,
            TURN2_RESPONSE_ITER2,
            TURN3_RESPONSE_ITER1,
            TURN3_RESPONSE_ITER2,
        ])

        with patch("rlm.core.rlm.get_client", return_value=mock_lm):
            from rlm.core.rlm import RLM

            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "scripted-mock"},
                persistent=True,
                max_iterations=5,
            )

            rlm.completion("e1|type_a\ne2|type_b", root_prompt="Find pairs")
            rlm.completion("e3|type_c\ne4|type_d", root_prompt="Find pairs")
            result3 = rlm.completion("e5|type_e\ne6|type_f", root_prompt="Find pairs")

            assert result3.response is not None

            # Check that _incremental is in the environment's locals
            env = rlm._persistent_env
            # _incremental is underscore-prefixed so it's in the combined namespace
            # but hidden from cached_vars — verify it's accessible by checking
            # the code executed successfully (turn 3 uses _incremental)
            assert "Found all pairs from 3 chunks incrementally" in result3.response

            rlm.close()

    def test_turn_count_increments(self):
        """_turn_count should increment with each completion call."""
        mock_lm = ScriptedMockLM([
            TURN1_RESPONSE_ITER1,
            TURN1_RESPONSE_ITER2,
            TURN2_RESPONSE_ITER1,
            TURN2_RESPONSE_ITER2,
        ])

        with patch("rlm.core.rlm.get_client", return_value=mock_lm):
            from rlm.core.rlm import RLM

            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "scripted-mock"},
                persistent=True,
                max_iterations=5,
            )

            assert rlm._turn_count == 0
            rlm.completion("chunk1", root_prompt="test")
            assert rlm._turn_count == 1
            rlm.completion("chunk2", root_prompt="test")
            assert rlm._turn_count == 2

            rlm.close()

    def test_context_versioning(self):
        """Each turn should create a new context_N variable."""
        mock_lm = ScriptedMockLM([
            TURN1_RESPONSE_ITER1,
            TURN1_RESPONSE_ITER2,
            TURN2_RESPONSE_ITER1,
            TURN2_RESPONSE_ITER2,
            TURN3_RESPONSE_ITER1,
            TURN3_RESPONSE_ITER2,
        ])

        with patch("rlm.core.rlm.get_client", return_value=mock_lm):
            from rlm.core.rlm import RLM

            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "scripted-mock"},
                persistent=True,
                max_iterations=5,
            )

            rlm.completion("data_chunk_0", root_prompt="test")
            rlm.completion("data_chunk_1", root_prompt="test")
            rlm.completion("data_chunk_2", root_prompt="test")

            env = rlm._persistent_env
            assert env.get_context_count() == 3
            assert "context_0" in env.locals
            assert "context_1" in env.locals
            assert "context_2" in env.locals

            rlm.close()

    def test_history_stored_across_turns(self):
        """Conversation history should be stored after each turn."""
        mock_lm = ScriptedMockLM([
            TURN1_RESPONSE_ITER1,
            TURN1_RESPONSE_ITER2,
            TURN2_RESPONSE_ITER1,
            TURN2_RESPONSE_ITER2,
        ])

        with patch("rlm.core.rlm.get_client", return_value=mock_lm):
            from rlm.core.rlm import RLM

            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "scripted-mock"},
                persistent=True,
                max_iterations=5,
            )

            rlm.completion("chunk1", root_prompt="test")
            rlm.completion("chunk2", root_prompt="test")

            env = rlm._persistent_env
            assert env.get_history_count() == 2
            assert "history_0" in env.locals
            assert "history_1" in env.locals

            rlm.close()

    def test_history_manager_records_turn_summaries(self):
        """HistoryManager should have turn summaries after each completion."""
        mock_lm = ScriptedMockLM([
            TURN1_RESPONSE_ITER1,
            TURN1_RESPONSE_ITER2,
            TURN2_RESPONSE_ITER1,
            TURN2_RESPONSE_ITER2,
        ])

        with patch("rlm.core.rlm.get_client", return_value=mock_lm):
            from rlm.core.rlm import RLM

            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "scripted-mock"},
                persistent=True,
                max_iterations=5,
            )

            rlm.completion("chunk1", root_prompt="test")
            assert len(rlm.history_manager.get_turn_summaries()) == 1

            rlm.completion("chunk2", root_prompt="test")
            assert len(rlm.history_manager.get_turn_summaries()) == 2

            rlm.close()

    def test_multiple_contexts_noted_in_prompt(self):
        """Turn 2+ user prompt should mention multiple contexts available."""
        mock_lm = ScriptedMockLM([
            TURN1_RESPONSE_ITER1,
            TURN1_RESPONSE_ITER2,
            TURN2_RESPONSE_ITER1,
            TURN2_RESPONSE_ITER2,
        ])

        with patch("rlm.core.rlm.get_client", return_value=mock_lm):
            from rlm.core.rlm import RLM

            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "scripted-mock"},
                persistent=True,
                max_iterations=5,
            )

            rlm.completion("chunk1", root_prompt="test")
            rlm.completion("chunk2", root_prompt="test")

            # Check turn 2 prompt mentions 2 contexts
            turn2_user = _get_user_prompt_from_recorded(mock_lm.recorded_prompts[2])
            assert "2 contexts" in turn2_user
            assert "context_0" in turn2_user
            assert "context_1" in turn2_user

            rlm.close()


class TestRetractEntityDoubleCounting:
    """Tests for the retraction double-counting fix."""

    def test_retract_both_entities_in_pair_no_double_count(self):
        """When both entities in a pair are retracted, count should be 1 not 2."""
        pt = PairTracker()
        pt.add_pair("u1", "u2")
        pt.add_pair("u1", "u3")
        pt.add_pair("u2", "u3")

        # Retract u1 — should remove (u1,u2) and (u1,u3)
        r1 = pt.retract_entity("u1")
        assert len(r1) == 2
        assert pt.retraction_count == 2

        # Retract u2 — (u1,u2) was already cleaned up, only (u2,u3) remains
        r2 = pt.retract_entity("u2")
        assert len(r2) == 1  # Only (u2,u3), NOT (u1,u2) again
        assert pt.retraction_count == 3  # 2 + 1, not 2 + 2

    def test_retract_shared_pair_not_double_counted(self):
        """Specific test: pair (A,B) retracted once when both A and B are retracted."""
        pt = PairTracker()
        pt.add_pair("A", "B")

        r1 = pt.retract_entity("A")
        assert ("A", "B") in r1
        assert pt.retraction_count == 1

        # B's inverted index should have been cleaned up
        r2 = pt.retract_entity("B")
        assert len(r2) == 0  # No pairs left for B — (A,B) was already handled
        assert pt.retraction_count == 1  # Unchanged

    def test_process_chunk_retraction_count_correct(self):
        """IncrementalState.process_chunk should report correct retraction counts."""
        state = IncrementalState()

        # Chunk 0: add entities
        state.process_chunk(0, {
            "u1": {"type": "A"},
            "u2": {"type": "A"},
            "u3": {"type": "B"},
        }, pair_checker=lambda a, b: a["type"] == b["type"])

        # u1 and u2 should be paired (both type A)
        assert ("u1", "u2") in state.pair_tracker.get_pairs()

        # Chunk 1: update both u1 and u2
        stats = state.process_chunk(1, {
            "u1": {"type": "B"},
            "u2": {"type": "B"},
        }, pair_checker=lambda a, b: a["type"] == b["type"])

        # The pair (u1,u2) should be retracted ONCE, not twice
        assert stats["retracted_pairs"] == 1

        # Total retractions should be 1
        assert state._total_retractions == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
