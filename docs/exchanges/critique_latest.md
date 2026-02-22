# Critique — Iteration 1

STATUS: CONTINUE

## Overall Assessment (2-3 sentences)

The RLM codebase is well-engineered with a clean architecture for recursive LM decomposition via REPL environments. The OOLONG-Pairs results (77.8% F1 vs 3.0% base) are genuinely impressive and under-reported. However, the research has a critical gap: **Thrust 2 (Dynamic/Incremental RLM) exists only as infrastructure (the `persistent` mode) with zero experimental evidence** — no benchmarks, no measurements of incremental advantage, and no comparison against the naive "re-read everything" baseline. The persistent REPL is plumbing, not yet science.

## Scores
| Criterion | Score | Delta | Comment |
|-----------|-------|-------|---------|
| Novelty | 3/10 | — | Persistent REPL is engineering; no evidence it saves computation vs re-reading. Dynamic metrics gap is well-identified but unfilled. |
| Technical Soundness | 6/10 | — | Core RLM loop is solid. Persistent mode has correctness issues (see below). Eval code works. |
| Benchmark Performance | 7/10 | — | 77.8% vs 3.0% on OOLONG-Pairs is strong. But only one benchmark has results. No OOLONG or S-NIAH results exist. |
| Scalability | 4/10 | — | Message history grows linearly with turns, no pruning. Socket-per-request model won't scale. No token budget tracking. |
| Research Maturity | 2/10 | — | Research log is empty ("No experiments run yet"). No hypotheses tested. Persistent mode untested beyond unit tests. |

## Architecture Review

### Strengths
1. **Clean separation of concerns**: The `BaseEnv` / `SupportsPersistence` protocol pattern is well-designed. Adding new environments is straightforward.
2. **Batched LM queries**: `llm_query_batched` is a meaningful optimization that demonstrates systems thinking.
3. **The eval harness** is functional and the scoring code for OOLONG-Pairs is correct (verified the `compute_gold_pairs` logic against the task descriptions).

### Weaknesses — What breaks first at scale

1. **Message history is unbounded**. In `rlm.py:274`, `message_history.extend(new_messages)` appends every iteration's full response + code output. In persistent mode across turns, this means the root LM sees *every prior turn's full message history*. For a 10-turn conversation with 5 iterations each, this is 50+ messages with up to 20K chars each in execution results. **This will exceed any model's context window within a few turns.**

2. **The `_default_answer` method (line 324-347) puts a final-answer request as an `assistant` message**, not a `user` message. This violates the expected alternating user/assistant pattern for most LLM APIs and may cause undefined behavior with some providers.

3. **`execute_code` in `local_repl.py` uses `exec()` with a merged globals+locals dict** (line 365-366: `combined = {**self.globals, **self.locals}; exec(code, combined, combined)`). This means every execution creates a full copy of all variables. For large contexts (131K chars), this is copying ~131K per code execution. Over 30 iterations with multiple code blocks, this is significant memory pressure.

4. **No timeout on code execution in LocalREPL**. A malicious or buggy LLM-generated code block could hang forever. The 300s socket timeout in `comms_utils.py` doesn't protect against infinite loops in `exec()`.

5. **Thread-safety of `_capture_output`** (local_repl.py:334-344) uses a lock but replaces `sys.stdout`/`sys.stderr` globally. If any other thread writes to stdout during execution, output will be captured incorrectly or lost.

## Novelty Assessment

### What's genuinely new
- **The RLM concept itself** (from the paper) — recursive LM calls via REPL is novel and demonstrably effective.
- **The dynamic metrics gap observation** is a genuine insight. No existing benchmark tests incremental context evolution.

### What's incremental
- The `persistent` mode is standard session management (keeping a Python namespace alive across calls). This is engineering, not research.
- Context versioning (`context_0`, `context_1`, ...) is a naming convention, not an architecture.

### What would make this more novel — the key gap

The thesis claims Dynamic/Incremental RLM avoids re-reading the whole context every turn. But **the current persistent mode doesn't actually do this**. Look at `rlm.py:215-231`:

```python
message_history = self._setup_prompt(prompt)  # Fresh system prompt every call
for i in range(self.max_iterations):
    current_prompt = message_history + [build_user_prompt(...)]
    iteration = self._completion_turn(prompt=current_prompt, ...)
```

Every `completion()` call builds a fresh `message_history` from scratch. The root LM **does not see** prior turns' reasoning — only `context_N` and `history_N` variables are available in the REPL, but the root LM's prompt doesn't contain prior conversation context. The model has to re-discover everything each time by executing code to read `history_0`, etc.

**This is not incremental computation — it's state accumulation.** The model still does O(n) work per turn to process all prior state. True incremental RLM would mean the model only processes the *delta* (new context) and can build on *cached intermediate results* without re-executing.

To make this publishable:
1. **Measure the actual token cost** of persistent mode vs. non-persistent (re-creating environment each time). If persistent mode doesn't reduce total tokens consumed by the root LM, the claim is hollow.
2. **Implement actual incremental computation** — e.g., if the model has already summarized context_0 into a variable `summary_0`, it shouldn't need to re-summarize it when context_1 arrives. This requires either (a) prompt engineering to tell the model what's already been computed, or (b) architectural changes to carry forward intermediate results in the LM prompt itself.

## Experiment Critique

### What exists
- OOLONG-Pairs: RLM (77.8% F1) vs base model (3.0%). **This is a strong result that should be front and center.**
- No OOLONG results, no S-NIAH results despite the eval code being ready.

### What's missing

1. **No cost analysis**. The RLM on Task 1 used 6 root calls + 168 sub-calls (252K input + 321K output tokens for sub-model alone). What does this cost? Is the 74.8 percentage-point improvement worth it economically? **Compute cost-per-F1-point** for both methods.

2. **No ablation studies**. What drives the improvement? Is it:
   - The REPL execution? (ablation: allow code execution but no sub-LM calls)
   - The sub-LM calls? (ablation: allow sub-LM calls but no code execution)
   - The iterative refinement? (ablation: single-turn RLM with 1 iteration)
   - The chunking strategy? (ablation: vary chunk sizes)

3. **No error analysis on the failures**. Tasks 3 (0.551), 11 (0.591), 13 (0.641), 19 (0.443) underperform. What's different about these tasks? Task 19 requires matching "at least two instances with location AND at least one instance with entity" against "exactly one description AND exactly one abbreviation" — this is a complex asymmetric condition. **Do the failures correlate with task complexity?**

4. **No dynamic context experiments**. The persistent mode has been built and unit-tested but never benchmarked. This is the core thesis.

## The One Big Thing

**Design and run a dynamic context benchmark.** This is the single highest-leverage action because it:
- Tests the actual thesis (Dynamic/Incremental RLM)
- Fills the identified "dynamic metrics gap" (itself a potential contribution)
- Gives persistent mode something real to demonstrate
- Differentiates from the static-context results in the original RLM paper

Concrete proposal — **"Streaming OOLONG-Pairs"**:
1. Take the OOLONG-Pairs corpus.
2. Instead of giving all context at once, stream it in 5 chunks (simulate new data arriving).
3. After each chunk, ask the model for its current answer.
4. Measure: (a) final F1, (b) F1 at each intermediate step, (c) total tokens consumed.
5. Compare: persistent RLM (reuses REPL state) vs. non-persistent RLM (fresh environment per chunk, full re-read).
6. If persistent mode achieves comparable F1 with fewer tokens, that's the incremental advantage. If not, that's an important negative result that should redirect Thrust 2.

## Specific Experiments to Run

1. **Complete the existing benchmarks**: Run OOLONG (static, already coded) and at least one S-NIAH length. This fills out the baseline numbers and takes minimal effort.

2. **Token cost analysis**: Add token counting to the existing OOLONG-Pairs results. Parse the `usage` field already saved in `results/oolong_pairs/rlm.json`. Compute total input/output tokens, approximate cost, and cost-per-F1-point. Code is trivial:
   ```python
   for r in results:
       usage = r['usage']['model_usage_summaries']
       total_input = sum(m['total_input_tokens'] for m in usage.values())
       total_output = sum(m['total_output_tokens'] for m in usage.values())
   ```

3. **Failure analysis on OOLONG-Pairs**: For tasks with F1 < 0.6, inspect the RLM logs. Did the model fail at classification, at pair enumeration, or at following instructions? Categorize failures.

4. **Streaming context benchmark** (described above): This is the priority experiment. Even a simplified version (2-3 chunks, 5 tasks) would generate the first data point for Thrust 2.

5. **Persistent vs non-persistent comparison**: Run the same multi-turn task with `persistent=True` vs `persistent=False`. Measure tokens and accuracy. This directly tests whether the engineering work on persistent mode has research value.

## Code Issues Found

### Bug: `_default_answer` uses wrong message role
In `rlm.py:329-332`:
```python
current_prompt = message_history + [
    {
        "role": "assistant",
        "content": "Please provide a final answer...",
    }
]
```
This appends an **assistant** message asking for a final answer, then calls `lm_handler.completion()` which would generate another assistant response. Most LLM APIs expect the last message before generation to be `user` or `system`, not `assistant`. This should be `"role": "user"`.

### Bug: `REPLResult` dataclass has mismatched field name
In `types.py:121-140`, the class is declared as a `@dataclass` with field `llm_calls`, but `__init__` accepts `rlm_calls` and assigns to `self.rlm_calls`. The `@dataclass` field `llm_calls` on line 126 is never used because the custom `__init__` overrides it. This is confusing and the field names should be consistent.

### Issue: Output truncation hides critical information
In `parsing.py:87-91`:
```python
if len(result) > max_character_length:
    result = result[:max_character_length] + f"... + [{len(result) - max_character_length} chars...]"
```
The default `max_character_length=20000` truncates execution results. For large context operations (e.g., printing all pairs), the model may never see its own output. Consider logging the full output while truncating only the prompt portion.

### Issue: `find_final_answer` regex can match inside code blocks
The FINAL pattern `r"^\s*FINAL\((.*)\)\s*$"` with `re.MULTILINE | re.DOTALL` will match `FINAL(...)` that appears in a code block or execution output. If the model generates code that prints "FINAL(something)", the parser would incorrectly terminate. Consider requiring FINAL to appear outside of code blocks.

### Performance: `execute_code` copies all locals every call
In `local_repl.py:365-371`:
```python
combined = {**self.globals, **self.locals}
exec(code, combined, combined)
for key, value in combined.items():
    if key not in self.globals and not key.startswith("_"):
        self.locals[key] = value
```
This creates a full copy of globals+locals (potentially 100K+ chars of context), executes, then copies back. For 30 iterations with large state, this is quadratic in total state size.

### Missing: No `__import__` restriction despite blocking `eval`/`exec`
The safe builtins include `__import__: __import__` and `open: open` (lines 83-84). This means LLM-generated code can `import os; os.system('rm -rf /')` or read any file on the host. The sandbox is cosmetic — it blocks `eval` but allows unrestricted imports and file I/O.

## Acknowledged Limitations

- **No access to run benchmarks** (requires API keys and compute budget). The critique is based on existing results and code review.
- **Open models for Thrust 1** not yet explored — this is a separate workstream and doesn't block Thrust 2 progress.
- **Isolated environment persistence** (Docker, Modal, Prime) is hard to implement and not critical for initial research validation. Local-only persistence is sufficient for now.
