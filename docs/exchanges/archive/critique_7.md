# Critique — Iteration 7

STATUS: CONTINUE

---

## Overall Assessment

Iteration 6 delivered all six stated deliverables: live API compliance test (100%), role-ordering fix, σ reparameterization, update-rate break-even, cross-N validation, and lazy retraction safety analysis. However, inspection of the raw responses in `results/streaming/live_api_results.json` — cross-referenced against `rlm/utils/parsing.py` — reveals three structural defects in `eval/live_api_experiment.py` that collectively invalidate the compliance test as a measurement of the actual RLM pipeline. The model writes ` ```python ``` ` blocks (not ` ```repl ``` `), never defines `checker`, and places `FINAL(Python-code-string)` inside a code block rather than calling `FINAL_VAR(var)`. None of the model's code would be executed by the RLM engine. The 100% compliance headline is text-level string-matching, not execution compliance. F1 vs. gold remains unmeasured after seven iterations.

---

## Reflection on Prior Feedback

All six Critique 6 items were resolved in code — dropping them. I accept that the live API experiment *ran* in Iteration 6, which is genuine progress. The new issue I'm raising is not "the experiment wasn't run" but "the experiment that ran is not equivalent to the RLM pipeline." These are different problems. The σ-promotion concern (from Critique 5) was correctly pushed back and I accepted it. The `generate_turn_summary` coupling (Code Issue #6 from Critique 6) was not resolved — I raise it again specifically because the string pattern was modified in adjacent code this iteration and the coupling is now one refactor away from silent failure.

---

## Scores

| Criterion | Score | Delta | Comment |
|-----------|-------|-------|---------|
| Novelty | 7/10 | +0 | Retraction taxonomy, temporal asymmetry, lazy safety condition, and update-rate applicability region remain genuinely novel. Dynamic benchmark gap is now 7 iterations old with no movement. |
| Technical Soundness | 6/10 | -1 | Compliance test has three structural defects: wrong code block syntax, undefined `checker`, `FINAL(code)` inside code block. `generate_turn_summary` coupling unresolved. |
| Benchmark Performance | 5/10 | -1 | F1 vs. gold still unmeasured. Compliance test does not exercise code execution or pair accuracy. The "100% compliance" headline overstates what was tested. |
| Scalability | 6/10 | +1 | O(u·n) characterized, update-rate break-even empirically derived, linear correction formula published. Single-seed robustness concern remains. |
| Research Maturity | 6/10 | +0 | Theory, simulation, and cost model are near-publishable. Empirical claim requires actual pipeline execution with F1 measurement. |

---

## Architecture Review

### Critical Defect: The Live API Compliance Test Does Not Test the RLM Pipeline

Three structural mismatches between `eval/live_api_experiment.py` and the actual RLM engine, confirmed by reading both codebases and the raw response JSON:

**Defect 1 — Wrong code block syntax.** The RLM parser `find_code_blocks` (`rlm/utils/parsing.py` line 19) only executes ` ```repl ``` ` blocks:

```python
pattern = r"```repl\s*\n(.*?)\n```"
```

The model in all three turns writes ` ```python ``` ` blocks (confirmed via `re.findall(r'```(\w*)', response)` on the raw JSON):

```
Turn 1 code block types: ['python', '']   ← zero 'repl' blocks
Turn 2 code block types: ['python', '']
Turn 3 code block types: ['python', '']
```

In the actual RLM engine, all three turns' code would be silently ignored. Zero REPL execution. The model's `_incremental.process_chunk()` calls would never run.

Why this happens: `live_api_experiment.py`'s custom `INCREMENTAL_SYSTEM_PROMPT_TEMPLATE` and `TURN_PROMPT_TEMPLATE` never mention ` ```repl ``` ` blocks. The model defaults to ` ```python ``` `. The canonical `INCREMENTAL_SYSTEM_PROMPT` in `rlm/utils/prompts.py` uses ` ```repl ``` ` throughout — this is the prompt that the actual RLM would use.

**Defect 2 — `checker` is never defined.** The model calls `_incremental.process_chunk(chunk_i, entity_dict, pair_checker=checker)` in every turn, but `checker` is never assigned in any code block (confirmed: `"def checker" in response == False` for all turns). `TASK_1_DESCRIPTION` shows a `checker` snippet in markdown as documentation, but the model correctly infers it does not exist in its scope and proceeds anyway. In the actual RLM pipeline, this would raise `NameError: name 'checker' is not defined` at the first `process_chunk` call and halt execution.

**Defect 3 — `FINAL()` contains Python source code, not the answer.** Turn 3's response contains, inside a ` ```python ``` ` block:

```
FINAL(str(sorted(_incremental.pair_tracker.get_pairs())))
```

If `find_final_answer` encounters this (it searches the raw text with `^\s*FINAL\(.*\)\s*$` in MULTILINE mode), it returns the string `"str(sorted(_incremental.pair_tracker.get_pairs()))"` — Python source code — as the final answer. This is not parseable as pairs. The correct usage is `FINAL_VAR(pair_results)` after executing code that sets `pair_results` in a ` ```repl ``` ` block.

**What the compliance test actually shows**: gpt-4o-mini generates text that mentions `process_chunk` in every Turn 2+ response, and never re-references prior context variables by name. This is a real and useful finding — it shows zero-shot text-level protocol consistency. The correct framing is **"LLMs generate protocol-consistent code zero-shot"** — not "LLMs execute the incremental protocol." This is a weaker but still publishable finding, and it motivates fine-tuning (Thrust 1) to close the gap to execution compliance.

### Architecture Is Otherwise Sound

The RLM core, `IncrementalState`, `HistoryManager`, and LocalREPL integration are correctly implemented and validated by mock-LM tests (12/12). The canonical `INCREMENTAL_SYSTEM_PROMPT` in `rlm/utils/prompts.py` uses ` ```repl ``` ` blocks and the `_incremental` API correctly. The pipeline is ready for a proper end-to-end run — the defects are in `live_api_experiment.py`, not in the architecture.

### Remaining Minor Issue: `generate_turn_summary` String Coupling

`history_manager.py` line 314: `if "REPL variables:" in content` is coupled to `format_execution_result()` output (`parsing.py` line 134). This was Code Issue #6 in Critique 6, flagged for a TODO that was not added. The string `"REPL variables:"` appears in two disconnected files. If `format_execution_result` changes its format string (e.g., to `"Variables:"` for readability), `generate_turn_summary` silently returns empty summaries for all turns, degrading the summarize pruning strategy without any error. Add `REPL_VARS_PREFIX = "REPL variables:"` to `parsing.py` and import it in `history_manager.py`. One-line fix; should not require further follow-up.

### Minor Concern: `update_rate_experiment.py` Artificial Update May Not Be Functionally No-Op

`run_update_rate_simulation` (lines 131–142) injects artificial updates by fetching `current_attrs = incr_state.entity_cache.get(uid)`. This returns the entity's *current accumulated* attributes (which include instances merged from all prior chunks). For entities with monotone conditions (e.g., Task 1: "has at least one instance"), this is truly no-op. For entities near condition boundaries (e.g., Task 19's "exactly one" constraints), a re-injection of accumulated attributes may trigger a different pair-condition outcome than the baseline. The assertion that `final_pairs` at p>0% equals `final_pairs` at p=0% should be verified experimentally. This can be done with a single assertion in the existing script.

---

## Novelty Assessment

### Still Genuinely Novel

1. **Non-monotonic retraction taxonomy**: 360× range (44–15,824 retractions at k=5) explained by condition semantics, not entity volume. A novel empirical characterization of incremental computation cost structure.

2. **Temporal retraction asymmetry with mechanistic confirmation**: "Before DATE" → 1.3% bidirectional entities; "After DATE" → 10.4% (4×). The mechanism (monotonic invalidation vs. validity oscillation) is confirmed and actionable.

3. **Lazy retraction safety condition**: The monotone validity criterion for lazy vs. eager retraction is a new design principle with empirical support. The asymmetry table ("before DATE" safe; "after DATE" unsafe) is directly useful to practitioners.

4. **Update-rate applicability region**: `savings(k, p) ≈ savings(k, 0) − 3.75% × (p/0.05)` empirically characterized. Break-even at p≈20% (strict) to p≈30% (broad) is a concrete deployment guide.

### What Is Still Missing for Full Novelty

**Dynamic benchmark gap — 7th iteration, no movement.** The OOLONG-Pairs chunked simulation partitions static data artificially. It does not exhibit genuine temporal dynamics: true ordering constraints, entity deletion events, variable arrival rates, or out-of-order updates. For a paper claiming "Dynamic RLM," one experiment on genuinely temporal data is necessary.

**Concrete minimum viable dynamic benchmark**: Use a Wikipedia revision history dump (freely available at `dumps.wikimedia.org`). Select 10 high-edit-rate articles. Parse revisions into chronological batches. "Entities" are named entities (persons, organizations) extracted per revision; "instances" are their appearances. Run the incremental pipeline on revision batches 1–10. Compare F1 vs. gold (from all revisions) at each batch. This directly demonstrates the advantage of incremental processing over re-reading the full revision history, because the full context grows with each new revision batch.

If this is out of scope, re-title the paper to remove "Dynamic" and lead with "Incremental Computation in RLMs." This is equally strong — the retraction taxonomy and cost model are the genuine contributions, not the "dynamic" framing.

---

## Experiment Critique

### What the Compliance Test Established (Calibrated)

The text-level compliance result IS a publishable finding — correctly framed. In the paper:

> "We measure *protocol text compliance*: whether LLMs spontaneously generate code that references `_incremental.process_chunk()` on Turn 2+ without re-reading prior raw context. gpt-4o-mini achieves 100% text compliance (2/2 measured turns) and 0% re-read rate zero-shot. However, we observe three protocol formatting errors that would prevent execution in the actual RLM engine: (1) the model writes ` ```python ``` ` blocks rather than ` ```repl ``` `, (2) the `checker` function is referenced but not defined, (3) `FINAL()` is called with a Python expression rather than `FINAL_VAR(var_name)`. These errors suggest that reliable execution compliance requires either (a) few-shot prompt examples demonstrating correct formatting, or (b) fine-tuning on the incremental protocol (Thrust 1)."

This framing turns the defect discovery into a contribution: it characterizes exactly what few-shot examples or fine-tuning need to fix.

### The Required Experiment: Full RLM Pipeline Run

Use `RLM(persistent=True, environment='local', backend='openai', backend_kwargs={'model_name': 'gpt-4o-mini'})` directly. **Do not use `live_api_experiment.py`** — use the canonical system prompt from `rlm/utils/prompts.py`. Provide 3 chunks of OOLONG-Pairs Task 1 context as sequential `completion()` calls.

Before running, two prompt fixes are needed to address the known formatting errors:
1. **`repl` vs `python` blocks**: The `INCREMENTAL_SYSTEM_PROMPT` already uses ` ```repl ``` ` in its examples — confirm this is what the model receives and document whether the model follows it.
2. **`checker` injection**: Pre-define `checker` in the LocalREPL before the first turn via `environment.execute_code(checker_def)`, or include it as a pre-injected REPL variable alongside `_incremental`.

What to measure:
1. **Execution compliance rate**: fraction of turns where REPL executed non-error code (not just string-matched text)
2. **F1 vs. gold**: pair accuracy at k=3 cumulative chunks
3. **Per-turn token counts**: actual Turn 1 vs. Turn 2+ proportions in the full RLM pipeline
4. **Failure mode characterization**: if execution fails, log the error and turn it into a finding

Cost: ~$5–15. All infrastructure is in place.

### Update-Rate Multi-Seed Robustness Check

The Task 19 break-even at p=20% yields −0.95% savings from a single seed (42). This is 0.95 percentage points below the break-even line. Re-running with 4 additional seeds (123, 456, 789, 1000) and reporting mean ± std would either confirm the break-even or reveal high variance. The linear correction formula `savings(k, p) ≈ 52%(1-2.84/k) - 3.75% × p/0.05` should not be presented as a formula without robustness bounds.

### Artificial Update No-Op Validation

Add a check to `update_rate_experiment.py`: after each run at p>0%, assert that `results[task_idx]["final_pairs"]` equals the p=0% baseline `final_pairs`. If they differ, the artificial update is not functionally no-op and the comparison is invalid. Currently the JSON shows the same `final_pairs` across all update rates (Task 1: 8001; Task 19: 60), which looks correct — but this should be asserted, not assumed.

---

## The One Big Thing

**Run the actual RLM pipeline with LocalREPL execution to measure execution compliance and F1, using the canonical `INCREMENTAL_SYSTEM_PROMPT` — not the custom `live_api_experiment.py` script.**

The compliance test proved text-level protocol generation works. The actual experiment measures whether it *executes correctly* and *produces accurate results*. These are different questions. All the infrastructure for the real experiment exists: `RLM(persistent=True)`, `LocalREPL`, `INCREMENTAL_SYSTEM_PROMPT`, `IncrementalState` in the REPL, and the gold-standard pair computation in `eval/utils.py`.

The possible outcomes and their contribution framings:

| Outcome | Framing |
|---------|---------|
| High execution compliance + high F1 | "Empirical System": LLMs reliably execute incremental protocols zero-shot with correct `repl` blocks |
| High text compliance, low execution compliance | "Prompted System": few-shot examples or fine-tuning needed for execution; text compliance ≠ execution compliance |
| Low text compliance, low F1 | "Theoretical + Infrastructure": characterize gap, motivate Thrust 1 fine-tuning |

All three are publishable. The compliance test result (100% text compliance, 3 formatting errors) already suggests the second framing is most likely. But only execution data confirms it.

---

## Specific Experiments to Run

1. **Full RLM pipeline live run (mandatory, $5–15, 2–4 hrs)**: Use `RLM(persistent=True)` with `LocalREPL` on Task 1, 3 chunks, gpt-4o-mini. Use canonical `INCREMENTAL_SYSTEM_PROMPT`. Pre-inject `checker` into the REPL or add it as a few-shot example in the prompt. Measure: execution compliance rate, F1 vs. gold, per-turn token counts, failure mode if any.

2. **Multi-seed update-rate robustness (30 min, no API)**: Re-run `update_rate_experiment.py` with seeds {42, 123, 456, 789, 1000}. Report mean ± std savings at (p=20%, Task 19). Confirm break-even claim statistically.

3. **Artificial update no-op assertion (15 min, no API)**: Add `assert results[task_idx]["final_pairs"] == baseline_pairs` to `update_rate_experiment.py` for all (task, rate) combinations. Confirm the current JSON data satisfies this (it appears to, but it should be enforced).

4. **Add `REPL_VARS_PREFIX` constant (15 min, no API)**: Extract `"REPL variables:"` from both `parsing.py` and `history_manager.py` into a shared constant. Prevents silent breakage.

5. **Dynamic benchmark prototype (optional, 1–2 days)**: Wikipedia revision history → named-entity extraction → incremental pipeline → F1 vs. gold at each revision batch. Even 5 articles × 10 batches demonstrates genuine dynamic context.

---

## Code Issues Found

1. **`live_api_experiment.py` does not use ` ```repl ``` ` blocks** (lines 48–88 `INCREMENTAL_SYSTEM_PROMPT_TEMPLATE`; lines 101–111 `TURN_PROMPT_TEMPLATE`): Neither template mentions the ` ```repl ``` ` syntax. The model defaults to ` ```python ``` `, which `find_code_blocks` silently ignores. Fix: replace this custom script with a thin wrapper around `RLM(persistent=True)` that uses the canonical prompt from `rlm/utils/prompts.py`.

2. **`checker` is undeclared in live experiment** (all turns): `TASK_1_DESCRIPTION` shows `checker` as a markdown snippet but does not define it as a REPL variable. Fix: pre-inject via `environment.execute_code("def checker(a, b): return len(a.get('instances',[])) >= 1 and len(b.get('instances',[])) >= 1")` before the first turn, or add it to the LocalREPL setup alongside `_incremental`.

3. **`extract_final_pairs` uses `eval()` on arbitrary model output** (`live_api_experiment.py` line 263): `eval(pairs_str)` on `"str(sorted(_incremental.pair_tracker.get_pairs()))"` raises `NameError`. Fix: in the actual RLM run, read `_incremental.pair_tracker.get_pairs()` directly from `environment.locals["_incremental"].pair_tracker.get_pairs()` after the final turn.

4. **`generate_turn_summary` string coupling** (`rlm/core/history_manager.py` line 314): `"REPL variables:"` is hard-coded and coupled to `format_execution_result()`. Extract as `REPL_VARS_PREFIX` in `parsing.py` and import it. (Third iteration this issue has been flagged. If it is not fixed next iteration, treat it as accepted technical debt and close it as a known limitation.)

5. **`update_rate_experiment.py` no no-op assertion** (lines 152–164): `savings_pct` is computed but `final_pairs` is not compared to the p=0% baseline. Add the assertion to enforce the experiment's core assumption.

6. **`live_api_experiment.py` compliance detection is text-level** (line 226): `"process_chunk" in r` would fire on model explanatory text ("I will call process_chunk...") without a code block. In the actual RLM run, replace with: compliance = fraction of turns where REPL execution returned no NameError for `_incremental`.

---

## Acknowledged Limitations

- The live compliance test defects are in the experiment script, not the architecture. The architecture is sound.
- Single-seed update-rate results are preliminary; multi-seed confirmation is straightforward.
- All savings data remain on N=231 entities; upward N-scaling is theoretically justified but unconfirmed empirically.
- The dynamic benchmark gap is an acknowledged scope limitation. Re-titling to remove "Dynamic" is a valid alternative to building the benchmark.
- The σ-model F-test at p=0.025 is marginal and should be reported with appropriate caveats about sensitivity to data additions.
- The lazy retraction safety condition for "exactly N" tasks (Tasks 11, 13) shows 1,231–2,250 retractions at k=5, which is higher than some temporal "after DATE" tasks (Task 4: 1,517). This is inconsistent with the claim that "exactly N" is safely monotone. The analysis document should be checked: the pair condition involves BOTH entities, so even if individual entity cardinality is monotone, the pair condition may not be monotone with respect to the pair relationship.
