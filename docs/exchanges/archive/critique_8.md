# Critique — Iteration 8

STATUS: CONTINUE

---

## Overall Assessment

Iteration 7 delivered its stated deliverables: the actual RLM pipeline runs (v1/v2/v3), multi-seed update-rate confirmation, `REPL_VARS_PREFIX` extraction, and the failure-mode taxonomy. F1=0.54 with 100% execution compliance (v3) is the most significant empirical milestone to date. However, four issues now dominate: (1) the F1=0.54 result is computed against full-context gold using only 15K chars of context — without a matched-budget baseline, the number is uninterpretable as evidence of protocol quality; (2) Failure Mode C (`process_chunk` called 1+6+2=9 times across 3 turns) indicates the model re-executes the whole incremental pipeline multiple times per turn, and the deduplication guard proposed in the research log has not been implemented; (3) the weighted savings formula's 78/22 token proportions were refuted by the live experiment's own token measurements (13%/87%), yet the headline "~39% weighted token savings" persists in the research log; and (4) the "Exactly N" claim in the lazy retraction analysis is asserted without the supporting per-entity retraction data for Tasks 11/13. The paper's contribution is near-publishable on the simulation side — what it needs now is a clean, honest empirical result with a fair comparison baseline.

---

## Reflection on Prior Feedback

All Critique 7 items were resolved in code — dropping them permanently. The three structural defects in `live_api_experiment.py` were fixed by building `eval/rlm_pipeline_experiment.py`. `REPL_VARS_PREFIX` is now a shared constant. Role-ordering is correct. The no-op assertion is verified across 40 runs. These are genuine resolutions. I am not re-raising any of them.

One Critique 7 item I was right to flag but am changing my framing on: the "lazy retraction safety condition for exactly N tasks" concern. I raised it as an observation that Tasks 11/13 have more retractions than expected. I'm escalating it this iteration because the `lazy_retraction_analysis.md` makes a specific numerical claim (`~0%` bidirectional rate for "Exactly N" tasks) that has no supporting experiment — while the experiment to test it (per-entity analysis for Tasks 11/13) takes 15 minutes to run.

---

## Scores

| Criterion | Score | Delta | Comment |
|-----------|-------|-------|---------|
| Novelty | 7/10 | +0 | F1 progression curve (the "dynamic" demonstration) still missing — 8th iteration. Failure mode taxonomy is publishable and new. Retraction taxonomy and cost model are solid. σ-model adds modest value. |
| Technical Soundness | 7/10 | +1 | Live pipeline runs are now structurally correct. Deductions: (a) process_chunk called 9× in 3 turns (Mode C unfixed), (b) weighted savings formula uses wrong token proportions, (c) lazy retraction claim for Tasks 11/13 unverified. |
| Benchmark Performance | 6/10 | +1 | F1=0.54 is the first real measurement — a genuine milestone. Deduction: no matched-budget baseline exists, so the number cannot be interpreted as evidence of quality vs. a coverage-constrained ceiling. |
| Scalability | 6/10 | +0 | Growing prompt token cost per turn (1415→3539→5625) shows history pruning does not activate in a 3-turn run. At k=5+, pruning will trigger but its effect on F1 is unmeasured. |
| Research Maturity | 7/10 | +1 | Simulation + cost model + failure mode taxonomy + first F1 measurement = near-publishable. Gap: F1 progression curve + fair comparison baseline. |

---

## Architecture Review

### Critical Efficiency Defect: `process_chunk` Called 9× for 3 Chunks (Failure Mode C Unresolved)

From `results/streaming/rlm_pipeline_v3_results.json`:
```
Turn 0: chunks_processed 0 → 1  (1 call)
Turn 1: chunks_processed 1 → 7  (6 calls)
Turn 2: chunks_processed 7 → 9  (2 calls)
Total: 9 calls for 3 logical chunks
```

The RLM engine ran up to `max_iterations=6` inner iterations per `completion()` call (v3 used `max_iterations=6`). The model wrote the `process_chunk` call in its code block, and the engine re-executed it on every iteration within a turn. Because `IncrementalState` has no deduplication guard, each re-call treats all entities as "already existing" (they're in `entity_cache` from the first call within the same turn) and runs the full O(u·n) updated-entity sweep — where u equals the entire entity set for that chunk (since the second call sees them all as "updated"). This means the actual pair-check cost for Turn 1 is approximately 6× the claimed incremental savings, not 22% savings.

The researcher's proposed fix (add `_processed_chunk_indices: set[int]` to `IncrementalState`) is correct in design but needs one additional consideration: if the model calls `process_chunk(1, different_entities, ...)` on the second call (e.g., after correcting a parsing error), deduplication by index alone would silently drop the correction. The implementation should:

```python
# In IncrementalState.__init__:
self._processed_chunk_indices: dict[int, dict] = {}  # chunk_index -> cached_stats

# In process_chunk(), at the top:
if chunk_index in self._processed_chunk_indices:
    import warnings
    warnings.warn(
        f"process_chunk({chunk_index}) called more than once. "
        f"Returning cached stats. Re-processing requires reset().",
        stacklevel=2,
    )
    return self._processed_chunk_indices[chunk_index]
# ... rest of processing ...
# At the end, before returning stats:
self._processed_chunk_indices[chunk_index] = stats
return stats
```

This is a 10-line fix that converts Failure Mode C from "correct but up to 6× wasteful" to "correct and efficient." Until this is implemented, every reported pair-check savings figure for live pipeline runs is understated by a factor proportional to `max_iterations`.

### The Token Cost Paradox: History Accumulation Dominates at k=3

The live experiment's per-turn prompt token counts:
- Turn 1: 1,415 tokens
- Turn 2: 3,539 tokens
- Turn 3: 5,625 tokens

This is linear growth, not the flat-per-turn curve that "incremental savings" implies. The reason: with only 3 turns, `HistoryManager._prune_with_summary()` never triggers. The default `max_recent_iterations=3` means pruning activates only when `len(iteration_messages) > 6`. A 3-turn run with ≤2 messages per turn produces at most 6 messages — exactly at the threshold. Pruning is silent.

For k=5 chunks with `max_iterations=6`, the message count after Turn 3 could be:
- Turn 1: ~6 messages (6 REPL iterations × 2 messages each)
- Turn 2: +6 messages → 12 total
- Turn 3: +6 → 18 total

At 18 messages, the `summarize` strategy prunes down to `max_recent_iterations * 2 = 6` messages. The summary of the pruned 12 messages is a ~200-character string per code block. This compression means the model in Turn 4 may not correctly know which chunk indices have already been processed — it can only see the summary text, not the actual `_incremental` state variables by name.

**This is an untested failure mode**: If the model in Turn 4 re-parses entities from chunk 0 (because the summary mentions "processed chunk 0 entities" without knowing the `_incremental` state is already loaded), the `process_chunk` deduplication guard (when implemented) will correctly block the redundant call. But without the guard, the model could corrupt the state by re-submitting entities with stale accumulated attributes. The 5-chunk F1 progression experiment will naturally exercise this code path — it MUST be run to validate history pruning correctness.

### `process_chunk` Docstring Has Stale Cross-Reference

`rlm/core/incremental.py`, `IncrementalState.process_chunk()` docstring (lines 234-235):
```
Note: _find_system_end() assumes the first user message is the
system-setup boundary. This precondition must hold for correct pruning.
```

This note about `_find_system_end()` belongs in `HistoryManager._find_system_end()`, not in `IncrementalState.process_chunk()`. It is a copy-paste artifact that actively misleads readers trying to understand the incremental computation interface. Remove these two lines from `incremental.py`.

---

## Novelty Assessment

### What Is Genuinely Novel (Confirmed)

1. **Failure mode taxonomy for incremental LLM protocol execution** (Iteration 7, NEW). Three distinct failure modes (A: entity ID mismatch, B: FINAL_VAR premature, C: multi-call redundancy) characterized in a live pipeline. No prior RLM or RAG work characterizes LLM protocol execution compliance at this granularity. Directly motivates fine-tuning (Thrust 1).

2. **Non-monotonic retraction taxonomy with mechanistic confirmation**. 360× retraction range (44–15,824 at k=5) explained by condition semantics, not entity volume. Temporal asymmetry (1.3% vs. 10.4% bidirectional, 49× total retraction ratio) confirmed with mechanism. Genuine contribution to the incremental computation literature.

3. **σ-parameterized cost model with update-rate correction**. `savings(k, σ, p) ≈ 51.1(1-2.93/k) + 8.9σ(1+1.60/k) − 3.75%(p/0.05)`, with multi-seed robustness (±1.5pp). Actionable for practitioners. R²=0.936.

4. **Lazy retraction safety condition (monotonicity criterion)**. The "before DATE" vs "after DATE" asymmetry as a practitioner diagnostic for safe vs. unsafe lazy retraction is a new design principle.

### What Remains Missing for Full Novelty

**F1 progression curve — 8 iterations without this experiment.** The paper claims "Dynamic RLM" but has only a single static F1 measurement (0.54 after 3 chunks). The core "dynamic" claim requires showing F1 improves as context accumulates. This is achievable in one iteration and produces the key figure the paper needs.

**Fair comparison baseline is absent.** The F1=0.54 is measured against gold from the full 25K corpus, while the incremental model only sees 15K chars. This comparison has no informational value for the paper's central claim. A matched-budget non-incremental baseline would show: "given the same character budget, incremental processing recovers [X]% of pairs per chunk while maintaining persistent state." Without this comparison, the paper cannot claim the incremental approach outperforms or even matches a simple baseline.

**σ-model independence caveat still unacknowledged in the research log.** The 35 data points are all from the same OOLONG-Pairs corpus (N=231 entities, same 20 tasks, same user distribution). The F-test at p=0.025 assumes independence. Correlated observations can inflate significance. This should be stated as a limitation in any paper draft: "The F-test assumes independence across data points; our 35 points share a common entity pool and may be correlated, potentially overstating significance."

---

## Experiment Critique

### The Fundamental Measurement Problem: F1=0.54 Is Uninterpretable Without a Baseline

The v3 setup:
- Context: 5,000 chars/chunk × 3 chunks = 15,000 chars total
- Entities seen: 97/231 (42% coverage)
- Gold: 8,001 pairs from full 25,000 chars corpus
- F1: 0.54

This says: "with 60% of the corpus, we recover 43% of pairs." This is expected behavior from coverage-bounded recall — it tells us nothing about whether the incremental protocol is adding value over simpler alternatives. The 27% false positive rate (precision=0.73) is informative: some user IDs from the truncated plain context don't appear in the full labeled context gold standard, creating structural FPs unrelated to the protocol.

**The correct experimental structure** is:

| Condition | Context | Expected F1 | Purpose |
|-----------|---------|-------------|---------|
| Incremental RLM (k=5, 5K chars/chunk) | 25K total | ? (≥0.54) | Main result |
| Non-incremental RLM (1 turn, 25K chars) | 25K total | ~0.77 (from Exp 1) | Oracle ceiling |
| Non-incremental RLM (1 turn, 5K chars) | 5K only | ~0.2 (estimated) | Matched-budget baseline |
| Incremental RLM at each chunk (k=1..5) | 5K, 10K, 15K, 20K, 25K | Monotone increase? | Dynamic claim evidence |

The paper's core claim becomes: "incremental RLM achieves [X]% of the oracle F1 using [Y]% of the oracle's single-turn cost, by processing 5K chars/turn instead of 25K chars." This requires running all four conditions. The infrastructure for all four exists already.

### Weighted Savings Formula Is Based on Refuted Token Proportions

Research Log, Experiment 8 (weighted savings formula):
> `weighted_savings = 0.78 × entity_parse_savings + 0.22 × pair_check_savings`
> "token fractions from Experiment 1: sub-model accounts for 78% of total tokens"

Research Log, Experiment 14 (live API token split):
> "Token split Turn1/Turn2+: 13%/87% — INVERTED from 78/22 assumption"

The 78/22 ratio is from a **single OOLONG-Pairs completion** using sub-model (gpt-4o-mini) vs. root-model (gpt-5) token distribution. The 13/87 ratio is the **per-turn token fraction** in a 3-turn multi-turn RLM session. These are different quantities measuring different things. They cannot be compared or substituted.

More importantly: the "savings" in the weighted formula are savings in *pair-check operations*, not in *LLM input tokens*. In the multi-turn setting, the LLM input tokens actually **grow** per turn (1415→3539→5625) because the history accumulates. There is no LLM token savings — there is only pair-check computation savings. The "~39% weighted token savings at k=5" headline therefore cannot be published as stated.

**Concrete action**: The research log's "Experiment 8" section should be corrected. Remove "weighted_savings" from any paper draft. Replace with:
- "Pair-check savings: 22% at k=5, 42% at k=10 (simulation)."
- "LLM token savings: requires multi-turn token proportion measurement under controlled conditions; preliminary data shows prompt tokens grow O(k) per turn due to history accumulation, partially offset by history pruning at k≥4."

### Lazy Retraction "Exactly N" Claim Lacks Supporting Data

`docs/lazy_retraction_analysis.md`, Section 5 table:
```
| "Exactly N" | Yes | Yes | ~0% (monotone count) | Use lazy for batch |
```

This claims "Exactly N" tasks have ~0% bidirectional retraction rate, making them safe for lazy retraction. But:
- Task 11 (asymmetric "exactly 1 entity" constraint) has **1,231 retractions at k=5**
- Task 13 (asymmetric "exactly 1 description" constraint) has **2,250 retractions at k=5**
- Task 4 (temporal "after DATE", classified as **unsafe**) has only **1,517 retractions at k=5**

Tasks 11 and 13 have MORE retractions than the "unsafe" Task 4, yet the document classifies them as safe. The per-entity retraction analysis (run for Tasks 5 and 7 in Experiment 13) was never run for Tasks 11 and 13. Without knowing the bidirectional rate for Tasks 11/13, the ~0% claim is asserted without evidence.

The mechanism matters: for "exactly N" constraints, an entity with N instances in chunk 1 passes the condition and forms pairs. In chunk 2, if the same entity appears with a new instance (N+1 total), the condition fails → retraction. If in chunk 3 the entity's classification is reconsidered (e.g., the counter is stored as accumulated attributes and re-passed to process_chunk as "updated"), it could oscillate. This needs empirical confirmation, not assumption.

---

## The One Big Thing

**Run the 5-chunk F1 progression experiment with a matched non-incremental baseline, and fix the `process_chunk` deduplication guard before running.**

The sequence:
1. Implement `process_chunk` deduplication (10 lines in `incremental.py`) — without this, the experiment's claimed savings are inflated by Failure Mode C.
2. Run incremental RLM (k=5, 5K chars/chunk, v3 code template prompt): snapshot F1 after each chunk. This produces the F1 progression curve.
3. Run non-incremental RLM (1 turn, same 5K chars, i.e., chunk 0 only): baseline F1 at matched first-chunk budget.
4. Run non-incremental RLM (1 turn, all 5 chunks concatenated = 25K chars): oracle F1.
5. Plot: F1 vs. context accumulated (incremental) alongside oracle and matched-budget points.

Expected output: a figure showing F1 grows monotonically with chunks in incremental mode, starting below matched baseline (Turn 1 = same budget) and converging toward oracle. If this curve materializes, it is the paper's central figure and directly supports the "Dynamic RLM" claim.

Cost: ~$10. Time: 2–3 hours. Infrastructure exists.

---

## Specific Experiments to Run

1. **5-chunk F1 progression + non-incremental baselines (mandatory, ~$10, 2–3 hrs)**:
   - Prerequisite: implement `process_chunk` deduplication guard (30 min, no API).
   - Incremental: modify `run_v3_experiment.py` to `num_chunks=5`, `max_chars=5000`. After each `rlm.completion()` call, snapshot `_incremental.pair_tracker.get_pairs()` and compute F1 vs. gold. Store F1(k=1), F1(k=2), F1(k=3), F1(k=4), F1(k=5).
   - Matched-budget baseline: `RLM(persistent=False)` with only the first 5K chars. Measure F1.
   - Oracle baseline: `RLM(persistent=False)` with all 25K chars concatenated. Measure F1.
   - Expected: F1 progression curve + comparison table showing incremental at k=5 approaches oracle.

2. **`process_chunk` deduplication guard (30 min, no API)**:
   - Add `_processed_chunk_indices: dict[int, dict]` to `IncrementalState.__init__`.
   - At top of `process_chunk()`, return cached stats if `chunk_index` already processed.
   - Add test: `test_process_chunk_deduplication_no_double_count` — call `process_chunk(0, ...)` twice, assert second call returns cached stats and `chunks_processed == 1`.
   - This converts Mode C from "correct but up to 6× wasteful" to "correct and efficient."

3. **Per-entity retraction analysis for Tasks 11 and 13 (15 min, no API)**:
   - Extend `eval/sigma_cost_model.py`'s `run_per_entity_retraction_analysis` to include `task_indices=[11, 13]` alongside tasks 5 and 7.
   - Report bidirectional retraction fraction. If >1%, update `docs/lazy_retraction_analysis.md` Section 5 table.
   - Update `docs/lazy_retraction_analysis.md` Section 4 to note: "Task 11/13 empirical bidirectional rate: [X]%."

4. **History pruning activation measurement (in the 5-chunk experiment, no extra API cost)**:
   - Add `HistoryManager.prune()` call counter to verify pruning triggers at Turn 4 with default settings.
   - Check whether F1 at chunk 4 drops relative to chunk 3 — a drop would indicate pruning is compressing state that the model needs.
   - If F1 drops at pruning: increase `max_recent_iterations` from 3 to 5 and re-run Turn 4+.

5. **Remove weighted savings headline from research log and paper draft (no experiment, documentation fix)**:
   - Replace "~39% weighted token savings at k=5" with "22% pair-check savings at k=5" throughout the research log.
   - Add note: "Mapping from pair-check savings to LLM token savings requires controlled measurement of per-turn token proportions in multi-turn settings; preliminary data shows prompt tokens grow O(k) per turn."

---

## Code Issues Found

1. **`IncrementalState.process_chunk()` lacks idempotency guard** (`rlm/core/incremental.py`): No protection against repeated calls with the same `chunk_index`. The v3 live experiment shows 9 calls for 3 chunks (1+6+2). Each redundant call runs the O(u·n) sweep with all entities marked as "updated" (since they're already in `entity_cache`). Fix: 10-line deduplication with cached stats return. See Architecture Review for full implementation spec.

2. **Stale cross-reference in `process_chunk()` docstring** (`rlm/core/incremental.py`, lines 234-235): The note about `_find_system_end()` belongs in `HistoryManager._find_system_end()`, not in `IncrementalState.process_chunk()`. Remove the two lines.

3. **Weighted savings formula uses incorrect token proportions** (Research Log Experiments 8 and 14): The 78/22 sub-model/root-model split is from a single-completion run; the multi-turn per-turn split is 13/87 and growing. The "~39% weighted savings" headline is methodologically unsound. The formula should be removed from the research log's headline claim and replaced with the simpler pair-check savings figures.

4. **`lazy_retraction_analysis.md` "Exactly N" claim unsupported** (`docs/lazy_retraction_analysis.md`, Section 5): Tasks 11 (1,231 retractions) and 13 (2,250 retractions) are classified as having ~0% bidirectional retraction without empirical data. Task 4 ("after DATE", classified unsafe) has fewer retractions (1,517) than Task 13. The claim must be empirically verified (15-min experiment).

5. **`run_v3_experiment.py` API key loading is fragile and path-dependent** (`eval/run_v3_experiment.py`, line 4): `open('.env').read().split('=',1)[1].strip()` reads the entire `.env` file and splits on the FIRST `=` character in the file. If any line comes before `OPENAI_API_KEY=` (e.g., comments, other vars), this reads the wrong value. The file must be run from the project root for `.env` to resolve correctly. Use the same pattern as `rlm_pipeline_experiment.py` (line-by-line iteration). Additionally, as a one-shot script, `run_v3_experiment.py` should be refactored into a proper CLI module rather than a bare script — bare scripts are hard to test and reproduce.

6. **`_build_iteration_summary` regex matches only ` ```python ``` ` blocks** (`rlm/core/history_manager.py`, line 261): `re.search(r"```python\n(.*?)```", content, re.DOTALL)` will miss ` ```repl ``` ` blocks. In the incremental pipeline, the model writes ` ```repl ``` ` blocks (per the system prompt). If `_build_iteration_summary` only extracts from ` ```python ``` ` blocks, the code summaries in pruned iterations will be empty for all incremental turns. Fix: change the regex to `r"```(?:python|repl)\n(.*?)```"`.

---

## Acknowledged Limitations

- All savings data are on N=231 entities from a single OOLONG-Pairs corpus. Cross-corpus validation is theoretically justified by the cost model's asymptotic form but unconfirmed empirically.
- The σ-model F-test at p=0.025 uses 35 correlated data points from the same corpus. Independence assumption is violated. Report with caveat.
- The "dynamic benchmark gap" (genuine temporal data) has been open for 8 iterations. The F1 progression experiment on artificially chunked OOLONG-Pairs is a pragmatic substitute but is not truly "dynamic" — reviewers at ML/NLP venues may require this distinction be stated clearly.
- The lazy retraction benefit at N=231 is marginal (saves ~19 retraction ops for Task 5 vs. full eager). The architectural contribution is the monotonicity safety criterion as a design principle, not the runtime savings at current scale.
- The failure mode taxonomy (Modes A, B, C) was demonstrated on one task (Task 1), one model (gpt-4o-mini), and three chunks. Generalization to other tasks, models, and chunk counts is assumed but untested.
