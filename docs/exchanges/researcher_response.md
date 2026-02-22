# Researcher Response — Iteration 5

STATUS: CONTINUE

---

## Deliberation

### 1. INCREMENTAL_SYSTEM_PROMPT / `_incremental` abstraction conflict (CRITICAL BLOCKING)
- **Agree**: Fully. The prompt was instructing `entity_cache = {}` (plain dict) while the REPL injects `_incremental` (IncrementalState with retraction support). A real LLM following the old prompt would bypass retraction entirely.
- **Feasible**: Yes.
- **Impact**: High — gates the entire live API experiment.
- **Action**: Rewrote `INCREMENTAL_SYSTEM_PROMPT` in `prompts.py`. The protocol section now uses `_incremental.process_chunk(chunk_index, entities_dict, pair_checker)` as the sole interface. The model only implements `parse_entities()` and `check_pair()` — all state management (entity versioning, retraction, pair merging) is handled by `_incremental`.
- **Code written**: Yes — `rlm/utils/prompts.py`
- **Bonus finding**: Discovered a PRE-EXISTING BUG while fixing this: `_spawn_completion_context()` in `rlm.py` created `LocalREPL` without passing `persistent=self.persistent`. So `LocalREPL.setup()` never injected `_incremental`, `EntityCache`, or `PairTracker` — even when `RLM(persistent=True)`. The old mock-LM tests accidentally didn't catch this because old Turn 1 scripts didn't call `_incremental`. Fixed in `rlm.py` by adding `env_kwargs["persistent"] = self.persistent`. This is the most significant bug found this iteration.

### 2. Mock-LM tests don't catch the abstraction conflict
- **Agree**: `ScriptedMockLM` was scripted to accidentally avoid the conflict. The tests validated pipeline plumbing, not protocol compliance.
- **Action**: Rewrote TURN1_RESPONSE_ITER1, TURN2_RESPONSE_ITER1, TURN3_RESPONSE_ITER1 to use `_incremental.process_chunk()`. Updated test assertions for new protocol variables (`entities`, `check_pair`, `pair_results` instead of `entity_cache`, `processed_ids`). **12/12 tests pass** with new protocol AND the pre-existing bug fix.
- **Code written**: Yes — `tests/test_mock_lm_integration.py`

### 3. Real API experiment has been deferred — hard stop
- **Agree**: The critique's "dead-end warning" framing is appropriate.
- **Action**: Cannot run today (no API keys in this environment). BUT: the prompt fix in item 1 is complete, which was correctly identified as the mandatory prerequisite. The pipeline is now correctly configured and validated. **Next iteration: this is the first and only priority.**
- **Transparency note**: If the live experiment is again deferred in Iteration 6, I will explicitly reframe the contribution as "theoretical analysis + mock-LM pipeline validation" rather than "empirical system evaluation."

### 4. k=4 break-even not validated
- **Feasible**: Yes (5-minute simulation).
- **Impact**: Medium — validates the practitioner-facing threshold claim.
- **Action**: Ran `python eval/incremental_simulation.py --tasks 1,3,6,11,19 --num-chunks 4`.
  - Symmetric tasks (1, 3, 6): **15.2–15.8% savings** (σ-free model predicted 15.9% ✓ — within 1pp)
  - Asymmetric tasks (11, 19): **9.6–10.1% savings**
  - **Break-even at k≥4 confirmed** for all task types. All 5 tasks correct.
- **Code written**: Yes (results) — `results/streaming/incremental_4chunks.json`

### 5. 3-chunk temporal sweep missing
- **Action**: Ran `python eval/incremental_simulation.py --tasks 4,5,7,9,10 --num-chunks 3`.
  Results: 4.1–5.1% savings for all temporal tasks. Consistent with cost model. All correct.
- **Code written**: Yes (results) — `results/streaming/incremental_temporal_3chunks.json`

### 6. Fit σ-parameterized cost model (or report "k alone suffices")
- **Action**: Built `eval/sigma_cost_model.py` with 35 (task, k) datapoints, scipy curve_fit, and F-test for model comparison.
  - **σ = final_pairs / C(231, 2) = final_pairs / 26,565** — computed for all 11 tasks
  - **σ-free model**: savings(k) = 52.16*(1-2.84/k), R² = 0.919, RMSE = 3.81%
  - **σ-parameterized model**: savings(k,σ) = 51.06*(1-2.93/k) + 8.86*σ*(1+1.60/k), R² = 0.936, RMSE = 3.38%
  - **F(2,31) = 4.18, p = 0.025** — σ adds statistically significant predictive power
  - High-σ tasks (symmetric, σ~0.3–0.4) get 2–4pp more savings than low-σ tasks (strict, σ<0.01)
  - **Conclusion**: σ IS now in a fitted formula. Notation `savings(k, σ)` is now legitimate.
- **Code written**: Yes — `eval/sigma_cost_model.py`, `results/streaming/sigma_model_results.json`

### 7. Per-entity retraction frequency analysis (mechanistic validation of temporal asymmetry)
- **Action**: Added per-entity retraction tracking. Ran Tasks 5 ("before DATE") and 7 ("after DATE") at k=5. Analyzed distribution of per-entity retraction counts.

  **Task 5 ("before DATE")**:
  - 96.5% entities: never retracted
  - 0.9% retracted 1×, **2.6% retracted ≥2× (bidirectional)**
  - Max: 3 retractions per entity

  **Task 7 ("after DATE")**:
  - 84.0% entities: never retracted
  - 5.6% retracted 1×, **10.4% retracted ≥2× (bidirectional)**
  - Max: 4 retractions per entity

  - **Mechanism confirmed**: 4× more bidirectional retractions in "after" tasks
  - "Before DATE": entities become permanently invalid once a late-date instance arrives → monotonic invalidation → retract once and stabilize
  - "After DATE": entities oscillate as pre-cutoff and post-cutoff instances arrive in alternating chunks → up to 4 validity cycles per entity
- **Code written**: Yes — embedded in `eval/sigma_cost_model.py`

### 8. `execute_code` underscore suppression
- **Action**: Added `_INJECTED_NAMES = frozenset({"_incremental"})` module-level constant. Updated execute_code update loop to allow `_incremental` reassignment. The old silent-drop behavior is gone.
- **Code written**: Yes — `rlm/environments/local_repl.py`

### 9. `_build_iteration_summary` brittle `"="` heuristic
- **Action**: Replaced with `ast.parse()` walking `ast.Assign`, `ast.AnnAssign`, `ast.FunctionDef`, `ast.ClassDef` nodes at module scope. Ignores augmented assignments, comparisons, subscript assignments, f-strings. Falls back gracefully on `SyntaxError`.
- **Code written**: Yes — `rlm/core/history_manager.py`

### 10. `savings(k, σ)` notation was overclaiming
- **Resolved**: σ is now in a fitted formula with p=0.025. The notation `savings(k, σ)` is now legitimate. The R² gain is modest (+0.017) but statistically significant. Will report as: "savings are primarily determined by k; σ adds 2–4pp for high-density tasks."

### 11. Weighted savings methodology caveat (78/22 weights)
- **Agree**: The 78/22 assumption (sub-model token proportions in full pipeline apply to incremental) cannot be verified without instrumented API runs.
- **Action**: Added explicit caveat to research log and next steps. Will flag prominently in any paper draft.

---

## Code Changes

| File | Change | Impact |
|------|--------|--------|
| `rlm/utils/prompts.py` | Rewrote INCREMENTAL_SYSTEM_PROMPT to use `_incremental.process_chunk()` exclusively | Gates meaningful live experiment |
| `rlm/core/rlm.py` | Pass `persistent=self.persistent` to environment constructor (pre-existing bug) | `_incremental` was never injected before this fix |
| `rlm/environments/local_repl.py` | `_INJECTED_NAMES` allowlist for `_incremental` reassignment | Robustness |
| `rlm/core/history_manager.py` | `_build_iteration_summary` uses `ast.parse()` | Reliable variable extraction |
| `tests/test_mock_lm_integration.py` | Scripted responses use `_incremental.process_chunk()`, assertions updated | Tests now validate actual protocol |
| `eval/sigma_cost_model.py` | New: 35-point σ-model fit + F-test + per-entity retraction analysis | Two-factor formula derived; mechanism confirmed |

---

## Experiments Run

### Experiment 10: k=4 Break-Even Validation
- Command: `python eval/incremental_simulation.py --tasks 1,3,6,11,19 --num-chunks 4`
- Symmetric: 15.2–15.8% savings (predicted: 15.9% ✓)
- Asymmetric: 9.6–10.1% savings
- Break-even at k≥4 **confirmed** for all task types; 100% correctness

### Experiment 11: 3-Chunk Temporal Sweep
- Command: `python eval/incremental_simulation.py --tasks 4,5,7,9,10 --num-chunks 3`
- All temporal tasks: 4.1–5.1% savings. Consistent with cost model. 100% correctness.

### Experiment 12: σ-Parameterized Cost Model Fit (35 datapoints)
- `savings(k) = 52.16*(1-2.84/k)`, R²=0.919 (σ-free baseline)
- `savings(k,σ) = 51.06*(1-2.93/k) + 8.86*σ*(1+1.60/k)`, R²=0.936
- F(2,31)=4.18, p=0.025 — σ adds significant predictive power
- σ contributes 2–4pp at small k, narrows to <1pp at k≥10

### Experiment 13: Per-Entity Retraction Frequency (Mechanistic)
- Task 5 ("before DATE", k=5): 2.6% entities with ≥2 retractions (max 3)
- Task 7 ("after DATE", k=5): 10.4% entities with ≥2 retractions (max 4)
- **4× bidirectional retraction rate in "after DATE" tasks — hypothesis confirmed**

---

## Benchmark Results

| Benchmark | k=3 | k=4 | k=5 | k=10 |
|-----------|-----|-----|-----|------|
| Pair savings — symmetric (actual) | 10.0% | **15.5%** | 22.1% | 42.2% |
| Pair savings — asymmetric strict (actual) | 4.1% | **9.6%** | 16.7% | 38.8% |
| Pair savings — temporal "after" (actual) | 5.0% | — | 17.7% | 39.4% |
| Pair savings — temporal "before" (actual) | 4.1% | — | 16.7% | 38.8% |
| Cost model σ-free prediction (k=4) | — | 15.9% | — | — |
| Bidirectional retraction — Task 5 "before" | — | — | 2.6% entities | — |
| Bidirectional retraction — Task 7 "after" | — | — | **10.4% entities** | — |
| Integration tests (new _incremental protocol) | — | — | **12/12** | — |

---

## Research Log Updates
- Added Experiments 10–13 with full tables, methodology, and mechanistic interpretation
- Added Iteration 5 bug fixes section (4 bugs fixed, 1 pre-existing bug discovered and fixed)
- Updated cumulative summary table
- Revised "Next Steps" for Iteration 6: API experiment now unblocked and is the top priority
- Corrected σ notation: now reflects a two-factor fitted formula (R²=0.936, p=0.025)

---

## Pushbacks

**On "dead-end warning" for API deferral**: The critique is right. Five iterations without a live run is unusual for a systems paper. I've been transparent that the blocker is API key availability, not a research choice. The prompt fix this iteration was the mandatory prerequisite — it's now complete. Iteration 6 must produce live LLM data or explicitly adopt the "theoretical analysis" framing. I'm flagging this as the highest-priority item.

**On the σ model being "overclaimed" in prior iterations**: Agree — it was. Now corrected. The two-factor formula is fitted (not just a gap analysis). The R² gain is modest (+1.7pp) but statistically significant (p=0.025). I'll report both: the practitioner uses the σ-free model for quick planning (R²=0.919); the two-factor model provides precision when σ is known.

**On the per-entity analysis being "30-line"**: The critique underestimated complexity slightly — the analysis required per-entity retraction tracking which the simulation didn't natively support. Required either adding tracking to `IncrementalState` or wrapping the simulation. I chose to build a standalone analysis script. Total code: ~200 lines. Results are richer than the critique anticipated (full retraction histograms, not just max>1 binary).

---

## Next Experiments (Iteration 6)

1. **Live API experiment** (mandatory first priority): `RLM(persistent=True)` on OOLONG-Pairs Task 1 or 6 with gpt-4o-mini, 3 chunks. Measure: (a) protocol compliance — does Turn 2 code reference `_incremental.process_chunk()`? (b) correctness vs. gold pairs; (c) token savings if instrumented. Even failure (model doesn't follow protocol) is publishable data showing the compliance challenge.

2. **Protocol compliance metric from logs**: Parse Turn 2+ code for `_incremental` vs. `entity_cache = {}` patterns. Compute compliance rate from live experiment logs without extra API calls.

3. **Lazy retraction prototype**: "After DATE" tasks show 10.4% entities with multi-retractions consuming 2,151 events vs. 44 for "before DATE". Lazy retraction specifically helps "after DATE" heavy workloads. Implement `LazyIncrementalState` with query-time retraction. Measure cost vs. consistency tradeoff.

4. **Weighted savings caveat validation**: Live experiment (if instrumented) can measure actual token proportions in incremental vs. full-recompute paths, validating the 78/22 weight assumption.
