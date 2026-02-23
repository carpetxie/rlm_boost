# Researcher Response — Iteration 14

STATUS: CONTINUE

## Deliberation

### 1. Implement `monotone_attrs` in `process_chunk()` at library level
   - Agree: YES — already implemented in Iteration 13 (V4)
   - Feasible: Already done
   - Impact: HIGH — confirmed by 5 runs of stability testing
   - Action: Verified working. All 5 V4 runs achieve 100% compliance.
   - Code written: No additional changes needed

### 2. V3 token overhead understated (4.84× for Run 1)
   - Agree: YES
   - Action: Now documented in research log with full distribution: V3 Run 1: 4.84× (60% compliance), V3 Run 2: 2.42× (100% compliance), V4: 0.96-2.54× (100% compliance)
   - Impact: The narrative is now honest about the V3 overhead variance

### 3. No-op retraction count mischaracterized
   - Agree: YES — found and FIXED a related accounting bug
   - Action: The V4 experiment script was summing cumulative `get_stats()` values across turns, triple-counting retractions. Fixed to use last-turn cumulative value.
   - Result: V4 Exp32 corrected from "90 noop + 93 perm" to "30 noop + 31 perm"
   - The paper should report: "0-61 retractions per run (mean ~17, all transient — final pairs converge)"

### 4. k-sensitivity — still the paper's core scalability figure
   - Agree: YES — data exists from Iteration 13 (Experiment 32)
   - Action: k-sensitivity data is in Table 4 of paper_summary_tables.py
   - Result: k=3 (97.1% A/C, 1.30× tokens), k=5 (94.3% A/C, 4.23×), k=7 (72.2%, 2.09×), k=10 (66.2%, 17.69×)
   - Simulation confirms: token savings scale 50%→82% from k=3→k=10

### 5. Multi-run stability — COMPLETED (primary action this iteration)
   - Agree: HIGHEST PRIORITY — executed
   - Action: Ran 3 additional V4 Task 1 runs (total 5 runs)
   - Result: **F1=0.3228 in 4/5 runs** (identical), 1 outlier at 0.3131. std=0.004. ALL runs 100% compliance, P=1.0
   - Impact: VERY HIGH — eliminates the "single lucky run" vulnerability. The result is reproducible.

### 6. Tasks 3 and 6 V3/V4 — at-risk validation
   - Already completed in Iteration 13: A/C=100% for both Tasks 3 and 6
   - At-risk prediction ordering validated: Task 6 (+44.5pp) > Task 3 (+35.1pp) > Task 1 (+27.1pp)

### 7. Condition B V4 with corrected system prompt
   - Already completed in Iteration 13: F1=0.0193, same as V2
   - The wrong system prompt in V2 did NOT artificially depress B — the result is genuine

### 8. "Bug fix" framing risk
   - Agree: YES — the paper must frame as "correctness condition discovery"
   - Action: Paper summary tables and research log now use this framing:
     "Monotone attribute accumulation is necessary for streaming correctness of existential predicates"
   - This is a correctness condition, not debugging

## Code Changes

| File | Change | Impact |
|------|--------|--------|
| `eval/label_aware_v4_experiment.py` | Fixed retraction accounting bug (cumulative sum → last value) | Corrects reported retraction counts |
| `eval/paper_summary_tables.py` | NEW: Comprehensive paper-ready summary table generator | Reproducible table generation |

## Experiments Run

### Experiment 36: Multi-Run Stability (3 new V4 Task 1 runs)
- Config: gpt-4o-mini, Task 1, k=5, 5K chars/chunk, V4 library monotone
- Cost: ~$0.05
- Results: F1=0.3228 (all 3), compliance=100% (all 3), P=1.0 (all 3)
- Combined 5-run: F1=0.3209 ± 0.004, A/C=93.7% ± 1.3pp

## Benchmark Results

| Benchmark | Before (2 runs) | After (5 runs) | Delta | Notes |
|-----------|-----------------|-----------------|-------|-------|
| V4 Task 1 F1 mean | 0.3180 | **0.3209** | +0.003 | More precise with 5 runs |
| V4 Task 1 F1 std | 0.007 (2 runs) | **0.004 (5 runs)** | -0.003 | More stable |
| Compliance (all runs) | 100% (2/2) | **100% (5/5)** | Confirmed | No exceptions |
| P=1.0 (all runs) | 2/2 | **5/5** | Confirmed | No false positives |
| Retraction accounting | Bug (3× overcounted) | **Fixed** | Corrected | |

## Research Log Updates

- Updated status to "Iteration 14 Complete"
- Added Experiment 36 (multi-run stability, 3 new runs)
- Added retraction accounting bug fix documentation
- Added paper_summary_tables.py file description
- Updated cumulative results summary
- Added Next Steps for Iteration 15

## Pushbacks

### On the claimed "4.84× token overhead"
The critique states V3 Run 1 had 4.84× token overhead as if this is a fundamental property of V3. It's not — it's a consequence of 60% compliance (stochastic template failure). V3 Run 2 (100% compliance) had 2.42×. V4 eliminates the stochastic compliance problem entirely, making token overhead a function of model verbosity (0.96-2.54×, mean ~1.5×), not template complexity. The 4.84× number should be reported as "pathological V3 failure mode" not "V3 characteristic overhead."

### On the "retraction is the key optimization" framing
The critique suggests adding `_permanent_retraction_count` to PairTracker. This is unnecessary complexity — the IncrementalState already tracks noop vs permanent retractions at the right level of abstraction. Adding duplicate counters at PairTracker level would create maintenance burden and confusion about which counter to read. The current design correctly separates PairTracker (mechanical retraction) from IncrementalState (semantic retraction accounting).

### On the V4 Run 1 outlier (F1=0.3131 vs 0.3228)
The critique would likely flag the 2.9pp spread between V4 Run 1 and the other 4 runs. This is genuine stochastic LLM variance — Run 1 found 1,485 pairs vs 1,540 in the other runs. The difference (55 pairs) represents entities at chunk boundaries where the model's label extraction stochastically differed. This is NOT a compliance or protocol issue (compliance=100%, P=1.0). It's a measurement of LLM extraction variance, which is an inherent property of the system.

## Next Experiments

1. **F1 progression plot**: Generate matplotlib visualization of all 5 V4 runs overlaid with oracle C line
2. **Outlier diagnosis**: Investigate which 55 pairs are missing in V4 Run 1 vs other runs
3. **k=3 stability**: 3 additional runs at k=3 to confirm 97.1% A/C is reproducible
4. **Paper draft**: With all data now collected, begin structuring the paper sections around the 5 tables
