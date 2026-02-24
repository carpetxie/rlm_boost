# Critique — Iteration 11

STATUS: CONTINUE

## Overall Assessment

The project has reached a significant milestone: the full-corpus live API experiment (Exp 49) delivers F1=1.0 for both incremental and full-recompute at 84% token savings, 81% cost savings, and 65% wall-clock speedup. This is a strong headline result. The separated counterfactual (Exp 50) cleanly attributes retraction's value at 68% of F1 protection. The architecture is sound, `apply_edits()` is properly integrated with tests, and all 195 tests pass. However, this is still a **single-run, single-model, single-task** result at full corpus scale, and the paper needs robustness evidence before the claim "F1=1.0 at 84% savings" is credible to reviewers.

## Reflection on Prior Feedback

**Resolved — not re-raising:**
- Full-corpus live API run (Critique 14 "One Big Thing"). DONE spectacularly — F1=1.0.
- No-retraction counterfactual (Critique 14 #2). DONE — 620 invalid pairs at 96K, 10 edits.
- `apply_edits()` library method (Critique 14 #3). DONE — implemented with 7 tests.
- Separated counterfactual (Critique 14 implied). DONE — clean 3-way ablation.
- Full-corpus counterfactual at scale (Critique 14 #4 partial). DONE.
- Code issues (docstrings, Gemini test, chunk comment). All fixed.

**Pushbacks accepted — not re-raising:**
- Cross-model validation deferred. Accepted as documented limitation.
- `PairTracker._retracted` unbounded growth with `clear_retracted()`. Accepted.

## Scores

| Criterion | Score | Delta | Comment |
|-----------|-------|-------|---------|
| Novelty | 8/10 | +0.5 | Full-corpus live result makes the efficiency story compelling. Separated counterfactual is a genuine contribution (clean precision/recall attribution for retraction). Still single-domain. |
| Technical Soundness | 8.5/10 | +0 | `apply_edits()` is well-implemented with good tests. `process_chunk()` architecture remains solid. No new bugs found. BUT headline result is n=1 (see below). |
| Benchmark Performance | 8.5/10 | +2.0 | F1=1.0 at 96K chars with 84% savings is a dramatic improvement from F1=0.32 at 25K. This is now a publishable result. Downgrading from 9 because it's a single run. |
| Scalability | 6.5/10 | +0 | Everything is still gpt-4o-mini on OOLONG-Pairs Task 1 at full corpus. No cross-model, no cross-task at full corpus, no evidence of behavior at 200K+ chars. |
| Research Maturity | 8/10 | +0.5 | 11 documented contributions, paper-ready tables, fair comparison, clean ablation. Close to submission but needs multi-run stability at full corpus. |

## Architecture Review

### Core Architecture Remains Strong

Reviewed `rlm/core/incremental.py` end-to-end (646 lines). The code is clean and well-documented:
- `process_chunk()` with monotone merge, idempotency guard, deduplication — all correct.
- `apply_edits()` correctly handles the 3-phase pipeline (retract → re-evaluate → discover new).
- Partner cleanup in `retract_entity()` prevents double-counting.
- 7 `apply_edits` tests cover downgrade, upgrade, mixed, telemetry, no-op re-add, overlapping edits, and upgrade-only discovery.

### Chunk Boundary Behavior at 19K chars/chunk

At 19K chars/chunk, entity data can be split across chunk boundaries — some lines of user X in chunk i, remaining lines in chunk i+1. The `monotone_attrs={"qualifying"}` merge correctly preserves qualifying=True from earlier chunks, so this doesn't affect correctness. But the paper should note this: reviewers may ask "what happens at chunk boundaries?" The answer is "monotone merge handles it," which is actually a strength of the architecture worth highlighting.

### Full-Corpus Experiment Architecture

`eval/full_corpus_and_counterfactual.py` is well-structured. The `run_full_corpus_live` path correctly delegates to `run_condition_a_v4` and `run_condition_d_full_recompute`. The chunking logic (lines 462-466) handles remainder distribution to the last chunk.

## Novelty Assessment

### What's Genuinely Novel (Strong — Paper-Ready)

1. **IncrementalState as a reusable library** with entity cache + pair tracker + retraction: clean, tested, and now with `apply_edits()` for dynamic context.
2. **P=1.0 across ALL runs, all turns, all tasks, all scales**: Zero false positives from structured decomposition. This is the paper's most surprising and robust finding.
3. **F1=1.0 at 84% token savings**: The headline result. Equal quality at 5.2× lower cost.
4. **Separated counterfactual ablation**: Clean attribution — retraction protects precision (68%), new pair discovery protects recall (32%). This is a methodological contribution.
5. **Library-vs-template design principle**: V3→V4 showing that invariants belong in library code (deterministic compliance) vs LLM prompts (stochastic compliance). Broadly useful insight.

### What's Missing for Maximum Novelty

6. **No cross-task result at full corpus**: Tasks 3 and 6 are validated at 25K (F1≈0.32) but NOT at full corpus. A reviewer will ask: "Does F1=1.0 hold for other tasks, or is Task 1 uniquely easy?" This is the most impactful missing data point after multi-run stability.
7. **No multi-run stability at full corpus**: The 25K experiments have 5-run stability (σ=0.004). The 96K headline result is n=1. If F1=1.0 is fragile (drops to 0.85 on a bad run), the headline is misleading.

## 3rd-Party Clarity Test

### Table 11 (Full-Corpus Live, A vs D): ✅ PASSES — STRONG

A skeptical engineer reads: "Both conditions use the same IncrementalState framework. A processes each 19K-char chunk once. D resets and replays all chunks 1..k on each turn. F1=1.0 for both. A uses 38K tokens; D uses 236K tokens." Clear, fair, meaningful. The wall-clock comparison (174s vs 500s) adds practical impact.

**Minor improvement**: Add a "per-turn token breakdown" showing D's quadratic growth (Turn 5 = 107K) vs A's flat profile (Turn 5 = 4.6K). This makes the O(k²) vs O(k) claim visually obvious.

### Table 12 (Separated Counterfactual): ✅ PASSES

Three conditions are clearly defined and mutually exclusive. Attribution math is transparent. A skeptical reader can verify: F1(full) - F1(retract-only) = recall contribution; F1(retract-only) - F1(neither) = precision contribution.

### Table 13 (Full-Corpus Counterfactual): ✅ PASSES

Clear "with vs without" comparison. 620 invalid pairs at 96K vs 240 at 25K demonstrates retraction scales with corpus size.

### Headline Result (F1=1.0): ⚠️ PARTIAL PASS — n=1 Problem

**Blocking issue**: The headline claim "F1=1.0 at 84% token savings" is based on a SINGLE API run. At 25K, the researcher demonstrated σ=0.004 across 5 runs. But we don't know the variance at 96K. If one unlucky run produces F1=0.92 (due to a single non-compliant turn at 19K chars), the headline claim collapses.

The fix is trivial: run the full-corpus A condition 2 more times (cost: ~$0.02). If all 3 runs hit F1=1.0, the claim is solid. If variance appears, report it honestly — F1=0.98±0.02 at 84% savings is still excellent.

### Per-Turn Token Growth (D): ✅ PASSES BUT UNDEREXPLOITED

D's per-turn breakdown shows exactly the predicted O(k²) pattern:
- Turn 1: 37K tokens
- Turn 5: 107K tokens (2.9× growth)

This should be a FIGURE in the paper, not buried in a table. A log-scale plot of D's cumulative tokens vs A's flat profile is the single most visually compelling evidence for the incremental advantage.

## Experiment Critique

### What's Solid
1. **Full-corpus live A vs D**: Fair head-to-head, identical F1, 84% savings. Publication-grade.
2. **Separated counterfactual**: Clean 3-way ablation with transparent attribution.
3. **`apply_edits()` tests**: 7 tests covering edge cases including overlapping edits.
4. **Full experimental pipeline**: simulation → live API → counterfactual → ablation. Methodical.

### What's Missing (in priority order)

1. **Multi-run stability at full corpus (HIGH — ~$0.02, 30 min)**
   - Run full-corpus A condition 2-3 more times
   - Report mean ± std for F1, tokens, wall-clock
   - If F1=1.0 is stable (σ=0), state "F1=1.000 across N=3 runs"
   - If variance exists, report honestly: "F1=X±Y"

2. **Full-corpus Tasks 3 and 6 (MEDIUM — ~$0.10, 1 hr)**
   - Cross-task validation removes "Task 1 is uniquely easy" concern
   - Task 3 had compliance issues at 5K/chunk. Does 19K/chunk help or hurt?
   - Expected: similar savings pattern, possibly F1 < 1.0 (Task 3's selectivity is lower)

3. **Per-turn token comparison figure ($0, 15 min) — MEDIUM**
   - Plot A's per-turn input_tokens (flat ~6K) vs D's (growing to 107K)
   - This IS the visual proof of O(k) vs O(k²) — should be Figure 1 or 2 in the paper

4. **Cross-model spot check (~$0.50, 30 min) — LOW-MEDIUM**
   - Full-corpus Task 1, k=5 with gpt-4o (single run)
   - Even P=1.0 + F1=0.95 with similar savings would be sufficient

## The One Big Thing

**Run 2 more full-corpus A replicates to confirm F1=1.0 is stable (not a lucky n=1).**

Cost: ~$0.02. Time: ~30 minutes. This converts the headline from "we observed F1=1.0" (anecdotal) to "F1=1.000 ± 0.000 across 3 runs" (statistical). Without this, any reviewer can dismiss the result as cherry-picked.

The 25K experiments showed σ=0.004 across 5 runs, so stability is likely — but "likely" is not "demonstrated." At 19K chars/chunk, a single non-compliant turn could drop F1 below 1.0, and we need to know the probability of that happening.

## Specific Experiments to Run

1. **Multi-run full-corpus stability (~$0.02, 30 min) — HIGHEST**
   - Run `eval/full_corpus_and_counterfactual.py --full-corpus-live --task 1 --k 5` 2 more times
   - Record F1, tokens, wall-clock, compliance for each run
   - Report mean ± std
   - If any run has compliance failure, document and analyze

2. **Full-corpus Tasks 3 and 6 (~$0.10, 1 hr) — HIGH**
   - `--full-corpus-live --task 3 --k 5` and `--full-corpus-live --task 6 --k 5`
   - Cross-task validation removes "Task 1 is uniquely easy" concern
   - Expected: P=1.0, similar savings, F1 may vary by task selectivity

3. **Per-turn token comparison figure ($0, 15 min) — MEDIUM**
   - Extract per-turn `input_tokens` from the Exp 49 JSON for both A and D
   - Plot as a simple line chart (matplotlib)
   - Caption: "Incremental (A) maintains constant per-turn token usage while full-recompute (D) grows linearly with turn number"

4. **Cross-model spot check (~$0.50, 30 min) — LOW-MEDIUM**
   - Full-corpus Task 1, k=5 with gpt-4o (single run)
   - Addresses the "gpt-4o-mini-specific" concern

## Code Issues Found

1. **`apply_edits()` missing `_total_pair_checks` tracking**:
   In `process_chunk()`, every pair check increments `self._total_pair_checks`. But `apply_edits()` doesn't track pair checks at all — neither Phase 2 (re-evaluate retracted) nor Phase 3 (new pair discovery) increment `_total_pair_checks`. This means `get_stats()["total_pair_checks"]` underreports when `apply_edits()` is used. Fix:
   ```python
   # Add a pair_checks counter in apply_edits:
   pair_checks = 0
   # Phase 2: count each re-evaluation
   for p in all_retracted:
       pair_checks += 1
       ...
   # Phase 3: count each new pair check
   for eid in edited_ids:
       for other_id in self.entity_cache.get_ids():
           if other_id == eid: continue
           if self.pair_tracker.has_pair(eid, other_id): continue
           pair_checks += 1
           ...
   self._total_pair_checks += pair_checks
   # Add to return dict: "pair_checks": pair_checks
   ```
   This is a telemetry gap, not a correctness bug — but it would cause confusion if someone compares `get_stats()` between runs.

2. **`select_entities_to_edit()` iterates unsorted dicts** (carried over from Critique 14, not addressed):
   In `eval/dynamic_context_experiment.py`, `qualifying.items()` iteration depends on insertion order. For reproducibility across different Python environments:
   ```python
   for i, (uid, attrs) in enumerate(sorted(qualifying.items())):
   ```

3. **`compute_gold_pairs_with_edits()` doesn't use `_check_pair_condition()`** (carried over from Critique 14):
   Uses a simplified qualifying check instead of the real task condition. Fine for Task 1 but would produce wrong gold pairs for asymmetric tasks. Add a `# TODO` comment.

4. **No `--seed` or `temperature` parameter for live API experiments**:
   `full_corpus_and_counterfactual.py` doesn't expose temperature control. For multi-run stability testing, adding `temperature=0` as an option would distinguish model stochasticity from protocol fragility. This is a ~3-line change in the experiment runner.

## Acknowledged Limitations

- Single model (gpt-4o-mini), single corpus (OOLONG-Pairs). Accepted as proof-of-concept scope.
- All counterfactuals use synthetic hand-crafted edits. Real-world edit distributions would be less predictable.
- The "dynamic context" story is validated through counterfactual simulation, not a real multi-turn conversation with genuine context evolution. The paper should be clear about what's demonstrated (mechanism correctness) vs what's hypothesized (production applicability).
- Non-monotone tasks (Task 11) show F1=0.047. Documented scope boundary.
- The structural savings formula 1-2/(k+1) applies to pair-check counts, not total system tokens. The live API shows system token savings (84%) exceed structural pair-check savings (64%) due to D's prompt overhead — frame the formula as a LOWER BOUND on actual savings.
