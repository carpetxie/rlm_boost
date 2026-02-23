# Critique — Iteration 14

STATUS: CONTINUE

---

## Overall Assessment

The research has reached a genuine inflection point: 5-run stability confirms F1=0.322±0.004 with P=1.0 and 100% compliance, the naive vs incremental head-to-head exists with real API numbers (84.3% token savings), and the at-risk fraction prediction is validated across 3 tasks. The paper's empirical skeleton is now present. However, two critical issues remain: (1) the headline comparison table (Table 2) conflates a *structurally unfair* naive baseline (which lacks IncrementalState entirely, thus failing at F1=0) with a fair efficiency comparison, and (2) the k=3 operating point (97.1% A/C, 1.30× tokens) — the paper's strongest result — rests on a single run.

---

## Reflection on Prior Feedback

**Resolved — not re-raising:**
- Library-level `monotone_attrs` in `process_chunk()` — implemented, tested (4 unit tests), validated across 5 runs. Done.
- Multi-run stability — 5 runs, σ=0.004, 100% compliance across 25 turns. Done.
- V3 token overhead distribution documented honestly (Run 1: 4.84×, Run 2: 2.42×). Done.
- Retraction accounting bug (cumulative sum → last value). Fixed. Done.
- "Bug fix" framing risk — researcher correctly frames as "correctness condition discovery." Done.
- Tasks 3 and 6 V4 — A/C=100% for both. At-risk prediction validated with correct ordering. Done.
- Condition B V4 with corrected system prompt — confirmed F1=0.0193. Done.
- `_permanent_retraction_count` in PairTracker — researcher's abstraction-level argument accepted. Done.

**Pushbacks accepted — not re-raising:**
- V3 Run 1 4.84× as "pathological V3 failure mode": agreed, V4 eliminates stochastic compliance entirely. Accepted.
- V4 Run 1 outlier (F1=0.3131 vs 0.3228): 55 pairs on 8001 is 0.7% — within reasonable LLM variance. Accepted as plausible; diagnostic still recommended below but not blocking.

---

## Scores

| Criterion | Score | Delta | Comment |
|-----------|-------|-------|---------|
| Novelty | 7/10 | 0 | P=1.0 protocol, monotone accumulation correctness condition, at-risk fraction predictor, retraction taxonomy — independently publishable. Score holds; see novelty section for what would push to 8. |
| Technical Soundness | 8/10 | +1 | 5-run stability, retraction accounting fix, 100% compliance across 25 turns, library-level monotone merge with 4 unit tests, `process_chunk()` code is well-structured with proper deduplication. Upgrade from 7. |
| Benchmark Performance | 8/10 | 0 | A/C=93.7%±1.3pp (5 runs), P=1.0 everywhere, 84.3% token savings vs naive. Strong, but Table 2 fairness issue prevents upgrade. |
| Scalability | 6/10 | +1 | k-sensitivity data now exists (Table 4: k∈{3,5,7,10}). k=3 at 97.1% A/C is strongest operating point. Upgrade from 5 — data exists, even if thin (single run at k=3). |
| Research Maturity | 7/10 | +1 | Paper-ready tables exist (`paper_summary_tables.py`). 5-run stability. Cross-task validation. The paper skeleton is present. Needs the Table 2 fairness fix and k=3 stability before submission. Upgrade from 6. |

---

## Architecture Review

### The `process_chunk()` Implementation Is Sound

The monotone merge logic (lines 331-358 of `incremental.py`) is correctly implemented:
1. Reads cached attrs BEFORE updating EntityCache — correct ordering
2. For each monotone attr: preserves truthy cached values when new value is falsy — correct merge
3. If ALL monotone attrs unchanged → `is_noop_update = True` → entity excluded from `updated_ids` — correct optimization
4. Retraction loop, new×existing, new×new, retracted pair re-evaluation, and updated×all sweep are properly ordered with deduplication via `checked_in_updated_sweep` set

**One subtle safety invariant is undocumented**: The `is_noop_update` optimization skips `updated_ids` when only monotone attrs change. This is correct ONLY if `pair_checker` depends exclusively on the declared monotone attributes. If a future pair_checker reads non-monotone attrs (e.g., `attrs.get("label_count")`), the optimization silently produces incorrect results — the entity's non-monotone attrs changed but retraction was skipped. Add a docstring warning to `process_chunk()`:

```python
# SAFETY INVARIANT: When monotone_attrs is provided, pair_checker must depend
# exclusively on the declared monotone attributes. If pair_checker reads other
# entity attributes, set monotone_attrs=None to preserve correct retraction.
```

### Token Variance Is a Deployment Caveat

Across 5 V4 runs at k=5, input tokens range from 18K to 67K (3.7× range). F1 is stable but cost is not. This is worth one sentence in limitations: "Per-run LLM token cost varies ~3.7× due to stochastic model iteration behavior, though output quality (F1, precision) is deterministic."

---

## Novelty Assessment

### What's Genuinely Novel (Strong)

1. **Monotone accumulation as a correctness condition for streaming LLM computation**: The deepest contribution. Prior incremental systems (CQL, incremental view maintenance) don't face LLM-specific stochastic attribute assignment across chunks. The insight that "once qualifying, always qualifying" must be enforced at the library level (not prompt level) is non-obvious and empirically validated (+27-44pp A/C improvement).

2. **At-risk fraction as a deployment diagnostic**: Predicting monotone fix impact from corpus statistics before running experiments. Validated across 3 tasks with correct ordering (Task 6 > Task 3 > Task 1). Practical tool.

3. **P=1.0 as a protocol invariant**: Zero false positives across 5 runs × 5 turns × 3 tasks = 75 multi-turn measurements. Perfect precision is a structural guarantee.

4. **Library vs template-level fix as a general design principle**: The V3→V4 upgrade (template to library) is a reusable insight for any LLM-in-the-loop system: push invariants into the runtime, not the prompt.

### What Would Push Novelty to 8/10

**Formalize the retraction taxonomy into predictive bounds.** Currently the taxonomy is descriptive (360× range across task types, from 44 to 15,824 retractions). To make it a publishable theoretical contribution, derive: "Given predicate class C ∈ {existential, cardinality, temporal-before, temporal-after} and entity selectivity σ, the expected retraction rate r(C, σ, k) is bounded by [formula]." Even order-of-magnitude bounds would turn descriptive data into a predictive tool, completing the analogy with the at-risk fraction predictor.

---

## 3rd-Party Clarity Test

### ⚠️ BLOCKING: Table 2 (Naive vs Incremental) Is a Structurally Unfair Comparison

A skeptical engineer reading Table 2 would immediately ask: **"Why does Naive achieve F1=0? Is the 'incremental' advantage just from having a structured framework, not from being incremental?"**

The answer is YES — the naive arm lacks `IncrementalState` entirely, so it has no entity-pair decomposition framework. It's not "incremental computation vs full recompute" — it's "structured framework vs no framework." The naive arm fails at the prompting/structure level, not the computation level.

**This is a strawman.** The paper claims efficiency savings, but the comparison conflates two independent benefits:
1. **Structural benefit**: Having `IncrementalState` for entity-pair decomposition (explains F1=0→0.32)
2. **Incremental benefit**: Processing only new chunks vs all chunks (explains token savings)

The fair efficiency comparison requires:

| Approach | Framework | Strategy | F1 | Tokens |
|----------|-----------|----------|-----|--------|
| Incremental RLM | IncrementalState | New chunk only | 0.3228 | 23,187 |
| **Full-Recompute RLM** | **IncrementalState** | **reset() + all chunks each turn** | **≈0.3424** | **~100K+** |
| Oracle (single-turn) | None | All context, 1 shot | 0.3424 | 24,184 |

The missing "Full-Recompute RLM" row uses the SAME IncrementalState framework but calls `_incremental.reset()` at each turn start, then `process_chunk(0, ...), ..., process_chunk(t, ...)` over all accumulated chunks. This isolates the incremental efficiency advantage from the structural advantage.

**Implementation**: ~20 lines in `label_aware_v4_experiment.py`:
```python
def run_condition_d_full_recompute(task_idx, num_chunks, ...):
    """Full-recompute using IncrementalState (reset each turn)."""
    # Same template as Condition A, but at each turn:
    #   _incremental.reset()
    #   for i in range(current_turn + 1):
    #       process_chunk(i, entities_from_chunk_i, check_pair, monotone_attrs={"qualifying"})
```

Without this row, a reviewer WILL say: "You're comparing apples to oranges. The naive baseline doesn't even have the same tools."

**The simulation data (Table 3) partially fills this gap** — it shows 58.5% pair-check savings with both sides using IncrementalState. But Table 2's live API numbers conflate two effects. The paper needs a live API full-recompute arm.

### ✅ PASS: Table 1 (Cross-Version Comparison)

Clear V2→V3→V4 progression. Same task, chunks, model, oracle C. Honest about stochastic V3 compliance. Reader instantly understands what changed and why.

### ✅ PASS: Table 3 (Cross-Task V2→V4 Improvement)

At-risk prediction validated with correct ordering. Clear mechanism. Fair comparison (all V4, same framework).

### ⚠️ CONCERN: Table 4 (k-Sensitivity) — k=3 Is a Single Run

k=3 is the paper's best operating point (97.1% A/C, 1.30× tokens, 100% compliance). This is ONE run. The 5-run stability at k=5 shows σ=1.3pp — we don't know k=3's variance. If k=3's true mean is 90%, the "97.1%" headline is misleading.

### ✅ PASS: Naive vs Incremental Simulation (Table 3 in paper_summary_tables.py)

Deterministic computation, structurally fair (same entity sets, checkers, chunks). Cleanest efficiency evidence.

---

## Experiment Critique

### Table 2 Needs a Full-Recompute Arm (Highest Priority)

Covered in 3rd-party clarity test above. This is the single experiment that transforms the paper from "nice framework" to "provably efficient framework."

### k=3 Stability (Second Highest Priority)

3 additional k=3 runs. Cost: ~$0.05. If mean A/C > 95%: k=3 is the paper's headline operating point ("97% of oracle at 30% token premium"). If mean A/C < 90%: k=5 remains canonical.

### The Outlier Diagnosis Is Low-Effort, High-Value

V4 Exp32 found 1,485 pairs vs 1,540 in the other 4 runs. The 55 missing pairs are fully determinable from the saved result JSONs (zero API cost):
1. Load pair sets from both files
2. Compute set difference
3. For each missing pair: which entities, which chunks?

This takes 15 minutes and adds one diagnostic paragraph to the paper: "The 3.6% outlier is explained by stochastic label extraction affecting N entities at chunk M boundaries."

### k=7/k=10 Compliance Degradation Is Under-Documented

k=7: 86% compliance, k=10: 90%. The paper says "small chunks sometimes produce near-empty contexts." This is testable from existing results: check which turns were non-compliant and how many entities their chunks contained. If there's a threshold (e.g., <5 entities → non-compliant), document it: "Compliance degrades when chunks contain fewer than N entities."

### Missing: Any Dynamic Context Experiment

After 14 iterations, ALL experiments use STATIC context chunked sequentially. The thesis motivation is dynamic context, but every experiment takes a fixed document and splits it. A skeptical reviewer: "This is just batched processing of a static document."

The update-rate experiment (Experiment 15) is the closest, but it's a simulation. A minimal live API proof-of-concept with genuine entity attribute updates mid-stream would strengthen the dynamic context claim.

**Not blocking for the current paper** (which can be scoped as "streaming static context"), but blocking for the broader thesis.

---

## The One Big Thing

**Implement a Full-Recompute RLM arm (Condition D) for a fair head-to-head efficiency comparison.**

The current Table 2 compares IncrementalState + incremental computation vs NO framework. This conflates structural and efficiency benefits. The paper's efficiency claim requires comparing incremental vs full-recompute WITHIN the same framework. This is ~20 lines of code: at each turn, `_incremental.reset()` then loop `process_chunk(i, ...)` for i=0..t. Run once on Task 1 k=5. Report F1 (expect ≈0.3424) and tokens (expect ~100K+, matching naive but with correct F1). The comparison becomes: "Same framework, same correctness, 84% fewer tokens."

---

## Specific Experiments to Run

In priority order:

1. **Full-Recompute RLM arm — Condition D ($0.05, ~1-2 hours code + run)**:
   - Add `run_condition_d_full_recompute()` to `label_aware_v4_experiment.py`
   - Each turn t: `_incremental.reset()`, then `process_chunk(i, entities_i, check_pair, monotone_attrs={"qualifying"})` for i=0..t
   - Same system prompt and REPL template as Condition A (just adds reset + loop)
   - Measure: F1 (expect ≈0.3424), input tokens, wall-clock
   - The comparison A (0.3228 F1, 23K tokens) vs D (≈0.3424 F1, ~100K+ tokens) isolates the efficiency advantage
   - If D achieves higher F1 than A (expected), the paper honestly states: "Incremental achieves 94% of full-recompute quality at 77% token savings"
   - **This is the paper's cleanest efficiency number**

2. **k=3 stability (3 runs, ~$0.05, 30 min)**:
   - `--multi-run 3 --k 3 --task 1`
   - Report mean±std of A/C and F1
   - If mean A/C > 95%: k=3 is the paper headline
   - If mean A/C < 90%: k=5 remains canonical

3. **Outlier diagnosis (0 API cost, 15 min)**:
   - Load Exp32 and MR1 pair sets from JSON
   - Compute set difference → identify 55 missing pairs
   - Report: which entities, which chunks, what pattern
   - One paragraph in paper

4. **k=7/10 non-compliance diagnosis (0 API cost, 15 min)**:
   - From existing k-sensitivity results: which turns non-compliant?
   - How many entities in those chunks?
   - Document minimum-entity threshold for compliance

5. **Update `paper_summary_tables.py` to use 5-run means (0 cost, 15 min)**:
   - Table 3 line 63: change Task 1 F1 from 0.3131 (single worst run) to 0.3209 (5-run mean)
   - Table 4: update k=5 token ratio from 4.23× (single run) to ~1.8× (5-run mean)
   - These corrections prevent reviewers from citing the worst single run

---

## Code Issues Found

1. **`process_chunk()` monotone optimization safety invariant undocumented (`incremental.py` lines 331-358)**:
   The no-op update optimization (skip `updated_ids` when only monotone attrs change) is correct ONLY if `pair_checker` depends exclusively on declared monotone attributes. If a future pair_checker reads non-monotone attrs, the optimization silently produces wrong results. Add a docstring warning: "SAFETY: pair_checker must depend only on attributes in monotone_attrs when this parameter is set. If pair_checker reads other attributes, set monotone_attrs=None."

2. **`paper_summary_tables.py` Table 3 uses V4 Run 1 F1 (0.3131) not 5-run mean (0.3209)**:
   Line 63 hardcodes `0.3131` for Task 1. With 5 runs available, use the mean (0.3209) and A/C of 93.7%. Using the single worst run understates the result.

3. **Table 4 k=5 token ratio (4.23×) is from the outlier run**:
   Line 86 of `paper_summary_tables.py` shows tok_ratio "4.23×" at k=5. This is from the k-sensitivity sweep (Exp32, the outlier run). The 5-run mean input tokens at k=5 ≈ 43K → ratio ≈ 1.8×. The paper should use multi-run means.

4. **`_processed_chunk_indices` grows without bound**:
   Stores full stats dict per chunk, never cleared (except by `reset()`). Low priority for paper scope (k≤10) but document for production use.

---

## Acknowledged Limitations

- All results on single model (gpt-4o-mini), single corpus (OOLONG-Pairs, N=231), single domain. Cross-generalization is explicitly out of scope.
- "Streaming" evaluation uses static context chunked sequentially. Genuine streaming (real-time, unknown length) would be stronger but is beyond current scope.
- Token cost variance (~3.7× across runs) means per-run cost estimates are unreliable. Report means with confidence intervals.
- k=3 operating point (97.1% A/C) is a single run — flagged as priority experiment above.
- The naive baseline (Table 2, F1=0) fails at the framework level, not the computation level. This conflates two benefits. Flagged as blocking issue above.
