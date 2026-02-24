# Critique — Iteration 14

STATUS: CONTINUE

## Overall Assessment

The project has matured substantially over 17 researcher iterations. The CRITICAL GAP (fair D vs A comparison) is fully resolved with strong, reproducible results: 77-86% token savings with 100% quality retention across 3 tasks. The dynamic context proof-of-concept (Iteration 17) validates the retraction mechanism on live entity edits — a meaningful step toward the "Dynamic RLM" thesis. However, the paper's claims still rest on a narrow empirical base: one model (gpt-4o-mini), one corpus (OOLONG-Pairs), and crucially, only 25K of 96K available chars — capping absolute F1 at ~0.34. The full-corpus run is now the single highest-leverage remaining experiment: it would simultaneously raise absolute F1, demonstrate scalability, and remove the most obvious reviewer objection ("you only find 32% of pairs").

## Reflection on Prior Feedback

**Resolved — not re-raising:**
- Fair D vs A comparison (CRITICAL GAP). Fully resolved across 3 tasks with 2 runs on Task 1. Done.
- Multi-run stability. 5 runs k=5 (σ=0.004), 4 runs k=3 (σ=0.000). Done.
- k-sensitivity sweep. k∈{3,5,7,10} live API data exists. Done.
- Cross-task V4 validation. Tasks 1, 3, 6 all complete with at-risk fraction validated. Done.
- Library-level monotone_attrs. Implemented, tested, eliminates stochastic compliance. Done.
- Non-monotone sanity check (Task 11). Backward compatibility confirmed. Done.
- Dynamic context experiment deferred across iterations 15-16. Now COMPLETED in Iteration 17. Done.
- Structural savings formula. Derived: 1-2/(k+1). Done.
- Retraction accounting bug. Fixed (cumulative→last). Done.

**Pushbacks accepted — not re-raising:**
- Cross-model validation deferred as documented limitation. Accepted.
- `PairTracker._retracted` unbounded growth. `clear_retracted()` exists, documented. Accepted.

## Scores

| Criterion | Score | Delta | Comment |
|-----------|-------|-------|---------|
| Novelty | 7.5/10 | +0.5 | Dynamic context experiment adds a genuine new capability demonstration. Retraction mechanism validated on live entity edits. But the experiment is narrow: 1 task, 1 model, only chunk-0 edits, hand-crafted balanced edits. |
| Technical Soundness | 8.5/10 | +0 | Condition D comparison remains the gold standard. Dynamic context simulation matches live API (91 retractions in both). Structural formula is clean. 187 tests passing. Code is well-engineered. |
| Benchmark Performance | 6.5/10 | -1.5 | The 77-86% savings claim is strong. But I'm downgrading because absolute F1 ≈ 0.32 on a task where oracle F1 = 1.0 (C Full) is a severe presentation problem. The researcher has 96K chars available and only uses 25K. A reviewer will see this as the system "working but barely." The full-corpus run is the fix. |
| Scalability | 6.5/10 | +0.5 | k-sensitivity + dynamic context add scalability dimensions. But all experiments use N=231 entities in 25K chars. The full-corpus run (N=231 in 96K chars) would test whether the framework scales with context window size. |
| Research Maturity | 7.5/10 | +0 | 8 documented contributions, paper-ready tables, 5-run stability, fair comparison, dynamic context POC. Close to submission but needs the full-corpus experiment to make F1 numbers presentable. |

## Architecture Review

### The Core Architecture Is Solid

`rlm/core/incremental.py` is clean, well-documented, and correctly implements the claimed algorithm. After reviewing the full file:

- Monotone merge logic (lines 346-364): correctly handles all four cases (truthy→truthy, truthy→falsy preserve, falsy→truthy genuine change, falsy→falsy).
- No-op update detection (lines 360-364): correctly skips `updated_ids` when all monotone attrs unchanged.
- Idempotency guard (lines 311-319): prevents Failure Mode C with O(1) cached return.
- `checked_in_updated_sweep` deduplication (lines 441-460): prevents inflated pair_checks.
- `retract_entity()` partner cleanup (lines 167-170): prevents double-counting.

**No new core library bugs found.** The implementation quality is high.

### Dynamic Context Edit Path Bypasses the Framework

In `eval/dynamic_context_experiment.py`, the edit turn (Turn 3, lines 100-136) bypasses `process_chunk()` entirely — it directly calls `entity_cache.add()` + `pair_tracker.retract_entity()` + manual re-evaluation loop. This means the edit path doesn't benefit from:
1. The monotone merge logic in `process_chunk()`
2. The idempotency guard
3. The `_total_retractions` counter (hence the telemetry gap noted in the researcher response)
4. The `checked_in_updated_sweep` deduplication

This is pragmatically fine for the POC, but creates an awkward gap between the claim ("our framework handles dynamic updates") and the implementation (the dynamic update code path is manual, not framework-level).

**Concrete suggestion** — add an `apply_edits()` method to `IncrementalState`:

```python
def apply_edits(
    self,
    edits: dict[str, dict[str, Any]],
    pair_checker: Any = None,
) -> dict[str, Any]:
    """Apply entity attribute edits and retract/re-evaluate affected pairs.

    For dynamic context scenarios where entity attributes change between turns
    (document edits, streaming corrections, etc.).

    Returns stats dict with: entities_edited, total_retracted, new_pairs, pairs_readded.
    """
```

This is ~30 lines, directly extracted from the experiment code. It makes the "dynamic context" claim architecturally honest — the framework itself supports edits, not just the experiment script.

### Token Variance in Condition D

D token counts vary 3× across runs (246K vs 80K for Task 1, identical F1). The structural formula resolves this cleanly — but Table 2c should include a "structural" column alongside empirical numbers to give the reader a variance-free anchor. The researcher is already doing this in Table 7; suggest merging the structural prediction INTO Table 2c rather than having a separate table.

## Novelty Assessment

### What's Genuinely Novel (Strong)

1. **IncrementalState as a reusable library** for LLM-program incremental computation. Entity cache + pair tracker + retraction is a clean abstraction.
2. **P=1.0 across all runs, all turns, all tasks** — zero false positives from structured decomposition. The paper's most surprising result.
3. **At-risk fraction as a predictive diagnostic** — validated ordering across 3 tasks.
4. **Library-vs-template design principle** — V3→V4 demonstrating that invariants belong in library code, not LLM prompts. Broadly useful insight.
5. **Monotone attribute accumulation as a correctness condition** — clean result with practical value.

### What's Incrementally Novel (Medium)

6. **Retraction mechanism for non-monotonic incremental computation** — novel in the LLM context, but prior art exists in database stream processing (RETE networks, incremental view maintenance). The paper should cite and differentiate.
7. **Structural savings formula** 1-2/(k+1) — correct but trivially derived. Present as a useful bound, not a standalone contribution.
8. **Dynamic context POC** — validates the mechanism works on live API. The superlinear retraction scaling (18.2/edit→78.1/edit) is expected combinatorial behavior, not a surprising finding. Frame accurately.

### What Would Increase Novelty

- **Full-corpus results** showing F1 >> 0.34 would make the efficiency story dramatically more compelling. Currently "we save 80% of tokens to get F1=0.32" is a hard sell; "we save 80% of tokens to get F1=0.90" is a strong paper.
- **No-retraction counterfactual** for the dynamic context experiment: what happens if you edit entities but DON'T retract? How many invalid pairs persist? This quantifies the retraction mechanism's VALUE, not just its existence.

## 3rd-Party Clarity Test

### Table 2c (D vs A vs C, Cross-Task): ✅ PASSES

A skeptical engineer reads: "Both D and A use the same framework. D resets and replays all chunks; A processes only the new one. F1 is identical across 4 experiments and 3 tasks. A saves 77-86% of tokens." Clear, fair, meaningful. Minor: note the D token variance (246K vs 80K) so readers don't think the 79.8% and 77.1% savings represent different capabilities.

### Table 3 (Cross-Task V2→V4 Improvement): ✅ PASSES

At-risk fraction ordering matches ΔA/C ordering. Clean falsifiable prediction, validated across 3 tasks. Unambiguous.

### Table 8 (Dynamic Context): ⚠️ PARTIAL PASS — Missing Counterfactual Baseline

**Blocking issue**: The table shows the retraction mechanism WORKS (retractions fire, P=1.0 maintained). But a skeptical engineer asks: **"What would happen WITHOUT retraction?"** If the answer is "nothing bad, because the edits barely affect the pair set," then the mechanism is correct but unimportant. If the answer is "precision drops from 1.0 to 0.7 because 30% of pairs become invalid," then the mechanism is essential.

Currently, the 5-edit experiment has pair delta = 0 (net unchanged). The 10-edit experiment has delta = -61 (from 496 to 435). But neither tells us how many INVALID pairs would persist without retraction. The no-retraction counterfactual is zero-cost (simulation only) and directly answers the value question.

**Suggested addition to Table 8**:

| Metric | With Retraction | Without Retraction |
|--------|----------------|-------------------|
| Invalid pairs remaining | 0 | X |
| Precision | 1.0 | Y |
| Correctness | ✓ | ✗ |

### Naive vs Incremental (Table 2b): ✅ PASSES

Correctly labeled as structural advantage. Not a strawman — properly separated from the fair D vs A efficiency comparison.

### Superlinear Retraction Scaling Claim: ⚠️ NEEDS REFRAMING

The "superlinear" claim (18.2/edit at 5 edits → 78.1/edit at 10 edits) is presented as a "novel finding." It's actually expected combinatorial behavior: with more edits, edited entities interact with each other (A's retraction involves B, B's involves A). Present as "expected quadratic interaction between edited entities" rather than a novel finding. The interesting observation is that the PairTracker's partner cleanup prevents double-counting despite the combinatorial explosion — that's the engineering contribution.

## Experiment Critique

### What's Solid

1. **Multi-run stability**: 5 runs k=5 (σ=0.004), 4 runs k=3 (σ=0.000). Publication-grade.
2. **Cross-task generalization**: 3 tasks, consistent results, at-risk predictor validated.
3. **Condition D fair comparison**: 4 experiments, 3 tasks, 100% quality retention. The paper's strongest claim.
4. **Dynamic context POC**: Simulation matches live API. Retraction mechanism validated end-to-end.
5. **k-sensitivity**: Practical recommendation (k≤5) is data-driven.

### What's Missing (in priority order)

1. **Full-corpus run (HIGH — removes the "F1=0.32" presentation problem)**: All experiments use 25K of 96K chars. C Full achieves F1=1.0 on 96K. Running A and D on the full corpus (k=5, ~19K chars/chunk) would show F1 >> 0.34 while maintaining the same savings pattern. Cost: $0.50-1.00. This single experiment transforms the paper's impact.

2. **No-retraction counterfactual for dynamic context (HIGH — zero cost)**: Simulation showing what happens if edits are applied without calling `retract_entity()`. Quantifies how many invalid pairs persist and the resulting precision drop. Makes the dynamic context section convincing rather than just "mechanism fires correctly."

3. **`apply_edits()` library method (MEDIUM — architecture claim)**: Makes the dynamic context capability first-class. ~30 lines + 4 tests. The paper claims "the framework handles dynamic updates" — the framework should actually expose this as an API.

4. **Cross-model spot check (LOW-MEDIUM)**: One run of Task 1, k=5 with gpt-4o. Even a single data point showing the same pattern (P=1.0, similar savings) would address the "gpt-4o-mini-specific" concern.

## The One Big Thing

**Run the full-corpus experiment: Condition A and D on Task 1, k=5, ~19K chars/chunk, 96K total context.**

Cost: ~$0.50-1.00. This is the single experiment that transforms the paper.

**Why**: Every result table currently shows F1 ≈ 0.32. A reviewer's first reaction is "the system barely works." The explanation ("we only use 25K of 96K chars, the oracle gets F1=1.0 on 96K") is technically correct but unconvincing without showing what happens at full scale. The full-corpus run is expected to show:
- F1(A) = F1(D) >> 0.34 (likely 0.85-0.95 based on 93.7% A/C ratio and C Full F1=1.0)
- Same 77-86% token savings (structural formula still 66.7% at k=5)
- Same P=1.0
- Same 100% compliance

This transforms the narrative from "we save 80% of tokens but only find 32% of pairs" to "we save 80% of tokens while finding 90%+ of pairs." Both are true, but one is publishable and the other invites immediate rejection.

## Specific Experiments to Run

In priority order:

1. **Full-corpus incremental run ($0.50-1.00, ~2 hrs) — HIGHEST**
   - Task 1, k=5, ~19K chars/chunk, 96K total labeled context
   - Run Condition A (incremental) and Condition D (full-recompute)
   - Expected: F1 >> 0.34, savings ~77-86%, quality ratio 100%
   - Also run Condition C (oracle on full 96K) if not already available — but C Full F1=1.0 already exists from Iteration 11
   - **This is the paper's most important missing data point**

2. **No-retraction counterfactual ($0, 30 min) — HIGH**
   - Modify `run_dynamic_context_simulation()`: after applying edits via `entity_cache.add()`, DON'T call `retract_entity()`. Count invalid pairs remaining. Compute precision.
   - Add a "Without retraction" column to Table 8
   - Directly answers "why does retraction matter?"

3. **`apply_edits()` method on IncrementalState ($0, 1 hr) — MEDIUM**
   - Extract the edit loop from `dynamic_context_experiment.py` lines 100-136 into `IncrementalState.apply_edits()`
   - 4 unit tests: (a) downgrade removes pairs, (b) upgrade adds pairs, (c) precision=1.0 maintained, (d) stats track edit retractions correctly
   - Updates `_total_retractions` counter (fixes the telemetry gap)
   - Makes the dynamic context claim architecturally honest

4. **Full-corpus Tasks 3 and 6 ($1.00, ~2 hrs) — MEDIUM**
   - If full-corpus Task 1 succeeds, replicate on Tasks 3 and 6
   - Cross-task validation at full corpus scale

5. **Cross-model spot check ($0.50, 30 min) — LOW-MEDIUM**
   - Task 1, k=5, V4 with gpt-4o (single run)
   - Verify: same pattern (P=1.0, similar A/C ratio, similar savings)

## Code Issues Found

1. **Dynamic context edit path bypasses `IncrementalState.process_chunk()` — `eval/dynamic_context_experiment.py` lines 100-136**:
   The REPL code template and simulation both manually call `entity_cache.add()` + `pair_tracker.retract_entity()` + manual pair re-evaluation. This duplicates logic from `process_chunk()` without deduplication guards, monotone merge, or telemetry. The `_total_retractions` counter in IncrementalState doesn't track edit retractions. Fix: add `apply_edits()` to IncrementalState (see experiment #3 above).

2. **`compute_gold_pairs_with_edits()` doesn't use `_check_pair_condition()` — `eval/dynamic_context_experiment.py` lines 211-238**:
   The function checks if both entities are qualifying, but doesn't use the real task condition from `eval/utils.py::_check_pair_condition`. For Task 1 (symmetric, "at least one qualifying label"), this produces correct results. For asymmetric tasks, this will produce wrong gold pairs. Add a TODO or replace with the real checker if extending beyond Task 1.

3. **`select_entities_to_edit()` iterates unsorted dicts — `eval/dynamic_context_experiment.py` lines 187-208**:
   `qualifying.items()` iteration order depends on insertion order. For reproducibility across runs, sort by entity ID:
   ```python
   for i, (uid, attrs) in enumerate(sorted(qualifying.items())):
   ```

4. **Table 8 F1 values inconsistency between simulation and live API**:
   `paper_summary_tables.py` Table 8 reports "F1 vs updated gold (T4)" as 0.5445 for 5 edits (live API). But the simulation JSON shows `f1_final: 0.8327` for 5 edits (post-T4). The discrepancy exists because the simulation processes entities deterministically while the live API relies on LLM extraction (which misses entities). The table should either (a) note this is the live API result, or (b) include both simulation and live numbers to show the gap.

5. **Missing `max_retries` for oracle C runs**:
   Task 6 V4 Run 2 oracle C failed (F1=0.0, 0 entities found — stochastic LLM failure). This has happened at least twice. Adding `max_retries=3` to oracle runs with F1>0 as success criterion is a ~5-line defensive change that prevents wasted experiment budget.

6. **`_retracted` set in PairTracker still grows unboundedly in dynamic context experiments**:
   The 10-edit live experiment fires 781 retractions. Each retracted pair is added to `_retracted`. If an entity is retracted and its pairs are re-added, those pairs are removed from `_retracted` via `discard()` in `add_pair()`. But pairs that are permanently invalidated accumulate forever. For 781 retractions this is trivial, but for production streaming use, the auto-clear recommendation from Critique 10 still applies.

## Acknowledged Limitations

- Single model (gpt-4o-mini), single corpus (OOLONG-Pairs, N=231). Cross-generalization is out of scope for proof-of-concept.
- All experiments use 25K of 96K chars, capping absolute F1 at ~0.34. The full-corpus experiment is the top-priority fix.
- Dynamic context experiment uses hand-crafted, balanced edits (2 down / 3 up or 5/5). Real-world edit patterns would be less predictable.
- The "streaming" metaphor is applied to static data chunked sequentially — accepted scoping.
- Token variance in Condition D makes absolute token comparisons fragile. The structural formula mitigates this.
- Non-monotone tasks (Task 11) show F1=0.047 — the framework works but accuracy is poor. This is a documented scope boundary.
