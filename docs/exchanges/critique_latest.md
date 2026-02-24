# Critique — Iteration 13

STATUS: CONTINUE

## Overall Assessment

The research has reached a strong position: full-corpus simulation demonstrates F1≈1.0 with 64% pair-check savings (matching the structural formula within 3pp), the no-retraction counterfactual provides compelling evidence that retraction is essential (precision drops to 0.81 with 10 edits), and `apply_edits()` makes the dynamic context claim architecturally honest. The **single most important remaining gap** is the absence of a **live API full-corpus run** — the simulation proves pair-check savings but does NOT prove token savings or LLM compliance at ~19K chars/chunk. This is a $1 experiment that transforms the paper from "simulation shows F1=1.0" to "live LLM achieves F1≈1.0 with 77-86% token savings."

## Reflection on Prior Feedback

**Resolved — not re-raising:**
- Full-corpus F1 presentation problem (F1=0.32 → F1=1.0 in simulation). Done.
- No-retraction counterfactual (precision 1.0 → 0.81 without retraction). Done.
- `apply_edits()` library method. Implemented with 5 tests. Done.
- Sorted dict iteration fix in `select_entities_to_edit()`. Done.
- Superlinear retraction reframing. Researcher agrees, will reframe. Done.
- Fair Condition D vs A comparison across 3 tasks. Done (77-86% savings, 100% quality). Done.
- Multi-run stability (5 runs k=5, σ=0.004). Done.
- k-sensitivity sweep. Done.
- Library-level monotone_attrs. Done.
- Structural savings formula. Done.

**Pushback accepted — not re-raising:**
- Cross-model validation deferred as documented limitation. Accepted (but still recommended).
- `PairTracker._retracted` unbounded growth. `clear_retracted()` exists. Accepted.
- Simulation vs live API for full-corpus: researcher's argument that IncrementalState simulation uses the same library code is valid for pair-check savings. But token savings requires a live API run (see below).

## Scores

| Criterion | Score | Delta | Comment |
|-----------|-------|-------|---------|
| Novelty | 8.0/10 | +0.5 | No-retraction counterfactual + apply_edits() strengthen the retraction narrative. Full-corpus F1≈1.0 makes the efficiency claim dramatically more compelling. |
| Technical Soundness | 8.5/10 | +0 | apply_edits() implementation is correct for multi-entity edits (verified by code review). Minor encapsulation violation and telemetry inflation issue (see Code Issues). 193 tests passing. |
| Benchmark Performance | 8.0/10 | +1.5 | Major improvement. Full-corpus F1=1.0 (Task 1), 0.993 (Tasks 3/6) with 64% check savings. This is the result the paper needed. But it's simulation-only — the live API confirmation is still the missing piece for the headline table. |
| Scalability | 6.5/10 | +0 | Still single model (gpt-4o-mini), single corpus. The full-corpus simulation shows the framework scales to 96K chars. But ~19K chars/chunk is a real chunk size that needs live API validation — does the LLM comply when the chunk is 4× larger? |
| Research Maturity | 8.0/10 | +0.5 | 9 documented contributions, paper-ready tables, full-corpus numbers, counterfactual evidence. Close to submission. The live API run is the one remaining must-do. |

## Architecture Review

### apply_edits() — Correct but with Minor Issues

Reviewed the full implementation (lines 518-595 of `rlm/core/incremental.py`). The multi-entity interaction semantics are correct:

- **Ordering**: When entities A and B are both edited and share a pair, A is processed first (retract, re-evaluate, check new), then B. The pair (A,B) may be retracted and re-added during A's pass, then retracted again during B's pass with B's updated attributes. The final pair set is correct because B's pass sees A's already-updated attributes.
- **Upgrade-upgrade**: Both A and B non-qualifying → qualifying. A's pass doesn't discover (A,B) because B isn't updated yet. B's pass discovers it. Correct.
- **Downgrade-downgrade**: Both downgraded. A's pass retracts (A,B) and cleans up B's index. B's pass has nothing to retract. Correct.

**Issue 1 — Encapsulation violation (line 575)**: `if canonical in self.pair_tracker._pairs` directly accesses the private `_pairs` set. Add a `has_pair()` or `__contains__` method to PairTracker. This is a code smell that will cause problems if the internal representation changes.

**Issue 2 — Telemetry double-counting for multi-entity edits**: When two edited entities share a pair, `total_retracted` counts the same pair twice (once per entity's retraction). `pairs_readded` also counts the re-add twice. The net `permanent_retractions` calculation (`total_retracted - pairs_readded`) is correct because the inflation cancels, but the individual `total_retracted` and `pairs_readded` fields overstate the true values. For 5-10 edits this is negligible; for larger edit batches it could mislead analysis.

**Concrete fix**: Process all retractions first (collect retracted pairs into a deduplicated set), then re-evaluate once. This separates the retract and evaluate phases:

```python
def apply_edits(self, edits, pair_checker=None, edit_chunk_index=-1):
    # Phase 1: Update all entities and collect ALL retracted pairs
    all_retracted = set()
    for eid, new_attrs in edits.items():
        self.entity_cache.add(eid, new_attrs, chunk_index=edit_chunk_index)
        retracted = self.pair_tracker.retract_entity(eid)
        all_retracted |= retracted

    total_retracted = len(all_retracted)  # Deduplicated count

    # Phase 2: Re-evaluate all retracted pairs with updated attributes
    pairs_readded = 0
    new_pairs_from_edits = 0
    if pair_checker:
        for p in all_retracted:
            a1 = self.entity_cache.get(p[0])
            a2 = self.entity_cache.get(p[1])
            if a1 and a2 and pair_checker(a1, a2):
                self.pair_tracker.add_pair(p[0], p[1])
                pairs_readded += 1

        # Phase 3: Check for new pairs (edited entities × all)
        edited_ids = set(edits.keys())
        for eid in edited_ids:
            updated_attrs = self.entity_cache.get(eid)
            for other_id in self.entity_cache.get_ids():
                if other_id == eid:
                    continue
                if self.pair_tracker.has_pair(eid, other_id):
                    continue
                other_attrs = self.entity_cache.get(other_id)
                if other_attrs and pair_checker(updated_attrs, other_attrs):
                    self.pair_tracker.add_pair(eid, other_id)
                    new_pairs_from_edits += 1
    # ... telemetry updates
```

This is also more efficient: O(|all_retracted|) re-evaluations instead of potentially re-evaluating the same pair twice.

### dynamic_context_experiment.py Still Uses Manual Edit Path

The researcher noted "Update dynamic context experiment to use apply_edits()" as a next step but didn't implement it. The live API REPL code (lines 95-136) still manually calls `entity_cache.add()` + `pair_tracker.retract_entity()` + manual loop. This means the live API experiment validates the MANUAL path, not the `apply_edits()` library method.

**Impact**: The paper claims "apply_edits() handles dynamic context" but the only live API evidence uses a different code path. The counterfactual experiment (Exp 48) DOES use `apply_edits()`, which is good, but that's a simulation.

**Recommendation**: Update the REPL template in `dynamic_context_experiment.py` to call `_incremental.apply_edits(edits_dict, pair_checker=check_pair)`. This is a ~10-line change that aligns the live API evidence with the library API.

## Novelty Assessment

### Genuinely Novel (Strong — quantify these in the paper)

1. **F1≈1.0 at 64% pair-check savings on full corpus** — the headline result. Transforms the contribution from "efficient but barely works" to "efficient AND nearly perfect."
2. **P=1.0 across ALL runs, ALL turns, ALL tasks** — the most surprising empirical finding. Frame prominently.
3. **Retraction is essential, not optional**: No-retraction counterfactual shows precision drops from 1.0 to 0.81 with 10 edits. Concrete evidence that the mechanism provides real value.
4. **Library-vs-template design principle**: V3→V4 demonstrating that invariants belong in library code. Broadly useful for LLM program design.
5. **At-risk fraction as a predictive diagnostic**: Validated ordering across 3 tasks.

### The Contribution That's Underemphasized

The **"library-level correctness guarantees for LLM programs"** idea is more general than the pair-finding application. The insight: when you decompose an LLM task into structured stateful computation (EntityCache, PairTracker, retraction), the library can enforce invariants that the LLM cannot reliably maintain via prompts alone. V3→V4 is a clean demonstration:

| | Template-level (V3) | Library-level (V4) |
|---|---|---|
| Compliance | 60-100% (stochastic) | 100% (deterministic) |
| Token overhead | 2.4-4.8× | 1.3× |
| Retractions | ~1,078 no-op | 0 |

This is a broadly applicable finding for any "LLM programs with REPL execution" system. Frame it as a design principle, not just an optimization.

## 3rd-Party Clarity Test

### Table 9 (Full-Corpus A vs D Simulation): ⚠️ PARTIAL PASS — Simulation Only

A skeptical engineer reads: "F1=1.0 on the full corpus, 64% pair-check savings." Strong result. But then asks: **"These are simulated pair checks, not actual LLM token savings. Where's the live API run?"**

The simulation uses `IncrementalState` directly — no LLM involved. It proves:
- ✅ The library correctly computes incremental pair checks
- ✅ The monotone merge produces identical results to full recompute
- ✅ Pair-check savings match structural formula

It does NOT prove:
- ❌ Token savings (which depend on prompt overhead, LLM iteration counts, etc.)
- ❌ LLM compliance at ~19K chars/chunk (4× larger than the validated 5K chunks)
- ❌ F1≈1.0 through the full RLM pipeline (LLM extraction errors could reduce this)

**Blocking issue**: The paper cannot claim "77-86% token savings at F1≈1.0" without a live API full-corpus run. It CAN claim "64% pair-check savings at F1=1.0 (simulation)" and "77-86% token savings at F1=0.32 (live API, 25K context)" separately. But the combined claim requires combined evidence.

### Table 10 (No-Retraction Counterfactual): ✅ PASSES

Clear comparison: with vs without retraction, same edits, same starting state. Precision drops from 1.0 to 0.81-0.92. 99-240 invalid pairs persist. Unambiguous.

Minor note: this uses the 25K subset (4 chunks × 5K). Running on the full corpus would show more dramatic numbers.

### Table 2c (Cross-Task D vs A, Live API): ✅ PASSES (unchanged)

Still the strongest table. 4 experiments, 3 tasks, 100% quality retention, 77-86% token savings. Real API numbers.

### Headline Paper Table — STILL INCOMPLETE

The paper needs ONE table that tells the complete story. Current state:

| Approach | Context | F1 | Pair Savings | Token Savings | Cost | Time |
|----------|---------|-----|-------------|---------------|------|------|
| Naive RLM (no framework) | 25K | 0.0 | — | — | $0.025 | 135s |
| Full-recompute D (25K) | 25K | 0.3228 | baseline | baseline | ~$0.05 | ~250-540s |
| Incremental A (25K) | 25K | 0.3228 | 58% (sim) | **77-86%** | ~$0.007 | ~120s |
| Oracle C (25K) | 25K | 0.3424 | — | — | ~$0.004 | ~30s |
| Oracle C (96K) | 96K | 1.0 | — | — | — | — |
| **Full-corpus A (96K)** | 96K | **1.0 (sim)** | **64% (sim)** | **? (no live data)** | **?** | **?** |
| **Full-corpus D (96K)** | 96K | **1.0 (sim)** | baseline | **?** | **?** | **?** |

The "?" cells are why the live API full-corpus run is essential. A reviewer will see: "The best F1 numbers are simulation-only. The live API numbers cap at F1=0.32."

## Experiment Critique

### What's Solid

1. **Full-corpus simulation**: Clean, reproducible, matches structural prediction. 3 tasks. Good.
2. **No-retraction counterfactual**: Well-designed ablation study. Quantifies retraction value concretely.
3. **apply_edits() with 5 tests**: Good software engineering. Makes the API claim honest.
4. **All prior work**: D vs A comparison, multi-run stability, k-sensitivity, at-risk fraction — all remain valid and strong.

### What's Missing (in priority order)

1. **Live API full-corpus run (HIGHEST — $0.50-1.00, ~2 hrs)**:
   Run Condition A and D on 96K chars with live gpt-4o-mini calls. Expected: F1 close to 1.0 (sim shows 1.0), token savings similar to structural (66.7%), same P=1.0. The `--full-corpus-live` flag is already implemented.

   **Why simulation isn't enough**: At ~19K chars/chunk, the LLM receives 4× more context per turn than the validated 5K setting. Compliance, extraction accuracy, and iteration count may all differ.

2. **Migrate dynamic_context_experiment.py to use apply_edits() ($0, 30 min)**:
   Replace REPL template lines 95-136 with `_incremental.apply_edits(edits_dict, pair_checker=check_pair)`. Aligns live API evidence with library API.

3. **apply_edits() two-phase refactor ($0, 1 hr)**:
   Separate retract-all from re-evaluate-all. Fix telemetry inflation. Add multi-entity interaction test.

4. **Cross-model spot check (LOW — $0.50, 30 min)**:
   Single gpt-4o run. Still deferred, still recommended.

## The One Big Thing

**Run the live API full-corpus experiment.** The code exists (`--full-corpus-live`). It costs ~$1 and takes ~2 hours. This single experiment:

1. Fills the "?" cells in the headline table
2. Proves F1≈1.0 through the full RLM pipeline (not just simulation)
3. Validates LLM compliance at ~19K chars/chunk
4. Provides real token savings numbers at full-corpus scale
5. Makes the paper submission-ready

Without it, the paper has a split personality: "our simulation shows F1=1.0" but "our actual system shows F1=0.32." A reviewer will fixate on the live number.

## Specific Experiments to Run

1. **Live API full-corpus run ($1, ~2 hrs) — MUST-DO**
   ```bash
   python eval/full_corpus_and_counterfactual.py --full-corpus-live --task 1 --k 5
   python eval/full_corpus_and_counterfactual.py --full-corpus-live --task 3 --k 5
   python eval/full_corpus_and_counterfactual.py --full-corpus-live --task 6 --k 5
   ```
   Expected: F1 ≈ 0.9-1.0, token savings ~65-80%, P=1.0.
   If compliance breaks at ~19K chars/chunk, try k=3 (~32K chars/chunk) or k=7 (~14K chars/chunk).

2. **Migrate dynamic_context_experiment.py to apply_edits() ($0, 30 min)**
   Update REPL template lines 95-136 to use `_incremental.apply_edits()`.

3. **apply_edits() encapsulation fix + two-phase refactor ($0, 1 hr)**
   Add `PairTracker.has_pair(id1, id2) -> bool`. Separate retract/evaluate phases in apply_edits(). Add test: edit 2 entities sharing a pair, verify deduplicated retraction count.

4. **Full-corpus no-retraction counterfactual ($0, 15 min)**
   Run on 96K chars. More entities → more dramatic precision drop. Strengthens the retraction argument.

5. **Chunk boundary alignment check ($0, 15 min)**
   Verify that `full_corpus_and_counterfactual.py`'s character-boundary chunking doesn't split mid-entity-profile. If it does, align to user profile boundaries (use the regex-based chunking from `_make_sequential_chunks`).

## Code Issues Found

1. **`apply_edits()` accesses private `_pairs` — `rlm/core/incremental.py` line 575**:
   ```python
   if canonical in self.pair_tracker._pairs:  # Private access
   ```
   Fix: Add `has_pair(id1, id2) -> bool` to PairTracker, then use `self.pair_tracker.has_pair(eid, other_id)`.

2. **`apply_edits()` telemetry inflated for multi-entity edits — lines 549-581**:
   When two edited entities share a pair, `total_retracted` and `pairs_readded` both count the pair twice. Net calculation is correct (inflation cancels). Fix with two-phase refactor above.

3. **`dynamic_context_experiment.py` REPL template bypasses `apply_edits()` — lines 95-136**:
   Live API evidence misaligned with library API. Update to call `_incremental.apply_edits()`.

4. **`compute_gold_pairs_with_edits()` doesn't use `_check_pair_condition()` — `eval/dynamic_context_experiment.py` lines 211-238**:
   Simplified check, correct for Task 1 only. Add `assert task_idx == 1` guard or use the real checker.

5. **`tests/clients/test_gemini.py` collection error**: Import error prevents full test suite from running. Pre-existing but should be fixed.

6. **`full_corpus_and_counterfactual.py` character-boundary chunking (lines 271-275)**: May split mid-entity-profile. Verify this doesn't cause entity loss in the live API path.

## Acknowledged Limitations

- Single model (gpt-4o-mini). Cross-model validation deferred as documented limitation. Accepted.
- OOLONG-Pairs corpus only (N=231 entities). Cross-corpus validation out of scope. Accepted.
- Dynamic context experiment uses hand-crafted balanced edits. Accepted for POC.
- Non-monotone tasks (Task 11) show F1=0.047 — documented scope boundary. Accepted.
- Token variance in Condition D (3× range across runs). Structural formula mitigates. Accepted.
