# Critique — Iteration 12

STATUS: CONTINUE

## Overall Assessment

The project is approaching publishable quality after 18 researcher iterations. The full-corpus simulation (F1=1.0 on Task 1, 64% check savings) resolves the most damaging presentation problem. The no-retraction counterfactual provides the missing "why retraction matters" evidence. The `apply_edits()` API makes the dynamic context claim architecturally honest. However, the core empirical claims still rest entirely on **simulations** — no live API run has validated the full-corpus (96K chars) claim, and the headline comparison table remains simulation-only. The next step is clear: a live API run on the full corpus to close the gap between simulation and reality.

## Reflection on Prior Feedback

**Resolved — not re-raising:**
- Full-corpus experiment (simulation). Done — F1=1.0/0.993 at 64% savings. ✅
- No-retraction counterfactual. Done — 99-240 invalid pairs, P drops to 0.81-0.92. ✅
- `apply_edits()` library method. Done — 80 lines, 5 tests, telemetry tracked. ✅
- Sorted dict iteration reproducibility fix. Done. ✅
- Superlinear retraction reframing. Accepted — expected combinatorial behavior. ✅
- Fair D vs A comparison (CRITICAL GAP). Fully resolved across 3 tasks, 2 runs on Task 1. ✅
- Multi-run stability. 5 runs k=5 (σ=0.004), 4 runs k=3 (σ=0.000). ✅
- k-sensitivity sweep. k∈{3,5,7,10}. ✅
- Cross-task V4 validation. Tasks 1, 3, 6 complete. ✅
- Library-level monotone_attrs. Done. ✅
- Structural savings formula 1-2/(k+1). Done. ✅
- Dynamic context POC (live API). Done with 5 and 10 edits. ✅

**Pushbacks accepted — not re-raising:**
- Cross-model validation deferred as documented limitation. Accepted.
- `PairTracker._retracted` unbounded growth. `clear_retracted()` exists. Accepted.
- Full-corpus as simulation first (not live API). Researcher's reasoning is sound for the pair-check claim. Accepted — but the live API run remains necessary (see below).

## Scores

| Criterion | Score | Delta | Comment |
|-----------|-------|-------|---------|
| Novelty | 7.5/10 | +0 | Full-corpus sim + counterfactual strengthen existing contributions but don't add new ones. The apply_edits() API is a clean engineering contribution. No new conceptual insight this iteration. |
| Technical Soundness | 9.0/10 | +0.5 | Full-corpus simulation validates F1=1.0 at scale. No-retraction counterfactual is a clean ablation. apply_edits() properly tracks telemetry. 195 tests passing. Code quality is high. |
| Benchmark Performance | 8.0/10 | +1.5 | F1=1.0 at full corpus is transformative for the paper. 64% check savings matches structural prediction. But this is all simulation — no live API numbers at 96K scale yet. |
| Scalability | 6.5/10 | +0 | Still one model (gpt-4o-mini), one corpus (OOLONG-Pairs). Full-corpus simulation shows the framework works at 96K chars, but N=231 entities is still small. No cross-model data. |
| Research Maturity | 8.0/10 | +0.5 | 9 documented contributions. Full-corpus sim removes the biggest reviewer objection. Paper tables are nearly complete. The live API confirmation is the last major gap. |

## Architecture Review

### Core Library: Solid

`rlm/core/incremental.py` is well-engineered. The new `apply_edits()` method (lines 527-606) is clean:
- Phase 1 (update + retract) correctly deduplicates across multiple edited entities
- Phase 2 (re-evaluate retracted) and Phase 3 (discover new pairs) are correctly ordered
- Telemetry (`_total_retractions`, `_noop_retractions`, `_permanent_retractions`) properly updated
- The `has_pair()` check in Phase 3 prevents duplicate additions

One minor concern: `apply_edits()` Phase 3 (lines 577-591) iterates `self.entity_cache.get_ids()` for each edited entity, giving O(E × N) complexity where E = edited entities and N = total entities. For small edit batches this is fine, but for bulk edits (E >> 10), this could be expensive. The comment on line 583 notes this with `continue` on existing pairs, but doesn't document the worst-case complexity. Consider adding a complexity note similar to `process_chunk()`'s docstring.

### Full-Corpus Simulation Code

`eval/full_corpus_and_counterfactual.py` is well-structured. One observation:

**Lines 270-274**: Chunk creation uses integer division `total_chars // num_chunks` and appends remainder to last chunk. This means the last chunk can be up to `num_chunks - 1` chars larger than others. At 96K chars / 5 chunks, this is negligible (~4 extra chars). But the code should document this asymmetry for reproducibility.

**Lines 306-307**: Condition A uses `{uid: dict(attrs) for uid, attrs in chunk_entities[ci].items()}` — a shallow copy of attrs dicts. Since `process_chunk()` with `monotone_attrs` can mutate the attrs dicts (documented in the method's docstring), this shallow copy is necessary and correct. Good.

### No-Retraction Counterfactual: Clean Design

The counterfactual (lines 58-237) correctly implements the comparison:
- WITH: Uses `apply_edits()` (architecturally honest)
- WITHOUT: Manually calls `entity_cache.add()` without retraction

One issue: the "without retraction" path (lines 146-172) also skips Phase 3 (discovering new pairs from upgrades). This means the counterfactual conflates two separate failure modes: (1) stale pairs from downgrades, and (2) missing pairs from upgrades. The `missing_new_pairs` metric (line 195) counts the second, but the F1 computation blends both. For the paper, it would be cleaner to run **two separate ablations**: (a) without retraction but WITH new pair discovery, (b) without either. This separates the precision impact (retraction) from the recall impact (new pair discovery).

## Novelty Assessment

### What's Genuinely Novel (Strong — unchanged from Critique 14)

1. **IncrementalState as a reusable library** with EntityCache + PairTracker + retraction
2. **P=1.0 across all runs, all turns, all tasks** — zero false positives from structured decomposition
3. **At-risk fraction as a predictive diagnostic** — validated ordering across 3 tasks
4. **Library-vs-template design principle** (V3→V4)
5. **Monotone attribute accumulation** as a correctness condition

### What's New This Iteration

6. **`apply_edits()` as first-class API** — makes dynamic context architecturally honest
7. **No-retraction counterfactual** — first quantitative evidence that retraction is essential (not just correct)
8. **Full-corpus simulation** — removes the F1=0.32 presentation problem entirely

### What Would Further Increase Novelty

- **Full-corpus LIVE API run**: The simulation proves the framework logic works. But the paper's claim is about an LLM system — and the LLM hasn't been tested on 19K-char chunks. At 5K chars/chunk, compliance is 100%; at 3.5K (k=7), compliance drops to 86%. How does 19K behave? The simulation can't answer this.

- **A second dynamic benchmark domain**: Everything is OOLONG-Pairs entity matching. A 30-line demo on a different domain (e.g., streaming document summarization with edits) would dramatically broaden the contribution's perceived scope.

## 3rd-Party Clarity Test

### Table 2c (D vs A vs C, Cross-Task): ✅ PASSES — Unchanged

Same-framework comparison, F1 identical in all 4 experiments, 77-86% savings. Clear, fair, reproducible.

### Table 9 (Full-Corpus Simulation, A vs D): ⚠️ PARTIAL PASS — Simulation, Not Live

A skeptical engineer reads: "F1=1.0 at 64% savings — but these are simulation numbers, not actual LLM runs." The simulation uses `IncrementalState` directly without any LLM in the loop. It proves the **library logic** is correct but does not prove the **end-to-end system** works at 96K scale. The simulation F1=1.0 is actually a correctness test of the library code, not a benchmark result.

**This is not a blocking issue** — the simulation IS the definitive result for the pair-check savings claim. But the paper must clearly distinguish between:
- **Library-level results** (simulation): F1=1.0, 64% check savings (deterministic, no LLM variance)
- **System-level results** (live API): F1=0.32, 77-86% token savings, P=1.0 (includes LLM compliance and coverage effects)

If the paper presents the simulation as "our system achieves F1=1.0" without this distinction, reviewers will (correctly) object.

### Table 10 (No-Retraction Counterfactual): ✅ PASSES

Clear comparison: with retraction → 0 invalid pairs, P=1.0; without → 99-240 invalid pairs, P=0.81-0.92. Directly answers "why does retraction matter?" with concrete numbers. The distinction between "invalid pairs remaining" (from downgrades) and "missing new pairs" (from upgrades) is properly reported.

### Naive vs Incremental (Table 2b): ✅ PASSES — Unchanged

Correctly labeled as structural advantage.

### Headline Comparison Table (MANDATORY): ⚠️ INCOMPLETE

The mandatory head-to-head table from the critique template asks for:

| Approach | Total Context | Turns | Pair Checks | Tokens | Cost | F1 | Time |
|----------|-------------|-------|-------------|--------|------|-----|------|
| Naive RLM (full recompute) | same | same | X | X | X | X | X |
| Incremental RLM | same | same | Y | Y | Y | Y | Y |
| Oracle (single-turn) | same | 1 | Z | Z | Z | Z | Z |

This table EXISTS for the 25K-char experiments (Table 2c). It does NOT exist for the 96K full-corpus case. At full corpus, we only have simulation pair checks — no tokens, cost, or time. The full-corpus live API run would fill this table completely.

## Experiment Critique

### What's Solid

1. **Full-corpus simulation** (Exp 47): Clean validation of F1=1.0 with matching A/D savings. Well-executed.
2. **No-retraction counterfactual** (Exp 48): Zero-cost, high-impact ablation. Good experimental design.
3. **apply_edits() integration**: Used in both `full_corpus_and_counterfactual.py` (counterfactual) and `dynamic_context_experiment.py` (live API REPL template). Architecture-code alignment is now honest.
4. **Test suite**: 195 tests passing, up from 187. Comprehensive coverage of the new functionality.

### What's Missing (in priority order)

1. **Full-corpus LIVE API run ($0.50-1.00) — HIGHEST PRIORITY**

   The simulation proves the library works. The live API run proves the SYSTEM works. At 19K chars/chunk, the LLM must parse entities from ~19K chars of labeled text in a single REPL turn. This is 3.8× more text per turn than the k=5 experiments (5K chars/chunk). Compliance at k=7 (3.5K/chunk) was already degraded to 86%. The question is: does 19K/chunk cause compliance failure, or does the model handle it fine because fewer turns means less cumulative complexity?

   **Specific suggestion**: Run `python eval/full_corpus_and_counterfactual.py --full-corpus-live --task 1 --k 5`. Report: F1(A), F1(D), tokens(A), tokens(D), compliance rate, wall-clock time. This fills in the mandatory headline table.

   **If compliance fails at 19K/chunk**: This is still valuable data! It tells us the operational envelope of the incremental RLM and motivates a "chunk size vs. compliance" tradeoff analysis. Try k=10 (~9.6K/chunk) as a fallback.

2. **Separated counterfactual ablation ($0, 30 min) — MEDIUM**

   Currently, the "without retraction" path disables BOTH retraction (stale pair removal) AND new pair discovery (from upgrades). These are two distinct mechanisms. Split into:
   - **(a) Without retraction, with new pair discovery**: Only precision degrades (stale pairs from downgrades). Recall is maintained.
   - **(b) Without retraction, without new pair discovery** (current): Both precision and recall degrade.

   This separation cleanly attributes the F1 drop: how much is from stale pairs (precision), how much from missed upgrades (recall)?

3. **Cross-model spot check ($0.50, 30 min) — LOW-MEDIUM**

   One run of Task 1, k=5, V4 with gpt-4o or claude-3.5-sonnet. The entire empirical contribution rests on gpt-4o-mini. Even a single run showing: P=1.0, compliance=100%, A/C ≈ 90%+ would address the "model-specific" concern. Deferred across 5 critique cycles; each deferral weakens the paper more.

4. **Full-corpus counterfactual ($0, 15 min) — LOW**

   Run the no-retraction counterfactual on the full 96K corpus (not just 25K). At full corpus with F1=1.0 baseline, the precision drop from skipping retraction would be even more dramatic. The `--full-corpus-counterfactual` flag already exists in the CLI.

## The One Big Thing

**Run the full-corpus live API experiment: Condition A and D on Task 1, k=5, 96K total context, ~19K chars/chunk.**

This is the single experiment that completes the paper's evidence base. The simulation proves the algorithm; the live run proves the system. Cost: ~$0.50-1.00. The `--full-corpus-live` flag is already implemented in `eval/full_corpus_and_counterfactual.py`.

**Why this is the #1 priority**: The paper currently has two disconnected evidence streams:
1. **Live API results**: F1=0.32 at 25K chars, P=1.0, 77-86% savings — proves the system works but looks weak
2. **Simulation results**: F1=1.0 at 96K chars, 64% savings — proves the algorithm but isn't a real system run

The live full-corpus run MERGES these streams into: "F1≈1.0 at 96K chars with live LLM, P=1.0, ~64-80% savings." This is the paper's headline result.

## Specific Experiments to Run

In priority order:

1. **Full-corpus live API, Task 1, k=5 ($0.50-1.00, ~2 hrs) — HIGHEST**
   ```bash
   python eval/full_corpus_and_counterfactual.py --full-corpus-live --task 1 --k 5
   ```
   Report: F1(A), F1(D), tokens, compliance, wall-clock. If compliance fails at 19K/chunk, try k=10 (~9.6K/chunk).

2. **Separated counterfactual ablation ($0, 30 min) — MEDIUM**
   Add a `--counterfactual-retract-only` mode that applies retraction but NOT new pair discovery. This isolates the precision impact of retraction from the recall impact of pair discovery.

3. **Cross-model spot check, Task 1, k=5 ($0.50, 30 min) — LOW-MEDIUM**
   ```bash
   # Modify run_condition_a_v4 to accept model parameter
   python eval/label_aware_v4_experiment.py --task 1 --model gpt-4o
   ```

4. **Full-corpus counterfactual ($0, 15 min) — LOW**
   ```bash
   python eval/full_corpus_and_counterfactual.py --full-corpus-counterfactual --task 1 --k 5
   ```

## Code Issues Found

1. **`apply_edits()` Phase 3 missing complexity docstring** — `rlm/core/incremental.py` lines 577-591:
   The method iterates `get_ids()` for each edited entity but doesn't document O(E × N) worst-case complexity (where E = edited entities, N = total entities). The `process_chunk()` method has an excellent complexity docstring; `apply_edits()` should match that standard.

   Suggested addition:
   ```python
   """
   Complexity: O(E × N) for Phase 3 (new pair discovery), where E = len(edits)
   and N = total entities. Phase 2 (retraction re-evaluation) is O(R) where
   R = total retracted pairs. For small edit batches (E < 20), this is negligible.
   For bulk edits (E > 100), consider batching or using process_chunk() with
   a synthetic edit chunk instead.
   """
   ```

2. **`full_corpus_and_counterfactual.py` chunk creation asymmetry undocumented** — lines 270-274:
   The last chunk may be up to `num_chunks - 1` chars larger than others. Add a comment noting this for reproducibility.

3. **`full_corpus_and_counterfactual.py` line 130**: `compute_f1(list(pairs_with_retraction), gold_post_edit)` — the `compute_f1` function receives a `list` of pairs but `gold_post_edit` type isn't verified. If `gold_post_edit` returns pairs in non-canonical order, the comparison silently fails. Verify `compute_f1` handles this (it likely does via set comparison internally, but worth confirming).

4. **Gemini test import error**: `tests/clients/test_gemini.py` fails to collect due to `ImportError: cannot import name 'genai' from 'google'`. This causes `pytest --co` to error. The test is skipped in CI (per CLAUDE.md), but local developers see it as a failure. Consider adding a `pytest.importorskip("google.genai")` at the top of the test file.

5. **Researcher response item #3 (use apply_edits in dynamic_context_experiment.py)**: The REPL template in `dynamic_context_experiment.py` now uses `apply_edits()` (confirmed via grep), but the simulation path (`--simulate`) should also be verified to use `apply_edits()` for consistency. Check that the simulation doesn't still use the manual retraction loop.

## Acknowledged Limitations

- Single model (gpt-4o-mini), single corpus (OOLONG-Pairs). Accepted scope boundary for proof-of-concept.
- Full-corpus results are simulation-only (no live API). The live run is the top priority.
- Cross-model validation deferred across 5 critique cycles. Documented limitation.
- Non-monotone tasks (Task 11) show F1=0.047. Documented scope boundary.
- Dynamic context experiment uses hand-crafted, balanced edits. Real-world patterns are less predictable.
