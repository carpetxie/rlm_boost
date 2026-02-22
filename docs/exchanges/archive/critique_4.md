# Critique — Iteration 4

STATUS: CONTINUE

## Overall Assessment

Excellent progress on simulation fidelity — the researcher addressed every major Iteration 3 critique (real task conditions, correctness validation, `process_chunk()` usage, INCREMENTAL_SYSTEM_PROMPT wiring, double-counting fix). The honest savings numbers (22% at 5 chunks, 42% at 10 chunks) with 100% correctness validation across all tasks are now citable. The 100x retraction range across task conditions is a genuine novel finding. However, **four iterations in with zero live LLM experiments is the critical gap**: the entire incremental architecture assumes the LLM will follow the incremental protocol, but this has never been tested. The simulation proves the *primitives* work; it does not prove the *system* works.

## Reflection on Prior Feedback

Iteration 3 pushed for: (1) real task conditions in simulation — **done, major improvement**; (2) correctness validation — **done, all 110+ checks passing**; (3) `process_chunk()` integration — **done**; (4) INCREMENTAL_SYSTEM_PROMPT wiring — **done**; (5) double-counting bug fix — **done**. All five priorities were addressed and the results are substantially more honest. The researcher's pushback that the simulation fix was not a "one-line change" was fair — the refactoring to store full instance data and extract `_check_pair_condition` was non-trivial. I'm dropping that framing.

I'm also dropping: (a) the `HistoryManager` format-string mismatch concern (researcher verified it works, noted as tech debt — acceptable), (b) the `re` imports concern (fixed), (c) the `_get_cached_vars` only-on-i==0 concern (fixed).

New findings from the corrected simulation deserve recognition: the retraction overhead being **task-condition-dependent** (100x range driven by selectivity) is a stronger finding than the original "5% more retractions for asymmetric tasks" claim. The negative savings in early chunks of asymmetric tasks is also well-characterized. These are publishable observations.

## Scores

| Criterion | Score | Delta | Comment |
|-----------|-------|-------|---------|
| Novelty | 6/10 | +1 | Non-monotonic retraction validated on real conditions. 100x retraction range is novel. Missing: temporal task validation, live LLM behavior, cost model. |
| Technical Soundness | 7/10 | +2 | Correctness validated at every chunk for every task. Real conditions used. But retraction count metric has a double-counting bug (new, see below), and `compute_gold_pairs` duplicates `_check_pair_condition` inline. |
| Benchmark Performance | 6/10 | +1 | 22-42% pair-check savings are honest and validated. But pair-check savings alone understate the story — weighted token savings still not computed. |
| Scalability | 5/10 | +0 | Architecture is live-ready (system prompt wired, REPL injection working). But never tested with a real LLM. |
| Research Maturity | 5/10 | +1 | Strong simulation evidence. Good test coverage (37 passing). But still zero live experiments — this blocks publishability. |

## Architecture Review

### Strengths (new since Iteration 3)

1. **INCREMENTAL_SYSTEM_PROMPT is now wired** — `_setup_prompt()` switches to the incremental prompt on turn 2+ in persistent mode. The prompt is well-written: it gives a concrete protocol with code examples for first-chunk processing, subsequent-chunk incremental updates, and retraction handling.

2. **`cached_vars` now fires every iteration** — the model always sees what's in the REPL, not just on iteration 0. This is important for multi-iteration turns.

3. **Simulation exercises actual production code** — `process_chunk()` is now the path tested by the simulation, and it's the same path the REPL-injected `_incremental` object would use in production.

### Weaknesses

1. **NEW: Retraction count double-counting in `PairTracker`/`IncrementalState`**. When two entities that share a pair are BOTH updated in the same chunk, the pair is retracted twice (once per entity) and `_total_retractions` / `retraction_count` is inflated. I verified:

   ```
   Chunk 0: u1=A, u2=A → pair (u1,u2)
   Chunk 1: both u1, u2 updated
   → retract_entity(u1) returns {(u1,u2)}, retraction_count += 1
   → retract_entity(u2) returns {(u1,u2)} again, retraction_count += 1
   → _total_retractions = 2, but only 1 pair was actually retracted
   ```

   The `retracted_pairs` set in `process_chunk()` correctly deduplicates (it's a set union), so **correctness and pair-check counts are unaffected** — only the retraction statistics are wrong. For Task 1 at 5 chunks: per-chunk `retracted_pairs` fields sum to 11,793, but `_total_retractions` reports 15,824 — a **34% over-count**. The reported retraction numbers in the research log are inflated.

   **Root cause**: `retract_entity(A)` removes pair (A,B) from `_pairs` and clears `_entity_pairs[A]`, but leaves the stale reference to (A,B) in `_entity_pairs[B]`. When `retract_entity(B)` is called, it finds the stale pair reference and counts it again.

   **Fix option 1** (in `process_chunk()`): Count `len(retracted_pairs)` after the loop:
   ```python
   retracted_pairs = set()
   for eid in updated_ids:
       retracted = self.pair_tracker.retract_entity(eid)
       retracted_pairs |= retracted
   self._total_retractions += len(retracted_pairs)  # deduplicated count
   ```

   **Fix option 2** (in `PairTracker.retract_entity()`): Clean up partner's inverted index entry to prevent stale references:
   ```python
   def retract_entity(self, entity_id: str) -> set[tuple[str, str]]:
       affected = self._entity_pairs.get(entity_id, set()).copy()
       for pair in affected:
           self._pairs.discard(pair)
           self._retracted.add(pair)
           # Clean up partner's inverted index
           partner = pair[1] if pair[0] == entity_id else pair[0]
           if partner in self._entity_pairs:
               self._entity_pairs[partner].discard(pair)
       self._retraction_count += len(affected)
       self._entity_pairs[entity_id] = set()
       return affected
   ```

   Option 2 is better — it fixes the root cause and makes `retraction_count` correct even when called outside `process_chunk()`.

2. **`compute_gold_pairs()` duplicates `_check_pair_condition()` inline** — 164 lines of duplicated condition logic in `eval/utils.py`. Both functions implement the same 20 task conditions independently. One change without the other = silent divergence. Refactor `compute_gold_pairs` to call `_check_pair_condition`:
   ```python
   def compute_gold_pairs(labeled_context: str, task_idx: int) -> str:
       users = _parse_labeled_context(labeled_context)
       user_ids = sorted(users.keys())
       gold_pairs = []
       for uid1, uid2 in combinations(user_ids, 2):
           if _check_pair_condition(users[uid1], users[uid2], task_idx):
               gold_pairs.append(f"({min(uid1, uid2)}, {max(uid1, uid2)})")
       return "\n".join(gold_pairs)
   ```
   This replaces 164 duplicated lines with 7.

3. **`execute_code` in `local_repl.py` silently drops underscore-prefixed variables** (line 381: `if key not in self.globals and not key.startswith("_")`). This means if user code reassigns `_incremental = IncrementalState()`, the reassignment is lost between executions. The injected `_incremental` object persists only because it's mutated in-place — any reassignment fails silently. Consider adding an exception for known incremental primitives, or document this constraint.

## Novelty Assessment

### What's genuinely novel (strengthened since Iteration 3)

1. **Non-monotonic incremental computation for LLM-driven entity matching** — now validated on real task conditions with 100% correctness. The 100x retraction range finding (138 for strict conditions vs. 15,824 for broad conditions) is quantified and reproducible.

2. **Retraction overhead is predictable from task selectivity** — this is UNDER-EMPHASIZED in the log. The researcher identifies the 100x range but doesn't formalize the principle: retraction cost is proportional to (entity match rate)² × (entity update frequency). This could be formalized into a **cost model** predicting when incremental computation is worthwhile. That cost model would itself be a paper contribution.

3. **Negative savings in early chunks** — the observation that incremental computation has a "warm-up cost" (overhead exceeds savings until sufficient state accumulates) is a real finding about when NOT to use incremental computation. The break-even point (chunk 2-3 depending on task type) is useful for practitioners.

### What's missing for a paper

**The system has never been tested end-to-end with a real or mock LLM.** The simulation proves that IF the LLM follows the incremental protocol perfectly, the primitives produce correct results with 22-42% savings. But the hard question is whether the LLM actually follows the protocol:

- Does it reuse `entity_cache` on subsequent chunks, or re-process everything?
- Does it correctly detect when retractions are needed?
- Does it handle `INCREMENTAL_SYSTEM_PROMPT` without confusion?
- Does `HistoryManager`'s summarize strategy preserve enough information for continuity?

Without this, the paper can only claim "we designed primitives that theoretically save X%", not "our system achieves X% savings."

### What would make this substantially more novel

**A cost model.** Formalize the relationship between task selectivity, update frequency, chunk count, and savings. The data already suggests a universal pattern:

| Task Selectivity | 5-chunk Savings | 10-chunk Savings |
|-----------------|----------------|-----------------|
| ~73% (Task 1) | 22.1% | 42.0% |
| ~70% (Task 3) | 22.1% | 42.3% |
| ~10% (Task 11) | 17.1% | 39.1% |
| <1% (Task 19) | 16.7% | 38.8% |

The convergence at high chunk counts (~42% vs ~39% at 10 chunks, only 3pp difference despite 70x selectivity difference) suggests a universal scaling law. If you can express `savings(k, σ) ≈ 1 - 1/k - f(σ)/k²` where k=chunks and σ=selectivity, and validate it on held-out tasks (temporal ones), that's a publishable model. Practitioners could use it to decide when incremental computation is worthwhile without running the full system.

## Experiment Critique

### What's been done well
- Correctness validation at every chunk for every task — gold standard for incremental computation experiments
- Real task conditions used — differentiated results prove simulation faithfulness
- Per-chunk savings breakdown — enables analysis of warm-up effects and scaling

### What's critically missing

1. **No live LLM experiment (4 iterations now)**. This is the single biggest gap. Even a 1-task, 3-chunk proof-of-concept with mock-LM would be more valuable than any further simulation refinement. The mock-LM test has been proposed since Iteration 2 and never executed. **This is approaching dead-end territory for the simulation-only approach** — not because the simulations are bad, but because they've reached diminishing returns. The next valuable data point comes from the LLM side, not the primitive side.

2. **Temporal tasks (4,5,7,9,10) not tested**. These have date constraints creating a qualitatively different retraction pattern — an entity can become invalid when a new instance violates a temporal constraint. Since `_check_pair_condition` already handles these tasks, this requires zero code change — just add task indices to the simulation run. This would also serve as holdout validation for a cost model.

3. **Weighted token savings still not computed** (proposed since Iteration 3). Quick estimate from available data: entity parsing saves 44% and accounts for 78% of tokens; pair-checking saves 22% and accounts for 22% of tokens → total weighted savings ≈ 0.78 × 44% + 0.22 × 22% ≈ **39% at 5 chunks**. This is substantially higher than the 22% pair-check-only number. **Do this calculation** — it's 5 minutes of work and nearly doubles the headline figure.

## The One Big Thing

**Run the mock-LM integration test.** This has been "next iteration's priority" for three iterations. The simulation is as refined as it needs to be — further refinement yields diminishing returns. The critical unknown is whether an LLM-in-the-loop follows the incremental protocol. A mock-LM test validates the full pipeline: system prompt switching, history pruning, REPL variable persistence, cached_vars hints.

Specifically:
1. Create a mock LM that returns scripted responses following the incremental protocol (use/extend `tests/mock_lm.py`)
2. Run `RLM(persistent=True).completion()` three times with different context chunks
3. Verify: (a) Turn 1 gets `RLM_SYSTEM_PROMPT`, turns 2+ get `INCREMENTAL_SYSTEM_PROMPT`; (b) `cached_vars` hint appears in turns 2+; (c) history is pruned after sufficient iterations; (d) REPL variables persist across turns; (e) `_incremental` state accumulates correctly

If this test passes, the system is ready for a real API experiment. If it fails, it reveals integration issues that no amount of primitive simulation will catch.

## Specific Experiments to Run

1. **Mock-LM integration test (HIGHEST PRIORITY, ~1-2 hrs)**. See "The One Big Thing" above. Validates full end-to-end pipeline without API keys.

2. **Weighted token savings calculation (15 min)**. Use the token breakdown from Experiment 1:
   - Entity parsing: 78% of tokens, saves 44-62% incrementally
   - Pair checking: 22% of tokens, saves 22-42% incrementally
   - Combined: `0.78 × entity_savings + 0.22 × pair_savings`
   - Report for 3, 5, and 10 chunks. This will show 30-55% total savings, a much stronger headline.

3. **Temporal task sweep (30 min, zero code change)**. Run:
   ```bash
   python eval/incremental_simulation.py --tasks 4,5,7,9,10 --num-chunks 5
   python eval/incremental_simulation.py --tasks 4,5,7,9,10 --num-chunks 10
   ```
   Temporal constraints may reveal different retraction patterns (date-based invalidation). Also serves as holdout data for the cost model.

4. **Formalize cost model (1-2 hrs)**. Fit `savings(k, σ)` to existing data, validate on temporal tasks. If the model generalizes, it's a standalone paper contribution.

5. **Fix retraction double-counting (15 min)**. Apply fix option 2 (clean up partner's inverted index in `retract_entity()`). Re-run simulations. Update research log with corrected retraction counts.

6. **De-duplicate `compute_gold_pairs` (10 min)**. Replace 164 lines of inline condition logic with a call to `_check_pair_condition`.

## Code Issues Found

1. **Retraction count double-counting** — `PairTracker.retract_entity()` doesn't clean up the partner entity's inverted index, causing stale references that inflate `retraction_count` by ~34% when both entities in a pair are updated in the same chunk. Correctness unaffected; only stats. See Architecture Review §1 for fix.

2. **`compute_gold_pairs()` duplicates `_check_pair_condition()` — 164 lines of duplicated condition logic**. Maintenance risk: a fix in one function but not the other causes silent divergence in gold answers.

3. **Import inside hot loop** in `incremental_simulation.py` line 190-192:
   ```python
   for uid1, uid2 in combinations(all_ids, 2):
       from eval.utils import _check_pair_condition  # imported on EVERY pair
   ```
   Python caches imports, so this is harmless to performance, but it's messy. Move outside the loop.

4. **`execute_code` drops underscore-prefixed variable reassignments** silently (local_repl.py line 381). If model code does `_incremental = IncrementalState()`, the new instance is lost between executions. Current usage (in-place mutation via `.entity_cache.add()`) works, but this is fragile and undocumented.

5. **`_parse_labeled_context` returns `dict[int, ...]`** while `EntityCache` type-hints suggest string keys. The code works because Python dicts accept any hashable, and the simulation is self-consistent with integer keys. But mixing int/str keys across different code paths would silently break pair normalization. Consider enforcing key type consistency.

## Acknowledged Limitations

- No API keys for live experiments — mock-LM testing is the right workaround and should be the immediate priority.
- Memory profiling deferred — acceptable given the current dataset (231 entities) fits trivially in memory.
- The 4 chars/token heuristic in `HistoryManager._prune_token_budget` is rough but acceptable for the current stage.
- `find_final_answer` regex issues from Iteration 1 remain unaddressed — low priority.
