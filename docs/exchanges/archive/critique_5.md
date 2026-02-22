# Critique — Iteration 5

STATUS: CONTINUE

## Overall Assessment

The Iteration 4 work is strong: mock-LM integration tests pass (12/12), retraction double-counting is fixed and verified, temporal sweep reveals a 49x asymmetry in retraction rates ("before" vs "after" DATE constraints), and the cost model achieves R²=0.90/0.98. However, after five iterations without a live LLM experiment, a newly identified **abstraction conflict in `INCREMENTAL_SYSTEM_PROMPT`** now threatens to make that experiment meaningless when it finally runs: the prompt instructs `entity_cache = {}` (a plain Python dict), while the REPL injects `_incremental.entity_cache` (an `EntityCache` object with retraction support). A real LLM following the prompt literally will bypass the retraction primitives entirely. Fixing this conflict is the gate condition for a meaningful API experiment — and the API experiment is now iteration 5's mandatory deliverable.

---

## Reflection on Prior Feedback

Iteration 4 addressed all five priorities I raised: mock-LM test (12/12 pass), retraction double-counting fix (verified via three dedicated unit tests), `compute_gold_pairs` deduplication (done), hot-loop imports (cleaned up), weighted savings computed (39% at k=5, 57% at k=10). I'm dropping all five from this critique. I'm also dropping the `re` import concern (fixed in Iteration 3) and the `compute_gold_pairs` duplication concern (resolved).

The researcher's pushback that the temporal sweep produced a novel finding (49x before/after retraction asymmetry) is valid — I concede that point. However, I maintain that the σ-parameterized cost model exists only as a gap analysis, not as an actual predictive formula that takes σ as input.

---

## Scores

| Criterion | Score | Delta | Comment |
|-----------|-------|-------|---------|
| Novelty | 7/10 | +1 | Temporal retraction asymmetry (49x) is publishable. Cost model is a contribution. But `savings(k, σ)` notation overstates what's been fitted — σ is not in any formula. |
| Technical Soundness | 6/10 | -1 | Double-counting fixed and tested. Correctness validated across 11 tasks. But INCREMENTAL_SYSTEM_PROMPT / `_incremental` abstraction conflict is a correctness risk that invalidates the planned live experiment in its current form. |
| Benchmark Performance | 6/10 | +0 | Mock pipeline validated. Cost model fits well. Still zero live LLM results after five iterations. |
| Scalability | 5/10 | +0 | Architecture is ready but untested with a real model. Persistence still only supported on `local` environment. |
| Research Maturity | 6/10 | +1 | Temporal finding and cost model are near-publishable. Mock-LM validates machinery. Gap: LLM protocol compliance entirely unknown. |

---

## Architecture Review

### Strengths

1. **The incremental pipeline is mechanically correct** — `process_chunk()` validated across 11 tasks × 3 chunk counts × correctness at every chunk. The double-counting fix is complete and properly tested (3 dedicated unit tests verifying `retraction_count == 1` when both entities in a pair are updated). This is solid.

2. **Retraction mechanism handles non-monotonic updates** — the inverted index enabling O(degree) retraction is the right data structure. The 49x temporal asymmetry is a concrete empirical demonstration that retraction cost depends on constraint directionality, not just entity count.

3. **Mock-LM integration confirms pipeline plumbing** — system prompt switching, cached_vars hints, context versioning, history storage all work without API keys. This is the correct precedent to the live experiment.

### Critical Weakness: INCREMENTAL_SYSTEM_PROMPT / Primitive Abstraction Conflict

Reading `INCREMENTAL_SYSTEM_PROMPT` (prompts.py lines 125–181) and `local_repl.py` (lines 168–176) together reveals a serious inconsistency.

**The prompt instructs** (from the "On SUBSEQUENT chunks" section):
```python
entity_cache = {}          # ← plain Python dict (no retraction support)
entity_cache.update(new_classifications)
pair_results.add((new_id, cached_id))
```

**The REPL injects** (local_repl.py lines 172–176):
```python
self.locals["EntityCache"] = EntityCache      # class with versioning
self.locals["PairTracker"] = PairTracker      # class with retraction
self.locals["_incremental"] = IncrementalState()  # instance with process_chunk()
```

These are two parallel, partially overlapping abstractions for the same task. A real LLM receiving `INCREMENTAL_SYSTEM_PROMPT` and following it literally will:
- Create `entity_cache = {}` (a plain dict that shadows nothing)
- Manage retractions manually (or not at all)
- Never call `_incremental.pair_tracker.retract_entity()`

This has two consequences:

1. **No retraction safety**: The LLM following the prompt bypasses the entire retraction mechanism. The claimed savings assume the model handles non-monotonic updates — but the prompt doesn't instruct it to use the retraction-capable primitives.

2. **The mock-LM test doesn't catch this**: `ScriptedMockLM` responses use `entity_cache = {}` (plain dict) in turns 1–2 and `_incremental` in turn 3. This is a scripted coincidence, not a test that a real LLM will do either coherently.

**Concrete fix** — Rewrite the `INCREMENTAL_SYSTEM_PROMPT` protocol section to use `_incremental` as the sole interface:

```python
### On the FIRST chunk (context_0):
# Parse entities into a dict {entity_id: attributes}
entities_chunk0 = parse_entities(context_0)  # implement this yourself
# process_chunk() handles classification storage, pair checking, and retraction:
stats = _incremental.process_chunk(0, entities_chunk0, pair_checker=check_pair)
pair_results = _incremental.pair_tracker.get_pairs()

### On SUBSEQUENT chunks (context_N):
# Parse ONLY the new chunk
entities_new = parse_entities(context_N)
# process_chunk() handles: new vs updated, retraction, re-evaluation, merge
stats = _incremental.process_chunk(N, entities_new, pair_checker=check_pair)
pair_results = _incremental.pair_tracker.get_pairs()  # always current
```

This reduces the model's responsibility to implementing `parse_entities()` and `check_pair()` — the hard incremental state management is handled by `_incremental`. The model no longer needs to manually track `processed_ids`, manage retractions, or deduplicate pairs.

After this fix, update the mock-LM scripted responses to match the new protocol, re-run the 12 integration tests, then proceed to the live experiment.

### Secondary Weakness: Underscore Variable Suppression (carry-over, still unfixed)

`execute_code` (local_repl.py line 380): `if key not in self.globals and not key.startswith("_"): self.locals[key] = value`

If model code executes `_incremental = IncrementalState()` (a reasonable reset), the new instance is silently dropped. `_incremental` persists from `setup()` and is never overwritten because the filter prevents it. Current usage (in-place mutation via `.entity_cache.add()`) works, but this is fragile and undocumented. The fix is an explicit allowlist:

```python
INJECTED_NAMES = {"_incremental"}
for key, value in combined.items():
    if key not in self.globals and (not key.startswith("_") or key in INJECTED_NAMES):
        self.locals[key] = value
```

This was raised in Critique 4 and not addressed. With the prompt fix pointing users to `_incremental`, this becomes more important — the model may reasonably try to reset it.

---

## Novelty Assessment

### What's genuinely novel

1. **Non-monotonic incremental computation** — characterized across a 360x retraction range (44 to 15,824 per 5 chunks), validated with 100% correctness. The finding that retraction cost is determined by condition selectivity (not just entity count) is the key insight.

2. **Temporal retraction asymmetry** — "before DATE" constraints produce 10-49x fewer retractions than "after DATE" constraints. Mechanism hypothesis: "before" cutoffs cause monotonic invalidation (entities stabilize quickly), "after" cutoffs cause bidirectional flips (valid→invalid→valid as new pre/post-cutoff instances arrive in later chunks). This is testable via per-entity retraction frequency analysis.

3. **Savings scaling law** — the cost model (`pair_savings ≈ 0.52*(1-2.78/k)`, R²=0.90) with break-even at k≥4 is a practitioner-relevant contribution.

### What's overclaimed: the `savings(k, σ)` notation

The research log and researcher response describe a function `savings(k, σ)` where σ = task selectivity. But the fitted models are `savings(k) = a*(1-b/k)` — there is **no σ term**. What exists is a gap analysis showing that σ explains 3–6 percentage points of residual variance, narrowing as k increases. This is worth reporting, but it's not the same as having a formula that takes σ as an input to produce a prediction.

Until `savings(k, σ)` is actually derived and fitted, the notation should be `savings(k)` with a footnote that task selectivity adds ≤6pp variance at small k. Overclaiming the formula will be noticed by reviewers who check equations.

**Path to a genuine `savings(k, σ)` formula**:
Define σ as `final_pairs / C(n_users, 2)` (fraction of pairs satisfying the condition — computable from existing simulation results). Fit `savings(k, σ) = a*(1 - b/k) + c*σ*(1 - d/k)` or a similar additive two-factor form. You already have 28 datapoints — sufficient for a 4-parameter fit. If σ doesn't add meaningful R² (< 0.02 gain), report that explicitly: "savings are well-predicted by chunk count alone, regardless of task selectivity at k≥5." That negative result is itself clean and publishable.

### What needs scrutiny: weighted savings methodology

The headline "~39% weighted savings at k=5" uses:
```
weighted_savings = 0.78 × entity_parse_savings + 0.22 × pair_check_savings
```

The 78%/22% weights come from Experiment 1: a full (non-incremental) API run where sub-model tokens were 78% of total. But there's a hidden assumption: that "entity parsing" and "pair checking" each consume the same *fraction* of sub-model tokens in both the full and incremental pipelines.

In the incremental pipeline, entity parsing tokens drop ~44% and pair-checking tokens drop ~22%, so the total token budget is ~61% of the full pipeline. Within that smaller budget, the proportions may differ. More importantly, the 78% figure from Experiment 1 includes ALL sub-model overhead (including context loading, prompt overhead, etc.) — not just entity classification. The clean separation into "entity parsing = 78%" and "pair checking = 22%" may not reflect how sub-model tokens are actually allocated.

This doesn't necessarily invalidate the 39% headline, but it needs a one-paragraph caveat in the paper: the weighting assumes that sub-model token proportions in the incremental pipeline are the same as in the full pipeline. This assumption can only be verified with instrumented incremental API runs.

---

## Experiment Critique

### What's well-designed

- Correctness validation at every chunk for every task — this is the right standard for incremental computation experiments.
- Per-chunk savings curves showing the warm-up effect — negative savings in early chunks of strict-condition tasks is a publishable finding.
- Temporal sweep (5 and 10 chunks, 5 tasks) with correctness validation.
- 28-datapoint cost model with holdout validation on temporal tasks.

### Critical gaps

**1. The real LLM experiment has been deferred for 5 iterations — this is now a hard stop.**

The mock-LM test validates that *if* a model follows the protocol, the pipeline works. It does not test what a real LLM does. The paper's central claim — "an LLM instructed with our incremental protocol achieves X% savings" — cannot be supported by simulation alone. Even a single 3-chunk task with gpt-4o-mini (~$2–5) would produce data that no simulation can substitute.

The specific metric that must be measured: **protocol compliance rate** = fraction of turns where model code references `_incremental`/`entity_cache` (on turns 2+) rather than re-reading `context_0` from scratch. This can be computed by log parsing without additional API calls.

**Dead-end warning**: If the live experiment is again deferred to Iteration 6, the simulation-only approach should be explicitly labelled as a *theoretical analysis* section, not an empirical result. That changes the framing of the contribution from "our system achieves X% savings" to "our architecture enables up to X% savings, assuming LLM compliance." The weaker framing is defensible, but it needs to be chosen consciously.

**2. k=4 break-even not validated** — the cost model predicts ~10% savings at k=4. This is the most practically important threshold (when does incremental computation become worthwhile?). It costs exactly 5 minutes and zero code changes:
```bash
python eval/incremental_simulation.py --tasks 1,3,6,11,19 --num-chunks 4
```

**3. 3-chunk temporal sweep missing** — needed to complete the cost model dataset. This was on the researcher's own Iteration 5 to-do list and requires only:
```bash
python eval/incremental_simulation.py --tasks 4,5,7,9,10 --num-chunks 3
```

**4. Mechanistic validation of temporal asymmetry** — the "monotonic invalidation vs bidirectional flips" hypothesis is testable by analyzing per-entity retraction frequency from existing chunk_log data. Do "after DATE" entities retract multiple times across chunks (bidirectional), while "before DATE" entities retract at most once (monotonic)? A 30-line analysis of `incremental_temporal_5chunks.json` would confirm or refute this mechanistic claim.

**5. Lazy retraction not prototyped** — at k=10, symmetric tasks show 24,686–27,528 retractions (~18–20% of all pair checks). Lazy retraction (deferring retractions to query time) could reduce per-chunk overhead at the cost of a weaker consistency guarantee. This is a genuine architectural alternative. Even a paper-and-pencil analysis of the correctness tradeoff (strong consistency vs eventual consistency at query time) would strengthen the "design space" contribution.

---

## The One Big Thing

**Fix `INCREMENTAL_SYSTEM_PROMPT` to use `_incremental` primitives exclusively, update mock-LM tests, then run the real API experiment.**

These are three steps but they form a gate: running the API experiment before fixing the prompt tests the wrong interface (dict-based protocol that bypasses retraction). Fixing the prompt first makes the experiment meaningful. Updating mock-LM tests confirms the fixed prompt doesn't break existing pipeline validation. Then the live experiment measures real LLM compliance.

Estimated time: 30 min (prompt fix) + 30 min (test update + re-run) + 2–3 hrs (live experiment run + log analysis) = ~4 hrs total. This is the highest-ROI investment remaining.

---

## Specific Experiments to Run

1. **Fix INCREMENTAL_SYSTEM_PROMPT (30 min)**. Replace the "On SUBSEQUENT chunks" code example with `_incremental.process_chunk(N, entities_new, pair_checker)`. Remove manual `entity_cache = {}`, `pair_results.add()`, `processed_ids.update()` guidance. The model only needs to implement `parse_entities()` and `check_pair()`.

2. **Update and re-run mock-LM tests (30 min)**. Rewrite `TURN2_RESPONSE_ITER1` and `TURN3_RESPONSE_ITER1` in `test_mock_lm_integration.py` to use the new `_incremental.process_chunk()` interface. Confirm 12/12 still pass.

3. **Real API experiment (2–3 hrs, ~$5)**. Run `RLM(persistent=True)` on Task 1 (or Task 6, broadest condition) of OOLONG-Pairs with 3 chunks using gpt-4o-mini. Measure: (a) does turn 2 response reference `_incremental`? (b) does it re-read `context_0`? (c) is `pair_results` correct vs gold? Log parsing gives (a) and (b) for free. Even failure (model doesn't follow protocol) is publishable data.

4. **k=4 simulation (5 min)**:
   ```bash
   python eval/incremental_simulation.py --tasks 1,3,6,11,19 --num-chunks 4
   ```
   Validates break-even claim. If savings at k=4 are in the range 8–12% (as predicted), the cost model is confirmed. If negative, break-even shifts to k=5 — that's still a publishable finding.

5. **3-chunk temporal sweep (5 min)**:
   ```bash
   python eval/incremental_simulation.py --tasks 4,5,7,9,10 --num-chunks 3
   ```
   Completes cost model dataset. Needed before fitting σ-parameterized formula.

6. **Fit σ-parameterized cost model (1 hr)**. Compute σ for each task as `final_pairs / C(n_users, 2)`. Fit `savings(k, σ) = a*(1 - b/k) + c*σ*(1 - d/k)`. Report R² improvement over σ-free model. If <0.02 gain: report "k alone suffices" — a clean negative result. If ≥0.02 gain: report the formula with σ and provide the practitioner lookup table.

7. **Per-entity retraction frequency analysis (30 min)**. Load `incremental_temporal_5chunks.json` chunk_log. For tasks 5 and 7, compute max_retractions_per_entity. Test: is max > 1 significantly more common in task 7 ("after") than task 5 ("before")? This validates the mechanistic hypothesis for the 49x asymmetry.

---

## Code Issues Found

1. **`INCREMENTAL_SYSTEM_PROMPT` creates shadow dict instead of using primitives** (prompts.py lines 151–164). The protocol section shows `entity_cache = {}` — a plain dict — rather than `_incremental.entity_cache`. A real LLM following the prompt will bypass `EntityCache`, `PairTracker`, and retraction entirely. **This is the highest-priority code fix**: it gates the meaningful live experiment.

2. **`execute_code` silently suppresses `_incremental` reassignment** (local_repl.py line 380). The condition `not key.startswith("_")` prevents model code from resetting `_incremental`. In-place mutation works (current usage), but reassignment fails silently. After the prompt fix drives the model to use `_incremental` more actively, this will become a real failure mode. Add an allowlist for `INJECTED_NAMES = {"_incremental"}`.

3. **Cost model formula notation inconsistency in research log**. The log describes `savings(k, σ)` but the fitted formulas contain only `k`. Either fit the σ term or change notation to `savings(k)` with a "σ contributes ≤6pp" footnote. Reviewers will check the formulas.

4. **`_build_iteration_summary` uses brittle `"="` heuristic for variable extraction** (history_manager.py lines 217–224). The pattern `if "=" in stripped and not stripped.startswith("#")` incorrectly captures comparison expressions (`if a == b`), subscript assignments (`entity_cache[k] = v`), f-strings with `=`, and augmented assignments (`count += 1`). The extracted "variable names" fed into the summary are unreliable for complex REPL code. Consider using `ast.parse()` and walking `ast.Assign` nodes instead.

5. **`generate_turn_summary` is tightly coupled to `format_iteration()` string format** (history_manager.py line 275). The pattern `if "REPL variables:" in content` will silently fail if `format_iteration()` changes its output format. These two functions need to share a constant, or the test coverage needs to include a round-trip test that `generate_turn_summary` produces non-empty summaries from `format_iteration()` output.

6. **`_all_after` / `_all_before` return `False` for zero-instance users** (eval/utils.py lines 278–289). `labeled = [inst for inst in instances if inst["label"] == label]` followed by `bool(labeled)` means a user with zero instances of the constrained label automatically fails the condition. This is the correct semantics (you can't have "all instances after DATE" if you have zero instances), but it's worth documenting since it causes task 4's condition to fail for location-only users even when the location requirement is satisfied. The asymmetric handling of "must have at least one [label]" vs "all [label] instances must be [direction] DATE" is a subtle interaction that reviewers may question.

---

## Acknowledged Limitations

- No API keys in this environment — the live experiment requires external execution. The mock-LM workaround is correct scaffolding, but cannot substitute for real LLM compliance measurement past iteration 5.
- Memory profiling deferred — acceptable at current dataset scale (231 entities).
- The 4 chars/token heuristic in `HistoryManager._prune_token_budget` remains approximate but acceptable at this stage.
- Persistence only supported on `local` environment — Docker/Modal/Daytona would require implementing `SupportsPersistence`, which is a substantive extension deferred appropriately.
