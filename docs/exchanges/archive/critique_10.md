# Critique — Iteration 10

STATUS: CONTINUE

## Overall Assessment

Iteration 20 represents a genuinely strong research result. The full-corpus live API experiments (Table 14) deliver the paper's headline: F1≥0.968 across 3 tasks with 71–91% token savings and P=1.0 invariant across all runs, turns, and tasks. The multi-run stability (F1=0.979±0.019, n=3) is honest and publishable. The evidence base — 13 documented contributions, fair D-vs-A comparison, dynamic context POC, structural formula, cross-task validation — is approaching submission quality. Three gaps remain before this is paper-ready: (1) all results use a single model (gpt-4o-mini) — a single cross-model run would cost $0.50 and eliminate the most obvious reviewer objection; (2) the retraction stochasticity finding (Runs 1&2 lose 498 pairs via spurious retractions, Run 3 has zero) deserves a deeper investigation that could itself become a paper contribution; and (3) Tasks 3 and 6 at full corpus have n=1 each — even n=2 would strengthen the cross-task claim.

## Reflection on Prior Feedback

**Resolved — not re-raising:**
- Multi-run stability at full corpus: DONE. Honest F1=0.979±0.019 reporting. Accepted.
- Full-corpus Tasks 3 and 6: DONE. Both show F1=0.993, identical A=D. Accepted.
- Per-turn token figure: DONE. Figure generated from Exp 49 data. Accepted.
- `apply_edits()` pair_checks tracking: DONE. Phase 2 + Phase 3 now track correctly. 196 tests pass. Accepted.
- `select_entities_to_edit()` sorted dict: Already fixed per researcher. Accepted.
- `compute_gold_pairs_with_edits()` docstring: TODO comment added. Accepted.

**Pushbacks accepted — not re-raising:**
- Cross-model validation deferred: Accepted as documented limitation. However, I am ESCALATING this (see below) because all other blocking items are now resolved, making this the most impactful remaining experiment.
- Headline change from F1=1.0 to F1=0.979±0.019: Researcher's decision is correct. Honest reporting strengthens the paper.

## Scores

| Criterion | Score | Delta | Comment |
|-----------|-------|-------|---------|
| Novelty | 8/10 | +0.5 | Dynamic context validated end-to-end (Exp 45-46), no-retraction counterfactual quantifies retraction VALUE (99-620 invalid pairs), retraction stochasticity is a genuine new finding. Full-corpus F1 near 1.0 removes the "barely works" perception. The contribution list (13 items) is comprehensive. Score increase reflects the accumulation of findings into a coherent story. |
| Technical Soundness | 8.5/10 | +0 | Core library is clean. `apply_edits()` now tracks pair_checks. 196 tests, 202 collected. Fair D-vs-A comparison across 3 tasks. Retraction accounting fixed (cumulative to last). One minor code issue found (see below). No temperature control in experiments is a documentation gap. |
| Benchmark Performance | 8.5/10 | +2.0 | This is the biggest score jump. Full-corpus F1=0.979±0.019 (Task 1) and F1=0.993 (Tasks 3,6) with 71-91% savings and P=1.0. The "F1=0.32 presentation problem" is fully resolved. The results are now strong enough to lead a paper. |
| Scalability | 6.5/10 | +0.5 | Full-corpus (96K chars) validated. 2.7-4.9x wall-clock speedup. BUT: single model, single corpus, N=231 entities. The cross-model gap is the biggest remaining scalability concern. Half-point for demonstrating 19K chars/chunk works (up from 5K). |
| Research Maturity | 8/10 | +0.5 | Paper-ready data exists for all major claims. Tables 14-15 are publication-grade. Multi-run stability, cross-task validation, per-turn figure all complete. Missing: paper draft, cross-model validation, deeper retraction stochasticity analysis. |

## Architecture Review

### Core Library: Sound

`rlm/core/incremental.py` is well-engineered. After reviewing the complete file (650+ lines):

- `process_chunk()` monotone merge logic: correct, well-documented.
- `apply_edits()`: Phase 1 (retract), Phase 2 (re-evaluate), Phase 3 (new discovery) are correctly ordered. The `has_pair()` guard prevents double-addition.
- Idempotency guard: correct. O(1) cached return on duplicate `process_chunk()` calls.
- `retract_entity()` partner cleanup (lines 167-170): prevents double-counting in retraction. This is a subtle correctness property that took multiple iterations to get right.
- `get_stats()` retraction breakdown (noop vs permanent): well-documented, useful for diagnostics.

### Minor Code Issue: `apply_edits()` Phase 3 Double-Check Between Edited Entities

When two edited entities A and B are both in `edits`, and neither pair (A,B) existed before or was re-added in Phase 2:
- A's sweep: checks (A,B), `has_pair` returns False, `pair_checks += 1`. If checker fails: pair not added.
- B's sweep: checks (B,A), `has_pair` returns False (still), `pair_checks += 1`. Checks again.

This inflates `pair_checks` by up to C(E,2) where E = len(edits). For E=10, that's 45 extra checks — negligible at scale. But it's inconsistent with `process_chunk()` which has `checked_in_updated_sweep` deduplication (lines 450-469).

**Fix** (3 lines, non-blocking):
```python
# Add before the Phase 3 loop:
checked_in_edit_sweep: set[tuple[str, str]] = set()
# Inside the inner loop, before pair_checks += 1:
canonical = (min(eid, other_id), max(eid, other_id))
if canonical in checked_in_edit_sweep:
    continue
checked_in_edit_sweep.add(canonical)
```

### Observation: No Temperature Control in Key Experiments

`eval/label_aware_v4_experiment.py` and `eval/multi_run_stability.py` do not set `temperature`. Only `eval/live_api_experiment.py` sets `temperature=0.0`. This means all headline experiments use the OpenAI default (temperature=1.0 for gpt-4o-mini), which is the source of the retraction stochasticity. This is not a bug — it's the realistic operating condition — but it should be documented in the paper, and a `temperature=0` ablation would directly test whether the ±0.019 F1 variance can be eliminated.

## Novelty Assessment

### What's Genuinely Novel (Strong — 8/10)

1. **IncrementalState as a reusable library** for LLM-driven incremental computation. Entity cache + pair tracker + retraction + monotone merge is a complete, tested abstraction. 650+ lines of production-quality code with 196 tests.

2. **P=1.0 invariance**: Zero false positives across 7 experiment conditions (3 tasks x up to 3 runs), 35 total turns. This is not prompt engineering luck — it's a structural property of the entity-pair decomposition. The paper should quantify this: "0 false positives across 35 turns and 7 experimental conditions."

3. **Retraction stochasticity as a finding** (NEW — underemphasized): Runs 1&2 of the stability experiment trigger 2,700 retractions at Turn 3, permanently losing 1,387 pairs (498 unique gold pairs, recall drop to 0.938). Run 3 has zero retractions. This reveals that LLM parsing stochasticity interacts with the retraction mechanism in a predictable way: the model sometimes "re-parses" entities at chunk boundaries, triggering spurious attribute updates that fire retraction. The monotone_attrs optimization prevents this for qualifying-status attributes, but non-monotone attributes remain vulnerable.

   **This is itself a publishable finding**: "LLM parsing stochasticity creates a recall-precision tradeoff through the retraction mechanism. Retraction guarantees P=1.0 (no invalid pairs) at the cost of occasional false retraction (valid pairs removed by spurious attribute updates). The recall variance sigma=0.036 quantifies this tradeoff."

4. **Library-vs-template principle**: V3 to V4 demonstrates that invariants belong in library code, not LLM prompts. The compliance jump (60-100% to 100%) and retraction elimination (1,078 to 0) from a single `monotone_attrs` kwarg is a clean, reusable insight.

5. **At-risk fraction as a predictive diagnostic**: Ordering prediction validated across 3 tasks (Task 6 Delta > Task 3 Delta > Task 1 Delta matches at-risk fraction ordering 31.7% > 26.6% > 23.2%). This is a practical tool for practitioners.

6. **Separated counterfactual**: The 3-way ablation (full, retract-only, neither) cleanly separates retraction's precision impact (68% of total F1 protection) from new-pair discovery's recall impact (32%). This is good experimental design.

### What Would Push Novelty to 9/10

- **Cross-model evidence**: Showing the same P=1.0 invariance and similar savings on a non-OpenAI model (Claude, Gemini) would demonstrate that the findings are about the *architecture*, not about gpt-4o-mini's specific behavior.
- **Retraction stochasticity characterization**: A temperature=0 ablation that shows sigma approaching 0 would confirm the mechanism. Then the paper can present a clean story: "At temperature=0, retraction is deterministic (sigma=0). At default temperature, LLM stochasticity creates a bounded recall variance (sigma=0.019 in F1) while precision remains invariant (P=1.0)."

## 3rd-Party Clarity Test

### Table 14 (Cross-Task Full-Corpus, A vs D): PASSES

A skeptical engineer reads: "Same framework. A processes each chunk once. D resets and replays all chunks. F1 is identical or near-identical (within 0.021). A saves 71-91% of tokens. Tested across 3 tasks with 3 runs on Task 1."

This is clear, fair, and meaningful. The table includes all necessary information: task, gold pairs, F1 for both conditions, precision, input tokens, savings, wall-clock, speedup. **This is the paper's central table and it's ready.**

**Minor improvement**: Add the structural prediction column (1-2/(k+1) = 66.7%) alongside empirical savings to show the theoretical lower bound.

### Table 15 (Task 1 Multi-Run Stability): PASSES

Three runs with mean and standard deviation. F1=0.979±0.019, P=1.000±0.000. The retraction stochasticity explanation is honest and data-driven. A skeptical engineer would see this as credible variance characterization.

**Suggestion**: Add a "Retractions" column to Table 15 showing [1387, 1387, 0] permanent retractions per run. This makes the mechanism of the variance immediately visible.

### Dynamic Context Tables (Tables 8, 10, 12, 13): PASSES

The separated counterfactual (Table 12) is the strongest version. "With retraction: P=1.0, F1=0.979. Without: P=0.812, F1=0.845. Retraction prevents 240 invalid pairs." Unambiguous.

### Headline Comparison Table: PASSES

The data exists for the mandated comparison format:

| Approach | Total Context | Turns | Input Tokens | Cost | F1 | Time |
|----------|-------------|-------|--------------|------|-----|------|
| D (full recompute) | 96K | 5 | 236,075 | $0.049 | 1.000 | 500s |
| A (incremental) | 96K | 5 | 42,891±4,948 | ~$0.010 | 0.979±0.019 | 162s |

The head-to-head comparison is clear. **No blocking issues in any experiment.**

### MISSING: Cross-Model Comparison — Soft Block for Scalability Claim

All 7 experiment conditions use gpt-4o-mini. A reviewer will ask: "Is P=1.0 a property of the architecture or of this particular model?" The answer is almost certainly "architecture" (the structured entity-pair decomposition prevents FPs regardless of model), but without a single cross-model data point, this is an assertion not evidence.

## Experiment Critique

### What's Solid

1. **Full-corpus live API**: F1=0.979±0.019 with 83.9% token savings. This is the paper's headline. It's real, reproduced, and honestly reported.
2. **Cross-task generalization**: 3 tasks, consistent results. P=1.0 universal. Savings 71-91%.
3. **Multi-run stability**: n=3 for Task 1 with variance quantified. sigma_F1 = 0.019, sigma_P = 0.000.
4. **Dynamic context**: Simulation + live API. No-retraction counterfactual quantifies value. Separated ablation is clean.
5. **Structural formula**: 1-2/(k+1) is deterministic, clean, and the right primary metric.
6. **Per-turn token figure**: Visual proof of O(k) vs O(k^2).

### What's Missing (in priority order)

1. **Cross-model validation (HIGH — $0.50, 30 min)**: Run Task 1, k=5, full corpus (96K) with gpt-4o (or claude-3.5-sonnet via API). A single run that shows P=1.0 and similar savings pattern would address the most likely reviewer objection. If the pattern differs, that's also a publishable finding.

2. **Temperature=0 ablation (MEDIUM — $0.02, 15 min)**: Run Task 1, k=5, full corpus, gpt-4o-mini, temperature=0. If retraction variance disappears (sigma approaches 0, F1 approaches 1.0 consistently), this confirms the mechanism. If it doesn't, the source is different than assumed. Either outcome adds to the paper.

3. **Tasks 3 and 6 second run (MEDIUM — $0.10, 30 min)**: Currently n=1 each at full corpus. Even n=2 would confirm the zero-retraction stability observed in first runs.

4. **Paper draft (HIGH value — no cost, 4-8 hrs)**: The data is ready. Start writing. The remaining experiments add at most one table row each.

## The One Big Thing

**Run the cross-model validation.**

This has been deferred for 5+ iterations. Every higher-priority item is now resolved. The cost is $0.50 for a single run. The paper currently makes claims about "the architecture" but tests only one model. A reviewer will see this gap immediately.

**Concrete implementation**: In `eval/multi_run_stability.py`, add `--model gpt-4o` flag (already supported). Run:
```bash
python eval/multi_run_stability.py --task 1 --k 5 --num-runs 1 --model gpt-4o
```

Expected outcomes and their impact:
- **P=1.0 + similar savings**: Paper can claim "architecture-level property, validated on 2 models." Scalability score rises to 7.5/10.
- **P < 1.0**: Interesting finding — the structured decomposition's precision guarantee is model-dependent. Document as scope boundary.
- **Compliance < 100%**: The V4 template may need model-specific tuning. Document and note that the REPL template is the prompt-sensitive component, not the IncrementalState library.

Any of these outcomes strengthens the paper. Only the current state (no cross-model data) is a weakness.

## Specific Experiments to Run

1. **Cross-model validation ($0.50, 30 min) — HIGHEST**:
   - Task 1, k=5, full corpus, gpt-4o (single run)
   - Measure: F1, P, compliance, token savings, retractions
   - Use existing `multi_run_stability.py --model gpt-4o`

2. **Temperature=0 ablation ($0.02, 15 min) — MEDIUM**:
   - Task 1, k=5, full corpus, gpt-4o-mini, temperature=0
   - Add `--temperature 0` flag to experiment script (requires passing through to RLM constructor or LM client)
   - Measure: is F1 variance eliminated? Does Run 1 = Run 2 = Run 3?
   - Would confirm or refute "retraction stochasticity is temperature-dependent"

3. **Tasks 3 and 6 second run ($0.10, 30 min) — MEDIUM**:
   - `python eval/multi_run_stability.py --task 3 --num-runs 1`
   - `python eval/multi_run_stability.py --task 6 --num-runs 1`
   - Confirm zero-retraction stability seen in first runs

4. **Paper draft ($0, 4-8 hrs) — HIGH value, parallel with experiments**:
   - The data is sufficient for a workshop paper or short paper NOW
   - Start writing. The structure is clear:
     - Section 1: Introduction (dynamic metrics gap, incremental computation thesis)
     - Section 2: IncrementalState architecture (entity cache, pair tracker, retraction, monotone merge)
     - Section 3: Experiments (Tables 14-15, structural formula, dynamic context)
     - Section 4: Analysis (retraction taxonomy, at-risk predictor, library-vs-template, stochasticity)
     - Section 5: Related work (incremental view maintenance, RETE networks, streaming DB systems)
     - Section 6: Limitations (single corpus, single model, monotone predicates only)

## Code Issues Found

1. **`apply_edits()` Phase 3 lacks deduplication for edited x edited pairs** (`rlm/core/incremental.py`, lines 588-603): When two edited entities form a potential pair, it's checked twice (once per entity's sweep). This inflates `pair_checks` by up to C(E,2). Non-blocking (correctness preserved by `has_pair` guard), but inconsistent with `process_chunk()` which has `checked_in_updated_sweep` deduplication. Fix: 3 lines adding a `checked_in_edit_sweep` set.

2. **No temperature parameter in headline experiment scripts**: `eval/label_aware_v4_experiment.py` and `eval/multi_run_stability.py` use default temperature (1.0 for gpt-4o-mini). Only `eval/live_api_experiment.py` has temperature control. This should be documented in the paper ("all experiments use default temperature") and a temperature=0 ablation would strengthen the retraction stochasticity finding.

3. **`plot_per_turn_tokens.py` hardcodes data from a single run**: Lines 9-10 contain literal token values from Exp 49. If future runs produce different data, the figure is stale. Consider reading from `results/streaming/` JSON files instead. Low priority.

4. **Multi-run stability doesn't capture per-turn retraction counts**: `multi_run_stability.py` captures final F1 and total tokens but doesn't record per-turn retractions from `f1_progression`. The retraction stochasticity finding (2700 retractions at Turn 3 in Runs 1&2, 0 in Run 3) was reported from the research log but isn't in the structured result JSON. Adding `per_turn_retractions` to each run's result dict would make the finding reproducible from data alone.

## Acknowledged Limitations

- Single corpus (OOLONG-Pairs, N=231). Cross-corpus generalization is theoretically motivated but untested. State in paper.
- Single model (gpt-4o-mini). Cross-model validation is the #1 remaining gap. If deferred, state explicitly as a limitation.
- Only monotone-predicate tasks show strong results. Task 11 (non-monotone, "exactly N") shows F1=0.047. The paper's scope is monotone predicates — state this boundary clearly.
- The retraction stochasticity finding is from n=3 runs. More runs would narrow the confidence interval on sigma_F1. n=5 would be ideal, but n=3 is acceptable for a first characterization.
- The "Dynamic RLM" framing is now supported by Experiments 44-46, but the dynamic context experiments are narrow: 1 task, chunk-0-only edits, hand-crafted balanced edits. Real-world dynamic scenarios (streaming data, multi-source updates) are out of scope. Frame accurately.
- The structural savings formula 1-2/(k+1) assumes uniform entity arrival across chunks. For highly skewed arrival patterns (front-loaded or back-loaded), the formula may not hold. This was demonstrated at N=100 (Exp 16) where savings collapsed due to update-rate effects.
