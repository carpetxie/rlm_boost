# Critique — Iteration 6

STATUS: CONTINUE

---

## Overall Assessment

Iteration 5 completed all nine deliverables: INCREMENTAL_SYSTEM_PROMPT rewritten, persistent flag propagation bug fixed, `_INJECTED_NAMES` allowlist added, AST-based variable extraction, mock-LM tests re-validated (12/12), k=4 break-even confirmed (15.2–15.8%), 3-chunk temporal sweep done, σ-parameterized model fitted (R²=0.936, p=0.025), and temporal asymmetry mechanism confirmed (4× more bidirectional retractions in "after DATE" tasks). The pipeline now uses the correct `_incremental.process_chunk()` interface throughout. However, after six iterations the live API experiment has not been run, which remains the contribution's most serious deficiency. Three newly identified issues demand attention before paper submission: an O(u·n) complexity gap in the scalability claim, a role-ordering defect in the history pruner that can produce ill-formed message sequences, and a sign-convention inconsistency in the σ model parameterization.

---

## Reflection on Prior Feedback

All nine Iteration 5 items are resolved — dropping them entirely. I accept the researcher's pushback that the per-entity retraction analysis was ~200 lines of non-trivial code (not my estimated "30 lines"), and that the temporal asymmetry finding was richer than a simple binary test. I also accept that the `savings(k, σ)` notation is now legitimate (p=0.025, fitted formula). I'm dropping the σ-notation overclaim, the `_build_iteration_summary` brittleness concern (fixed with AST parsing), the `_INJECTED_NAMES` issue (fixed), and the `generate_turn_summary` coupling concern (demoted to minor, captured as code issue #6 below).

---

## Scores

| Criterion | Score | Delta | Comment |
|-----------|-------|-------|---------|
| Novelty | 7/10 | +0 | Temporal retraction asymmetry, retraction taxonomy (360× range), and cost model are genuine contributions. σ adds statistical significance but <1pp to weighted savings — reduce its prominence. Dynamic benchmark gap identified but still unfilled. |
| Technical Soundness | 7/10 | +1 | All Iteration 5 bugs fixed. New: O(u·n) complexity of updated-entity loop undocumented; role-ordering defect in `_prune_with_summary`; σ parameterization sign inconsistency. |
| Benchmark Performance | 6/10 | +0 | Simulation validated across 11 tasks × 4 chunk counts, 100% correctness. Weighted savings (39%@k=5, 57%@k=10) rest on unvalidated 78/22 token split. Zero live LLM results. |
| Scalability | 5/10 | +0 | O(u·n) complexity of updated-entity loop challenges scalability at large N. All results on N=231 entities; no cross-N model validation. |
| Research Maturity | 6/10 | +0 | Strong theory and simulation. Retraction taxonomy, temporal asymmetry, cost model are near-publishable. Missing: live compliance data, cross-dataset validation, and a genuinely dynamic benchmark. |

---

## Architecture Review

### Strength: Mechanically Correct Pipeline

All previously identified correctness issues are fixed. `process_chunk()` is validated to produce pairs identical to full-recompute at every chunk for every tested task. The `_incremental` interface is now the consistent control point across system prompt, REPL injection, and tests. This is a solid foundation for a live run.

### New Issue 1: O(u·n) Updated-Entity Sweep Breaks the Scalability Claim

The paper's central complexity claim is O(k·n) for incremental processing vs. O(n²) for full recompute. This holds for NEW entities (k_new × n existing pair checks). However, UPDATED entities follow a different code path in `process_chunk()` (lines 269–287 of `incremental.py`):

```python
for updated_id in updated_ids:          # O(u) outer loop
    ...
    for other_id in all_ids:            # O(n) inner loop
        if other_id == updated_id: continue
        if other_id in new_ids: continue
        canonical = (min(updated_id, other_id), max(updated_id, other_id))
        if canonical in retracted_pairs: continue
        ...check...
```

This is **O(u × n)** per chunk where `u = len(updated_ids)`. At N=231 entities, u is small (most entities appear only once across chunks), making this invisible. At production scale with N=10,000 entities and a 10% update rate per chunk:

- u = 1,000 updated entities per chunk
- Inner loop: 1,000 × 10,000 = **10M pair checks per chunk**
- Compare: full-recompute at N=10,000 is C(10,000, 2)/k = 5M pair checks per chunk at k=10

The incremental approach is **worse** than full recompute in this scenario. The theoretical savings only hold when u ≪ k (update rate much lower than new-entity arrival rate). This is satisfied in OOLONG-Pairs (users rarely appear in multiple chunks), but may NOT hold in real streaming applications (e.g., a user submits multiple queries per time window).

**Required actions before paper submission:**
1. Correct the complexity claim: state O((k + u) × n) per chunk, where k = new entities per chunk and u = updated entities per chunk.
2. Run a parametric experiment: vary the artificial update rate from 0% to 20% and plot pair-check savings vs. update rate. This shows where the incremental approach breaks even with full recompute. Even 2 hours of simulation would produce a publishable "applicability region" characterization.
3. Add a docstring to `process_chunk()`: "Complexity: O((k+u)·n) per chunk where k = new entities and u = updated entities. When u is large relative to n, the updated-entity sweep dominates. Consider reducing chunk granularity to lower u."

### New Issue 2: Role-Ordering Defect in `_prune_with_summary`

In `history_manager.py`, `_prune_with_summary()` returns:

```python
return system_messages + [summary_message, ack_message] + recent_messages
```

`system_messages` (per `_find_system_end`) always ends with the **first user message**. So the concatenated sequence at the seam is:

```
[... user: first_user_prompt]   ← last element of system_messages
[user: PRIOR COMPUTATION SUMMARY]  ← summary_message ← CONSECUTIVE USER TURNS
[assistant: I understand...]    ← ack_message
[assistant: model_response]     ← first element of recent_messages ← CONSECUTIVE ASSISTANT TURNS
[user: repl_output]
```

Two consecutive user messages followed by two consecutive assistant messages is structurally valid for most LLM APIs (OpenAI accepts this), but it creates a semantically incorrect discourse: the "PRIOR COMPUTATION SUMMARY" appears as *user input* rather than as prior *model output*, which is what it actually represents. A model receiving this context may discount the summary as "something the user said" rather than trusting it as its own prior computation.

**Concrete fix** — swap roles so the summary appears as prior model output:
```python
summary_message = {
    "role": "assistant",
    "content": summary,  # model's prior computation — correct role
}
ack_message = {
    "role": "user",
    "content": "Continue with the current task using the state described above.",
}
```

This produces: `user(first_prompt) → assistant(summary) → user(continue) → assistant(recent_response) → user(repl_output) → ...` — properly alternating. Run the 12 existing integration tests after the fix to confirm no regression.

### New Issue 3: σ Model Sign Inconsistency

`sigma_cost_model.py` defines `model_sigma_param(X, a, b, c, d)` with functional form `a*(1-b/k) + c*σ*(1-d/k)`. The optimizer drives `d` to -1.60 (negative, because high-σ savings are LARGER at small k, requiring the σ term to grow as k→0). With d=-1.60, the formula evaluates as `c*σ*(1-(-1.60)/k) = c*σ*(1+1.60/k)` — which is what the research log correctly reports.

The issue: the code's `bounds=([0, 0, -500, -10], [200, 10, 500, 10])` allows d<0 silently, making it easy to accidentally interpret `d` as positive in the paper formula. A reader seeing `savings(k,σ) = a(1-b/k) + c·σ·(1-d/k)` with no stated sign for d would assume all parameters positive, evaluate `(1-1.60/k)` instead of `(1+1.60/k)`, and get wrong predictions.

**Fix**: Reparameterize with an explicitly positive coefficient:
```python
def model_sigma_param(X, a, b, c, e):
    """savings(k,σ) = a*(1-b/k) + c*σ*(1+e/k), all params positive."""
    k, sigma = X
    return a * (1.0 - b / k) + c * sigma * (1.0 + e / k)
```
With `bounds=([0, 0, 0, 0], [200, 10, 500, 10])`. Report the formula in the paper as:
```
savings(k, σ) = 51.1·(1 - 2.93/k) + 8.9·σ·(1 + 1.60/k)
```
This matches the current result (e=1.60) with unambiguous sign conventions.

---

## Novelty Assessment

### Genuinely novel (maintain these as primary contributions)

1. **Retraction taxonomy**: 360× range (44–15,824 retractions at k=5) explained by condition type (symmetric, temporal-direction, asymmetric cardinality), not entity count. The finding that retraction cost is primarily determined by condition semantics — not entity volume — is a novel empirical characterization of incremental computation cost.

2. **Temporal retraction asymmetry**: "Before DATE" constraints → monotonic invalidation → 44 retractions; "After DATE" → bidirectional entity validity oscillation → 2,151 retractions. 4× higher bidirectional rate (10.4% vs 2.6% of entities) confirmed mechanistically. Directly actionable for practitioners (know your constraint direction before choosing between lazy and eager retraction).

3. **Empirical cost model with break-even**: `savings(k) ≈ 52%(1-2.84/k)`, k≥4 positive break-even, R²=0.919. Practitioner-ready: given chunk count, predict savings within 5pp for any task type.

### What σ actually contributes (reduce prominence)

The σ term adds p=0.025 significance but only **0.9pp to weighted savings** at typical parameters:
```
weighted_savings_σ_contribution = 0.22 × σ_contribution_to_pair_savings
                                = 0.22 × [8.9 × 0.35 × (1 + 1.60/5)]   # at σ=0.35, k=5
                                = 0.22 × 4.1pp ≈ 0.9pp
```

For symmetric tasks (σ=0.3–0.4), σ adds <1pp to the 39% headline. This is below noise for any practical comparison. **Recommendation**: Move the σ model to a supplementary section. Lead with the σ-free formula in the main paper. Report: "Task selectivity (σ) is statistically significant (p=0.025) but practically negligible (<1pp impact on weighted savings). Practitioners can use the σ-free formula for all task types."

### What's still missing for full novelty

**The dynamic benchmark gap is identified but not filled.** The OOLONG-Pairs chunked simulation is a valid proxy, but it is not a genuine dynamic benchmark. Key properties absent from the simulation:
- True temporal ordering (chunks are artificial partitions of static data)
- Entity deletion events (pairs involving "removed" entities should be retracted)
- Variable chunk arrival rates
- Out-of-order entity instances

For the paper to claim "Dynamic RLM" in the title, at least one experiment should use data with genuine temporal dynamics. A minimal addition: take a subset of the arXiv or Wikipedia edit history where "users" are documents and "instances" are edits arriving over time. Even 5 documents × 100 edits each would demonstrate the approach on real dynamic data, distinguishing it from "batch processing in smaller batches."

---

## Experiment Critique

### Well-designed (don't revisit)
- Correctness validation at every chunk for every task — right standard
- Per-chunk savings curves (negative savings at k≤3 for strict tasks — publishable boundary)
- Temporal sweep with mechanism confirmation (tasks 4,5,7,9,10 at k=3,5,10)
- 35-point cost model with F-test and holdout validation

### Critical Gap 1: Live API Experiment — Sixth Consecutive Deferral (Terminal)

The researcher's Iteration 5 response included an explicit commitment: "If the live experiment is again deferred in Iteration 6, I will explicitly reframe the contribution as 'theoretical analysis + mock-LM pipeline validation.'"

I am holding this commitment. If the experiment ran, provide:
1. **Protocol compliance rate**: fraction of Turn 2+ model responses containing `process_chunk` (log grep)
2. **Re-read rate**: fraction of Turn 2+ responses referencing `context_0` (a compliance failure)
3. **F1 vs. gold pairs**: accuracy of the incremental result vs. ground truth
4. **Actual token split**: Turn 1 vs. Turn 2 token counts (validates or refutes the 78/22 weight assumption)

If the experiment did not run, the contribution must be reframed immediately. The reframing is:

> "We design and formally characterize an incremental computation architecture for RLMs. We prove correctness on 11 task types and derive a practitioner cost model. LLM protocol compliance — whether LLMs reliably follow the incremental protocol without fine-tuning — is an open empirical question; our mock-LM tests validate the pipeline infrastructure under scripted compliance."

This reframing is defensible and honest. It also creates a natural bridge to Thrust 1 (fine-tuning): fine-tuning may be a prerequisite for achieving the predicted savings in practice.

**If compliance is low (< 50%)**, this is a stronger finding than compliance >50%, not a weaker one. It means: "Zero-shot LLMs do not naturally follow incremental computation protocols, which motivates fine-tuning as a necessary prerequisite." This connects both thrusts of the research into a single coherent narrative.

### Critical Gap 2: No Cross-N Validation of Cost Model

The formula `savings(k) = 52.16*(1-2.84/k)` was fitted on N=231 entities. All 35 data points come from the same dataset. The model may not generalize to different entity counts — the `b` parameter (break-even numerator) could depend on N.

**Theoretical expectation**: Since savings is a ratio (incremental_checks / full_checks), the formula should be N-invariant when chunks are proportional to N. But this needs empirical confirmation.

**Required**: Run `incremental_simulation.py` on tasks 1 and 19 at k=5 and k=10 using a subsampled dataset (take the first 100 users from OOLONG-Pairs) and a denser dataset (repeat the 231 users to create N=462). If the predicted savings (22.1% and 38.8%) hold within 5pp for different N, the formula is N-invariant — report this as a robustness finding. If they don't, report the N-dependence as a calibration requirement.

**Cost**: 30 minutes. Data already on disk. Only requires adding a `--max-entities` flag to `incremental_simulation.py`.

### Critical Gap 3: Lazy Retraction Analysis — Three-Iteration Deferral → Downgrade to Paper-and-Pencil

Lazy retraction has been deferred across Iterations 2, 4, and 5. Implementing `LazyIncrementalState` is no longer the right ask — that's premature engineering without a correctness analysis.

**Revised ask**: Write a 1-page formal analysis of lazy retraction safety as a section in the paper:

- **Definition**: Lazy = retract pairs at query time (when `get_pairs()` is called), not on chunk arrival. Eager = current behavior (retract immediately in `process_chunk()`).
- **Consistency guarantee**: Lazy is query-consistent (correct at query time); eager is always-consistent. Between queries, lazy may hold stale pairs.
- **Safety condition**: Task conditions with *monotonically decreasing* pair validity are safe for lazy retraction (pairs invalidated once stay invalid — never re-validated). This holds for "Before DATE" and strict cardinality ("exactly N") constraints. It does NOT hold for "After DATE" constraints where entities oscillate between valid and invalid.
- **Empirical support already in hand**: Task 5 ("before DATE") has only 44 retractions, 2.6% bidirectional entities — lazy retraction would change only 6 of 21 final pairs at worst. Task 7 ("after DATE") has 2,151 retractions, 10.4% bidirectional — lazy retraction can produce permanently stale results.

No new code or experiments needed. This analysis completes the design space characterization that the contribution needs.

---

## The One Big Thing

**Run the live API experiment, then use the compliance rate to determine whether the paper's contribution framing is "Empirical System" or "Theoretical Architecture + Infrastructure."**

Both framings are publishable. The wrong path is to maintain the "empirical system" framing while indefinitely deferring the experiment that would validate it. After six iterations, the choice must be made explicitly:

- **If compliance ≥ 50%**: Lead with empirical savings (with caveats on the 78/22 token split assumption). Add actual token counts from the live run to replace/supplement the weighted-savings calculation. This is the stronger paper.
- **If compliance < 50%**: Lead with theoretical analysis + infrastructure. Add a "compliance gap" section showing that zero-shot LLMs don't naturally follow the protocol. Conclude with a call to Thrust 1 (fine-tuning) as the solution. This connects both research thrusts and produces an equally valid — arguably more interesting — contribution.

The specific numbers needed from a single 3-chunk run with gpt-4o-mini (~$2–5):
1. compliance_rate = count(turns with "process_chunk") / count(turns 2+)
2. reread_rate = count(turns referencing "context_0") / count(turns 2+)
3. F1 of final `pair_results` vs. gold
4. token_turn1, token_turn2, token_turn3 (from usage object)

These four numbers determine the entire paper's empirical narrative.

---

## Specific Experiments to Run

1. **Live API experiment (mandatory ≤$5)**: `RLM(persistent=True)` on Task 1, 3 chunks, gpt-4o-mini. Log all turn responses and token counts. Extract compliance rate (grep `process_chunk`), re-read rate (grep `context_0`), F1 vs. gold, and actual token proportions. Even zero compliance is publishable data.

2. **Cross-N cost model validation (30 min, no API)**: Add `--max-entities N` flag to `incremental_simulation.py`. Run tasks 1 and 19 at N=100, 231 (existing), and 462 (doubled). Check if `savings(k=5) ≈ 22.1%` holds within 5pp across all three N values. Report as either "N-invariant (robustness finding)" or "N-dependent (calibration needed)."

3. **Update-rate parametric experiment (2 hrs, no API)**: Modify `run_incremental_simulation` to artificially mark a fraction `p` of existing entities as "updated" on each chunk (with the same attributes — no functional change, just testing the sweep overhead). Run tasks 1 and 19 at p=0%, 5%, 10%, 20% with k=5. Plot pair-check savings vs. update rate. Find the break-even update rate where incremental ≈ full recompute. This empirically characterizes the O(u·n) regime.

4. **Fix σ model parameterization (30 min)**: Reparameterize to `(1+e/k)` with e>0. Re-fit. Confirm R²=0.936 unchanged. Update all formula references in research log and paper draft.

5. **Fix `_prune_with_summary` role ordering (30 min)**: Swap `summary_message` to `"assistant"` role, `ack_message` to `"user"` role. Re-run 12 integration tests. Required before live experiment to ensure pruner doesn't produce semantically ill-formed history.

6. **Lazy retraction safety analysis (2 hrs, no code)**: Write the 1-page analysis section: definition, consistency guarantee, safety condition (monotonic vs. bidirectional validity), empirical support from temporal sweep. No new experiments needed — data is already in `sigma_model_results.json`.

---

## Code Issues Found

1. **O(u·n) complexity of updated-entity sweep undocumented** (`rlm/core/incremental.py`, lines 269–287): The inner `for other_id in all_ids` loop is O(n) per updated entity. This breaks the scalability claim at high update rates. Add docstring: "Complexity: O((k+u)·n) per chunk. When update rate u is high (>k), performance may exceed full recompute cost. Use fewer, larger chunks to minimize u."

2. **Consecutive same-role messages in pruned history** (`rlm/core/history_manager.py`, `_prune_with_summary`): The seam between `system_messages` (ends with user) and `[summary_message, ack_message]` produces `(user, user)` then `(assistant, assistant)`. Swap summary to `"assistant"` role and ack to `"user"` role.

3. **σ model sign ambiguity** (`eval/sigma_cost_model.py`, `model_sigma_param`): Parameter `d` goes negative (d=-1.60) to produce `(1+1.60/k)`, but the code writes `(1-d/k)`. Reparameterize to `e = |d| > 0` and write `(1+e/k)` explicitly.

4. **`EntityCache` has no deletion support** (`rlm/core/incremental.py`): Entities are added but never removed. For streaming scenarios with entity disappearance, stale pairs persist. Add a docstring limitation note: "No deletion support. Entities are retained indefinitely once added."

5. **`_find_system_end` assumption undocumented** (`rlm/core/history_manager.py`, lines 178–194): The function assumes the first user message is always the system-setup boundary. Add docstring asserting this precondition.

6. **`generate_turn_summary` string pattern coupled to `format_iteration` output** (`rlm/core/history_manager.py`, line 275): `"REPL variables:" in content` will silently return empty summaries if `format_iteration()` changes its string format. Add a TODO: `# TODO: Share 'REPL variables:' as a module-level constant with format_iteration()`.

---

## Acknowledged Limitations

- No API keys in this environment — live experiment requires external execution. The six-iteration deferral is an environmental constraint, but the commitment made in Iteration 5 must now produce a result or an explicit contribution reframing.
- Persistence supported only on `local` environment — Docker/Modal/Daytona deferred appropriately.
- Memory profiling deferred — acceptable at N=231.
- The 4 chars/token heuristic in `_prune_token_budget` remains approximate — acceptable.
- "Dynamic benchmark gap" identified but not filled — OOLONG chunking is a proxy, not a genuine streaming benchmark. Acknowledged as a scope limitation for this contribution.
