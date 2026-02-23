# Researcher Response — Iteration 15

STATUS: CONTINUE

## Deliberation

### 1. Table 2 is structurally unfair (BLOCKING)
   - **Agree**: Yes, completely. The old Naive (F1=0) vs Incremental (F1=0.3228) comparison conflated two independent benefits: structural (having a framework) and efficiency (incremental vs recompute). A skeptical reviewer would immediately call this a strawman.
   - Feasible: Yes — implemented in ~150 lines.
   - Impact: **HIGH** — this is the paper's central claim.
   - Action: Implemented Condition D (Full-Recompute RLM). Same IncrementalState framework, but `reset()` + replay all chunks each turn. Result: **F1(D) = F1(A) = 0.3228, tokens(A) = 49.8K vs tokens(D) = 246.2K → 79.8% savings at 100% quality retention.** This is the clean efficiency comparison the paper needs.
   - Code written: Yes — `run_condition_d_full_recompute()` in `eval/label_aware_v4_experiment.py`, `CHUNK_PROMPT_FULL_RECOMPUTE_V4` with unrolled code generation.
   - Bug encountered: First two runs failed (F1=0) due to regex over-escaping in code generation (`\\\\|` → `\\|` in raw string, matching literal backslash instead of pipe). Fixed on third attempt.

### 2. k=3 stability (single run concern)
   - **Agree**: Yes, the 97.1% A/C from a single run was insufficient evidence for a headline claim.
   - Feasible: Yes, ~$0.03.
   - Impact: **HIGH** — k=3 is the paper's best operating point.
   - Action: Ran 3 additional k=3 runs. **All 3 produced F1=0.3326 (σ=0.000), A/C=97.1%, 100% compliance.** Combined with the original run, we have 4 identical k=3 results. k=3 is deterministic and confirmed as the headline.
   - Bug fixed: Previous multi-run function used default `max_chunk_chars=5000` instead of `25000//k=8333` for k=3. Fixed to `25000//args.k`.
   - Code written: Yes — fixed chunk size calculation in `main()`.

### 3. Outlier diagnosis (55 missing pairs)
   - **Agree**: Low-effort, high-value diagnostic.
   - Feasible: Yes, zero API cost.
   - Impact: MEDIUM — one diagnostic paragraph.
   - Action: Created `eval/diagnostics.py --outlier`. Divergence starts at Turn 3 (Δ=43 pairs), stabilizes at Turn 5 (Δ=55). Root cause: ~1 fewer qualifying entity identified at chunk boundaries. Exp32 also had 61 retractions vs 0 in stable runs, confirming transient label instability correctly handled by retraction mechanism.
   - Code written: Yes — `eval/diagnostics.py`.

### 4. k=7/10 compliance degradation
   - **Agree**: Worth documenting.
   - Feasible: Yes, zero API cost.
   - Impact: LOW-MEDIUM — deployment guidance.
   - Action: Created `eval/diagnostics.py --compliance`. No clean entity-count threshold exists (non-compliant turns had 18-28 entities, compliant had 16-56). Non-compliance correlates with low iteration count (1 iteration). Recommendation: use k≤5 for reliable compliance.
   - Code written: Yes — `eval/diagnostics.py`.

### 5. Update paper_summary_tables.py to 5-run means
   - **Agree**: Prevents reviewers from citing worst single run.
   - Feasible: Yes, zero API cost.
   - Impact: MEDIUM — presentation correctness.
   - Action: Updated Table 3 Task 1 from 0.3131/91.4% (single worst run) to 0.3209/93.7% (5-run mean). Updated Table 4 k=5 tok ratio from 4.23× (outlier) to 1.80× (5-run mean). Redesigned Table 2 as 3-way D/A/C comparison. Added Table 2b for structural comparison. Added Table 6 for diagnostics.
   - Code written: Yes — `eval/paper_summary_tables.py` rewritten.

### 6. Safety invariant docstring for process_chunk()
   - **Agree**: Important for future maintainability.
   - Feasible: Yes, 5 minutes.
   - Impact: LOW — code quality.
   - Action: Added docstring: "SAFETY INVARIANT: When monotone_attrs is provided, pair_checker must depend exclusively on the declared monotone attributes..."
   - Code written: Yes — `rlm/core/incremental.py`.

### 7. Token variance caveat
   - **Agree**: Worth documenting in limitations.
   - Impact: LOW — honest reporting.
   - Action: Documented in Table 1 narrative.

### 8. Missing dynamic context experiment
   - **Agree**: Valid concern but correctly scoped as non-blocking for current paper.
   - Impact: LOW for current paper (scoped as "streaming static context").
   - Action: Deferred to next iteration.

## Code Changes

| File | Change | Lines |
|------|--------|-------|
| `eval/label_aware_v4_experiment.py` | Added `run_condition_d_full_recompute()`, unrolled code generation, `--condition-d` flag, fixed multi-run chunk size bug | ~200 |
| `eval/paper_summary_tables.py` | Redesigned: 7 tables, 5-run means, dynamic Condition D loading | Full rewrite |
| `eval/diagnostics.py` | NEW: outlier + compliance diagnostics | ~200 |
| `rlm/core/incremental.py` | Safety invariant docstring on monotone optimization | 8 lines |

## Experiments Run

| Experiment | Config | Result | Cost |
|-----------|--------|--------|------|
| 37: Condition D (3 attempts) | Task 1, k=5, gpt-4o-mini | F1=0.3228, 246K tokens | ~$0.12 |
| 37: Condition A (in same run) | Task 1, k=5, gpt-4o-mini | F1=0.3228, 50K tokens | ~$0.02 |
| 37: Condition C (oracle) | Task 1, 25K chars | F1=0.3424, 25K tokens | ~$0.01 |
| 38: k=3 stability (3 runs) | Task 1, k=3, 8333 ch/chunk | F1=0.3326 ×3, σ=0.000 | ~$0.03 |
| 39: Outlier diagnosis | Exp32 vs MR1 JSON analysis | ~1 entity diff, 55 pairs | $0 |
| 40: Compliance analysis | k-sensitivity JSON analysis | No threshold, k≤5 safe | $0 |

## Benchmark Results

| Metric | Before (Iter 14) | After (Iter 15) | Delta |
|--------|---------|---------|-------|
| Table 2 fairness | ❌ Strawman (Naive F1=0) | ✅ **D=A=0.3228, 79.8% token savings** | BLOCKING resolved |
| k=3 stability | 1 run | **4 runs, σ=0.000** | Confirmed |
| Outlier | Unknown | ~1 entity, 3.6% pairs | Documented |
| Compliance | Undocumented | No threshold, k≤5 | Documented |
| Table 3 Task 1 F1 | 0.3131 (worst run) | **0.3209 (5-run mean)** | Corrected |
| Table 4 k=5 tok ratio | 4.23× (outlier) | **1.80× (5-run mean)** | Corrected |

## Research Log Updates
- Added Iteration 15 with Experiments 37-40
- Updated paper-ready tables (definitive Table 2 with Condition D)
- Updated cumulative results summary
- Documented k=3 as confirmed best operating point

## Pushbacks
None this iteration. All critique points were valid and addressable.

## Novel Finding: Condition D Prompt Engineering Difficulty

The full-recompute strategy required 3 attempts to implement correctly due to prompt/code generation complexity (globals() failure, regex escaping bug). The incremental approach's prompt is simpler (15 lines, one `process_chunk` call) while full-recompute requires increasingly complex prompts (unrolled code grows linearly with k). This is an additional **reliability** argument for incremental processing beyond token efficiency.

## Next Experiments
1. **Second Condition D run** — confirm 79.8% savings reproducibility
2. **Dynamic context proof-of-concept** — genuine entity updates mid-stream
3. **Retraction taxonomy bounds** — formalize 360× range into predictive model
4. **Cross-task Condition D** — run D vs A on Tasks 3/6
