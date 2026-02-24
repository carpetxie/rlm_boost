# Researcher Response — Iteration 20

STATUS: CONTINUE

## Deliberation

### 1. Multi-run stability at full corpus (HIGHEST priority)
   - **Agree**: Completely. A single n=1 result claiming F1=1.0 is not credible.
   - **Feasible**: Yes, ~$0.02 per run.
   - **Impact**: HIGH — this converts the headline from anecdotal to statistical.
   - **Action**: Ran 3 full-corpus Task 1 runs. **Result: F1=0.979±0.019, NOT 1.0.**
     - 2 of 3 runs hit F1=0.968 due to retraction stochasticity at Turn 3
     - 1 run achieved F1=1.0 (the original Exp 49 was this lucky run)
     - P=1.000 in ALL runs — precision is deterministic
   - **Code written**: Yes — `eval/multi_run_stability.py`, `scripts/run_stability.sh`
   - **Honest reporting**: The headline must change from "F1=1.0" to "F1=0.979±0.019". This is actually a stronger paper — it shows we report variance honestly. P=1.0 invariance is the real structural result.

### 2. Full-corpus Tasks 3 and 6 (HIGH priority)
   - **Agree**: Essential. Without cross-task, "Task 1 is uniquely easy" is a valid dismissal.
   - **Feasible**: Yes, ~$0.05 per task.
   - **Impact**: HIGH — removes single-task criticism entirely.
   - **Action**: Ran both tasks with A+D live API.
     - Task 3: F1=0.993 (identical A=D), 71.4% savings, 2.7× speedup
     - Task 6: F1=0.993 (identical A=D), 90.9% savings, 4.9× speedup
     - P=1.0, 100% compliance for both tasks
   - **Code written**: Used `eval/multi_run_stability.py --task {3,6} --include-d`

### 3. Per-turn token comparison figure (MEDIUM priority)
   - **Agree**: Visual proof of O(k) vs O(k²) is essential for paper.
   - **Feasible**: Yes, free — uses existing data.
   - **Impact**: MEDIUM — makes the efficiency story visually obvious.
   - **Action**: Created `eval/plot_per_turn_tokens.py`, saved figure.
   - **Code written**: Yes

### 4. `apply_edits()` pair_checks tracking (code bug)
   - **Agree**: Telemetry gap that would confuse future users.
   - **Action**: Fixed. Phase 2 and Phase 3 now track pair_checks, increment `_total_pair_checks`, and return count in result dict. 196 tests pass.
   - **Code written**: Yes — `rlm/core/incremental.py`

### 5. `select_entities_to_edit()` sorted dict iteration
   - **Already fixed**: The code already uses `sorted(qualifying.items())`. The critique was about a version that no longer exists.

### 6. `compute_gold_pairs_with_edits()` doesn't use `_check_pair_condition()`
   - **Agree**: Should be documented. Added TODO comment in docstring.
   - **Impact**: Low — only matters for asymmetric tasks (not in scope).

### 7. Cross-model spot check (LOW-MEDIUM)
   - **Deferred**: Would cost ~$0.50 and the cross-task validation is more impactful for the paper. Cross-model remains a documented limitation.

### 8. Temperature/seed for live experiments
   - **Partial agree**: Adding temperature=0 would help, but the multi-run stability test already characterizes the variance. The retraction variance is an important finding — it shows model stochasticity interacts with the retraction mechanism.

## Code Changes

| File | What | Result |
|------|------|--------|
| `rlm/core/incremental.py` | Fixed `apply_edits()` to track `_total_pair_checks` in Phase 2 (re-evaluate) and Phase 3 (new discovery). Returns `pair_checks` in result dict. | 196 tests pass |
| `eval/dynamic_context_experiment.py` | Added TODO comment to `compute_gold_pairs_with_edits()` about simplified qualifying check | Documentation |
| `eval/multi_run_stability.py` | **NEW**: Multi-run stability experiment runner with --task, --k, --num-runs, --include-d | Used for Exps 51-52 |
| `scripts/run_stability.sh` | **NEW**: Bash runner for stability experiments across tasks | Convenience wrapper |
| `eval/plot_per_turn_tokens.py` | **NEW**: Per-turn token comparison figure generator | Figure saved |
| `docs/research_log.md` | Updated headline, added Exps 51-53, updated cumulative summary | Research log current |

## Experiments Run

### Exp 51: Multi-Run Stability (Task 1, k=5, n=3, live API)
- **Config**: gpt-4o-mini, 96,689 chars, 5 chunks of ~19K chars
- **Results**: F1=0.979±0.019 (range: 0.968–1.000), P=1.000±0.000, Compliance=100%
- **Cost**: ~$0.03 total (3 runs × ~$0.01)
- **Key finding**: Retraction stochasticity at chunk boundaries causes ±2% F1 variance

### Exp 52a: Full-Corpus Task 3 (A+D, live API)
- **Config**: gpt-4o-mini, 96,689 chars, 5 chunks, Task 3
- **Results**: F1(A)=F1(D)=0.993, 71.4% token savings, 2.7× speedup
- **Cost**: ~$0.04

### Exp 52b: Full-Corpus Task 6 (A+D, live API)
- **Config**: gpt-4o-mini, 96,689 chars, 5 chunks, Task 6
- **Results**: F1(A)=F1(D)=0.993, 90.9% token savings, 4.9× speedup
- **Cost**: ~$0.05

### Exp 53: Per-Turn Token Figure
- **Config**: Existing Exp 49 data
- **Output**: results/streaming/per_turn_token_comparison.png

## Benchmark Results

### Paper-Ready Cross-Task Comparison (Table 14)

| Task | F1(A) | F1(D) | P(A) | Tokens(A) | Tokens(D) | Savings | Speedup |
|------|-------|-------|------|-----------|-----------|---------|---------|
| Task 1 (n=3) | 0.979±0.019 | 1.000 | 1.000 | 42,891 | 236,075 | 81.8% | 3.1× |
| Task 3 | 0.993 | 0.993 | 1.000 | 51,132 | 179,033 | 71.4% | 2.7× |
| Task 6 | 0.993 | 0.993 | 1.000 | 27,277 | 301,263 | 90.9% | 4.9× |

### Task 1 Multi-Run Stability (Table 15)

| Metric | Mean±Std | Min | Max |
|--------|----------|-----|-----|
| F1 | 0.979±0.019 | 0.968 | 1.000 |
| Precision | 1.000±0.000 | 1.000 | 1.000 |
| Recall | 0.959±0.036 | 0.938 | 1.000 |
| Compliance | 100%±0% | 100% | 100% |
| Input Tokens | 42,891±4,948 | 38,567 | 48,287 |
| Wall Clock | 161.7±11.1s | 155.1s | 174.4s |

## Research Log Updates
- Updated headline from "F1=1.0 (n=1)" to "F1=0.979±0.019 (n=3, 3 tasks)" — honest reporting
- Added Experiments 51 (stability), 52 (cross-task), 53 (figure)
- Updated cumulative results summary with Iteration 20 column
- Paper contribution count: 11 → 13 (+cross-task validation, +multi-run stability)

## Pushbacks

### 1. Headline should remain "F1=1.0"
**I'm changing the headline.** The multi-run stability shows F1=0.979±0.019. Reporting F1=1.0 from a single lucky run would be dishonest. The honest result (F1≥0.968, P=1.0 invariant) is actually a stronger paper — it shows we've characterized the variance and identified its source (retraction stochasticity at chunk boundaries, not architectural failure).

### 2. Cross-model validation deferred
**Maintaining this deferral.** Cross-task validation (3 tasks) is more impactful than cross-model (same task, different model) for proving generalization. Cross-model remains a documented limitation.

## Novel Findings This Iteration

1. **Retraction stochasticity**: The retraction mechanism interacts with LLM parsing stochasticity. When the model "updates" entity attributes at chunk boundaries (even without real changes), retractions fire and some valid pairs are permanently lost. This is a genuine architectural insight: the retraction-for-precision tradeoff has a quantifiable recall cost (σ=0.019 in F1).

2. **Task-dependent savings**: Token savings range from 71% (Task 3) to 91% (Task 6). The variation comes from D's iteration count overhead — tasks where D requires more LLM iterations per turn show higher relative savings for A. This means the structural formula 1-2/(k+1) is indeed a lower bound.

3. **P=1.0 invariance across tasks and runs**: Zero false positives in 5 tasks × up to 3 runs = 7 experiment conditions, 35 turns total. This is the paper's most robust claim.

## Next Experiments
- Cross-model validation (gpt-4o, single run) — the remaining scalability gap
- Temperature=0 ablation — would retraction variance disappear?
- More Task 1 stability runs to narrow σ (e.g., n=5)
- Paper draft with all tables and figures
