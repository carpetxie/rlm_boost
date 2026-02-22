# Critique — Iteration 11

STATUS: CONTINUE

---

## Overall Assessment

Iteration 10 delivered its commitments cleanly: label-aware check_pair is implemented and ran, prune_count telemetry is confirmed working (0→1 transition at Turn 4 matches token evidence), and the AST scope fix for `_extract_assigned_names` is correct. The P=1.0 finding — zero false positives across all turns and conditions under actual task conditions — is the paper's strongest empirical result and is publishable as-is. However, close inspection of `label_aware_task1_results.json` against `split_context_by_users()` reveals two previously unidentified issues that structurally compromise the core A/C comparison: (1) Conditions A and C are evaluated on **different slices of the labeled corpus** — A samples shallowly across all 96K chars while C covers the first 25K contiguously — making their F1 comparison neither fair nor interpretable as "same budget, different strategies"; and (2) in Turn 1 the model advanced `chunks_processed` to 2 (processing a phantom chunk 1 with empty entities alongside chunk 0), causing Turn 2 to be non-compliant by deduplication. Together, these mean the F1=0.1695 vs F1=0.3424 comparison (49.5% of oracle) cannot yet be cited as the paper's headline without a targeted experiment redesign.

---

## Reflection on Prior Feedback

**Resolved — not re-raising:**
- Label-aware check_pair implemented and run. Done.
- prune_count telemetry confirmed via direct attribute access; 0→1 at Turn 4. Done.
- `_extract_assigned_names` uses `tree.body` not `ast.walk()`. Fixed and unit-tested. Done.
- `EntityCache.get_from_chunk()` docstring corrected; `get_new_in_chunk()` added. Done.
- `PairTracker.clear_retracted()` method added. Done.
- `_build_iteration_summary` chunk_index tracking via `_PROCESS_CHUNK_RE`. Done.
- Condition B token anomaly attributed to ORACLE_PROMPT_SINGLE template. Accepted.

**Pushback accepted:** The critique's prediction that "F1 should rise toward 0.716" conflated proxy-condition ceiling (0.716) with actual-task ceiling at 25K labeled chars (0.3424). Researcher is correct. The proxy ceiling is not the actual ceiling. This point is dropped.

---

## Scores

| Criterion | Score | Delta | Comment |
|-----------|-------|-------|---------|
| Novelty | 7/10 | +0 | P=1.0, REPL-state-as-correctness-ground-truth, and retraction taxonomy are genuinely novel. However, novelty claims rest on a comparison that has a coverage non-equivalence flaw. Fix the experiment first, then the novelty claims are independently defensible. |
| Technical Soundness | 6/10 | −1 | New: Conditions A and C see different 25K-char slices of the 96K labeled corpus (A: breadth across full corpus; C: depth in first 25K). This is a structural correctness issue in the experiment design, not a minor bug. The comparison as reported is not internally valid. |
| Benchmark Performance | 6/10 | −1 | F1=0.1695 (A) vs 0.3424 (C) gives "49.5% of oracle," but this comparison is confounded by coverage non-equivalence. Condition A likely achieves substantially higher F1 when redesigned with sequential chunks from the same first 25K chars. Cannot cite 49.5% honestly without the fix. |
| Scalability | 6/10 | +0 | N=231, 3 tasks, 1 model. Turn 1 phantom-chunk behavior (model processes 2 chunks in one turn) is uncharacterized and could worsen at higher k. |
| Research Maturity | 7/10 | +0 | P=1.0 is paper-worthy. The A/C comparison needs one targeted experiment redesign. With that fix, this moves to 8/10. |

---

## Architecture Review

### Critical New Finding 1: Data Slice Non-Equivalence Invalidates the A/C Comparison

Inspecting `split_context_by_users()` (lines 100–124 of `eval/rlm_pipeline_experiment.py`) and the label-aware results together reveals a fundamental mismatch between what Conditions A and C actually process.

The labeled context is **96,689 characters** — roughly 3× the plain context — because each record gains a `|| Label: [cat]` field. When `split_context_by_users(labeled_context, num_chunks=5)` runs:

1. It finds all user-boundary positions across the 96K corpus (231 positions)
2. Divides into 5 groups of ~46 users each, giving ~19K chars per group
3. Each group is truncated to `max_chunk_chars=5000`

**Condition A's five chunks therefore cover**:
- Chunk 0: chars ~0–5K (first 5K of users 1–46)
- Chunk 1: chars ~19K–24K (first 5K of users 47–92)
- Chunk 2: chars ~38K–43K (first 5K of users 93–138)
- Chunk 3: chars ~57K–62K (first 5K of users 139–185)
- Chunk 4: chars ~76K–81K (first 5K of users 186–231)

**Condition C's oracle covers**: `labeled_context[:25000]` — the first 25K chars of the corpus, which spans roughly users 1–60 and overlaps only with Chunk 0 (and slightly into the beginning of Chunk 1's region).

These are **entirely different slices**. Condition A covers all 231 users shallowly (5K chars per user group). Condition C covers ~60 users deeply (all their instances within the first 25K chars). The two conditions have **different coverage ceilings** — not the same 1,653 pairs.

The observed numbers confirm this mechanically:
- Condition C: 113 entities, 58 qualifying, 1,653 pairs (all from chars 0–25K)
- Condition A: 741 pairs from 5 disjoint windows across the 96K corpus

The "49.5% of oracle" headline is therefore a comparison of two different tasks, not two different strategies on the same data. Reviewers who inspect the chunking logic will catch this immediately.

**The fix requires one code change** in `run_label_aware_condition_a()`:

```python
# CURRENT (wrong): split full 96K corpus into 5 user groups, truncate each to 5K
chunks = split_context_by_users(labeled_context, num_chunks)
chunks = [c[:max_chunk_chars] for c in chunks]

# FIXED: split only the first 25K chars into 5 sequential windows
context_window = labeled_context[:num_chunks * max_chunk_chars]   # same 25K as oracle C
step = max_chunk_chars  # 5000
chunks = [context_window[i*step : (i+1)*step] for i in range(num_chunks)]
```

With this fix, A and C see identical content and the comparison measures exactly what it claims: incremental streaming vs single-pass oracle on the same data.

### Critical New Finding 2: Turn 1 Phantom Chunk (Model Processes Two Chunks in One Turn)

From the actual `label_aware_task1_results.json`:

| Turn | Prompt chunk_idx | chunks_processed after | Compliant (current check) |
|------|-----------------|----------------------|--------------------------|
| 1 | 0 | **2** | True (wrong: jumped by 2) |
| 2 | 1 | 2 | False (dedup blocked) |
| 3 | 2 | 3 | True |
| 4 | 3 | 4 | True |
| 5 | 4 | 5 | True |

Turn 1's `pair_checks_total=1,907` at `chunks_processed=2` reveals the model ran `process_chunk(0, entities, ...)` AND `process_chunk(1, {}, ...)` in consecutive REPL iterations within a single completion. The second call created chunk_idx=1 in `_processed_chunk_indices` with empty entities. When Turn 2 arrived with chunk 1's actual data and called `process_chunk(1, entities, ...)`, the deduplication guard returned cached stats (empty) and the incremental state for chunk 1 was permanently polluted with zero entities.

The current compliance metric `chunks_processed > prev_chunks_processed` accepts a jump of 2 as compliant. The correct metric is `chunks_processed == prev_chunks_processed + 1`. Under the correct metric, **Turn 1 is also non-compliant** (advanced by 2), giving a true compliance rate of **60%** (Turns 3, 4, 5 correctly advance by exactly 1), not the reported 80%.

Additionally, chunk 1's real data was **never processed** — the incremental state has 0 entities for chunk 1. This artificially reduces Condition A's coverage and F1. The total pairs found (741) would be higher if chunk 1 had been processed correctly.

This is a prompt engineering issue: `max_iterations=6` lets the model call `process_chunk` with arbitrary chunk indices across REPL iterations. The fix is to restrict the template: add "Call `_incremental.process_chunk({chunk_idx}, entities, pair_checker=check_pair)` EXACTLY ONCE. Do not call process_chunk with any other chunk index in this turn."

### Finding 3: Turn 4 Token Spike (45,275 tokens) — Still Mechanistically Unexplained

prune fired at Turn 4 (prune_count: 0→1), yet input tokens jumped from 4,564 (Turn 3) to **45,275** (Turn 4), then dropped to 6,819 (Turn 5). Pruning should reduce history and thus reduce tokens. The 10× spike is the opposite.

Two hypotheses:
1. **Many LM iterations within Turn 4**: If Turn 4 ran 6 LM iterations (max_iterations=6) while Turn 3 ran ~2, total token usage is 3× higher even if each individual call is smaller. `_extract_tokens()` aggregates across all iterations in a completion.
2. **The prune summary itself is large**: `_build_iteration_summary()` could produce a large summary if many code blocks and outputs are included, partially or fully offsetting the compression.

`run_label_aware_condition_b()` has `iteration_count_proxy = 0` that is never populated — the promised per-completion iteration logging was not implemented. This is a code gap that must be closed to resolve the spike.

---

## Novelty Assessment

### Defend These — They're Independently Robust

**P=1.0 (zero false positives)**: This holds regardless of the A/C coverage non-equivalence. Every predicted pair in Condition A was a true gold pair at every turn. This is a precision guarantee that survives the experiment redesign — the data slice changes which pairs are available but doesn't change whether the label-aware checker is correct. This should be the lead result.

**REPL-state as correctness ground truth**: The deduplication guard in `_processed_chunk_indices` provides O(1) idempotency. Turn 2's non-compliance (dedup blocked the phantom chunk from being re-processed correctly) and the post-pruning continuity at Turn 4–5 both confirm that REPL-level Python state is the ground truth, not message history. This architectural insight generalizes beyond OOLONG-Pairs.

**Retraction taxonomy and temporal asymmetry**: The 44–15,824 range (360×) in retraction counts across task types, mechanistically confirmed for temporal asymmetry (bidirectional vs monotonic invalidation), is empirically original. No prior work characterizes incremental computation overhead as a function of constraint type.

### What the Redesigned Experiment Adds

With sequential chunking, the A/C comparison becomes:
- Same 25K chars, same entities, same qualifying set (58 users), same ceiling (1,653 pairs)
- F1 gap measures only: latency (A sees data incrementally, C sees all at once) + inter-chunk accumulation effects
- If A achieves ≥ 0.28 F1 (≥80% of oracle): strong headline (near-oracle streaming quality)
- If A achieves ~0.22 F1 (same as current k=1): reveals that per-chunk coverage (not total coverage) is the binding constraint, and increasing chunk size is more valuable than increasing k

Either result is a finding. The P=1.0 guarantee is the novelty; the F1 level quantifies the coverage-vs-accuracy tradeoff.

---

## Experiment Critique

### Priority 1: Redesigned A/C Label-Aware (Mandatory, ~$5, 2 hrs)

The single experiment that resolves the paper's central claim. Change `run_label_aware_condition_a()` to use sequential 5K windows from the same first 25K labeled chars that oracle C sees. Simultaneously:
- Fix the compliance metric to `chunks_processed == prev_chunks_processed + 1`
- Add phantom-chunk detection: warn if `chunks_processed > prev_chunks_processed + 1`
- Add per-completion iteration count: log `len(completion.iterations)` or equivalent
- Add the one-sentence prompt fix to prevent phantom chunk calls
- Rerun all three conditions (A, B, C) for Task 1

Expected outcome: Condition A F1 improves substantially. If P=1.0 still holds (expected, since the checker is correct regardless of data slice), the paper has: "Incremental RLM achieves P=1.0 and F1=[X]% of oracle on Task 1, processing the same 25K-char context as oracle in 5 streaming turns of 5K chars each."

### Priority 2: Tasks 3 and 6 Label-Aware with Sequential Chunking (~$8, 3 hrs)

Not yet run. Apply the same sequential-chunk redesign. Task 3 (~70% qualifying entities) should yield higher F1 and may push "X% of oracle" into a more impressive range. If P=1.0 holds for all three tasks, the paper can claim: "The incremental protocol achieves zero false positives across tasks and turns."

### Priority 3: Full-Context Oracle (~$2, 1 API call)

Run Condition C on all 96,689 labeled chars (not just 25K). Expected F1 approaches 1.0 (all 231 users visible, all qualifying pairs findable). This establishes the definitive coverage ceiling and anchors the 25K-window F1 (0.3424) as a coverage fraction: "At 25K chars, the oracle achieves 20.7% of all pairs (1,653/8,001). Our incremental approach achieves [Y]% of the same ceiling with the same budget."

---

## The One Big Thing

**Redesign the Condition A chunking to use sequential 5K windows from the same first 25K labeled chars that Condition C (oracle) sees.** This is a ~10-line code change. Everything else in Iteration 11 is secondary to this. Without it, the paper's core A/C comparison is not internally valid and cannot be published. With it:

1. The coverage ceilings for A and C become identical
2. Condition A's F1 likely improves substantially from 0.1695
3. P=1.0 still holds (the label-aware checker is correct independently of data slice)
4. The "X% of oracle" headline becomes a clean claim about streaming vs single-pass processing

The phantom-chunk fix (one sentence added to the prompt template) and the compliance-metric correction (strict `==` instead of `>`) should be applied simultaneously — both take under 30 minutes and are prerequisite to interpreting any compliance numbers correctly.

---

## Specific Experiments to Run

1. **Redesigned A/C label-aware, sequential chunking, Task 1 (mandatory, ~$5, 2 hrs)**:
   - Replace `split_context_by_users(labeled_context, 5)` + per-chunk truncation with sequential 5K windows from `labeled_context[:25000]`
   - Fix compliance check: `chunks_processed == prev_chunks_processed + 1` (not `>`)
   - Add phantom-chunk warning when delta > 1
   - Add per-completion iteration count logging to diagnose Turn 4 token spike
   - Add to prompt template: "Call `_incremental.process_chunk({chunk_idx}, entities, pair_checker=check_pair)` EXACTLY ONCE with chunk_idx={chunk_idx}. Do not call process_chunk with any other chunk index."
   - Run all three conditions and record F1 alongside prune_count and iteration_count

2. **Tasks 3 and 6 with sequential chunking (~$8, 3 hrs)**:
   - Apply same sequential-chunk design
   - Report P, R, F1 per condition per task
   - If P=1.0 for all tasks: generalization claim is supported

3. **Full-context oracle on all labeled chars (~$2, 30 min)**:
   - Run Condition C with `max_chars=len(labeled_context)` (~96K)
   - Anchors the F1 ceiling definitively
   - One-sentence result: "Oracle on full labeled corpus achieves F1=[Z]≈1.0, confirming the label-aware checker is correct and coverage is the only limiting factor"

4. **Resolve Turn 4 token spike by logging per-completion iteration count (free, 20 min)**:
   - Inspect `rlm._persistent_env.locals` or add `rlm._lm_call_count` instrumentation
   - Confirm whether Turn 4 used 6 LM iterations while other turns used 2
   - If confirmed: "Token cost per turn is iteration-count-dependent; the 45K spike at Turn 4 reflects 6 LM iterations triggered by history pruning generating a complex summary context"

5. **Emit per-chunk coverage count in Condition A output (free, 10 min)**:
   - After each turn, print `qualifying_this_chunk = sum(1 for e in entities.values() if e.get('qualifying'))` and `entities_total_so_far = len(_incremental.entity_cache)`
   - With sequential chunking, this will confirm whether qualifying entities accumulate monotonically across turns (expected: yes, since all 5 chunks now cover the same 25K chars as oracle C)

---

## Code Issues Found

1. **`run_label_aware_condition_a()` compliance check is too permissive** (`eval/label_aware_experiment.py`, line 303):
   ```python
   # CURRENT (wrong — accepts phantom-chunk jump of 2 as compliant):
   compliant = chunks_processed > prev_chunks_processed
   # CORRECT:
   compliant = chunks_processed == prev_chunks_processed + 1
   ```
   The current check accepts any positive increase, including jumps of 2. The strict check enforces exactly one new chunk per turn.

2. **`run_label_aware_condition_b()` declares `iteration_count_proxy = 0` but never populates it** (`eval/label_aware_experiment.py`, line 442):
   The variable is set and never assigned or logged. The Turn 4 token spike investigation requires per-completion iteration counts. This is dead code.

3. **`split_context_by_users()` is not suitable for labeled context without `max_context_chars` parameter** (`eval/rlm_pipeline_experiment.py`, line 100):
   The function splits across the full input, which is fine for ~32K plain context (5K/chunk ≈ 16% coverage per chunk) but produces disjoint 5K windows across 96K labeled context (≈5% coverage per chunk). A `max_context_chars` parameter would make the caller's intent explicit and prevent silent slicing across the wrong range.

4. **`CHUNK_PROMPT_LABEL_AWARE` does not restrict model to exactly one `process_chunk` call** (`eval/label_aware_experiment.py`, line 119):
   The template says "Run this code (chunk index {chunk_idx})" but does not say "EXACTLY ONCE." The model can and does call `process_chunk` with other chunk indices in subsequent REPL iterations, causing the Turn 1 phantom chunk behavior.

5. **History manager lacks per-completion LM call counter** (`rlm/core/history_manager.py`):
   `_prune_count` tracks pruning events; there is no analogous `_total_lm_iterations` counter. The Turn 4 spike requires this to be explained. Add `self._total_lm_calls: int = 0` and increment in `RLM.completion()` for each LM call (or expose the completion's iteration count via `UsageSummary`).

---

## Acknowledged Limitations

- All label-aware experiments use a single OOLONG-Pairs corpus (N=231, 1 domain, 1 model: gpt-4o-mini). Cross-domain generalization is unverified and cannot be fixed without additional resources. Scope as "proof-of-concept on OOLONG-Pairs."
- The σ-parameterized cost model (R²=0.936, p=0.025) is fitted on 35 datapoints from one corpus. Report as empirical approximation; the modest F-statistic and single-corpus origin prevent stronger claims.
- The "dynamic benchmark" is a static dataset chunked artificially. Accepted scoping decision. Frame as "sequential context revelation" with streaming as future work.
- History pruning's correctness guarantee via the deduplication guard is empirically observed but not formally proven. One clear paragraph in the paper suffices; a formal proof is out of scope.
