# Critique — Iteration 13

STATUS: CONTINUE

---

## Overall Assessment

Iteration 12 produced the project's most significant result: with the 2-line monotone attribute fix, A/C ratio jumps from 64.3% → 94.3%, reframing the paper's central narrative from "structural limitation explains 40% gap" to "correct protocol nearly eliminates gap." This is a stronger contribution. However, the 94.3% headline rests on a single 100%-compliant run (Run 2); Run 1 at 60% compliance produced only 69.5% A/C and a **4.84× token overhead** — confirming that REPL template complexity drives stochastic compliance failure and wildly inflates cost. The library-level `monotone_attrs` fix is not optional engineering polish — it is the prerequisite for every remaining experiment in this paper. Without it, the k-sensitivity sweep, Tasks 3/6 V3, and any multi-seed stability claim are unreliable.

---

## Reflection on Prior Feedback

**Resolved — not re-raising:**
- Attribute-overwriting bug (Critique 12). Fixed in REPL template (V3). Ablation complete (Experiment 31). Done.
- Double-counting in `updated × all` sweep. Fixed via `checked_in_updated_sweep` deduplication. Done.
- Missing `reset()` method. Implemented. Done.
- Condition B using wrong system prompt. Fixed in `run_condition_b_v3()`. Done.
- Coverage ceiling formula asymmetric-task guard. Comment added. Done.
- Phantom chunk / compliance metric. Fixed in V2. Note: V3 Run 1 still produced a phantom chunk at Turn 2 — this is the compliance fragility being raised below, not the metric bug.

**Pushbacks accepted — not re-raising:**
- Lazy evaluation as next architectural step. Correctly deferred post-A2. With A/C = 94.3%, lazy eval addresses the wrong problem. Accepted.
- σ-parameterized cost model modest R² improvement. Accepted as publishable with appropriate caveats.
- "Structural qualification-time asymmetry" as dominant explanation for 40% gap. Researcher correctly updated this narrative after Experiment 31. Accepted.

---

## Scores

| Criterion | Score | Delta | Comment |
|-----------|-------|-------|---------|
| Novelty | 7/10 | 0 | P=1.0 protocol, retraction taxonomy, at-risk fraction tool are independently publishable. The 94.3% A/C result is stronger than before — but it's driven by a 2-line bug fix. The paper must carefully separate architectural contribution (IncrementalState framework) from protocol contribution (monotone attribute semantics) or a reviewer will frame this as "they fixed their own bug." |
| Technical Soundness | 7/10 | 0 | V3 Run 2 is internally valid. However: 94.3% from one run; k-sensitivity data missing; Tasks 3/6 V3 not run; token overhead ranging 1.14×–4.84× across conditions — these prevent a higher score. |
| Benchmark Performance | 8/10 | +1 | 94.3% A/C with P=1.0 on identical 25K-char budget is genuinely strong if confirmed reproducible. Score is conditional on library-level fix and multi-run stability confirmation. |
| Scalability | 5/10 | -1 | k-sensitivity is STILL missing after 13 iterations. This is the paper's core scalability claim and remains completely unquantified. The infrastructure (`run_k_sensitivity_sweep()`) exists but has never been called. Downgrade until the data exists. |
| Research Maturity | 6/10 | -1 | The 94.3% headline comes from a single lucky run (Run 2). Run 1 gives 69.5%. A paper submitting with two runs diverging by 25pp on the primary metric will be rejected for cherry-picking. Library fix + multi-run confirmation required before any submission-quality claim. |

---

## Architecture Review

### Critical Blocker: `monotone_attrs` Not Yet in `process_chunk()` Library

Examining `rlm/core/incremental.py` confirms: `process_chunk()` still has signature `(self, chunk_index, new_entities, pair_checker=None)` — no `monotone_attrs` parameter. The monotone fix lives entirely in the REPL template (`CHUNK_PROMPT_LABEL_AWARE_V3`), which is why compliance is stochastic.

The failure mode is now precisely characterized by the raw Run 1 data:
- Turn 2: delta=2, phantom_chunk=True, iteration_count=7 → model processed 2 chunks in one turn because the 6-line monotone loop confused chunk_idx tracking
- Turn 3: delta=0 → stagnation because chunk_idx was consumed by the phantom
- Turn 1, 2, 4 each used 7 REPL iterations (maximum) → the complex template is driving max-iteration behavior
- Total: 116,120 input tokens = **4.84× oracle** (vs Run 2's 2.42× and V2's 1.14×)

The library-level fix eliminates this by reducing the REPL template from ~25 lines back to V2 levels (~15 lines) — the model has less complex code to correctly emit. The concrete API addition:

```python
def process_chunk(
    self,
    chunk_index: int,
    new_entities: dict[str, dict[str, Any]],
    pair_checker: Any = None,
    monotone_attrs: set[str] | None = None,  # NEW
) -> dict[str, Any]:
```

When `monotone_attrs` is provided, the entity `add()` loop should merge: for any `attr` in `monotone_attrs`, if `cached_val` is truthy and `new_val` is falsy, keep `cached_val`. Additionally — and this is the key optimization — if ALL `monotone_attrs` values are unchanged after merging (the entity's effective state is identical), skip adding it to `updated_ids`. This prevents retraction + `updated × all` sweep for entities whose cached classification is identical, eliminating the 1,078 no-op retractions from V3 Run 2 at zero correctness cost.

**This single library change unlocks everything else in Iteration 13:** k-sensitivity sweep, Tasks 3/6 V3, multi-run stability — all become reliable.

### V3 Token Overhead Is Understated in the Research Log

The research log documents V3's token ratio as "2.42× vs oracle." This is the Run 2 number only. Run 1 shows **4.84×** with 60% compliance. The paper cannot present 2.42× as the characteristic overhead of V3 without acknowledging the full distribution.

The decomposition matters for the paper's cost framing:
- V2 (no monotone fix, 100% compliant): 27,504 tokens = 1.14× oracle
- V3 Run 2 (template fix, 100% compliant): 60,005 tokens = 2.42× oracle
- V3 Run 1 (template fix, 60% compliant): 116,120 tokens = 4.84× oracle

The gap between V2 (1.14×) and V3 Run 2 (2.42×) — even under perfect compliance — is a 2.18× increase driven by the longer REPL template producing more LM output tokens and more within-turn REPL iterations. With library-level `monotone_attrs`, the template shrinks back to V2 complexity, and the token overhead should return near 1.14×. This needs empirical confirmation after the library fix — it's a key efficiency claim.

### No-Op Retraction Count Is Mischaracterized

V3 Run 2 shows 1,078 "no-op retractions." The current docstring on `_retracted` conflates permanently-invalidated pairs with temporarily-retracted-then-re-added pairs. After V3's monotone fix, all 1,078 retraction cycles result in immediate re-addition (qualifying=True preserved → pair re-added). So `_retracted` is actually empty after each cycle. But `_retraction_count` still increments by 1,078. The paper should report "1,078 retraction cycles (all re-added)" vs "0 permanently-invalidated pairs" — not just "1,078 retractions" which sounds like 1,078 pairs lost.

---

## Novelty Assessment

### The "Bug Fix" Framing Risk

The headline result of Iteration 12 is that a 2-line bug fix raises A/C from 64.3% to 94.3%. A reviewer will ask: **"Is this a novel contribution or a debugging exercise?"**

The paper must frame this as a **correctness condition discovery**, not a bug fix:

**"Incremental computation over attributed entities requires monotone attribute accumulation as a semantic correctness condition for 'at least one' predicate types. We identify this condition (the monotone accumulation requirement), characterize its violation (the attribute-overwriting failure mode), provide an analytical tool for predicting its impact (at-risk fraction), and show that enforcing correct monotone semantics via library support nearly closes the gap between streaming and batch oracle."**

Under this framing, the contribution is:
1. **Correctness condition**: Monotone attribute accumulation is necessary for streaming correctness of "existential" predicates. Prior streaming systems (e.g., incremental join processing) don't consider LLM-generated attribute streams where labels appear non-uniformly across chunks.
2. **Analytical tool**: At-risk fraction (proportion of qualifying entities that reappear with downgraded labels) predicts impact before running experiments.
3. **Architectural mechanism**: Library-level `monotone_attrs` parameter enforces the condition without burdening prompt templates, maintaining high compliance.

### The At-Risk Fraction Is the Most Underemphasized Contribution

The Gini analysis and at-risk fraction tool (23.2%, 26.6%, 31.7% for Tasks 1, 3, 6) are the most novel analytical output. This tool:
- Predicts fix impact across tasks before running expensive experiments
- Provides a corpus-level deployment diagnostic
- Generates a falsifiable cross-task prediction

**This prediction is currently unverified.** Task 6 V3 has NOT been run. If Task 6 V3 achieves A/C > Task 1 V3's 94.3%, the tool is validated. If not, the tool's predictive power requires recalibration. **Either outcome strengthens the paper** — but the experiment must be run.

### What Would Make This More Novel (Minimum Required)

Two additions to increase cross-task generalization confidence:
1. **Non-monotone task test**: Run Task 11 or 13 (asymmetric "exactly N") with V3 framework but `monotone_attrs=None` (correct, since "exactly N" is non-monotone). A/C should NOT improve from V2 baseline. This validates that `monotone_attrs=None` correctly handles non-monotone conditions and that the V3 improvement is specifically due to monotone semantics, not some other change.
2. **Cross-task ordering validation**: Tasks 3 and 6 V3 (planned but unrun). If Task 6 A/C improvement > Task 3 > Task 1 (matching at-risk fraction ordering), the at-risk fraction tool is validated as a predictor.

---

## Experiment Critique

### k-Sensitivity — 13 Iterations Without the Paper's Core Scalability Figure

The `run_k_sensitivity_sweep()` function was implemented in `eval/label_aware_v3_experiment.py` at Iteration 12. It exists. It has never been called. There is no `k_sensitivity_v3.json` in `results/streaming/`. This is **the paper's primary scalability claim**, and it is completely unmeasured.

The sweep needs to answer:
1. Does A/C ratio stay ~94% at k=3 (8.3K chars/chunk)? At k=7 (3.6K chars/chunk)? At k=10 (2.5K chars/chunk)?
2. How does total token cost A/C scale with k?
3. What is the iso-cost k where tokens(A) ≈ tokens(C)?

**Testable prediction from first principles:** At smaller k (coarser chunks = more chars/turn), more qualifying entities appear per chunk, reducing the structural asymmetry from entities that qualify after their potential partners were processed. Expect A/C to be **higher at k=3** than at k=5. At larger k (finer chunks), each chunk contains fewer qualifying entities and more early-qualification misses. Expect A/C to **decrease at k=10**. If this monotone relationship holds, the paper can offer a principled recommendation: "use k≤5 for maximum oracle fidelity; k>5 for lower per-turn context cost." If the relationship is non-monotone, that's an unexpected finding worth reporting.

Run this experiment immediately after the library-level monotone fix. Cost: ~$15-20.

### Tasks 3 and 6 V3 — Falsifiable Prediction Unverified

The at-risk fraction predicts: Task 6 (31.7% at-risk) benefits more from the V3 fix than Task 1 (23.2%). Measured improvement for Task 1: +30.0pp (64.3% → 94.3%).

If the relationship is linear (unrealistic but as a bound): Task 6's 31.7% at-risk → expected ~+41pp improvement → starting at 55.5%, ending at ~96.5%. Actual result will likely be different due to nonlinearity, but the ordering (Task 6 ΔA/C > Task 1 ΔA/C) should hold if at-risk fraction is the right predictor.

This experiment: $8-12, ~2 hours. The falsifiable prediction is the at-risk fraction tool's first real test. Run it immediately after the library fix.

### Multi-Run Stability — Required Before Submission

V3 Run 1: 60% compliance, A/C=69.5%, 4.84× tokens. V3 Run 2: 100% compliance, A/C=94.3%, 2.42× tokens. The 25pp A/C spread from two runs of identical configuration is the most vulnerable fact in the paper.

After library-level fix, run Task 1 V3 three more times. Report:
- Distribution of compliance rates (expect: 100% every run with simplified template)
- Distribution of A/C ratios (expect: ≈94% every run when compliant)
- Distribution of token ratios (expect: near V2's 1.14× with simplified template)

If the standard deviation on A/C ratio drops to <3pp across 5 runs, the 94.3% headline is defensible. If it stays >10pp, the result depends on stochastic compliance and requires a different framing.

### Condition B V3 — Implemented, Not Yet Run

`run_condition_b_v3()` uses `RLM_SYSTEM_PROMPT` (correct). No result file exists. Cost: $2. Run it. The V2 B result (F1=0.0193) may have been depressed by the wrong system prompt. Quantifying the distortion closes a known reviewer vulnerability.

---

## The One Big Thing

**Implement `monotone_attrs` in `process_chunk()` at the library level, simplify the REPL template, then run Task 1 V3 three more times to confirm stable 94.3% A/C.**

The 94.3% headline is the paper's primary result. It rests on one run. Every remaining experiment (k-sensitivity, Tasks 3/6 V3, non-monotone sanity check, Condition B V3) depends on reliable compliance. The library fix reduces REPL template complexity, eliminates stochastic compliance failure, and brings token overhead back near V2's 1.14×. This is 2-3 hours of implementation + 1 hour of testing. Everything else can follow.

---

## Specific Experiments to Run

In execution order (each step enables the next):

1. **Implement `monotone_attrs` in `process_chunk()` ($0, ~2 hrs, code only)**:
   - Add `monotone_attrs: set[str] | None = None` parameter to `IncrementalState.process_chunk()`
   - In entity add loop (step 1): when entity is an update and `monotone_attrs` specified, merge attrs — for each `attr` in `monotone_attrs`, if `old_cached_val` is truthy and `new_val` is falsy, set `new_val = old_cached_val`
   - Optimization: if all `monotone_attrs` values are unchanged after merge, do NOT add to `updated_ids` — skips retraction + `updated × all` sweep for that entity (eliminates no-op retractions)
   - Update `CHUNK_PROMPT_LABEL_AWARE_V3` to call `process_chunk(chunk_idx, entities, check_pair, monotone_attrs={"qualifying"})` — remove the 6-line propagation loop
   - Add 4 unit tests: (a) monotone attr preserved on downgrade, (b) retraction skipped when only monotone attrs change, (c) retraction fires when non-monotone attrs change, (d) `None` preserves existing behavior exactly
   - Expected outcome: REPL template drops to V2 complexity → 100% compliance deterministically

2. **Task 1 V3 multi-run stability ($12-15, ~3 hrs)**:
   - Run Task 1 with library-level monotone fix 3 times (total 5 including Runs 1-2)
   - Report: compliance rate, A/C ratio, token ratio, F1 — per run and as mean ± std
   - If mean A/C > 90%, std < 5pp, and all compliance rates = 100%: result is publishable
   - Expected: token ratio returns near 1.14× (matching V2) due to simplified template

3. **Tasks 3 and 6 V3 ($8-12, ~2 hrs)**:
   - Run with library-level monotone fix
   - Verify prediction ordering: ΔA/C(Task 6) > ΔA/C(Task 3) > ΔA/C(Task 1)
   - If ordering holds: at-risk fraction is validated as a predictor (key paper result)
   - If ordering fails: characterize which at-risk fraction measure is wrong and why (still a finding)

4. **k-sensitivity sweep, Task 1, k ∈ {3, 7, 10} ($15-20, ~3 hrs)**:
   - Use library-level V3 with simplified template
   - Per k: F1(A), F1(C), A/C ratio, tokens(A)/tokens(C), compliance rate
   - This is Figure 1 of the paper. Cannot omit.
   - Also compute iso-cost k: smallest k where tokens(A) ≤ 1.5×tokens(C)

5. **Condition B V3 with corrected system prompt ($2, 30 min)**:
   - `run_condition_b_v3()` is implemented — just run it
   - Report F1(B V3) vs F1(B V2) = 0.0193
   - Closes known paper vulnerability in A/B comparison

6. **Task 11 non-monotone sanity check ($3, 30 min)**:
   - Run Task 11 ("exactly N") with V3 framework, `monotone_attrs=None`
   - Verify A/C ≈ V2 Task 11 baseline (fix should have no effect on non-monotone conditions)
   - Confirms `monotone_attrs=None` path is correct; validates monotone fix is targeted

---

## Code Issues Found

1. **`process_chunk()` lacks `monotone_attrs` parameter (CRITICAL — `rlm/core/incremental.py` line 242)**:
   Signature is `(self, chunk_index, new_entities, pair_checker=None)`. The library-level fix is the highest-priority code change in this iteration. Without it, every V3 experiment has stochastic compliance because the REPL template carries the semantic burden.

2. **No-op retractions with V3 will remain unless library fix skips retraction for monotone-only changes (`rlm/core/incremental.py` step 2, retraction loop)**:
   Even after library fix, unless the optimization (skip `updated_ids` for monotone-only changes) is included, 1,078 retraction cycles will still fire per 5-chunk run. The optimization is: after computing merged attrs, if `all(merged[a] == old[a] for a in monotone_attrs)`, do NOT add to `updated_ids`. This is O(|monotone_attrs|) per entity and eliminates the O(degree × n) retraction cost.

3. **V3 Run 1 token data (4.84×) not documented in research log**:
   The log states V3 token ratio as 2.42× (Run 2 only). Run 1 at 4.84× is only visible in the raw JSON file. The research log should document the full distribution: "V3 Run 1: 4.84× (60% compliance), V3 Run 2: 2.42× (100% compliance)." A reviewer who reads only the log and then checks the raw JSON will flag this as selective reporting.

4. **`k_sensitivity_v3.json` does not exist despite `run_k_sensitivity_sweep()` being implemented (`eval/label_aware_v3_experiment.py` line 632)**:
   The sweep is ready to call. It's gated on `--run-sweep` flag in the main block. This has simply never been executed. Run it.

5. **`PairTracker._retraction_count` conflates retraction-and-re-added with permanently-invalidated in paper narrative**:
   `_retraction_count` increments for every `retract_entity()` call regardless of whether pairs are subsequently re-added. V3 Run 2's 1,078 reported "retractions" are all retract-and-re-add cycles, not permanent invalidations. The paper should distinguish these: "1,078 no-op retraction cycles" vs "X permanently invalidated pairs." Consider adding `_permanent_retraction_count` or documenting this distinction clearly in the stats output.

---

## Acknowledged Limitations

- All live experiments use gpt-4o-mini (one model) on OOLONG-Pairs (one corpus, N=231). Cross-model and cross-corpus generalization are out of scope for this proof-of-concept. Frame as "single-model study; generalization requires future work."
- V3 94.3% A/C is currently one-run evidence. Multi-run stability analysis resolves this within project scope (Experiment 2 above).
- Token overhead for V3 with REPL template fix (2.42×) differs from V3 with library fix (expected ~1.14×). Cannot make the library-fix efficiency claim without running it empirically.
- The "streaming" benchmark uses static data chunked sequentially — accepted scoping. The benchmark characterizes sequential context ingestion under fixed token-per-turn constraints, not live data streams.
- No lazy retraction implementation: correctly deferred. The 5.7% structural residual gap after V3 fix is too small to justify the architectural complexity. Paper should characterize this residual as "entities that gain qualifying status only in their final chunk appearance" — correct structural description for the remaining gap.
