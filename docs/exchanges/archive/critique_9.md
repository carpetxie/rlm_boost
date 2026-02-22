# Critique — Iteration 9

STATUS: CONTINUE

---

## Overall Assessment

Iteration 8 delivered the paper's central figure: a strictly monotone F1 progression [0.22, 0.35, 0.45, 0.49, 0.51] with 100% compliance across 5 turns and empirically confirmed 22.3% pair-check savings. This is genuine progress. However, three paper-blocking issues remain: (1) the comparison table is broken — Conditions B and C both record F1=0.0 due to a Failure Mode B recurrence in the baseline code path, making the "A vs B vs C" comparison structurally invalid; (2) all per-turn token counts in `f1_progression_results.json` are zero, meaning no cost data exists for the central experiment; and (3) the precision decline from 0.88→0.54 over 5 chunks (3,312 FPs at k=5) is unexplained and represents a hypothesis about check_pair/gold mismatch that has not been tested. These must be resolved before any version of this paper can be submitted. The generalization to Tasks 3 and 6 — promised since Iteration 6 — also remains absent.

---

## Reflection on Prior Feedback

**Resolved — not re-raising:**
- Deduplication guard for `process_chunk`: implemented and tested (3 new tests). Failure Mode C eliminated.
- `_build_iteration_summary` regex: fixed to match both ` ```python ``` ` and ` ```repl ``` ` blocks.
- Stale `_find_system_end()` docstring cross-reference: removed.
- `run_v3_experiment.py` fragile `.env` loading: fixed to line-by-line parser.
- Weighted savings headline ("~39% token savings"): removed and corrected to pair-check savings.
- Tasks 11/13 bidirectional retraction claim (~0%): empirically refuted (6.1%/9.5%) and corrected in `lazy_retraction_analysis.md`.
- Role-ordering in history pruning: fixed (summary=assistant, ack=user).

**Escalated from Iteration 8 — now paper-blocking:**
- Condition B/C baseline: Iteration 8 critique said "run Conditions B and C." Researcher ran them. Both returned F1=0.0. This is not a resolution; it is a confirmed failure that must now be fixed.

---

## Scores

| Criterion | Score | Delta | Comment |
|-----------|-------|-------|---------|
| Novelty | 7/10 | +0 | F1 progression curve is the paper's central figure but the comparison story (incremental vs. baseline) is broken. Non-monotonic retraction taxonomy (360× range) + failure mode taxonomy are genuinely novel. Generalization to multiple tasks still missing. |
| Technical Soundness | 6/10 | −1 | Condition B/C produce F1=0.0 from a parse failure — the comparison table in the paper is structurally invalid. Token tracking shows all zeros in the central experiment. Precision decline mechanism at k=5 unexplained. |
| Benchmark Performance | 7/10 | +1 | F1 progression [0.22→0.51] is monotone and 22.3% pair-check savings are confirmed. BUT: without a working Condition C oracle, "F1=0.51 is X% of oracle" cannot be stated. |
| Scalability | 6/10 | +0 | History pruning effect at turn 4+ unverified. Precision declines at scale (k=5). No data beyond N=231 or 5 chunks. |
| Research Maturity | 6/10 | −1 | One step backward: the comparison experiment was run but produced invalid results. The paper cannot be submitted with B=0.0 and C=0.0 in the comparison table. |

---

## Architecture Review

### Critical Code Bug: Token Tracking Silent Failure

**In `eval/f1_progression_experiment.py`, `run_condition_a_incremental()`, lines 167–175:**

```python
usage = completion.usage_summary
usage_dict = usage.to_dict() if hasattr(usage, "to_dict") else {}
for _model_name, mu in usage_dict.items():
    if isinstance(mu, dict):
        input_tokens += mu.get("input_tokens", 0)
```

**All five turns in `results/streaming/f1_progression_results.json` record `input_tokens: 0, output_tokens: 0`.** This is a silent failure — the code runs without error but produces no data.

The likely cause: `usage.to_dict()` returns `{model_name: UsageSummary_object}` where values are objects, not plain dicts. The `isinstance(mu, dict)` check is False for every entry so `input_tokens` is never incremented. The token tracking in `eval/live_api_experiment.py` used a different pattern and correctly measured 1415→3539→5625 tokens across 3 turns.

**Impact**: The paper has zero cost data from its central experiment. The claim "22.3% pair-check savings" is correct (pair_checks_total is accurate) but cannot be expressed in token or dollar terms. **Concrete fix**: Print `type(completion.usage_summary)` and `dir(completion.usage_summary)` on one debug run, then fix the extraction to use attribute access (`.input_tokens`) instead of dict lookup.

### Condition B/C: Confirmed Failure Mode B Recurrence

From `results/streaming/f1_progression_results.json` (confirmed by direct inspection):

```
condition_b: { f1: 0.0, predicted_pairs: 0,
  response_preview: "FINAL(\"There are 45 users with at least 1 instance,
  resulting in a total of 990 unique pairs of users.\")" }

condition_c: { f1: 0.0, predicted_pairs: 0,
  response_preview: "...128 unique users...8128 pairs...
  FINAL(f\"Total unique users: 128; Total unique user pairs: 8128\")" }
```

Both models returned `FINAL()` with natural language instead of `FINAL_VAR(pair_results)`. The extraction code `eval(response_str.strip())` requires a Python list literal starting with `[`. Both responses → `pairs = []` → F1=0.0.

This is Failure Mode B (FINAL_VAR premature/skipped) from the failure mode taxonomy, now confirmed in the baseline conditions. The explicit code template in Condition A avoids this by forcing the model to execute a specific code block and then call FINAL_VAR. The BASELINE_PROMPT_SINGLE in Condition B provides no such scaffolding. This demonstrates that **the code template is a necessary condition for structured extraction**, not merely a convenience — this is an underemphasized finding.

**How to fix Condition B**: Use `persistent=True` with `num_chunks=1` and the exact same `CHUNK_PROMPT_INCREMENTAL` template. After `completion()`, extract pairs directly from `env.locals["_incremental"].pair_tracker.get_pairs()` — NOT from `completion.response`. This is the same extraction path as Condition A.

**How to fix Condition C**: Use `persistent=True` with a single turn, all 25K chars as one context. Embed explicit extraction code in the prompt (analogous to Condition A's template, but extracting all pairs at once without `_incremental`). Extract from `env.locals["pair_results"]` directly. Note: gpt-4o-mini at 25K chars will differ from the 77.8% F1 in Experiment 1 (which used gpt-5/gpt-5-mini) — this is intentional; same-model comparison is necessary.

### Precision Decline: Unexplained and Growing (0.88→0.54)

Detailed FP trajectory by chunk:

| k | Pairs | TP | FP | FP-rate |
|---|-------|----|----|---------|
| 1 | 1128 | 990 | 138 | 12.2% |
| 2 | 2701 | 1891 | 810 | 30.0% |
| 3 | 4095 | 2701 | 1394 | 34.0% |
| 4 | 5565 | 3321 | 2244 | 40.3% |
| 5 | 7140 | 3828 | 3312 | 46.4% |

FP rate nearly quadruples from k=1 to k=5. The gold condition (Task 1) is "both users have >= 1 instance." The injected `check_pair` is `len(instances1) >= 1 and len(instances2) >= 1`. These should be identical. The discrepancy must come from a grounding difference:

**Hypothesis**: The gold pairs use `labeled_context` where a user's "instance" is a pre-labeled appearance. The model extracts "instances" from `plain_context` (raw text) where any mention of a user_id on a line counts as an instance. A user_id string may appear in plain_context in a non-instance context (e.g., in header/metadata lines), creating entities that "have instances" in the model's parse but not in the gold. As k increases, more such phantom entities accumulate, and every phantom entity forms O(n) false-positive pairs.

**This is measurable without API calls in 30 minutes**: For each chunk k, compute the set intersection of model-predicted entity IDs with labeled_context entity IDs, and count how many predicted pairs have both IDs in the intersection (true candidates) vs. at least one ID outside it (phantom). If phantoms explain the FP growth, the fix is to add a user_id validation step to `parse_entities`.

**Paper impact**: If this is the mechanism, the F1=0.51 plateau is not an incremental protocol limitation — it is an entity extraction noise problem. The paper should separate these concerns: "incremental F1 on clean entities: [higher number]; F1 with raw plain-text extraction: 0.51." Alternatively, fix the extraction and re-run.

### History Pruning Effect at k≥4 Unverified

With `max_recent_iterations=3` and `max_iterations=6` per turn, by Turn 4 the message history could have 18+ messages (6 iterations × 2–3 messages each per turn × 3 turns). Pruning would activate, compressing earlier chunks into a summary. The f1_progression data shows Turn 5 is compliant (chunks_processed: 5), but gives no visibility into whether pruning fired and whether any F1 degradation occurred from history compression.

**Risk**: If pruning compresses the Turn 1 processing into a text summary and the model in Turn 5 cannot reconstruct which entity IDs were seen in chunk 0 (because the deduplication guard only prevents re-processing of the same chunk_index, not re-parsing the wrong entities), the deduplication guard and pruning could interact in unexpected ways.

**Fix**: Add `self._prune_count: int = 0` to `HistoryManager.__init__()`, increment in `_prune_with_summary()` when `len(iteration_messages) > max_iter_messages`. Expose in `get_stats()`. This is 5 lines and requires no API calls.

---

## Novelty Assessment

### What is Genuinely Novel (Confirmed by Iteration 8)

1. **Non-monotonic retraction taxonomy**: 360× range in retraction counts (44 to 15,824 at k=5) explained by condition semantics and temporal direction. "Exactly N" conditions empirically classified as UNSAFE (refuting prior theory at 6.1%/9.5% bidirectional rates). This taxonomy has no direct precedent in incremental computation literature.

2. **Failure Mode Taxonomy for LLM protocol execution**: Three modes (A: entity ID mismatch, B: FINAL_VAR skip, C: redundant process_chunk) characterized with root causes and fixes. Mode C fixed by deduplication guard. The v3 code template as a necessary protocol constraint is an underemphasized but genuine finding.

3. **Strictly monotone F1 progression with 100% compliance**: [0.22→0.35→0.45→0.49→0.51] across 5 turns with zero violations and 22.3% pair-check savings confirmed. First empirical demonstration of incremental context accumulation yielding monotone F1 improvement in an LLM-driven pipeline.

4. **σ-parameterized cost model (R²=0.936)** validated across 11 task types, 4 chunk counts, 5 seeds. Actionable for practitioners. Break-even at k≥4 confirmed for all task types.

### Critical Remaining Gap: The Comparative Claim Is Still Missing

After 9 iterations, there is no valid comparison between the incremental RLM and a non-incremental baseline. The paper can show monotone F1 growth. It cannot yet claim "incremental achieves F1=X vs. non-incremental baseline F1=Y on the same budget." Both Condition B and Condition C produced F1=0.0. This is the single most important gap in the research.

### Underemphasized Finding: Code Template as a Necessary Protocol Constraint

The contrast between Condition A (100% compliance, F1=0.51) and Conditions B/C (0% structured output, F1=0.0) reveals something important: gpt-4o-mini cannot produce structured pair extraction output without explicit code scaffolding. When given `BASELINE_PROMPT_SINGLE` ("Return FINAL_VAR(pair_results)"), the model defaults to natural language. When given `CHUNK_PROMPT_INCREMENTAL` (explicit code block with assignments), it follows the protocol perfectly.

This is a finding about LLM protocol compliance, not just a prompt engineering quirk. It should be stated in the paper: "We find that explicit executable code templates are necessary — not merely helpful — for structured output extraction in RLM pipelines. Instruction-level prompts alone (without explicit code blocks) cause models to revert to natural language responses."

---

## Experiment Critique

### The Comparison Table Cannot Be Published in Its Current Form

The `f1_progression_results.json` comparison table would read:
- Condition A (Incremental, k=5): F1=0.51
- Condition B (Non-incremental, 5K): F1=0.0
- Condition C (Non-incremental, 25K oracle): F1=0.0

Publishing B=0.0 and C=0.0 as baselines would be actively misleading — a reviewer would interpret these as evidence that non-incremental approaches fail completely, when in fact both are implementation failures of the extraction code. This must be fixed.

**The right comparison once fixed (expected)**:
- Condition A (Incremental, k=5, 25K total): F1≈0.51
- Condition B (Non-incremental, 5K only): F1≈0.22 (same coverage as k=1; trivially same result since coverage determines recall)
- Condition C (Non-incremental, 25K oracle, same model): F1≈0.55–0.75 (unknown; likely higher than A)

If Condition C > Condition A, that's a limitation: incremental accumulation has overhead (false positives, FP rate 46%) that a single-turn oracle avoids by seeing the full clean context. Honest reporting of this limitation is fine and actually publishable.

### F1 Saturation at k=4→5 (+0.02 Gain) Needs Interpretation

The marginal F1 gain collapses: +0.13, +0.10, +0.04, +0.02. At k=5, TP=3828 but FN=4173 — more than half of gold pairs are still unfound despite having processed 25K chars. What explains the remaining 4173 FN at k=5?

**Experiment (no API, 5 min)**: Count gold pairs (from labeled_context) where both user IDs appear somewhere in the first 25K chars of plain_context. If this count is significantly less than 8001, the FN at k=5 are coverage-bounded (those users simply don't appear in the first 25K chars). This would explain the saturation and set an appropriate ceiling for the paper's claims.

### Tasks 3 and 6 F1 Progression: 9 Iterations Overdue

This is the third consecutive iteration where Tasks 3 and 6 appear in Next Steps but have not been run. The paper's F1 progression claim currently rests on a single task (Task 1, σ=0.30). Reviewers will ask: does this generalize? The cost model predicts similar savings for Tasks 3 and 6 (σ=0.39 and 0.34), but without running the full F1 progression experiment, there is no empirical answer.

If these experiments show the same monotone F1 pattern, the paper can claim "monotone F1 across all symmetric tasks." If they don't, that's a finding that needs explaining. Either way, this experiment is mandatory before submission.

---

## The One Big Thing

**Run corrected Conditions B and C using direct REPL environment extraction (not `completion.response` parsing), and fix the token tracking bug before the run.**

The sequence:
1. **Fix token tracking** (no API, 30 min): Add debug print to identify `usage_summary` object structure; fix extraction to use attribute access.
2. **Implement `run_condition_b_template`** (no API, 30 min): Use `persistent=True`, `num_chunks=1`, same `CHUNK_PROMPT_INCREMENTAL`, extract from `env.locals["_incremental"].pair_tracker.get_pairs()`.
3. **Implement `run_condition_c_oracle`** (no API, 30 min): Use `persistent=True`, single turn, 25K chars, explicit code template building `pair_results` without `_incremental`. Extract from `env.locals["pair_results"]`.
4. **Run both conditions** (~$5, 1 hr).
5. **Report the comparison table** with honest F1 values.

This single experiment resolves the paper's most critical gap. Expected cost: $3–8. Expected time: 2–3 hours including implementation.

---

## Specific Experiments to Run (Iteration 9)

1. **Fix and re-run Conditions B/C (mandatory, ~$5, 2–3 hrs total)**:
   - Fix token extraction: debug `type(completion.usage_summary)` and correct the isinstance check
   - Implement template-based extraction for both conditions (using `env.locals` not `completion.response`)
   - Run and report the three-condition comparison table

2. **Precision decline root cause analysis (no API, 30 min)**:
   - For each chunk k=1..5, split predicted pairs into: (a) both user IDs in labeled_context, (b) at least one user ID NOT in labeled_context
   - Count FP from each category
   - If category (b) dominates: entity extraction from plain text is over-inclusive → propose validation step

3. **Coverage ceiling for k=5 (no API, 5 min)**:
   - Count gold pairs where both user IDs appear in the first 25K chars of plain_context
   - This gives the hard upper bound on F1 achievable by any system using 25K chars of plain context

4. **F1 progression for Tasks 3 and 6 (~$5, 1–2 hrs)**:
   - Add `--task-idx` argument to `f1_progression_experiment.py`
   - Run with task_idx=3, task_idx=6 (incremental-only to save cost)
   - Report whether F1 is strictly monotone; compare final F1 to Task 1 at same k

5. **Add HistoryManager prune_count telemetry (no API, 15 min)**:
   - `self._prune_count: int = 0` in `__init__`
   - `self._prune_count += 1` in `_prune_with_summary()` when pruning fires
   - Expose in `get_stats()`
   - Verify in the next 5-chunk run that pruning activates at expected turn

---

## Code Issues Found

1. **Token tracking silent failure** (`eval/f1_progression_experiment.py`, lines 167–175): `isinstance(mu, dict)` is False when `usage.to_dict()` returns object values, not dicts. All token counts in `f1_progression_results.json` are zero. No cost data available from the central experiment.

2. **Condition B/C extraction uses `completion.response` parsing** (`eval/f1_progression_experiment.py`, lines 304–318): `eval(response_str.strip().startswith("["))` fails for natural language responses. Both baselines record F1=0.0. Must switch to `env.locals` extraction (same as Condition A's `env.locals["_incremental"].pair_tracker.get_pairs()`).

3. **`persistent=False` in `run_condition_b_baseline`** (`eval/f1_progression_experiment.py`, line 287): With `persistent=False`, the REPL environment is discarded after `completion()`, making `env.locals` inaccessible. Must use `persistent=True` to allow direct pair extraction from the live environment.

4. **No pruning telemetry in HistoryManager** (`rlm/core/history_manager.py`): `_prune_with_summary()` fires silently. Adding `self._prune_count += 1` and exposing in `get_stats()` takes 5 lines and makes the pruning behavior observable.

5. **`check_pair` / gold grounding mismatch unquantified**: The FP rate grows from 12% to 46% across k=1..5. The hypothesis is that plain-text entity extraction over-includes user IDs that lack labeled instances. This should be quantified using existing `_parse_labeled_context` infrastructure before the paper's precision numbers are finalized.

---

## Acknowledged Limitations

- All results are on a single OOLONG-Pairs corpus (N=231 users, 20 tasks, one domain). Cross-corpus generalization is theoretically motivated but empirically unconfirmed. State explicitly in any paper draft.
- The σ-model F-test uses 35 data points from the same entity pool — independence assumption violated. Report with caveat at α=0.05.
- The "dynamic benchmark gap" is addressed by artificially chunking a static dataset. Reviewers at top venues may require truly dynamic context data. Scope the paper's contribution as "incremental computation over sequentially-revealed static context" — not real-time streaming.
- The failure mode taxonomy (A, B, C) was characterized on Task 1, one model (gpt-4o-mini), and 3–5 chunks. Failure rates have not been tested on other tasks, models, or chunk sizes.
- Lazy retraction provides negligible wall-clock benefit at N=231 for monotone tasks. The architectural contribution is the safety condition as a design principle, not the runtime savings.
- The Condition C oracle (single-turn, 25K chars, gpt-4o-mini) is currently unknown. If C > A (likely), this is an honest limitation: single-turn on full context outperforms incremental. The paper's claim is not "incremental beats single-turn" but "incremental achieves near-oracle F1 at 1/5 the per-turn context cost with 22.3% pair-check savings."
