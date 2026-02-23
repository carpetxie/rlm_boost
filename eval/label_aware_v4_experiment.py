"""
Label-Aware Check_Pair Experiment V4 — Iteration 13.

This is the library-level monotone_attrs fix. The V3 monotone qualifying fix lived
entirely in the REPL template (a 6-line propagation loop), making the template ~25
lines long and causing stochastic compliance failures (~40% of runs in V3 Run 1).

V4 moves the monotone semantics to the library:
    process_chunk(chunk_idx, entities, check_pair, monotone_attrs={"qualifying"})

The REPL template shrinks back to V2 complexity (~15 lines), eliminating the
compliance fragility. Expected outcome: 100% compliance on every run, token
overhead near V2's 1.14× (vs V3's 2.42× at 100% compliance, 4.84× at 60%).

## Key architectural change

V3 REPL template (25 lines, stochastic compliance):
    # Parse entities from chunk text
    ...
    # MONOTONE FIX (6 lines):
    for uid, attrs in entities.items():
        cached = _incremental.entity_cache.get(uid)
        if cached and cached.get("qualifying", False):
            attrs["qualifying"] = True
    # Then call process_chunk
    stats = _incremental.process_chunk(chunk_idx, entities, pair_checker=check_pair)

V4 REPL template (15 lines, deterministic compliance):
    # Parse entities from chunk text
    ...
    # Library handles monotone merge internally:
    stats = _incremental.process_chunk(chunk_idx, entities, pair_checker=check_pair,
                                       monotone_attrs={"qualifying"})

Additional V4 improvements:
- Reports noop_retractions vs permanent_retractions (new get_stats() fields)
- Tracks per-chunk retraction type breakdown
- k-sensitivity sweep uses V4 template

## Experiments

- Experiment A3: Task 1, k=5, library-level monotone_attrs (multi-run stability)
- Experiment K-Sweep V4: k ∈ {3, 5, 7, 10} with simplified template
- Task 3, Task 6 V4: Cross-task validation (at-risk fraction prediction)
- Condition B V4: Corrected system prompt + library monotone (sanity check)
- Task 11 Non-monotone: monotone_attrs=None on asymmetric task (sanity check)

## Usage

    export OPENAI_API_KEY=sk-...
    python eval/label_aware_v4_experiment.py --task 1          # Task 1, k=5 (A3)
    python eval/label_aware_v4_experiment.py --task 1 --k-sweep  # k ∈ {3,5,7,10}
    python eval/label_aware_v4_experiment.py --all-tasks       # Tasks 1, 3, 6
    python eval/label_aware_v4_experiment.py --task 11         # Non-monotone sanity
    python eval/label_aware_v4_experiment.py --condition-b     # Condition B V4
    python eval/label_aware_v4_experiment.py --multi-run N     # N runs of Task 1
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.rlm_pipeline_experiment import compute_f1, compute_gold_pairs
from eval.f1_progression_experiment import _extract_tokens
from eval.label_aware_experiment import (
    TASK_QUALIFYING_LABELS,
    TASK_LABEL_DESCRIPTION,
    make_label_checker_setup,
    load_labeled_data,
)
from eval.label_aware_v2_experiment import (
    _extract_iteration_count,
    _make_sequential_chunks,
    run_condition_c_v2,
)


# ---------------------------------------------------------------------------
# V4: SIMPLIFIED chunk prompt — library handles monotone qualifying merge
# ---------------------------------------------------------------------------

CHUNK_PROMPT_LABEL_AWARE_V4 = """Task (OOLONG-Pairs Task {task_idx}): Find pairs where BOTH users have at least one
instance labeled {label_desc}. This is Chunk {chunk_num} of {total_chunks}.

Context format: "Date: [date] || User: [user_id] || Instance: [text] || Label: [cat]"
Entity ID = the numeric user_id (e.g. "44436"). Parse the Label field directly.

Run this code (chunk index {chunk_idx}):

```repl
import re
entities = {{}}
qualifying_labels = {qualifying_labels_repr}
for line in context_{chunk_idx}.split('\\n'):
    m = re.search(r'User: (\\d+).*?\\|\\| Label: (.+?)$', line)
    if m:
        uid = m.group(1)
        label = m.group(2).strip().lower()
        if uid not in entities:
            entities[uid] = {{"labels": [], "qualifying": False}}
        entities[uid]["labels"].append(label)
        if label in qualifying_labels:
            entities[uid]["qualifying"] = True

stats = _incremental.process_chunk({chunk_idx}, entities, pair_checker=check_pair, monotone_attrs={{"qualifying"}})
pair_results = list(_incremental.pair_tracker.get_pairs())
print(f"Chunk {chunk_num}: {{stats['new_entities']}} new entities, {{stats['updated_entities']}} updated, {{stats['total_pairs']}} pairs, {{stats['pair_checks']}} checks")
print(f"  Noop retractions: {{stats.get('noop_retractions', 0)}}  Permanent: {{stats.get('permanent_retractions', 0)}}")
print(f"  Total retractions so far: {{_incremental.get_stats()['total_retractions']}}")
```

IMPORTANT: Call `_incremental.process_chunk({chunk_idx}, entities, pair_checker=check_pair, monotone_attrs={{"qualifying"}})` EXACTLY ONCE
with chunk_idx={chunk_idx}. Do NOT call process_chunk with any other chunk index in this turn.
The library automatically preserves qualifying=True for entities that qualified in prior chunks.

After the repl block runs successfully, return FINAL_VAR(pair_results).
Do NOT re-read context_0, context_1 etc. on this turn.
"""


# ---------------------------------------------------------------------------
# V4 non-monotone prompt — for Task 11 (exactly-N conditions, no monotone merge)
# ---------------------------------------------------------------------------

CHUNK_PROMPT_LABEL_AWARE_V4_NON_MONOTONE = """Task (OOLONG-Pairs Task {task_idx}): {task_description}
This is Chunk {chunk_num} of {total_chunks}.

Context format: "Date: [date] || User: [user_id] || Instance: [text] || Label: [cat]"
Entity ID = the numeric user_id. Parse the Label field directly.

Run this code (chunk index {chunk_idx}):

```repl
import re
entities = {{}}
qualifying_labels = {qualifying_labels_repr}
for line in context_{chunk_idx}.split('\\n'):
    m = re.search(r'User: (\\d+).*?\\|\\| Label: (.+?)$', line)
    if m:
        uid = m.group(1)
        label = m.group(2).strip().lower()
        if uid not in entities:
            entities[uid] = {{"labels": [], "qualifying": False}}
        entities[uid]["labels"].append(label)

stats = _incremental.process_chunk({chunk_idx}, entities, pair_checker=check_pair)
pair_results = list(_incremental.pair_tracker.get_pairs())
print(f"Chunk {chunk_num}: {{stats['new_entities']}} new entities, {{stats['updated_entities']}} updated, {{stats['total_pairs']}} pairs")
```

IMPORTANT: Call `_incremental.process_chunk({chunk_idx}, entities, pair_checker=check_pair)` EXACTLY ONCE
with chunk_idx={chunk_idx}. Do NOT call process_chunk with any other chunk index in this turn.
Note: monotone_attrs is NOT used here — this task has non-monotone conditions.

After the repl block runs successfully, return FINAL_VAR(pair_results).
"""


# Non-monotone task descriptions (for Task 11)
TASK_DESCRIPTIONS = {
    11: "Find pairs where User A has exactly 1 entity label AND User B has abbreviation + entity labels",
    13: "Find pairs where User A has exactly 1 description label AND User B has abbreviation + entity labels",
}


# ---------------------------------------------------------------------------
# Condition A V4: Incremental RLM with library-level monotone_attrs
# ---------------------------------------------------------------------------

def run_condition_a_v4(
    labeled_context: str,
    gold_pairs: set,
    api_key: str,
    task_idx: int = 1,
    num_chunks: int = 5,
    max_chunk_chars: int = 5000,
    model: str = "gpt-4o-mini",
    verbose: bool = False,
    run_id: int = 1,
) -> dict:
    """
    Condition A V4 (Library-Level Monotone Fix): Incremental RLM.

    Key improvement from V3: monotone qualifying semantics moved to library
    (process_chunk monotone_attrs parameter) instead of REPL template loop.

    V3 REPL template: ~25 lines (6-line propagation loop causes stochastic
    compliance failure — V3 Run 1 had 60% compliance, 4.84× token overhead).

    V4 REPL template: ~15 lines (library call with monotone_attrs={"qualifying"}).
    Expected: 100% compliance deterministically, token overhead near V2's 1.14×.

    The library-level fix also:
    - Skips retraction entirely for no-op updates (qualifying True→True after merge)
    - Eliminates 1,078 no-op retraction cycles from V3 Run 2
    - Tracks noop_retractions vs permanent_retractions in get_stats()
    """
    from rlm.core.rlm import RLM
    from rlm.utils.prompts import INCREMENTAL_SYSTEM_PROMPT

    os.environ["OPENAI_API_KEY"] = api_key

    checker_setup = make_label_checker_setup(task_idx)
    qualifying_labels = TASK_QUALIFYING_LABELS[task_idx]
    label_desc = TASK_LABEL_DESCRIPTION[task_idx]

    print(f"\n{'=' * 70}")
    print(f"CONDITION A V4 (Library Monotone Fix): Incremental RLM")
    print(f"  Run {run_id} | k={num_chunks}, {max_chunk_chars} chars/chunk")
    print(f"  Task {task_idx} | Qualifying: {label_desc}")
    print(f"  FIX: Library-level monotone_attrs={{\"qualifying\"}} in process_chunk()")
    print(f"  Expected: 100% compliance, ~1.14× token overhead (vs V3's ~2.42×)")
    print(f"{'=' * 70}")

    chunks = _make_sequential_chunks(labeled_context, num_chunks, max_chunk_chars)
    print(f"Sequential chunk sizes: {[len(c) for c in chunks]} chars")

    import re as re_mod
    for i, chunk in enumerate(chunks):
        label_count = len(re_mod.findall(r'\|\| Label: .+?$', chunk, re_mod.MULTILINE))
        qual_hits = sum(
            1 for m in re_mod.finditer(r'\|\| Label: (.+?)$', chunk, re_mod.MULTILINE)
            if m.group(1).strip().lower() in qualifying_labels
        )
        print(f"  Chunk {i}: {label_count} labeled records, {qual_hits} qualifying-label instances")

    rlm = RLM(
        backend="openai",
        backend_kwargs={"model_name": model},
        environment="local",
        environment_kwargs={"setup_code": checker_setup},
        persistent=True,
        custom_system_prompt=INCREMENTAL_SYSTEM_PROMPT,
        max_iterations=6,
        verbose=verbose,
    )

    f1_progression = []
    turn_tokens = []
    prev_chunks_processed = 0
    phantom_chunks_detected = 0

    for chunk_i, chunk in enumerate(chunks):
        chunk_num = chunk_i + 1
        root_prompt = CHUNK_PROMPT_LABEL_AWARE_V4.format(
            task_idx=task_idx,
            label_desc=label_desc,
            chunk_num=chunk_num,
            total_chunks=num_chunks,
            chunk_idx=chunk_i,
            qualifying_labels_repr=repr(qualifying_labels),
        )

        print(f"\n  --- Turn {chunk_num}/{num_chunks} ---")
        t0 = time.perf_counter()
        completion = rlm.completion(chunk, root_prompt=root_prompt)
        elapsed = time.perf_counter() - t0

        env = rlm._persistent_env
        incr = env.locals.get("_incremental") if env and hasattr(env, "locals") else None

        chunks_processed = 0
        direct_pairs = []
        pair_checks_total = 0
        total_retractions = 0
        noop_retractions = 0
        permanent_retractions = 0

        if incr:
            stats = incr.get_stats()
            chunks_processed = stats.get("chunks_processed", 0)
            pair_checks_total = stats.get("total_pair_checks", 0)
            total_retractions = stats.get("total_retractions", 0)
            noop_retractions = stats.get("noop_retractions", 0)
            permanent_retractions = stats.get("permanent_retractions", 0)
            direct_pairs = list(incr.pair_tracker.get_pairs())

        prune_count_direct = 0
        if hasattr(rlm, "history_manager") and rlm.history_manager is not None:
            prune_count_direct = rlm.history_manager._prune_count

        delta = chunks_processed - prev_chunks_processed
        compliant = (delta == 1)

        phantom = (delta > 1)
        if phantom:
            phantom_chunks_detected += 1
            print(f"    ⚠ PHANTOM CHUNK DETECTED: chunks_processed advanced by {delta}")
        elif delta == 0:
            print(f"    ⚠ NON-COMPLIANT: chunks_processed did not advance (delta=0)")

        prev_chunks_processed = chunks_processed

        iteration_count = _extract_iteration_count(completion.usage_summary)
        f1_result = compute_f1(direct_pairs, gold_pairs)
        input_tokens, output_tokens = _extract_tokens(completion.usage_summary)
        turn_tokens.append({"turn": chunk_num, "input": input_tokens, "output": output_tokens})

        print(f"    compliant: {compliant}  chunks_processed: {chunks_processed}  delta: {delta}")
        print(f"    pairs: {len(direct_pairs)}  pair_checks: {pair_checks_total}")
        print(f"    retractions: {total_retractions} (noop={noop_retractions}, permanent={permanent_retractions})")
        print(f"    F1={f1_result['f1']}  P={f1_result['precision']}  R={f1_result['recall']}")
        print(f"    input_tokens: {input_tokens}  output_tokens: {output_tokens}  elapsed: {elapsed:.1f}s")
        print(f"    iteration_count: {iteration_count}  prune_count: {prune_count_direct}")

        f1_progression.append({
            "chunk": chunk_num,
            "chunk_idx": chunk_i,
            "compliant": compliant,
            "delta": delta,
            "phantom_chunk": phantom,
            "chunks_processed": chunks_processed,
            "pairs": len(direct_pairs),
            "pair_checks_total": pair_checks_total,
            "total_retractions": total_retractions,
            "noop_retractions": noop_retractions,
            "permanent_retractions": permanent_retractions,
            "f1": f1_result["f1"],
            "precision": f1_result["precision"],
            "recall": f1_result["recall"],
            "tp": f1_result.get("tp"),
            "fp": f1_result.get("fp"),
            "fn": f1_result.get("fn"),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "elapsed_sec": round(elapsed, 2),
            "prune_count": prune_count_direct,
            "iteration_count": iteration_count,
        })

    rlm.close()

    compliant_turns = sum(1 for t in f1_progression if t["compliant"])
    compliance_rate = compliant_turns / num_chunks if num_chunks > 0 else 0.0
    total_input = sum(t["input"] for t in turn_tokens)
    total_output = sum(t["output"] for t in turn_tokens)
    # Use LAST turn's cumulative values (get_stats() returns cumulative totals,
    # NOT per-turn deltas). Summing across turns would triple-count.
    final_noop = f1_progression[-1]["noop_retractions"] if f1_progression else 0
    final_permanent = f1_progression[-1]["permanent_retractions"] if f1_progression else 0
    final_total_retractions = f1_progression[-1]["total_retractions"] if f1_progression else 0

    print(f"\n  Condition A V4 Summary (Task {task_idx}, k={num_chunks}, Run {run_id}):")
    print(f"    Compliance (strict ==): {compliance_rate:.0%} ({compliant_turns}/{num_chunks} turns)")
    print(f"    Phantom chunks: {phantom_chunks_detected}")
    print(f"    F1 progression: {[round(t['f1'], 4) for t in f1_progression]}")
    print(f"    Final F1={f1_progression[-1]['f1'] if f1_progression else None}")
    print(f"    Total retractions: {final_total_retractions} "
          f"(noop={final_noop}, permanent={final_permanent})")
    print(f"    Total input tokens: {total_input}  Total output tokens: {total_output}")

    return {
        "condition": "A_incremental_label_aware_v4_library_monotone",
        "version": 4,
        "run_id": run_id,
        "task_idx": task_idx,
        "model": model,
        "num_chunks": num_chunks,
        "max_chunk_chars": max_chunk_chars,
        "context_type": "labeled_sequential",
        "chunking_strategy": "sequential_from_first_25k",
        "check_pair_type": "label_aware",
        "qualifying_labels": list(qualifying_labels),
        "attribute_fix": "library_level_monotone_attrs",
        "compliance_rate": compliance_rate,
        "phantom_chunks_detected": phantom_chunks_detected,
        "f1_progression": f1_progression,
        "final_f1": f1_progression[-1]["f1"] if f1_progression else None,
        "final_precision": f1_progression[-1]["precision"] if f1_progression else None,
        "final_recall": f1_progression[-1]["recall"] if f1_progression else None,
        "total_retractions": final_total_retractions,
        "total_noop_retractions": final_noop,
        "total_permanent_retractions": final_permanent,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "per_turn_tokens": turn_tokens,
    }


# ---------------------------------------------------------------------------
# Non-monotone sanity check (Task 11 / 13) — monotone_attrs=None
# ---------------------------------------------------------------------------

def run_condition_a_v4_non_monotone(
    labeled_context: str,
    gold_pairs: set,
    api_key: str,
    task_idx: int = 11,
    num_chunks: int = 5,
    max_chunk_chars: int = 5000,
    model: str = "gpt-4o-mini",
    verbose: bool = False,
) -> dict:
    """
    Non-monotone sanity check: run V4 framework with monotone_attrs=None.

    For "exactly N" tasks (Task 11: exactly 1 entity label), the qualifying
    condition is non-monotone — a user who has exactly 1 entity label in chunk 0
    may have 2+ entity labels by chunk 3, becoming non-qualifying. The monotone
    merge would incorrectly preserve the qualifying=True status.

    EXPECTED RESULT: A/C ratio ≈ V2 baseline for Task 11 (~64%, not ~94%).
    This validates that monotone_attrs=None correctly handles non-monotone conditions
    and that the V4 improvement is specifically due to monotone semantics.
    """
    from rlm.core.rlm import RLM
    from rlm.utils.prompts import INCREMENTAL_SYSTEM_PROMPT

    os.environ["OPENAI_API_KEY"] = api_key

    if task_idx == 11:
        # Task 11: role-asymmetric, count-based condition — not in TASK_QUALIFYING_LABELS.
        # Role A: has entity AND abbreviation; Role B: exactly 1 entity label.
        checker_setup = """
# Task 11 non-monotone check_pair: role-asymmetric, count-based.
def check_pair(attrs1, attrs2):
    def role_a(a):
        return a.get("entity", 0) >= 1 and a.get("abbreviation", 0) >= 1
    def role_b(a):
        return a.get("entity", 0) == 1
    return (role_a(attrs1) and role_b(attrs2)) or (role_a(attrs2) and role_b(attrs1))
"""
    elif task_idx == 13:
        # Task 13: role-asymmetric — role A: exactly 1 description; role B: abbreviation+entity.
        checker_setup = """
# Task 13 non-monotone check_pair: role-asymmetric, count-based.
def check_pair(attrs1, attrs2):
    def role_a(a):
        return a.get("description and abstract concept", 0) == 1
    def role_b(a):
        return a.get("abbreviation", 0) >= 1 and a.get("entity", 0) >= 1
    return (role_a(attrs1) and role_b(attrs2)) or (role_a(attrs2) and role_b(attrs1))
"""
    else:
        checker_setup = make_label_checker_setup(task_idx)
    qualifying_labels = TASK_QUALIFYING_LABELS.get(task_idx, set())
    label_desc = TASK_LABEL_DESCRIPTION.get(task_idx, f"Task {task_idx}")
    task_description = TASK_DESCRIPTIONS.get(task_idx, label_desc)

    print(f"\n{'=' * 70}")
    print(f"NON-MONOTONE SANITY CHECK (V4, monotone_attrs=None): Task {task_idx}")
    print(f"  k={num_chunks}, {max_chunk_chars} chars/chunk")
    print(f"  EXPECTED: A/C ≈ V2 baseline (~64%), NOT ~94% (fix has no effect)")
    print(f"{'=' * 70}")

    chunks = _make_sequential_chunks(labeled_context, num_chunks, max_chunk_chars)

    rlm = RLM(
        backend="openai",
        backend_kwargs={"model_name": model},
        environment="local",
        environment_kwargs={"setup_code": checker_setup},
        persistent=True,
        custom_system_prompt=INCREMENTAL_SYSTEM_PROMPT,
        max_iterations=6,
        verbose=verbose,
    )

    f1_progression = []
    turn_tokens = []
    prev_chunks_processed = 0

    for chunk_i, chunk in enumerate(chunks):
        chunk_num = chunk_i + 1
        root_prompt = CHUNK_PROMPT_LABEL_AWARE_V4_NON_MONOTONE.format(
            task_idx=task_idx,
            task_description=task_description,
            chunk_num=chunk_num,
            total_chunks=num_chunks,
            chunk_idx=chunk_i,
            qualifying_labels_repr=repr(qualifying_labels),
        )

        print(f"\n  --- Turn {chunk_num}/{num_chunks} ---")
        t0 = time.perf_counter()
        completion = rlm.completion(chunk, root_prompt=root_prompt)
        elapsed = time.perf_counter() - t0

        env = rlm._persistent_env
        incr = env.locals.get("_incremental") if env and hasattr(env, "locals") else None

        chunks_processed = 0
        direct_pairs = []
        if incr:
            chunks_processed = incr.get_stats().get("chunks_processed", 0)
            direct_pairs = list(incr.pair_tracker.get_pairs())

        delta = chunks_processed - prev_chunks_processed
        compliant = (delta == 1)
        prev_chunks_processed = chunks_processed

        f1_result = compute_f1(direct_pairs, gold_pairs)
        input_tokens, output_tokens = _extract_tokens(completion.usage_summary)
        turn_tokens.append({"turn": chunk_num, "input": input_tokens, "output": output_tokens})

        print(f"    compliant: {compliant}  pairs: {len(direct_pairs)}")
        print(f"    F1={f1_result['f1']}  P={f1_result['precision']}  R={f1_result['recall']}")
        print(f"    input_tokens: {input_tokens}  elapsed: {elapsed:.1f}s")

        f1_progression.append({
            "chunk": chunk_num,
            "compliant": compliant,
            "pairs": len(direct_pairs),
            "f1": f1_result["f1"],
            "precision": f1_result["precision"],
            "recall": f1_result["recall"],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "elapsed_sec": round(elapsed, 2),
        })

    rlm.close()

    compliant_turns = sum(1 for t in f1_progression if t["compliant"])
    compliance_rate = compliant_turns / num_chunks if num_chunks > 0 else 0.0
    total_input = sum(t["input"] for t in turn_tokens)

    print(f"\n  Non-Monotone Sanity Check Summary (Task {task_idx}):")
    print(f"    Compliance: {compliance_rate:.0%} ({compliant_turns}/{num_chunks} turns)")
    print(f"    F1 progression: {[round(t['f1'], 4) for t in f1_progression]}")
    print(f"    Final F1={f1_progression[-1]['f1'] if f1_progression else None}")
    print(f"    Total input tokens: {total_input}")

    return {
        "condition": "A_non_monotone_sanity_check_v4",
        "version": 4,
        "task_idx": task_idx,
        "model": model,
        "num_chunks": num_chunks,
        "max_chunk_chars": max_chunk_chars,
        "monotone_attrs": None,
        "compliance_rate": compliance_rate,
        "f1_progression": f1_progression,
        "final_f1": f1_progression[-1]["f1"] if f1_progression else None,
        "total_input_tokens": total_input,
        "total_output_tokens": sum(t["output"] for t in turn_tokens),
        "per_turn_tokens": turn_tokens,
    }


# ---------------------------------------------------------------------------
# Multi-run stability: run Condition A V4 N times
# ---------------------------------------------------------------------------

def run_multi_run_stability(
    labeled_context: str,
    gold_pairs: set,
    api_key: str,
    task_idx: int = 1,
    num_runs: int = 3,
    num_chunks: int = 5,
    max_chunk_chars: int = 5000,
    model: str = "gpt-4o-mini",
    verbose: bool = False,
) -> dict:
    """
    Run Condition A V4 N times to measure stability.

    Reports: mean ± std of compliance_rate, final_f1, total_input_tokens.
    Target for publishable claim: std(compliance_rate) = 0 (all 100%), std(F1) < 0.05.
    """
    import statistics

    print(f"\n{'#' * 70}")
    print(f"# MULTI-RUN STABILITY: Task {task_idx}, {num_runs} runs, k={num_chunks}")
    print(f"# Target: compliance=100% every run, F1 std < 0.05")
    print(f"{'#' * 70}")

    run_results = []
    for run_id in range(1, num_runs + 1):
        result = run_condition_a_v4(
            labeled_context=labeled_context,
            gold_pairs=gold_pairs,
            api_key=api_key,
            task_idx=task_idx,
            num_chunks=num_chunks,
            max_chunk_chars=max_chunk_chars,
            model=model,
            verbose=verbose,
            run_id=run_id,
        )
        run_results.append(result)

    compliance_rates = [r["compliance_rate"] for r in run_results]
    final_f1s = [r["final_f1"] or 0 for r in run_results]
    token_totals = [r["total_input_tokens"] for r in run_results]

    mean_compliance = statistics.mean(compliance_rates)
    mean_f1 = statistics.mean(final_f1s)
    mean_tokens = statistics.mean(token_totals)
    std_compliance = statistics.stdev(compliance_rates) if len(compliance_rates) > 1 else 0.0
    std_f1 = statistics.stdev(final_f1s) if len(final_f1s) > 1 else 0.0
    std_tokens = statistics.stdev(token_totals) if len(token_totals) > 1 else 0.0

    print(f"\n{'=' * 70}")
    print(f"MULTI-RUN STABILITY SUMMARY (Task {task_idx}, {num_runs} runs)")
    print(f"{'=' * 70}")
    print(f"{'Metric':<30} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 70)
    print(f"{'Compliance Rate':<30} {mean_compliance:>10.1%} {std_compliance:>10.3f}"
          f" {min(compliance_rates):>10.1%} {max(compliance_rates):>10.1%}")
    print(f"{'Final F1':<30} {mean_f1:>10.4f} {std_f1:>10.4f}"
          f" {min(final_f1s):>10.4f} {max(final_f1s):>10.4f}")
    print(f"{'Total Input Tokens':<30} {mean_tokens:>10.0f} {std_tokens:>10.0f}"
          f" {min(token_totals):>10} {max(token_totals):>10}")

    publishable = (mean_compliance >= 0.9 and std_f1 < 0.05)
    print(f"\nPublishable claim: {'YES' if publishable else 'NO — std too high or compliance < 90%'}")
    if publishable:
        print(f"  Paper claim: 'F1={mean_f1:.4f} ± {std_f1:.4f} across {num_runs} runs (compliance={mean_compliance:.0%})'")

    return {
        "experiment": "multi_run_stability_v4",
        "task_idx": task_idx,
        "num_runs": num_runs,
        "num_chunks": num_chunks,
        "model": model,
        "run_results": run_results,
        "summary": {
            "compliance_rate": {"mean": mean_compliance, "std": std_compliance,
                                "min": min(compliance_rates), "max": max(compliance_rates)},
            "final_f1": {"mean": mean_f1, "std": std_f1,
                         "min": min(final_f1s), "max": max(final_f1s)},
            "total_input_tokens": {"mean": mean_tokens, "std": std_tokens,
                                   "min": min(token_totals), "max": max(token_totals)},
        },
        "publishable": publishable,
    }


# ---------------------------------------------------------------------------
# k-sensitivity sweep with V4 (simplified template)
# ---------------------------------------------------------------------------

def run_k_sensitivity_sweep_v4(
    labeled_context: str,
    gold_pairs: set,
    api_key: str,
    task_idx: int = 1,
    k_values: list[int] | None = None,
    total_chars: int = 25000,
    model: str = "gpt-4o-mini",
    verbose: bool = False,
) -> dict:
    """
    k-sensitivity sweep with V4 (library-level monotone fix, simplified template).

    Tests k ∈ {3, 5, 7, 10} — the paper's primary scalability figure.
    Prediction from first principles:
    - k=3 (8.3K chars/chunk): more entities per chunk → higher A/C (less asymmetry)
    - k=10 (2.5K chars/chunk): fewer entities per chunk → lower A/C (more asymmetry)

    Reports:
    - A/C ratio vs k (Figure 1 of the paper)
    - tokens(A)/tokens(C) vs k (cost premium)
    - Compliance rate vs k
    - Iso-cost k: smallest k where tokens(A) ≤ 1.5×tokens(C)
    """
    if k_values is None:
        k_values = [3, 5, 7, 10]

    print(f"\n{'#' * 70}")
    print(f"# K-SENSITIVITY SWEEP V4 — Task {task_idx}, k ∈ {k_values}")
    print(f"# V4: library monotone_attrs, simplified template, expected 100% compliance")
    print(f"# Fixed 25K window: each k divides {total_chars} chars")
    print(f"{'#' * 70}")

    results_by_k: dict[int, dict] = {}

    for k in k_values:
        max_chunk_chars = total_chars // k
        print(f"\n{'=' * 60}")
        print(f"k={k}: {k} chunks × {max_chunk_chars} chars/chunk")
        print(f"{'=' * 60}")

        result_a = run_condition_a_v4(
            labeled_context=labeled_context,
            gold_pairs=gold_pairs,
            api_key=api_key,
            task_idx=task_idx,
            num_chunks=k,
            max_chunk_chars=max_chunk_chars,
            model=model,
            verbose=verbose,
            run_id=1,
        )

        result_c = run_condition_c_v2(
            labeled_context=labeled_context,
            gold_pairs=gold_pairs,
            api_key=api_key,
            task_idx=task_idx,
            max_chars=total_chars,
            model=model,
            verbose=verbose,
        )

        f1_a = result_a.get("final_f1", 0) or 0
        f1_c = result_c.get("f1", 0) or 0
        ratio = f1_a / f1_c if f1_c > 0 else 0
        tokens_a = result_a.get("total_input_tokens", 0)
        tokens_c = result_c.get("input_tokens", 0)
        token_ratio = tokens_a / tokens_c if tokens_c > 0 else 0
        compliance = result_a.get("compliance_rate", 0)
        noop_ret = result_a.get("total_noop_retractions", 0)
        perm_ret = result_a.get("total_permanent_retractions", 0)

        results_by_k[k] = {
            "k": k,
            "max_chunk_chars": max_chunk_chars,
            "result_a": result_a,
            "result_c": result_c,
            "f1_a": f1_a,
            "f1_c": f1_c,
            "ac_ratio": round(ratio, 4),
            "total_tokens_a": tokens_a,
            "total_tokens_c": tokens_c,
            "token_ratio_a_over_c": round(token_ratio, 4),
            "compliance_rate": compliance,
            "noop_retractions": noop_ret,
            "permanent_retractions": perm_ret,
        }

        print(f"\nk={k}: F1(A)={f1_a:.4f}  F1(C)={f1_c:.4f}  A/C={ratio:.1%}"
              f"  tok(A)/tok(C)={token_ratio:.2f}  compliance={compliance:.0%}"
              f"  noop_ret={noop_ret}  perm_ret={perm_ret}")

    # Summary table
    print(f"\n{'=' * 90}")
    print(f"K-SENSITIVITY SWEEP V4 SUMMARY — Task {task_idx}")
    print(f"{'=' * 90}")
    print(f"{'k':>4} {'chars/chunk':>12} {'F1(A)':>8} {'F1(C)':>8} {'A/C':>8}"
          f" {'tok(A)/tok(C)':>14} {'Compl':>8} {'noop_ret':>10} {'perm_ret':>10}")
    print("-" * 90)
    for k in sorted(results_by_k.keys()):
        r = results_by_k[k]
        print(f"{r['k']:>4} {r['max_chunk_chars']:>12} {r['f1_a']:>8.4f} {r['f1_c']:>8.4f}"
              f" {r['ac_ratio']:>7.1%} {r['token_ratio_a_over_c']:>14.2f} {r['compliance_rate']:>7.0%}"
              f" {r['noop_retractions']:>10} {r['permanent_retractions']:>10}")

    # Compute iso-cost k
    iso_cost_k = None
    for k in sorted(results_by_k.keys()):
        r = results_by_k[k]
        if r["token_ratio_a_over_c"] <= 1.5:
            iso_cost_k = k
            break

    if iso_cost_k:
        print(f"\nIso-cost k (tokens(A) ≤ 1.5×tokens(C)): k={iso_cost_k}")
    else:
        print(f"\nIso-cost k: tokens(A) > 1.5×tokens(C) at all tested k values")

    return {
        "task_idx": task_idx,
        "model": model,
        "version": 4,
        "total_chars": total_chars,
        "k_values": k_values,
        "results_by_k": {str(k): v for k, v in results_by_k.items()},
        "iso_cost_k": iso_cost_k,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Label-Aware Experiment V4 (Library Monotone Fix)")
    parser.add_argument("--task", type=int, default=1, choices=[1, 3, 6, 11, 13],
                        help="Task index (default=1)")
    parser.add_argument("--k", type=int, default=5, help="Number of chunks (default=5)")
    parser.add_argument("--k-sweep", action="store_true", help="Run k-sensitivity sweep k∈{3,5,7,10}")
    parser.add_argument("--k-values", type=str, default="3,5,7,10",
                        help="Comma-separated k values for sweep (default=3,5,7,10)")
    parser.add_argument("--all-tasks", action="store_true", help="Run Tasks 1, 3, 6")
    parser.add_argument("--non-monotone", action="store_true",
                        help="Non-monotone sanity check (Task 11, monotone_attrs=None)")
    parser.add_argument("--multi-run", type=int, default=0, metavar="N",
                        help="Run Task 1 N times for stability analysis")
    parser.add_argument("--condition-b", action="store_true", help="Run Condition B V4")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-dir", default="results/streaming",
                        help="Output directory for results JSON")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("OPENAI_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, labeled_context = load_labeled_data()

    tasks_to_run = [1, 3, 6] if args.all_tasks else [args.task]

    if args.multi_run > 0:
        # Multi-run stability analysis
        gold_pairs = compute_gold_pairs(labeled_context, task_idx=1)
        result = run_multi_run_stability(
            labeled_context=labeled_context,
            gold_pairs=gold_pairs,
            api_key=api_key,
            task_idx=1,
            num_runs=args.multi_run,
            num_chunks=args.k,
            model=args.model,
            verbose=args.verbose,
        )
        out_path = output_dir / f"label_aware_task1_v4_multi_run_{args.multi_run}.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"\nSaved to {out_path}")

    elif args.k_sweep:
        # k-sensitivity sweep
        gold_pairs = compute_gold_pairs(labeled_context, task_idx=args.task)
        k_values = [int(x) for x in args.k_values.split(",")]
        result = run_k_sensitivity_sweep_v4(
            labeled_context=labeled_context,
            gold_pairs=gold_pairs,
            api_key=api_key,
            task_idx=args.task,
            k_values=k_values,
            model=args.model,
            verbose=args.verbose,
        )
        out_path = output_dir / f"label_aware_task{args.task}_v4_k_sensitivity.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"\nSaved to {out_path}")

    elif args.non_monotone:
        # Non-monotone sanity check
        task_idx = 11
        gold_pairs = compute_gold_pairs(labeled_context, task_idx=task_idx)
        result = run_condition_a_v4_non_monotone(
            labeled_context=labeled_context,
            gold_pairs=gold_pairs,
            api_key=api_key,
            task_idx=task_idx,
            num_chunks=args.k,
            model=args.model,
            verbose=args.verbose,
        )
        out_path = output_dir / f"label_aware_task{task_idx}_v4_non_monotone.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"\nSaved to {out_path}")

    elif args.condition_b:
        # Condition B V4 — uses RLM_SYSTEM_PROMPT
        from eval.label_aware_v3_experiment import run_condition_b_v3
        task_idx = args.task
        gold_pairs = compute_gold_pairs(labeled_context, task_idx=task_idx)
        result = run_condition_b_v3(
            labeled_context=labeled_context,
            gold_pairs=gold_pairs,
            api_key=api_key,
            task_idx=task_idx,
            model=args.model,
            verbose=args.verbose,
        )
        out_path = output_dir / f"label_aware_task{task_idx}_v4_condition_b.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"\nSaved to {out_path}")

    else:
        # Single task run
        all_results = {}
        for task_idx in tasks_to_run:
            gold_pairs = compute_gold_pairs(labeled_context, task_idx=task_idx)
            result_a = run_condition_a_v4(
                labeled_context=labeled_context,
                gold_pairs=gold_pairs,
                api_key=api_key,
                task_idx=task_idx,
                num_chunks=args.k,
                max_chunk_chars=25000 // args.k,
                model=args.model,
                verbose=args.verbose,
            )
            result_c = run_condition_c_v2(
                labeled_context=labeled_context,
                gold_pairs=gold_pairs,
                api_key=api_key,
                task_idx=task_idx,
                max_chars=25000,
                model=args.model,
                verbose=args.verbose,
            )

            f1_a = result_a.get("final_f1", 0) or 0
            f1_c = result_c.get("f1", 0) or 0
            ac_ratio = f1_a / f1_c if f1_c > 0 else 0
            tok_a = result_a.get("total_input_tokens", 0)
            tok_c = result_c.get("input_tokens", 0)
            tok_ratio = tok_a / tok_c if tok_c > 0 else 0

            print(f"\nTask {task_idx} SUMMARY:")
            print(f"  A: F1={f1_a:.4f}  tokens={tok_a}")
            print(f"  C: F1={f1_c:.4f}  tokens={tok_c}")
            print(f"  A/C: {ac_ratio:.1%}  tok(A)/tok(C): {tok_ratio:.2f}")
            print(f"  Noop ret: {result_a.get('total_noop_retractions', 0)}"
                  f"  Permanent ret: {result_a.get('total_permanent_retractions', 0)}")

            all_results[task_idx] = {
                "task_idx": task_idx,
                "result_a": result_a,
                "result_c": result_c,
                "f1_a": f1_a,
                "f1_c": f1_c,
                "ac_ratio": round(ac_ratio, 4),
                "token_ratio": round(tok_ratio, 4),
            }

            out_path = output_dir / f"label_aware_task{task_idx}_v4_results.json"
            out_path.write_text(json.dumps(all_results[task_idx], indent=2))
            print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
