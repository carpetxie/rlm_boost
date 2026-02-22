"""
Label-Aware Check_Pair Experiment V3 — Iteration 12.

This is the MANDATORY ablation identified in the Iteration 12 critique.
It fixes the attribute-overwriting bug found in V2's REPL template and adds
three additional experiments:

## Critical Fix: Attribute-Overwriting Bug (Iteration 12)

In V2's CHUNK_PROMPT_LABEL_AWARE_V2, the `entities` dict is rebuilt from scratch
each chunk. When User X has a qualifying label in chunk 0 (qualifying=True in
EntityCache) but only non-qualifying labels in chunk 2, the V2 template sends
{X: {qualifying: False}} to process_chunk(2, ...), which overwrites EntityCache[X]
with qualifying=False. Then retract_entity(X) removes all X's pairs. At
re-evaluation: X is non-qualifying → pairs NOT re-added. X is permanently wrong.

V3 Fix: After building entities dict from current chunk text, propagate cached
qualifying status for "at least one" monotone conditions:

    for uid, attrs in entities.items():
        cached = _incremental.entity_cache.get(uid)
        if cached and cached.get("qualifying", False):
            attrs["qualifying"] = True  # monotone: once qualifying, stays qualifying

This 2-line fix is the ablation. If A/C jumps from ~64% to ~80%+, the gap was
primarily the bug. If A/C stays ~64%, qualification-time asymmetry is the bottleneck.

## Additional Fixes (Iteration 12)

1. Condition B now uses RLM_SYSTEM_PROMPT (not INCREMENTAL_SYSTEM_PROMPT)
   — B is a single-turn non-incremental baseline; the incremental prompt is wrong.
2. Token cost table: tokens(A) / tokens(C) per task, explicitly stated.
3. k-sensitivity sweep: k ∈ {3, 7, 10} on Task 1 (after knowing A2 result).
4. Task 6 qualifying distribution: Gini coefficient across chunks.

## Experiments

- Experiment A2 (Attribute Fix Ablation): Condition A, Task 1, k=5, monotone fix
- Experiment K-Sensitivity: Condition A, Task 1, k ∈ {3, 7, 10}
- Task 6 Distribution Analysis: Gini coefficient per task (no API needed)

## Usage

    export OPENAI_API_KEY=sk-...
    python eval/label_aware_v3_experiment.py --ablation-only   # Experiment A2 only
    python eval/label_aware_v3_experiment.py --task-idx 1      # Full task 1
    python eval/label_aware_v3_experiment.py --k-sweep         # k in {3, 5, 7, 10}
    python eval/label_aware_v3_experiment.py --all-tasks       # Tasks 1, 3, 6
    python eval/label_aware_v3_experiment.py --gini-analysis   # Free, no API
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
    ORACLE_PROMPT_LABEL_AWARE,
)
from eval.label_aware_v2_experiment import (
    _extract_iteration_count,
    _make_sequential_chunks,
    run_condition_c_v2,
)


# ---------------------------------------------------------------------------
# V3: Updated chunk prompt — attribute-overwriting fix (monotone qualifying)
# ---------------------------------------------------------------------------

CHUNK_PROMPT_LABEL_AWARE_V3 = """Task (OOLONG-Pairs Task {task_idx}): Find pairs where BOTH users have at least one
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

# MONOTONE FIX (Iteration 12): propagate cached qualifying status.
# For "at least one qualifying label" tasks: once an entity qualifies in any
# prior chunk, it remains qualifying forever. The entities dict is rebuilt
# from scratch each chunk, so without this, a user who qualified in chunk 0
# but only has non-qualifying labels in chunk 2 would be incorrectly downgraded.
for uid, attrs in entities.items():
    cached = _incremental.entity_cache.get(uid)
    if cached and cached.get("qualifying", False):
        attrs["qualifying"] = True  # monotone: once qualifying, stays qualifying

stats = _incremental.process_chunk({chunk_idx}, entities, pair_checker=check_pair)
pair_results = list(_incremental.pair_tracker.get_pairs())
print(f"Chunk {chunk_num}: {{stats['new_entities']}} new entities, {{stats['updated_entities']}} updated, {{stats['total_pairs']}} pairs, {{stats['pair_checks']}} checks")
print(f"  Qualifying entities this chunk (incl. cached): {{sum(1 for e in entities.values() if e['qualifying'])}}")
print(f"  Total retractions so far: {{_incremental.get_stats()['total_retractions']}}")
```

IMPORTANT: Call `_incremental.process_chunk({chunk_idx}, entities, pair_checker=check_pair)` EXACTLY ONCE
with chunk_idx={chunk_idx}. Do NOT call process_chunk with any other chunk index in this turn.

After the repl block runs successfully, return FINAL_VAR(pair_results).
Do NOT re-read context_0, context_1 etc. on this turn.
"""


# ---------------------------------------------------------------------------
# V3: Corrected Condition B — uses RLM_SYSTEM_PROMPT (not INCREMENTAL)
# ---------------------------------------------------------------------------

CONDITION_B_PROMPT_V3 = """Task (OOLONG-Pairs Task {task_idx}): Find pairs where BOTH users have at least one
instance labeled {label_desc}.

Context format: "Date: [date] || User: [user_id] || Instance: [text] || Label: [cat]"
Entity ID = the numeric user_id. Parse the Label field directly.

Run this code:

```repl
import re
entities = {{}}
qualifying_labels = {qualifying_labels_repr}
for line in context_0.split('\\n'):
    m = re.search(r'User: (\\d+).*?\\|\\| Label: (.+?)$', line)
    if m:
        uid = m.group(1)
        label = m.group(2).strip().lower()
        if uid not in entities:
            entities[uid] = {{"labels": [], "qualifying": False}}
        entities[uid]["labels"].append(label)
        if label in qualifying_labels:
            entities[uid]["qualifying"] = True
stats = _incremental.process_chunk(0, entities, pair_checker=check_pair)
pair_results = list(_incremental.pair_tracker.get_pairs())
print(f"Entities: {{len(entities)}}  Qualifying: {{sum(1 for e in entities.values() if e['qualifying'])}}")
print(f"Pairs found: {{len(pair_results)}}")
```

After the repl block runs successfully, return FINAL_VAR(pair_results).
"""


# ---------------------------------------------------------------------------
# Condition A V3: Incremental RLM with attribute-overwriting fix
# ---------------------------------------------------------------------------

def run_condition_a_v3(
    labeled_context: str,
    gold_pairs: set,
    api_key: str,
    task_idx: int = 1,
    num_chunks: int = 5,
    max_chunk_chars: int = 5000,
    model: str = "gpt-4o-mini",
    verbose: bool = False,
) -> dict:
    """
    Condition A V3 (Attribute-Overwriting Fix): Incremental RLM.

    Key fix from V2: Monotone qualifying status propagation.
    For "at least one qualifying label" tasks: entities that qualified in
    prior chunks retain their qualifying status even when they reappear in
    later chunks with only non-qualifying labels.

    Without this fix:
    - User X qualifies in chunk 0 (EntityCache: qualifying=True)
    - User X appears in chunk 2 with non-qualifying labels only
    - V2 template builds entities = {X: {qualifying: False}}
    - process_chunk(2, {X: {qualifying: False}}) overwrites EntityCache[X]
    - retract_entity(X) removes all X's pairs
    - Re-evaluation: X is now non-qualifying → pairs NOT re-added
    - X is permanently wrong for the rest of the run

    With the monotone fix:
    - After building entities from chunk 2 text, check cache for X
    - If EntityCache[X].qualifying=True, set entities[X]["qualifying"]=True
    - process_chunk(2, {X: {qualifying: True}}) — no downgrade
    """
    from rlm.core.rlm import RLM
    from rlm.utils.prompts import INCREMENTAL_SYSTEM_PROMPT

    os.environ["OPENAI_API_KEY"] = api_key

    checker_setup = make_label_checker_setup(task_idx)
    qualifying_labels = TASK_QUALIFYING_LABELS[task_idx]
    label_desc = TASK_LABEL_DESCRIPTION[task_idx]

    print(f"\n{'=' * 70}")
    print(f"CONDITION A V3 (Attribute Fix): Incremental RLM (k={num_chunks}, {max_chunk_chars} chars/chunk)")
    print(f"Task {task_idx} | Qualifying: {label_desc}")
    print(f"FIX: Monotone qualifying propagation from EntityCache before process_chunk")
    print(f"{'=' * 70}")

    # Sequential chunks from same first 25K chars as Condition C
    chunks = _make_sequential_chunks(labeled_context, num_chunks, max_chunk_chars)
    print(f"Sequential chunk sizes: {[len(c) for c in chunks]} chars")

    # Verify labels in each chunk
    import re
    for i, chunk in enumerate(chunks):
        label_count = len(re.findall(r'\|\| Label: .+?$', chunk, re.MULTILINE))
        qual_hits = sum(1 for m in re.finditer(r'\|\| Label: (.+?)$', chunk, re.MULTILINE)
                        if m.group(1).strip().lower() in qualifying_labels)
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
        root_prompt = CHUNK_PROMPT_LABEL_AWARE_V3.format(
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

        # Extract from REPL state
        env = rlm._persistent_env
        incr = env.locals.get("_incremental") if env and hasattr(env, "locals") else None

        chunks_processed = 0
        direct_pairs = []
        pair_checks_total = 0
        total_retractions = 0
        if incr:
            stats = incr.get_stats()
            chunks_processed = stats.get("chunks_processed", 0)
            pair_checks_total = stats.get("total_pair_checks", 0)
            total_retractions = stats.get("total_retractions", 0)
            direct_pairs = list(incr.pair_tracker.get_pairs())

        # Prune count telemetry
        prune_count_direct = 0
        if hasattr(rlm, "history_manager") and rlm.history_manager is not None:
            prune_count_direct = rlm.history_manager._prune_count

        # Strict compliance metric (== not >)
        delta = chunks_processed - prev_chunks_processed
        compliant = (delta == 1)

        # Phantom chunk detection
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
        print(f"    pairs: {len(direct_pairs)}  pair_checks_total: {pair_checks_total}  retractions: {total_retractions}")
        print(f"    F1={f1_result['f1']}  P={f1_result['precision']}  R={f1_result['recall']}")
        print(f"    input_tokens: {input_tokens}  output_tokens: {output_tokens}  elapsed: {elapsed:.1f}s")

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

    print(f"\n  Condition A V3 Summary (Task {task_idx}, k={num_chunks}):")
    print(f"    Compliance (strict ==): {compliance_rate:.0%} ({compliant_turns}/{num_chunks} turns)")
    print(f"    Phantom chunks: {phantom_chunks_detected}")
    print(f"    F1 progression: {[t['f1'] for t in f1_progression]}")
    print(f"    Final F1={f1_progression[-1]['f1'] if f1_progression else None}")
    print(f"    Total input tokens: {total_input}  Total output tokens: {total_output}")

    return {
        "condition": "A_incremental_label_aware_v3_attribute_fix",
        "version": 3,
        "task_idx": task_idx,
        "model": model,
        "num_chunks": num_chunks,
        "max_chunk_chars": max_chunk_chars,
        "context_type": "labeled_sequential",
        "chunking_strategy": "sequential_5k_from_first_25k",
        "check_pair_type": "label_aware",
        "qualifying_labels": list(qualifying_labels),
        "attribute_fix": "monotone_qualifying_propagation",
        "compliance_rate": compliance_rate,
        "phantom_chunks_detected": phantom_chunks_detected,
        "f1_progression": f1_progression,
        "final_f1": f1_progression[-1]["f1"] if f1_progression else None,
        "final_precision": f1_progression[-1]["precision"] if f1_progression else None,
        "final_recall": f1_progression[-1]["recall"] if f1_progression else None,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "per_turn_tokens": turn_tokens,
    }


# ---------------------------------------------------------------------------
# Condition B V3: Non-incremental baseline — FIXED to use RLM_SYSTEM_PROMPT
# ---------------------------------------------------------------------------

def run_condition_b_v3(
    labeled_context: str,
    gold_pairs: set,
    api_key: str,
    task_idx: int = 1,
    max_chars: int = 5000,
    model: str = "gpt-4o-mini",
    verbose: bool = False,
) -> dict:
    """
    Condition B V3 (Label-Aware): Non-incremental baseline, first 5K chars.

    Fixed from V2: now uses RLM_SYSTEM_PROMPT instead of INCREMENTAL_SYSTEM_PROMPT.
    Condition B is a single-turn non-incremental baseline; the multi-turn incremental
    system prompt is semantically incorrect for it and may have depressed V2's F1.

    If V2's B F1 (0.0193) was artificially depressed by the wrong system prompt,
    the A vs B comparison (A=0.2202, B=0.0193, 11× improvement) would overstate
    the incremental advantage. V3 measures this correctly.
    """
    from rlm.core.rlm import RLM
    from rlm.utils.prompts import RLM_SYSTEM_PROMPT  # FIXED: was INCREMENTAL_SYSTEM_PROMPT

    os.environ["OPENAI_API_KEY"] = api_key

    checker_setup = make_label_checker_setup(task_idx)
    qualifying_labels = TASK_QUALIFYING_LABELS[task_idx]
    label_desc = TASK_LABEL_DESCRIPTION[task_idx]

    context_chunk = labeled_context[:max_chars]

    print(f"\n{'=' * 70}")
    print(f"CONDITION B V3 (Label-Aware): Non-incremental baseline (1 turn, {len(context_chunk)} chars)")
    print(f"Task {task_idx} | Qualifying: {label_desc}")
    print(f"FIXED: Uses RLM_SYSTEM_PROMPT (not INCREMENTAL_SYSTEM_PROMPT)")
    print(f"{'=' * 70}")

    root_prompt = CONDITION_B_PROMPT_V3.format(
        task_idx=task_idx,
        label_desc=label_desc,
        qualifying_labels_repr=repr(qualifying_labels),
    )

    rlm = RLM(
        backend="openai",
        backend_kwargs={"model_name": model},
        environment="local",
        environment_kwargs={"setup_code": checker_setup},
        persistent=True,
        custom_system_prompt=RLM_SYSTEM_PROMPT,  # FIXED
        max_iterations=6,
        verbose=verbose,
    )

    t0 = time.perf_counter()
    completion = rlm.completion(context_chunk, root_prompt=root_prompt)
    elapsed = time.perf_counter() - t0

    # Extract from env.locals
    env = rlm._persistent_env
    pairs = []
    chunks_processed = 0
    pair_checks = 0
    if env and hasattr(env, "locals"):
        incr = env.locals.get("_incremental")
        if incr:
            pairs = list(incr.pair_tracker.get_pairs())
            stats = incr.get_stats()
            chunks_processed = stats.get("chunks_processed", 0)
            pair_checks = stats.get("total_pair_checks", 0)

    iteration_count = _extract_iteration_count(completion.usage_summary)

    prune_count = 0
    if hasattr(rlm, "history_manager") and rlm.history_manager is not None:
        prune_count = rlm.history_manager._prune_count

    rlm.close()

    compliant = chunks_processed >= 1
    f1_result = compute_f1(pairs, gold_pairs)
    input_tokens, output_tokens = _extract_tokens(completion.usage_summary)

    print(f"    compliant: {compliant}  chunks_processed: {chunks_processed}")
    print(f"    pairs: {len(pairs)}  pair_checks: {pair_checks}")
    print(f"    iteration_count: {iteration_count}")
    print(f"    F1={f1_result['f1']}  P={f1_result['precision']}  R={f1_result['recall']}")
    print(f"    input_tokens: {input_tokens}  output_tokens: {output_tokens}  elapsed: {elapsed:.1f}s")

    return {
        "condition": "B_matched_budget_label_aware_v3",
        "version": 3,
        "task_idx": task_idx,
        "model": model,
        "context_chars": len(context_chunk),
        "context_type": "labeled",
        "check_pair_type": "label_aware",
        "system_prompt": "RLM_SYSTEM_PROMPT",  # FIXED from V2's INCREMENTAL_SYSTEM_PROMPT
        "qualifying_labels": list(qualifying_labels),
        "compliant": compliant,
        "chunks_processed": chunks_processed,
        "pair_checks": pair_checks,
        "iteration_count": iteration_count,
        "f1": f1_result["f1"],
        "precision": f1_result["precision"],
        "recall": f1_result["recall"],
        "tp": f1_result.get("tp"),
        "fp": f1_result.get("fp"),
        "fn": f1_result.get("fn"),
        "predicted_pairs": len(pairs),
        "gold_pairs": len(gold_pairs),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "elapsed_sec": round(elapsed, 2),
        "prune_count": prune_count,
    }


# ---------------------------------------------------------------------------
# Qualifying distribution analysis (free — no API calls needed)
# ---------------------------------------------------------------------------

def analyze_qualifying_distribution(
    labeled_context: str,
    task_idx: int,
    num_chunks: int = 5,
    max_chunk_chars: int = 5000,
) -> dict:
    """
    Compute qualifying entity distribution across chunks for a given task.

    Returns per-chunk qualifying entity counts, Gini coefficient, and
    cross-chunk overlap statistics. Used to characterize Task 6's lower
    A/C ratio vs Tasks 1 and 3.

    Gini coefficient interpretation:
    - 0.0 = perfectly uniform (same qualifying entities per chunk)
    - 1.0 = all qualifying entities in one chunk

    High Gini for Task 6 would explain its lower A/C ratio:
    when qualifying entities cluster in specific chunks, the incremental
    algorithm sees most of them together, reducing opportunities for
    incremental pair savings.
    """
    import re

    qualifying_labels = TASK_QUALIFYING_LABELS[task_idx]
    chunks = _make_sequential_chunks(labeled_context, num_chunks, max_chunk_chars)

    # Per-chunk: entity → set of labels seen in THIS chunk
    per_chunk_entities: list[dict[str, set[str]]] = []
    for chunk in chunks:
        chunk_entities: dict[str, set[str]] = {}
        for line in chunk.split("\n"):
            m = re.search(r"User: (\d+).*?\|\| Label: (.+?)$", line)
            if m:
                uid = m.group(1)
                label = m.group(2).strip().lower()
                chunk_entities.setdefault(uid, set()).add(label)
        per_chunk_entities.append(chunk_entities)

    # Per-chunk: qualifying entities IN THAT CHUNK (label visible in chunk text)
    per_chunk_qualifying: list[set[str]] = []
    for chunk_ents in per_chunk_entities:
        qualifying = {
            uid for uid, labels in chunk_ents.items()
            if labels & qualifying_labels
        }
        per_chunk_qualifying.append(qualifying)

    # Cumulative qualifying (monotone, as seen by incremental algorithm with fix)
    cumulative_qualifying: set[str] = set()
    cumulative_per_chunk: list[set[str]] = []
    for q in per_chunk_qualifying:
        cumulative_qualifying |= q
        cumulative_per_chunk.append(cumulative_qualifying.copy())

    # Cross-chunk qualifying overlap
    all_qualifying = cumulative_qualifying
    n_qualifying_total = len(all_qualifying)

    # Per-chunk qualifying counts (entities with qualifying label IN that chunk)
    per_chunk_counts = [len(q) for q in per_chunk_qualifying]

    # Gini coefficient of per-chunk qualifying entity counts
    def gini(values: list[float]) -> float:
        if not values or sum(values) == 0:
            return 0.0
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        cumulative = sum((i + 1) * v for i, v in enumerate(sorted_vals))
        total = sum(sorted_vals)
        return (2 * cumulative) / (n * total) - (n + 1) / n

    gini_coeff = gini([float(c) for c in per_chunk_counts])

    # Entities that FIRST qualify in each chunk (new qualifying per chunk)
    seen_qualifying: set[str] = set()
    first_qualifying_per_chunk: list[int] = []
    for q in per_chunk_qualifying:
        new = q - seen_qualifying
        first_qualifying_per_chunk.append(len(new))
        seen_qualifying |= q

    # Cross-chunk reappearance: entities seen in multiple chunks
    entity_chunk_count: dict[str, int] = {}
    for chunk_ents in per_chunk_entities:
        for uid in chunk_ents:
            entity_chunk_count[uid] = entity_chunk_count.get(uid, 0) + 1

    # Among qualifying entities: how many appear in multiple chunks?
    qual_multi_chunk = sum(
        1 for uid in all_qualifying if entity_chunk_count.get(uid, 0) > 1
    )

    # Theoretical attribute-overwriting exposure:
    # Entities that QUALIFY in chunk i but REAPPEAR in chunk j>i with NO qualifying label
    # These are the entities the bug affects for sure.
    downgrade_risk: list[dict] = []
    for uid in all_qualifying:
        first_qualifying_chunk = None
        for i, q in enumerate(per_chunk_qualifying):
            if uid in q:
                first_qualifying_chunk = i
                break
        if first_qualifying_chunk is None:
            continue
        # Check if uid reappears in later chunks with non-qualifying labels only
        at_risk_chunks = []
        for j in range(first_qualifying_chunk + 1, num_chunks):
            if uid in per_chunk_entities[j]:
                chunk_labels = per_chunk_entities[j][uid]
                if not (chunk_labels & qualifying_labels):
                    at_risk_chunks.append(j)
        if at_risk_chunks:
            downgrade_risk.append({
                "uid": uid,
                "first_qualifying_chunk": first_qualifying_chunk,
                "non_qualifying_reappearance_chunks": at_risk_chunks,
            })

    return {
        "task_idx": task_idx,
        "task_label": TASK_LABEL_DESCRIPTION[task_idx],
        "num_chunks": num_chunks,
        "max_chunk_chars": max_chunk_chars,
        "total_qualifying_entities": n_qualifying_total,
        "per_chunk_qualifying_counts": per_chunk_counts,
        "per_chunk_first_qualifying": first_qualifying_per_chunk,
        "gini_coefficient": round(gini_coeff, 4),
        "qualifying_multi_chunk_count": qual_multi_chunk,
        "qualifying_multi_chunk_fraction": round(qual_multi_chunk / n_qualifying_total, 4) if n_qualifying_total else 0,
        "attribute_overwriting_at_risk_count": len(downgrade_risk),
        "attribute_overwriting_at_risk_fraction": round(len(downgrade_risk) / n_qualifying_total, 4) if n_qualifying_total else 0,
        "at_risk_entities": downgrade_risk[:20],  # first 20 for inspection
    }


# ---------------------------------------------------------------------------
# k-sensitivity sweep (Condition A V3 with varying k)
# ---------------------------------------------------------------------------

def run_k_sensitivity_sweep(
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
    Run Condition A V3 with k ∈ {3, 5, 7, 10} chunks over the same 25K window.

    Per k, reports:
    - F1(A), F1(C), A/C ratio
    - Total input tokens: A vs C (ratio = streaming cost premium)
    - Compliance rate

    The paper's core scalability figure: A/C ratio vs k.
    At k=10 (2.5K chars/chunk): expect lower A/C (less qualifying density per chunk).
    At k=3 (8.3K chars/chunk): expect higher A/C (more qualifying per chunk).

    Also computes the iso-cost k: where total_tokens(A) ≈ total_tokens(C).
    """
    if k_values is None:
        k_values = [3, 7, 10]

    print(f"\n{'#' * 70}")
    print(f"# K-SENSITIVITY SWEEP — Task {task_idx}, k ∈ {k_values}")
    print(f"# Fixed 25K window: each k divides {total_chars} chars")
    print(f"{'#' * 70}")

    results_by_k: dict[int, dict] = {}

    for k in k_values:
        max_chunk_chars = total_chars // k
        print(f"\n{'=' * 60}")
        print(f"k={k}: {k} chunks × {max_chunk_chars} chars/chunk")
        print(f"{'=' * 60}")

        result_a = run_condition_a_v3(
            labeled_context=labeled_context,
            gold_pairs=gold_pairs,
            api_key=api_key,
            task_idx=task_idx,
            num_chunks=k,
            max_chunk_chars=max_chunk_chars,
            model=model,
            verbose=verbose,
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
            "compliance_rate": result_a.get("compliance_rate", 0),
        }

        print(f"\nk={k} summary: F1(A)={f1_a:.4f}  F1(C)={f1_c:.4f}  A/C={ratio:.1%}"
              f"  tokens(A)/tokens(C)={token_ratio:.2f}")

    # Summary table
    print(f"\n{'=' * 80}")
    print(f"K-SENSITIVITY SWEEP SUMMARY — Task {task_idx}")
    print(f"{'=' * 80}")
    print(f"{'k':>4} {'chars/chunk':>12} {'F1(A)':>8} {'F1(C)':>8} {'A/C':>8}"
          f" {'tok(A)':>8} {'tok(C)':>8} {'tok(A)/tok(C)':>14} {'Compl':>8}")
    print("-" * 80)
    for k, r in sorted(results_by_k.items()):
        print(f"{r['k']:>4} {r['max_chunk_chars']:>12} {r['f1_a']:>8.4f} {r['f1_c']:>8.4f}"
              f" {r['ac_ratio']:>7.1%} {r['total_tokens_a']:>8} {r['total_tokens_c']:>8}"
              f" {r['token_ratio_a_over_c']:>14.2f} {r['compliance_rate']:>7.0%}")

    return {
        "task_idx": task_idx,
        "model": model,
        "total_chars": total_chars,
        "k_values": k_values,
        "results_by_k": {str(k): v for k, v in results_by_k.items()},
    }


# ---------------------------------------------------------------------------
# Run single task (full V3: A, B, C with attribute fix)
# ---------------------------------------------------------------------------

def run_task_v3(
    task_idx: int,
    labeled_context: str,
    api_key: str,
    model: str,
    num_chunks: int,
    max_chunk_chars: int,
    run_b: bool,
    verbose: bool,
) -> dict:
    """Run V3 experiment (Conditions A + B + C) for a single task."""
    import re

    gold_pairs = compute_gold_pairs(labeled_context, task_idx=task_idx)
    qualifying_labels = TASK_QUALIFYING_LABELS[task_idx]
    print(f"\nGold pairs (Task {task_idx}, full labeled context): {len(gold_pairs)}")

    # Count qualifying entities in first 25K chars
    window = labeled_context[: num_chunks * max_chunk_chars]
    entities_25k: dict[str, dict] = {}
    for line in window.split("\n"):
        m = re.search(r"User: (\d+).*?\|\| Label: (.+?)$", line)
        if m:
            uid = m.group(1)
            label = m.group(2).strip().lower()
            if uid not in entities_25k:
                entities_25k[uid] = {"qualifying": False}
            if label in qualifying_labels:
                entities_25k[uid]["qualifying"] = True
    qualifying_25k = [uid for uid, a in entities_25k.items() if a["qualifying"]]
    pairs_25k_count = len(qualifying_25k) * (len(qualifying_25k) - 1) // 2
    coverage_ceiling = pairs_25k_count / len(gold_pairs) if gold_pairs else 0
    print(f"  Within first {num_chunks * max_chunk_chars} chars: {len(entities_25k)} entities,"
          f" {len(qualifying_25k)} qualifying, {pairs_25k_count} pairs (ceiling: {coverage_ceiling:.1%})")

    results: dict = {
        "task_idx": task_idx,
        "model": model,
        "version": 3,
        "gold_pairs_count": len(gold_pairs),
        "pairs_in_first_25k": pairs_25k_count,
        "entities_in_first_25k": len(entities_25k),
        "qualifying_entities_in_first_25k": len(qualifying_25k),
        "coverage_ceiling_25k": round(coverage_ceiling, 4),
        "qualifying_labels": list(qualifying_labels),
        "iteration": 12,
        "fix": "monotone_qualifying_propagation",
    }

    result_a = run_condition_a_v3(
        labeled_context=labeled_context,
        gold_pairs=gold_pairs,
        api_key=api_key,
        task_idx=task_idx,
        num_chunks=num_chunks,
        max_chunk_chars=max_chunk_chars,
        model=model,
        verbose=verbose,
    )
    results["condition_a"] = result_a

    if run_b:
        result_b = run_condition_b_v3(
            labeled_context=labeled_context,
            gold_pairs=gold_pairs,
            api_key=api_key,
            task_idx=task_idx,
            max_chars=max_chunk_chars,
            model=model,
            verbose=verbose,
        )
        results["condition_b"] = result_b

    result_c = run_condition_c_v2(
        labeled_context=labeled_context,
        gold_pairs=gold_pairs,
        api_key=api_key,
        task_idx=task_idx,
        max_chars=num_chunks * max_chunk_chars,
        model=model,
        verbose=verbose,
    )
    results["condition_c"] = result_c

    # A vs C comparison
    f1_a = results["condition_a"].get("final_f1", 0) or 0
    f1_c = results["condition_c"].get("f1", 0) or 0
    ratio = f1_a / f1_c if f1_c > 0 else 0
    tokens_a = results["condition_a"].get("total_input_tokens", 0)
    tokens_c = results["condition_c"].get("input_tokens", 0)
    token_ratio = tokens_a / tokens_c if tokens_c > 0 else 0

    results["a_vs_c_ratio"] = round(ratio, 4)
    results["token_ratio_a_over_c"] = round(token_ratio, 4)

    # Summary table
    print(f"\n{'=' * 75}")
    print(f"V3 COMPARISON TABLE — Task {task_idx} ({TASK_LABEL_DESCRIPTION[task_idx]})")
    print(f"FIX: Monotone qualifying propagation (attribute-overwriting ablation)")
    print(f"{'=' * 75}")
    ra = results["condition_a"]
    print(f"A V3 (Incremental, k=5, attribute fix):  F1={ra.get('final_f1', 'N/A')}"
          f"  P={ra.get('final_precision', 'N/A')}  R={ra.get('final_recall', 'N/A')}"
          f"  tokens={ra.get('total_input_tokens', 'N/A')}")
    rc = results["condition_c"]
    print(f"C V2 (Oracle, 1T, 25K):                  F1={rc.get('f1', 'N/A')}"
          f"  P={rc.get('precision', 'N/A')}  R={rc.get('recall', 'N/A')}"
          f"  tokens={rc.get('input_tokens', 'N/A')}")
    print(f"\n  A/C ratio (V3, with attribute fix): {ratio:.1%}")
    print(f"  Token ratio A/C: {token_ratio:.2f}x")
    print(f"  Coverage ceiling: {coverage_ceiling:.1%}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Label-Aware Experiment V3 — Attribute-Overwriting Ablation (Iteration 12)"
    )
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--num-chunks", type=int, default=5)
    parser.add_argument("--max-chunk-chars", type=int, default=5000)
    parser.add_argument("--task-idx", type=int, default=1, choices=[1, 3, 6])
    parser.add_argument("--all-tasks", action="store_true", help="Run Tasks 1, 3, 6")
    parser.add_argument("--ablation-only", action="store_true",
                        help="Only run Condition A (attribute fix ablation, skip B and C)")
    parser.add_argument("--run-b", action="store_true",
                        help="Also run Condition B with corrected RLM_SYSTEM_PROMPT")
    parser.add_argument("--k-sweep", action="store_true",
                        help="Run k-sensitivity sweep (k ∈ {3, 7, 10}) on Task 1")
    parser.add_argument("--k-values", type=int, nargs="+", default=[3, 7, 10],
                        help="k values for k-sensitivity sweep")
    parser.add_argument("--gini-analysis", action="store_true",
                        help="Run qualifying distribution analysis (free, no API)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # Load API key
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key and not args.gini_analysis:
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("OPENAI_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break
        if not api_key:
            print("ERROR: No OPENAI_API_KEY found")
            sys.exit(1)

    print("Loading OOLONG-Pairs labeled data...")
    _, labeled_context = load_labeled_data()
    print(f"Labeled context length: {len(labeled_context)} chars")

    all_results: dict = {}

    # --- Gini analysis (free, always runs if requested) ---
    if args.gini_analysis:
        print("\n" + "=" * 70)
        print("QUALIFYING DISTRIBUTION ANALYSIS (no API needed)")
        print("=" * 70)
        gini_results: dict = {}
        for t in [1, 3, 6]:
            dist = analyze_qualifying_distribution(
                labeled_context,
                task_idx=t,
                num_chunks=args.num_chunks,
                max_chunk_chars=args.max_chunk_chars,
            )
            gini_results[f"task_{t}"] = dist
            print(f"\nTask {t} ({TASK_LABEL_DESCRIPTION[t]}):")
            print(f"  Total qualifying: {dist['total_qualifying_entities']}")
            print(f"  Per-chunk qualifying counts: {dist['per_chunk_qualifying_counts']}")
            print(f"  Per-chunk first-qualifying: {dist['per_chunk_first_qualifying']}")
            print(f"  Gini coefficient: {dist['gini_coefficient']:.4f}")
            print(f"  Multi-chunk qualifying fraction: {dist['qualifying_multi_chunk_fraction']:.1%}")
            print(f"  Attribute-overwriting at-risk: {dist['attribute_overwriting_at_risk_count']}"
                  f" ({dist['attribute_overwriting_at_risk_fraction']:.1%} of qualifying)")

        gini_path = Path("results/streaming/qualifying_distribution_v3.json")
        gini_path.parent.mkdir(parents=True, exist_ok=True)
        with open(gini_path, "w") as f:
            json.dump(gini_results, f, indent=2, default=str)
        print(f"\nGini analysis saved to {gini_path}")
        all_results["gini_analysis"] = gini_results

        if not api_key:
            return all_results

    # --- k-sensitivity sweep ---
    if args.k_sweep:
        gold_pairs_t1 = compute_gold_pairs(labeled_context, task_idx=1)
        sweep_results = run_k_sensitivity_sweep(
            labeled_context=labeled_context,
            gold_pairs=gold_pairs_t1,
            api_key=api_key,
            task_idx=1,
            k_values=args.k_values,
            total_chars=args.num_chunks * args.max_chunk_chars,
            model=args.model,
            verbose=not args.quiet,
        )
        all_results["k_sweep"] = sweep_results
        sweep_path = Path("results/streaming/k_sensitivity_v3.json")
        sweep_path.parent.mkdir(parents=True, exist_ok=True)
        with open(sweep_path, "w") as f:
            json.dump(sweep_results, f, indent=2, default=str)
        print(f"k-sensitivity sweep saved to {sweep_path}")

    # --- Main task experiments ---
    tasks_to_run = [1, 3, 6] if args.all_tasks else [args.task_idx]

    for task_idx in tasks_to_run:
        print(f"\n{'#' * 75}")
        print(f"# TASK {task_idx}: {TASK_LABEL_DESCRIPTION[task_idx]}")
        print(f"{'#' * 75}")

        labeled_context_local = labeled_context
        gold_pairs = compute_gold_pairs(labeled_context_local, task_idx=task_idx)

        if args.ablation_only:
            # Only run Condition A (attribute fix ablation)
            result_a = run_condition_a_v3(
                labeled_context=labeled_context_local,
                gold_pairs=gold_pairs,
                api_key=api_key,
                task_idx=task_idx,
                num_chunks=args.num_chunks,
                max_chunk_chars=args.max_chunk_chars,
                model=args.model,
                verbose=not args.quiet,
            )
            task_results = {
                "task_idx": task_idx,
                "version": 3,
                "condition_a": result_a,
                "mode": "ablation_only",
            }
        else:
            task_results = run_task_v3(
                task_idx=task_idx,
                labeled_context=labeled_context_local,
                api_key=api_key,
                model=args.model,
                num_chunks=args.num_chunks,
                max_chunk_chars=args.max_chunk_chars,
                run_b=args.run_b,
                verbose=not args.quiet,
            )

        all_results[f"task_{task_idx}"] = task_results

        # Save per-task
        if args.output is None:
            out_path = Path(f"results/streaming/label_aware_task{task_idx}_v3_results.json")
        else:
            out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(task_results, f, indent=2, default=str)
        print(f"Task {task_idx} V3 results saved to {out_path}")

    # Save combined results if multiple tasks
    if args.all_tasks or args.k_sweep or args.gini_analysis:
        combined_path = Path("results/streaming/label_aware_v3_combined.json")
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nCombined V3 results saved to {combined_path}")

    return all_results


if __name__ == "__main__":
    main()
