"""
Dynamic Context Proof-of-Concept — Iteration 17.

This experiment validates the retraction mechanism on genuinely dynamic context:
entities whose attributes CHANGE between turns (not just new entities arriving).

## Design

- Turn 1: Process chunk 0 normally (entities parsed with labels)
- Turn 2: Process chunk 1 normally (incremental processing)
- Turn 3: "Edit" — Re-inject a MODIFIED version of chunk 0 where N entities
  have their labels changed (e.g., qualifying -> non-qualifying or vice versa).
  This simulates a document edit / streaming correction.
- Turn 4: Process chunk 2 normally (continued incremental processing after edit)

## What This Tests

1. Retraction mechanism fires correctly for modified entities (permanent retractions)
2. PairTracker correctly removes pairs involving entities that lost qualification
3. PairTracker correctly adds new pairs for entities that gained qualification
4. Final F1 is computed against UPDATED ground truth (reflecting the edits)
5. The system continues operating correctly after an edit (Turn 4)

## Why This Matters

All prior experiments process static OOLONG-Pairs data chunked sequentially.
The "Dynamic RLM" thesis motivation requires demonstrating that the system handles
genuinely changing context — not just new context arriving. This experiment is the
minimum viable proof that the retraction mechanism works end-to-end in a live pipeline.

## Implementation

Uses the existing V4 framework (run_condition_a_v4) with modifications:
- After Turn 2, mutate chunk 0's data to change N entities' labels
- Re-submit the modified chunk as a new chunk (chunk_index=0 would be idempotent,
  so we use a special "edit chunk" approach: reset the relevant entities and
  reprocess them)

Actually, the cleanest approach: use IncrementalState directly in the REPL code.
The edit turn calls:
1. For each modified entity: update its attributes in entity_cache
2. pair_tracker.retract_entity() for each modified entity
3. Re-check pairs for the modified entities

This tests the retraction mechanism at the library level through a live API call.

Usage:
    export OPENAI_API_KEY=sk-...
    python eval/dynamic_context_experiment.py
    python eval/dynamic_context_experiment.py --num-edits 5
    python eval/dynamic_context_experiment.py --num-edits 10
"""

from __future__ import annotations

import json
import os
import re
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
from eval.label_aware_v4_experiment import CHUNK_PROMPT_LABEL_AWARE_V4


# ---------------------------------------------------------------------------
# Dynamic context: edit prompt — tells the model to update entities in-place
# ---------------------------------------------------------------------------

EDIT_PROMPT = """DYNAMIC CONTEXT UPDATE: Some entities from a previous chunk have changed.
This simulates a document edit where {num_edits} entities had their labels corrected.

The following entities need to be UPDATED (their qualifying status may have changed):

{edit_description}

Run this code to apply the edits using the apply_edits() API:

```repl
# Dynamic context update: use the library-level apply_edits() method
edits = {edits_dict_repr}

edit_stats = _incremental.apply_edits(edits, pair_checker=check_pair, edit_chunk_index={edit_chunk_idx})

pair_results = list(_incremental.pair_tracker.get_pairs())
print(f"Edit complete: {{edit_stats['total_retracted']}} pairs retracted, {{edit_stats['pairs_readded']}} re-added, {{edit_stats['new_pairs_from_edits']}} new")
print(f"Pairs: {{edit_stats['pairs_before']}} -> {{edit_stats['pairs_after']}}")
print(f"Stats: {{_incremental.get_stats()}}")
```

IMPORTANT: Run the code block above EXACTLY. This applies the entity edits and
correctly retracts/re-evaluates affected pairs via the library API.

After the repl block runs successfully, return FINAL_VAR(pair_results).
"""


# ---------------------------------------------------------------------------
# Helper: find entities in chunk text and select N to modify
# ---------------------------------------------------------------------------

def find_entities_in_chunk(chunk_text: str, qualifying_labels: set[str]) -> dict[str, dict]:
    """Parse entities from a chunk, returning {uid: {labels: [...], qualifying: bool}}."""
    entities = {}
    for line in chunk_text.split('\n'):
        m = re.search(r'User: (\d+).*?\|\| Label: (.+?)$', line)
        if m:
            uid = m.group(1)
            label = m.group(2).strip().lower()
            if uid not in entities:
                entities[uid] = {"labels": [], "qualifying": False}
            entities[uid]["labels"].append(label)
            if label in qualifying_labels:
                entities[uid]["qualifying"] = True
    return entities


def select_entities_to_edit(
    entities: dict[str, dict],
    num_edits: int,
    qualifying_labels: set[str],
    all_labels: set[str] | None = None,
) -> dict[str, dict]:
    """Select entities to edit: flip their qualifying status.

    Strategy:
    - Pick num_edits//2 qualifying entities -> make non-qualifying (downgrade)
    - Pick num_edits//2 non-qualifying entities -> make qualifying (upgrade)
    This tests both retraction (pair removal) and new pair creation.
    """
    qualifying = {uid: e for uid, e in entities.items() if e["qualifying"]}
    non_qualifying = {uid: e for uid, e in entities.items() if not e["qualifying"]}

    num_downgrade = min(num_edits // 2, len(qualifying))
    num_upgrade = min(num_edits - num_downgrade, len(non_qualifying))

    edits = {}

    # Downgrade: qualifying -> non-qualifying
    for i, (uid, attrs) in enumerate(sorted(qualifying.items())):
        if i >= num_downgrade:
            break
        # Replace all labels with a non-qualifying one
        edits[uid] = {
            "labels": ["entity"],  # non-qualifying label
            "qualifying": False,
            "edit_type": "downgrade",
        }

    # Upgrade: non-qualifying -> qualifying
    target_label = sorted(qualifying_labels)[0]  # pick first qualifying label
    for i, (uid, attrs) in enumerate(sorted(non_qualifying.items())):
        if i >= num_upgrade:
            break
        edits[uid] = {
            "labels": [target_label],  # qualifying label
            "qualifying": True,
            "edit_type": "upgrade",
        }

    return edits


def compute_gold_pairs_with_edits(
    labeled_context_window: str,
    qualifying_labels: set[str],
    edits: dict[str, dict],
) -> set[tuple[str, str]]:
    """Compute gold pairs from the labeled context with edits applied.

    This is the UPDATED ground truth after applying entity edits.
    """
    # Parse all entities from the full context window
    all_entities = find_entities_in_chunk(labeled_context_window, qualifying_labels)

    # Apply edits
    for uid, edit_attrs in edits.items():
        if uid in all_entities:
            all_entities[uid] = {
                "labels": edit_attrs["labels"],
                "qualifying": edit_attrs["qualifying"],
            }

    # Compute qualifying pairs
    qualifying_ids = sorted(uid for uid, e in all_entities.items() if e["qualifying"])
    pairs = set()
    for i, id1 in enumerate(qualifying_ids):
        for id2 in qualifying_ids[i + 1:]:
            pairs.add((min(id1, id2), max(id1, id2)))

    return pairs


# ---------------------------------------------------------------------------
# Main experiment: dynamic context with entity edits
# ---------------------------------------------------------------------------

def run_dynamic_context_experiment(
    labeled_context: str,
    gold_pairs_original: set,
    api_key: str,
    task_idx: int = 1,
    num_chunks: int = 4,  # 4 turns: chunk0, chunk1, EDIT, chunk2
    max_chunk_chars: int = 5000,
    num_edits: int = 5,
    model: str = "gpt-4o-mini",
    verbose: bool = False,
) -> dict:
    """
    Dynamic context experiment: 4-turn pipeline with an entity edit in Turn 3.

    Turn 1: Process chunk 0 (normal incremental)
    Turn 2: Process chunk 1 (normal incremental)
    Turn 3: EDIT — modify num_edits entities from chunk 0 (flip qualifying status)
    Turn 4: Process chunk 2 (normal incremental, post-edit)

    Measures:
    - Retractions fired correctly for edited entities
    - Pairs updated correctly (removed for downgraded, added for upgraded)
    - F1 against UPDATED ground truth
    - System continues correctly after edit (Turn 4)
    """
    from rlm.core.rlm import RLM
    from rlm.utils.prompts import INCREMENTAL_SYSTEM_PROMPT

    os.environ["OPENAI_API_KEY"] = api_key

    checker_setup = make_label_checker_setup(task_idx)
    qualifying_labels = TASK_QUALIFYING_LABELS[task_idx]
    label_desc = TASK_LABEL_DESCRIPTION[task_idx]

    print(f"\n{'=' * 70}")
    print(f"DYNAMIC CONTEXT EXPERIMENT: Entity Edits in Live Pipeline")
    print(f"  Task {task_idx} | k={num_chunks} chunks + 1 edit turn")
    print(f"  Qualifying: {label_desc}")
    print(f"  num_edits: {num_edits} entities will be modified in Turn 3")
    print(f"  Tests: retraction mechanism, pair update correctness, post-edit continuation")
    print(f"{'=' * 70}")

    # Create chunks — we need num_chunks data chunks plus one edit turn
    # Total context = (num_chunks) * max_chunk_chars
    total_context_chars = num_chunks * max_chunk_chars
    chunks = _make_sequential_chunks(labeled_context, num_chunks, max_chunk_chars)
    print(f"Data chunk sizes: {[len(c) for c in chunks]} chars")

    # Pre-parse chunk 0 entities to plan edits
    chunk0_entities = find_entities_in_chunk(chunks[0], qualifying_labels)
    print(f"Chunk 0: {len(chunk0_entities)} entities ({sum(1 for e in chunk0_entities.values() if e['qualifying'])} qualifying)")

    # Select entities to edit
    edits = select_entities_to_edit(chunk0_entities, num_edits, qualifying_labels)
    num_downgrade = sum(1 for e in edits.values() if e.get("edit_type") == "downgrade")
    num_upgrade = sum(1 for e in edits.values() if e.get("edit_type") == "upgrade")
    print(f"Edits planned: {len(edits)} total ({num_downgrade} downgrade, {num_upgrade} upgrade)")
    for uid, edit in edits.items():
        orig = chunk0_entities.get(uid, {})
        print(f"  Entity {uid}: {orig.get('qualifying', '?')} -> {edit['qualifying']} ({edit['edit_type']})")

    # Compute gold pairs for pre-edit and post-edit ground truth
    context_window = labeled_context[:total_context_chars]
    gold_pairs_post_edit = compute_gold_pairs_with_edits(
        context_window, qualifying_labels, edits,
    )
    print(f"\nGold pairs (original, within {total_context_chars} chars): {len(gold_pairs_original)}")
    print(f"Gold pairs (post-edit): {len(gold_pairs_post_edit)}")
    print(f"Gold pair delta: {len(gold_pairs_post_edit) - len(gold_pairs_original)}")

    # Initialize RLM
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

    results_by_turn = []
    total_wall_clock = 0.0
    total_input_tokens = 0
    total_output_tokens = 0

    # --- Turn 1: Process chunk 0 ---
    print(f"\n{'─' * 50}")
    print(f"Turn 1/4: Process chunk 0 (normal incremental)")
    print(f"{'─' * 50}")
    t0 = time.perf_counter()
    root_prompt_1 = CHUNK_PROMPT_LABEL_AWARE_V4.format(
        task_idx=task_idx,
        label_desc=label_desc,
        chunk_num=1,
        total_chunks=num_chunks,
        chunk_idx=0,
        qualifying_labels_repr=repr(qualifying_labels),
    )
    completion = rlm.completion(chunks[0], root_prompt=root_prompt_1)
    elapsed = time.perf_counter() - t0
    total_wall_clock += elapsed

    env = rlm._persistent_env
    incr = env.locals.get("_incremental") if env and hasattr(env, "locals") else None
    pairs_t1 = list(incr.pair_tracker.get_pairs()) if incr else []
    stats_t1 = incr.get_stats() if incr else {}
    input_tokens, output_tokens = _extract_tokens(completion.usage_summary)
    total_input_tokens += input_tokens
    total_output_tokens += output_tokens
    f1_t1 = compute_f1(pairs_t1, gold_pairs_original)

    print(f"  Pairs: {len(pairs_t1)}, Entities: {stats_t1.get('total_entities', 0)}")
    print(f"  F1 (vs original gold): {f1_t1['f1']:.4f}")
    print(f"  Tokens: {input_tokens} in, {output_tokens} out, {elapsed:.1f}s")

    results_by_turn.append({
        "turn": 1, "type": "chunk", "chunk_idx": 0,
        "pairs": len(pairs_t1), "f1_original": f1_t1["f1"],
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "elapsed_sec": round(elapsed, 2),
        "stats": stats_t1,
    })

    # --- Turn 2: Process chunk 1 ---
    print(f"\n{'─' * 50}")
    print(f"Turn 2/4: Process chunk 1 (normal incremental)")
    print(f"{'─' * 50}")
    t0 = time.perf_counter()
    root_prompt_2 = CHUNK_PROMPT_LABEL_AWARE_V4.format(
        task_idx=task_idx,
        label_desc=label_desc,
        chunk_num=2,
        total_chunks=num_chunks,
        chunk_idx=1,
        qualifying_labels_repr=repr(qualifying_labels),
    )
    completion = rlm.completion(chunks[1], root_prompt=root_prompt_2)
    elapsed = time.perf_counter() - t0
    total_wall_clock += elapsed

    incr = env.locals.get("_incremental") if env and hasattr(env, "locals") else None
    pairs_t2 = list(incr.pair_tracker.get_pairs()) if incr else []
    stats_t2 = incr.get_stats() if incr else {}
    input_tokens, output_tokens = _extract_tokens(completion.usage_summary)
    total_input_tokens += input_tokens
    total_output_tokens += output_tokens
    f1_t2 = compute_f1(pairs_t2, gold_pairs_original)
    pre_edit_pairs = len(pairs_t2)

    print(f"  Pairs: {len(pairs_t2)}, Entities: {stats_t2.get('total_entities', 0)}")
    print(f"  F1 (vs original gold): {f1_t2['f1']:.4f}")
    print(f"  Retractions so far: {stats_t2.get('total_retractions', 0)}")
    print(f"  Tokens: {input_tokens} in, {output_tokens} out, {elapsed:.1f}s")

    results_by_turn.append({
        "turn": 2, "type": "chunk", "chunk_idx": 1,
        "pairs": len(pairs_t2), "f1_original": f1_t2["f1"],
        "pre_edit_pairs": pre_edit_pairs,
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "elapsed_sec": round(elapsed, 2),
        "stats": stats_t2,
    })

    # --- Turn 3: EDIT — Modify entities from chunk 0 ---
    print(f"\n{'─' * 50}")
    print(f"Turn 3/4: DYNAMIC EDIT — Modify {len(edits)} entities")
    print(f"{'─' * 50}")

    # Build the edit description for the prompt
    edit_lines = []
    for uid, edit_attrs in edits.items():
        orig = chunk0_entities.get(uid, {})
        old_q = orig.get("qualifying", False)
        new_q = edit_attrs["qualifying"]
        direction = "DOWNGRADE (was qualifying, now not)" if edit_attrs["edit_type"] == "downgrade" else "UPGRADE (was not qualifying, now is)"
        edit_lines.append(f"- Entity {uid}: {direction}")

    # Build the edits dict repr for the REPL code
    edits_for_repl = {}
    for uid, edit_attrs in edits.items():
        edits_for_repl[uid] = {
            "labels": edit_attrs["labels"],
            "qualifying": edit_attrs["qualifying"],
        }

    edit_prompt = EDIT_PROMPT.format(
        num_edits=len(edits),
        edit_description="\n".join(edit_lines),
        edits_dict_repr=repr(edits_for_repl),
        edit_chunk_idx=99,  # Use a special chunk index for the edit
    )

    t0 = time.perf_counter()
    # Pass a minimal context (the edit is self-contained in the prompt)
    completion = rlm.completion(
        "DYNAMIC CONTEXT UPDATE — see instructions above.",
        root_prompt=edit_prompt,
    )
    elapsed = time.perf_counter() - t0
    total_wall_clock += elapsed

    incr = env.locals.get("_incremental") if env and hasattr(env, "locals") else None
    pairs_t3 = list(incr.pair_tracker.get_pairs()) if incr else []
    stats_t3 = incr.get_stats() if incr else {}
    input_tokens, output_tokens = _extract_tokens(completion.usage_summary)
    total_input_tokens += input_tokens
    total_output_tokens += output_tokens

    # Compute F1 against BOTH original and updated ground truth
    f1_t3_original = compute_f1(pairs_t3, gold_pairs_original)
    f1_t3_updated = compute_f1(pairs_t3, gold_pairs_post_edit)
    post_edit_pairs = len(pairs_t3)

    # Note: retractions via direct pair_tracker.retract_entity() in the edit code
    # update pair_tracker.retraction_count but NOT incr._total_retractions
    # (which is only updated by process_chunk). Use pair_tracker's own counter.
    retractions_t3_process_chunk = stats_t3.get("total_retractions", 0) - stats_t2.get("total_retractions", 0)
    retractions_t3_tracker = incr.pair_tracker.retraction_count if incr else 0
    # The tracker count includes all retractions from the entire run; subtract
    # the count at end of T2 (which was stats_t2 total_retractions)
    retractions_t3 = max(retractions_t3_process_chunk, retractions_t3_tracker - stats_t2.get("total_retractions", 0))

    print(f"  Pairs before edit: {pre_edit_pairs}")
    print(f"  Pairs after edit: {post_edit_pairs}")
    print(f"  Pair delta: {post_edit_pairs - pre_edit_pairs}")
    print(f"  Retractions this turn: {retractions_t3} (tracker: {retractions_t3_tracker})")
    print(f"  F1 (vs original gold): {f1_t3_original['f1']:.4f}")
    print(f"  F1 (vs UPDATED gold): {f1_t3_updated['f1']:.4f}")
    print(f"  Tokens: {input_tokens} in, {output_tokens} out, {elapsed:.1f}s")

    results_by_turn.append({
        "turn": 3, "type": "edit",
        "num_edits": len(edits),
        "num_downgrade": num_downgrade,
        "num_upgrade": num_upgrade,
        "pairs_before_edit": pre_edit_pairs,
        "pairs_after_edit": post_edit_pairs,
        "pair_delta": post_edit_pairs - pre_edit_pairs,
        "retractions_this_turn": retractions_t3,
        "f1_original": f1_t3_original["f1"],
        "f1_updated": f1_t3_updated["f1"],
        "precision_updated": f1_t3_updated["precision"],
        "recall_updated": f1_t3_updated["recall"],
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "elapsed_sec": round(elapsed, 2),
        "stats": stats_t3,
        "edits": {uid: {"qualifying": e["qualifying"], "edit_type": e["edit_type"]}
                  for uid, e in edits.items()},
    })

    # --- Turn 4: Process chunk 2 (post-edit continuation) ---
    print(f"\n{'─' * 50}")
    print(f"Turn 4/4: Process chunk 2 (post-edit continuation)")
    print(f"{'─' * 50}")
    t0 = time.perf_counter()
    root_prompt_4 = CHUNK_PROMPT_LABEL_AWARE_V4.format(
        task_idx=task_idx,
        label_desc=label_desc,
        chunk_num=3,
        total_chunks=num_chunks,
        chunk_idx=2,
        qualifying_labels_repr=repr(qualifying_labels),
    )
    completion = rlm.completion(chunks[2], root_prompt=root_prompt_4)
    elapsed = time.perf_counter() - t0
    total_wall_clock += elapsed

    incr = env.locals.get("_incremental") if env and hasattr(env, "locals") else None
    pairs_t4 = list(incr.pair_tracker.get_pairs()) if incr else []
    stats_t4 = incr.get_stats() if incr else {}
    input_tokens, output_tokens = _extract_tokens(completion.usage_summary)
    total_input_tokens += input_tokens
    total_output_tokens += output_tokens

    f1_t4_updated = compute_f1(pairs_t4, gold_pairs_post_edit)

    print(f"  Pairs: {len(pairs_t4)}, Entities: {stats_t4.get('total_entities', 0)}")
    print(f"  F1 (vs UPDATED gold): {f1_t4_updated['f1']:.4f}")
    print(f"  Chunks processed: {stats_t4.get('chunks_processed', 0)}")
    print(f"  Tokens: {input_tokens} in, {output_tokens} out, {elapsed:.1f}s")

    results_by_turn.append({
        "turn": 4, "type": "chunk", "chunk_idx": 2,
        "pairs": len(pairs_t4), "f1_updated": f1_t4_updated["f1"],
        "precision_updated": f1_t4_updated["precision"],
        "recall_updated": f1_t4_updated["recall"],
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "elapsed_sec": round(elapsed, 2),
        "stats": stats_t4,
    })

    rlm.close()

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print(f"DYNAMIC CONTEXT EXPERIMENT SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Task {task_idx} | {num_edits} entity edits ({num_downgrade} down, {num_upgrade} up)")
    print(f"  Pre-edit pairs: {pre_edit_pairs}")
    print(f"  Post-edit pairs: {post_edit_pairs} (delta: {post_edit_pairs - pre_edit_pairs})")
    print(f"  Retractions from edit: {retractions_t3}")
    print(f"  Final pairs (after T4): {len(pairs_t4)}")
    print(f"  Final F1 (vs updated gold): {f1_t4_updated['f1']:.4f}")
    print(f"  Final P/R: P={f1_t4_updated['precision']:.4f} R={f1_t4_updated['recall']:.4f}")
    print(f"  Total tokens: {total_input_tokens} in + {total_output_tokens} out")
    print(f"  Total wall-clock: {total_wall_clock:.1f}s")
    print(f"  Total cost est: ${(total_input_tokens * 0.15 + total_output_tokens * 0.6) / 1_000_000:.4f}")

    edit_worked = retractions_t3 > 0  # retractions fired
    pairs_changed = pre_edit_pairs != post_edit_pairs  # pairs actually changed
    post_edit_correct = f1_t3_updated["f1"] > 0  # some correct pairs after edit
    continuation_works = len(pairs_t4) >= post_edit_pairs  # T4 continued correctly

    print(f"\n  VALIDATION CHECKS:")
    print(f"    ✓ Retractions fired: {edit_worked} ({retractions_t3} retractions)")
    print(f"    ✓ Pairs changed after edit: {pairs_changed}")
    print(f"    ✓ Post-edit pairs correct (F1>0): {post_edit_correct}")
    print(f"    ✓ Post-edit continuation (T4 works): {continuation_works}")

    success = edit_worked and post_edit_correct and continuation_works

    return {
        "experiment": "dynamic_context_proof_of_concept",
        "task_idx": task_idx,
        "model": model,
        "num_chunks": num_chunks,
        "max_chunk_chars": max_chunk_chars,
        "num_edits": len(edits),
        "num_downgrade": num_downgrade,
        "num_upgrade": num_upgrade,
        "gold_pairs_original": len(gold_pairs_original),
        "gold_pairs_post_edit": len(gold_pairs_post_edit),
        "gold_pair_delta": len(gold_pairs_post_edit) - len(gold_pairs_original),
        "pre_edit_pairs": pre_edit_pairs,
        "post_edit_pairs": post_edit_pairs,
        "final_pairs": len(pairs_t4),
        "retractions_from_edit": retractions_t3,
        "final_f1_updated": f1_t4_updated["f1"],
        "final_precision_updated": f1_t4_updated["precision"],
        "final_recall_updated": f1_t4_updated["recall"],
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_wall_clock_sec": round(total_wall_clock, 2),
        "success": success,
        "validation": {
            "retractions_fired": edit_worked,
            "pairs_changed": pairs_changed,
            "post_edit_correct": post_edit_correct,
            "continuation_works": continuation_works,
        },
        "results_by_turn": results_by_turn,
    }


# ---------------------------------------------------------------------------
# Offline simulation: validate the edit logic without API calls
# ---------------------------------------------------------------------------

def run_dynamic_context_simulation(
    labeled_context: str,
    task_idx: int = 1,
    num_chunks: int = 4,
    max_chunk_chars: int = 5000,
    num_edits: int = 5,
) -> dict:
    """
    Simulate the dynamic context experiment offline (no API calls).

    Uses IncrementalState directly to:
    1. Process chunks 0 and 1
    2. Apply edits to chunk 0 entities
    3. Process chunk 2
    4. Validate retraction mechanism

    This confirms the library-level correctness before spending API budget.
    """
    from rlm.core.incremental import IncrementalState

    qualifying_labels = TASK_QUALIFYING_LABELS[task_idx]

    print(f"\n{'=' * 70}")
    print(f"DYNAMIC CONTEXT SIMULATION (offline, no API calls)")
    print(f"  Task {task_idx} | k={num_chunks}, {max_chunk_chars} chars/chunk, {num_edits} edits")
    print(f"{'=' * 70}")

    chunks = _make_sequential_chunks(labeled_context, num_chunks, max_chunk_chars)
    total_context_chars = num_chunks * max_chunk_chars

    # Parse entities from each chunk
    chunk_entities = []
    for i, chunk in enumerate(chunks):
        entities = find_entities_in_chunk(chunk, qualifying_labels)
        chunk_entities.append(entities)
        q_count = sum(1 for e in entities.values() if e["qualifying"])
        print(f"  Chunk {i}: {len(entities)} entities ({q_count} qualifying)")

    # Select entities to edit
    edits = select_entities_to_edit(chunk_entities[0], num_edits, qualifying_labels)
    num_downgrade = sum(1 for e in edits.values() if e.get("edit_type") == "downgrade")
    num_upgrade = sum(1 for e in edits.values() if e.get("edit_type") == "upgrade")
    print(f"\nEdits: {len(edits)} total ({num_downgrade} downgrade, {num_upgrade} upgrade)")

    # Compute gold pairs
    context_window = labeled_context[:total_context_chars]
    all_entities_original = find_entities_in_chunk(context_window, qualifying_labels)
    gold_original = set()
    q_ids_orig = sorted(uid for uid, e in all_entities_original.items() if e["qualifying"])
    for i, id1 in enumerate(q_ids_orig):
        for id2 in q_ids_orig[i + 1:]:
            gold_original.add((min(id1, id2), max(id1, id2)))

    gold_post_edit = compute_gold_pairs_with_edits(context_window, qualifying_labels, edits)
    print(f"Gold pairs: {len(gold_original)} original, {len(gold_post_edit)} post-edit")

    # Define pair checker
    def check_pair(attrs1, attrs2):
        q1 = attrs1.get("qualifying", False) if isinstance(attrs1, dict) else False
        q2 = attrs2.get("qualifying", False) if isinstance(attrs2, dict) else False
        return q1 and q2

    # --- Process chunk 0 ---
    incr = IncrementalState()
    stats0 = incr.process_chunk(0, chunk_entities[0], pair_checker=check_pair, monotone_attrs={"qualifying"})
    pairs_0 = incr.pair_tracker.get_pairs()
    print(f"\nAfter chunk 0: {len(pairs_0)} pairs, {stats0['new_entities']} entities")

    # --- Process chunk 1 ---
    stats1 = incr.process_chunk(1, chunk_entities[1], pair_checker=check_pair, monotone_attrs={"qualifying"})
    pairs_1 = incr.pair_tracker.get_pairs()
    pre_edit_pairs = len(pairs_1)
    print(f"After chunk 1: {len(pairs_1)} pairs, {stats1['new_entities']} new entities")

    # --- Apply edits ---
    print(f"\n--- Applying {len(edits)} entity edits ---")
    total_retracted = 0
    for uid, edit_attrs in edits.items():
        old_attrs = incr.entity_cache.get(uid)
        old_q = old_attrs.get("qualifying", False) if old_attrs else False

        # Update entity
        edit_attrs_clean = {"labels": edit_attrs["labels"], "qualifying": edit_attrs["qualifying"]}
        incr.entity_cache.add(uid, edit_attrs_clean, chunk_index=99)

        # Retract pairs
        retracted = incr.pair_tracker.retract_entity(uid)
        total_retracted += len(retracted)

        # Re-evaluate retracted pairs
        updated_attrs = incr.entity_cache.get(uid)
        for p in retracted:
            partner_id = p[1] if p[0] == uid else p[0]
            partner_attrs = incr.entity_cache.get(partner_id)
            if partner_attrs and check_pair(updated_attrs, partner_attrs):
                incr.pair_tracker.add_pair(uid, partner_id)

        # Check for new pairs
        for other_id in incr.entity_cache.get_ids():
            if other_id == uid:
                continue
            other_attrs = incr.entity_cache.get(other_id)
            if other_attrs and check_pair(updated_attrs, other_attrs):
                incr.pair_tracker.add_pair(uid, other_id)

        new_q = edit_attrs["qualifying"]
        print(f"  Entity {uid}: qualifying {old_q} -> {new_q} ({edit_attrs['edit_type']}), retracted {len(retracted)}")

    pairs_post_edit = incr.pair_tracker.get_pairs()
    post_edit_pairs = len(pairs_post_edit)
    f1_post_edit = compute_f1(list(pairs_post_edit), gold_post_edit)

    print(f"\nAfter edit: {post_edit_pairs} pairs (was {pre_edit_pairs}, delta: {post_edit_pairs - pre_edit_pairs})")
    print(f"  Retractions: {total_retracted}")
    print(f"  F1 vs updated gold: {f1_post_edit['f1']:.4f} P={f1_post_edit['precision']:.4f} R={f1_post_edit['recall']:.4f}")

    # --- Process chunk 2 (clear idempotency for chunk 99 if needed) ---
    stats2 = incr.process_chunk(2, chunk_entities[2], pair_checker=check_pair, monotone_attrs={"qualifying"})
    pairs_final = incr.pair_tracker.get_pairs()
    f1_final = compute_f1(list(pairs_final), gold_post_edit)

    print(f"\nAfter chunk 2: {len(pairs_final)} pairs, {stats2['new_entities']} new entities")
    print(f"  F1 vs updated gold: {f1_final['f1']:.4f} P={f1_final['precision']:.4f} R={f1_final['recall']:.4f}")

    # Validation
    success = total_retracted > 0 and f1_final["f1"] > 0 and len(pairs_final) > 0
    print(f"\n  SIMULATION VALIDATION:")
    print(f"    ✓ Retractions fired: {total_retracted > 0} ({total_retracted} total)")
    print(f"    ✓ Pairs changed: {pre_edit_pairs != post_edit_pairs}")
    print(f"    ✓ Post-edit F1 > 0: {f1_post_edit['f1'] > 0}")
    print(f"    ✓ Post-edit continuation: {len(pairs_final) >= post_edit_pairs}")
    print(f"    SUCCESS: {success}")

    return {
        "experiment": "dynamic_context_simulation",
        "task_idx": task_idx,
        "num_chunks": num_chunks,
        "num_edits": len(edits),
        "num_downgrade": num_downgrade,
        "num_upgrade": num_upgrade,
        "gold_original": len(gold_original),
        "gold_post_edit": len(gold_post_edit),
        "pre_edit_pairs": pre_edit_pairs,
        "post_edit_pairs": post_edit_pairs,
        "final_pairs": len(pairs_final),
        "total_retracted": total_retracted,
        "f1_post_edit": f1_post_edit["f1"],
        "f1_final": f1_final["f1"],
        "precision_final": f1_final["precision"],
        "recall_final": f1_final["recall"],
        "success": success,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Dynamic context experiment")
    parser.add_argument("--task", type=int, default=1, help="Task index (1, 3, or 6)")
    parser.add_argument("--num-edits", type=int, default=5, help="Number of entities to edit")
    parser.add_argument("--num-chunks", type=int, default=4, help="Number of data chunks")
    parser.add_argument("--max-chunk-chars", type=int, default=5000, help="Max chars per chunk")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--simulate", action="store_true", help="Run offline simulation only (no API calls)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    _, labeled_context = load_labeled_data()
    total_chars = args.num_chunks * args.max_chunk_chars
    gold_pairs = compute_gold_pairs(labeled_context[:total_chars], args.task)

    if args.simulate:
        result = run_dynamic_context_simulation(
            labeled_context=labeled_context,
            task_idx=args.task,
            num_chunks=args.num_chunks,
            max_chunk_chars=args.max_chunk_chars,
            num_edits=args.num_edits,
        )
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required for live experiment")

        result = run_dynamic_context_experiment(
            labeled_context=labeled_context,
            gold_pairs_original=gold_pairs,
            api_key=api_key,
            task_idx=args.task,
            num_chunks=args.num_chunks,
            max_chunk_chars=args.max_chunk_chars,
            num_edits=args.num_edits,
            model=args.model,
            verbose=args.verbose,
        )

    output_path = args.output or f"results/streaming/dynamic_context_task{args.task}_edits{args.num_edits}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
