"""
Iteration 18 Experiments:

1. No-Retraction Counterfactual (zero API cost) — quantifies the VALUE of retraction
2. Full-Corpus Experiment Runner (A+D on 96K chars) — removes the F1=0.32 presentation problem

## No-Retraction Counterfactual

Simulates what happens if entity edits are applied WITHOUT calling retract_entity().
Quantifies: how many invalid pairs persist, what precision drops to.
This is the missing "why retraction matters" evidence for Table 8.

## Full-Corpus Experiment

Runs Condition A (incremental) and Condition D (full recompute) on the FULL 96K-char
labeled corpus instead of the 25K subset. Expected: F1 >> 0.34 while maintaining
77-86% token savings.

Usage:
    # No-retraction counterfactual (zero cost, instant)
    python eval/full_corpus_and_counterfactual.py --counterfactual

    # Full-corpus A+D (needs API key, ~$1)
    python eval/full_corpus_and_counterfactual.py --full-corpus

    # Both
    python eval/full_corpus_and_counterfactual.py --counterfactual --full-corpus
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.rlm_pipeline_experiment import compute_f1
from eval.dynamic_context_experiment import (
    find_entities_in_chunk,
    select_entities_to_edit,
    compute_gold_pairs_with_edits,
)
from eval.label_aware_experiment import (
    TASK_QUALIFYING_LABELS,
    load_labeled_data,
)
from eval.label_aware_v2_experiment import _make_sequential_chunks


# ===========================================================================
# 1. No-Retraction Counterfactual
# ===========================================================================


def run_no_retraction_counterfactual(
    labeled_context: str,
    task_idx: int = 1,
    num_chunks: int = 4,
    max_chunk_chars: int = 5000,
    num_edits_list: list[int] | None = None,
) -> list[dict]:
    """Run the no-retraction counterfactual for multiple edit counts.

    Compares:
    - WITH retraction: apply edits, retract affected pairs, re-evaluate (correct)
    - WITHOUT retraction: apply edits to entity cache, but DON'T retract pairs

    This quantifies how many invalid pairs persist without retraction and the
    resulting precision drop. Directly answers: "why does retraction matter?"
    """
    from rlm.core.incremental import IncrementalState

    if num_edits_list is None:
        num_edits_list = [5, 10]

    qualifying_labels = TASK_QUALIFYING_LABELS[task_idx]
    chunks = _make_sequential_chunks(labeled_context, num_chunks, max_chunk_chars)
    total_context_chars = num_chunks * max_chunk_chars
    context_window = labeled_context[:total_context_chars]

    # Parse entities per chunk
    chunk_entities = []
    for i, chunk in enumerate(chunks):
        entities = find_entities_in_chunk(chunk, qualifying_labels)
        chunk_entities.append(entities)

    def check_pair(attrs1, attrs2):
        q1 = attrs1.get("qualifying", False) if isinstance(attrs1, dict) else False
        q2 = attrs2.get("qualifying", False) if isinstance(attrs2, dict) else False
        return q1 and q2

    results = []

    for num_edits in num_edits_list:
        print(f"\n{'=' * 70}")
        print(f"NO-RETRACTION COUNTERFACTUAL: {num_edits} edits")
        print(f"  Task {task_idx} | k={num_chunks}, {max_chunk_chars} chars/chunk")
        print(f"{'=' * 70}")

        # Select edits (deterministic due to sorted iteration)
        edits = select_entities_to_edit(chunk_entities[0], num_edits, qualifying_labels)
        num_downgrade = sum(1 for e in edits.values() if e.get("edit_type") == "downgrade")
        num_upgrade = sum(1 for e in edits.values() if e.get("edit_type") == "upgrade")
        print(f"  Edits: {len(edits)} ({num_downgrade} downgrade, {num_upgrade} upgrade)")

        # Compute gold pairs
        all_entities = find_entities_in_chunk(context_window, qualifying_labels)
        gold_original = set()
        q_ids = sorted(uid for uid, e in all_entities.items() if e["qualifying"])
        for i, id1 in enumerate(q_ids):
            for id2 in q_ids[i + 1:]:
                gold_original.add((min(id1, id2), max(id1, id2)))
        gold_post_edit = compute_gold_pairs_with_edits(context_window, qualifying_labels, edits)

        # === WITH RETRACTION (using apply_edits) ===
        incr_with = IncrementalState()
        for ci in range(num_chunks):
            incr_with.process_chunk(ci, chunk_entities[ci].copy(), pair_checker=check_pair,
                                     monotone_attrs={"qualifying"})

        pairs_before = len(incr_with.pair_tracker)
        # Use the new apply_edits API
        edit_attrs = {uid: {"labels": e["labels"], "qualifying": e["qualifying"]}
                      for uid, e in edits.items()}
        edit_stats = incr_with.apply_edits(edit_attrs, pair_checker=check_pair, edit_chunk_index=99)
        pairs_with_retraction = incr_with.pair_tracker.get_pairs()
        f1_with = compute_f1(list(pairs_with_retraction), gold_post_edit)

        # Verify all pairs are valid (precision = 1.0)
        invalid_with_retraction = 0
        for p in pairs_with_retraction:
            a1 = incr_with.entity_cache.get(p[0])
            a2 = incr_with.entity_cache.get(p[1])
            if not check_pair(a1, a2):
                invalid_with_retraction += 1

        print(f"\n  WITH RETRACTION:")
        print(f"    Pairs: {pairs_before} → {len(pairs_with_retraction)} (delta: {len(pairs_with_retraction) - pairs_before})")
        print(f"    Retractions: {edit_stats['total_retracted']}")
        print(f"    Invalid pairs remaining: {invalid_with_retraction}")
        print(f"    F1 vs updated gold: {f1_with['f1']:.4f} P={f1_with['precision']:.4f}")

        # === WITHOUT RETRACTION ===
        incr_without = IncrementalState()
        for ci in range(num_chunks):
            # Deep copy entities to avoid mutation
            chunk_ents = {uid: dict(attrs) for uid, attrs in chunk_entities[ci].items()}
            incr_without.process_chunk(ci, chunk_ents, pair_checker=check_pair,
                                        monotone_attrs={"qualifying"})

        pairs_before_no_retract = len(incr_without.pair_tracker)

        # Apply edits to entity cache but DON'T retract pairs
        for uid, edit_data in edits.items():
            edit_attrs_clean = {"labels": edit_data["labels"], "qualifying": edit_data["qualifying"]}
            incr_without.entity_cache.add(uid, edit_attrs_clean, chunk_index=99)
            # NO retract_entity() call — this is the counterfactual

        pairs_without_retraction = incr_without.pair_tracker.get_pairs()

        # Count invalid pairs (pairs where at least one entity is no longer qualifying)
        invalid_without_retraction = 0
        invalid_pair_details = []
        for p in pairs_without_retraction:
            a1 = incr_without.entity_cache.get(p[0])
            a2 = incr_without.entity_cache.get(p[1])
            if not check_pair(a1, a2):
                invalid_without_retraction += 1
                invalid_pair_details.append(p)

        # Compute precision
        total_predicted = len(pairs_without_retraction)
        true_positives_without = total_predicted - invalid_without_retraction
        precision_without = true_positives_without / total_predicted if total_predicted > 0 else 1.0
        f1_without = compute_f1(list(pairs_without_retraction), gold_post_edit)

        # Also check: are there missing pairs (entities that got upgraded but
        # their new pairs weren't created)?
        # Without retraction, no new pairs are created either
        all_ids = incr_without.entity_cache.get_ids()
        missing_new_pairs = 0
        for uid, edit_data in edits.items():
            if edit_data.get("edit_type") == "upgrade":
                # This entity is now qualifying but no pairs were created
                updated_attrs = incr_without.entity_cache.get(uid)
                for other_id in all_ids:
                    if other_id == uid:
                        continue
                    other_attrs = incr_without.entity_cache.get(other_id)
                    canonical = (min(uid, other_id), max(uid, other_id))
                    if other_attrs and check_pair(updated_attrs, other_attrs):
                        if canonical not in incr_without.pair_tracker._pairs:
                            missing_new_pairs += 1

        print(f"\n  WITHOUT RETRACTION:")
        print(f"    Pairs: {pairs_before_no_retract} → {len(pairs_without_retraction)} (UNCHANGED)")
        print(f"    Invalid pairs remaining: {invalid_without_retraction}")
        print(f"    Missing new pairs (from upgrades): {missing_new_pairs}")
        print(f"    Precision: {precision_without:.4f} (vs 1.0 with retraction)")
        print(f"    F1 vs updated gold: {f1_without['f1']:.4f}")

        print(f"\n  COMPARISON:")
        print(f"    | Metric                  | With Retraction | Without Retraction |")
        print(f"    |-------------------------|-----------------|-------------------|")
        print(f"    | Invalid pairs remaining | {invalid_with_retraction:>15} | {invalid_without_retraction:>17} |")
        print(f"    | Missing new pairs       | {'0':>15} | {missing_new_pairs:>17} |")
        print(f"    | Precision               | {f1_with['precision']:>15.4f} | {precision_without:>17.4f} |")
        print(f"    | F1 vs updated gold      | {f1_with['f1']:>15.4f} | {f1_without['f1']:>17.4f} |")
        print(f"    | Correctness             | {'✓':>15} | {'✗':>17} |")

        results.append({
            "num_edits": num_edits,
            "num_downgrade": num_downgrade,
            "num_upgrade": num_upgrade,
            "gold_original": len(gold_original),
            "gold_post_edit": len(gold_post_edit),
            "with_retraction": {
                "pairs_before": pairs_before,
                "pairs_after": len(pairs_with_retraction),
                "retractions": edit_stats["total_retracted"],
                "invalid_pairs": invalid_with_retraction,
                "precision": f1_with["precision"],
                "recall": f1_with["recall"],
                "f1": f1_with["f1"],
            },
            "without_retraction": {
                "pairs_before": pairs_before_no_retract,
                "pairs_after": len(pairs_without_retraction),
                "invalid_pairs": invalid_without_retraction,
                "missing_new_pairs": missing_new_pairs,
                "precision": precision_without,
                "recall": f1_without["recall"],
                "f1": f1_without["f1"],
            },
        })

    return results


# ===========================================================================
# 2. Full-Corpus Experiment Runner
# ===========================================================================


def run_full_corpus_simulation(
    labeled_context: str,
    task_idx: int = 1,
    num_chunks: int = 5,
) -> dict:
    """Simulate full-corpus A vs D (no API calls) to validate parameters.

    Uses the FULL labeled context (~96K chars), chunked into num_chunks pieces.
    Runs both incremental (A) and full-recompute (D) through IncrementalState.
    """
    from rlm.core.incremental import IncrementalState

    qualifying_labels = TASK_QUALIFYING_LABELS[task_idx]
    total_chars = len(labeled_context)
    max_chunk_chars = total_chars // num_chunks

    print(f"\n{'=' * 70}")
    print(f"FULL-CORPUS SIMULATION: A vs D (no API calls)")
    print(f"  Task {task_idx} | k={num_chunks}, {max_chunk_chars} chars/chunk")
    print(f"  Total context: {total_chars} chars")
    print(f"{'=' * 70}")

    # Create chunks from full corpus
    chunks = [labeled_context[i * max_chunk_chars:(i + 1) * max_chunk_chars]
              for i in range(num_chunks)]
    # Include any remaining chars in last chunk
    if num_chunks * max_chunk_chars < total_chars:
        chunks[-1] += labeled_context[num_chunks * max_chunk_chars:]

    print(f"  Chunk sizes: {[len(c) for c in chunks]} chars")

    # Parse entities per chunk
    chunk_entities = []
    for i, chunk in enumerate(chunks):
        entities = find_entities_in_chunk(chunk, qualifying_labels)
        chunk_entities.append(entities)
        q_count = sum(1 for e in entities.values() if e["qualifying"])
        print(f"  Chunk {i}: {len(entities)} entities ({q_count} qualifying)")

    def check_pair(attrs1, attrs2):
        q1 = attrs1.get("qualifying", False) if isinstance(attrs1, dict) else False
        q2 = attrs2.get("qualifying", False) if isinstance(attrs2, dict) else False
        return q1 and q2

    # Compute full-corpus gold pairs
    all_entities = find_entities_in_chunk(labeled_context, qualifying_labels)
    q_ids = sorted(uid for uid, e in all_entities.items() if e["qualifying"])
    gold_pairs = set()
    for i, id1 in enumerate(q_ids):
        for id2 in q_ids[i + 1:]:
            gold_pairs.add((min(id1, id2), max(id1, id2)))
    print(f"\n  Total entities: {len(all_entities)} ({len(q_ids)} qualifying)")
    print(f"  Gold pairs: {len(gold_pairs)}")

    # === Condition A: Incremental ===
    incr_a = IncrementalState()
    a_pair_checks = 0
    a_progression = []
    for ci in range(num_chunks):
        chunk_ents = {uid: dict(attrs) for uid, attrs in chunk_entities[ci].items()}
        stats = incr_a.process_chunk(ci, chunk_ents, pair_checker=check_pair,
                                      monotone_attrs={"qualifying"})
        a_pair_checks += stats["pair_checks"]
        pairs = list(incr_a.pair_tracker.get_pairs())
        f1 = compute_f1(pairs, gold_pairs)
        a_progression.append({
            "chunk": ci, "pairs": len(pairs),
            "f1": f1["f1"], "precision": f1["precision"], "recall": f1["recall"],
            "pair_checks": stats["pair_checks"],
            "new_entities": stats["new_entities"],
            "updated_entities": stats["updated_entities"],
        })
        print(f"  A chunk {ci}: {len(pairs)} pairs, F1={f1['f1']:.4f}, P={f1['precision']:.4f}, checks={stats['pair_checks']}")

    final_a = a_progression[-1]
    a_stats = incr_a.get_stats()

    # === Condition D: Full recompute each turn ===
    d_pair_checks = 0
    d_progression = []
    for turn in range(num_chunks):
        incr_d = IncrementalState()
        for ci in range(turn + 1):
            chunk_ents = {uid: dict(attrs) for uid, attrs in chunk_entities[ci].items()}
            stats = incr_d.process_chunk(ci, chunk_ents, pair_checker=check_pair,
                                          monotone_attrs={"qualifying"})
        d_pair_checks += incr_d.get_stats()["total_pair_checks"]
        pairs = list(incr_d.pair_tracker.get_pairs())
        f1 = compute_f1(pairs, gold_pairs)
        d_progression.append({
            "turn": turn, "pairs": len(pairs),
            "f1": f1["f1"], "precision": f1["precision"], "recall": f1["recall"],
            "pair_checks": incr_d.get_stats()["total_pair_checks"],
        })
        print(f"  D turn {turn}: {len(pairs)} pairs, F1={f1['f1']:.4f}, checks={incr_d.get_stats()['total_pair_checks']}")

    final_d = d_progression[-1]

    # Comparison
    savings_pair_checks = 1 - a_pair_checks / d_pair_checks if d_pair_checks > 0 else 0
    structural = 1 - 2 / (num_chunks + 1)

    print(f"\n  {'=' * 60}")
    print(f"  FULL-CORPUS SIMULATION SUMMARY")
    print(f"  {'=' * 60}")
    print(f"  | Metric           | A (Incremental) | D (Full Recompute) |")
    print(f"  |------------------|-----------------|-------------------|")
    print(f"  | Final F1         | {final_a['f1']:>15.4f} | {final_d['f1']:>17.4f} |")
    print(f"  | Final Precision  | {final_a['precision']:>15.4f} | {final_d['precision']:>17.4f} |")
    print(f"  | Final Pairs      | {final_a['pairs']:>15} | {final_d['pairs']:>17} |")
    print(f"  | Total Pair Checks| {a_pair_checks:>15} | {d_pair_checks:>17} |")
    print(f"  | Check Savings    | {savings_pair_checks:>14.1%}  | {'baseline':>17} |")
    print(f"  | Structural pred. | {structural:>14.1%}  |                   |")
    print(f"  | F1 match         | {'✓' if abs(final_a['f1'] - final_d['f1']) < 0.01 else '✗':>15} |                   |")

    return {
        "experiment": "full_corpus_simulation",
        "task_idx": task_idx,
        "num_chunks": num_chunks,
        "total_chars": total_chars,
        "max_chunk_chars": max_chunk_chars,
        "gold_pairs": len(gold_pairs),
        "total_entities": len(all_entities),
        "qualifying_entities": len(q_ids),
        "condition_a": {
            "final_f1": final_a["f1"],
            "final_precision": final_a["precision"],
            "final_recall": final_a["recall"],
            "final_pairs": final_a["pairs"],
            "total_pair_checks": a_pair_checks,
            "total_retractions": a_stats["total_retractions"],
            "progression": a_progression,
        },
        "condition_d": {
            "final_f1": final_d["f1"],
            "final_precision": final_d["precision"],
            "final_recall": final_d["recall"],
            "final_pairs": final_d["pairs"],
            "total_pair_checks": d_pair_checks,
            "progression": d_progression,
        },
        "pair_check_savings": savings_pair_checks,
        "structural_prediction": structural,
        "f1_match": abs(final_a["f1"] - final_d["f1"]) < 0.01,
    }


# ===========================================================================
# CLI
# ===========================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Full-corpus + counterfactual experiments")
    parser.add_argument("--counterfactual", action="store_true", help="Run no-retraction counterfactual")
    parser.add_argument("--full-corpus", action="store_true", help="Run full-corpus A vs D simulation")
    parser.add_argument("--full-corpus-live", action="store_true",
                        help="Run full-corpus A+D with live API (needs OPENAI_API_KEY)")
    parser.add_argument("--task", type=int, default=1, help="Task index")
    parser.add_argument("--k", type=int, default=5, help="Number of chunks")
    parser.add_argument("--edits", type=str, default="5,10", help="Comma-separated edit counts")
    parser.add_argument("--output-dir", default="results/streaming")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, labeled_context = load_labeled_data()

    if args.counterfactual:
        edit_counts = [int(x) for x in args.edits.split(",")]
        results = run_no_retraction_counterfactual(
            labeled_context=labeled_context,
            task_idx=args.task,
            num_edits_list=edit_counts,
        )
        out_path = output_dir / f"no_retraction_counterfactual_task{args.task}.json"
        out_path.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nSaved to {out_path}")

    if args.full_corpus:
        result = run_full_corpus_simulation(
            labeled_context=labeled_context,
            task_idx=args.task,
            num_chunks=args.k,
        )
        out_path = output_dir / f"full_corpus_simulation_task{args.task}_k{args.k}.json"
        out_path.write_text(json.dumps(result, indent=2, default=str))
        print(f"\nSaved to {out_path}")

    if args.full_corpus_live:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            env_path = Path(__file__).parent.parent / ".env"
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    if line.startswith("OPENAI_API_KEY="):
                        api_key = line.split("=", 1)[1].strip()
                        break
        if not api_key:
            print("ERROR: OPENAI_API_KEY required for --full-corpus-live")
            sys.exit(1)

        from eval.label_aware_v4_experiment import (
            run_condition_a_v4,
            run_condition_d_full_recompute,
        )
        from eval.rlm_pipeline_experiment import compute_gold_pairs

        # Full corpus params
        total_chars = len(labeled_context)
        max_chunk_chars = total_chars // args.k
        gold_pairs = compute_gold_pairs(labeled_context, task_idx=args.task)

        print(f"\n{'=' * 70}")
        print(f"FULL-CORPUS LIVE EXPERIMENT")
        print(f"  Task {args.task} | k={args.k}, {max_chunk_chars} chars/chunk, {total_chars} total")
        print(f"  Gold pairs: {len(gold_pairs)}")
        print(f"{'=' * 70}")

        # Run Condition A (incremental)
        result_a = run_condition_a_v4(
            labeled_context=labeled_context,
            gold_pairs=gold_pairs,
            api_key=api_key,
            task_idx=args.task,
            num_chunks=args.k,
            max_chunk_chars=max_chunk_chars,
            run_id=1,
        )
        out_a = output_dir / f"full_corpus_task{args.task}_k{args.k}_condition_a.json"
        out_a.write_text(json.dumps(result_a, indent=2, default=str))
        print(f"Condition A saved to {out_a}")

        # Run Condition D (full recompute)
        result_d = run_condition_d_full_recompute(
            labeled_context=labeled_context,
            gold_pairs=gold_pairs,
            api_key=api_key,
            task_idx=args.task,
            num_chunks=args.k,
            max_chunk_chars=max_chunk_chars,
        )
        out_d = output_dir / f"full_corpus_task{args.task}_k{args.k}_condition_d.json"
        out_d.write_text(json.dumps(result_d, indent=2, default=str))
        print(f"Condition D saved to {out_d}")

        # Print comparison
        fa = result_a.get("final_f1", 0)
        fd = result_d.get("final_f1", 0)
        ta = result_a.get("total_input_tokens", 0)
        td = result_d.get("total_input_tokens", 0)
        savings = 1 - ta / td if td > 0 else 0

        print(f"\n  FULL-CORPUS LIVE COMPARISON:")
        print(f"  | Metric        | A (Incremental) | D (Full Recompute) |")
        print(f"  |---------------|-----------------|-------------------|")
        print(f"  | Final F1      | {fa:>15.4f} | {fd:>17.4f} |")
        print(f"  | Input Tokens  | {ta:>15,} | {td:>17,} |")
        print(f"  | Token Savings | {savings:>14.1%}  | {'baseline':>17} |")

    if not (args.counterfactual or args.full_corpus or args.full_corpus_live):
        print("No experiment selected. Use --counterfactual, --full-corpus, or --full-corpus-live")


if __name__ == "__main__":
    main()
