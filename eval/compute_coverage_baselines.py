"""
Compute coverage-bounded F1 baselines for the F1 progression experiment.

For conditions B and C (non-incremental baselines), we compute:
- B: coverage-bounded F1 ceiling for first 5K chars (what ANY perfect system could get)
- C: coverage-bounded F1 ceiling for full 25K chars (what ANY perfect system could get)
- B_incremental_k1: same as the incremental RLM's k=1 snapshot (direct comparison)

These are ORACLE baselines — upper bounds for systems with given context budgets.
They isolate the "context coverage" limitation from "model quality" limitations.

No API calls needed: purely data-driven analysis.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.rlm_pipeline_experiment import (
    compute_gold_pairs,
    load_oolong_data,
    split_context_by_users,
)


def compute_coverage_ceiling(plain_context: str, labeled_context: str, max_chars: int, task_idx: int = 1) -> dict:
    """
    Compute the coverage-bounded F1 ceiling for a given context budget.

    This answers: "If a perfect extractor processed the first max_chars of context,
    what's the maximum F1 it could achieve against the full-corpus gold?"

    Method:
    1. Truncate labeled_context to max_chars
    2. Extract all visible users and their labels from the truncated context
    3. Compute pairs satisfying task condition among visible users
    4. Compute F1 vs full-corpus gold (which includes pairs from the full context)

    This is the upper bound for any system with this context budget.
    """
    from eval.utils import _parse_labeled_context, _check_pair_condition

    # Truncate to budget
    truncated_labeled = labeled_context[:max_chars]

    # Parse visible users from truncated labeled context
    visible_users = _parse_labeled_context(truncated_labeled)

    # Compute pairs among visible users using task condition
    visible_pairs = set()
    user_ids = sorted(visible_users.keys())
    for uid1, uid2 in combinations(user_ids, 2):
        if _check_pair_condition(visible_users[uid1], visible_users[uid2], task_idx):
            visible_pairs.add((min(uid1, uid2), max(uid1, uid2)))

    # Full gold pairs (against full corpus)
    full_gold = compute_gold_pairs(labeled_context, task_idx)

    # F1 computation
    tp = len(visible_pairs & full_gold)
    fp = len(visible_pairs - full_gold)
    fn = len(full_gold - visible_pairs)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "context_chars": max_chars,
        "visible_users": len(visible_users),
        "visible_pairs": len(visible_pairs),
        "gold_pairs": len(full_gold),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def main():
    print("Loading OOLONG-Pairs data...")
    plain_context, labeled_context = load_oolong_data()
    gold_pairs = compute_gold_pairs(labeled_context, task_idx=1)
    print(f"Full gold pairs (Task 1): {len(gold_pairs)}")

    # Compute coverage ceilings at each chunk boundary
    num_chunks = 5
    max_chunk_chars = 5000

    print("\n=== Coverage-Bounded F1 Ceilings (Perfect Oracle Extractor) ===")
    print(f"{'Chars':>8} {'Visible Users':>14} {'Visible Pairs':>14} {'F1':>6} {'Precision':>10} {'Recall':>8}")
    print("-" * 65)

    ceilings = []
    for k in range(1, num_chunks + 1):
        max_chars = k * max_chunk_chars
        result = compute_coverage_ceiling(plain_context, labeled_context, max_chars, task_idx=1)
        ceilings.append({"k": k, **result})
        print(f"{max_chars:>8} {result['visible_users']:>14} {result['visible_pairs']:>14} "
              f"{result['f1']:>6} {result['precision']:>10} {result['recall']:>8}")

    # Single-turn oracle on full context
    full_result = compute_coverage_ceiling(plain_context, labeled_context,
                                           len(labeled_context), task_idx=1)
    print(f"{'Full':>8} {full_result['visible_users']:>14} {full_result['visible_pairs']:>14} "
          f"{full_result['f1']:>6} {full_result['precision']:>10} {full_result['recall']:>8}")

    # Load the incremental F1 progression results for comparison
    incr_path = Path("results/streaming/f1_progression_results.json")
    if incr_path.exists():
        with open(incr_path) as f:
            incr_results = json.load(f)

        incr_progression = incr_results.get("condition_a", {}).get("f1_progression", [])
        if incr_progression:
            print("\n=== Comparison: Incremental RLM vs Coverage Ceiling ===")
            print(f"{'k':>3} {'Chars':>8} {'Incr F1':>8} {'Ceiling F1':>11} {'Gap':>8} {'Coverage%':>10}")
            print("-" * 55)
            for i, (prog, ceil) in enumerate(zip(incr_progression, ceilings)):
                chars = (i + 1) * max_chunk_chars
                incr_f1 = prog.get("f1") or 0.0
                ceil_f1 = ceil["f1"]
                gap = ceil_f1 - incr_f1
                coverage_pct = 100 * incr_f1 / ceil_f1 if ceil_f1 > 0 else 0.0
                print(f"{i+1:>3} {chars:>8} {incr_f1:>8.4f} {ceil_f1:>11.4f} {gap:>+8.4f} {coverage_pct:>9.1f}%")

    # Save results
    out_data = {
        "task_idx": 1,
        "gold_pairs_full": len(gold_pairs),
        "coverage_ceilings_by_chunk": ceilings,
        "coverage_ceiling_full_context": full_result,
    }
    out_path = Path("results/streaming/coverage_baselines.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved to {out_path}")

    return out_data


if __name__ == "__main__":
    main()
