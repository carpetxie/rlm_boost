"""
Incremental Advantage Analysis.

Measures the theoretical and empirical token savings of incremental computation
in the streaming context setting. This demonstrates the core thesis: persistent
RLM with cached state processes O(k*n) new pairs per chunk (k new users × n
existing users) vs O((n+k)²) for full re-computation.

This script runs without API keys — it simulates the computation patterns.
"""

import json
import re
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.utils import _parse_labeled_context, compute_gold_pairs


def load_data(context_len: int = 32768):
    from datasets import load_dataset

    ds = load_dataset("oolongbench/oolong-synth", split="validation")
    corpus = [x for x in ds if x["dataset"] == "trec_coarse" and x["context_len"] == context_len][0]
    return corpus["context_window_text"], corpus["context_window_text_with_labels"]


def analyze_incremental_vs_full(labeled_context: str, num_chunks: int, task_indices: list[int]):
    """Analyze the computation cost of incremental vs full recomputation.

    For each chunk arrival:
    - Full recompute: parse ALL users, check ALL C(n,2) pairs
    - Incremental: parse only NEW users, check pairs between (new, existing) and (new, new)

    We measure:
    - Number of user-parses required
    - Number of pair-checks required
    - Pairs discovered per chunk
    """
    # Split labeled context by user boundaries
    user_pattern = re.compile(r"(?=Date:.*?\|\| User:)")
    positions = [m.start() for m in user_pattern.finditer(labeled_context)]

    users_per_chunk = max(1, len(positions) // num_chunks)

    results = {}
    for task_idx in task_indices:
        task_result = {
            "task_id": task_idx,
            "chunks": [],
            "total_full_pair_checks": 0,
            "total_incremental_pair_checks": 0,
            "total_full_user_parses": 0,
            "total_incremental_user_parses": 0,
        }

        all_seen_users = {}  # uid -> instances
        all_discovered_pairs = set()

        for chunk_i in range(num_chunks):
            start_pos = (
                positions[chunk_i * users_per_chunk]
                if chunk_i * users_per_chunk < len(positions)
                else len(labeled_context)
            )
            if chunk_i < num_chunks - 1:
                end_idx = min((chunk_i + 1) * users_per_chunk, len(positions))
                end_pos = positions[end_idx] if end_idx < len(positions) else len(labeled_context)
            else:
                end_pos = len(labeled_context)

            chunk_text = labeled_context[start_pos:end_pos]
            chunk_users = _parse_labeled_context(chunk_text)
            new_user_ids = set(chunk_users.keys()) - set(all_seen_users.keys())

            # --- Full recompute cost ---
            # Must parse ALL users up to this point
            cumulative_text = labeled_context[:end_pos]
            cumulative_users = _parse_labeled_context(cumulative_text)
            full_user_parses = len(cumulative_users)
            full_pair_checks = len(list(combinations(sorted(cumulative_users.keys()), 2)))

            # --- Incremental cost ---
            # Only parse new users in this chunk
            incremental_user_parses = len(chunk_users)
            # Check pairs: (new_user, existing_user) + (new_user, new_user)
            existing_user_ids = set(all_seen_users.keys())
            incremental_pair_checks = (
                len(new_user_ids) * len(existing_user_ids)  # new × existing
                + len(list(combinations(sorted(new_user_ids), 2)))  # new × new
            )

            # Compute actual gold pairs
            gold = compute_gold_pairs(cumulative_text, task_idx)
            gold_set = set()
            for pair_str in gold.split("\n"):
                if pair_str.strip():
                    m = re.match(r"\((\d+),\s*(\d+)\)", pair_str.strip())
                    if m:
                        gold_set.add((int(m.group(1)), int(m.group(2))))

            new_pairs = gold_set - all_discovered_pairs
            all_discovered_pairs = gold_set.copy()

            # Update seen users
            for uid, instances in chunk_users.items():
                if uid in all_seen_users:
                    all_seen_users[uid].extend(instances)
                else:
                    all_seen_users[uid] = list(instances)

            savings = (
                (1 - incremental_pair_checks / full_pair_checks) * 100
                if full_pair_checks > 0
                else 0
            )

            chunk_info = {
                "chunk_index": chunk_i,
                "new_users": len(new_user_ids),
                "total_users": len(all_seen_users),
                "full_user_parses": full_user_parses,
                "full_pair_checks": full_pair_checks,
                "incremental_user_parses": incremental_user_parses,
                "incremental_pair_checks": incremental_pair_checks,
                "pair_check_savings_pct": savings,
                "total_pairs_discovered": len(all_discovered_pairs),
                "new_pairs_this_chunk": len(new_pairs),
            }
            task_result["chunks"].append(chunk_info)
            task_result["total_full_pair_checks"] += full_pair_checks
            task_result["total_incremental_pair_checks"] += incremental_pair_checks
            task_result["total_full_user_parses"] += full_user_parses
            task_result["total_incremental_user_parses"] += incremental_user_parses

        results[task_idx] = task_result

    return results


def estimate_token_savings(results: dict, num_chunks: int):
    """Estimate token savings based on computation patterns.

    Key insight: in RLM, each "pair check" requires the model to read and
    classify user instances. The number of pair checks directly correlates
    with token usage because:
    1. Full recompute: model reads ALL user data and checks ALL pairs
    2. Incremental: model reads only NEW data and checks NEW pairs against cache
    """
    print("\n" + "=" * 100)
    print(f"Incremental Advantage Analysis ({num_chunks} chunks)")
    print("=" * 100)

    print(
        f"\n{'Task':>5} | {'Full Pair Checks':>16} | {'Incr. Pair Checks':>18} | {'Savings':>8} | "
        f"{'Full User Parses':>16} | {'Incr. User Parses':>18} | {'User Parse Sav.':>14}"
    )
    print("-" * 110)

    total_full = 0
    total_incr = 0
    total_full_users = 0
    total_incr_users = 0

    for task_idx, task_result in sorted(results.items()):
        fp = task_result["total_full_pair_checks"]
        ip = task_result["total_incremental_pair_checks"]
        fu = task_result["total_full_user_parses"]
        iu = task_result["total_incremental_user_parses"]
        pair_savings = (1 - ip / fp) * 100 if fp > 0 else 0
        user_savings = (1 - iu / fu) * 100 if fu > 0 else 0

        total_full += fp
        total_incr += ip
        total_full_users += fu
        total_incr_users += iu

        print(
            f"{task_idx:>5} | {fp:>16,} | {ip:>18,} | {pair_savings:>7.1f}% | "
            f"{fu:>16,} | {iu:>18,} | {user_savings:>13.1f}%"
        )

    overall_savings = (1 - total_incr / total_full) * 100 if total_full > 0 else 0
    overall_user_savings = (
        (1 - total_incr_users / total_full_users) * 100 if total_full_users > 0 else 0
    )
    print("-" * 110)
    print(
        f"{'TOTAL':>5} | {total_full:>16,} | {total_incr:>18,} | {overall_savings:>7.1f}% | "
        f"{total_full_users:>16,} | {total_incr_users:>18,} | {overall_user_savings:>13.1f}%"
    )

    # Per-chunk breakdown for a representative task
    print("\n--- Per-Chunk Detail (Task 1) ---")
    task1 = results.get(1, results.get(list(results.keys())[0]))
    print(
        f"  {'Chunk':>6} | {'New Users':>10} | {'Tot Users':>10} | {'Full Checks':>12} | "
        f"{'Incr. Checks':>12} | {'Savings':>8} | {'New Pairs':>10}"
    )
    for c in task1["chunks"]:
        print(
            f"  {c['chunk_index'] + 1:>6} | {c['new_users']:>10} | {c['total_users']:>10} | "
            f"{c['full_pair_checks']:>12,} | {c['incremental_pair_checks']:>12,} | "
            f"{c['pair_check_savings_pct']:>7.1f}% | {c['new_pairs_this_chunk']:>10,}"
        )

    # Token cost model
    # Assume ~100 tokens per user-parse and ~10 tokens per pair-check
    TOKENS_PER_USER_PARSE = 100  # reading user's instances
    TOKENS_PER_PAIR_CHECK = 10  # evaluating one pair condition

    full_tokens = total_full * TOKENS_PER_PAIR_CHECK + total_full_users * TOKENS_PER_USER_PARSE
    incr_tokens = total_incr * TOKENS_PER_PAIR_CHECK + total_incr_users * TOKENS_PER_USER_PARSE
    token_savings = (1 - incr_tokens / full_tokens) * 100 if full_tokens > 0 else 0

    print("\n--- Estimated Token Model ---")
    print(
        f"  Assuming {TOKENS_PER_USER_PARSE} tokens/user-parse, {TOKENS_PER_PAIR_CHECK} tokens/pair-check"
    )
    print(f"  Full recompute:  {full_tokens:>12,} estimated tokens")
    print(f"  Incremental:     {incr_tokens:>12,} estimated tokens")
    print(f"  Token savings:   {token_savings:.1f}%")

    return {
        "total_full_pair_checks": total_full,
        "total_incremental_pair_checks": total_incr,
        "pair_check_savings_pct": overall_savings,
        "estimated_full_tokens": full_tokens,
        "estimated_incremental_tokens": incr_tokens,
        "estimated_token_savings_pct": token_savings,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-chunks", type=int, default=5)
    parser.add_argument("--tasks", type=str, default="1,2,3,6,8,11,13,19")
    parser.add_argument(
        "--output", type=str, default="results/streaming/incremental_advantage.json"
    )
    args = parser.parse_args()

    task_indices = [int(t) for t in args.tasks.split(",")]

    print("Loading data...")
    _, labeled_context = load_data()

    for n_chunks in [3, 5, 10]:
        results = analyze_incremental_vs_full(labeled_context, n_chunks, task_indices)
        summary = estimate_token_savings(results, n_chunks)

        if n_chunks == args.num_chunks:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            output = {
                "num_chunks": n_chunks,
                "tasks": {str(k): v for k, v in results.items()},
                "summary": summary,
            }
            with open(args.output, "w") as f:
                json.dump(output, f, indent=2)
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
