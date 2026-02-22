"""
Update-Rate Parametric Experiment.

This script measures how pair-check savings degrade as the artificial update
rate increases. It directly characterizes the O(u·n) regime identified in
the complexity analysis.

For each update rate p in [0%, 5%, 10%, 20%]:
    On each chunk after the first, artificially mark p fraction of existing
    entities as "updated" (same attributes — no functional change, just
    triggering the updated-entity sweep). Then measure total pair-check savings
    vs. full recompute.

This shows:
    - At p=0%: baseline savings (only new entities drive incremental work)
    - At p=5%: moderate overhead from updated-entity sweeps
    - At p=10%, 20%: regime where O(u·n) dominates → savings collapse
    - Break-even update rate: where incremental ≈ full recompute

Hypothesis: savings break even (0%) at approximately u_breakeven = k_new,
i.e., when the number of updates per chunk equals the number of new entities.

Usage:
    python eval/update_rate_experiment.py
    python eval/update_rate_experiment.py --tasks 1,19 --num-chunks 5
    python eval/update_rate_experiment.py --output results/streaming/update_rate_results.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.utils import _check_pair_condition
from rlm.core.incremental import IncrementalState


UPDATE_RATES = [0.0, 0.05, 0.10, 0.20]  # fractions of existing entities to mark as "updated"


def load_data(context_len: int = 32768):
    from datasets import load_dataset
    ds = load_dataset("oolongbench/oolong-synth", split="validation")
    corpus = [x for x in ds if x["dataset"] == "trec_coarse" and x["context_len"] == context_len][0]
    return corpus["context_window_text"], corpus["context_window_text_with_labels"]


def parse_users_from_labeled(labeled_text: str) -> dict:
    import re
    from eval.utils import _parse_labeled_context
    return _parse_labeled_context(labeled_text)


def split_labeled_context(labeled_context: str, num_chunks: int) -> list[str]:
    import re
    user_pattern = re.compile(r"(?=Date:.*?\|\| User:)")
    positions = [m.start() for m in user_pattern.finditer(labeled_context)]
    if not positions:
        chunk_size = len(labeled_context) // num_chunks
        return [labeled_context[i*chunk_size:(i+1)*chunk_size if i<num_chunks-1 else len(labeled_context)]
                for i in range(num_chunks)]
    users_per_chunk = max(1, len(positions) // num_chunks)
    chunks = []
    for i in range(num_chunks):
        start = positions[i*users_per_chunk] if i*users_per_chunk < len(positions) else len(labeled_context)
        if i < num_chunks - 1:
            end_idx = min((i+1)*users_per_chunk, len(positions))
            end = positions[end_idx] if end_idx < len(positions) else len(labeled_context)
        else:
            end = len(labeled_context)
        if start < end:
            chunks.append(labeled_context[start:end])
    while len(chunks) < num_chunks:
        chunks.append("")
    return chunks[:num_chunks]


def run_update_rate_simulation(
    labeled_context: str,
    task_indices: list[int],
    num_chunks: int,
    update_rate: float,
    rng_seed: int = 42,
    baseline_final_pairs: dict | None = None,
) -> dict:
    """
    Run incremental simulation with artificial update injection.

    For each chunk i > 0, randomly select (update_rate × n_existing) entities
    from the entity cache and include them in chunk_entities with their CURRENT
    attributes (no functional change). This forces the O(u·n) sweep without
    altering the final pair set.
    """
    chunks = split_labeled_context(labeled_context, num_chunks)
    rng = random.Random(rng_seed)
    results = {}

    for task_idx in task_indices:
        def checker(attrs1, attrs2):
            return _check_pair_condition(attrs1["instances"], attrs2["instances"], task_idx)

        incr_state = IncrementalState()
        incr_pair_checks = 0
        full_pair_checks = 0
        cumulative_labeled = ""
        all_user_instances: dict = {}

        for chunk_i, chunk in enumerate(chunks):
            cumulative_labeled += chunk
            chunk_users = parse_users_from_labeled(chunk)

            # Build entity dict for natural new + updated entities
            chunk_entities = {}
            for uid, instances in chunk_users.items():
                if uid in all_user_instances:
                    merged = all_user_instances[uid] + instances
                    all_user_instances[uid] = merged
                    chunk_entities[uid] = {"instances": merged}
                else:
                    all_user_instances[uid] = instances
                    chunk_entities[uid] = {"instances": instances}

            # Artificial update injection (chunk_i > 0 only)
            if chunk_i > 0 and update_rate > 0:
                existing_ids = list(incr_state.entity_cache.get_ids())
                n_artificial_updates = int(len(existing_ids) * update_rate)
                if n_artificial_updates > 0:
                    artificial_update_ids = rng.sample(
                        existing_ids, min(n_artificial_updates, len(existing_ids))
                    )
                    for uid in artificial_update_ids:
                        if uid not in chunk_entities:
                            # Include with current attributes (no functional change)
                            current_attrs = incr_state.entity_cache.get(uid)
                            if current_attrs is not None:
                                chunk_entities[uid] = current_attrs

            chunk_stats = incr_state.process_chunk(chunk_i, chunk_entities, pair_checker=checker)
            incr_pair_checks += chunk_stats["pair_checks"]

            # Full recompute
            cumulative_users = parse_users_from_labeled(cumulative_labeled)
            chunk_full_checks = len(list(combinations(sorted(cumulative_users.keys()), 2)))
            full_pair_checks += chunk_full_checks

        savings_pct = (1 - incr_pair_checks / full_pair_checks) * 100 if full_pair_checks > 0 else 0
        stats = incr_state.get_stats()
        final_pairs_count = stats["total_pairs"]

        # No-op assertion: artificial updates inject entities with CURRENT accumulated attrs,
        # which should produce exactly the same final pairs as the p=0% baseline.
        # If final_pairs differs from baseline, the artificial injection is NOT functionally
        # no-op (e.g., for "exactly N" tasks near condition boundaries) and the savings
        # comparison is invalid.
        if baseline_final_pairs is not None and task_idx in baseline_final_pairs:
            expected = baseline_final_pairs[task_idx]
            if final_pairs_count != expected:
                raise AssertionError(
                    f"Task {task_idx} at update_rate={update_rate:.0%}: "
                    f"final_pairs={final_pairs_count} != baseline={expected}. "
                    f"Artificial update injection is NOT functionally no-op for this task. "
                    f"The savings comparison against p=0% baseline is invalid."
                )

        results[task_idx] = {
            "task_id": task_idx,
            "update_rate": update_rate,
            "num_chunks": num_chunks,
            "incr_pair_checks": incr_pair_checks,
            "full_pair_checks": full_pair_checks,
            "savings_pct": round(savings_pct, 2),
            "total_retractions": stats["total_retractions"],
            "final_pairs": final_pairs_count,
            "total_entities": stats["total_entities"],
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Update-Rate Parametric Experiment")
    parser.add_argument("--tasks", type=str, default="1,19")
    parser.add_argument("--num-chunks", type=int, default=5)
    parser.add_argument(
        "--update-rates", type=str, default="0.0,0.05,0.10,0.20",
        help="Comma-separated list of artificial update rates (fractions of existing entities)",
    )
    parser.add_argument(
        "--seeds", type=str, default="42",
        help="Comma-separated list of RNG seeds for robustness analysis",
    )
    parser.add_argument(
        "--output", type=str, default="results/streaming/update_rate_results.json"
    )
    args = parser.parse_args()

    task_indices = [int(t) for t in args.tasks.split(",")]
    update_rates = [float(r) for r in args.update_rates.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    print("=" * 70)
    print("UPDATE-RATE PARAMETRIC EXPERIMENT")
    print("=" * 70)
    print(f"Tasks: {task_indices}, Chunks: {args.num_chunks}")
    print(f"Update rates: {[f'{r:.0%}' for r in update_rates]}")
    print(f"Seeds: {seeds}")
    print()

    print("Loading data...")
    _, labeled_context = load_data()

    # First: get p=0% baseline final_pairs (used to assert no-op for p>0%)
    print("\nComputing p=0% baseline final_pairs for no-op assertion...")
    baseline_run = run_update_rate_simulation(
        labeled_context, task_indices, args.num_chunks, update_rate=0.0, rng_seed=42
    )
    baseline_final_pairs = {tid: r["final_pairs"] for tid, r in baseline_run.items()}
    print(f"  Baseline final_pairs: {baseline_final_pairs}")
    print("  No-op assertion will verify all p>0% runs produce identical final_pairs.")

    all_results = {}
    summary_rows = []
    # {rate_key -> {task_id -> [savings across seeds]}}
    seed_savings: dict[str, dict[str, list[float]]] = {}

    for rate in update_rates:
        rate_key = f"{rate:.2f}"
        seed_savings[rate_key] = {str(tid): [] for tid in task_indices}

        for seed in seeds:
            print(f"\n{'=' * 60}")
            print(f"Update rate: {rate:.0%}, seed: {seed}")
            print(f"{'=' * 60}")
            t0 = time.perf_counter()
            run_results = run_update_rate_simulation(
                labeled_context,
                task_indices,
                args.num_chunks,
                rate,
                rng_seed=seed,
                baseline_final_pairs=baseline_final_pairs if rate > 0 else None,
            )
            elapsed = time.perf_counter() - t0

            if len(seeds) == 1:
                all_results[rate_key] = {str(tid): r for tid, r in run_results.items()}

            for task_id, r in sorted(run_results.items()):
                seed_savings[rate_key][str(task_id)].append(r["savings_pct"])
                if seed == seeds[0]:
                    print(
                        f"  Task {task_id}: savings={r['savings_pct']:+.1f}%  "
                        f"incr={r['incr_pair_checks']:,}  full={r['full_pair_checks']:,}  "
                        f"retractions={r['total_retractions']}  final_pairs={r['final_pairs']}"
                    )
                else:
                    print(f"  Task {task_id}: savings={r['savings_pct']:+.1f}%  (seed {seed})")
                summary_rows.append({
                    "update_rate": rate,
                    "task_id": task_id,
                    "seed": seed,
                    "savings_pct": r["savings_pct"],
                    "incr_checks": r["incr_pair_checks"],
                    "full_checks": r["full_pair_checks"],
                    "retractions": r["total_retractions"],
                    "final_pairs": r["final_pairs"],
                })
            print(f"  [Elapsed: {elapsed:.2f}s]")

        # After all seeds, aggregate
        if len(seeds) > 1:
            rate_agg = {}
            for tid in task_indices:
                tidstr = str(tid)
                savings_list = seed_savings[rate_key][tidstr]
                import statistics
                mean_s = statistics.mean(savings_list)
                std_s = statistics.stdev(savings_list) if len(savings_list) > 1 else 0.0
                rate_agg[tidstr] = {
                    "mean_savings_pct": round(mean_s, 2),
                    "std_savings_pct": round(std_s, 2),
                    "seeds_savings": savings_list,
                }
                print(
                    f"  Task {tid} AGGREGATE (n={len(seeds)} seeds): "
                    f"savings = {mean_s:.1f}% ± {std_s:.1f}%"
                )
            all_results[rate_key] = rate_agg

    # Summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY: Savings vs. Update Rate")
    print(f"{'=' * 70}")
    if len(seeds) == 1:
        print(f"{'Update Rate':>12} | {'Task 1 Savings':>14} | {'Task 19 Savings':>15}")
        print("-" * 46)
        for rate in update_rates:
            rate_key = f"{rate:.2f}"
            t1 = all_results[rate_key].get("1", {}).get("savings_pct", float("nan"))
            t19 = all_results[rate_key].get("19", {}).get("savings_pct", float("nan"))
            print(f"  {rate:>8.0%}   | {t1:>+12.1f}% | {t19:>+13.1f}%")
    else:
        print(f"{'Update Rate':>12} | {'Task 1 Mean±Std':>18} | {'Task 19 Mean±Std':>19}")
        print("-" * 54)
        for rate in update_rates:
            rate_key = f"{rate:.2f}"
            t1 = all_results[rate_key].get("1", {})
            t19 = all_results[rate_key].get("19", {})
            t1_str = f"{t1.get('mean_savings_pct', float('nan')):+.1f}% ± {t1.get('std_savings_pct', 0):.1f}%"
            t19_str = f"{t19.get('mean_savings_pct', float('nan')):+.1f}% ± {t19.get('std_savings_pct', 0):.1f}%"
            print(f"  {rate:>8.0%}   | {t1_str:>18} | {t19_str:>19}")

    # Break-even analysis
    print(f"\nBreak-even update rate analysis (Task 1, primary seed):")
    prev_savings = None
    prev_rate = None
    for rate in update_rates:
        rate_key = f"{rate:.2f}"
        if len(seeds) == 1:
            savings = all_results[rate_key].get("1", {}).get("savings_pct", float("nan"))
        else:
            savings = all_results[rate_key].get("1", {}).get("mean_savings_pct", float("nan"))
        if prev_savings is not None and prev_savings > 0 and savings <= 0:
            print(f"  Break-even occurs between {prev_rate:.0%} and {rate:.0%} update rate")
        elif savings <= 0:
            print(f"  Already at/below break-even at {rate:.0%}")
        prev_savings = savings
        prev_rate = rate

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output = {
        "config": {
            "task_indices": task_indices,
            "num_chunks": args.num_chunks,
            "update_rates": update_rates,
            "seeds": seeds,
        },
        "baseline_final_pairs": baseline_final_pairs,
        "no_op_assertion": "verified" if any(r > 0 for r in update_rates) else "not_tested",
        "results_by_rate": all_results,
        "summary": summary_rows,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
