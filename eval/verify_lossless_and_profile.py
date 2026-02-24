"""
Losslessness Verification & Memory Profiling Experiment.

Addresses two external reviewer concerns:
1. "Caching is lossy compression" → proves EntityCache is LOSSLESS
2. "Memory will blow up" → profiles actual memory at each turn, extrapolates

This is a deterministic experiment ($0, no API calls). It:
- Processes OOLONG-Pairs context in chunks using IncrementalState.process_chunk()
- After each chunk, verifies entity cache contains EXACTLY the union of all
  entities seen so far (losslessness proof)
- After each chunk, reports memory usage of EntityCache, PairTracker, etc.
- Extrapolates memory to n=1K, 10K, 100K entities
- Compares REPL state memory vs LLM context window sizes

Usage:
  python eval/verify_lossless_and_profile.py --tasks 1,3,6 --num-chunks 5
  python eval/verify_lossless_and_profile.py --verify-lossless  # strict assertion mode
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.incremental_simulation import (
    load_data,
    make_task_checker,
    parse_users_from_labeled,
    split_labeled_context,
)
from eval.utils import _check_pair_condition
from rlm.core.incremental import IncrementalState


def run_verification_and_profiling(
    labeled_context: str,
    task_indices: list[int],
    num_chunks: int,
    verify_lossless: bool = False,
) -> dict:
    """Run losslessness verification + memory profiling on incremental pipeline.

    Args:
        labeled_context: Full labeled OOLONG-Pairs text
        task_indices: List of task IDs
        num_chunks: Number of chunks
        verify_lossless: If True, assert losslessness (fail-fast on violation)
    """
    chunks = split_labeled_context(labeled_context, num_chunks)
    results = {}

    for task_idx in task_indices:
        print(f"\n{'=' * 70}")
        print(f"Task {task_idx} — Losslessness Verification + Memory Profiling")
        print(f"{'=' * 70}")

        checker = make_task_checker(task_idx)
        state = IncrementalState()

        # Track ALL entities seen so far (ground truth for losslessness)
        all_entities_seen: set[str] = set()
        all_user_instances: dict[int, list[dict]] = {}

        lossless_results = []
        memory_results = []
        turn_details = []

        for chunk_i, chunk in enumerate(chunks):
            chunk_users = parse_users_from_labeled(chunk)

            # Build entity dict and track ground truth
            chunk_entities = {}
            for uid, instances in chunk_users.items():
                eid = str(uid)  # normalize to string
                all_entities_seen.add(eid)
                if uid in all_user_instances:
                    merged = all_user_instances[uid] + instances
                    all_user_instances[uid] = merged
                    chunk_entities[eid] = {"instances": merged}
                else:
                    all_user_instances[uid] = instances
                    chunk_entities[eid] = {"instances": instances}

            # Process chunk
            chunk_stats = state.process_chunk(
                chunk_i, chunk_entities, pair_checker=checker
            )

            # === LOSSLESSNESS VERIFICATION ===
            lossless_check = state.verify_lossless(all_entities_seen)
            lossless_results.append(lossless_check)

            if verify_lossless:
                assert lossless_check["is_lossless"], (
                    f"LOSSLESSNESS VIOLATION at chunk {chunk_i}! "
                    f"Missing: {lossless_check['missing_ids']}, "
                    f"Extra: {lossless_check['extra_ids']}"
                )

            # === MEMORY PROFILING ===
            mem = state.memory_usage()
            memory_results.append(mem)

            # Print turn summary
            status = "✓ LOSSLESS" if lossless_check["is_lossless"] else "✗ LOSSY"
            print(
                f"  Turn {chunk_i + 1}/{num_chunks}: "
                f"{status} | "
                f"entities={lossless_check['cached_count']}/{lossless_check['expected_count']} | "
                f"pairs={chunk_stats['total_pairs']} | "
                f"memory={mem['total_kb']:.1f} KB "
                f"(entity={mem['component_breakdown']['entity_cache']:.1f} KB, "
                f"pairs={mem['component_breakdown']['pair_set']:.1f} KB, "
                f"index={mem['component_breakdown']['inverted_index']:.1f} KB)"
            )

            turn_details.append({
                "turn": chunk_i + 1,
                "lossless": lossless_check,
                "memory": mem,
                "chunk_stats": chunk_stats,
            })

        # === SUMMARY ===
        all_lossless = all(r["is_lossless"] for r in lossless_results)
        final_mem = memory_results[-1]

        print(f"\n  === LOSSLESSNESS SUMMARY ===")
        print(f"  All turns lossless: {'YES' if all_lossless else 'NO'}")
        print(f"  Final entity count: {lossless_results[-1]['cached_count']}")
        print(f"  Total entities ever seen: {len(all_entities_seen)}")

        print(f"\n  === MEMORY SUMMARY ===")
        print(f"  Final total memory: {final_mem['total_kb']:.1f} KB ({final_mem['total_mb']:.3f} MB)")
        print(f"  Breakdown:")
        for component, kb in final_mem["component_breakdown"].items():
            print(f"    {component}: {kb:.1f} KB")

        # === SCALING PROJECTION ===
        n_final = final_mem["counts"]["entities"]
        p_final = final_mem["counts"]["pairs"]
        total_bytes = final_mem["total_bytes"]

        # Per-entity and per-pair cost
        entity_cache_bytes = final_mem["entity_cache_bytes"] + final_mem["chunk_index_bytes"]
        pair_bytes = final_mem["pair_tracker_bytes"] + final_mem["inverted_index_bytes"]
        bytes_per_entity = entity_cache_bytes / max(n_final, 1)
        bytes_per_pair = pair_bytes / max(p_final, 1)

        print(f"\n  === SCALING PROJECTION ===")
        print(f"  Measured: {n_final} entities, {p_final} pairs → {total_bytes:,} bytes")
        print(f"  Per-entity cost: {bytes_per_entity:.0f} bytes")
        print(f"  Per-pair cost: {bytes_per_pair:.0f} bytes")
        print()

        projections = []
        for n_proj in [1_000, 10_000, 100_000]:
            # Assume pair density scales as p/C(n,2) ratio stays constant
            if n_final > 1:
                pair_density = p_final / (n_final * (n_final - 1) / 2)
            else:
                pair_density = 0
            p_proj = int(pair_density * n_proj * (n_proj - 1) / 2)
            entity_mem = n_proj * bytes_per_entity
            pair_mem = p_proj * bytes_per_pair
            total_proj = entity_mem + pair_mem
            proj = {
                "n": n_proj,
                "projected_pairs": p_proj,
                "entity_memory_mb": round(entity_mem / (1024 * 1024), 2),
                "pair_memory_mb": round(pair_mem / (1024 * 1024), 2),
                "total_memory_mb": round(total_proj / (1024 * 1024), 2),
            }
            projections.append(proj)
            print(
                f"  n={n_proj:>7,}: ~{p_proj:>12,} pairs → "
                f"{proj['total_memory_mb']:>8.2f} MB "
                f"(entity={proj['entity_memory_mb']:.2f} MB, "
                f"pair={proj['pair_memory_mb']:.2f} MB)"
            )

        # Compare vs LLM context
        print(f"\n  === COMPARISON: REPL State vs LLM Context ===")
        for ctx_tokens, ctx_name in [
            (32_000, "32K context"),
            (128_000, "128K context"),
            (1_000_000, "1M context"),
        ]:
            ctx_bytes = ctx_tokens * 4  # ~4 bytes per token
            ratio = total_bytes / ctx_bytes
            print(
                f"  REPL state ({final_mem['total_kb']:.0f} KB) vs "
                f"{ctx_name} ({ctx_bytes // 1024} KB): "
                f"{ratio:.4f}x ({ratio * 100:.2f}%)"
            )

        results[task_idx] = {
            "task_id": task_idx,
            "num_chunks": num_chunks,
            "all_lossless": all_lossless,
            "final_entity_count": lossless_results[-1]["cached_count"],
            "total_entities_seen": len(all_entities_seen),
            "final_memory": final_mem,
            "per_entity_bytes": round(bytes_per_entity, 1),
            "per_pair_bytes": round(bytes_per_pair, 1),
            "projections": projections,
            "turn_details": turn_details,
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Losslessness Verification & Memory Profiling"
    )
    parser.add_argument("--tasks", type=str, default="1,3,6")
    parser.add_argument("--num-chunks", type=int, default=5)
    parser.add_argument(
        "--verify-lossless",
        action="store_true",
        help="Assert losslessness (fail on violation)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/streaming/verify_lossless_and_profile.json",
    )
    args = parser.parse_args()

    task_indices = [int(t) for t in args.tasks.split(",")]

    print("Loading data...")
    _, labeled_context = load_data()

    print(
        f"Running verification + profiling: tasks={task_indices}, "
        f"chunks={args.num_chunks}, verify_lossless={args.verify_lossless}"
    )
    results = run_verification_and_profiling(
        labeled_context,
        task_indices,
        args.num_chunks,
        verify_lossless=args.verify_lossless,
    )

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")

    # Final verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    all_lossless = all(r["all_lossless"] for r in results.values())
    print(f"Losslessness: {'ALL TASKS VERIFIED LOSSLESS' if all_lossless else 'VIOLATIONS DETECTED'}")
    for task_idx, r in sorted(results.items()):
        mem = r["final_memory"]
        print(
            f"  Task {task_idx}: "
            f"{'✓' if r['all_lossless'] else '✗'} lossless | "
            f"{r['final_entity_count']} entities | "
            f"{mem['counts']['pairs']} pairs | "
            f"{mem['total_kb']:.1f} KB total"
        )


if __name__ == "__main__":
    main()
