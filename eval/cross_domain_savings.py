#!/usr/bin/env python3
"""
Cross-domain savings quantification for IncrementalState.

Measures pair-check savings, memory usage, and rebuild correctness across
4 non-OOLONG domains to prove the library is domain-agnostic.

No API calls needed — purely synthetic data.

Usage:
    python eval/cross_domain_savings.py
"""

import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from rlm.core.incremental import IncrementalState


def keyword_overlap_checker(a: dict, b: dict) -> bool:
    kw1 = a.get("keywords", frozenset())
    kw2 = b.get("keywords", frozenset())
    return len(kw1 & kw2) >= 3


def interest_checker(a: dict, b: dict) -> bool:
    i1 = a.get("interests", frozenset())
    i2 = b.get("interests", frozenset())
    return len(i1 & i2) >= 2


def category_checker(a: dict, b: dict) -> bool:
    if a.get("type") == b.get("type"):
        return False
    c1 = a.get("categories", frozenset())
    c2 = b.get("categories", frozenset())
    return len(c1 & c2) >= 1


def threshold_checker(a: dict, b: dict) -> bool:
    return a.get("score", 0) > 0.5 and b.get("score", 0) > 0.5


def make_doc_chunks(n_chunks: int = 5, per_chunk: int = 20) -> list[dict[str, dict[str, Any]]]:
    return [
        {
            f"doc_{c * per_chunk + i}": {
                "keywords": frozenset(f"kw_{j}" for j in range(c * per_chunk + i, c * per_chunk + i + 5))
            }
            for i in range(per_chunk)
        }
        for c in range(n_chunks)
    ]


def make_user_chunks(n_chunks: int = 5, per_chunk: int = 20) -> list[dict[str, dict[str, Any]]]:
    return [
        {
            f"user_{c * per_chunk + i}": {
                "interests": frozenset(f"int_{j}" for j in range(i % 8, i % 8 + 4))
            }
            for i in range(per_chunk)
        }
        for c in range(n_chunks)
    ]


def make_product_chunks(n_chunks: int = 5, per_chunk: int = 20) -> list[dict[str, dict[str, Any]]]:
    cats = ["electronics", "books", "sports", "music", "food", "art", "tech"]
    rng = random.Random(42)
    return [
        {
            f"{'cust' if i < per_chunk // 2 else 'prod'}_{c * per_chunk + i}": {
                "type": "customer" if i < per_chunk // 2 else "product",
                "categories": frozenset(rng.sample(cats, k=rng.randint(1, 3))),
            }
            for i in range(per_chunk)
        }
        for c in range(n_chunks)
    ]


def make_threshold_chunks(
    n_chunks: int = 5, per_chunk: int = 20, update_rate: float = 0.0
) -> list[dict[str, dict[str, Any]]]:
    rng = random.Random(42)
    ground_truth: dict[str, dict] = {}
    chunks = []
    for c in range(n_chunks):
        chunk = {}
        for i in range(per_chunk):
            eid = f"e_{c * per_chunk + i}"
            chunk[eid] = {"score": rng.random()}
            ground_truth[eid] = chunk[eid]
        # Add updates
        if update_rate > 0 and c > 0:
            existing = [eid for eid in ground_truth if eid not in chunk]
            n_updates = int(len(existing) * update_rate)
            for eid in rng.sample(existing, min(n_updates, len(existing))):
                chunk[eid] = {"score": rng.random()}
                ground_truth[eid] = chunk[eid]
        chunks.append(chunk)
    return chunks


def run_comparison(name: str, checker, chunks: list[dict]) -> dict:
    """Run incremental vs full recompute comparison for a domain."""
    # Incremental
    state_incr = IncrementalState()
    incr_checks = 0
    t0 = time.perf_counter()
    for idx, chunk in enumerate(chunks):
        stats = state_incr.process_chunk(idx, chunk, checker)
        incr_checks += stats["pair_checks"]
    incr_time = time.perf_counter() - t0
    incr_pairs = state_incr.pair_tracker.get_pairs()

    # Full recompute (reset + replay each turn)
    state_full = IncrementalState()
    full_checks = 0
    t0 = time.perf_counter()
    for turn in range(len(chunks)):
        state_full.reset()
        for idx in range(turn + 1):
            state_full.process_chunk(idx, chunks[idx], checker)
            full_checks += state_full.chunk_log[-1]["pair_checks"]
    full_time = time.perf_counter() - t0
    full_pairs = state_full.pair_tracker.get_pairs()

    # Verify correctness
    pairs_match = incr_pairs == full_pairs

    # Rebuild check
    rebuild = state_incr.rebuild_pairs(checker)

    # Memory
    mem = state_incr.memory_usage()

    savings = 1 - incr_checks / full_checks if full_checks > 0 else 0

    return {
        "domain": name,
        "k": len(chunks),
        "total_entities": sum(len(c) for c in chunks),
        "unique_entities": len(set().union(*(c.keys() for c in chunks))),
        "pairs": len(incr_pairs),
        "incr_checks": incr_checks,
        "full_checks": full_checks,
        "savings_pct": savings * 100,
        "incr_time_ms": incr_time * 1000,
        "full_time_ms": full_time * 1000,
        "speedup": full_time / incr_time if incr_time > 0 else float("inf"),
        "pairs_match": pairs_match,
        "rebuild_match": rebuild["match"],
        "memory_kb": mem["total_kb"],
    }


def main():
    print("=" * 90)
    print("Cross-Domain IncrementalState Savings Report")
    print("=" * 90)
    print()

    domains = [
        ("Document-Keyword (≥3 shared keywords)", keyword_overlap_checker, make_doc_chunks(5, 20)),
        ("User-Interest (≥2 shared interests)", interest_checker, make_user_chunks(5, 20)),
        ("Product-Category (cross-type, ≥1 cat)", category_checker, make_product_chunks(5, 20)),
        ("Threshold (score > 0.5, no churn)", threshold_checker, make_threshold_chunks(5, 20, 0.0)),
        ("High-Churn Threshold (50% updates)", threshold_checker, make_threshold_chunks(5, 20, 0.5)),
    ]

    results = []
    for name, checker, chunks in domains:
        result = run_comparison(name, checker, chunks)
        results.append(result)

    # Print table
    print(f"{'Domain':<45} {'k':>3} {'N':>5} {'Pairs':>6} {'Incr':>8} {'Full':>8} "
          f"{'Savings':>8} {'Speedup':>8} {'Match':>6} {'Rebuild':>8} {'Mem(KB)':>8}")
    print("-" * 130)
    for r in results:
        print(f"{r['domain']:<45} {r['k']:>3} {r['unique_entities']:>5} {r['pairs']:>6} "
              f"{r['incr_checks']:>8} {r['full_checks']:>8} {r['savings_pct']:>7.1f}% "
              f"{r['speedup']:>7.1f}× {('✓' if r['pairs_match'] else '✗'):>6} "
              f"{('✓' if r['rebuild_match'] else '✗'):>8} {r['memory_kb']:>8}")
    print()

    # Summary
    print("Summary:")
    all_match = all(r["pairs_match"] and r["rebuild_match"] for r in results)
    avg_savings = sum(r["savings_pct"] for r in results) / len(results)
    print(f"  All domains correct (incr==full, rebuild match): {'✓ YES' if all_match else '✗ NO'}")
    print(f"  Average pair-check savings vs full recompute:    {avg_savings:.1f}%")
    print(f"  Domains tested:                                  {len(results)}")
    print()

    # Structural formula comparison
    k = 5
    structural = (1 - 2 / (k + 1)) * 100
    print(f"Structural savings formula 1-2/(k+1) for k={k}:  {structural:.1f}%")
    print("Note: Structural formula applies to linear token reads (O(k) vs O(k²)).")
    print("Pair-check savings reflect quadratic within-chunk costs and may differ.")


def scale_test():
    """Test savings at different scales to show they hold."""
    print()
    print("=" * 90)
    print("Scale Sensitivity (Document-Keyword domain, k=5)")
    print("=" * 90)
    print()
    print(f"{'N (entities)':>12} {'Incr Checks':>12} {'Full Checks':>12} "
          f"{'Savings':>8} {'Speedup':>8} {'Memory(KB)':>10}")
    print("-" * 70)

    for per_chunk in [10, 20, 50, 100, 200]:
        chunks = make_doc_chunks(5, per_chunk)
        result = run_comparison(f"DocKW-{per_chunk * 5}", keyword_overlap_checker, chunks)
        print(f"{result['unique_entities']:>12} {result['incr_checks']:>12} "
              f"{result['full_checks']:>12} {result['savings_pct']:>7.1f}% "
              f"{result['speedup']:>7.1f}× {result['memory_kb']:>10}")


if __name__ == "__main__":
    main()
    scale_test()
