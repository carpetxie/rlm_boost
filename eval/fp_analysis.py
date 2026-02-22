"""
FP Root Cause Analysis — Iteration 9.

Diagnoses the precision decline from 0.88 → 0.54 across 5 chunks (Task 1).

Hypothesis (from critique): FP rate grows because plain-text entity extraction
over-includes user IDs that appear in non-instance contexts (metadata, headers),
creating "phantom entities" not present in labeled_context gold. These phantom
entities form O(n) false-positive pairs each.

This script requires NO API calls. It uses:
1. The OOLONG-Pairs dataset (via HuggingFace datasets)
2. The existing f1_progression_results.json (from Iteration 8 Condition A run)
3. Pure Python analysis of entity extraction

Analysis:
  For each chunk k=1..5:
  A. Parse entity IDs from plain_context (first k*5000 chars) using same regex as model
  B. Parse entity IDs from labeled_context (all) — these are "gold entities"
  C. Classify predicted pairs (from pair_tracker at chunk k) as:
     (a) Both user IDs in gold entities ("clean pairs") — could be TP or FP from condition
     (b) At least one user ID NOT in gold entities ("phantom pairs") — always FP
  D. Count FPs from each category

This separates two distinct FP sources:
  Source 1: Entity extraction noise (phantom entities from plain text over-inclusion)
  Source 2: check_pair approximation (user HAS instances in gold but check_pair condition
            doesn't precisely match task condition, so some valid entity pairs are FPs)

Usage:
    python eval/fp_analysis.py
    python eval/fp_analysis.py --output results/streaming/fp_analysis.json
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_NUM_CHUNKS = 5
DEFAULT_CHUNK_CHARS = 5000
RESULTS_FILE = Path("results/streaming/f1_progression_results.json")


# ---------------------------------------------------------------------------
# Entity extraction from plain context (mirrors model's code template)
# ---------------------------------------------------------------------------

def extract_entities_from_plain(text: str) -> dict[str, list[str]]:
    """
    Extract entity IDs from plain context using the same regex as the model's code template.

    The model uses: re.search(r'User: (\\d+)', line)
    Returns: {user_id: [instance_lines]}
    """
    entities: dict[str, list[str]] = {}
    for line in text.split("\n"):
        m = re.search(r"User: (\d+)", line)
        if m:
            uid = m.group(1)
            if uid not in entities:
                entities[uid] = []
            entities[uid].append(line.strip())
    return entities


# ---------------------------------------------------------------------------
# Entity extraction from labeled context (gold)
# ---------------------------------------------------------------------------

def extract_gold_entity_ids(labeled_context: str) -> set[str]:
    """
    Extract gold entity IDs from labeled_context.

    Uses the same _LINE_RE as eval.utils._parse_labeled_context.
    Returns set of user_id strings (as strings to match model's extraction).
    """
    from eval.utils import _parse_labeled_context
    users = _parse_labeled_context(labeled_context)
    # Convert int keys to str (model extracts user IDs as strings)
    return {str(uid) for uid in users.keys()}


# ---------------------------------------------------------------------------
# Coverage ceiling analysis
# ---------------------------------------------------------------------------

def analyze_coverage_ceiling(
    plain_context: str,
    gold_pairs: set,
    num_chunks: int = DEFAULT_NUM_CHUNKS,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
) -> list[dict]:
    """
    For each chunk k, count gold pairs where BOTH user IDs appear in plain_context[:k*chunk_chars].
    This gives the hard upper bound on F1 achievable by any system using k*chunk_chars of plain context.

    Key question: are the remaining FN at k=5 due to:
    (a) Coverage: gold users don't appear in the first 25K chars of plain context?
    (b) Protocol: incremental protocol fails to find pairs that ARE present?
    """
    results = []
    plain_chunks = plain_context[:num_chunks * chunk_chars]

    for k in range(1, num_chunks + 1):
        context_k = plain_context[:k * chunk_chars]
        entities_k = extract_entities_from_plain(context_k)
        entity_ids_k = set(entities_k.keys())

        # Count gold pairs where both users appear in context_k
        covered_gold = set()
        for (uid1, uid2) in gold_pairs:
            u1, u2 = str(uid1), str(uid2)
            if u1 in entity_ids_k and u2 in entity_ids_k:
                covered_gold.add((min(u1, u2), max(u1, u2)))

        coverage_pct = len(covered_gold) / len(gold_pairs) * 100 if gold_pairs else 0.0
        results.append({
            "k": k,
            "chars_seen": k * chunk_chars,
            "entity_ids_in_plain": len(entity_ids_k),
            "gold_pairs_covered": len(covered_gold),
            "gold_pairs_total": len(gold_pairs),
            "coverage_pct": round(coverage_pct, 2),
            "f1_ceiling": round(2 * coverage_pct / (100 + coverage_pct), 4) if coverage_pct > 0 else 0.0,
            # F1 ceiling at perfect precision: F1 = 2*R/(1+R) when P=1 and R=coverage_pct/100
        })
    return results


# ---------------------------------------------------------------------------
# FP categorization analysis
# ---------------------------------------------------------------------------

def analyze_fp_categories(
    plain_context: str,
    labeled_context: str,
    gold_pairs: set,
    f1_progression_results: dict,
    num_chunks: int = DEFAULT_NUM_CHUNKS,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
) -> list[dict]:
    """
    For each chunk k=1..5, categorize FPs as phantom or clean.

    Phantom FPs: pairs where >= 1 user ID is NOT in labeled_context gold entities.
                 These are FPs caused by plain-text over-extraction.
    Clean FPs: pairs where both user IDs ARE in labeled_context gold, but the pair
               doesn't satisfy the gold condition.
               These are FPs from check_pair approximation mismatch.

    Also categorizes FNs:
    Covered FNs: gold pairs where both users appear in plain_context[:k*chunk_chars]
                 but the model missed them. These are PROTOCOL failures (should be findable).
    Uncovered FNs: gold pairs where >= 1 user doesn't appear in plain_context[:k*chunk_chars].
                   These are COVERAGE failures (can't be found with limited context).
    """
    gold_entity_ids = extract_gold_entity_ids(labeled_context)

    # Normalize gold pairs to string tuples
    gold_str = set()
    for p in gold_pairs:
        gold_str.add((min(str(p[0]), str(p[1])), max(str(p[0]), str(p[1]))))

    results = []
    f1_prog = f1_progression_results.get("condition_a", {}).get("f1_progression", [])

    for k in range(1, num_chunks + 1):
        context_k = plain_context[:k * chunk_chars]
        entities_k = extract_entities_from_plain(context_k)
        entity_ids_k = set(entities_k.keys())

        # Get the number of pairs/FPs from existing results (if available)
        existing_k = next((t for t in f1_prog if t["chunk"] == k), None)
        tp_existing = existing_k.get("tp", 0) if existing_k else 0
        fp_existing = existing_k.get("fp", 0) if existing_k else 0
        fn_existing = existing_k.get("fn", 0) if existing_k else 0
        pairs_existing = existing_k.get("pairs", 0) if existing_k else 0

        # Predicted pairs = all C(entity_ids_k, 2) pairs where both have >= 1 instance
        # (This mirrors what the model's check_pair does: len(instances) >= 1)
        qualifying_ids = sorted([uid for uid, insts in entities_k.items() if len(insts) >= 1])
        predicted_pairs = set()
        for i, uid1 in enumerate(qualifying_ids):
            for uid2 in qualifying_ids[i + 1:]:
                predicted_pairs.add((min(uid1, uid2), max(uid1, uid2)))

        # Categorize FPs
        fp_set = predicted_pairs - gold_str
        phantom_fps = set()  # >= 1 user NOT in gold_entity_ids
        clean_fps = set()    # both users in gold_entity_ids but pair not gold
        for (u1, u2) in fp_set:
            if u1 not in gold_entity_ids or u2 not in gold_entity_ids:
                phantom_fps.add((u1, u2))
            else:
                clean_fps.add((u1, u2))

        # Categorize FNs
        fn_set = gold_str - predicted_pairs
        covered_fns = set()   # both users in entity_ids_k (protocol failure)
        uncovered_fns = set() # >= 1 user not in entity_ids_k (coverage failure)
        for (u1, u2) in fn_set:
            if u1 in entity_ids_k and u2 in entity_ids_k:
                covered_fns.add((u1, u2))
            else:
                uncovered_fns.add((u1, u2))

        tp = len(predicted_pairs & gold_str)
        fp = len(fp_set)
        fn = len(fn_set)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Phantom entity count
        phantom_entity_ids = entity_ids_k - gold_entity_ids
        gold_entity_ids_in_context = entity_ids_k & gold_entity_ids

        results.append({
            "k": k,
            "chars_seen": k * chunk_chars,
            # Entities
            "entity_ids_in_plain": len(entity_ids_k),
            "entity_ids_gold_total": len(gold_entity_ids),
            "entity_ids_in_context_AND_gold": len(gold_entity_ids_in_context),
            "phantom_entity_ids": len(phantom_entity_ids),
            "phantom_entity_pct": round(len(phantom_entity_ids) / len(entity_ids_k) * 100, 1)
                                  if entity_ids_k else 0.0,
            # Pairs
            "predicted_pairs": len(predicted_pairs),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            # FP categories
            "fp_phantom": len(phantom_fps),  # >= 1 user NOT in gold
            "fp_clean": len(clean_fps),       # both in gold but pair condition mismatch
            "fp_phantom_pct": round(len(phantom_fps) / fp * 100, 1) if fp > 0 else 0.0,
            "fp_clean_pct": round(len(clean_fps) / fp * 100, 1) if fp > 0 else 0.0,
            # FN categories
            "fn_covered": len(covered_fns),   # protocol failure (both users in context)
            "fn_uncovered": len(uncovered_fns), # coverage failure
            "fn_covered_pct": round(len(covered_fns) / fn * 100, 1) if fn > 0 else 0.0,
            "fn_uncovered_pct": round(len(uncovered_fns) / fn * 100, 1) if fn > 0 else 0.0,
            # From existing experiment (for cross-validation)
            "existing_tp": tp_existing,
            "existing_fp": fp_existing,
            "existing_pairs": pairs_existing,
        })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="FP Root Cause Analysis (no API)")
    parser.add_argument("--output", type=str,
                        default="results/streaming/fp_analysis.json")
    parser.add_argument("--num-chunks", type=int, default=DEFAULT_NUM_CHUNKS)
    parser.add_argument("--chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS)
    parser.add_argument("--results-file", type=str, default=str(RESULTS_FILE))
    args = parser.parse_args()

    print("=" * 70)
    print("FP ROOT CAUSE ANALYSIS (no API required)")
    print("=" * 70)

    # Load data
    print("\nLoading OOLONG-Pairs data...")
    from eval.rlm_pipeline_experiment import (
        compute_gold_pairs,
        load_oolong_data,
    )
    plain_context, labeled_context = load_oolong_data()
    gold_pairs = compute_gold_pairs(labeled_context, task_idx=1)
    print(f"Gold pairs (Task 1): {len(gold_pairs)}")
    print(f"Plain context length: {len(plain_context)} chars")
    print(f"Labeled context length: {len(labeled_context)} chars")

    # Load existing F1 progression results (for cross-validation)
    results_path = Path(args.results_file)
    f1_results = {}
    if results_path.exists():
        with open(results_path) as f:
            f1_results = json.load(f)
        print(f"Loaded existing F1 progression results from {results_path}")
    else:
        print(f"WARNING: {results_path} not found — will compute from scratch (no cross-validation)")

    # Coverage ceiling analysis
    print("\n" + "=" * 70)
    print("COVERAGE CEILING ANALYSIS")
    print("(How many gold pairs have BOTH users in the first k*5K chars of plain context?)")
    print("=" * 70)

    coverage = analyze_coverage_ceiling(
        plain_context, gold_pairs, args.num_chunks, args.chunk_chars
    )
    print(f"\n{'k':>3} {'chars':>8} {'entities':>8} {'gold_covered':>12} {'coverage%':>10} {'F1_ceiling':>10}")
    print("-" * 55)
    for row in coverage:
        print(f"{row['k']:>3} {row['chars_seen']:>8} {row['entity_ids_in_plain']:>8} "
              f"{row['gold_pairs_covered']:>12} {row['coverage_pct']:>10.1f}% {row['f1_ceiling']:>10.4f}")

    max_coverage = coverage[-1]
    uncoverable_pct = 100 - max_coverage["coverage_pct"]
    print(f"\nKEY FINDING: At k=5 ({args.num_chunks * args.chunk_chars} chars), "
          f"{max_coverage['coverage_pct']:.1f}% of gold pairs are COVERED by plain context.")
    print(f"  → {uncoverable_pct:.1f}% of gold pairs are UNCOVERABLE with {args.num_chunks * args.chunk_chars} chars")
    print(f"  → F1 ceiling (perfect P+R up to coverage): {max_coverage['f1_ceiling']:.4f}")

    # FP categorization analysis
    print("\n" + "=" * 70)
    print("FP CATEGORIZATION ANALYSIS")
    print("(Phantom = user ID not in labeled gold; Clean = both in gold but condition mismatch)")
    print("=" * 70)

    fp_cats = analyze_fp_categories(
        plain_context, labeled_context, gold_pairs, f1_results,
        args.num_chunks, args.chunk_chars
    )

    print(f"\n{'k':>3} {'entities':>8} {'phantom_e':>9} {'pred_pairs':>10} "
          f"{'fp':>6} {'fp_phantom':>10} {'fp_clean':>8} {'precision':>9}")
    print("-" * 65)
    for row in fp_cats:
        print(f"{row['k']:>3} {row['entity_ids_in_plain']:>8} "
              f"{row['phantom_entity_ids']:>9} ({row['phantom_entity_pct']:.0f}%) "
              f"{row['predicted_pairs']:>10} "
              f"{row['fp']:>6} {row['fp_phantom']:>7} ({row['fp_phantom_pct']:.0f}%) "
              f"{row['fp_clean']:>6} ({row['fp_clean_pct']:.0f}%) "
              f"{row['precision']:>9.4f}")

    # FN categorization
    print(f"\n{'k':>3} {'fn':>6} {'fn_covered':>10} {'fn_uncovered':>12} {'recall':>8}")
    print("-" * 45)
    for row in fp_cats:
        print(f"{row['k']:>3} {row['fn']:>6} "
              f"{row['fn_covered']:>7} ({row['fn_covered_pct']:.0f}%) "
              f"{row['fn_uncovered']:>9} ({row['fn_uncovered_pct']:.0f}%) "
              f"{row['recall']:>8.4f}")

    # Summary diagnosis
    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)

    k5 = fp_cats[-1]
    print(f"\nAt k=5 ({args.num_chunks * args.chunk_chars} chars):")
    print(f"  Phantom entities (plain text over-extraction): {k5['phantom_entity_ids']} "
          f"({k5['phantom_entity_pct']:.1f}% of predicted entities)")
    print(f"  FPs from phantoms: {k5['fp_phantom']} ({k5['fp_phantom_pct']:.1f}% of all FPs)")
    print(f"  FPs from check_pair mismatch: {k5['fp_clean']} ({k5['fp_clean_pct']:.1f}% of all FPs)")
    print(f"  FNs from coverage (uncoverable): {k5['fn_uncovered']} ({k5['fn_uncovered_pct']:.1f}% of FNs)")
    print(f"  FNs from protocol (coverable but missed): {k5['fn_covered']} ({k5['fn_covered_pct']:.1f}% of FNs)")

    if k5["fp_phantom_pct"] > 50:
        dominant = "PHANTOM ENTITIES dominate FPs"
        fix = "Entity validation step: filter predicted user IDs to those matching labeled entity ID pattern"
    else:
        dominant = "check_pair CONDITION MISMATCH dominates FPs"
        fix = "More precise check_pair that matches gold label semantics"

    print(f"\n  Dominant FP source: {dominant}")
    print(f"  Recommended fix: {fix}")

    if k5["fn_uncovered_pct"] > 50:
        print(f"  Dominant FN source: COVERAGE LIMITATION ({k5['fn_uncovered_pct']:.0f}% of FNs)")
        print(f"  → F1=0.51 plateau is mostly coverage-bounded, not protocol-bounded")
        print(f"  → Protocol improvement alone cannot reach F1 > {coverage[-1]['f1_ceiling']:.3f}")
    else:
        print(f"  Dominant FN source: PROTOCOL FAILURE ({k5['fn_covered_pct']:.0f}% of FNs)")
        print(f"  → Protocol improvement could meaningfully raise F1 within current 25K budget")

    # Save results
    output = {
        "coverage_ceiling": coverage,
        "fp_categorization": fp_cats,
        "summary": {
            "gold_pairs": len(gold_pairs),
            "k5_phantom_entity_pct": k5["phantom_entity_pct"],
            "k5_fp_phantom_pct": k5["fp_phantom_pct"],
            "k5_fp_clean_pct": k5["fp_clean_pct"],
            "k5_fn_covered_pct": k5["fn_covered_pct"],
            "k5_fn_uncovered_pct": k5["fn_uncovered_pct"],
            "k5_f1_ceiling": coverage[-1]["f1_ceiling"],
            "k5_coverage_pct": coverage[-1]["coverage_pct"],
            "dominant_fp_source": dominant,
            "recommended_fix": fix,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
