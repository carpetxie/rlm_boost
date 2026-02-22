"""
Sigma Cost Model Analysis — Selectivity-Parameterized Savings Prediction.

This script:
1. Collects all simulation data points from results/streaming/ JSON files
   and hard-coded experiment logs.
2. Computes σ (selectivity) = final_pairs / C(231,2) = final_pairs / 26565
   for each task.
3. Fits two models:
   (a) σ-free:          savings(k)   = a * (1 - b/k)
   (b) σ-parameterized: savings(k,σ) = a * (1 - b/k) + c * σ * (1 - d/k)
4. Reports R² improvement and F-test for significance of σ term.
5. Runs per-entity retraction analysis for tasks 5 ("before DATE") and
   7 ("after DATE") at 5 chunks, tracking how many times each entity_id
   is retracted across chunks.

Usage:
  python eval/sigma_cost_model.py
  python eval/sigma_cost_model.py --output results/streaming/sigma_model_results.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import f as f_dist

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# C(231, 2) — total possible pairs for 231 entities
# ---------------------------------------------------------------------------
C231_2 = 26565  # C(231, 2) = 231 * 230 / 2


# ---------------------------------------------------------------------------
# 1. Data collection
# ---------------------------------------------------------------------------

def collect_data_points() -> list[dict]:
    """
    Collect (task_id, k, savings_pct, sigma) tuples from all sources.

    Sources:
    - Hard-coded Experiment 5 (Iteration 3) results
    - Hard-coded k=4 simulation results
    - Hard-coded 3-chunk temporal results
    - JSON files: incremental_temporal_5chunks.json, incremental_temporal_10chunks.json

    Returns list of dicts with keys: task_id, k, savings_pct, sigma
    """

    # σ values: final_pairs / 26565
    SIGMA = {
        1:  8001 / C231_2,   # 0.3012
        3:  10440 / C231_2,  # 0.3930
        4:  990 / C231_2,    # 0.03726
        5:  21 / C231_2,     # 0.000790
        6:  8911 / C231_2,   # 0.3354
        7:  1485 / C231_2,   # 0.05591
        9:  741 / C231_2,    # 0.02789
        10: 190 / C231_2,    # 0.007153
        11: 689 / C231_2,    # 0.02594
        13: 1524 / C231_2,   # 0.05736
        19: 60 / C231_2,     # 0.002259
    }

    # Hard-coded experiment data points (task_id, k, savings_pct)
    # From Experiment 5 (Iteration 3) results — Experiment 5 tasks
    exp5_data = [
        # Task 1: k=3→9.8%, k=5→22.1%, k=10→42.0%
        (1, 3, 9.8),
        (1, 5, 22.1),
        (1, 10, 42.0),
        # Task 3: k=5→22.1%, k=10→42.3%
        (3, 5, 22.1),
        (3, 10, 42.3),
        # Task 6: k=3→10.3%, k=5→22.2%, k=10→42.2%
        (6, 3, 10.3),
        (6, 5, 22.2),
        (6, 10, 42.2),
        # Task 11: k=5→17.1%, k=10→39.1%
        (11, 5, 17.1),
        (11, 10, 39.1),
        # Task 13: k=5→17.4%, k=10→39.4%
        (13, 5, 17.4),
        (13, 10, 39.4),
        # Task 19: k=3→4.1%, k=4→9.6%, k=5→16.7%, k=10→38.8%
        (19, 3, 4.1),
        (19, 4, 9.6),
        (19, 5, 16.7),
        (19, 10, 38.8),
    ]

    # From k=4 simulation
    k4_data = [
        (1, 4, 15.2),
        (3, 4, 15.8),
        (6, 4, 15.7),
        (11, 4, 10.1),
        (19, 4, 9.6),
    ]

    # From 3-chunk temporal simulation
    k3_temporal_data = [
        (4, 3, 5.0),
        (5, 3, 4.1),
        (7, 3, 5.1),
        (9, 3, 4.9),
        (10, 3, 4.2),
    ]

    # Read k=5 temporal results from JSON
    json_5chunk_path = Path(__file__).parent.parent / "results/streaming/incremental_temporal_5chunks.json"
    json_5chunk_data = []
    if json_5chunk_path.exists():
        with open(json_5chunk_path) as f:
            data = json.load(f)
        for task_str, task_result in data.items():
            task_id = int(task_str)
            savings_pct = task_result["savings"]["pair_check_pct"]
            json_5chunk_data.append((task_id, 5, savings_pct))
        print(f"Loaded {len(json_5chunk_data)} data points from {json_5chunk_path.name}")

    # Read k=10 temporal results from JSON
    json_10chunk_path = Path(__file__).parent.parent / "results/streaming/incremental_temporal_10chunks.json"
    json_10chunk_data = []
    if json_10chunk_path.exists():
        with open(json_10chunk_path) as f:
            data = json.load(f)
        for task_str, task_result in data.items():
            task_id = int(task_str)
            savings_pct = task_result["savings"]["pair_check_pct"]
            json_10chunk_data.append((task_id, 10, savings_pct))
        print(f"Loaded {len(json_10chunk_data)} data points from {json_10chunk_path.name}")

    # Combine all data sources, deduplicating by (task_id, k) — JSON data takes precedence
    # over hard-coded data for the same (task_id, k) pair.
    all_raw = exp5_data + k4_data + k3_temporal_data
    json_raw = json_5chunk_data + json_10chunk_data

    # Build lookup: (task_id, k) -> savings_pct (JSON overrides hard-coded)
    lookup: dict[tuple[int, int], float] = {}
    for task_id, k, savings in all_raw:
        lookup[(task_id, k)] = savings
    for task_id, k, savings in json_raw:
        lookup[(task_id, k)] = savings  # JSON takes precedence

    # Assemble final data points with σ values
    data_points = []
    for (task_id, k), savings_pct in sorted(lookup.items()):
        if task_id not in SIGMA:
            print(f"  WARNING: No σ for task {task_id}, skipping")
            continue
        data_points.append({
            "task_id": task_id,
            "k": k,
            "savings_pct": savings_pct,
            "sigma": SIGMA[task_id],
        })

    return data_points


# ---------------------------------------------------------------------------
# 2. Model fitting
# ---------------------------------------------------------------------------

def model_sigma_free(k, a, b):
    """σ-free model: savings(k) = a * (1 - b/k)"""
    return a * (1.0 - b / k)


def model_sigma_param(X, a, b, c, e):
    """σ-parameterized model: savings(k, σ) = a*(1-b/k) + c*σ*(1+e/k), all params positive.

    Reparameterized from the original (1-d/k) form where d went negative (d=-1.60).
    The optimizer drove d negative because high-σ tasks show LARGER savings at small k,
    requiring the σ term to grow as k→0. With d=-1.60, (1-d/k) = (1+1.60/k).

    This e-parameterization makes the sign convention explicit and unambiguous:
    e = |d| > 0, with bounds enforced. The paper formula is:
        savings(k, σ) = a·(1 - b/k) + c·σ·(1 + e/k)
    All four parameters (a, b, c, e) are non-negative.
    """
    k, sigma = X
    return a * (1.0 - b / k) + c * sigma * (1.0 + e / k)


def r_squared(y_true, y_pred):
    """Compute R² coefficient of determination."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def f_test_r2_improvement(r2_full, r2_reduced, n, p_full, p_reduced):
    """
    F-test for improvement in R² by adding extra parameters.

    H0: the additional parameters do not improve fit.
    Returns (F_statistic, p_value).
    """
    if r2_reduced >= 1.0 or p_full == p_reduced:
        return np.nan, np.nan
    f_stat = ((r2_full - r2_reduced) / (p_full - p_reduced)) / (
        (1 - r2_full) / (n - p_full)
    )
    p_value = 1.0 - f_dist.cdf(f_stat, p_full - p_reduced, n - p_full)
    return f_stat, p_value


def fit_models(data_points: list[dict]) -> dict:
    """Fit both models and return results dict."""

    k_arr = np.array([d["k"] for d in data_points], dtype=float)
    savings_arr = np.array([d["savings_pct"] for d in data_points], dtype=float)
    sigma_arr = np.array([d["sigma"] for d in data_points], dtype=float)

    n = len(data_points)

    print(f"\n{'=' * 70}")
    print("MODEL FITTING")
    print(f"{'=' * 70}")
    print(f"Total data points: {n}")

    # ---- Model (a): σ-free ----
    try:
        popt_free, pcov_free = curve_fit(
            model_sigma_free,
            k_arr,
            savings_arr,
            p0=[50.0, 1.0],
            bounds=([0, 0], [200, 10]),
            maxfev=10000,
        )
        a_free, b_free = popt_free
        pred_free = model_sigma_free(k_arr, *popt_free)
        r2_free = r_squared(savings_arr, pred_free)
        rmse_free = np.sqrt(np.mean((savings_arr - pred_free) ** 2))
        fit_free_ok = True
    except Exception as e:
        print(f"WARNING: σ-free model fit failed: {e}")
        a_free = b_free = np.nan
        r2_free = rmse_free = np.nan
        pred_free = np.full_like(savings_arr, np.nan)
        fit_free_ok = False

    # ---- Model (b): σ-parameterized ----
    # Note: reparameterized to use e>0 explicitly, where e = |d_old|.
    # Old form: c*σ*(1-d/k) with d=-1.60 → (1+1.60/k).
    # New form: c*σ*(1+e/k) with e=+1.60 → same result, unambiguous sign.
    # Bounds enforce all parameters non-negative.
    try:
        popt_param, pcov_param = curve_fit(
            model_sigma_param,
            (k_arr, sigma_arr),
            savings_arr,
            p0=[45.0, 1.0, 20.0, 1.0],
            bounds=([0, 0, 0, 0], [200, 10, 500, 10]),
            maxfev=20000,
        )
        a_p, b_p, c_p, e_p = popt_param
        pred_param = model_sigma_param((k_arr, sigma_arr), *popt_param)
        r2_param = r_squared(savings_arr, pred_param)
        rmse_param = np.sqrt(np.mean((savings_arr - pred_param) ** 2))
        fit_param_ok = True
    except Exception as exc:
        print(f"WARNING: σ-parameterized model fit failed: {exc}")
        a_p = b_p = c_p = e_p = np.nan
        r2_param = rmse_param = np.nan
        pred_param = np.full_like(savings_arr, np.nan)
        fit_param_ok = False

    # ---- F-test ----
    f_stat = p_value = np.nan
    if fit_free_ok and fit_param_ok:
        f_stat, p_value = f_test_r2_improvement(r2_param, r2_free, n, 4, 2)

    # ---- Report ----
    print(f"\n(a) σ-free model: savings(k) = a * (1 - b/k)")
    if fit_free_ok:
        print(f"    a = {a_free:.4f},  b = {b_free:.4f}")
        print(f"    R² = {r2_free:.4f},  RMSE = {rmse_free:.3f}%")
    else:
        print("    FIT FAILED")

    print(f"\n(b) σ-parameterized model: savings(k,σ) = a*(1-b/k) + c*σ*(1+e/k)  [all params ≥ 0]")
    if fit_param_ok:
        print(f"    a = {a_p:.4f},  b = {b_p:.4f},  c = {c_p:.4f},  e = {e_p:.4f}")
        print(f"    R² = {r2_param:.4f},  RMSE = {rmse_param:.3f}%")
    else:
        print("    FIT FAILED")

    print(f"\nR² improvement: {r2_param - r2_free:+.4f} (σ-param vs σ-free)")
    if not np.isnan(f_stat):
        significance = "SIGNIFICANT" if p_value < 0.05 else "NOT significant"
        print(f"F-test: F({2}, {n - 4}) = {f_stat:.3f},  p = {p_value:.4f}  → σ term is {significance} (α=0.05)")

    # Per-point residuals
    print(f"\n{'Task':>5} {'k':>3} {'σ':>8} {'Actual':>8} {'PredFree':>9} {'PredParam':>10} {'ResFree':>8} {'ResParam':>9}")
    print("-" * 72)
    for i, d in enumerate(data_points):
        task_id = d["task_id"]
        k_v = d["k"]
        sigma_v = d["sigma"]
        actual = d["savings_pct"]
        pred_f = pred_free[i] if fit_free_ok else float("nan")
        pred_p = pred_param[i] if fit_param_ok else float("nan")
        print(
            f"{task_id:>5} {k_v:>3} {sigma_v:>8.5f} {actual:>8.2f}%"
            f" {pred_f:>8.2f}% {pred_p:>9.2f}%"
            f" {actual - pred_f:>+7.2f}% {actual - pred_p:>+8.2f}%"
        )

    return {
        "n_points": n,
        "model_free": {
            "formula": "savings(k) = a*(1 - b/k)",
            "params": {"a": float(a_free), "b": float(b_free)},
            "r_squared": float(r2_free),
            "rmse_pct": float(rmse_free),
        },
        "model_sigma": {
            "formula": "savings(k,sigma) = a*(1-b/k) + c*sigma*(1+e/k)  [all params >= 0]",
            "params": {"a": float(a_p), "b": float(b_p), "c": float(c_p), "e": float(e_p)},
            "r_squared": float(r2_param),
            "rmse_pct": float(rmse_param),
        },
        "r2_improvement": float(r2_param - r2_free),
        "f_test": {
            "F_statistic": float(f_stat) if not np.isnan(f_stat) else None,
            "p_value": float(p_value) if not np.isnan(p_value) else None,
            "df_numerator": 2,
            "df_denominator": n - 4,
            "significant_at_0.05": bool(p_value < 0.05) if not np.isnan(p_value) else None,
        },
    }


# ---------------------------------------------------------------------------
# 3. Per-entity retraction analysis
# ---------------------------------------------------------------------------

def load_simulation_data():
    """Load the corpus from the HuggingFace dataset."""
    from datasets import load_dataset
    ds = load_dataset("oolongbench/oolong-synth", split="validation")
    corpus = [
        x for x in ds
        if x["dataset"] == "trec_coarse" and x["context_len"] == 32768
    ][0]
    return corpus["context_window_text"], corpus["context_window_text_with_labels"]


def parse_users_from_labeled(labeled_text: str) -> dict:
    """Parse labeled context into {user_id: [{"date": datetime|None, "label": str}]}."""
    from eval.utils import _parse_labeled_context
    return _parse_labeled_context(labeled_text)


def split_labeled_context(labeled_context: str, num_chunks: int) -> list[str]:
    """Split labeled context into chunks at user boundaries."""
    user_pattern = re.compile(r"(?=Date:.*?\|\| User:)")
    positions = [m.start() for m in user_pattern.finditer(labeled_context)]

    if not positions:
        chunk_size = len(labeled_context) // num_chunks
        return [
            labeled_context[
                i * chunk_size : (i + 1) * chunk_size
                if i < num_chunks - 1
                else len(labeled_context)
            ]
            for i in range(num_chunks)
        ]

    users_per_chunk = max(1, len(positions) // num_chunks)
    chunks = []
    for i in range(num_chunks):
        start = (
            positions[i * users_per_chunk]
            if i * users_per_chunk < len(positions)
            else len(labeled_context)
        )
        if i < num_chunks - 1:
            end_idx = min((i + 1) * users_per_chunk, len(positions))
            end = positions[end_idx] if end_idx < len(positions) else len(labeled_context)
        else:
            end = len(labeled_context)
        if start < end:
            chunks.append(labeled_context[start:end])

    while len(chunks) < num_chunks:
        chunks.append("")
    return chunks[:num_chunks]


class EntityRetractionTracker:
    """
    Wrapper around IncrementalState that tracks per-entity retraction counts.

    Instruments the PairTracker.retract_entity() calls to count how many
    times each entity_id has been retracted across all chunks.
    """

    def __init__(self):
        from rlm.core.incremental import IncrementalState
        self.state = IncrementalState()
        # Per-entity retraction count: entity_id -> retraction_count
        self.entity_retraction_counts: dict = defaultdict(int)
        self._original_retract = self.state.pair_tracker.retract_entity
        self._patch_retract()

    def _patch_retract(self):
        """Monkey-patch retract_entity to track per-entity counts."""
        tracker_ref = self

        def tracked_retract(entity_id: str):
            affected = tracker_ref._original_retract(entity_id)
            if affected:  # Only count if there were actual pairs to retract
                tracker_ref.entity_retraction_counts[entity_id] += 1
            return affected

        self.state.pair_tracker.retract_entity = tracked_retract

    def process_chunk(self, chunk_index, new_entities, pair_checker=None):
        return self.state.process_chunk(chunk_index, new_entities, pair_checker=pair_checker)

    def get_stats(self):
        return self.state.get_stats()

    @property
    def pair_tracker(self):
        return self.state.pair_tracker

    @property
    def chunk_log(self):
        return self.state.chunk_log


def run_per_entity_retraction_analysis(
    labeled_context: str,
    task_indices: list[int],
    num_chunks: int,
) -> dict:
    """
    Run per-entity retraction analysis for specified tasks.

    For each task, tracks how many times each entity_id is retracted
    across all chunk processings. Returns per-entity retraction distributions.
    """
    from eval.utils import _check_pair_condition

    def make_checker(task_idx):
        def checker(attrs1, attrs2):
            return _check_pair_condition(attrs1["instances"], attrs2["instances"], task_idx)
        return checker

    chunks = split_labeled_context(labeled_context, num_chunks)
    results = {}

    for task_idx in task_indices:
        print(f"\n{'=' * 60}")
        print(f"Per-Entity Retraction Analysis — Task {task_idx}")
        print(f"{'=' * 60}")

        checker = make_checker(task_idx)
        tracker = EntityRetractionTracker()

        all_user_instances: dict = {}

        for chunk_i, chunk in enumerate(chunks):
            chunk_users = parse_users_from_labeled(chunk)

            chunk_entities = {}
            for uid, instances in chunk_users.items():
                if uid in all_user_instances:
                    merged = all_user_instances[uid] + instances
                    all_user_instances[uid] = merged
                    chunk_entities[uid] = {"instances": merged}
                else:
                    all_user_instances[uid] = instances
                    chunk_entities[uid] = {"instances": instances}

            chunk_stats = tracker.process_chunk(chunk_i, chunk_entities, pair_checker=checker)
            print(
                f"  Chunk {chunk_i + 1}/{num_chunks}: "
                f"new={chunk_stats['new_entities']}, "
                f"updated={chunk_stats['updated_entities']}, "
                f"retracted_pairs={chunk_stats['retracted_pairs']}, "
                f"total_pairs={chunk_stats['total_pairs']}"
            )

        # Analyze per-entity retraction distribution
        entity_counts = dict(tracker.entity_retraction_counts)

        # Distribution analysis
        if entity_counts:
            max_retractions = max(entity_counts.values())
            counts_by_bucket = defaultdict(int)
            for count in entity_counts.values():
                counts_by_bucket[count] += 1

            # Entities with 0 retractions (never retracted)
            total_entities = len(all_user_instances)
            n_retracted = len(entity_counts)
            n_zero = total_entities - n_retracted

            # Bidirectional: entities retracted more than once (max > 1)
            n_bidirectional = sum(1 for c in entity_counts.values() if c > 1)
            n_unidirectional = sum(1 for c in entity_counts.values() if c == 1)

            print(f"\n  Entity Retraction Distribution:")
            print(f"    Total entities: {total_entities}")
            print(f"    Never retracted (0×): {n_zero}")
            print(f"    Retracted 1×: {n_unidirectional}")
            print(f"    Retracted 2+× (bidirectional): {n_bidirectional}")
            print(f"    Max retractions for one entity: {max_retractions}")
            if entity_counts:
                avg_retractions = sum(entity_counts.values()) / len(entity_counts)
                print(f"    Avg retractions (among retracted entities): {avg_retractions:.2f}")

            print(f"\n  Retraction count histogram:")
            for bucket in sorted(counts_by_bucket.keys()):
                bar = "#" * counts_by_bucket[bucket]
                print(f"    {bucket:>2}×: {counts_by_bucket[bucket]:>3} entities  {bar}")

            # Top 10 most retracted entities
            top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            print(f"\n  Top retracted entities:")
            for eid, cnt in top_entities:
                print(f"    entity {eid}: {cnt} retraction(s)")

        else:
            max_retractions = 0
            n_zero = len(all_user_instances)
            n_bidirectional = 0
            n_unidirectional = 0
            avg_retractions = 0.0
            counts_by_bucket = {}
            print(f"    No entities were retracted.")

        overall_stats = tracker.get_stats()
        results[task_idx] = {
            "task_id": task_idx,
            "num_chunks": num_chunks,
            "total_entities": len(all_user_instances),
            "final_pairs": len(tracker.pair_tracker.get_pairs()),
            "total_retractions": overall_stats["total_retractions"],
            "per_entity_retraction_counts": {
                str(k): v for k, v in entity_counts.items()
            },
            "distribution": {
                "never_retracted": len(all_user_instances) - len(entity_counts),
                "retracted_once": sum(1 for c in entity_counts.values() if c == 1),
                "retracted_2plus": sum(1 for c in entity_counts.values() if c > 1),
                "max_retractions": max(entity_counts.values()) if entity_counts else 0,
                "histogram": {str(k): v for k, v in sorted(counts_by_bucket.items())},
            },
        }

    return results


def compare_retraction_distributions(results: dict, task_before: int, task_after: int) -> dict:
    """
    Compare retraction distributions between a 'before DATE' task and an 'after DATE' task.
    """
    r_before = results.get(task_before)
    r_after = results.get(task_after)

    if not r_before or not r_after:
        print("WARNING: Missing results for comparison")
        return {}

    print(f"\n{'=' * 70}")
    print(f"COMPARISON: Task {task_before} ('before DATE') vs Task {task_after} ('after DATE')")
    print(f"{'=' * 70}")

    def pct(a, b):
        return f"{100 * a / b:.1f}%" if b > 0 else "N/A"

    for label, r in [(f"Task {task_before} (before DATE)", r_before),
                     (f"Task {task_after} (after DATE)", r_after)]:
        d = r["distribution"]
        n = r["total_entities"]
        print(f"\n  {label}:")
        print(f"    Total entities:           {n}")
        print(f"    Final pairs:              {r['final_pairs']}")
        print(f"    Total retractions:        {r['total_retractions']}")
        print(f"    Never retracted:          {d['never_retracted']} ({pct(d['never_retracted'], n)})")
        print(f"    Retracted 1× (unidirect): {d['retracted_once']} ({pct(d['retracted_once'], n)})")
        print(f"    Retracted 2+× (bidir):    {d['retracted_2plus']} ({pct(d['retracted_2plus'], n)})")
        print(f"    Max retractions:          {d['max_retractions']}")

    # Analysis: do 'after DATE' tasks show more bidirectional retractions?
    before_bidir_frac = r_before["distribution"]["retracted_2plus"] / r_before["total_entities"]
    after_bidir_frac = r_after["distribution"]["retracted_2plus"] / r_after["total_entities"]

    print(f"\n  Bidirectional retraction fraction (max>1):")
    print(f"    Task {task_before} (before DATE): {100*before_bidir_frac:.2f}%")
    print(f"    Task {task_after} (after DATE):  {100*after_bidir_frac:.2f}%")

    if after_bidir_frac > before_bidir_frac:
        print(f"    → 'after DATE' task ({task_after}) shows MORE bidirectional retractions")
        print(f"      ({100*after_bidir_frac:.2f}% vs {100*before_bidir_frac:.2f}%)")
        hypothesis_supported = True
    elif after_bidir_frac < before_bidir_frac:
        print(f"    → 'before DATE' task ({task_before}) shows MORE bidirectional retractions")
        hypothesis_supported = False
    else:
        print(f"    → Equal bidirectional retraction rates")
        hypothesis_supported = False

    # Interpretation
    print(f"\n  Interpretation:")
    print(f"    'Before DATE' tasks (e.g., task 5): entities that appear after the cutoff")
    print(f"    date invalidate the condition, causing pair retractions. Once an entity")
    print(f"    is disqualified by a late-date instance, it stays disqualified — so")
    print(f"    retractions are largely monotone (entities retract once).")
    print(f"")
    print(f"    'After DATE' tasks (e.g., task 7): entities must have instances AFTER")
    print(f"    the cutoff. As new chunks arrive, early-chunk entities may initially")
    print(f"    qualify (with only pre-cutoff instances), then become disqualified when")
    print(f"    pre-cutoff instances appear, then re-qualify when post-cutoff instances")
    print(f"    arrive — creating more volatile, multi-directional retractions.")

    return {
        "task_before": task_before,
        "task_after": task_after,
        "before_bidir_fraction": before_bidir_frac,
        "after_bidir_fraction": after_bidir_frac,
        "hypothesis_supported": hypothesis_supported,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sigma Cost Model Analysis")
    parser.add_argument(
        "--output",
        default="results/streaming/sigma_model_results.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--skip-retraction",
        action="store_true",
        help="Skip per-entity retraction analysis (faster, no dataset download needed)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("SIGMA COST MODEL — Selectivity-Parameterized Savings Analysis")
    print("=" * 70)

    # Step 1: Collect data
    print("\n[1] Collecting data points...")
    data_points = collect_data_points()
    print(f"\nData point summary ({len(data_points)} total):")
    print(f"{'Task':>5} {'k':>3} {'σ':>8} {'Savings%':>9}")
    print("-" * 35)
    for d in sorted(data_points, key=lambda x: (x["task_id"], x["k"])):
        print(f"{d['task_id']:>5} {d['k']:>3} {d['sigma']:>8.5f} {d['savings_pct']:>8.1f}%")

    # Step 2: Fit models
    print(f"\n[2] Fitting cost models...")
    model_results = fit_models(data_points)

    # Step 3: Per-entity retraction analysis
    retraction_results = {}
    comparison_result = {}

    if not args.skip_retraction:
        print(f"\n[3] Per-entity retraction analysis (tasks 5 and 7, k=5)...")
        try:
            _, labeled_context = load_simulation_data()
            retraction_results = run_per_entity_retraction_analysis(
                labeled_context,
                task_indices=[5, 7],
                num_chunks=5,
            )
            comparison_result = compare_retraction_distributions(retraction_results, 5, 7)
        except Exception as e:
            print(f"  WARNING: Retraction analysis failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n[3] Skipping per-entity retraction analysis (--skip-retraction flag set)")

    # Step 4: Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "data_points": data_points,
        "model_results": model_results,
        "retraction_analysis": {
            str(k): v for k, v in retraction_results.items()
        },
        "retraction_comparison": comparison_result,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Final summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    mr = model_results
    print(f"  Data points: {mr['n_points']}")
    print(f"\n  Model (a) — σ-free:")
    print(f"    savings(k) = {mr['model_free']['params']['a']:.3f} * (1 - {mr['model_free']['params']['b']:.3f}/k)")
    print(f"    R² = {mr['model_free']['r_squared']:.4f}")
    print(f"\n  Model (b) — σ-parameterized:")
    p = mr['model_sigma']['params']
    print(f"    savings(k,σ) = {p['a']:.3f}*(1-{p['b']:.3f}/k) + {p['c']:.3f}*σ*(1+{p['e']:.3f}/k)")
    print(f"    R² = {mr['model_sigma']['r_squared']:.4f}")
    print(f"\n  R² improvement: {mr['r2_improvement']:+.4f}")
    ft = mr["f_test"]
    if ft["F_statistic"] is not None:
        sig_str = "YES (p<0.05)" if ft["significant_at_0.05"] else "NO (p≥0.05)"
        print(f"  σ adds significant predictive power: {sig_str}")
        print(f"  F({ft['df_numerator']}, {ft['df_denominator']}) = {ft['F_statistic']:.3f},  p = {ft['p_value']:.4f}")

    if comparison_result:
        print(f"\n  Per-entity retraction (tasks 5 vs 7, k=5):")
        print(f"    Task 5 (before DATE) bidirectional fraction: {100*comparison_result['before_bidir_fraction']:.2f}%")
        print(f"    Task 7 (after DATE)  bidirectional fraction: {100*comparison_result['after_bidir_fraction']:.2f}%")
        supp = comparison_result.get("hypothesis_supported")
        print(f"    Hypothesis ('after DATE' shows more bidirectional): {'SUPPORTED' if supp else 'NOT SUPPORTED'}")


if __name__ == "__main__":
    main()
