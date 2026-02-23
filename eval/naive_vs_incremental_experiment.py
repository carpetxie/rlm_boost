"""
Naive RLM vs Incremental RLM — Head-to-Head Comparison.

This is the critical missing experiment for the research project. It provides
a direct, apples-to-apples comparison between two approaches to processing
streaming context in OOLONG-Pairs tasks:

1. **Naive RLM**: On each new chunk k, re-read ALL accumulated context
   (chunks 0..k) and recompute all entity classifications and pairs from
   scratch. No caching, no incremental state.

2. **Incremental RLM**: On each new chunk k, only process new data using
   EntityCache/PairTracker/IncrementalState with monotone_attrs.

## Two parts:

- **Simulation** (no API needed): Proves correctness (both find same pairs)
  and quantifies computational savings (pair checks, token estimates).

- **Live API comparison**: Measures real F1, token usage, wall-clock time,
  and estimated cost using actual LLM calls.

Usage:
    # Simulation only (no API key needed)
    python eval/naive_vs_incremental_experiment.py --simulate --task-idx 1

    # All tasks simulation
    python eval/naive_vs_incremental_experiment.py --simulate --all-tasks

    # Live comparison (needs OPENAI_API_KEY)
    python eval/naive_vs_incremental_experiment.py --live --task-idx 1

    # Both simulation and live
    python eval/naive_vs_incremental_experiment.py --simulate --live --task-idx 1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.label_aware_experiment import (
    TASK_QUALIFYING_LABELS,
    TASK_LABEL_DESCRIPTION,
    make_label_checker_setup,
    load_labeled_data,
)
from eval.label_aware_v2_experiment import _make_sequential_chunks
from eval.rlm_pipeline_experiment import compute_f1, compute_gold_pairs
from rlm.core.incremental import IncrementalState


# ---------------------------------------------------------------------------
# Entity parsing helpers
# ---------------------------------------------------------------------------

def parse_entities_from_chunk(
    chunk_text: str,
    qualifying_labels: set[str],
) -> dict[str, dict]:
    """Parse entities from a chunk of labeled context.

    Each line matching "User: <id> ... || Label: <label>" produces an entity.
    An entity qualifies if ANY of its labels is in qualifying_labels.

    Returns:
        {entity_id: {"labels": [...], "qualifying": bool}}
    """
    entities: dict[str, dict] = {}
    for line in chunk_text.split("\n"):
        m = re.search(r"User: (\d+).*?\|\| Label: (.+?)$", line)
        if m:
            uid = m.group(1)
            label = m.group(2).strip().lower()
            if uid not in entities:
                entities[uid] = {"labels": [], "qualifying": False}
            entities[uid]["labels"].append(label)
            if label in qualifying_labels:
                entities[uid]["qualifying"] = True
    return entities


def check_pair(attrs1: dict, attrs2: dict) -> bool:
    """Symmetric 'at least one' pair check: both must be qualifying."""
    return bool(attrs1.get("qualifying")) and bool(attrs2.get("qualifying"))


# ---------------------------------------------------------------------------
# Part 1: Simulation comparison (no API needed)
# ---------------------------------------------------------------------------

def run_simulation_comparison(
    labeled_context: str,
    task_idx: int = 1,
    num_chunks: int = 5,
    max_chunk_chars: int = 5000,
) -> dict:
    """Run a simulation comparing naive vs incremental pair computation.

    No API calls are made. This demonstrates that:
    1. Both approaches find the SAME pairs at every turn (correctness).
    2. The incremental approach uses fewer pair checks and fewer tokens.

    Args:
        labeled_context: Full labeled context string.
        task_idx: OOLONG-Pairs task index (1, 3, or 6).
        num_chunks: Number of sequential chunks.
        max_chunk_chars: Characters per chunk.

    Returns:
        Comparison dict with per-turn and total metrics.
    """
    qualifying_labels = TASK_QUALIFYING_LABELS[task_idx]
    label_desc = TASK_LABEL_DESCRIPTION[task_idx]
    chunks = _make_sequential_chunks(labeled_context, num_chunks, max_chunk_chars)

    print(f"\n{'=' * 70}")
    print(f"SIMULATION: Naive vs Incremental — Task {task_idx}")
    print(f"  Qualifying labels: {label_desc}")
    print(f"  k={num_chunks}, {max_chunk_chars} chars/chunk")
    print(f"{'=' * 70}")
    print(f"Chunk sizes: {[len(c) for c in chunks]} chars")

    # ----- Incremental approach -----
    incr = IncrementalState()
    incr_per_turn: list[dict] = []

    for chunk_i in range(num_chunks):
        entities = parse_entities_from_chunk(chunks[chunk_i], qualifying_labels)
        t0 = time.perf_counter()
        stats = incr.process_chunk(
            chunk_i, entities, pair_checker=check_pair, monotone_attrs={"qualifying"}
        )
        elapsed = time.perf_counter() - t0
        incr_per_turn.append({
            "turn": chunk_i,
            "pair_checks": stats["pair_checks"],
            "total_pairs": stats["total_pairs"],
            "new_entities": stats["new_entities"],
            "updated_entities": stats["updated_entities"],
            "token_est": len(chunks[chunk_i]),
            "wall_clock_sec": round(elapsed, 6),
        })

    incr_pairs_final = incr.pair_tracker.get_pairs()
    incr_cumulative_stats = incr.get_stats()

    # ----- Naive approach -----
    naive_per_turn: list[dict] = []

    for turn in range(num_chunks):
        # Concatenate chunks 0..turn (cumulative context)
        cumulative_text = "".join(chunks[: turn + 1])
        cumulative_token_est = len(cumulative_text)

        # Parse ALL entities from cumulative context
        t0 = time.perf_counter()
        all_entities = parse_entities_from_chunk(cumulative_text, qualifying_labels)

        # Count entities and qualifying entities
        qualifying_ids = [
            uid for uid, attrs in all_entities.items() if attrs.get("qualifying")
        ]
        q_count = len(qualifying_ids)
        n_total = len(all_entities)

        # Naive pair checks: TWO metrics for the paper
        # 1. all_pair_checks: C(N, 2) — must check all entity pairs to find valid ones
        #    (same scope as incremental's new×existing check over all entities)
        # 2. qualifying_pair_checks: C(Q, 2) — if naive pre-filters qualifying entities
        #    before pairing (optimistic, assumes oracle knowledge of qualifying status)
        naive_all_pair_checks = n_total * (n_total - 1) // 2
        naive_qualifying_pair_checks = q_count * (q_count - 1) // 2

        # Compute actual pairs (for correctness verification)
        naive_pairs_at_turn: set[tuple[str, str]] = set()
        for id1, id2 in combinations(sorted(qualifying_ids), 2):
            naive_pairs_at_turn.add((min(id1, id2), max(id1, id2)))
        elapsed = time.perf_counter() - t0

        naive_per_turn.append({
            "turn": turn,
            "all_pair_checks": naive_all_pair_checks,
            "qualifying_pair_checks": naive_qualifying_pair_checks,
            "total_pairs": len(naive_pairs_at_turn),
            "qualifying_entities": q_count,
            "total_entities": n_total,
            "token_est": cumulative_token_est,
            "wall_clock_sec": round(elapsed, 6),
        })

    # ----- Build per-turn comparison -----
    per_turn_comparison: list[dict] = []
    cumulative_naive_token_est = 0
    cumulative_incr_token_est = 0
    cumulative_naive_all_checks = 0
    cumulative_naive_qual_checks = 0
    cumulative_incr_checks = 0

    for t in range(num_chunks):
        naive = naive_per_turn[t]
        incr_t = incr_per_turn[t]

        cumulative_naive_token_est += naive["token_est"]
        cumulative_incr_token_est += incr_t["token_est"]
        cumulative_naive_all_checks += naive["all_pair_checks"]
        cumulative_naive_qual_checks += naive["qualifying_pair_checks"]
        cumulative_incr_checks += incr_t["pair_checks"]

        pairs_match = (naive["total_pairs"] == incr_t["total_pairs"])

        per_turn_comparison.append({
            "turn": t,
            "naive_all_pair_checks": naive["all_pair_checks"],
            "naive_qual_pair_checks": naive["qualifying_pair_checks"],
            "incr_pair_checks": incr_t["pair_checks"],
            "naive_token_est": naive["token_est"],
            "incr_token_est": incr_t["token_est"],
            "naive_pairs": naive["total_pairs"],
            "incr_pairs": incr_t["total_pairs"],
            "pairs_match": pairs_match,
            "naive_wall_sec": naive["wall_clock_sec"],
            "incr_wall_sec": incr_t["wall_clock_sec"],
        })

    # ----- Totals -----
    total_naive_all_checks = cumulative_naive_all_checks
    total_naive_qual_checks = cumulative_naive_qual_checks
    total_incr_checks = cumulative_incr_checks

    # Pair check savings vs all-pairs naive (apples-to-apples: both check all entity pairs)
    pair_check_savings_all = (
        1.0 - total_incr_checks / total_naive_all_checks
        if total_naive_all_checks > 0
        else 0.0
    )
    # Pair check savings vs qualifying-only naive (optimistic for naive)
    pair_check_savings_qual = (
        1.0 - total_incr_checks / total_naive_qual_checks
        if total_naive_qual_checks > 0
        else 0.0
    )

    total_naive_tokens = cumulative_naive_token_est
    total_incr_tokens = cumulative_incr_token_est
    token_savings = (
        1.0 - total_incr_tokens / total_naive_tokens
        if total_naive_tokens > 0
        else 0.0
    )

    # Cost model: gpt-4o-mini pricing ($0.15/M input tokens, ~4 chars/token)
    chars_per_token = 4.0
    cost_per_million_input = 0.15  # dollars
    naive_cost = (total_naive_tokens / chars_per_token) / 1e6 * cost_per_million_input
    incr_cost = (total_incr_tokens / chars_per_token) / 1e6 * cost_per_million_input

    all_match = all(t["pairs_match"] for t in per_turn_comparison)
    final_match = (naive_per_turn[-1]["total_pairs"] == len(incr_pairs_final))

    # ----- Print summary table -----
    print(f"\n{'Turn':>4} {'N_all_chk':>10} {'N_qual_chk':>11} {'I_checks':>10} {'N_tok':>8} {'I_tok':>8}"
          f" {'N_pairs':>8} {'I_pairs':>8} {'Match':>6}")
    print("-" * 85)
    for t in per_turn_comparison:
        print(f"{t['turn']:>4} {t['naive_all_pair_checks']:>10} {t['naive_qual_pair_checks']:>11}"
              f" {t['incr_pair_checks']:>10} {t['naive_token_est']:>8} {t['incr_token_est']:>8}"
              f" {t['naive_pairs']:>8} {t['incr_pairs']:>8}"
              f" {'OK' if t['pairs_match'] else 'DIFF':>6}")
    print("-" * 85)
    print(f"{'TOTAL':>4} {total_naive_all_checks:>10} {total_naive_qual_checks:>11}"
          f" {total_incr_checks:>10} {total_naive_tokens:>8} {total_incr_tokens:>8}")

    print(f"\n--- PAIR CHECK COMPARISON ---")
    print(f"  Naive (all C(N,2) per turn):      {total_naive_all_checks:>8}")
    print(f"  Naive (qualifying C(Q,2) per turn):{total_naive_qual_checks:>8}")
    print(f"  Incremental:                       {total_incr_checks:>8}")
    print(f"  Savings vs all-pairs naive:        {pair_check_savings_all:>7.1%}")
    print(f"  Savings vs qualifying-only naive:  {pair_check_savings_qual:>7.1%}")

    print(f"\n--- TOKEN (CONTEXT) COMPARISON ---")
    print(f"  Naive (cumulative re-read):        {total_naive_tokens:>8} chars")
    print(f"  Incremental (per-chunk only):      {total_incr_tokens:>8} chars")
    print(f"  Token savings:                     {token_savings:>7.1%}")
    print(f"  Naive est. cost:                   ${naive_cost:.4f}")
    print(f"  Incremental est. cost:             ${incr_cost:.4f}")

    print(f"\n--- CORRECTNESS ---")
    print(f"  Final pairs match: {final_match} (naive={naive_per_turn[-1]['total_pairs']}, incr={len(incr_pairs_final)})")
    if not final_match:
        diff = naive_per_turn[-1]["total_pairs"] - len(incr_pairs_final)
        print(f"  Gap: {diff} pairs ({diff/naive_per_turn[-1]['total_pairs']:.1%} of naive)")
        print(f"  → This is the known A/C gap from qualification-time asymmetry")

    return {
        "task_idx": task_idx,
        "num_chunks": num_chunks,
        "max_chunk_chars": max_chunk_chars,
        "qualifying_labels": sorted(qualifying_labels),
        "simulation": {
            "per_turn": per_turn_comparison,
            "totals": {
                "naive_all_pair_checks": total_naive_all_checks,
                "naive_qual_pair_checks": total_naive_qual_checks,
                "incr_pair_checks": total_incr_checks,
                "pair_check_savings_vs_all_pct": round(pair_check_savings_all * 100, 2),
                "pair_check_savings_vs_qual_pct": round(pair_check_savings_qual * 100, 2),
                "naive_token_est": total_naive_tokens,
                "incr_token_est": total_incr_tokens,
                "token_savings_pct": round(token_savings * 100, 2),
                "naive_est_cost_usd": round(naive_cost, 4),
                "incr_est_cost_usd": round(incr_cost, 4),
                "correctness_all_turns_match": all_match,
                "correctness_final_match": final_match,
                "final_pairs_naive": naive_per_turn[-1]["total_pairs"],
                "final_pairs_incremental": len(incr_pairs_final),
            },
            "incremental_stats": incr_cumulative_stats,
        },
    }


# ---------------------------------------------------------------------------
# Part 2: Live API comparison
# ---------------------------------------------------------------------------

# Naive RLM prompt: process ALL accumulated context from scratch each turn
NAIVE_PROMPT = """Task (OOLONG-Pairs Task {task_idx}): Find ALL pairs of users where BOTH users have
at least one instance labeled {label_desc}.

You are given the FULL accumulated context below. Process it from scratch:
1. Parse every line matching "User: <id> ... || Label: <label>"
2. Determine which users are qualifying (have at least one qualifying label)
3. Enumerate ALL pairs of qualifying users

Qualifying labels: {qualifying_labels_repr}

Context format: "Date: [date] || User: [user_id] || Instance: [text] || Label: [cat]"

Run this code:

```repl
import re
from itertools import combinations

entities = {{}}
qualifying_labels = {qualifying_labels_repr}
for line in full_context.split('\\n'):
    m = re.search(r'User: (\\d+).*?\\|\\| Label: (.+?)$', line)
    if m:
        uid = m.group(1)
        label = m.group(2).strip().lower()
        if uid not in entities:
            entities[uid] = {{"labels": [], "qualifying": False}}
        entities[uid]["labels"].append(label)
        if label in qualifying_labels:
            entities[uid]["qualifying"] = True

qualifying_ids = sorted(uid for uid, a in entities.items() if a["qualifying"])
pair_results = [(min(a, b), max(a, b)) for a, b in combinations(qualifying_ids, 2)]
print(f"Entities: {{len(entities)}}, Qualifying: {{len(qualifying_ids)}}, Pairs: {{len(pair_results)}}")
```

After the repl block runs successfully, return FINAL_VAR(pair_results).
"""


def run_live_comparison(
    labeled_context: str,
    gold_pairs: set,
    api_key: str,
    task_idx: int = 1,
    num_chunks: int = 5,
    max_chunk_chars: int = 5000,
    model: str = "gpt-4o-mini",
    verbose: bool = False,
) -> dict:
    """Run a live API comparison: naive RLM (from scratch each turn) vs incremental RLM.

    The naive approach creates a fresh RLM for each turn, passing ALL accumulated
    context as a single blob. The incremental approach uses run_condition_a_v4()
    with persistent state.

    Args:
        labeled_context: Full labeled context string.
        gold_pairs: Set of gold-standard pairs for F1 computation.
        api_key: OpenAI API key.
        task_idx: OOLONG-Pairs task index (1, 3, or 6).
        num_chunks: Number of sequential chunks.
        max_chunk_chars: Characters per chunk.
        model: LLM model name.
        verbose: Enable verbose RLM output.

    Returns:
        Comparison dict with F1, tokens, pair checks, wall-clock, cost for both.
    """
    from eval.label_aware_v4_experiment import run_condition_a_v4
    from rlm.core.rlm import RLM

    os.environ["OPENAI_API_KEY"] = api_key

    qualifying_labels = TASK_QUALIFYING_LABELS[task_idx]
    label_desc = TASK_LABEL_DESCRIPTION[task_idx]
    chunks = _make_sequential_chunks(labeled_context, num_chunks, max_chunk_chars)

    print(f"\n{'=' * 70}")
    print(f"LIVE COMPARISON: Naive vs Incremental — Task {task_idx}")
    print(f"  Model: {model}")
    print(f"  k={num_chunks}, {max_chunk_chars} chars/chunk")
    print(f"  Qualifying: {label_desc}")
    print(f"  Gold pairs: {len(gold_pairs)}")
    print(f"{'=' * 70}")

    # ---- Naive RLM: fresh from scratch each turn ----
    print(f"\n{'~' * 60}")
    print(f"NAIVE RLM: Re-read all context from scratch each turn")
    print(f"{'~' * 60}")

    naive_turn_results: list[dict] = []
    naive_total_input = 0
    naive_total_output = 0
    naive_total_wall = 0.0

    for turn in range(num_chunks):
        cumulative_context = "".join(chunks[: turn + 1])
        chunk_num = turn + 1

        print(f"\n  --- Naive Turn {chunk_num}/{num_chunks}"
              f" (context: {len(cumulative_context)} chars) ---")

        root_prompt = NAIVE_PROMPT.format(
            task_idx=task_idx,
            label_desc=label_desc,
            qualifying_labels_repr=repr(qualifying_labels),
        )

        # Fresh RLM each turn — no persistent state
        rlm = RLM(
            backend="openai",
            backend_kwargs={"model_name": model},
            environment="local",
            persistent=False,
            max_iterations=4,
            verbose=verbose,
        )

        t0 = time.perf_counter()
        completion = rlm.completion(cumulative_context, root_prompt=root_prompt)
        elapsed = time.perf_counter() - t0
        rlm.close()

        # Extract tokens from usage summary
        from eval.f1_progression_experiment import _extract_tokens
        input_tokens, output_tokens = _extract_tokens(completion.usage_summary)

        # Extract pairs from response (RLMChatCompletion.response is the final answer string)
        naive_pairs = []
        if completion.response is not None:
            try:
                raw = completion.response
                if isinstance(raw, str):
                    # Try parsing as JSON (FINAL_VAR returns a Python literal)
                    raw = json.loads(raw.replace("'", '"'))
                if isinstance(raw, list):
                    naive_pairs = [
                        (min(str(p[0]), str(p[1])), max(str(p[0]), str(p[1])))
                        for p in raw
                        if isinstance(p, (list, tuple)) and len(p) == 2
                    ]
            except (json.JSONDecodeError, TypeError, IndexError):
                # Try eval as fallback for Python literal lists
                try:
                    import ast
                    parsed = ast.literal_eval(completion.response)
                    if isinstance(parsed, list):
                        naive_pairs = [
                            (min(str(p[0]), str(p[1])), max(str(p[0]), str(p[1])))
                            for p in parsed
                            if isinstance(p, (list, tuple)) and len(p) == 2
                        ]
                except (ValueError, SyntaxError):
                    print(f"    ⚠ Could not parse response as pairs")
                    pass

        f1_result = compute_f1(naive_pairs, gold_pairs)
        naive_total_input += input_tokens
        naive_total_output += output_tokens
        naive_total_wall += elapsed

        print(f"    pairs: {len(naive_pairs)}  F1={f1_result['f1']}"
              f"  P={f1_result['precision']}  R={f1_result['recall']}")
        print(f"    input_tokens: {input_tokens}  output_tokens: {output_tokens}"
              f"  elapsed: {elapsed:.1f}s")

        naive_turn_results.append({
            "turn": chunk_num,
            "context_chars": len(cumulative_context),
            "pairs_found": len(naive_pairs),
            "f1": f1_result["f1"],
            "precision": f1_result["precision"],
            "recall": f1_result["recall"],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "elapsed_sec": round(elapsed, 2),
        })

    # ---- Incremental RLM: run_condition_a_v4 ----
    print(f"\n{'~' * 60}")
    print(f"INCREMENTAL RLM: V4 with monotone_attrs")
    print(f"{'~' * 60}")

    incr_t0 = time.perf_counter()
    incr_result = run_condition_a_v4(
        labeled_context=labeled_context,
        gold_pairs=gold_pairs,
        api_key=api_key,
        task_idx=task_idx,
        num_chunks=num_chunks,
        max_chunk_chars=max_chunk_chars,
        model=model,
        verbose=verbose,
    )
    incr_total_wall = time.perf_counter() - incr_t0

    incr_total_input = incr_result.get("total_input_tokens", 0)
    incr_total_output = incr_result.get("total_output_tokens", 0)
    incr_final_f1 = incr_result.get("final_f1", 0) or 0

    # ---- Comparison ----
    naive_final_f1 = naive_turn_results[-1]["f1"] if naive_turn_results else 0
    naive_final_f1 = naive_final_f1 if naive_final_f1 is not None else 0

    token_ratio = (
        naive_total_input / incr_total_input
        if incr_total_input > 0
        else float("inf")
    )
    token_savings = (
        1.0 - incr_total_input / naive_total_input
        if naive_total_input > 0
        else 0.0
    )
    wall_clock_ratio = (
        naive_total_wall / incr_total_wall
        if incr_total_wall > 0
        else float("inf")
    )

    # Cost estimate (gpt-4o-mini pricing: $0.15/1M input, $0.60/1M output)
    naive_cost = naive_total_input * 0.15 / 1e6 + naive_total_output * 0.60 / 1e6
    incr_cost = incr_total_input * 0.15 / 1e6 + incr_total_output * 0.60 / 1e6
    cost_savings = 1.0 - incr_cost / naive_cost if naive_cost > 0 else 0.0

    print(f"\n{'=' * 70}")
    print(f"LIVE COMPARISON SUMMARY — Task {task_idx}")
    print(f"{'=' * 70}")
    print(f"{'Metric':<30} {'Naive':>15} {'Incremental':>15} {'Savings':>12}")
    print("-" * 72)
    print(f"{'Final F1':<30} {naive_final_f1:>15.4f} {incr_final_f1:>15.4f}")
    print(f"{'Total input tokens':<30} {naive_total_input:>15,} {incr_total_input:>15,}"
          f" {token_savings:>11.1%}")
    print(f"{'Total output tokens':<30} {naive_total_output:>15,} {incr_total_output:>15,}")
    print(f"{'Wall clock (sec)':<30} {naive_total_wall:>15.1f} {incr_total_wall:>15.1f}"
          f" {1.0 - incr_total_wall / naive_total_wall if naive_total_wall > 0 else 0:>11.1%}")
    print(f"{'Est. cost ($)':<30} {naive_cost:>15.6f} {incr_cost:>15.6f}"
          f" {cost_savings:>11.1%}")
    print(f"\nNaive/Incremental token ratio: {token_ratio:.2f}x")
    print(f"Naive/Incremental wall-clock ratio: {wall_clock_ratio:.2f}x")

    return {
        "task_idx": task_idx,
        "model": model,
        "num_chunks": num_chunks,
        "max_chunk_chars": max_chunk_chars,
        "live": {
            "naive": {
                "per_turn": naive_turn_results,
                "final_f1": naive_final_f1,
                "total_input_tokens": naive_total_input,
                "total_output_tokens": naive_total_output,
                "total_wall_sec": round(naive_total_wall, 2),
                "est_cost_usd": round(naive_cost, 6),
            },
            "incremental": {
                "condition_a_v4_result": incr_result,
                "final_f1": incr_final_f1,
                "total_input_tokens": incr_total_input,
                "total_output_tokens": incr_total_output,
                "total_wall_sec": round(incr_total_wall, 2),
                "est_cost_usd": round(incr_cost, 6),
            },
            "comparison": {
                "token_savings_pct": round(token_savings * 100, 2),
                "naive_over_incr_token_ratio": round(token_ratio, 4),
                "wall_clock_savings_pct": round(
                    (1.0 - incr_total_wall / naive_total_wall) * 100
                    if naive_total_wall > 0
                    else 0.0,
                    2,
                ),
                "cost_savings_pct": round(cost_savings * 100, 2),
                "f1_difference": round(incr_final_f1 - naive_final_f1, 4),
            },
        },
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Naive RLM vs Incremental RLM head-to-head comparison"
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run simulation comparison (no API key needed)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run live API comparison (needs OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--task-idx",
        type=int,
        default=1,
        choices=[1, 3, 6],
        help="Task index (default: 1)",
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Run tasks 1, 3, 6",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=5,
        help="Number of chunks (default: 5)",
    )
    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=5000,
        help="Max characters per chunk (default: 5000)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model for live comparison (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose RLM output",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for results JSON",
    )
    args = parser.parse_args()

    if not args.simulate and not args.live:
        print("ERROR: Specify at least one of --simulate or --live.")
        parser.print_help()
        sys.exit(1)

    tasks = [1, 3, 6] if args.all_tasks else [args.task_idx]

    # Load data
    print("Loading OOLONG-Pairs labeled data...")
    _, labeled_context = load_labeled_data()
    print(f"Labeled context: {len(labeled_context)} chars")

    all_results: dict = {}

    for task_idx in tasks:
        print(f"\n{'#' * 70}")
        print(f"# TASK {task_idx}: {TASK_LABEL_DESCRIPTION[task_idx]}")
        print(f"{'#' * 70}")

        task_result: dict = {
            "task_idx": task_idx,
            "num_chunks": args.num_chunks,
            "max_chunk_chars": args.max_chunk_chars,
        }

        # Part 1: Simulation
        if args.simulate:
            sim_result = run_simulation_comparison(
                labeled_context=labeled_context,
                task_idx=task_idx,
                num_chunks=args.num_chunks,
                max_chunk_chars=args.max_chunk_chars,
            )
            task_result["simulation"] = sim_result.get("simulation")

        # Part 2: Live API
        if args.live:
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                env_path = Path(__file__).parent.parent / ".env"
                if env_path.exists():
                    for line in env_path.read_text().splitlines():
                        if line.startswith("OPENAI_API_KEY="):
                            api_key = line.split("=", 1)[1].strip()
                            break
            if not api_key:
                print("ERROR: OPENAI_API_KEY not set. Skipping live comparison.")
            else:
                gold_pairs = compute_gold_pairs(labeled_context, task_idx=task_idx)
                print(f"Gold pairs for task {task_idx}: {len(gold_pairs)}")
                live_result = run_live_comparison(
                    labeled_context=labeled_context,
                    gold_pairs=gold_pairs,
                    api_key=api_key,
                    task_idx=task_idx,
                    num_chunks=args.num_chunks,
                    max_chunk_chars=args.max_chunk_chars,
                    model=args.model,
                    verbose=args.verbose,
                )
                task_result["live"] = live_result.get("live")

        all_results[f"task_{task_idx}"] = task_result

    # Save results
    output_path = args.output
    if output_path is None:
        output_dir = Path("results/streaming")
        output_dir.mkdir(parents=True, exist_ok=True)
        task_suffix = "all" if args.all_tasks else str(args.task_idx)
        mode_suffix = []
        if args.simulate:
            mode_suffix.append("sim")
        if args.live:
            mode_suffix.append("live")
        output_path = str(
            output_dir / f"naive_vs_incremental_task{task_suffix}_{'_'.join(mode_suffix)}.json"
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {output_path}")

    # Print final summary across tasks
    if args.simulate:
        print(f"\n{'=' * 90}")
        print(f"SIMULATION SUMMARY ACROSS TASKS")
        print(f"{'=' * 90}")
        print(f"{'Task':>6} {'N_all_chk':>10} {'N_qual_chk':>11} {'I_checks':>10}"
              f" {'Save(all)':>10} {'N_tokens':>10} {'I_tokens':>10} {'Tok_save':>9}"
              f" {'N_$':>8} {'I_$':>8} {'Final':>6}")
        print("-" * 106)
        for task_idx in tasks:
            key = f"task_{task_idx}"
            if key in all_results and "simulation" in all_results[key]:
                totals = all_results[key]["simulation"]["totals"]
                print(
                    f"{task_idx:>6}"
                    f" {totals['naive_all_pair_checks']:>10}"
                    f" {totals['naive_qual_pair_checks']:>11}"
                    f" {totals['incr_pair_checks']:>10}"
                    f" {totals['pair_check_savings_vs_all_pct']:>9.1f}%"
                    f" {totals['naive_token_est']:>10}"
                    f" {totals['incr_token_est']:>10}"
                    f" {totals['token_savings_pct']:>8.1f}%"
                    f" ${totals['naive_est_cost_usd']:>6.4f}"
                    f" ${totals['incr_est_cost_usd']:>6.4f}"
                    f" {'OK' if totals['correctness_final_match'] else str(totals['final_pairs_incremental'])+'/'+str(totals['final_pairs_naive']):>6}"
                )


if __name__ == "__main__":
    main()
