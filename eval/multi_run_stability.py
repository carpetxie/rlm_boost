"""
Multi-Run Stability Experiment for Full-Corpus Condition A.

Runs run_condition_a_v4 multiple times on the same task/k configuration
to measure variance in F1, token usage, compliance, and wall-clock time.
Optionally includes a single Condition D run as a baseline.

Usage:
    # 3 runs of Task 1 k=5
    python eval/multi_run_stability.py --task 1 --k 5 --num-runs 3

    # 5 runs of Task 3 with Condition D baseline
    python eval/multi_run_stability.py --task 3 --num-runs 5 --include-d

    # Task 6, custom output dir
    python eval/multi_run_stability.py --task 6 --num-runs 3 --output-dir results/stability
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_api_key() -> str:
    """Load OPENAI_API_KEY from environment or .env file."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("OPENAI_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break
    if not api_key:
        print("ERROR: OPENAI_API_KEY required. Set in environment or .env file.")
        sys.exit(1)
    return api_key


def extract_wall_clock(result: dict) -> float:
    """Sum elapsed_sec from f1_progression to get total wall-clock time."""
    progression = result.get("f1_progression", [])
    return sum(t.get("elapsed_sec", 0) for t in progression)


def fmt_metric(values: list[float], fmt: str = ".4f") -> str:
    """Format mean +/- std for a list of values."""
    if not values:
        return "N/A"
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:{fmt}} +/- {std:{fmt}}"


def main():
    parser = argparse.ArgumentParser(
        description="Multi-run stability experiment for Condition A"
    )
    parser.add_argument("--task", type=int, default=1, help="Task index (1, 3, or 6)")
    parser.add_argument("--k", type=int, default=5, help="Number of chunks")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of Condition A runs")
    parser.add_argument("--output-dir", default="results/streaming", help="Output directory")
    parser.add_argument("--include-d", action="store_true", help="Also run Condition D once")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model to use")
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Temperature for LLM calls (default: model default)"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose RLM output")
    parser.add_argument(
        "--history-strategy", default=None,
        choices=["sliding_window", "summarize", "token_budget"],
        help="Override history pruning strategy (for aggressive pruning experiments)"
    )
    parser.add_argument(
        "--history-window", type=int, default=None,
        help="Max recent iterations to keep in history (default: 2 for sliding_window)"
    )
    args = parser.parse_args()

    # Validate task
    if args.task not in (1, 3, 6):
        print(f"WARNING: Task {args.task} is not one of the standard tasks (1, 3, 6).")
        print("Proceeding anyway...")

    api_key = load_api_key()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    from eval.label_aware_experiment import load_labeled_data
    from eval.rlm_pipeline_experiment import compute_gold_pairs

    print("Loading OOLONG labeled context...")
    _, labeled_context = load_labeled_data()

    total_chars = len(labeled_context)
    max_chunk_chars = total_chars // args.k
    gold_pairs = compute_gold_pairs(labeled_context, task_idx=args.task)

    print(f"\n{'=' * 70}")
    print(f"MULTI-RUN STABILITY EXPERIMENT")
    print(f"  Task {args.task} | k={args.k} | {args.num_runs} runs | model={args.model}")
    print(f"  Context: {total_chars:,} chars | {max_chunk_chars:,} chars/chunk")
    print(f"  Gold pairs: {len(gold_pairs)}")
    print(f"{'=' * 70}\n")

    from eval.label_aware_v4_experiment import run_condition_a_v4

    # -----------------------------------------------------------------------
    # Run Condition A num_runs times
    # -----------------------------------------------------------------------
    run_results = []
    for i in range(1, args.num_runs + 1):
        print(f"\n{'─' * 50}")
        print(f"  Condition A — Run {i}/{args.num_runs}")
        print(f"{'─' * 50}")

        t0 = time.time()
        result_a = run_condition_a_v4(
            labeled_context=labeled_context,
            gold_pairs=gold_pairs,
            api_key=api_key,
            task_idx=args.task,
            num_chunks=args.k,
            max_chunk_chars=max_chunk_chars,
            model=args.model,
            verbose=args.verbose,
            run_id=i,
            temperature=args.temperature,
            history_strategy=args.history_strategy,
            history_window=args.history_window,
        )
        wall_total = time.time() - t0

        # Save individual result
        out_path = (
            output_dir
            / f"full_corpus_task{args.task}_k{args.k}_condition_a_run{i}.json"
        )
        out_path.write_text(json.dumps(result_a, indent=2, default=str))
        print(f"  Saved to {out_path}")

        # Collect metrics
        f1 = result_a.get("final_f1") or 0
        precision = result_a.get("final_precision") or 0
        recall = result_a.get("final_recall") or 0
        input_tokens = result_a.get("total_input_tokens", 0)
        output_tokens = result_a.get("total_output_tokens", 0)
        compliance = result_a.get("compliance_rate", 0)
        progression_wall = extract_wall_clock(result_a)

        # Extract per-turn retraction counts from f1_progression
        progression = result_a.get("f1_progression", [])
        final_perm_retractions = progression[-1].get("permanent_retractions", 0) if progression else 0
        final_noop_retractions = progression[-1].get("noop_retractions", 0) if progression else 0
        per_turn_retractions = [
            {
                "turn": t.get("turn", j + 1),
                "noop_retractions": t.get("noop_retractions", 0),
                "permanent_retractions": t.get("permanent_retractions", 0),
            }
            for j, t in enumerate(progression)
        ]

        run_results.append({
            "run": i,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "total_input_tokens": input_tokens,
            "total_output_tokens": output_tokens,
            "compliance_rate": compliance,
            "wall_clock_sec": round(progression_wall, 2),
            "total_wall_sec": round(wall_total, 2),
            "permanent_retractions": final_perm_retractions,
            "noop_retractions": final_noop_retractions,
            "per_turn_retractions": per_turn_retractions,
        })

        print(f"  F1={f1:.4f} | P={precision:.4f} | R={recall:.4f} | "
              f"Compliance={compliance:.0%} | In={input_tokens:,} | Out={output_tokens:,} | "
              f"Wall={progression_wall:.1f}s")

    # -----------------------------------------------------------------------
    # Condition D (optional, single run)
    # -----------------------------------------------------------------------
    result_d = None
    if args.include_d:
        from eval.label_aware_v4_experiment import run_condition_d_full_recompute

        print(f"\n{'─' * 50}")
        print(f"  Condition D — Full Recompute (single run)")
        print(f"{'─' * 50}")

        result_d = run_condition_d_full_recompute(
            labeled_context=labeled_context,
            gold_pairs=gold_pairs,
            api_key=api_key,
            task_idx=args.task,
            num_chunks=args.k,
            max_chunk_chars=max_chunk_chars,
            model=args.model,
            verbose=args.verbose,
        )
        out_d = output_dir / f"full_corpus_task{args.task}_k{args.k}_condition_d.json"
        out_d.write_text(json.dumps(result_d, indent=2, default=str))
        print(f"  Saved to {out_d}")

    # -----------------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------------
    f1s = [r["f1"] for r in run_results]
    precisions = [r["precision"] for r in run_results]
    recalls = [r["recall"] for r in run_results]
    input_tokens_list = [r["total_input_tokens"] for r in run_results]
    output_tokens_list = [r["total_output_tokens"] for r in run_results]
    compliance_rates = [r["compliance_rate"] for r in run_results]
    wall_clocks = [r["wall_clock_sec"] for r in run_results]

    print(f"\n{'=' * 70}")
    print(f"STABILITY SUMMARY — Task {args.task}, k={args.k}, {args.num_runs} runs")
    print(f"{'=' * 70}")
    print(f"  {'Metric':<22} {'Mean +/- Std':<24} {'Min':>10} {'Max':>10}")
    print(f"  {'─' * 66}")
    print(f"  {'F1':<22} {fmt_metric(f1s):<24} {min(f1s):>10.4f} {max(f1s):>10.4f}")
    print(f"  {'Precision':<22} {fmt_metric(precisions):<24} "
          f"{min(precisions):>10.4f} {max(precisions):>10.4f}")
    print(f"  {'Recall':<22} {fmt_metric(recalls):<24} "
          f"{min(recalls):>10.4f} {max(recalls):>10.4f}")
    print(f"  {'Compliance Rate':<22} {fmt_metric(compliance_rates, '.2%'):<24} "
          f"{min(compliance_rates):>10.2%} {max(compliance_rates):>10.2%}")
    print(f"  {'Input Tokens':<22} {fmt_metric(input_tokens_list, ',.0f'):<24} "
          f"{min(input_tokens_list):>10,} {max(input_tokens_list):>10,}")
    print(f"  {'Output Tokens':<22} {fmt_metric(output_tokens_list, ',.0f'):<24} "
          f"{min(output_tokens_list):>10,} {max(output_tokens_list):>10,}")
    print(f"  {'Wall Clock (s)':<22} {fmt_metric(wall_clocks, '.1f'):<24} "
          f"{min(wall_clocks):>10.1f} {max(wall_clocks):>10.1f}")

    if result_d:
        fd = result_d.get("final_f1") or 0
        td = result_d.get("total_input_tokens", 0)
        mean_f1_a = statistics.mean(f1s)
        mean_tok_a = statistics.mean(input_tokens_list)
        savings = 1 - mean_tok_a / td if td > 0 else 0

        print(f"\n  {'─' * 66}")
        print(f"  Condition D (single run):  F1={fd:.4f}  Input Tokens={td:,}")
        print(f"  A vs D:  F1 delta = {mean_f1_a - fd:+.4f}  "
              f"Token savings = {savings:.1%}")

    # -----------------------------------------------------------------------
    # Save aggregate summary
    # -----------------------------------------------------------------------
    summary = {
        "task": args.task,
        "k": args.k,
        "num_runs": args.num_runs,
        "model": args.model,
        "temperature": args.temperature,
        "gold_pairs_count": len(gold_pairs),
        "total_context_chars": total_chars,
        "runs": run_results,
        "aggregate": {
            "f1": {
                "mean": statistics.mean(f1s),
                "std": statistics.stdev(f1s) if len(f1s) > 1 else 0.0,
                "min": min(f1s),
                "max": max(f1s),
            },
            "precision": {
                "mean": statistics.mean(precisions),
                "std": statistics.stdev(precisions) if len(precisions) > 1 else 0.0,
                "min": min(precisions),
                "max": max(precisions),
            },
            "recall": {
                "mean": statistics.mean(recalls),
                "std": statistics.stdev(recalls) if len(recalls) > 1 else 0.0,
                "min": min(recalls),
                "max": max(recalls),
            },
            "compliance_rate": {
                "mean": statistics.mean(compliance_rates),
                "std": statistics.stdev(compliance_rates) if len(compliance_rates) > 1 else 0.0,
                "min": min(compliance_rates),
                "max": max(compliance_rates),
            },
            "total_input_tokens": {
                "mean": statistics.mean(input_tokens_list),
                "std": statistics.stdev(input_tokens_list) if len(input_tokens_list) > 1 else 0.0,
                "min": min(input_tokens_list),
                "max": max(input_tokens_list),
            },
            "total_output_tokens": {
                "mean": statistics.mean(output_tokens_list),
                "std": (
                    statistics.stdev(output_tokens_list) if len(output_tokens_list) > 1 else 0.0
                ),
                "min": min(output_tokens_list),
                "max": max(output_tokens_list),
            },
            "wall_clock_sec": {
                "mean": statistics.mean(wall_clocks),
                "std": statistics.stdev(wall_clocks) if len(wall_clocks) > 1 else 0.0,
                "min": min(wall_clocks),
                "max": max(wall_clocks),
            },
        },
    }

    if result_d:
        summary["condition_d"] = {
            "final_f1": result_d.get("final_f1") or 0,
            "total_input_tokens": result_d.get("total_input_tokens", 0),
            "total_output_tokens": result_d.get("total_output_tokens", 0),
            "compliance_rate": result_d.get("compliance_rate", 0),
            "wall_clock_sec": extract_wall_clock(result_d),
        }

    temp_suffix = f"_temp{args.temperature}" if args.temperature is not None else ""
    model_suffix = f"_{args.model.replace('/', '_')}" if args.model != "gpt-4o-mini" else ""
    summary_path = (
        output_dir / f"stability_task{args.task}_k{args.k}_n{args.num_runs}{model_suffix}{temp_suffix}.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nAggregate summary saved to {summary_path}")


if __name__ == "__main__":
    main()
