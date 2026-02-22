"""
Analysis script for OOLONG-Pairs results.
Computes: token costs, per-task F1, failure analysis, cost-per-F1-point.

Usage:
  python eval/analyze_results.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.score import f1_pairs, parse_pairs

# Approximate pricing (GPT-5 and GPT-5-mini, as of Feb 2026)
# These are estimates — adjust if actual pricing differs
PRICING = {
    "gpt-5": {"input": 2.00 / 1_000_000, "output": 8.00 / 1_000_000},  # $/token
    "gpt-5-mini": {"input": 0.10 / 1_000_000, "output": 0.40 / 1_000_000},
}


def analyze_token_costs(results: list[dict]) -> dict:
    """Analyze token usage and costs from OOLONG-Pairs results."""
    total_input = 0
    total_output = 0
    per_task = []

    for r in results:
        usage = r.get("usage")
        task_id = r.get("id") or r.get("task_id") or "?"
        if not usage:
            per_task.append({"task_id": task_id, "input_tokens": 0, "output_tokens": 0, "cost": 0})
            continue

        model_summaries = usage.get("model_usage_summaries", {})
        task_input = 0
        task_output = 0
        task_cost = 0.0

        for model_name, summary in model_summaries.items():
            inp = summary.get("total_input_tokens", 0)
            out = summary.get("total_output_tokens", 0)
            task_input += inp
            task_output += out

            # Determine pricing tier
            pricing_key = None
            for key in PRICING:
                if key in model_name.lower():
                    pricing_key = key
                    break

            if pricing_key:
                task_cost += inp * PRICING[pricing_key]["input"]
                task_cost += out * PRICING[pricing_key]["output"]

        total_input += task_input
        total_output += task_output
        per_task.append(
            {
                "task_id": r["id"],
                "input_tokens": task_input,
                "output_tokens": task_output,
                "cost": task_cost,
            }
        )

    return {
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total_input + total_output,
        "per_task": per_task,
    }


def analyze_per_task_f1(results: list[dict]) -> list[dict]:
    """Compute per-task F1 and pair counts."""
    task_analysis = []
    for r in results:
        pred_text = r["prediction"]
        gold_text = r["answer"]
        f1 = f1_pairs(pred_text, gold_text)
        pred_set = parse_pairs(pred_text)
        gold_set = parse_pairs(gold_text)

        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)

        precision = tp / len(pred_set) if pred_set else 0
        recall = tp / len(gold_set) if gold_set else 0

        task_analysis.append(
            {
                "task_id": r["id"],
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "predicted_pairs": len(pred_set),
                "gold_pairs": len(gold_set),
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
            }
        )
    return task_analysis


def categorize_failures(task_analysis: list[dict]) -> dict:
    """Categorize failure modes."""
    failures = {"low_precision": [], "low_recall": [], "both_low": [], "good": []}
    for t in task_analysis:
        if t["f1"] >= 0.8:
            failures["good"].append(t["task_id"])
        elif t["precision"] < 0.5 and t["recall"] < 0.5:
            failures["both_low"].append(t["task_id"])
        elif t["precision"] < 0.5:
            failures["low_precision"].append(t["task_id"])
        elif t["recall"] < 0.5:
            failures["low_recall"].append(t["task_id"])
        else:
            failures["both_low"].append(t["task_id"])  # moderate failures

    return failures


def main():
    rlm_path = Path("results/oolong_pairs/rlm.json")
    base_path = Path("results/oolong_pairs/base_model.json")

    if not rlm_path.exists():
        print(f"ERROR: {rlm_path} not found")
        sys.exit(1)

    with open(rlm_path) as f:
        rlm_results = json.load(f)

    print("=" * 80)
    print("OOLONG-Pairs Analysis")
    print("=" * 80)

    # 1. Overall F1
    f1_scores = [f1_pairs(r["prediction"], r["answer"]) for r in rlm_results]
    avg_f1 = sum(f1_scores) / len(f1_scores) * 100
    print(f"\nOverall F1: {avg_f1:.1f}%")

    # 2. Per-task F1 analysis
    task_analysis = analyze_per_task_f1(rlm_results)
    print("\n--- Per-Task F1 ---")
    print(
        f"{'Task':>6} {'F1':>8} {'Prec':>8} {'Recall':>8} {'Pred':>6} {'Gold':>6} {'TP':>5} {'FP':>5} {'FN':>5}"
    )
    for t in sorted(task_analysis, key=lambda x: x["f1"]):
        print(
            f"{t['task_id']:>6} {t['f1']:>8.3f} {t['precision']:>8.3f} {t['recall']:>8.3f} "
            f"{t['predicted_pairs']:>6} {t['gold_pairs']:>6} {t['true_positive']:>5} "
            f"{t['false_positive']:>5} {t['false_negative']:>5}"
        )

    # 3. Failure categorization
    failures = categorize_failures(task_analysis)
    print("\n--- Failure Categorization ---")
    for cat, tasks in failures.items():
        print(f"  {cat}: {tasks}")

    # Symmetric vs Asymmetric analysis
    sym_f1 = [t["f1"] for t in task_analysis if t["task_id"] <= 10]
    asym_f1 = [t["f1"] for t in task_analysis if t["task_id"] > 10]
    print(f"\n  Symmetric tasks (1-10) avg F1:  {sum(sym_f1) / len(sym_f1) * 100:.1f}%")
    print(f"  Asymmetric tasks (11-20) avg F1: {sum(asym_f1) / len(asym_f1) * 100:.1f}%")

    # 4. Token cost analysis
    cost_analysis = analyze_token_costs(rlm_results)
    print("\n--- Token Usage ---")
    print(f"  Total input tokens:  {cost_analysis['total_input_tokens']:>12,}")
    print(f"  Total output tokens: {cost_analysis['total_output_tokens']:>12,}")
    print(f"  Total tokens:        {cost_analysis['total_tokens']:>12,}")

    total_cost = sum(t["cost"] for t in cost_analysis["per_task"])
    print(f"\n  Estimated total cost: ${total_cost:.2f}")
    print(f"  Cost per task:        ${total_cost / 20:.2f}")
    print(f"  Cost per F1 point:    ${total_cost / (avg_f1 / 100):.2f}")

    print("\n--- Per-Task Token Usage ---")
    print(f"{'Task':>6} {'Input':>12} {'Output':>12} {'Total':>12} {'Cost':>8}")
    for t in cost_analysis["per_task"]:
        total = t["input_tokens"] + t["output_tokens"]
        print(
            f"{t['task_id']:>6} {t['input_tokens']:>12,} {t['output_tokens']:>12,} "
            f"{total:>12,} ${t['cost']:>7.2f}"
        )

    # 5. Model breakdown
    print("\n--- Model Usage Breakdown ---")
    model_totals = {}
    for r in rlm_results:
        usage = r.get("usage", {})
        for model_name, summary in usage.get("model_usage_summaries", {}).items():
            if model_name not in model_totals:
                model_totals[model_name] = {"input": 0, "output": 0, "calls": 0}
            model_totals[model_name]["input"] += summary.get("total_input_tokens", 0)
            model_totals[model_name]["output"] += summary.get("total_output_tokens", 0)
            model_totals[model_name]["calls"] += summary.get("num_calls", 0)

    for model, totals in model_totals.items():
        print(f"\n  {model}:")
        print(f"    Calls:   {totals['calls']:>8,}")
        print(f"    Input:   {totals['input']:>12,} tokens")
        print(f"    Output:  {totals['output']:>12,} tokens")

    # 6. Base model comparison
    if base_path.exists():
        with open(base_path) as f:
            base_results = json.load(f)

        base_f1_scores = [f1_pairs(r["prediction"], r["answer"]) for r in base_results]
        base_avg_f1 = sum(base_f1_scores) / len(base_f1_scores) * 100

        base_costs = analyze_token_costs(base_results)
        base_total_cost = sum(t["cost"] for t in base_costs["per_task"])

        print("\n--- RLM vs Base Model ---")
        print(f"  {'':>20} {'RLM':>12} {'Base':>12} {'Delta':>12}")
        print(
            f"  {'F1 (%)':>20} {avg_f1:>12.1f} {base_avg_f1:>12.1f} {avg_f1 - base_avg_f1:>+12.1f}"
        )
        print(
            f"  {'Total Tokens':>20} {cost_analysis['total_tokens']:>12,} {base_costs['total_tokens']:>12,} {cost_analysis['total_tokens'] - base_costs['total_tokens']:>+12,}"
        )
        print(
            f"  {'Est. Cost ($)':>20} {total_cost:>12.2f} {base_total_cost:>12.2f} {total_cost - base_total_cost:>+12.2f}"
        )

        if avg_f1 > 0:
            print(f"  {'Cost/F1pt (RLM)':>20} ${total_cost / (avg_f1 / 100):>11.2f}")
        if base_avg_f1 > 0:
            print(f"  {'Cost/F1pt (Base)':>20} ${base_total_cost / (base_avg_f1 / 100):>11.2f}")

    # 7. Summary findings as JSON
    summary = {
        "overall_f1": avg_f1,
        "symmetric_f1": sum(sym_f1) / len(sym_f1) * 100,
        "asymmetric_f1": sum(asym_f1) / len(asym_f1) * 100,
        "total_tokens": cost_analysis["total_tokens"],
        "estimated_cost": total_cost,
        "per_task": task_analysis,
        "failure_categories": failures,
    }

    out_path = Path("results/oolong_pairs/analysis.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAnalysis saved to {out_path}")


if __name__ == "__main__":
    main()
