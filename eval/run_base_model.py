"""
Base model evaluation for OOLONG and OOLONG-Pairs.

Usage:
  python eval/run_base_model.py --benchmark oolong --output results/oolong/base_model.json
  python eval/run_base_model.py --benchmark oolong_pairs --output results/oolong_pairs/base_model.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

from eval.score import evaluate_oolong_results, evaluate_oolong_pairs_results

MODEL = "gpt-5"  # verify at platform.openai.com/docs/models

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def run_base_model(context: str, question: str) -> str:
    """Single base-model call: no system prompt, context + question as user message."""
    prompt = f"{context}\n\n{question}"
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
        {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or ""


def evaluate_oolong(output_path: str):
    from eval.utils import load_oolong
    dataset = load_oolong(context_len=131072)
    print(f"Loaded {len(dataset)} OOLONG tasks")

    results = []
    for i, example in enumerate(dataset):
        print(f"  Task {i + 1}/{len(dataset)}  id={example['id']}")
        try:
            pred = run_base_model(example["context"], example["question"])
        except Exception as e:
            pred = ""
            print(f"    Error: {e}")
        results.append({
            "id": example["id"],
            "prediction": pred,
            "answer": example["answer"],
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    score = evaluate_oolong_results(results)
    print(f"\nOOLONG Base Model score: {score:.1f}  (expected ~44.0)")
    print(f"Results saved to {output_path}")


def evaluate_oolong_pairs(output_path: str, gold_file: str | None = None):
    from eval.utils import load_oolong_pairs
    dataset = load_oolong_pairs(context_len=32768, gold_file=gold_file)
    print(f"Loaded {len(dataset)} OOLONG-Pairs tasks")

    results = []
    for i, example in enumerate(dataset):
        print(f"  Task {i + 1}/{len(dataset)}  task_id={example['task_id']}")
        try:
            pred = run_base_model(example["context"], example["question"])
        except Exception as e:
            pred = ""
            print(f"    Error: {e}")
        print(f"    Prediction: {pred[:200]}")
        results.append({
            "task_id": example["task_id"],
            "prediction": pred,
            "answer": example["answer"],
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    score = evaluate_oolong_pairs_results(results)
    print("Raw score:", score)
    if score > 0:
        print("score is non-zero, so we can round to 0.1")
    print(f"\nFormatted OOLONG-Pairs Base Model F1: {score:.1f}")
    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Base model evaluation")
    parser.add_argument(
        "--benchmark", choices=["oolong", "oolong_pairs"], required=True,
        help="Which benchmark to run"
    )
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--gold-file", default=None, help="Path to official gold answers JSON (OOLONG-Pairs only)")
    args = parser.parse_args()

    if args.benchmark == "oolong":
        evaluate_oolong(args.output)
    elif args.benchmark == "oolong_pairs":
        evaluate_oolong_pairs(args.output, gold_file=args.gold_file)


if __name__ == "__main__":
    main()
