"""
RLM evaluation for OOLONG and OOLONG-Pairs.

Usage:
  python eval/run_rlm.py --benchmark oolong \
    --output results/oolong/rlm.json \
    --log-dir logs/oolong/rlm

  python eval/run_rlm.py --benchmark oolong_pairs \
    --output results/oolong_pairs/rlm.json \
    --log-dir logs/oolong_pairs/rlm
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from rlm import RLM
from rlm.logger import RLMLogger

from eval.score import evaluate_oolong_results, evaluate_oolong_pairs_results

ROOT_MODEL = "gpt-5"       # verify at platform.openai.com/docs/models
SUB_MODEL  = "gpt-5-mini"  # verify at platform.openai.com/docs/models


def build_rlm(log_dir: str) -> RLM:
    logger = RLMLogger(log_dir=log_dir)
    return RLM(
        backend="openai",
        backend_kwargs={"model_name": ROOT_MODEL},
        other_backends=["openai"],
        other_backend_kwargs=[{"model_name": SUB_MODEL}],
        environment="local",
        max_iterations=30,
        logger=logger,
        verbose=False,
    )


def run_single_rlm(args_tuple):
    """Worker function for ProcessPoolExecutor.

    Args:
        args_tuple: (i, total, example_dict, log_dir_base)
    Returns:
        result dict
    """
    i, total, example, log_dir_base = args_tuple
    task_key = example.get("id") or example.get("task_id") or i
    print(f"  Task {i + 1}/{total}  key={task_key}", flush=True)
    rlm = build_rlm(log_dir=f"{log_dir_base}/task_{i}")
    pred = ""
    usage = None
    try:
        completion = rlm.completion(
            prompt=example["context"],
            root_prompt=example["question"],
        )
        pred = completion.response
        usage = completion.usage_summary.to_dict()
    except Exception as e:
        print(f"    Error: {e}", flush=True)

    return {
        "id": example.get("id") or example.get("task_id") or i,
        "prediction": pred,
        "answer": example["answer"],
        "usage": usage,
    }


def evaluate_oolong(output_path: str, log_dir: str, max_workers: int = 1):
    from eval.utils import load_oolong
    dataset = load_oolong(context_len=131072)
    print(f"Loaded {len(dataset)} OOLONG tasks")

    args_list = [(i, len(dataset), ex, log_dir) for i, ex in enumerate(dataset)]

    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(run_single_rlm, args_list))
    else:
        results = [run_single_rlm(a) for a in args_list]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    score = evaluate_oolong_results(results)
    print(f"\nOOLONG RLM score: {score:.1f}  (expected ~56.5)")
    print(f"Results saved to {output_path}")


def evaluate_oolong_pairs(output_path: str, log_dir: str, max_workers: int = 1, gold_file: str | None = None):
    from eval.utils import load_oolong_pairs
    dataset = load_oolong_pairs(context_len=32768, gold_file=gold_file)
    print(f"Loaded {len(dataset)} OOLONG-Pairs tasks")

    args_list = [(i, len(dataset), ex, log_dir) for i, ex in enumerate(dataset)]

    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(run_single_rlm, args_list))
    else:
        results = [run_single_rlm(a) for a in args_list]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    score = evaluate_oolong_pairs_results(results)
    print(f"\nOOLONG-Pairs RLM F1: {score:.1f}  (expected ~58.0)")
    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="RLM evaluation")
    parser.add_argument(
        "--benchmark", choices=["oolong", "oolong_pairs"], required=True,
        help="Which benchmark to run"
    )
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--log-dir", required=True, help="Directory for RLM logs")
    parser.add_argument(
        "--max-workers", type=int, default=1,
        help="Parallel processes (each RLM instance uses its own port)"
    )
    parser.add_argument("--gold-file", default=None, help="Path to official gold answers JSON (OOLONG-Pairs only)")
    args = parser.parse_args()

    if args.benchmark == "oolong":
        evaluate_oolong(args.output, args.log_dir, args.max_workers)
    elif args.benchmark == "oolong_pairs":
        evaluate_oolong_pairs(args.output, args.log_dir, args.max_workers, gold_file=args.gold_file)


if __name__ == "__main__":
    main()
