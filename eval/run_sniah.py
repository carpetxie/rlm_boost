"""
S-NIAH (Single Needle in a Haystack) evaluation for base model and RLM.

Requires pre-generated RULER tasks. See https://github.com/hsiehjackson/RULER
to generate tasks; save each length as {length}.json in --sniah-dir.

Usage:
  # Run one context length
  python eval/run_sniah.py \
    --sniah-dir data/sniah \
    --length 131072 \
    --method base \
    --output results/sniah/base_131072.json

  # Run all lengths (bash loop from implement.md §8.4)
  for LENGTH in 8192 16384 32768 65536 131072 262144 524288 1048576; do
    python eval/run_sniah.py --sniah-dir data/sniah --length $LENGTH \
      --method rlm --output results/sniah/rlm_${LENGTH}.json \
      --log-dir logs/sniah/rlm_${LENGTH}
  done
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

from eval.score import score_sniah, evaluate_sniah_results
from eval.utils import SNIAH_CONTEXT_LENGTHS, load_sniah_tasks

MODEL = "gpt-5"  # verify at platform.openai.com/docs/models


def run_base_model_sniah(haystack: str, question: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    prompt = f"{haystack}\n\n{question}"
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def run_rlm_sniah(haystack: str, question: str, log_dir: str) -> str:
    from rlm import RLM
    from rlm.logger import RLMLogger

    SUB_MODEL = "gpt-5-mini"
    logger = RLMLogger(log_dir=log_dir)
    rlm = RLM(
        backend="openai",
        backend_kwargs={"model_name": MODEL},
        other_backends=["openai"],
        other_backend_kwargs=[{"model_name": SUB_MODEL}],
        environment="local",
        max_iterations=30,
        logger=logger,
        verbose=False,
    )
    completion = rlm.completion(prompt=haystack, root_prompt=question)
    return completion.response


def evaluate_length(sniah_dir: str, length: int, method: str,
                    output_path: str, log_dir: str | None = None):
    tasks = load_sniah_tasks(sniah_dir, length)
    print(f"Running {len(tasks)} tasks at length={length} method={method}")

    results = []
    for i, task in enumerate(tasks):
        print(f"  Task {i + 1}/{len(tasks)}", end=" ", flush=True)
        try:
            if method == "base":
                pred = run_base_model_sniah(task["haystack"], task["question"])
            else:
                task_log = f"{log_dir}/task_{i}" if log_dir else f"logs/sniah/{length}/task_{i}"
                pred = run_rlm_sniah(task["haystack"], task["question"], task_log)
        except Exception as e:
            pred = ""
            print(f"  Error: {e}", end="")

        correct = score_sniah(pred, task["answer"])
        print(f"  {'✓' if correct else '✗'}", flush=True)
        results.append({
            "prediction": pred,
            "answer": task["answer"],
            "correct": bool(correct),
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    pct = sum(r["correct"] for r in results) / len(results) * 100
    print(f"\nLength {length}: {pct:.1f}% correct")
    print(f"Results saved to {output_path}")
    return pct


def main():
    parser = argparse.ArgumentParser(description="S-NIAH evaluation")
    parser.add_argument(
        "--sniah-dir", required=True,
        help="Directory containing RULER-generated {length}.json task files"
    )
    parser.add_argument(
        "--length", type=int,
        choices=SNIAH_CONTEXT_LENGTHS,
        required=True,
        help="Context length to evaluate"
    )
    parser.add_argument(
        "--method", choices=["base", "rlm"], required=True,
        help="base = vanilla GPT-5; rlm = RLM with REPL"
    )
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument(
        "--log-dir", default=None,
        help="RLM log directory (required for --method rlm)"
    )
    args = parser.parse_args()

    if args.method == "rlm" and args.log_dir is None:
        parser.error("--log-dir is required when --method rlm")

    evaluate_length(
        sniah_dir=args.sniah_dir,
        length=args.length,
        method=args.method,
        output_path=args.output,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
