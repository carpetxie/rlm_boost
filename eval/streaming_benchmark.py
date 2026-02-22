"""
Streaming OOLONG-Pairs Benchmark — Dynamic Context Evaluation.

Instead of giving all context at once, we deliver it in N chunks and ask
for the answer after each chunk. This tests whether persistent RLM (reusing
REPL state) saves computation vs non-persistent (fresh environment per chunk).

This benchmark measures:
  1. Final F1 accuracy
  2. F1 at each intermediate step (how fast does accuracy converge?)
  3. Total tokens consumed (persistent vs non-persistent)
  4. Per-chunk incremental cost

Usage:
  python eval/streaming_benchmark.py --mode persistent --num-chunks 3 --tasks 1,2,3
  python eval/streaming_benchmark.py --mode non-persistent --num-chunks 3 --tasks 1,2,3
  python eval/streaming_benchmark.py --mode simulate --num-chunks 5 --tasks 1,2,3
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.score import f1_pairs
from eval.utils import OOLONG_PAIRS_TASKS


def load_oolong_pairs_data(context_len: int = 32768, gold_file: str | None = None):
    """Load OOLONG-Pairs data with labeled context for gold computation."""
    from datasets import load_dataset

    ds = load_dataset("oolongbench/oolong-synth", split="validation")
    trec_at_len = [
        x for x in ds if x["dataset"] == "trec_coarse" and x["context_len"] == context_len
    ]
    if not trec_at_len:
        raise ValueError(f"No trec_coarse examples at context_len={context_len}")

    corpus = trec_at_len[0]
    return {
        "context": corpus["context_window_text"],
        "labeled_context": corpus["context_window_text_with_labels"],
    }


def split_context_by_users(context: str, num_chunks: int) -> list[str]:
    """Split context into chunks, preserving user boundaries.

    The OOLONG-Pairs context is structured as user entries separated by
    patterns like "User: XXXXX". We split at user boundaries to ensure
    each chunk contains complete user entries.
    """
    # Find all user boundaries
    user_pattern = re.compile(r"(?=Date:.*?\|\| User:)")
    positions = [m.start() for m in user_pattern.finditer(context)]

    if not positions:
        # Fallback: split evenly by character
        chunk_size = len(context) // num_chunks
        return [
            context[i * chunk_size : (i + 1) * chunk_size if i < num_chunks - 1 else len(context)]
            for i in range(num_chunks)
        ]

    # Distribute users evenly across chunks
    users_per_chunk = max(1, len(positions) // num_chunks)
    chunks = []
    for i in range(num_chunks):
        start_idx = (
            positions[i * users_per_chunk] if i * users_per_chunk < len(positions) else len(context)
        )
        if i < num_chunks - 1:
            end_user_idx = min((i + 1) * users_per_chunk, len(positions))
            end_idx = positions[end_user_idx] if end_user_idx < len(positions) else len(context)
        else:
            end_idx = len(context)
        if start_idx < end_idx:
            chunks.append(context[start_idx:end_idx])

    # Merge any trailing empty chunks
    while len(chunks) < num_chunks:
        chunks.append("")

    return chunks[:num_chunks]


def simulate_streaming(
    context: str,
    labeled_context: str,
    task_indices: list[int],
    num_chunks: int,
    gold_file: str | None = None,
) -> dict:
    """Simulate the streaming benchmark without API calls.

    This computes what the gold answers would be at each chunk level,
    establishing ground truth for how information arrives incrementally.
    """
    from eval.utils import _parse_labeled_context, compute_gold_pairs

    # Split both labeled and unlabeled context
    chunks = split_context_by_users(context, num_chunks)
    labeled_chunks = split_context_by_users(labeled_context, num_chunks)

    results = {}

    for task_idx in task_indices:
        question = f"In the above data, {OOLONG_PAIRS_TASKS[task_idx - 1]}"
        task_result = {
            "task_id": task_idx,
            "question": question,
            "chunks": [],
            "full_gold": compute_gold_pairs(labeled_context, task_idx),
        }

        # Compute gold at each cumulative chunk level
        cumulative_labeled = ""
        cumulative_unlabeled = ""
        for chunk_i in range(num_chunks):
            cumulative_labeled += labeled_chunks[chunk_i]
            cumulative_unlabeled += chunks[chunk_i]

            # Parse users visible so far
            users_so_far = _parse_labeled_context(cumulative_labeled)

            # Compute gold pairs with only visible data
            partial_gold = compute_gold_pairs(cumulative_labeled, task_idx)

            # How many gold pairs from the full answer are "discoverable" now?
            full_gold_set = set()
            for pair_str in task_result["full_gold"].split("\n"):
                if pair_str.strip():
                    m = re.match(r"\((\d+),\s*(\d+)\)", pair_str.strip())
                    if m:
                        full_gold_set.add((int(m.group(1)), int(m.group(2))))

            partial_gold_set = set()
            for pair_str in partial_gold.split("\n"):
                if pair_str.strip():
                    m = re.match(r"\((\d+),\s*(\d+)\)", pair_str.strip())
                    if m:
                        partial_gold_set.add((int(m.group(1)), int(m.group(2))))

            chunk_info = {
                "chunk_index": chunk_i,
                "cumulative_chars": len(cumulative_unlabeled),
                "users_visible": len(users_so_far),
                "partial_gold_pairs": len(partial_gold_set),
                "full_gold_pairs": len(full_gold_set),
                "discoverable_fraction": len(partial_gold_set) / len(full_gold_set)
                if full_gold_set
                else 0,
            }
            task_result["chunks"].append(chunk_info)

        results[task_idx] = task_result

    return results


def run_persistent_streaming(
    context: str,
    labeled_context: str,
    task_indices: list[int],
    num_chunks: int,
    gold_file: str | None = None,
) -> dict:
    """Run streaming benchmark with persistent RLM (reuses REPL state across chunks)."""
    from eval.utils import compute_gold_pairs
    from rlm import RLM
    from rlm.logger import RLMLogger

    chunks = split_context_by_users(context, num_chunks)
    results = {}

    for task_idx in task_indices:
        question = f"In the above data, {OOLONG_PAIRS_TASKS[task_idx - 1]}"
        full_gold = compute_gold_pairs(labeled_context, task_idx)

        task_result = {
            "task_id": task_idx,
            "question": question,
            "mode": "persistent",
            "chunks": [],
        }

        logger = RLMLogger(log_dir=f"logs/streaming/persistent/task_{task_idx}")

        with RLM(
            backend="openai",
            backend_kwargs={"model_name": "gpt-5"},
            other_backends=["openai"],
            other_backend_kwargs=[{"model_name": "gpt-5-mini"}],
            environment="local",
            max_iterations=15,  # fewer iterations per chunk
            logger=logger,
            persistent=True,
        ) as rlm:
            cumulative_context = ""
            for chunk_i, chunk in enumerate(chunks):
                cumulative_context += chunk
                chunk_start = time.perf_counter()

                # For persistent mode, each completion adds a new context_N
                # The question asks about "the above data" so we pass cumulative context
                # But the key insight: the model's REPL state persists, so any
                # summaries/variables from prior chunks remain available
                try:
                    if chunk_i == 0:
                        # First chunk: pass full context
                        completion = rlm.completion(
                            prompt=cumulative_context,
                            root_prompt=question
                            + f"\n\n[This is chunk {chunk_i + 1}/{num_chunks}. More data will follow. Give your best answer so far.]",
                        )
                    else:
                        # Subsequent chunks: pass only the delta
                        completion = rlm.completion(
                            prompt=chunk,
                            root_prompt=question
                            + f"\n\n[This is chunk {chunk_i + 1}/{num_chunks}. You have prior context in context_0 through context_{chunk_i - 1}. Update your answer with the new data.]",
                        )

                    chunk_time = time.perf_counter() - chunk_start
                    usage = completion.usage_summary.to_dict()
                    prediction = completion.response
                    f1 = f1_pairs(prediction, full_gold)

                    task_result["chunks"].append(
                        {
                            "chunk_index": chunk_i,
                            "prediction": prediction[:500],  # truncate for log
                            "f1": f1,
                            "execution_time": chunk_time,
                            "usage": usage,
                        }
                    )
                    print(
                        f"  Task {task_idx}, Chunk {chunk_i + 1}/{num_chunks}: F1={f1:.3f}, time={chunk_time:.1f}s"
                    )
                except Exception as e:
                    print(f"  Task {task_idx}, Chunk {chunk_i + 1}/{num_chunks}: ERROR: {e}")
                    task_result["chunks"].append(
                        {
                            "chunk_index": chunk_i,
                            "error": str(e),
                        }
                    )

        results[task_idx] = task_result

    return results


def run_non_persistent_streaming(
    context: str,
    labeled_context: str,
    task_indices: list[int],
    num_chunks: int,
    gold_file: str | None = None,
) -> dict:
    """Run streaming benchmark with non-persistent RLM (fresh environment per chunk).

    Each chunk gets the full cumulative context — simulating 're-read everything'.
    """
    from eval.utils import compute_gold_pairs
    from rlm import RLM
    from rlm.logger import RLMLogger

    chunks = split_context_by_users(context, num_chunks)
    results = {}

    for task_idx in task_indices:
        question = f"In the above data, {OOLONG_PAIRS_TASKS[task_idx - 1]}"
        full_gold = compute_gold_pairs(labeled_context, task_idx)

        task_result = {
            "task_id": task_idx,
            "question": question,
            "mode": "non-persistent",
            "chunks": [],
        }

        cumulative_context = ""
        for chunk_i, chunk in enumerate(chunks):
            cumulative_context += chunk
            chunk_start = time.perf_counter()

            logger = RLMLogger(
                log_dir=f"logs/streaming/non_persistent/task_{task_idx}/chunk_{chunk_i}"
            )

            # Fresh RLM each time with full cumulative context
            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "gpt-5"},
                other_backends=["openai"],
                other_backend_kwargs=[{"model_name": "gpt-5-mini"}],
                environment="local",
                max_iterations=15,
                logger=logger,
                persistent=False,
            )

            try:
                completion = rlm.completion(
                    prompt=cumulative_context,
                    root_prompt=question
                    + f"\n\n[You have {chunk_i + 1}/{num_chunks} chunks of data. Give your best answer.]",
                )

                chunk_time = time.perf_counter() - chunk_start
                usage = completion.usage_summary.to_dict()
                prediction = completion.response
                f1 = f1_pairs(prediction, full_gold)

                task_result["chunks"].append(
                    {
                        "chunk_index": chunk_i,
                        "prediction": prediction[:500],
                        "f1": f1,
                        "execution_time": chunk_time,
                        "usage": usage,
                    }
                )
                print(
                    f"  Task {task_idx}, Chunk {chunk_i + 1}/{num_chunks}: F1={f1:.3f}, time={chunk_time:.1f}s"
                )
            except Exception as e:
                print(f"  Task {task_idx}, Chunk {chunk_i + 1}/{num_chunks}: ERROR: {e}")
                task_result["chunks"].append(
                    {
                        "chunk_index": chunk_i,
                        "error": str(e),
                    }
                )

        results[task_idx] = task_result

    return results


def print_simulation_results(results: dict):
    """Pretty-print simulation results."""
    print("\n" + "=" * 80)
    print("Streaming OOLONG-Pairs — Simulation (Ground Truth Analysis)")
    print("=" * 80)

    for task_idx, task_result in sorted(results.items()):
        print(f"\nTask {task_idx}: {task_result['question'][:80]}...")
        print(
            f"  Full gold pairs: {task_result['chunks'][-1]['full_gold_pairs'] if task_result['chunks'] else 0}"
        )
        print(f"  {'Chunk':>6} {'Chars':>10} {'Users':>6} {'Pairs':>6} {'Discoverable':>12}")
        for c in task_result["chunks"]:
            print(
                f"  {c['chunk_index'] + 1:>6} {c['cumulative_chars']:>10,} {c['users_visible']:>6} "
                f"{c['partial_gold_pairs']:>6} {c['discoverable_fraction']:>11.1%}"
            )


def print_streaming_comparison(persistent_results: dict, non_persistent_results: dict):
    """Compare persistent vs non-persistent streaming results."""
    print("\n" + "=" * 80)
    print("Streaming OOLONG-Pairs — Persistent vs Non-Persistent Comparison")
    print("=" * 80)

    for task_idx in sorted(set(persistent_results.keys()) & set(non_persistent_results.keys())):
        p_task = persistent_results[task_idx]
        np_task = non_persistent_results[task_idx]

        print(f"\nTask {task_idx}:")
        print(
            f"  {'Chunk':>6} {'P-F1':>8} {'NP-F1':>8} {'P-Tokens':>12} {'NP-Tokens':>12} {'Token Savings':>14}"
        )

        p_total_tokens = 0
        np_total_tokens = 0

        for p_chunk, np_chunk in zip(p_task["chunks"], np_task["chunks"], strict=True):
            p_f1 = p_chunk.get("f1", 0)
            np_f1 = np_chunk.get("f1", 0)

            p_usage = p_chunk.get("usage", {})
            np_usage = np_chunk.get("usage", {})

            p_tokens = sum(
                s.get("total_input_tokens", 0) + s.get("total_output_tokens", 0)
                for s in p_usage.get("model_usage_summaries", {}).values()
            )
            np_tokens = sum(
                s.get("total_input_tokens", 0) + s.get("total_output_tokens", 0)
                for s in np_usage.get("model_usage_summaries", {}).values()
            )

            p_total_tokens += p_tokens
            np_total_tokens += np_tokens

            savings = (np_tokens - p_tokens) / np_tokens * 100 if np_tokens > 0 else 0
            print(
                f"  {p_chunk['chunk_index'] + 1:>6} {p_f1:>8.3f} {np_f1:>8.3f} "
                f"{p_tokens:>12,} {np_tokens:>12,} {savings:>13.1f}%"
            )

        total_savings = (
            (np_total_tokens - p_total_tokens) / np_total_tokens * 100 if np_total_tokens > 0 else 0
        )
        print(
            f"  {'TOTAL':>6} {'':>8} {'':>8} {p_total_tokens:>12,} {np_total_tokens:>12,} {total_savings:>13.1f}%"
        )


def main():
    parser = argparse.ArgumentParser(description="Streaming OOLONG-Pairs Benchmark")
    parser.add_argument(
        "--mode",
        choices=["persistent", "non-persistent", "both", "simulate"],
        required=True,
        help="Which mode to run",
    )
    parser.add_argument("--num-chunks", type=int, default=3, help="Number of context chunks")
    parser.add_argument(
        "--tasks", type=str, default="1,2,3", help="Comma-separated task indices (1-indexed)"
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--context-len", type=int, default=32768)
    parser.add_argument("--gold-file", type=str, default=None)
    args = parser.parse_args()

    task_indices = [int(t) for t in args.tasks.split(",")]

    print(f"Loading OOLONG-Pairs data (context_len={args.context_len})...")
    data = load_oolong_pairs_data(context_len=args.context_len, gold_file=args.gold_file)
    context = data["context"]
    labeled_context = data["labeled_context"]

    print(f"Context length: {len(context):,} chars")
    print(f"Tasks: {task_indices}")
    print(f"Chunks: {args.num_chunks}")

    if args.mode == "simulate":
        results = simulate_streaming(context, labeled_context, task_indices, args.num_chunks)
        print_simulation_results(results)

        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            # Convert results to serializable format
            serializable = {}
            for k, v in results.items():
                serializable[str(k)] = v
            with open(args.output, "w") as f:
                json.dump(serializable, f, indent=2)
            print(f"\nResults saved to {args.output}")

    elif args.mode == "persistent":
        results = run_persistent_streaming(context, labeled_context, task_indices, args.num_chunks)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    elif args.mode == "non-persistent":
        results = run_non_persistent_streaming(
            context, labeled_context, task_indices, args.num_chunks
        )
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    elif args.mode == "both":
        print("\n--- Running Persistent Mode ---")
        p_results = run_persistent_streaming(
            context, labeled_context, task_indices, args.num_chunks
        )
        print("\n--- Running Non-Persistent Mode ---")
        np_results = run_non_persistent_streaming(
            context, labeled_context, task_indices, args.num_chunks
        )
        print_streaming_comparison(p_results, np_results)

        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(
                    {
                        "persistent": {str(k): v for k, v in p_results.items()},
                        "non_persistent": {str(k): v for k, v in np_results.items()},
                    },
                    f,
                    indent=2,
                )


if __name__ == "__main__":
    main()
