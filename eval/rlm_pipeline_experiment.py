"""
Full RLM Pipeline Experiment — Execution Compliance + F1 Measurement.

This is the definitive live experiment that the critique (Iteration 7) requires.
It uses the ACTUAL RLM engine (RLM(persistent=True) + LocalREPL) rather than the
custom live_api_experiment.py script, which had three structural defects:
  1. Used ```python``` blocks instead of ```repl``` (silently ignored by find_code_blocks)
  2. Referenced undefined `checker` (would raise NameError in real pipeline)
  3. FINAL() contained Python source code instead of using FINAL_VAR(var_name)

This script fixes all three by:
  1. Using the canonical INCREMENTAL_SYSTEM_PROMPT (which uses ```repl``` throughout)
  2. Pre-injecting `check_pair` via environment setup_code BEFORE the first turn
  3. Instructing the model to use FINAL_VAR(pair_results) via the root_prompt

Metrics collected:
  1. execution_compliance_rate: fraction of turns where _incremental.process_chunk()
     was actually called (measured via _incremental.get_stats()["chunks_processed"])
  2. repl_error_rate: fraction of turns with NameError or other errors in REPL output
  3. f1_vs_gold: pair accuracy after all chunks
  4. per_turn_token_counts: prompt + completion tokens per turn
  5. failure_mode: if execution fails, what error occurred

All three outcomes are publishable:
  - High compliance + high F1 → "Empirical System": zero-shot execution works
  - High text compliance, low execution compliance → "Prompted System": need few-shot
  - Low compliance → "Theoretical + Infrastructure": motivates Thrust 1 fine-tuning

Usage:
    export OPENAI_API_KEY=sk-...
    python eval/rlm_pipeline_experiment.py
    python eval/rlm_pipeline_experiment.py --task 1 --num-chunks 3 --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Task 1 checker — pre-injected into REPL before the first turn.
# This addresses Defect 2 from live_api_experiment.py: `checker` was
# referenced but never defined. Here we inject `check_pair` (matching the
# INCREMENTAL_SYSTEM_PROMPT convention) into the REPL namespace as setup_code.
# The model may redefine it (which is fine and actually tests whether the model
# correctly adapts the checker to the task description).
# ---------------------------------------------------------------------------

TASK_1_CHECKER_SETUP = """
# Pre-injected check_pair for Task 1: symmetric co-appearance.
# Both users must have at least one appearance/instance.
# The model is expected to define its own check_pair — this is a safety fallback
# so that if the model skips defining it, the pipeline still executes.
def check_pair(attrs1, attrs2):
    instances1 = attrs1.get("instances", []) if isinstance(attrs1, dict) else []
    instances2 = attrs2.get("instances", []) if isinstance(attrs2, dict) else []
    return len(instances1) >= 1 and len(instances2) >= 1
"""

# Root prompt per chunk — tells the model what to do on THIS chunk.
# On intermediate chunks: process and report, then FINAL_VAR(pair_results)
# On the final chunk: process and produce definitive answer
CHUNK_ROOT_PROMPT_TEMPLATE = """Task (OOLONG-Pairs, Task 1): Find all pairs of users who BOTH have at least 1 appearance/instance.
This is chunk {chunk_num} of {total_chunks}.

Instructions:
1. Parse the users from context_{chunk_idx} into an entity dict: {{user_id: {{"instances": [...]}}}}
2. Define or reuse `check_pair(attrs1, attrs2) -> bool` (True if both have >= 1 instance)
3. Call: stats = _incremental.process_chunk({chunk_idx}, entities, pair_checker=check_pair)
4. Assign: pair_results = list(_incremental.pair_tracker.get_pairs())
5. Return FINAL_VAR(pair_results)

IMPORTANT: Do NOT re-read context_0, context_1, etc. Only process context_{chunk_idx}.
"""


# ---------------------------------------------------------------------------
# Data loading and chunking
# ---------------------------------------------------------------------------

def load_oolong_data():
    """Load OOLONG-Pairs dataset and return (plain_context, labeled_context)."""
    from datasets import load_dataset
    ds = load_dataset("oolongbench/oolong-synth", split="validation")
    corpus = [
        x for x in ds
        if x["dataset"] == "trec_coarse" and x["context_len"] == 32768
    ][0]
    return corpus["context_window_text"], corpus["context_window_text_with_labels"]


def split_context_by_users(context: str, num_chunks: int) -> list[str]:
    """Split context at user boundaries into num_chunks chunks."""
    import re
    user_pattern = re.compile(r"(?=Date:.*?\|\| User:)")
    positions = [m.start() for m in user_pattern.finditer(context)]

    if not positions:
        chunk_size = len(context) // num_chunks
        return [context[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]

    users_per_chunk = max(1, len(positions) // num_chunks)
    chunks = []
    for i in range(num_chunks):
        start = positions[i * users_per_chunk] if i * users_per_chunk < len(positions) else len(context)
        if i < num_chunks - 1:
            end_idx = min((i + 1) * users_per_chunk, len(positions))
            end = positions[end_idx] if end_idx < len(positions) else len(context)
        else:
            end = len(context)
        if start < end:
            chunks.append(context[start:end])

    while len(chunks) < num_chunks:
        chunks.append("[empty chunk]")
    return chunks[:num_chunks]


def compute_gold_pairs(labeled_context: str, task_idx: int = 1) -> set[tuple]:
    """Compute gold-standard pairs using the real task condition."""
    from eval.utils import _parse_labeled_context, _check_pair_condition
    users = _parse_labeled_context(labeled_context)
    pairs = set()
    for uid1, uid2 in combinations(sorted(users.keys()), 2):
        if _check_pair_condition(users[uid1], users[uid2], task_idx):
            pairs.add((min(uid1, uid2), max(uid1, uid2)))
    return pairs


# ---------------------------------------------------------------------------
# F1 computation
# ---------------------------------------------------------------------------

def compute_f1(predicted: list | set | None, gold: set) -> dict:
    """Compute precision, recall, F1 between predicted and gold pair sets."""
    if predicted is None:
        return {"precision": None, "recall": None, "f1": None, "note": "no pair_results in REPL"}

    # Normalize predicted pairs to frozensets for comparison
    pred_set = set()
    for p in predicted:
        if isinstance(p, (list, tuple)) and len(p) == 2:
            pred_set.add((min(str(p[0]), str(p[1])), max(str(p[0]), str(p[1]))))
        elif isinstance(p, frozenset) and len(p) == 2:
            elems = sorted(str(e) for e in p)
            pred_set.add((elems[0], elems[1]))

    # Normalize gold pairs to same string format
    gold_str = set()
    for g in gold:
        gold_str.add((min(str(g[0]), str(g[1])), max(str(g[0]), str(g[1]))))

    tp = len(pred_set & gold_str)
    fp = len(pred_set - gold_str)
    fn = len(gold_str - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
        "predicted_pairs": len(pred_set),
        "gold_pairs": len(gold_str),
    }


# ---------------------------------------------------------------------------
# Execution compliance measurement
# ---------------------------------------------------------------------------

def measure_execution_compliance(env, turn_idx: int, prev_chunks_processed: int) -> dict:
    """
    Measure execution compliance by inspecting the REPL state after a turn.

    Execution compliance = _incremental.process_chunk() was actually called,
    which we verify by checking chunks_processed increased.

    This is a STRONGER check than text-level compliance (looking for "process_chunk"
    in the response text) because it verifies REPL execution, not just text generation.
    """
    result = {"turn": turn_idx, "compliant": False, "chunks_processed": 0,
              "error": None, "pair_results_exists": False}

    if not hasattr(env, "locals"):
        result["error"] = "No locals accessible"
        return result

    incr = env.locals.get("_incremental")
    if incr is None:
        result["error"] = "NameError: _incremental not in locals"
        return result

    try:
        stats = incr.get_stats()
        chunks_processed = stats.get("chunks_processed", 0)
        result["chunks_processed"] = chunks_processed
        result["compliant"] = chunks_processed > prev_chunks_processed
        result["total_pairs"] = stats.get("total_pairs", 0)
        result["total_retractions"] = stats.get("total_retractions", 0)
        result["pair_checks"] = stats.get("pair_checks", 0)
    except Exception as e:
        result["error"] = f"get_stats() failed: {e}"
        return result

    # Check if pair_results was set in locals
    pair_results = env.locals.get("pair_results")
    result["pair_results_exists"] = pair_results is not None
    result["pair_results_count"] = len(pair_results) if pair_results is not None else 0

    return result


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_rlm_pipeline_experiment(
    model: str,
    task_idx: int,
    num_chunks: int,
    api_key: str,
    verbose: bool = True,
    max_chunk_chars: int = 4000,
) -> dict:
    """
    Run the actual RLM pipeline with LocalREPL and measure execution compliance + F1.

    Uses RLM(persistent=True) with the canonical INCREMENTAL_SYSTEM_PROMPT.
    Does NOT use live_api_experiment.py which had three structural defects.
    """
    from rlm.core.rlm import RLM
    from rlm.utils.prompts import INCREMENTAL_SYSTEM_PROMPT

    print(f"\n{'=' * 70}")
    print(f"RLM PIPELINE EXPERIMENT — Task {task_idx}, {num_chunks} chunks, {model}")
    print(f"{'=' * 70}")
    print(f"Using: RLM(persistent=True, environment='local', custom_system_prompt=INCREMENTAL_SYSTEM_PROMPT)")
    print(f"Defect fixes applied:")
    print(f"  1. ```repl``` blocks: canonical INCREMENTAL_SYSTEM_PROMPT uses them throughout")
    print(f"  2. check_pair injected: pre-injected via setup_code before first turn")
    print(f"  3. FINAL_VAR: root_prompt instructs model to use FINAL_VAR(pair_results)")

    # Set API key
    os.environ["OPENAI_API_KEY"] = api_key

    # Load data
    print("\nLoading OOLONG-Pairs data...")
    plain_context, labeled_context = load_oolong_data()
    chunks = split_context_by_users(plain_context, num_chunks)
    print(f"Split into {num_chunks} chunks: {[len(c) for c in chunks]} chars")

    # Truncate chunks to keep cost low
    chunks = [c[:max_chunk_chars] for c in chunks]
    print(f"Truncated to {max_chunk_chars} chars/chunk: {[len(c) for c in chunks]}")

    # Compute gold pairs on FULL labeled context (not truncated)
    print("Computing gold pairs (full labeled context)...")
    gold_pairs = compute_gold_pairs(labeled_context, task_idx)
    print(f"Gold pairs: {len(gold_pairs)}")

    # Create RLM with:
    # - INCREMENTAL_SYSTEM_PROMPT as custom system prompt (uses ```repl``` correctly)
    # - setup_code to pre-inject check_pair (fixes Defect 2)
    # - persistent=True for multi-turn state
    rlm = RLM(
        backend="openai",
        backend_kwargs={"model_name": model},
        environment="local",
        environment_kwargs={"setup_code": TASK_1_CHECKER_SETUP},
        persistent=True,
        custom_system_prompt=INCREMENTAL_SYSTEM_PROMPT,
        max_iterations=6,
        verbose=verbose,
    )

    turn_results = []
    prev_chunks_processed = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for chunk_i, chunk in enumerate(chunks):
        chunk_num = chunk_i + 1
        is_final = chunk_num == num_chunks

        root_prompt = CHUNK_ROOT_PROMPT_TEMPLATE.format(
            chunk_num=chunk_num,
            total_chunks=num_chunks,
            chunk_idx=chunk_i,
        )

        print(f"\n--- Turn {chunk_num}/{num_chunks} ---")
        print(f"Providing chunk {chunk_i} ({len(chunk)} chars)...")

        t0 = time.perf_counter()
        completion = rlm.completion(chunk, root_prompt=root_prompt)
        elapsed = time.perf_counter() - t0

        print(f"  Final answer: {str(completion.response)[:200]}")

        # Measure execution compliance
        compliance = measure_execution_compliance(
            rlm._persistent_env, chunk_i, prev_chunks_processed
        )
        prev_chunks_processed = compliance.get("chunks_processed", prev_chunks_processed)

        # Extract token counts from usage summary
        usage = completion.usage_summary
        turn_input_tokens = 0
        turn_output_tokens = 0
        if usage:
            usage_dict = usage.to_dict() if hasattr(usage, "to_dict") else {}
            # Sum across all models in the usage summary
            for model_name, model_usage in usage_dict.items():
                if isinstance(model_usage, dict):
                    turn_input_tokens += model_usage.get("input_tokens", 0)
                    turn_output_tokens += model_usage.get("output_tokens", 0)

        total_input_tokens += turn_input_tokens
        total_output_tokens += turn_output_tokens

        turn_result = {
            "turn": chunk_num,
            "chunk_idx": chunk_i,
            "chunk_chars": len(chunk),
            "elapsed_sec": round(elapsed, 2),
            "response_preview": str(completion.response)[:300],
            "execution_compliance": compliance,
            "input_tokens": turn_input_tokens,
            "output_tokens": turn_output_tokens,
            "is_final_chunk": is_final,
        }
        turn_results.append(turn_result)

        print(f"  Execution compliant: {compliance['compliant']}")
        print(f"  chunks_processed: {compliance['chunks_processed']}")
        print(f"  pair_results count: {compliance['pair_results_count']}")
        if compliance.get("error"):
            print(f"  ERROR: {compliance['error']}")

    # Read final pair_results from the REPL environment
    final_pair_results = None
    if rlm._persistent_env and hasattr(rlm._persistent_env, "locals"):
        final_pair_results = rlm._persistent_env.locals.get("pair_results")

    # Compute F1
    f1_result = compute_f1(final_pair_results, gold_pairs)

    # Execution compliance summary
    compliant_turns = sum(1 for t in turn_results if t["execution_compliance"]["compliant"])
    total_turns_measured = len(turn_results)
    # Compliance is measured for ALL turns (not just turn 2+) since we're using
    # process_chunk() on every turn including the first
    execution_compliance_rate = compliant_turns / total_turns_measured if total_turns_measured > 0 else 0.0

    # Check for REPL errors (NameError for _incremental etc.)
    error_turns = [t for t in turn_results if t["execution_compliance"].get("error")]

    # Framing determination
    if execution_compliance_rate >= 0.5:
        if f1_result["f1"] is not None and f1_result["f1"] >= 0.5:
            framing = "EMPIRICAL_SYSTEM"
        else:
            framing = "COMPLIANCE_OK_ACCURACY_LOW"
    elif execution_compliance_rate > 0:
        framing = "PARTIAL_COMPLIANCE"
    else:
        framing = "TEXT_LEVEL_ONLY"

    print(f"\n{'=' * 70}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"Execution compliance rate:  {execution_compliance_rate:.0%} ({compliant_turns}/{total_turns_measured} turns)")
    print(f"REPL error turns:           {len(error_turns)}")
    print(f"F1 vs gold:                 {f1_result.get('f1', 'N/A')}")
    print(f"  Precision:                {f1_result.get('precision', 'N/A')}")
    print(f"  Recall:                   {f1_result.get('recall', 'N/A')}")
    print(f"  Predicted pairs:          {f1_result.get('predicted_pairs', 'N/A')}")
    print(f"  Gold pairs:               {f1_result.get('gold_pairs', 'N/A')}")
    print(f"Total input tokens:         {total_input_tokens}")
    print(f"Total output tokens:        {total_output_tokens}")
    if len(turn_results) >= 2:
        t1_in = turn_results[0]["input_tokens"]
        t2plus_in = sum(t["input_tokens"] for t in turn_results[1:])
        total_in = t1_in + t2plus_in
        if total_in > 0:
            print(f"Turn 1 token fraction:      {t1_in/total_in:.1%}")
            print(f"Turn 2+ token fraction:     {t2plus_in/total_in:.1%}")
    print(f"Contribution framing:       {framing}")

    # Clean up
    rlm.close()

    return {
        "model": model,
        "task_idx": task_idx,
        "num_chunks": num_chunks,
        "max_chunk_chars": max_chunk_chars,
        "gold_pairs_count": len(gold_pairs),
        "execution_compliance_rate": round(execution_compliance_rate, 4),
        "compliant_turns": compliant_turns,
        "total_turns": total_turns_measured,
        "error_turns": len(error_turns),
        "f1": f1_result,
        "final_pair_results_count": len(final_pair_results) if final_pair_results is not None else None,
        "token_analysis": {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "per_turn": [
                {"turn": t["turn"], "input": t["input_tokens"], "output": t["output_tokens"]}
                for t in turn_results
            ],
        },
        "framing": framing,
        "turn_results": turn_results,
    }


def main():
    parser = argparse.ArgumentParser(description="RLM Pipeline Execution Compliance + F1 Experiment")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--task", type=int, default=1)
    parser.add_argument("--num-chunks", type=int, default=3)
    parser.add_argument("--max-chunk-chars", type=int, default=4000,
                        help="Max chars per chunk (truncated for cost control)")
    parser.add_argument("--output", type=str,
                        default="results/streaming/rlm_pipeline_results.json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # Load API key
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("OPENAI_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break
    if not api_key:
        print("ERROR: No OPENAI_API_KEY found")
        sys.exit(1)

    result = run_rlm_pipeline_experiment(
        model=args.model,
        task_idx=args.task,
        num_chunks=args.num_chunks,
        api_key=api_key,
        verbose=not args.quiet,
        max_chunk_chars=args.max_chunk_chars,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")

    return result


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Improved root prompt with explicit entity format description
# ---------------------------------------------------------------------------

CHUNK_ROOT_PROMPT_V2_TEMPLATE = """Task (OOLONG-Pairs, Task 1): Find all pairs of users who BOTH have at least 1 appearance.
This is chunk {chunk_num} of {total_chunks}.

CONTEXT FORMAT: Each line looks like:
  Date: [date] || User: [user_id] || Instance: [question] || Label: [category]
The ENTITY ID is the User number (e.g., "44436"), NOT the date.
One user can appear on multiple lines (multiple instances).

Instructions:
1. Parse context_{chunk_idx} line by line. For each line matching "|| User: [user_id] ||":
   - entity_id = the user_id number string (e.g. "44436")
   - attrs = {{"instances": [line]}}  (group all lines per user_id)
2. Call: stats = _incremental.process_chunk({chunk_idx}, entities, pair_checker=check_pair)
   where entities = {{user_id: {{"instances": [list_of_lines]}}}}
3. After process_chunk: pair_results = list(_incremental.pair_tracker.get_pairs())
4. Return FINAL_VAR(pair_results)

DO NOT re-read context_0, context_1, etc. Only process context_{chunk_idx}.
"""
