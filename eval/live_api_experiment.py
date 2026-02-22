"""
Live API Experiment — Incremental Protocol Compliance Test.

This is the mandatory live LLM experiment, deferred across 5 prior iterations.
It measures whether a real LLM (gpt-4o-mini) will follow the incremental
computation protocol described in the INCREMENTAL_SYSTEM_PROMPT.

Experiment design:
    - Task: OOLONG-Pairs Task 1 (symmetric co-appearance)
    - 3 chunks of context
    - Model: gpt-4o-mini (low cost, ~$2–5 total)
    - persistent=True (multi-turn, history is preserved)

Metrics collected:
    1. compliance_rate: fraction of Turn 2+ model responses containing
       "process_chunk" (following the incremental protocol)
    2. reread_rate: fraction of Turn 2+ responses referencing "context_0"
       or the raw context variable (a compliance failure — model re-reading)
    3. F1 of final pair_results vs gold (accuracy)
    4. token_turn1, token_turn2, token_turn3 (validates 78/22 weight assumption)

Both outcomes are publishable:
    - If compliance >= 50%: "LLMs can follow incremental protocol zero-shot"
    - If compliance < 50%: "Zero-shot LLMs don't naturally follow incremental
      protocols, motivating fine-tuning (Thrust 1) as a prerequisite"

Usage:
    export OPENAI_API_KEY=sk-...
    python eval/live_api_experiment.py
    python eval/live_api_experiment.py --task 1 --num-chunks 3 --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Incremental system prompt (task-adapted version of INCREMENTAL_SYSTEM_PROMPT)
# ---------------------------------------------------------------------------

INCREMENTAL_SYSTEM_PROMPT_TEMPLATE = """You are an intelligent data processing assistant implementing an INCREMENTAL computation protocol.

You have access to a Python REPL where you can run code. Use Python code blocks to perform computation.

## INCREMENTAL PROTOCOL

You are given context data ONE CHUNK AT A TIME. On each turn:
1. Process ONLY the new chunk provided in this turn
2. Use `_incremental.process_chunk()` to update your running state
3. Do NOT re-read or re-process chunks from previous turns
4. The `_incremental` object maintains all state across turns

Available tools in the REPL:
```python
# _incremental is pre-injected into the REPL
_incremental.process_chunk(chunk_index, entity_dict, pair_checker=checker_fn)
# Returns: {{'new_entities': int, 'updated_entities': int, 'pair_checks': int, ...}}
```

## TASK

{task_description}

## ENTITY FORMAT

Each entity (user) has a dict of attributes with their "instances" (list of appearances).

## OUTPUT FORMAT

After processing ALL chunks, output:
```python
FINAL(str(sorted(_incremental.pair_tracker.get_pairs())))
```

## IMPORTANT

- On Turn 1: Initialize state and process Chunk 1
- On Turn 2+: Call `_incremental.process_chunk()` with the NEW chunk only
- Never reference `context_0`, `chunk_0`, or prior raw context variables
- The incremental state automatically handles retractions and updates
"""

TASK_1_DESCRIPTION = """Find all pairs of users who BOTH appear in the dataset.
Two users form a valid pair if user A appears at least once AND user B appears at least once.
(Task 1: symmetric co-appearance — both users must have at least one appearance.)

A pair checker function is available:
```python
def checker(attrs1, attrs2):
    return len(attrs1.get("instances", [])) >= 1 and len(attrs2.get("instances", [])) >= 1
```
"""

TURN_PROMPT_TEMPLATE = """Here is Chunk {chunk_num} of {total_chunks}:

```text
{chunk_content}
```

Parse the users from this chunk. For each user, extract their instances (appearances).
Then call `_incremental.process_chunk({chunk_idx}, entity_dict, pair_checker=checker)`.

{final_instruction}
"""

TURN_FINAL_INSTRUCTION = """After calling process_chunk, call:
```python
FINAL(str(sorted(_incremental.pair_tracker.get_pairs())))
```
"""

TURN_NONFINAL_INSTRUCTION = """After processing, report:
- Number of new entities added
- Number of pairs found so far
- Current total in _incremental.pair_tracker
"""


# ---------------------------------------------------------------------------
# Simple direct LLM caller (no full RLM orchestration)
# ---------------------------------------------------------------------------

def call_openai(
    messages: list[dict],
    model: str,
    api_key: str,
    temperature: float = 0.0,
) -> tuple[str, dict]:
    """Call OpenAI API and return (content, usage)."""
    import urllib.request
    import urllib.error

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 4096,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return content, usage
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        raise RuntimeError(f"OpenAI API error {e.code}: {error_body}")


# ---------------------------------------------------------------------------
# Data loading and chunking
# ---------------------------------------------------------------------------

def load_data():
    from datasets import load_dataset
    ds = load_dataset("oolongbench/oolong-synth", split="validation")
    corpus = [x for x in ds if x["dataset"] == "trec_coarse" and x["context_len"] == 32768][0]
    return corpus["context_window_text"], corpus["context_window_text_with_labels"]


def split_context(context: str, num_chunks: int) -> list[str]:
    import re
    user_pattern = re.compile(r"(?=Date:.*?\|\| User:)")
    positions = [m.start() for m in user_pattern.finditer(context)]
    if not positions:
        chunk_size = len(context) // num_chunks
        return [context[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]
    users_per_chunk = max(1, len(positions) // num_chunks)
    chunks = []
    for i in range(num_chunks):
        start = positions[i*users_per_chunk] if i*users_per_chunk < len(positions) else len(context)
        if i < num_chunks - 1:
            end_idx = min((i+1)*users_per_chunk, len(positions))
            end = positions[end_idx] if end_idx < len(positions) else len(context)
        else:
            end = len(context)
        if start < end:
            chunks.append(context[start:end])
    while len(chunks) < num_chunks:
        chunks.append("[empty chunk]")
    return chunks[:num_chunks]


def get_gold_pairs(labeled_context: str) -> set[tuple]:
    """Get gold-standard pairs for Task 1 (both users appear at least once)."""
    from eval.utils import _parse_labeled_context, _check_pair_condition
    from itertools import combinations
    users = _parse_labeled_context(labeled_context)
    pairs = set()
    for uid1, uid2 in combinations(sorted(users.keys()), 2):
        if _check_pair_condition(users[uid1], users[uid2], 1):
            pairs.add((min(uid1, uid2), max(uid1, uid2)))
    return pairs


# ---------------------------------------------------------------------------
# Compliance measurement
# ---------------------------------------------------------------------------

def measure_compliance(responses: list[str]) -> dict:
    """Measure protocol compliance from Turn 2+ responses."""
    if len(responses) < 2:
        return {"compliance_rate": None, "reread_rate": None, "n_measured_turns": 0}

    later_responses = responses[1:]  # Turn 2+
    n = len(later_responses)

    has_process_chunk = [("process_chunk" in r) for r in later_responses]
    has_reread = [
        ("context_0" in r or "chunk_0" in r or "original_context" in r or "full_context" in r)
        for r in later_responses
    ]

    compliance_rate = sum(has_process_chunk) / n if n > 0 else None
    reread_rate = sum(has_reread) / n if n > 0 else None

    return {
        "compliance_rate": compliance_rate,
        "reread_rate": reread_rate,
        "n_measured_turns": n,
        "compliant_turns": sum(has_process_chunk),
        "reread_turns": sum(has_reread),
        "per_turn": [
            {
                "turn": i + 2,
                "has_process_chunk": has_process_chunk[i],
                "has_reread": has_reread[i],
            }
            for i in range(n)
        ],
    }


def extract_final_pairs(response: str) -> set[tuple] | None:
    """Try to extract pair results from a FINAL(...) call in the response."""
    import re
    # Look for FINAL(...) pattern
    m = re.search(r"FINAL\(([^)]+)\)", response)
    if not m:
        return None
    try:
        pairs_str = m.group(1)
        # Try to eval as a list of tuples
        pairs = eval(pairs_str)
        return set(tuple(p) for p in pairs)
    except Exception:
        return None


def compute_f1(predicted: set | None, gold: set) -> dict:
    """Compute F1 between predicted and gold pair sets."""
    if predicted is None:
        return {"precision": None, "recall": None, "f1": None, "note": "no FINAL() found"}
    if not gold:
        return {"precision": 1.0 if not predicted else 0.0, "recall": 1.0, "f1": 1.0}

    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "predicted_pairs": len(predicted),
        "gold_pairs": len(gold),
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_live_experiment(
    model: str,
    task_idx: int,
    num_chunks: int,
    api_key: str,
    verbose: bool = True,
) -> dict:
    """Run the live incremental protocol compliance experiment."""

    print(f"\n{'=' * 70}")
    print(f"LIVE API EXPERIMENT — Task {task_idx}, {num_chunks} chunks, {model}")
    print(f"{'=' * 70}")

    # Load data
    print("Loading OOLONG-Pairs data...")
    plain_context, labeled_context = load_data()

    # Get gold pairs
    print("Computing gold pairs...")
    gold_pairs = get_gold_pairs(labeled_context)
    print(f"Gold pairs: {len(gold_pairs)}")

    # Split context
    chunks = split_context(plain_context, num_chunks)
    print(f"Context split into {len(chunks)} chunks: {[len(c) for c in chunks]} chars")

    # Build conversation
    system_prompt = INCREMENTAL_SYSTEM_PROMPT_TEMPLATE.format(
        task_description=TASK_1_DESCRIPTION
    )

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    responses: list[str] = []
    usages: list[dict] = []
    turn_times: list[float] = []

    for chunk_i, chunk in enumerate(chunks):
        chunk_num = chunk_i + 1
        is_final = chunk_num == num_chunks

        turn_prompt = TURN_PROMPT_TEMPLATE.format(
            chunk_num=chunk_num,
            total_chunks=num_chunks,
            chunk_content=chunk[:3000],  # Truncate chunk to keep costs low
            chunk_idx=chunk_i,
            final_instruction=TURN_FINAL_INSTRUCTION if is_final else TURN_NONFINAL_INSTRUCTION,
        )

        messages.append({"role": "user", "content": turn_prompt})

        print(f"\n--- Turn {chunk_num}/{num_chunks} ---")
        print(f"Sending {len(turn_prompt)} chars to {model}...")

        t0 = time.perf_counter()
        response, usage = call_openai(messages, model, api_key)
        elapsed = time.perf_counter() - t0
        turn_times.append(elapsed)

        if verbose:
            print(f"Response ({len(response)} chars, {elapsed:.1f}s):")
            print(response[:800] + ("..." if len(response) > 800 else ""))

        messages.append({"role": "assistant", "content": response})
        responses.append(response)
        usages.append(usage)

        print(f"  Tokens: {usage.get('prompt_tokens', '?')} prompt, "
              f"{usage.get('completion_tokens', '?')} completion")
        print(f"  Has 'process_chunk': {'YES' if 'process_chunk' in response else 'NO'}")
        print(f"  Has reread signal:   {'YES' if any(w in response for w in ['context_0', 'chunk_0', 'original_context']) else 'NO'}")

    # Analysis
    print(f"\n{'=' * 70}")
    print("COMPLIANCE ANALYSIS")
    print(f"{'=' * 70}")
    compliance = measure_compliance(responses)
    print(f"Compliance rate (Turn 2+): {compliance['compliance_rate']:.0%}" if compliance['compliance_rate'] is not None else "N/A")
    print(f"Re-read rate (Turn 2+):    {compliance['reread_rate']:.0%}" if compliance['reread_rate'] is not None else "N/A")
    for pt in compliance.get("per_turn", []):
        print(f"  Turn {pt['turn']}: process_chunk={'✓' if pt['has_process_chunk'] else '✗'}  "
              f"reread={'⚠' if pt['has_reread'] else '✓'}")

    # Extract final pairs from last response
    final_pairs = extract_final_pairs(responses[-1]) if responses else None
    f1_result = compute_f1(final_pairs, gold_pairs)

    print(f"\n{'=' * 70}")
    print("ACCURACY ANALYSIS")
    print(f"{'=' * 70}")
    if f1_result.get("f1") is not None:
        print(f"F1:        {f1_result['f1']:.3f}")
        print(f"Precision: {f1_result['precision']:.3f}")
        print(f"Recall:    {f1_result['recall']:.3f}")
        print(f"Predicted: {f1_result.get('predicted_pairs', '?')} pairs, Gold: {f1_result.get('gold_pairs', '?')} pairs")
    else:
        print(f"No FINAL() found in last response: {f1_result.get('note', 'unknown')}")

    # Token analysis
    print(f"\n{'=' * 70}")
    print("TOKEN ANALYSIS")
    print(f"{'=' * 70}")
    total_prompt_tokens = sum(u.get("prompt_tokens", 0) for u in usages)
    total_completion_tokens = sum(u.get("completion_tokens", 0) for u in usages)
    for i, u in enumerate(usages):
        print(f"  Turn {i+1}: {u.get('prompt_tokens', '?')} prompt + "
              f"{u.get('completion_tokens', '?')} completion tokens")
    if len(usages) >= 2:
        t1_prompt = usages[0].get("prompt_tokens", 0)
        t2plus_prompt = sum(u.get("prompt_tokens", 0) for u in usages[1:])
        total_all_prompt = t1_prompt + t2plus_prompt
        if total_all_prompt > 0:
            t1_fraction = t1_prompt / total_all_prompt
            t2_fraction = t2plus_prompt / total_all_prompt
            print(f"\n  Turn 1 prompt token fraction: {t1_fraction:.1%}")
            print(f"  Turn 2+ prompt token fraction: {t2_fraction:.1%}")
            print(f"  (Paper assumes 78/22 split — actual: {t1_fraction:.0%}/{t2_fraction:.0%})")

    # Compliance framing
    compliance_rate = compliance.get("compliance_rate")
    print(f"\n{'=' * 70}")
    print("CONTRIBUTION FRAMING")
    print(f"{'=' * 70}")
    if compliance_rate is None:
        print("REFRAMING: Only 1 turn ran — not enough to measure compliance.")
    elif compliance_rate >= 0.5:
        print(f"EMPIRICAL SYSTEM: Compliance = {compliance_rate:.0%} ≥ 50%")
        print("→ LLMs follow incremental protocol zero-shot. Lead with empirical savings.")
    else:
        print(f"THEORETICAL + INFRASTRUCTURE: Compliance = {compliance_rate:.0%} < 50%")
        print("→ Zero-shot LLMs don't naturally follow incremental protocol.")
        print("→ Fine-tuning (Thrust 1) is a necessary prerequisite.")
        print("→ This connects both research thrusts into a coherent narrative.")

    result = {
        "model": model,
        "task_idx": task_idx,
        "num_chunks": num_chunks,
        "compliance": compliance,
        "f1": f1_result,
        "token_analysis": {
            "per_turn": [
                {"turn": i+1, "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens")}
                for i, u in enumerate(usages)
            ],
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
        },
        "turn_times_sec": turn_times,
        "raw_responses": responses,  # Full responses for manual analysis
        "gold_pairs_count": len(gold_pairs),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Live API Incremental Protocol Compliance Experiment")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--task", type=int, default=1)
    parser.add_argument("--num-chunks", type=int, default=3)
    parser.add_argument("--output", type=str, default="results/streaming/live_api_results.json")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose response printing")
    args = parser.parse_args()

    # Load API key
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        # Try .env file
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("OPENAI_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break
    if not api_key:
        print("ERROR: No OPENAI_API_KEY found in environment or .env file")
        sys.exit(1)

    result = run_live_experiment(
        model=args.model,
        task_idx=args.task,
        num_chunks=args.num_chunks,
        api_key=api_key,
        verbose=not args.quiet,
    )

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
