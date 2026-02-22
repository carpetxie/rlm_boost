"""
F1 Progression Experiment — Iteration 8 Main Deliverable.

Runs the three-condition comparison required to make F1=0.54 interpretable:

  Condition A: Incremental RLM (k=5, 5K chars/chunk)
    → F1 snapshot after each chunk (F1(k=1), F1(k=2), ..., F1(k=5))
    → Uses persistent=True + deduplication guard (Iteration 8 fix)
    → Produces the F1 progression curve showing dynamic context accumulation

  Condition B: Non-incremental RLM baseline (1 turn, 5K chars = chunk 0 only)
    → matched-budget baseline: same first-chunk budget as Condition A
    → Shows what a single-turn model achieves with the same context size

  Condition C: Non-incremental RLM oracle (1 turn, all 25K chars concatenated)
    → Upper bound: what a single-turn model achieves with full context
    → Estimated F1 ~0.77 from Experiment 1 (to be re-confirmed here)

The paper claim becomes:
  "Incremental RLM at k=5 achieves [X]% of oracle F1 using 1/5 of the
   oracle's single-turn context cost, by processing 5K chars/turn."

Usage:
    export OPENAI_API_KEY=sk-...
    python eval/f1_progression_experiment.py
    python eval/f1_progression_experiment.py --model gpt-4o-mini --num-chunks 5
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Shared imports
# ---------------------------------------------------------------------------

from eval.rlm_pipeline_experiment import (
    TASK_1_CHECKER_SETUP,
    compute_f1,
    compute_gold_pairs,
    load_oolong_data,
    split_context_by_users,
)


# ---------------------------------------------------------------------------
# Condition A: Incremental RLM with per-chunk F1 snapshots
# ---------------------------------------------------------------------------

# Explicit code template prompt (v3 design) with clear chunk instructions.
# This maximally constrains the model's behavior to follow the protocol.
CHUNK_PROMPT_INCREMENTAL = """Task (OOLONG-Pairs Task 1): Find pairs of users BOTH with >= 1 instance.
Chunk {chunk_num} of {total_chunks}.

Context format: each line is "Date: [date] || User: [user_id] || Instance: [text] || Label: [cat]"
Entity ID = the numeric user_id (e.g. "44436"), NOT the date.

Run this code (using the actual chunk index {chunk_idx}):

```repl
import re
entities = {{}}
for line in context_{chunk_idx}.split('\\n'):
    m = re.search(r'User: (\\d+)', line)
    if m:
        uid = m.group(1)
        if uid not in entities:
            entities[uid] = {{"instances": []}}
        entities[uid]["instances"].append(line.strip())
stats = _incremental.process_chunk({chunk_idx}, entities, pair_checker=check_pair)
pair_results = list(_incremental.pair_tracker.get_pairs())
print(f"Chunk {chunk_num}: {{stats['new_entities']}} new entities, {{stats['total_pairs']}} pairs, {{stats['pair_checks']}} checks")
```

After the repl block runs successfully, return FINAL_VAR(pair_results).
Do NOT re-read context_0, context_1 etc. on this turn.
"""


def run_condition_a_incremental(
    plain_context: str,
    labeled_context: str,
    gold_pairs: set,
    api_key: str,
    num_chunks: int = 5,
    max_chunk_chars: int = 5000,
    model: str = "gpt-4o-mini",
    verbose: bool = False,
) -> dict:
    """
    Condition A: Incremental RLM — snapshot F1 after each chunk.

    Uses persistent=True with the deduplication guard (Iteration 8 fix).
    After each turn, reads pair_tracker directly from the REPL to get F1.
    """
    from rlm.core.rlm import RLM
    from rlm.utils.prompts import INCREMENTAL_SYSTEM_PROMPT

    os.environ["OPENAI_API_KEY"] = api_key

    print(f"\n{'=' * 70}")
    print(f"CONDITION A: Incremental RLM (k={num_chunks}, {max_chunk_chars} chars/chunk)")
    print(f"{'=' * 70}")

    chunks = split_context_by_users(plain_context, num_chunks)
    chunks = [c[:max_chunk_chars] for c in chunks]
    print(f"Chunk sizes: {[len(c) for c in chunks]} chars")

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

    f1_progression = []  # F1(k=1), F1(k=2), ..., F1(k=num_chunks)
    turn_tokens = []
    prev_chunks_processed = 0

    for chunk_i, chunk in enumerate(chunks):
        chunk_num = chunk_i + 1
        root_prompt = CHUNK_PROMPT_INCREMENTAL.format(
            chunk_num=chunk_num,
            total_chunks=num_chunks,
            chunk_idx=chunk_i,
        )

        print(f"\n  --- Turn {chunk_num}/{num_chunks} ---")
        t0 = time.perf_counter()
        completion = rlm.completion(chunk, root_prompt=root_prompt)
        elapsed = time.perf_counter() - t0

        # Snapshot F1 from pair_tracker (not from pair_results variable —
        # pair_tracker is the authoritative state, pair_results may lag by 1 turn)
        env = rlm._persistent_env
        incr = env.locals.get("_incremental") if env and hasattr(env, "locals") else None

        chunks_processed = 0
        direct_pairs = []
        pair_checks_total = 0
        if incr:
            stats = incr.get_stats()
            chunks_processed = stats.get("chunks_processed", 0)
            pair_checks_total = stats.get("total_pair_checks", 0)
            direct_pairs = list(incr.pair_tracker.get_pairs())

        # Check if compliance increased
        compliant = chunks_processed > prev_chunks_processed
        prev_chunks_processed = chunks_processed

        # F1 snapshot at this chunk
        f1_result = compute_f1(direct_pairs, gold_pairs)

        # Token accounting
        usage = completion.usage_summary
        input_tokens = 0
        output_tokens = 0
        if usage:
            usage_dict = usage.to_dict() if hasattr(usage, "to_dict") else {}
            for _model_name, mu in usage_dict.items():
                if isinstance(mu, dict):
                    input_tokens += mu.get("input_tokens", 0)
                    output_tokens += mu.get("output_tokens", 0)
        turn_tokens.append({"turn": chunk_num, "input": input_tokens, "output": output_tokens})

        print(f"    compliant: {compliant}  chunks_processed: {chunks_processed}")
        print(f"    pairs: {len(direct_pairs)}  pair_checks_total: {pair_checks_total}")
        print(f"    F1={f1_result['f1']}  P={f1_result['precision']}  R={f1_result['recall']}")
        print(f"    input_tokens: {input_tokens}  elapsed: {elapsed:.1f}s")

        f1_progression.append({
            "chunk": chunk_num,
            "chunk_idx": chunk_i,
            "compliant": compliant,
            "chunks_processed": chunks_processed,
            "pairs": len(direct_pairs),
            "pair_checks_total": pair_checks_total,
            "f1": f1_result["f1"],
            "precision": f1_result["precision"],
            "recall": f1_result["recall"],
            "tp": f1_result.get("tp"),
            "fp": f1_result.get("fp"),
            "fn": f1_result.get("fn"),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "elapsed_sec": round(elapsed, 2),
        })

    rlm.close()

    compliant_turns = sum(1 for t in f1_progression if t["compliant"])
    compliance_rate = compliant_turns / num_chunks if num_chunks > 0 else 0.0

    print(f"\n  Condition A Summary:")
    print(f"    Compliance: {compliance_rate:.0%} ({compliant_turns}/{num_chunks} turns)")
    print(f"    F1 progression: {[t['f1'] for t in f1_progression]}")
    print(f"    Total input tokens: {sum(t['input'] for t in turn_tokens)}")

    return {
        "condition": "A_incremental",
        "model": model,
        "num_chunks": num_chunks,
        "max_chunk_chars": max_chunk_chars,
        "compliance_rate": compliance_rate,
        "f1_progression": f1_progression,
        "final_f1": f1_progression[-1]["f1"] if f1_progression else None,
        "final_precision": f1_progression[-1]["precision"] if f1_progression else None,
        "final_recall": f1_progression[-1]["recall"] if f1_progression else None,
        "total_input_tokens": sum(t["input"] for t in turn_tokens),
        "total_output_tokens": sum(t["output"] for t in turn_tokens),
        "per_turn_tokens": turn_tokens,
    }


# ---------------------------------------------------------------------------
# Condition B: Non-incremental RLM baseline (matched-budget)
# ---------------------------------------------------------------------------

BASELINE_PROMPT_SINGLE = """Task (OOLONG-Pairs Task 1): Find all pairs of users BOTH with >= 1 instance.

Context format: each line is "Date: [date] || User: [user_id] || Instance: [text] || Label: [cat]"
Entity ID = the numeric user_id (e.g. "44436"), NOT the date.

Run this code:

```repl
import re
entities = {{}}
for line in context.split('\\n'):
    m = re.search(r'User: (\\d+)', line)
    if m:
        uid = m.group(1)
        if uid not in entities:
            entities[uid] = {{"instances": []}}
        entities[uid]["instances"].append(line.strip())

pair_results = []
user_ids = [uid for uid, attrs in entities.items() if len(attrs["instances"]) >= 1]
for i, uid1 in enumerate(sorted(user_ids)):
    for uid2 in sorted(user_ids)[i+1:]:
        pair_results.append((min(uid1, uid2), max(uid1, uid2)))
print(f"Users with >= 1 instance: {{len(user_ids)}}")
print(f"Total pairs: {{len(pair_results)}}")
```

Return FINAL_VAR(pair_results).
"""


def run_condition_b_baseline(
    plain_context: str,
    labeled_context: str,
    gold_pairs: set,
    api_key: str,
    max_chars: int = 5000,
    model: str = "gpt-4o-mini",
    label: str = "B_matched_budget",
    verbose: bool = False,
) -> dict:
    """
    Condition B/C: Non-incremental single-turn RLM.

    B (matched budget): uses only first max_chars of context.
    C (oracle): uses full context (set max_chars = None or very large).
    """
    from rlm.core.rlm import RLM

    os.environ["OPENAI_API_KEY"] = api_key

    context = plain_context[:max_chars] if max_chars else plain_context
    print(f"\n{'=' * 70}")
    print(f"CONDITION {label}: Non-incremental RLM ({len(context)} chars, single turn)")
    print(f"{'=' * 70}")

    rlm = RLM(
        backend="openai",
        backend_kwargs={"model_name": model},
        environment="local",
        environment_kwargs={"setup_code": TASK_1_CHECKER_SETUP},
        persistent=False,
        max_iterations=4,
        verbose=verbose,
    )

    t0 = time.perf_counter()
    completion = rlm.completion(context, root_prompt=BASELINE_PROMPT_SINGLE)
    elapsed = time.perf_counter() - t0
    rlm.close()

    # Extract pair_results from completion response or from the env
    # Since persistent=False, the env is cleaned up — use the completion response
    response_str = str(completion.response) if completion.response else ""

    # Try to parse pairs from response string
    pairs = []
    try:
        # Attempt eval if it looks like a list of tuples
        if response_str.strip().startswith("["):
            parsed = eval(response_str.strip())
            if isinstance(parsed, list):
                pairs = parsed
    except Exception:
        pass

    # If response parsing failed, the model may have returned something else
    if not pairs and response_str:
        print(f"  WARNING: Could not parse pairs from response. Response preview: {response_str[:300]}")

    f1_result = compute_f1(pairs, gold_pairs)

    # Token accounting
    usage = completion.usage_summary
    input_tokens = 0
    output_tokens = 0
    if usage:
        usage_dict = usage.to_dict() if hasattr(usage, "to_dict") else {}
        for _model_name, mu in usage_dict.items():
            if isinstance(mu, dict):
                input_tokens += mu.get("input_tokens", 0)
                output_tokens += mu.get("output_tokens", 0)

    print(f"    F1={f1_result['f1']}  P={f1_result['precision']}  R={f1_result['recall']}")
    print(f"    Predicted pairs: {len(pairs)}  Gold pairs: {len(gold_pairs)}")
    print(f"    input_tokens: {input_tokens}  elapsed: {elapsed:.1f}s")

    return {
        "condition": label,
        "model": model,
        "context_chars": len(context),
        "f1": f1_result["f1"],
        "precision": f1_result["precision"],
        "recall": f1_result["recall"],
        "tp": f1_result.get("tp"),
        "fp": f1_result.get("fp"),
        "fn": f1_result.get("fn"),
        "predicted_pairs": len(pairs),
        "gold_pairs": len(gold_pairs),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "elapsed_sec": round(elapsed, 2),
        "response_preview": response_str[:500],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="F1 Progression Experiment (Iteration 8)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--num-chunks", type=int, default=5)
    parser.add_argument("--max-chunk-chars", type=int, default=5000)
    parser.add_argument("--output", type=str,
                        default="results/streaming/f1_progression_results.json")
    parser.add_argument("--incremental-only", action="store_true",
                        help="Skip condition B/C (saves API cost)")
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

    print("Loading OOLONG-Pairs data...")
    plain_context, labeled_context = load_oolong_data()
    gold_pairs = compute_gold_pairs(labeled_context, task_idx=1)
    print(f"Gold pairs (Task 1, full context): {len(gold_pairs)}")

    results = {"model": args.model, "gold_pairs_count": len(gold_pairs)}

    # Condition A: Incremental (k=5, 5K chars/chunk)
    result_a = run_condition_a_incremental(
        plain_context=plain_context,
        labeled_context=labeled_context,
        gold_pairs=gold_pairs,
        api_key=api_key,
        num_chunks=args.num_chunks,
        max_chunk_chars=args.max_chunk_chars,
        model=args.model,
        verbose=not args.quiet,
    )
    results["condition_a"] = result_a

    if not args.incremental_only:
        # Condition B: Matched-budget baseline (1 turn, 5K chars)
        result_b = run_condition_b_baseline(
            plain_context=plain_context,
            labeled_context=labeled_context,
            gold_pairs=gold_pairs,
            api_key=api_key,
            max_chars=args.max_chunk_chars,  # same as one chunk
            model=args.model,
            label="B_matched_budget",
            verbose=not args.quiet,
        )
        results["condition_b"] = result_b

        # Condition C: Oracle (1 turn, all 25K chars)
        total_chars = args.num_chunks * args.max_chunk_chars  # 25K
        result_c = run_condition_b_baseline(
            plain_context=plain_context,
            labeled_context=labeled_context,
            gold_pairs=gold_pairs,
            api_key=api_key,
            max_chars=total_chars,
            model=args.model,
            label="C_oracle",
            verbose=not args.quiet,
        )
        results["condition_c"] = result_c

    # Summary table
    print(f"\n{'=' * 70}")
    print("FINAL COMPARISON TABLE")
    print(f"{'=' * 70}")
    print(f"{'Condition':<30} {'F1':>6} {'Precision':>10} {'Recall':>8} {'Chars':>10} {'Tokens':>8}")
    print("-" * 70)

    a_final = result_a["final_f1"]
    print(f"{'A: Incremental (k=5, 5K/chunk)':<30} {a_final or 'N/A':>6} "
          f"{result_a['final_precision'] or 'N/A':>10} "
          f"{result_a['final_recall'] or 'N/A':>8} "
          f"{args.num_chunks * args.max_chunk_chars:>10} "
          f"{result_a['total_input_tokens']:>8}")

    if not args.incremental_only:
        print(f"{'B: Baseline (1 turn, 5K)':<30} {result_b['f1'] or 'N/A':>6} "
              f"{result_b['precision'] or 'N/A':>10} "
              f"{result_b['recall'] or 'N/A':>8} "
              f"{args.max_chunk_chars:>10} "
              f"{result_b['input_tokens']:>8}")
        print(f"{'C: Oracle (1 turn, 25K)':<30} {result_c['f1'] or 'N/A':>6} "
              f"{result_c['precision'] or 'N/A':>10} "
              f"{result_c['recall'] or 'N/A':>8} "
              f"{total_chars:>10} "
              f"{result_c['input_tokens']:>8}")

    print(f"\nF1 Progression (Condition A):")
    for t in result_a["f1_progression"]:
        chars_so_far = t["chunk"] * args.max_chunk_chars
        print(f"  k={t['chunk']} ({chars_so_far} chars): F1={t['f1']}  P={t['precision']}  R={t['recall']}  "
              f"pairs={t['pairs']}  tokens={t['input_tokens']}")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    main()
