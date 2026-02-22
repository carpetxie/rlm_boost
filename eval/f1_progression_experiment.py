"""
F1 Progression Experiment — Iteration 9 (fixed Conditions B/C + token tracking).

Runs the three-condition comparison required to make F1=0.51 interpretable:

  Condition A: Incremental RLM (k=5, 5K chars/chunk)
    → F1 snapshot after each chunk (F1(k=1), F1(k=2), ..., F1(k=5))
    → Uses persistent=True + deduplication guard (Iteration 8 fix)
    → Produces the F1 progression curve showing dynamic context accumulation

  Condition B: Non-incremental RLM baseline (1 turn, 5K chars = chunk 0 only)
    → matched-budget baseline: same first-chunk budget as Condition A
    → Uses same CHUNK_PROMPT_INCREMENTAL template + persistent=True
    → Extracts from env.locals["_incremental"].pair_tracker.get_pairs() (NOT completion.response)
    → Fix from Iteration 9: avoids Failure Mode B (FINAL_VAR skip)

  Condition C: Non-incremental RLM oracle (1 turn, all 25K chars concatenated)
    → Upper bound: what a single-turn model achieves with full context
    → Uses explicit ORACLE_PROMPT_SINGLE template + persistent=True
    → Extracts from env.locals["pair_results"] directly (no _incremental)
    → Fix from Iteration 9: avoids Failure Mode B

The paper claim becomes:
  "Incremental RLM at k=5 achieves [X]% of oracle F1 using 1/5 of the
   oracle's single-turn context cost, by processing 5K chars/turn incrementally."

Iteration 9 fixes:
  1. Token tracking: use usage.model_usage_summaries.values() with .total_input_tokens
     (prev bug: usage.to_dict() gave nested dict; isinstance(mu, dict) true but
      mu.get("input_tokens") always 0 since key was "model_usage_summaries" not model name)
  2. Condition B: persistent=True + CHUNK_PROMPT_INCREMENTAL + env.locals extraction
  3. Condition C: persistent=True + ORACLE_PROMPT_SINGLE + env.locals extraction
  4. --task-idx: generalize to Tasks 3 and 6

Usage:
    export OPENAI_API_KEY=sk-...
    python eval/f1_progression_experiment.py
    python eval/f1_progression_experiment.py --model gpt-4o-mini --num-chunks 5
    python eval/f1_progression_experiment.py --task-idx 3 --incremental-only
    python eval/f1_progression_experiment.py --task-idx 6 --incremental-only
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
# Task-specific checker setups
# ---------------------------------------------------------------------------

# Task 3: Both users have >= 1 instance with description/abstract concept OR abbreviation.
# In plain context (no labels visible), use >= 1 instance as proxy for protocol testing.
# Gold F1 is computed from labeled context with real condition; model's check_pair is
# intentionally simplified to test INCREMENTAL PROTOCOL generalization, not label accuracy.
TASK_3_CHECKER_SETUP = """
# Pre-injected check_pair for Task 3: description/abstract concept OR abbreviation (symmetric).
# Since plain context lacks labels, this uses >= 1 instance as a protocol-generalization proxy.
def check_pair(attrs1, attrs2):
    instances1 = attrs1.get("instances", []) if isinstance(attrs1, dict) else []
    instances2 = attrs2.get("instances", []) if isinstance(attrs2, dict) else []
    return len(instances1) >= 1 and len(instances2) >= 1
"""

# Task 6: Both users have >= 1 instance with location OR abbreviation.
# Same protocol-proxy approach as Task 3.
TASK_6_CHECKER_SETUP = """
# Pre-injected check_pair for Task 6: location OR abbreviation (symmetric).
# Since plain context lacks labels, this uses >= 1 instance as a protocol-generalization proxy.
def check_pair(attrs1, attrs2):
    instances1 = attrs1.get("instances", []) if isinstance(attrs1, dict) else []
    instances2 = attrs2.get("instances", []) if isinstance(attrs2, dict) else []
    return len(instances1) >= 1 and len(instances2) >= 1
"""

TASK_CHECKER_SETUPS = {
    1: TASK_1_CHECKER_SETUP,
    3: TASK_3_CHECKER_SETUP,
    6: TASK_6_CHECKER_SETUP,
}


# ---------------------------------------------------------------------------
# Condition A: Incremental RLM with per-chunk F1 snapshots
# ---------------------------------------------------------------------------

# Explicit code template prompt (v3 design) with clear chunk instructions.
# This maximally constrains the model's behavior to follow the protocol.
CHUNK_PROMPT_INCREMENTAL = """Task (OOLONG-Pairs Task {task_idx}): Find pairs of users BOTH with >= 1 instance.
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

# Condition C oracle: single turn, full 25K chars, explicit pair_results without _incremental.
# Uses context_0 (available via INCREMENTAL_SYSTEM_PROMPT with persistent=True).
ORACLE_PROMPT_SINGLE = """Task (OOLONG-Pairs Task {task_idx}): Find ALL pairs of users BOTH with >= 1 instance.
This is a SINGLE-TURN analysis of the FULL context ({total_chars} chars).

Context format: each line is "Date: [date] || User: [user_id] || Instance: [text] || Label: [cat]"
Entity ID = the numeric user_id (e.g. "44436"), NOT the date.

Run this code exactly:

```repl
import re
entities = {{}}
for line in context_0.split('\\n'):
    m = re.search(r'User: (\\d+)', line)
    if m:
        uid = m.group(1)
        if uid not in entities:
            entities[uid] = {{"instances": []}}
        entities[uid]["instances"].append(line.strip())

# Build all qualifying pairs directly (no _incremental needed for single-turn oracle)
user_ids = sorted([uid for uid, attrs in entities.items() if len(attrs["instances"]) >= 1])
pair_results = []
for i, uid1 in enumerate(user_ids):
    for uid2 in user_ids[i+1:]:
        pair_results.append((min(uid1, uid2), max(uid1, uid2)))
print(f"Users with >= 1 instance: {{len(user_ids)}}")
print(f"Total pairs: {{len(pair_results)}}")
```

After the repl block runs, return FINAL_VAR(pair_results).
"""


def _extract_tokens(usage) -> tuple[int, int]:
    """
    Extract input and output token counts from a UsageSummary object.

    Iteration 9 fix: the previous code did:
        usage_dict = usage.to_dict()          # {"model_usage_summaries": {model: {nested}}}
        for model_name, mu in usage_dict.items():
            if isinstance(mu, dict):           # True for the "model_usage_summaries" key
                input_tokens += mu.get("input_tokens", 0)  # Always 0 — wrong key

    The bug: usage.to_dict() returns {"model_usage_summaries": {...}}, so the loop gets
    ("model_usage_summaries", nested_dict) not (model_name, ModelUsageSummary). Even if
    you reached the model level, the key is "total_input_tokens" not "input_tokens".

    Fix: use usage.model_usage_summaries attribute directly (object access, not dict).
    """
    input_tokens = 0
    output_tokens = 0
    if usage is None:
        return input_tokens, output_tokens

    # Primary path: attribute access on UsageSummary object
    if hasattr(usage, "model_usage_summaries"):
        for mu in usage.model_usage_summaries.values():
            if hasattr(mu, "total_input_tokens"):
                input_tokens += mu.total_input_tokens
                output_tokens += mu.total_output_tokens
    else:
        # Fallback: try to_dict() with corrected nested access
        try:
            usage_dict = usage.to_dict()
            nested = usage_dict.get("model_usage_summaries", {})
            for mu in nested.values():
                if isinstance(mu, dict):
                    input_tokens += mu.get("total_input_tokens", 0)
                    output_tokens += mu.get("total_output_tokens", 0)
        except Exception:
            pass

    return input_tokens, output_tokens


def run_condition_a_incremental(
    plain_context: str,
    labeled_context: str,
    gold_pairs: set,
    api_key: str,
    num_chunks: int = 5,
    max_chunk_chars: int = 5000,
    model: str = "gpt-4o-mini",
    task_idx: int = 1,
    checker_setup: str = None,
    verbose: bool = False,
) -> dict:
    """
    Condition A: Incremental RLM — snapshot F1 after each chunk.

    Uses persistent=True with the deduplication guard (Iteration 8 fix).
    After each turn, reads pair_tracker directly from the REPL to get F1.

    Iteration 9: fixed token tracking via _extract_tokens().
    """
    from rlm.core.rlm import RLM
    from rlm.utils.prompts import INCREMENTAL_SYSTEM_PROMPT

    os.environ["OPENAI_API_KEY"] = api_key
    if checker_setup is None:
        checker_setup = TASK_CHECKER_SETUPS.get(task_idx, TASK_1_CHECKER_SETUP)

    print(f"\n{'=' * 70}")
    print(f"CONDITION A: Incremental RLM (k={num_chunks}, {max_chunk_chars} chars/chunk, Task {task_idx})")
    print(f"{'=' * 70}")

    chunks = split_context_by_users(plain_context, num_chunks)
    chunks = [c[:max_chunk_chars] for c in chunks]
    print(f"Chunk sizes: {[len(c) for c in chunks]} chars")

    rlm = RLM(
        backend="openai",
        backend_kwargs={"model_name": model},
        environment="local",
        environment_kwargs={"setup_code": checker_setup},
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
            task_idx=task_idx,
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
        prune_count = 0
        if incr:
            stats = incr.get_stats()
            chunks_processed = stats.get("chunks_processed", 0)
            pair_checks_total = stats.get("total_pair_checks", 0)
            direct_pairs = list(incr.pair_tracker.get_pairs())

        # Check HistoryManager prune_count (Iteration 9: new telemetry)
        # Note: attribute is rlm.history_manager (no leading underscore)
        if hasattr(rlm, "history_manager") and rlm.history_manager is not None:
            hm_stats = rlm.history_manager.get_stats()
            prune_count = hm_stats.get("prune_count", 0)

        # Check if compliance increased
        compliant = chunks_processed > prev_chunks_processed
        prev_chunks_processed = chunks_processed

        # F1 snapshot at this chunk
        f1_result = compute_f1(direct_pairs, gold_pairs)

        # Token accounting — Iteration 9 fix
        input_tokens, output_tokens = _extract_tokens(completion.usage_summary)
        turn_tokens.append({"turn": chunk_num, "input": input_tokens, "output": output_tokens})

        print(f"    compliant: {compliant}  chunks_processed: {chunks_processed}  prune_count: {prune_count}")
        print(f"    pairs: {len(direct_pairs)}  pair_checks_total: {pair_checks_total}")
        print(f"    F1={f1_result['f1']}  P={f1_result['precision']}  R={f1_result['recall']}")
        print(f"    input_tokens: {input_tokens}  output_tokens: {output_tokens}  elapsed: {elapsed:.1f}s")

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
            "prune_count": prune_count,
        })

    rlm.close()

    compliant_turns = sum(1 for t in f1_progression if t["compliant"])
    compliance_rate = compliant_turns / num_chunks if num_chunks > 0 else 0.0

    total_input = sum(t["input"] for t in turn_tokens)
    total_output = sum(t["output"] for t in turn_tokens)

    print(f"\n  Condition A Summary (Task {task_idx}):")
    print(f"    Compliance: {compliance_rate:.0%} ({compliant_turns}/{num_chunks} turns)")
    print(f"    F1 progression: {[t['f1'] for t in f1_progression]}")
    print(f"    Total input tokens: {total_input}  Total output tokens: {total_output}")

    return {
        "condition": "A_incremental",
        "task_idx": task_idx,
        "model": model,
        "num_chunks": num_chunks,
        "max_chunk_chars": max_chunk_chars,
        "compliance_rate": compliance_rate,
        "f1_progression": f1_progression,
        "final_f1": f1_progression[-1]["f1"] if f1_progression else None,
        "final_precision": f1_progression[-1]["precision"] if f1_progression else None,
        "final_recall": f1_progression[-1]["recall"] if f1_progression else None,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "per_turn_tokens": turn_tokens,
    }


# ---------------------------------------------------------------------------
# Condition B: Non-incremental RLM baseline (matched-budget, 5K chars)
# Iteration 9: use persistent=True + CHUNK_PROMPT_INCREMENTAL + env.locals extraction
# This avoids Failure Mode B (FINAL_VAR skip) by using the same code template as A.
# ---------------------------------------------------------------------------

def run_condition_b_template(
    plain_context: str,
    labeled_context: str,
    gold_pairs: set,
    api_key: str,
    max_chars: int = 5000,
    model: str = "gpt-4o-mini",
    task_idx: int = 1,
    checker_setup: str = None,
    verbose: bool = False,
) -> dict:
    """
    Condition B: Non-incremental single-turn, matched budget (5K chars).

    Iteration 9 fix: use persistent=True + same CHUNK_PROMPT_INCREMENTAL as Condition A.
    Extract pairs from env.locals["_incremental"].pair_tracker.get_pairs(), NOT from
    completion.response. This avoids Failure Mode B entirely.

    This is equivalent to Condition A at k=1, which is the matched-budget baseline:
    same context size, same template, same extraction path. F1 is expected to match
    Condition A's k=1 result (~0.22), confirming the extraction works correctly.
    """
    from rlm.core.rlm import RLM
    from rlm.utils.prompts import INCREMENTAL_SYSTEM_PROMPT

    os.environ["OPENAI_API_KEY"] = api_key
    if checker_setup is None:
        checker_setup = TASK_CHECKER_SETUPS.get(task_idx, TASK_1_CHECKER_SETUP)

    # Use the first max_chars only (matched budget)
    context_chunk = plain_context[:max_chars]

    print(f"\n{'=' * 70}")
    print(f"CONDITION B: Non-incremental baseline (1 turn, {len(context_chunk)} chars, Task {task_idx})")
    print(f"  Using: persistent=True + CHUNK_PROMPT_INCREMENTAL + env.locals extraction")
    print(f"{'=' * 70}")

    # Same prompt as Condition A, chunk 1 of 1
    root_prompt = CHUNK_PROMPT_INCREMENTAL.format(
        task_idx=task_idx,
        chunk_num=1,
        total_chunks=1,
        chunk_idx=0,
    )

    rlm = RLM(
        backend="openai",
        backend_kwargs={"model_name": model},
        environment="local",
        environment_kwargs={"setup_code": checker_setup},
        persistent=True,  # MUST be True to access env.locals after completion
        custom_system_prompt=INCREMENTAL_SYSTEM_PROMPT,
        max_iterations=6,
        verbose=verbose,
    )

    t0 = time.perf_counter()
    completion = rlm.completion(context_chunk, root_prompt=root_prompt)
    elapsed = time.perf_counter() - t0

    # Extract from env.locals["_incremental"] directly (same path as Condition A)
    env = rlm._persistent_env
    pairs = []
    chunks_processed = 0
    pair_checks = 0
    if env and hasattr(env, "locals"):
        incr = env.locals.get("_incremental")
        if incr:
            pairs = list(incr.pair_tracker.get_pairs())
            stats = incr.get_stats()
            chunks_processed = stats.get("chunks_processed", 0)
            pair_checks = stats.get("total_pair_checks", 0)

    rlm.close()

    compliant = chunks_processed >= 1
    f1_result = compute_f1(pairs, gold_pairs)

    # Token accounting — Iteration 9 fix
    input_tokens, output_tokens = _extract_tokens(completion.usage_summary)

    print(f"    compliant: {compliant}  chunks_processed: {chunks_processed}")
    print(f"    pairs: {len(pairs)}  pair_checks: {pair_checks}")
    print(f"    F1={f1_result['f1']}  P={f1_result['precision']}  R={f1_result['recall']}")
    print(f"    input_tokens: {input_tokens}  output_tokens: {output_tokens}  elapsed: {elapsed:.1f}s")

    return {
        "condition": "B_matched_budget",
        "task_idx": task_idx,
        "model": model,
        "context_chars": len(context_chunk),
        "compliant": compliant,
        "chunks_processed": chunks_processed,
        "pair_checks": pair_checks,
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
    }


# ---------------------------------------------------------------------------
# Condition C: Non-incremental RLM oracle (25K chars, single turn)
# Iteration 9: persistent=True + ORACLE_PROMPT_SINGLE + env.locals extraction
# ---------------------------------------------------------------------------

def run_condition_c_oracle(
    plain_context: str,
    labeled_context: str,
    gold_pairs: set,
    api_key: str,
    max_chars: int = 25000,
    model: str = "gpt-4o-mini",
    task_idx: int = 1,
    checker_setup: str = None,
    verbose: bool = False,
) -> dict:
    """
    Condition C: Non-incremental single-turn oracle (25K chars, same total budget as A at k=5).

    Iteration 9 fix: use persistent=True + ORACLE_PROMPT_SINGLE + env.locals extraction.
    Extracts from env.locals["pair_results"] directly (no _incremental).
    This avoids Failure Mode B entirely by using an explicit code template.

    Expected result: F1 likely >= Condition A (single-turn sees all 25K at once,
    no FP accumulation across chunks). If Condition C > A, that is an honest limitation:
    incremental has per-turn FP inflation that single-turn oracle avoids.
    """
    from rlm.core.rlm import RLM
    from rlm.utils.prompts import INCREMENTAL_SYSTEM_PROMPT

    os.environ["OPENAI_API_KEY"] = api_key
    if checker_setup is None:
        checker_setup = TASK_CHECKER_SETUPS.get(task_idx, TASK_1_CHECKER_SETUP)

    context_full = plain_context[:max_chars]

    print(f"\n{'=' * 70}")
    print(f"CONDITION C: Oracle (1 turn, {len(context_full)} chars, Task {task_idx})")
    print(f"  Using: persistent=True + ORACLE_PROMPT_SINGLE + env.locals[pair_results]")
    print(f"{'=' * 70}")

    root_prompt = ORACLE_PROMPT_SINGLE.format(
        task_idx=task_idx,
        total_chars=len(context_full),
    )

    rlm = RLM(
        backend="openai",
        backend_kwargs={"model_name": model},
        environment="local",
        environment_kwargs={"setup_code": checker_setup},
        persistent=True,  # MUST be True to access env.locals after completion
        custom_system_prompt=INCREMENTAL_SYSTEM_PROMPT,
        max_iterations=6,
        verbose=verbose,
    )

    t0 = time.perf_counter()
    completion = rlm.completion(context_full, root_prompt=root_prompt)
    elapsed = time.perf_counter() - t0

    # Extract pair_results from env.locals directly (no _incremental for oracle)
    env = rlm._persistent_env
    pairs = []
    entities_found = 0
    if env and hasattr(env, "locals"):
        pair_results_raw = env.locals.get("pair_results")
        if pair_results_raw is not None:
            pairs = list(pair_results_raw)
        # Also try _incremental if ORACLE_PROMPT template set it up that way
        if not pairs:
            incr = env.locals.get("_incremental")
            if incr:
                pairs = list(incr.pair_tracker.get_pairs())
        # Count entities for diagnostic
        entities_raw = env.locals.get("entities")
        if isinstance(entities_raw, dict):
            entities_found = len(entities_raw)

    rlm.close()

    f1_result = compute_f1(pairs, gold_pairs)

    # Token accounting — Iteration 9 fix
    input_tokens, output_tokens = _extract_tokens(completion.usage_summary)

    print(f"    pairs: {len(pairs)}  entities_found: {entities_found}")
    print(f"    F1={f1_result['f1']}  P={f1_result['precision']}  R={f1_result['recall']}")
    print(f"    input_tokens: {input_tokens}  output_tokens: {output_tokens}  elapsed: {elapsed:.1f}s")

    # Check response for diagnostic
    response_preview = str(completion.response)[:300] if completion.response else ""

    return {
        "condition": "C_oracle",
        "task_idx": task_idx,
        "model": model,
        "context_chars": len(context_full),
        "entities_found": entities_found,
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
        "response_preview": response_preview,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="F1 Progression Experiment (Iteration 9 — fixed B/C + token tracking)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--num-chunks", type=int, default=5)
    parser.add_argument("--max-chunk-chars", type=int, default=5000)
    parser.add_argument("--task-idx", type=int, default=1,
                        help="OOLONG-Pairs task index (1, 3, or 6)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: auto-named by task)")
    parser.add_argument("--incremental-only", action="store_true",
                        help="Skip condition B/C (saves API cost)")
    parser.add_argument("--conditions-only", action="store_true",
                        help="Skip condition A, run B and C only")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # Validate task index
    if args.task_idx not in TASK_CHECKER_SETUPS:
        print(f"ERROR: --task-idx must be one of {list(TASK_CHECKER_SETUPS.keys())}")
        sys.exit(1)

    # Auto-name output by task
    if args.output is None:
        if args.task_idx == 1:
            args.output = "results/streaming/f1_progression_results_iter9.json"
        else:
            args.output = f"results/streaming/f1_progression_task{args.task_idx}_results.json"

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

    checker_setup = TASK_CHECKER_SETUPS[args.task_idx]
    verbose = not args.quiet

    print("Loading OOLONG-Pairs data...")
    plain_context, labeled_context = load_oolong_data()
    gold_pairs = compute_gold_pairs(labeled_context, task_idx=args.task_idx)
    print(f"Gold pairs (Task {args.task_idx}, full context): {len(gold_pairs)}")

    results = {
        "model": args.model,
        "task_idx": args.task_idx,
        "gold_pairs_count": len(gold_pairs),
        "iteration": 9,
        "fixes": [
            "token_tracking: use model_usage_summaries.total_input_tokens (not to_dict() with isinstance check)",
            "condition_b: persistent=True + CHUNK_PROMPT_INCREMENTAL + env.locals extraction",
            "condition_c: persistent=True + ORACLE_PROMPT_SINGLE + env.locals[pair_results]",
        ],
    }

    # Condition A: Incremental (k=5, 5K chars/chunk)
    if not args.conditions_only:
        result_a = run_condition_a_incremental(
            plain_context=plain_context,
            labeled_context=labeled_context,
            gold_pairs=gold_pairs,
            api_key=api_key,
            num_chunks=args.num_chunks,
            max_chunk_chars=args.max_chunk_chars,
            model=args.model,
            task_idx=args.task_idx,
            checker_setup=checker_setup,
            verbose=verbose,
        )
        results["condition_a"] = result_a

    if not args.incremental_only:
        # Condition B: Matched-budget baseline (1 turn, 5K chars, same template as A)
        result_b = run_condition_b_template(
            plain_context=plain_context,
            labeled_context=labeled_context,
            gold_pairs=gold_pairs,
            api_key=api_key,
            max_chars=args.max_chunk_chars,  # same as one chunk
            model=args.model,
            task_idx=args.task_idx,
            checker_setup=checker_setup,
            verbose=verbose,
        )
        results["condition_b"] = result_b

        # Condition C: Oracle (1 turn, all 25K chars)
        total_chars = args.num_chunks * args.max_chunk_chars  # 25K
        result_c = run_condition_c_oracle(
            plain_context=plain_context,
            labeled_context=labeled_context,
            gold_pairs=gold_pairs,
            api_key=api_key,
            max_chars=total_chars,
            model=args.model,
            task_idx=args.task_idx,
            checker_setup=checker_setup,
            verbose=verbose,
        )
        results["condition_c"] = result_c

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"FINAL COMPARISON TABLE (Task {args.task_idx})")
    print(f"{'=' * 70}")
    print(f"{'Condition':<35} {'F1':>6} {'Precision':>10} {'Recall':>8} {'Chars':>8} {'InToks':>8}")
    print("-" * 75)

    if "condition_a" in results:
        ra = results["condition_a"]
        print(f"{'A: Incremental (k=5, 5K/chunk)':<35} "
              f"{ra.get('final_f1', 'N/A'):>6} "
              f"{ra.get('final_precision', 'N/A'):>10} "
              f"{ra.get('final_recall', 'N/A'):>8} "
              f"{args.num_chunks * args.max_chunk_chars:>8} "
              f"{ra.get('total_input_tokens', 'N/A'):>8}")
        print(f"\n  F1 Progression (Condition A):")
        for t in ra["f1_progression"]:
            chars_so_far = t["chunk"] * args.max_chunk_chars
            print(f"    k={t['chunk']} ({chars_so_far} chars): F1={t['f1']}  P={t['precision']}  "
                  f"R={t['recall']}  pairs={t['pairs']}  tokens={t['input_tokens']}  prune={t.get('prune_count', 0)}")

    if "condition_b" in results:
        rb = results["condition_b"]
        print(f"\n{'B: Baseline (1 turn, 5K, same tmpl)':<35} "
              f"{rb.get('f1', 'N/A'):>6} "
              f"{rb.get('precision', 'N/A'):>10} "
              f"{rb.get('recall', 'N/A'):>8} "
              f"{args.max_chunk_chars:>8} "
              f"{rb.get('input_tokens', 'N/A'):>8}")
        print(f"  compliant={rb.get('compliant')}, predicted={rb.get('predicted_pairs')}")

    if "condition_c" in results:
        rc = results["condition_c"]
        total_chars = args.num_chunks * args.max_chunk_chars
        print(f"\n{'C: Oracle (1 turn, 25K)':<35} "
              f"{rc.get('f1', 'N/A'):>6} "
              f"{rc.get('precision', 'N/A'):>10} "
              f"{rc.get('recall', 'N/A'):>8} "
              f"{total_chars:>8} "
              f"{rc.get('input_tokens', 'N/A'):>8}")
        print(f"  entities_found={rc.get('entities_found')}, predicted={rc.get('predicted_pairs')}")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    main()
