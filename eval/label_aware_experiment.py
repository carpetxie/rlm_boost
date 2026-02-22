"""
Label-Aware Check_Pair Experiment — Iteration 10.

This is the MANDATORY experiment identified in the Iteration 10 critique.
It re-runs the F1 progression experiment (Conditions A/B/C) using the ACTUAL
Task 1 condition (numeric_value OR location) instead of the proxy condition
(>= 1 instance). This uses the labeled context where "|| Label: [cat]" is
visible in each line.

## Why This Matters

The prior experiments (Iterations 8-9) used check_pair `>= 1 instance` as a
protocol compliance proxy — it tests whether the model follows the incremental
protocol but NOT whether it solves the actual OOLONG-Pairs task. The FP root
cause analysis (Experiment 25) confirmed: 100% of FPs are check_pair condition
mismatch, and the F1 ceiling at 25K chars is 0.716.

With label-aware check_pair using the ACTUAL condition, F1 is expected to
approach 0.716 (the coverage ceiling). The A vs C gap under fair conditions
is the paper's central empirical result.

## Context Format (Labeled)

The labeled context has lines like:
    Date: Jan 6, 2023 || User: 44436 || Instance: What is 5+5? || Label: numeric value

The model reads the `|| Label: [cat]` field directly and sets qualifying=True
when the label is in the task's qualifying set.

## Task-Specific Qualifying Labels

- Task 1: "numeric value" or "location"
- Task 3: "description and abstract concept" or "abbreviation"
- Task 6: "location" or "abbreviation"

## Expected Results

- Conditions A, B, C all should show substantially higher F1 than proxy version
- A at k=5 should approach 0.716 (coverage ceiling)
- A vs C gap narrows: the gap under proxy was partly coverage, partly proxy mismatch
- With label-aware check_pair, A vs C purely measures incremental vs oracle coverage

## Token Anomaly Investigation

This run also adds iteration-count logging (number of LM calls per completion())
to investigate the Condition B token anomaly (21,934 tokens vs A Turn 1's 6,401
for the same 5K context). Hypothesis: Condition B uses all 6 max_iterations while
Condition A Turn 1 terminates early (~2 iterations).

Usage:
    export OPENAI_API_KEY=sk-...
    python eval/label_aware_experiment.py
    python eval/label_aware_experiment.py --model gpt-4o-mini --task-idx 1
    python eval/label_aware_experiment.py --task-idx 3 --incremental-only
    python eval/label_aware_experiment.py --task-idx 6 --incremental-only
    python eval/label_aware_experiment.py --all-tasks  # runs Tasks 1, 3, 6
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.rlm_pipeline_experiment import compute_f1, compute_gold_pairs, split_context_by_users
from eval.f1_progression_experiment import _extract_tokens


# ---------------------------------------------------------------------------
# Task-specific qualifying label sets (ACTUAL task conditions, not proxy)
# ---------------------------------------------------------------------------

TASK_QUALIFYING_LABELS: dict[int, set[str]] = {
    1: {"numeric value", "location"},
    3: {"description and abstract concept", "abbreviation"},
    6: {"location", "abbreviation"},
}

TASK_LABEL_DESCRIPTION: dict[int, str] = {
    1: '"numeric value" or "location"',
    3: '"description and abstract concept" or "abbreviation"',
    6: '"location" or "abbreviation"',
}


# ---------------------------------------------------------------------------
# Label-aware checker setups (injected into REPL before first turn)
# ---------------------------------------------------------------------------

def make_label_checker_setup(task_idx: int) -> str:
    """Generate label-aware check_pair setup code for a given task.

    Uses attrs["qualifying"] which is set True when the entity has at least
    one instance with a qualifying label. This implements the ACTUAL task
    condition rather than the proxy '>= 1 instance' condition.
    """
    labels_repr = repr(TASK_QUALIFYING_LABELS[task_idx])
    return f"""
# Label-aware check_pair for Task {task_idx}: ACTUAL condition ({TASK_LABEL_DESCRIPTION[task_idx]}).
# Requires labeled context where "|| Label: [cat]" is visible in each line.
# Iteration 10: replaces proxy check_pair (>= 1 instance) with real task condition.
QUALIFYING_LABELS_{task_idx} = {labels_repr}

def check_pair(attrs1, attrs2):
    q1 = attrs1.get("qualifying", False) if isinstance(attrs1, dict) else False
    q2 = attrs2.get("qualifying", False) if isinstance(attrs2, dict) else False
    return q1 and q2
"""


# ---------------------------------------------------------------------------
# Label-aware chunk prompt (for Conditions A and B)
# ---------------------------------------------------------------------------

CHUNK_PROMPT_LABEL_AWARE = """Task (OOLONG-Pairs Task {task_idx}): Find pairs where BOTH users have at least one
instance labeled {label_desc}. This is Chunk {chunk_num} of {total_chunks}.

Context format: "Date: [date] || User: [user_id] || Instance: [text] || Label: [cat]"
Entity ID = the numeric user_id (e.g. "44436"). Parse the Label field directly.

Run this code (chunk index {chunk_idx}):

```repl
import re
entities = {{}}
qualifying_labels = {qualifying_labels_repr}
for line in context_{chunk_idx}.split('\\n'):
    m = re.search(r'User: (\\d+).*?\\|\\| Label: (.+?)$', line)
    if m:
        uid = m.group(1)
        label = m.group(2).strip().lower()
        if uid not in entities:
            entities[uid] = {{"labels": [], "qualifying": False}}
        entities[uid]["labels"].append(label)
        if label in qualifying_labels:
            entities[uid]["qualifying"] = True
stats = _incremental.process_chunk({chunk_idx}, entities, pair_checker=check_pair)
pair_results = list(_incremental.pair_tracker.get_pairs())
print(f"Chunk {chunk_num}: {{stats['new_entities']}} new entities, {{stats['total_pairs']}} pairs, {{stats['pair_checks']}} checks")
print(f"  Labels seen: {{set(l for e in entities.values() for l in e['labels'])}}")
print(f"  Qualifying entities this chunk: {{sum(1 for e in entities.values() if e['qualifying'])}}")
```

After the repl block runs successfully, return FINAL_VAR(pair_results).
Do NOT re-read context_0, context_1 etc. on this turn.
"""

# Condition C oracle prompt for labeled context
ORACLE_PROMPT_LABEL_AWARE = """Task (OOLONG-Pairs Task {task_idx}): Find ALL pairs where BOTH users have at least
one instance labeled {label_desc}. Single-turn analysis of FULL context ({total_chars} chars).

Context format: "Date: [date] || User: [user_id] || Instance: [text] || Label: [cat]"
The label is visible — read it directly to qualify entities.

Run this code:

```repl
import re
entities = {{}}
qualifying_labels = {qualifying_labels_repr}
for line in context_0.split('\\n'):
    m = re.search(r'User: (\\d+).*?\\|\\| Label: (.+?)$', line)
    if m:
        uid = m.group(1)
        label = m.group(2).strip().lower()
        if uid not in entities:
            entities[uid] = {{"labels": [], "qualifying": False}}
        entities[uid]["labels"].append(label)
        if label in qualifying_labels:
            entities[uid]["qualifying"] = True

# All users that qualify
qualifying_users = sorted([uid for uid, attrs in entities.items() if attrs["qualifying"]])
pair_results = []
for i, uid1 in enumerate(qualifying_users):
    for uid2 in qualifying_users[i+1:]:
        pair_results.append((min(uid1, uid2), max(uid1, uid2)))
print(f"Total entities: {{len(entities)}}, qualifying: {{len(qualifying_users)}}")
print(f"Total pairs: {{len(pair_results)}}")
```

After the repl block runs, return FINAL_VAR(pair_results).
"""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_labeled_data():
    """Load OOLONG-Pairs dataset and return (plain_context, labeled_context)."""
    from datasets import load_dataset
    ds = load_dataset("oolongbench/oolong-synth", split="validation")
    corpus = [
        x for x in ds
        if x["dataset"] == "trec_coarse" and x["context_len"] == 32768
    ][0]
    return corpus["context_window_text"], corpus["context_window_text_with_labels"]


# ---------------------------------------------------------------------------
# Condition A: Incremental RLM with label-aware check_pair
# ---------------------------------------------------------------------------

def run_label_aware_condition_a(
    labeled_context: str,
    gold_pairs: set,
    api_key: str,
    task_idx: int = 1,
    num_chunks: int = 5,
    max_chunk_chars: int = 5000,
    model: str = "gpt-4o-mini",
    verbose: bool = False,
) -> dict:
    """
    Condition A (Label-Aware): Incremental RLM on labeled context.

    Key difference from prior Condition A: uses labeled_context (with Label: field)
    and label-aware check_pair (actual task condition). Expected F1 substantially
    higher than proxy version, approaching 0.716 coverage ceiling.

    Also tracks iteration_count per completion() to investigate token anomaly.
    """
    from rlm.core.rlm import RLM
    from rlm.utils.prompts import INCREMENTAL_SYSTEM_PROMPT

    os.environ["OPENAI_API_KEY"] = api_key

    checker_setup = make_label_checker_setup(task_idx)
    qualifying_labels = TASK_QUALIFYING_LABELS[task_idx]
    label_desc = TASK_LABEL_DESCRIPTION[task_idx]

    print(f"\n{'=' * 70}")
    print(f"CONDITION A (Label-Aware): Incremental RLM (k={num_chunks}, {max_chunk_chars} chars/chunk)")
    print(f"Task {task_idx} | Qualifying: {label_desc}")
    print(f"{'=' * 70}")

    # Split labeled_context (not plain_context!) so model sees labels
    chunks = split_context_by_users(labeled_context, num_chunks)
    chunks = [c[:max_chunk_chars] for c in chunks]
    print(f"Labeled chunk sizes: {[len(c) for c in chunks]} chars")

    # Verify labels are visible in chunks
    import re
    label_sample = re.findall(r'\|\| Label: (.+?)$', chunks[0][:1000], re.MULTILINE)
    print(f"  Label sample (chunk 0, first 1K): {label_sample[:5]}")

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

    f1_progression = []
    turn_tokens = []
    prev_chunks_processed = 0

    for chunk_i, chunk in enumerate(chunks):
        chunk_num = chunk_i + 1
        root_prompt = CHUNK_PROMPT_LABEL_AWARE.format(
            task_idx=task_idx,
            label_desc=label_desc,
            chunk_num=chunk_num,
            total_chunks=num_chunks,
            chunk_idx=chunk_i,
            qualifying_labels_repr=repr(qualifying_labels),
        )

        print(f"\n  --- Turn {chunk_num}/{num_chunks} ---")
        t0 = time.perf_counter()
        completion = rlm.completion(chunk, root_prompt=root_prompt)
        elapsed = time.perf_counter() - t0

        # Extract from REPL state
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

        # Prune count telemetry — Iteration 10: direct attribute access
        prune_count_direct = 0
        if hasattr(rlm, "history_manager") and rlm.history_manager is not None:
            prune_count_direct = rlm.history_manager._prune_count
            hm_stats = rlm.history_manager.get_stats()
            print(f"    prune_count (direct): {prune_count_direct}  get_stats(): {hm_stats['prune_count']}")

        compliant = chunks_processed > prev_chunks_processed
        prev_chunks_processed = chunks_processed

        f1_result = compute_f1(direct_pairs, gold_pairs)
        input_tokens, output_tokens = _extract_tokens(completion.usage_summary)
        turn_tokens.append({"turn": chunk_num, "input": input_tokens, "output": output_tokens})

        print(f"    compliant: {compliant}  chunks_processed: {chunks_processed}")
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
            "prune_count": prune_count_direct,
        })

    rlm.close()

    compliant_turns = sum(1 for t in f1_progression if t["compliant"])
    compliance_rate = compliant_turns / num_chunks if num_chunks > 0 else 0.0
    total_input = sum(t["input"] for t in turn_tokens)
    total_output = sum(t["output"] for t in turn_tokens)

    print(f"\n  Condition A (Label-Aware) Summary (Task {task_idx}):")
    print(f"    Compliance: {compliance_rate:.0%} ({compliant_turns}/{num_chunks} turns)")
    print(f"    F1 progression: {[t['f1'] for t in f1_progression]}")
    print(f"    Final F1={f1_progression[-1]['f1'] if f1_progression else None}")
    print(f"    Total input tokens: {total_input}  Total output tokens: {total_output}")

    return {
        "condition": "A_incremental_label_aware",
        "task_idx": task_idx,
        "model": model,
        "num_chunks": num_chunks,
        "max_chunk_chars": max_chunk_chars,
        "context_type": "labeled",
        "check_pair_type": "label_aware",
        "qualifying_labels": list(qualifying_labels),
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
# Condition B: Non-incremental baseline (label-aware, matched budget)
# ---------------------------------------------------------------------------

def run_label_aware_condition_b(
    labeled_context: str,
    gold_pairs: set,
    api_key: str,
    task_idx: int = 1,
    max_chars: int = 5000,
    model: str = "gpt-4o-mini",
    verbose: bool = False,
) -> dict:
    """
    Condition B (Label-Aware): Non-incremental single-turn, matched budget (5K chars).

    Uses labeled context + label-aware check_pair. Same template as Condition A
    chunk 1. Expected F1 similar to A at k=1.

    Also logs iteration_count within completion() to investigate token anomaly.
    """
    from rlm.core.rlm import RLM
    from rlm.utils.prompts import INCREMENTAL_SYSTEM_PROMPT

    os.environ["OPENAI_API_KEY"] = api_key

    checker_setup = make_label_checker_setup(task_idx)
    qualifying_labels = TASK_QUALIFYING_LABELS[task_idx]
    label_desc = TASK_LABEL_DESCRIPTION[task_idx]

    context_chunk = labeled_context[:max_chars]

    print(f"\n{'=' * 70}")
    print(f"CONDITION B (Label-Aware): Non-incremental baseline (1 turn, {len(context_chunk)} chars)")
    print(f"Task {task_idx} | Qualifying: {label_desc}")
    print(f"{'=' * 70}")

    root_prompt = CHUNK_PROMPT_LABEL_AWARE.format(
        task_idx=task_idx,
        label_desc=label_desc,
        chunk_num=1,
        total_chunks=1,
        chunk_idx=0,
        qualifying_labels_repr=repr(qualifying_labels),
    )

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

    t0 = time.perf_counter()
    completion = rlm.completion(context_chunk, root_prompt=root_prompt)
    elapsed = time.perf_counter() - t0

    # Extract from env.locals
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

    # Iteration-count investigation: count how many times message_history grew
    # Proxy: check length of history_manager turn_summaries (each turn adds one)
    iteration_count_proxy = 0
    if hasattr(rlm, "history_manager") and rlm.history_manager is not None:
        prune_count = rlm.history_manager._prune_count
        print(f"    prune_count: {prune_count}")

    rlm.close()

    compliant = chunks_processed >= 1
    f1_result = compute_f1(pairs, gold_pairs)
    input_tokens, output_tokens = _extract_tokens(completion.usage_summary)

    print(f"    compliant: {compliant}  chunks_processed: {chunks_processed}")
    print(f"    pairs: {len(pairs)}  pair_checks: {pair_checks}")
    print(f"    F1={f1_result['f1']}  P={f1_result['precision']}  R={f1_result['recall']}")
    print(f"    input_tokens: {input_tokens}  output_tokens: {output_tokens}  elapsed: {elapsed:.1f}s")

    return {
        "condition": "B_matched_budget_label_aware",
        "task_idx": task_idx,
        "model": model,
        "context_chars": len(context_chunk),
        "context_type": "labeled",
        "check_pair_type": "label_aware",
        "qualifying_labels": list(qualifying_labels),
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
# Condition C: Oracle with label-aware check_pair (single-turn, 25K chars)
# ---------------------------------------------------------------------------

def run_label_aware_condition_c(
    labeled_context: str,
    gold_pairs: set,
    api_key: str,
    task_idx: int = 1,
    max_chars: int = 25000,
    model: str = "gpt-4o-mini",
    verbose: bool = False,
) -> dict:
    """
    Condition C (Label-Aware): Oracle single-turn on full labeled context.

    Uses ORACLE_PROMPT_LABEL_AWARE which extracts labels directly and builds
    pair_results without _incremental. Expected F1 near 0.716 (coverage ceiling
    at 25K chars) since the actual condition is implemented.
    """
    from rlm.core.rlm import RLM
    from rlm.utils.prompts import INCREMENTAL_SYSTEM_PROMPT

    os.environ["OPENAI_API_KEY"] = api_key

    checker_setup = make_label_checker_setup(task_idx)
    qualifying_labels = TASK_QUALIFYING_LABELS[task_idx]
    label_desc = TASK_LABEL_DESCRIPTION[task_idx]

    context_full = labeled_context[:max_chars]

    print(f"\n{'=' * 70}")
    print(f"CONDITION C (Label-Aware): Oracle (1 turn, {len(context_full)} chars)")
    print(f"Task {task_idx} | Qualifying: {label_desc}")
    print(f"{'=' * 70}")

    root_prompt = ORACLE_PROMPT_LABEL_AWARE.format(
        task_idx=task_idx,
        label_desc=label_desc,
        total_chars=len(context_full),
        qualifying_labels_repr=repr(qualifying_labels),
    )

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

    t0 = time.perf_counter()
    completion = rlm.completion(context_full, root_prompt=root_prompt)
    elapsed = time.perf_counter() - t0

    # Extract pair_results from env.locals
    env = rlm._persistent_env
    pairs = []
    entities_found = 0
    qualifying_count = 0
    if env and hasattr(env, "locals"):
        pair_results_raw = env.locals.get("pair_results")
        if pair_results_raw is not None:
            pairs = list(pair_results_raw)
        entities_raw = env.locals.get("entities")
        if isinstance(entities_raw, dict):
            entities_found = len(entities_raw)
            qualifying_count = sum(
                1 for e in entities_raw.values()
                if isinstance(e, dict) and e.get("qualifying", False)
            )

    rlm.close()

    f1_result = compute_f1(pairs, gold_pairs)
    input_tokens, output_tokens = _extract_tokens(completion.usage_summary)

    print(f"    entities_found: {entities_found}  qualifying: {qualifying_count}")
    print(f"    pairs: {len(pairs)}")
    print(f"    F1={f1_result['f1']}  P={f1_result['precision']}  R={f1_result['recall']}")
    print(f"    input_tokens: {input_tokens}  output_tokens: {output_tokens}  elapsed: {elapsed:.1f}s")

    return {
        "condition": "C_oracle_label_aware",
        "task_idx": task_idx,
        "model": model,
        "context_chars": len(context_full),
        "context_type": "labeled",
        "check_pair_type": "label_aware",
        "qualifying_labels": list(qualifying_labels),
        "entities_found": entities_found,
        "qualifying_entities": qualifying_count,
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
# Main
# ---------------------------------------------------------------------------

def run_task(
    task_idx: int,
    plain_context: str,
    labeled_context: str,
    api_key: str,
    model: str,
    num_chunks: int,
    max_chunk_chars: int,
    incremental_only: bool,
    conditions_only: bool,
    verbose: bool,
) -> dict:
    """Run Conditions A/B/C (label-aware) for a single task."""
    gold_pairs = compute_gold_pairs(labeled_context, task_idx=task_idx)
    print(f"\nGold pairs (Task {task_idx}, full labeled context): {len(gold_pairs)}")

    results = {
        "task_idx": task_idx,
        "model": model,
        "gold_pairs_count": len(gold_pairs),
        "check_pair_type": "label_aware",
        "qualifying_labels": list(TASK_QUALIFYING_LABELS[task_idx]),
        "iteration": 10,
    }

    if not conditions_only:
        result_a = run_label_aware_condition_a(
            labeled_context=labeled_context,
            gold_pairs=gold_pairs,
            api_key=api_key,
            task_idx=task_idx,
            num_chunks=num_chunks,
            max_chunk_chars=max_chunk_chars,
            model=model,
            verbose=verbose,
        )
        results["condition_a"] = result_a

    if not incremental_only:
        result_b = run_label_aware_condition_b(
            labeled_context=labeled_context,
            gold_pairs=gold_pairs,
            api_key=api_key,
            task_idx=task_idx,
            max_chars=max_chunk_chars,
            model=model,
            verbose=verbose,
        )
        results["condition_b"] = result_b

        total_chars = num_chunks * max_chunk_chars
        result_c = run_label_aware_condition_c(
            labeled_context=labeled_context,
            gold_pairs=gold_pairs,
            api_key=api_key,
            task_idx=task_idx,
            max_chars=total_chars,
            model=model,
            verbose=verbose,
        )
        results["condition_c"] = result_c

    # Summary table
    print(f"\n{'=' * 75}")
    print(f"LABEL-AWARE COMPARISON TABLE — Task {task_idx} ({TASK_LABEL_DESCRIPTION[task_idx]})")
    print(f"{'=' * 75}")
    print(f"{'Condition':<40} {'F1':>6} {'Prec':>8} {'Recall':>8} {'InToks':>8}")
    print("-" * 75)

    if "condition_a" in results:
        ra = results["condition_a"]
        print(f"{'A: Incremental (k=5, 5K/chunk, labeled)':<40} "
              f"{ra.get('final_f1', 'N/A'):>6} "
              f"{ra.get('final_precision', 'N/A'):>8} "
              f"{ra.get('final_recall', 'N/A'):>8} "
              f"{ra.get('total_input_tokens', 'N/A'):>8}")
        for t in ra["f1_progression"]:
            print(f"  k={t['chunk']}: F1={t['f1']}  P={t['precision']}  R={t['recall']}"
                  f"  pairs={t['pairs']}  tokens={t['input_tokens']}  prune={t.get('prune_count', 0)}")

    if "condition_b" in results:
        rb = results["condition_b"]
        print(f"{'B: Baseline (1T, 5K, labeled)':<40} "
              f"{rb.get('f1', 'N/A'):>6} "
              f"{rb.get('precision', 'N/A'):>8} "
              f"{rb.get('recall', 'N/A'):>8} "
              f"{rb.get('input_tokens', 'N/A'):>8}")

    if "condition_c" in results:
        rc = results["condition_c"]
        print(f"{'C: Oracle (1T, 25K, labeled)':<40} "
              f"{rc.get('f1', 'N/A'):>6} "
              f"{rc.get('precision', 'N/A'):>8} "
              f"{rc.get('recall', 'N/A'):>8} "
              f"{rc.get('input_tokens', 'N/A'):>8}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Label-Aware Check_Pair Experiment (Iteration 10)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--num-chunks", type=int, default=5)
    parser.add_argument("--max-chunk-chars", type=int, default=5000)
    parser.add_argument("--task-idx", type=int, default=1, choices=[1, 3, 6],
                        help="Task index (1=numeric_value|location, 3=description|abbrev, 6=location|abbrev)")
    parser.add_argument("--all-tasks", action="store_true",
                        help="Run Tasks 1, 3, and 6 sequentially")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--incremental-only", action="store_true")
    parser.add_argument("--conditions-only", action="store_true")
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

    print("Loading OOLONG-Pairs labeled data...")
    plain_context, labeled_context = load_labeled_data()
    print(f"Labeled context length: {len(labeled_context)} chars")

    # Verify labels are present
    import re
    label_count = len(re.findall(r'\|\| Label: .+?$', labeled_context[:5000], re.MULTILINE))
    print(f"Label occurrences in first 5K chars: {label_count}")

    tasks_to_run = [1, 3, 6] if args.all_tasks else [args.task_idx]
    all_results = {}

    for task_idx in tasks_to_run:
        print(f"\n{'#' * 75}")
        print(f"# TASK {task_idx}: {TASK_LABEL_DESCRIPTION[task_idx]}")
        print(f"{'#' * 75}")

        task_results = run_task(
            task_idx=task_idx,
            plain_context=plain_context,
            labeled_context=labeled_context,
            api_key=api_key,
            model=args.model,
            num_chunks=args.num_chunks,
            max_chunk_chars=args.max_chunk_chars,
            incremental_only=args.incremental_only,
            conditions_only=args.conditions_only,
            verbose=not args.quiet,
        )
        all_results[f"task_{task_idx}"] = task_results

        # Save per-task
        if args.output is None:
            out_path = Path(f"results/streaming/label_aware_task{task_idx}_results.json")
        else:
            out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(task_results, f, indent=2, default=str)
        print(f"Task {task_idx} results saved to {out_path}")

    # Save combined results if multiple tasks
    if args.all_tasks:
        combined_path = Path("results/streaming/label_aware_all_tasks.json")
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nCombined results saved to {combined_path}")

    return all_results


if __name__ == "__main__":
    main()
