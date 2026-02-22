"""
Label-Aware Check_Pair Experiment V2 — Iteration 11.

This is the MANDATORY redesign identified in the Iteration 11 critique.
It fixes two structural flaws in the Iteration 10 label-aware experiment:

## Flaw 1 Fixed: Data Slice Non-Equivalence

Iteration 10's Condition A used `split_context_by_users(labeled_context, 5)` which
splits across ALL 96K labeled chars into 5 disjoint groups (0–5K, 19K–24K, 38–43K,
57–62K, 76–81K). Condition C used `labeled_context[:25000]` (chars 0–25K only).
These are DIFFERENT slices, so F1(A)=0.1695 vs F1(C)=0.3424 was comparing two
different tasks, not two strategies on the same data.

V2 Fix: Condition A uses sequential 5K windows from the SAME first 25K chars as C:
    context_window = labeled_context[:25000]
    chunks = [context_window[i*5000:(i+1)*5000] for i in range(5)]

Now A and C see identical content — same entities, same qualifying set, same ceiling.

## Flaw 2 Fixed: Phantom Chunk and Compliance Metric

Iteration 10 compliance check was `chunks_processed > prev_chunks_processed`, which
accepted a jump of 2 (Turn 1: chunks_processed=2 from one completion). The correct
check is strict equality: `chunks_processed == prev_chunks_processed + 1`.

The prompt also lacked "EXACTLY ONCE" restriction, allowing the model to call
`process_chunk()` multiple times in one completion (causing the phantom chunk).

V2 Fix:
  - Compliance metric: strict `==` (not `>`)
  - Phantom detection: warn if delta > 1
  - Prompt addition: "Call process_chunk EXACTLY ONCE with chunk_idx={chunk_idx}"
  - Per-completion iteration count from usage_summary.total_calls

## Additional Experiments

- Condition C Full: Oracle on all 96K labeled chars (definitive coverage ceiling)
- Tasks 3 and 6: Sequential-chunk redesign applied to all three benchmark tasks

## Usage

    export OPENAI_API_KEY=sk-...
    python eval/label_aware_v2_experiment.py                    # Task 1
    python eval/label_aware_v2_experiment.py --task-idx 3       # Task 3
    python eval/label_aware_v2_experiment.py --task-idx 6       # Task 6
    python eval/label_aware_v2_experiment.py --all-tasks        # Tasks 1, 3, 6
    python eval/label_aware_v2_experiment.py --condition-c-full # Full-context oracle
    python eval/label_aware_v2_experiment.py --incremental-only # Only Condition A
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.rlm_pipeline_experiment import compute_f1, compute_gold_pairs
from eval.f1_progression_experiment import _extract_tokens
from eval.label_aware_experiment import (
    TASK_QUALIFYING_LABELS,
    TASK_LABEL_DESCRIPTION,
    make_label_checker_setup,
    load_labeled_data,
    ORACLE_PROMPT_LABEL_AWARE,
)


# ---------------------------------------------------------------------------
# V2: Updated chunk prompt — EXACTLY ONCE restriction (Flaw 2 fix)
# ---------------------------------------------------------------------------

CHUNK_PROMPT_LABEL_AWARE_V2 = """Task (OOLONG-Pairs Task {task_idx}): Find pairs where BOTH users have at least one
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

IMPORTANT: Call `_incremental.process_chunk({chunk_idx}, entities, pair_checker=check_pair)` EXACTLY ONCE
with chunk_idx={chunk_idx}. Do NOT call process_chunk with any other chunk index in this turn.

After the repl block runs successfully, return FINAL_VAR(pair_results).
Do NOT re-read context_0, context_1 etc. on this turn.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_iteration_count(usage_summary) -> int:
    """Extract total LM call count from usage_summary (total_calls field).

    This gives the number of LM iterations within a single completion() call.
    Turn 4 token spike investigation: if Turn 4 ran 6 iterations vs 2 for others,
    total_calls will reflect this directly.
    """
    if usage_summary is None:
        return 0
    total_calls = 0
    for model_usage in usage_summary.model_usage_summaries.values():
        total_calls += model_usage.total_calls
    return total_calls


def _make_sequential_chunks(labeled_context: str, num_chunks: int, max_chunk_chars: int) -> list[str]:
    """V2 fix: sequential 5K windows from SAME first 25K chars as Condition C.

    Iteration 10 used split_context_by_users() across the full 96K corpus,
    producing disjoint windows at chars 0-5K, 19K-24K, 38K-43K, 57K-62K, 76K-81K.
    Condition C used labeled_context[:25000]. These were different slices.

    This function creates sequential windows all within labeled_context[:25000]:
        Chunk 0: chars 0-4999
        Chunk 1: chars 5000-9999
        Chunk 2: chars 10000-14999
        Chunk 3: chars 15000-19999
        Chunk 4: chars 20000-24999

    Now A and C see identical content — same corpus slice, same entities, same ceiling.
    """
    total_chars = num_chunks * max_chunk_chars  # e.g., 5 * 5000 = 25000
    context_window = labeled_context[:total_chars]
    return [context_window[i * max_chunk_chars:(i + 1) * max_chunk_chars] for i in range(num_chunks)]


# ---------------------------------------------------------------------------
# Condition A V2: Incremental RLM with sequential chunking (fixed)
# ---------------------------------------------------------------------------

def run_condition_a_v2(
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
    Condition A V2 (Label-Aware, Sequential Chunking): Incremental RLM.

    Key fixes from Iteration 10 Condition A:
    1. Sequential chunking: all chunks from first 25K chars (same as Condition C)
    2. Strict compliance: chunks_processed == prev_chunks_processed + 1
    3. Phantom chunk detection: warn if model advances by > 1
    4. Iteration count logging per completion() (via usage_summary.total_calls)
    5. Prompt: "EXACTLY ONCE" restriction on process_chunk calls
    """
    from rlm.core.rlm import RLM
    from rlm.utils.prompts import INCREMENTAL_SYSTEM_PROMPT

    os.environ["OPENAI_API_KEY"] = api_key

    checker_setup = make_label_checker_setup(task_idx)
    qualifying_labels = TASK_QUALIFYING_LABELS[task_idx]
    label_desc = TASK_LABEL_DESCRIPTION[task_idx]

    print(f"\n{'=' * 70}")
    print(f"CONDITION A V2 (Sequential): Incremental RLM (k={num_chunks}, {max_chunk_chars} chars/chunk)")
    print(f"Task {task_idx} | Qualifying: {label_desc}")
    print(f"FIXED: Sequential chunks from first {num_chunks * max_chunk_chars} chars (same as Condition C)")
    print(f"{'=' * 70}")

    # V2 FIX: Sequential chunks from same first 25K chars as Condition C
    chunks = _make_sequential_chunks(labeled_context, num_chunks, max_chunk_chars)
    print(f"Sequential chunk sizes: {[len(c) for c in chunks]} chars")
    print(f"Chunk 0 starts at char 0, Chunk 4 ends at char {num_chunks * max_chunk_chars}")

    # Verify labels visible in each chunk
    import re
    for i, chunk in enumerate(chunks):
        label_count = len(re.findall(r'\|\| Label: .+?$', chunk, re.MULTILINE))
        print(f"  Chunk {i}: {label_count} labeled records")

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
    phantom_chunks_detected = 0

    for chunk_i, chunk in enumerate(chunks):
        chunk_num = chunk_i + 1
        root_prompt = CHUNK_PROMPT_LABEL_AWARE_V2.format(
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

        # Prune count telemetry
        prune_count_direct = 0
        if hasattr(rlm, "history_manager") and rlm.history_manager is not None:
            prune_count_direct = rlm.history_manager._prune_count
            hm_stats = rlm.history_manager.get_stats()
            print(f"    prune_count (direct): {prune_count_direct}  get_stats(): {hm_stats['prune_count']}")

        # V2 FIX: Strict compliance metric (== not >)
        delta = chunks_processed - prev_chunks_processed
        compliant = (delta == 1)

        # V2 FIX: Phantom chunk detection
        phantom = (delta > 1)
        if phantom:
            phantom_chunks_detected += 1
            print(f"    ⚠ PHANTOM CHUNK DETECTED: chunks_processed advanced by {delta} (expected 1)")
            print(f"      Model processed {delta} chunks in one turn — chunk idx poisoning possible")
        elif delta == 0:
            print(f"    ⚠ NON-COMPLIANT: chunks_processed did not advance (delta=0, dedup blocked?)")

        prev_chunks_processed = chunks_processed

        # V2 FIX: Per-completion iteration count via usage_summary.total_calls
        iteration_count = _extract_iteration_count(completion.usage_summary)
        print(f"    iteration_count (total LM calls this completion): {iteration_count}")

        f1_result = compute_f1(direct_pairs, gold_pairs)
        input_tokens, output_tokens = _extract_tokens(completion.usage_summary)
        turn_tokens.append({"turn": chunk_num, "input": input_tokens, "output": output_tokens})

        print(f"    compliant: {compliant}  chunks_processed: {chunks_processed}  delta: {delta}")
        print(f"    pairs: {len(direct_pairs)}  pair_checks_total: {pair_checks_total}")
        print(f"    F1={f1_result['f1']}  P={f1_result['precision']}  R={f1_result['recall']}")
        print(f"    input_tokens: {input_tokens}  output_tokens: {output_tokens}  elapsed: {elapsed:.1f}s")

        f1_progression.append({
            "chunk": chunk_num,
            "chunk_idx": chunk_i,
            "compliant": compliant,
            "delta": delta,
            "phantom_chunk": phantom,
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
            "iteration_count": iteration_count,  # V2: LM call count per completion
        })

    rlm.close()

    compliant_turns = sum(1 for t in f1_progression if t["compliant"])
    compliance_rate = compliant_turns / num_chunks if num_chunks > 0 else 0.0
    total_input = sum(t["input"] for t in turn_tokens)
    total_output = sum(t["output"] for t in turn_tokens)

    print(f"\n  Condition A V2 Summary (Task {task_idx}):")
    print(f"    Compliance (strict ==): {compliance_rate:.0%} ({compliant_turns}/{num_chunks} turns)")
    print(f"    Phantom chunks: {phantom_chunks_detected}")
    print(f"    F1 progression: {[t['f1'] for t in f1_progression]}")
    print(f"    Final F1={f1_progression[-1]['f1'] if f1_progression else None}")
    print(f"    Iteration counts: {[t['iteration_count'] for t in f1_progression]}")
    print(f"    Total input tokens: {total_input}  Total output tokens: {total_output}")

    return {
        "condition": "A_incremental_label_aware_v2_sequential",
        "version": 2,
        "task_idx": task_idx,
        "model": model,
        "num_chunks": num_chunks,
        "max_chunk_chars": max_chunk_chars,
        "context_type": "labeled_sequential",
        "chunking_strategy": "sequential_5k_from_first_25k",
        "check_pair_type": "label_aware",
        "qualifying_labels": list(qualifying_labels),
        "compliance_rate": compliance_rate,
        "phantom_chunks_detected": phantom_chunks_detected,
        "f1_progression": f1_progression,
        "final_f1": f1_progression[-1]["f1"] if f1_progression else None,
        "final_precision": f1_progression[-1]["precision"] if f1_progression else None,
        "final_recall": f1_progression[-1]["recall"] if f1_progression else None,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "per_turn_tokens": turn_tokens,
    }


# ---------------------------------------------------------------------------
# Condition B V2: Non-incremental baseline (same as V1, first 5K of labeled)
# ---------------------------------------------------------------------------

def run_condition_b_v2(
    labeled_context: str,
    gold_pairs: set,
    api_key: str,
    task_idx: int = 1,
    max_chars: int = 5000,
    model: str = "gpt-4o-mini",
    verbose: bool = False,
) -> dict:
    """
    Condition B V2 (Label-Aware): Non-incremental baseline, first 5K chars.

    Same as V1 but with iteration count logging (fixes dead code iteration_count_proxy=0).
    Uses first 5K chars of labeled context = same as Condition A's first chunk.
    """
    from rlm.core.rlm import RLM
    from rlm.utils.prompts import INCREMENTAL_SYSTEM_PROMPT

    os.environ["OPENAI_API_KEY"] = api_key

    checker_setup = make_label_checker_setup(task_idx)
    qualifying_labels = TASK_QUALIFYING_LABELS[task_idx]
    label_desc = TASK_LABEL_DESCRIPTION[task_idx]

    context_chunk = labeled_context[:max_chars]

    print(f"\n{'=' * 70}")
    print(f"CONDITION B V2 (Label-Aware): Non-incremental baseline (1 turn, {len(context_chunk)} chars)")
    print(f"Task {task_idx} | Qualifying: {label_desc}")
    print(f"{'=' * 70}")

    root_prompt = CHUNK_PROMPT_LABEL_AWARE_V2.format(
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

    # V2 FIX: Actually measure iteration count (was dead code in V1)
    iteration_count = _extract_iteration_count(completion.usage_summary)

    prune_count = 0
    if hasattr(rlm, "history_manager") and rlm.history_manager is not None:
        prune_count = rlm.history_manager._prune_count
        print(f"    prune_count: {prune_count}")

    rlm.close()

    compliant = chunks_processed >= 1
    f1_result = compute_f1(pairs, gold_pairs)
    input_tokens, output_tokens = _extract_tokens(completion.usage_summary)

    print(f"    compliant: {compliant}  chunks_processed: {chunks_processed}")
    print(f"    pairs: {len(pairs)}  pair_checks: {pair_checks}")
    print(f"    iteration_count: {iteration_count}")
    print(f"    F1={f1_result['f1']}  P={f1_result['precision']}  R={f1_result['recall']}")
    print(f"    input_tokens: {input_tokens}  output_tokens: {output_tokens}  elapsed: {elapsed:.1f}s")

    return {
        "condition": "B_matched_budget_label_aware_v2",
        "version": 2,
        "task_idx": task_idx,
        "model": model,
        "context_chars": len(context_chunk),
        "context_type": "labeled",
        "check_pair_type": "label_aware",
        "qualifying_labels": list(qualifying_labels),
        "compliant": compliant,
        "chunks_processed": chunks_processed,
        "pair_checks": pair_checks,
        "iteration_count": iteration_count,
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
        "prune_count": prune_count,
    }


# ---------------------------------------------------------------------------
# Condition C V2: Oracle on first 25K chars (same window as A's 5 chunks)
# ---------------------------------------------------------------------------

def run_condition_c_v2(
    labeled_context: str,
    gold_pairs: set,
    api_key: str,
    task_idx: int = 1,
    max_chars: int = 25000,
    model: str = "gpt-4o-mini",
    verbose: bool = False,
) -> dict:
    """
    Condition C V2 (Label-Aware): Oracle single-turn on first 25K chars.

    Now directly comparable to Condition A V2 — both see identical content
    (the same first 25K chars). This makes F1(A)/F1(C) a clean measure of
    incremental streaming vs single-pass oracle on the same budget.
    """
    from rlm.core.rlm import RLM
    from rlm.utils.prompts import INCREMENTAL_SYSTEM_PROMPT

    os.environ["OPENAI_API_KEY"] = api_key

    checker_setup = make_label_checker_setup(task_idx)
    qualifying_labels = TASK_QUALIFYING_LABELS[task_idx]
    label_desc = TASK_LABEL_DESCRIPTION[task_idx]

    context_full = labeled_context[:max_chars]

    print(f"\n{'=' * 70}")
    print(f"CONDITION C V2 (Label-Aware): Oracle (1 turn, {len(context_full)} chars, same as A V2)")
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

    iteration_count = _extract_iteration_count(completion.usage_summary)
    f1_result = compute_f1(pairs, gold_pairs)
    input_tokens, output_tokens = _extract_tokens(completion.usage_summary)

    print(f"    entities_found: {entities_found}  qualifying: {qualifying_count}")
    print(f"    pairs: {len(pairs)}")
    print(f"    iteration_count: {iteration_count}")
    print(f"    F1={f1_result['f1']}  P={f1_result['precision']}  R={f1_result['recall']}")
    print(f"    input_tokens: {input_tokens}  output_tokens: {output_tokens}  elapsed: {elapsed:.1f}s")

    return {
        "condition": "C_oracle_label_aware_v2",
        "version": 2,
        "task_idx": task_idx,
        "model": model,
        "context_chars": len(context_full),
        "context_type": "labeled_sequential",
        "check_pair_type": "label_aware",
        "qualifying_labels": list(qualifying_labels),
        "entities_found": entities_found,
        "qualifying_entities": qualifying_count,
        "iteration_count": iteration_count,
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
# Condition C Full: Oracle on COMPLETE labeled corpus (all 96K chars)
# ---------------------------------------------------------------------------

def run_condition_c_full(
    labeled_context: str,
    gold_pairs: set,
    api_key: str,
    task_idx: int = 1,
    model: str = "gpt-4o-mini",
    verbose: bool = False,
) -> dict:
    """
    Condition C Full: Oracle single-turn on ALL labeled chars (~96K).

    Priority 3 from Iteration 11 critique. This establishes the definitive
    coverage ceiling: F1 = [Z] ≈ 1.0 (if checker is correct and all pairs
    are findable in 96K chars). Anchors the 25K-window F1 (0.3424 from V1)
    as a fraction: "At 25K chars, we find [X]% of all pairs."

    Note: 96K labeled chars is a large context for gpt-4o-mini. The oracle
    prompt encodes all parsing in REPL code, so the model just executes it.
    """
    from rlm.core.rlm import RLM
    from rlm.utils.prompts import INCREMENTAL_SYSTEM_PROMPT

    os.environ["OPENAI_API_KEY"] = api_key

    checker_setup = make_label_checker_setup(task_idx)
    qualifying_labels = TASK_QUALIFYING_LABELS[task_idx]
    label_desc = TASK_LABEL_DESCRIPTION[task_idx]

    # Use FULL labeled context (all ~96K chars)
    context_full = labeled_context

    print(f"\n{'=' * 70}")
    print(f"CONDITION C FULL: Oracle on COMPLETE labeled corpus ({len(context_full)} chars)")
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

    iteration_count = _extract_iteration_count(completion.usage_summary)
    f1_result = compute_f1(pairs, gold_pairs)
    input_tokens, output_tokens = _extract_tokens(completion.usage_summary)

    print(f"    entities_found: {entities_found}  qualifying: {qualifying_count}")
    print(f"    pairs: {len(pairs)}")
    print(f"    iteration_count: {iteration_count}")
    print(f"    F1={f1_result['f1']}  P={f1_result['precision']}  R={f1_result['recall']}")
    print(f"    input_tokens: {input_tokens}  output_tokens: {output_tokens}  elapsed: {elapsed:.1f}s")

    return {
        "condition": "C_full_oracle_label_aware",
        "version": 2,
        "task_idx": task_idx,
        "model": model,
        "context_chars": len(context_full),
        "context_type": "labeled_full_corpus",
        "check_pair_type": "label_aware",
        "qualifying_labels": list(qualifying_labels),
        "entities_found": entities_found,
        "qualifying_entities": qualifying_count,
        "iteration_count": iteration_count,
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
# Run single task (all conditions)
# ---------------------------------------------------------------------------

def run_task_v2(
    task_idx: int,
    plain_context: str,
    labeled_context: str,
    api_key: str,
    model: str,
    num_chunks: int,
    max_chunk_chars: int,
    incremental_only: bool,
    conditions_only: bool,
    condition_c_full: bool,
    verbose: bool,
) -> dict:
    """Run Conditions A/B/C V2 (label-aware, sequential) for a single task."""
    gold_pairs = compute_gold_pairs(labeled_context, task_idx=task_idx)
    print(f"\nGold pairs (Task {task_idx}, full labeled context): {len(gold_pairs)}")

    # Count gold pairs within just the first 25K chars to anchor the comparison
    from eval.rlm_pipeline_experiment import compute_f1 as _cf1
    import re
    window_25k = labeled_context[:num_chunks * max_chunk_chars]
    entities_25k: dict = {}
    qualifying_labels = TASK_QUALIFYING_LABELS[task_idx]
    for line in window_25k.split('\n'):
        m = re.search(r'User: (\d+).*?\|\| Label: (.+?)$', line)
        if m:
            uid = m.group(1)
            label = m.group(2).strip().lower()
            if uid not in entities_25k:
                entities_25k[uid] = {"qualifying": False}
            if label in qualifying_labels:
                entities_25k[uid]["qualifying"] = True
    qualifying_25k = [uid for uid, attrs in entities_25k.items() if attrs["qualifying"]]
    pairs_25k_count = len(qualifying_25k) * (len(qualifying_25k) - 1) // 2
    print(f"  Within first {num_chunks * max_chunk_chars} chars: {len(entities_25k)} entities, {len(qualifying_25k)} qualifying, {pairs_25k_count} pairs")
    print(f"  Coverage ceiling for A/C V2: {pairs_25k_count}/{len(gold_pairs)} = {pairs_25k_count/len(gold_pairs):.1%} of all pairs")

    results = {
        "task_idx": task_idx,
        "model": model,
        "version": 2,
        "gold_pairs_count": len(gold_pairs),
        "pairs_in_first_25k": pairs_25k_count,
        "entities_in_first_25k": len(entities_25k),
        "qualifying_entities_in_first_25k": len(qualifying_25k),
        "coverage_ceiling_25k": round(pairs_25k_count / len(gold_pairs), 4) if gold_pairs else 0,
        "check_pair_type": "label_aware",
        "qualifying_labels": list(qualifying_labels),
        "iteration": 11,
        "chunking_strategy": "sequential_5k_windows",
    }

    if not conditions_only:
        result_a = run_condition_a_v2(
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
        result_b = run_condition_b_v2(
            labeled_context=labeled_context,
            gold_pairs=gold_pairs,
            api_key=api_key,
            task_idx=task_idx,
            max_chars=max_chunk_chars,
            model=model,
            verbose=verbose,
        )
        results["condition_b"] = result_b

        result_c = run_condition_c_v2(
            labeled_context=labeled_context,
            gold_pairs=gold_pairs,
            api_key=api_key,
            task_idx=task_idx,
            max_chars=num_chunks * max_chunk_chars,
            model=model,
            verbose=verbose,
        )
        results["condition_c"] = result_c

    if condition_c_full:
        result_c_full = run_condition_c_full(
            labeled_context=labeled_context,
            gold_pairs=gold_pairs,
            api_key=api_key,
            task_idx=task_idx,
            model=model,
            verbose=verbose,
        )
        results["condition_c_full"] = result_c_full

    # Summary table
    print(f"\n{'=' * 75}")
    print(f"V2 LABEL-AWARE COMPARISON TABLE — Task {task_idx} ({TASK_LABEL_DESCRIPTION[task_idx]})")
    print(f"FIXED: Sequential chunking, strict compliance metric, phantom-chunk detection")
    print(f"{'=' * 75}")
    print(f"{'Condition':<45} {'F1':>6} {'Prec':>8} {'Recall':>8} {'InToks':>8}")
    print("-" * 80)

    if "condition_a" in results:
        ra = results["condition_a"]
        print(f"{'A V2: Incremental (k=5, sequential 5K/chunk)':<45} "
              f"{ra.get('final_f1', 'N/A'):>6} "
              f"{ra.get('final_precision', 'N/A'):>8} "
              f"{ra.get('final_recall', 'N/A'):>8} "
              f"{ra.get('total_input_tokens', 'N/A'):>8}")
        for t in ra["f1_progression"]:
            phantom_marker = " ⚠PHANTOM" if t.get("phantom_chunk") else ""
            print(f"  k={t['chunk']}: F1={t['f1']}  P={t['precision']}  R={t['recall']}"
                  f"  pairs={t['pairs']}  iters={t.get('iteration_count', '?')}  "
                  f"prune={t.get('prune_count', 0)}{phantom_marker}")

    if "condition_b" in results:
        rb = results["condition_b"]
        print(f"{'B V2: Baseline (1T, 5K, labeled)':<45} "
              f"{rb.get('f1', 'N/A'):>6} "
              f"{rb.get('precision', 'N/A'):>8} "
              f"{rb.get('recall', 'N/A'):>8} "
              f"{rb.get('input_tokens', 'N/A'):>8}")

    if "condition_c" in results:
        rc = results["condition_c"]
        print(f"{'C V2: Oracle (1T, 25K, SAME window as A)':<45} "
              f"{rc.get('f1', 'N/A'):>6} "
              f"{rc.get('precision', 'N/A'):>8} "
              f"{rc.get('recall', 'N/A'):>8} "
              f"{rc.get('input_tokens', 'N/A'):>8}")

    if "condition_c_full" in results:
        rcf = results["condition_c_full"]
        print(f"{'C Full: Oracle (1T, 96K, full corpus)':<45} "
              f"{rcf.get('f1', 'N/A'):>6} "
              f"{rcf.get('precision', 'N/A'):>8} "
              f"{rcf.get('recall', 'N/A'):>8} "
              f"{rcf.get('input_tokens', 'N/A'):>8}")

    # A vs C comparison
    if "condition_a" in results and "condition_c" in results:
        f1_a = results["condition_a"].get("final_f1", 0) or 0
        f1_c = results["condition_c"].get("f1", 0) or 0
        ratio = f1_a / f1_c if f1_c > 0 else 0
        print(f"\n  A/C ratio (V2, valid comparison): {ratio:.1%} of oracle")
        print(f"  Coverage ceiling (first 25K): {results.get('coverage_ceiling_25k', '?'):.1%} of all pairs")
        results["a_vs_c_ratio"] = round(ratio, 4)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Label-Aware Experiment V2 — Sequential Chunking Fix (Iteration 11)"
    )
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--num-chunks", type=int, default=5)
    parser.add_argument("--max-chunk-chars", type=int, default=5000)
    parser.add_argument("--task-idx", type=int, default=1, choices=[1, 3, 6],
                        help="Task index")
    parser.add_argument("--all-tasks", action="store_true",
                        help="Run Tasks 1, 3, and 6 sequentially")
    parser.add_argument("--condition-c-full", action="store_true",
                        help="Also run oracle on full 96K labeled corpus (Priority 3)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--incremental-only", action="store_true",
                        help="Only run Condition A (skip B and C)")
    parser.add_argument("--conditions-only", action="store_true",
                        help="Only run Conditions B and C (skip A)")
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

    # Verify labels present
    import re
    label_count = len(re.findall(r'\|\| Label: .+?$', labeled_context[:5000], re.MULTILINE))
    print(f"Label occurrences in first 5K chars: {label_count}")

    # Show chunk layout for transparency
    total_chars = args.num_chunks * args.max_chunk_chars
    print(f"\nV2 CHUNKING LAYOUT (sequential, fixed):")
    print(f"  Total window: labeled_context[:{total_chars}] ({total_chars} chars)")
    for i in range(args.num_chunks):
        start = i * args.max_chunk_chars
        end = (i + 1) * args.max_chunk_chars
        print(f"  Chunk {i}: chars {start}-{end}")
    print(f"  Condition C (oracle): labeled_context[:{total_chars}] (SAME window)")
    print(f"  → A and C now see identical content. F1(A)/F1(C) is a valid comparison.\n")

    tasks_to_run = [1, 3, 6] if args.all_tasks else [args.task_idx]
    all_results = {}

    for task_idx in tasks_to_run:
        print(f"\n{'#' * 75}")
        print(f"# TASK {task_idx}: {TASK_LABEL_DESCRIPTION[task_idx]}")
        print(f"{'#' * 75}")

        task_results = run_task_v2(
            task_idx=task_idx,
            plain_context=plain_context,
            labeled_context=labeled_context,
            api_key=api_key,
            model=args.model,
            num_chunks=args.num_chunks,
            max_chunk_chars=args.max_chunk_chars,
            incremental_only=args.incremental_only,
            conditions_only=args.conditions_only,
            condition_c_full=args.condition_c_full,
            verbose=not args.quiet,
        )
        all_results[f"task_{task_idx}"] = task_results

        # Save per-task
        if args.output is None:
            out_path = Path(f"results/streaming/label_aware_task{task_idx}_v2_results.json")
        else:
            out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(task_results, f, indent=2, default=str)
        print(f"Task {task_idx} V2 results saved to {out_path}")

    # Save combined results if multiple tasks
    if args.all_tasks:
        combined_path = Path("results/streaming/label_aware_all_tasks_v2.json")
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nCombined V2 results saved to {combined_path}")

    return all_results


if __name__ == "__main__":
    main()
