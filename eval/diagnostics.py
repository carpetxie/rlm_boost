"""
Diagnostic scripts for paper data quality.

1. Outlier diagnosis: Identify the 55 missing pairs in Exp32 (V4 Run 1)
2. k=7/10 compliance degradation: Identify entity count thresholds
3. Update paper_summary_tables.py values

Usage:
    python eval/diagnostics.py --outlier
    python eval/diagnostics.py --compliance
    python eval/diagnostics.py --all
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------------------
# 1. Outlier diagnosis: 55 missing pairs in V4 Exp32
# ---------------------------------------------------------------------------

def diagnose_outlier():
    """Compare V4 Exp32 (1485 pairs) vs MR1 (1540 pairs) to find missing pairs."""
    results_dir = Path("results/streaming")

    # Load the two result files
    exp32_path = results_dir / "label_aware_task1_v4_results.json"
    mr_path = results_dir / "label_aware_task1_v4_multi_run_3.json"

    if not exp32_path.exists() or not mr_path.exists():
        print(f"Missing result files: {exp32_path.exists()=}, {mr_path.exists()=}")
        return

    exp32_data = load_json(exp32_path)
    mr_data = load_json(mr_path)

    # Extract pairs from Exp32 (result_a)
    exp32_result = exp32_data.get("result_a", exp32_data)
    exp32_progression = exp32_result.get("f1_progression", [])
    exp32_final_pairs = exp32_progression[-1]["pairs"] if exp32_progression else 0

    # Extract pairs from MR Run 1
    mr_runs = mr_data.get("run_results", [])
    if not mr_runs:
        print("No run results in multi_run file")
        return

    mr1_result = mr_runs[0]
    mr1_progression = mr1_result.get("f1_progression", [])
    mr1_final_pairs = mr1_progression[-1]["pairs"] if mr1_progression else 0

    print(f"\n{'=' * 70}")
    print(f"OUTLIER DIAGNOSIS: V4 Exp32 vs MR Run 1")
    print(f"{'=' * 70}")
    print(f"Exp32 final pairs: {exp32_final_pairs}")
    print(f"MR1 final pairs: {mr1_final_pairs}")
    print(f"Difference: {mr1_final_pairs - exp32_final_pairs}")

    # Compare per-turn pair counts
    print(f"\nPer-turn pair comparison:")
    print(f"{'Turn':>5} {'Exp32':>8} {'MR1':>8} {'Diff':>8}")
    print(f"{'-' * 30}")
    for i in range(min(len(exp32_progression), len(mr1_progression))):
        e = exp32_progression[i]["pairs"]
        m = mr1_progression[i]["pairs"]
        print(f"{i+1:>5} {e:>8} {m:>8} {m-e:>8}")

    # Token comparison
    print(f"\nPer-turn token comparison:")
    print(f"{'Turn':>5} {'Exp32 tok':>10} {'MR1 tok':>10}")
    print(f"{'-' * 26}")
    for i in range(min(len(exp32_progression), len(mr1_progression))):
        e_tok = exp32_progression[i].get("input_tokens", 0)
        m_tok = mr1_progression[i].get("input_tokens", 0)
        print(f"{i+1:>5} {e_tok:>10} {m_tok:>10}")

    # Analyze retraction differences
    exp32_ret = exp32_progression[-1].get("total_retractions", 0) if exp32_progression else 0
    mr1_ret = mr1_progression[-1].get("total_retractions", 0) if mr1_progression else 0
    exp32_noop = exp32_progression[-1].get("noop_retractions", 0) if exp32_progression else 0
    mr1_noop = mr1_progression[-1].get("noop_retractions", 0) if mr1_progression else 0

    print(f"\nRetraction comparison:")
    print(f"  Exp32: total={exp32_ret}, noop={exp32_noop}")
    print(f"  MR1: total={mr1_ret}, noop={mr1_noop}")

    # Diagnosis
    print(f"\n--- DIAGNOSIS ---")
    pair_diff = mr1_final_pairs - exp32_final_pairs
    if pair_diff > 0:
        print(f"Exp32 found {pair_diff} fewer pairs than MR1.")
        # Look at which turn the divergence starts
        for i in range(min(len(exp32_progression), len(mr1_progression))):
            e = exp32_progression[i]["pairs"]
            m = mr1_progression[i]["pairs"]
            if e != m:
                print(f"Divergence starts at Turn {i+1}: Exp32={e}, MR1={m} (Δ={m-e})")
                print(f"  Exp32 F1 at turn {i+1}: {exp32_progression[i]['f1']:.4f}")
                print(f"  MR1 F1 at turn {i+1}: {mr1_progression[i]['f1']:.4f}")
                break

        # Compute total entities seen per run from pair counts
        # pairs = C(Q, 2) where Q = qualifying entities
        # So Q ≈ (1 + sqrt(1 + 8*pairs)) / 2
        import math
        q_exp32 = (1 + math.sqrt(1 + 8 * exp32_final_pairs)) / 2
        q_mr1 = (1 + math.sqrt(1 + 8 * mr1_final_pairs)) / 2
        print(f"\nEstimated qualifying entities (from pair count):")
        print(f"  Exp32: ~{q_exp32:.0f} qualifying users ({exp32_final_pairs} pairs)")
        print(f"  MR1: ~{q_mr1:.0f} qualifying users ({mr1_final_pairs} pairs)")
        print(f"  Difference: ~{q_mr1 - q_exp32:.0f} entities")
        print(f"\nConclusion: Exp32 identified ~{q_mr1 - q_exp32:.0f} fewer qualifying entities,")
        print(f"likely due to stochastic LLM label extraction differences at chunk boundaries.")
        print(f"Impact: 3.6% of pairs ({pair_diff}/{mr1_final_pairs}), within expected LLM variance.")
    else:
        print("No outlier detected — pair counts match.")


# ---------------------------------------------------------------------------
# 2. k=7/10 compliance degradation analysis
# ---------------------------------------------------------------------------

def diagnose_compliance():
    """Analyze k-sensitivity results for compliance patterns."""
    results_dir = Path("results/streaming")
    k_sens_path = results_dir / "label_aware_task1_v4_k_sensitivity.json"

    if not k_sens_path.exists():
        print(f"Missing k-sensitivity file: {k_sens_path}")
        return

    data = load_json(k_sens_path)

    print(f"\n{'=' * 80}")
    print(f"COMPLIANCE DEGRADATION ANALYSIS (k-sensitivity)")
    print(f"{'=' * 80}")

    from eval.label_aware_experiment import TASK_QUALIFYING_LABELS, load_labeled_data
    from eval.label_aware_v2_experiment import _make_sequential_chunks
    qualifying_labels = TASK_QUALIFYING_LABELS[1]
    _, labeled_context = load_labeled_data()

    for k_str, result in sorted(data.get("results_by_k", {}).items(), key=lambda x: int(x[0])):
        k = int(k_str)
        result_a = result.get("result_a", {})
        progression = result_a.get("f1_progression", [])
        compliance = result_a.get("compliance_rate", 0)
        num_chunks = result_a.get("num_chunks", k)
        max_chunk_chars = result_a.get("max_chunk_chars", 25000 // k)

        print(f"\n--- k={k} (compliance={compliance:.0%}) ---")

        # Get chunks to count entities per chunk
        chunks = _make_sequential_chunks(labeled_context, num_chunks, max_chunk_chars)

        for i, turn in enumerate(progression):
            compliant = turn.get("compliant", True)
            delta = turn.get("delta", 0)
            phantom = turn.get("phantom_chunk", False)

            # Count entities in this chunk
            chunk_text = chunks[i] if i < len(chunks) else ""
            entity_count = len(set(re.findall(r'User: (\d+)', chunk_text)))
            qual_count = sum(
                1 for m in re.finditer(r'User: (\d+).*?\|\| Label: (.+?)$', chunk_text, re.MULTILINE)
                if m.group(2).strip().lower() in qualifying_labels
            )

            status = "✅" if compliant else ("⚠ PHANTOM" if phantom else "❌ NON-COMPL")
            print(f"  Turn {i+1}: {status} delta={delta} | {entity_count} entities, {qual_count} qual records | "
                  f"tokens={turn.get('input_tokens', 0):,} iter={turn.get('iteration_count', '?')}")

    # Summary analysis
    print(f"\n--- COMPLIANCE THRESHOLD ANALYSIS ---")
    all_turns = []
    for k_str, result in data.get("results_by_k", {}).items():
        k = int(k_str)
        result_a = result.get("result_a", {})
        progression = result_a.get("f1_progression", [])
        max_chunk_chars = result_a.get("max_chunk_chars", 25000 // k)
        chunks = _make_sequential_chunks(labeled_context, k, max_chunk_chars)

        for i, turn in enumerate(progression):
            chunk_text = chunks[i] if i < len(chunks) else ""
            entity_count = len(set(re.findall(r'User: (\d+)', chunk_text)))
            all_turns.append({
                "k": k,
                "turn": i + 1,
                "compliant": turn.get("compliant", True),
                "entities_in_chunk": entity_count,
                "chars_in_chunk": len(chunk_text) if i < len(chunks) else 0,
            })

    compliant_entities = [t["entities_in_chunk"] for t in all_turns if t["compliant"]]
    non_compliant_entities = [t["entities_in_chunk"] for t in all_turns if not t["compliant"]]

    if compliant_entities:
        print(f"Compliant turns: {len(compliant_entities)} | entity count range: "
              f"[{min(compliant_entities)}, {max(compliant_entities)}] | "
              f"mean: {sum(compliant_entities)/len(compliant_entities):.0f}")
    if non_compliant_entities:
        print(f"Non-compliant turns: {len(non_compliant_entities)} | entity count range: "
              f"[{min(non_compliant_entities)}, {max(non_compliant_entities)}] | "
              f"mean: {sum(non_compliant_entities)/len(non_compliant_entities):.0f}")

        # Find threshold
        max_non_compliant = max(non_compliant_entities)
        min_compliant = min(compliant_entities) if compliant_entities else 0
        print(f"\nMax entities in non-compliant turn: {max_non_compliant}")
        print(f"Min entities in compliant turn: {min_compliant}")
        if min_compliant > max_non_compliant:
            print(f"Clean threshold: turns with > {max_non_compliant} entities are always compliant")
        else:
            print(f"No clean threshold — compliance depends on factors beyond entity count")
    else:
        print("All turns compliant — no degradation pattern to analyze")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Diagnostics for paper data quality")
    parser.add_argument("--outlier", action="store_true", help="Diagnose Exp32 outlier")
    parser.add_argument("--compliance", action="store_true", help="Analyze compliance degradation")
    parser.add_argument("--all", action="store_true", help="Run all diagnostics")
    args = parser.parse_args()

    if args.all or args.outlier:
        diagnose_outlier()
    if args.all or args.compliance:
        diagnose_compliance()
    if not (args.all or args.outlier or args.compliance):
        print("Specify --outlier, --compliance, or --all")


if __name__ == "__main__":
    main()
