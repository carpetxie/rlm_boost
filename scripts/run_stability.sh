#!/usr/bin/env bash
# Runner script for multi-run stability experiments.
# Runs Condition A stability tests for Tasks 1, 3, and 6.
#
# Usage:
#   ./scripts/run_stability.sh              # all tasks, 3 runs each
#   ./scripts/run_stability.sh --task1-only  # just Task 1
#   NUM_RUNS=5 ./scripts/run_stability.sh   # 5 runs per task
#
# Prerequisites:
#   - OPENAI_API_KEY set in environment or in .env file
#   - Python environment activated (uv venv)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

NUM_RUNS="${NUM_RUNS:-3}"
K="${K:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-results/streaming}"
INCLUDE_D="${INCLUDE_D:-}"

# Parse flags
TASK1_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --task1-only) TASK1_ONLY=true ;;
        --include-d) INCLUDE_D="--include-d" ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

echo "========================================"
echo "Multi-Run Stability Experiment Runner"
echo "  NUM_RUNS=$NUM_RUNS  K=$K"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  INCLUDE_D=${INCLUDE_D:-no}"
echo "  TASK1_ONLY=$TASK1_ONLY"
echo "========================================"

run_task() {
    local task=$1
    echo ""
    echo ">>> Starting Task $task (k=$K, $NUM_RUNS runs) ..."
    python eval/multi_run_stability.py \
        --task "$task" \
        --k "$K" \
        --num-runs "$NUM_RUNS" \
        --output-dir "$OUTPUT_DIR" \
        $INCLUDE_D
    echo ">>> Task $task complete."
}

# Task 1 (always)
run_task 1

if [ "$TASK1_ONLY" = false ]; then
    # Task 3
    run_task 3

    # Task 6
    run_task 6
fi

echo ""
echo "========================================"
echo "All stability experiments complete."
echo "Results in: $OUTPUT_DIR"
echo "========================================"
