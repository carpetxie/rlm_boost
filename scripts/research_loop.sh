#!/usr/bin/env bash
# research_loop.sh — Automated researcher/critiquer loop for RLM architecture research
#
# Usage:
#   ./scripts/research_loop.sh [max_iterations] [model]
#
# Automatically resumes from the last completed iteration.
# If iteration N has both critique_N.md and researcher_response_N.md in the archive,
# it's considered complete. The loop starts from N+1.
#
# If iteration N has a critique but no researcher response, the loop resumes
# at iteration N (researcher phase).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAX_ITERATIONS="${1:-5}"
MODEL="${2:-opus}"
EXCHANGES="$REPO_ROOT/docs/exchanges"
ARCHIVE="$EXCHANGES/archive"
LOGFILE="$ARCHIVE/research_loop.log"

mkdir -p "$ARCHIVE"

# ── Auto-detect resume point ──────────────────────────────────────────
detect_resume_point() {
    local last_complete=0
    local has_partial_critique=false

    for f in "$ARCHIVE"/researcher_response_*.md; do
        [ -f "$f" ] || continue
        local n
        n=$(basename "$f" | sed 's/researcher_response_\([0-9]*\)\.md/\1/')
        if [ -f "$ARCHIVE/critique_${n}.md" ] && [ "$n" -gt "$last_complete" ]; then
            last_complete=$n
        fi
    done

    # Check if there's a critique for last_complete+1 without a researcher response
    local next=$(( last_complete + 1 ))
    if [ -f "$ARCHIVE/critique_${next}.md" ] && [ ! -f "$ARCHIVE/researcher_response_${next}.md" ]; then
        echo "${next}:researcher"
    else
        echo "$(( last_complete + 1 )):full"
    fi
}

RESUME_INFO=$(detect_resume_point)
START_ITERATION="${RESUME_INFO%%:*}"
RESUME_MODE="${RESUME_INFO##*:}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

LOOP_START=$(date +%s)

log() {
    local timestamp
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    local elapsed=$(( $(date +%s) - LOOP_START ))
    local mins=$(( elapsed / 60 ))
    local secs=$(( elapsed % 60 ))
    echo -e "${DIM}[${timestamp} +${mins}m${secs}s]${NC} $*"
    echo "[${timestamp} +${mins}m${secs}s] $(echo "$*" | sed 's/\x1b\[[0-9;]*m//g')" >> "$LOGFILE"
}

log_separator() {
    echo -e "${DIM}────────────────────────────────────────────────────────────${NC}"
}

file_stats() {
    local file="$1"
    local label="${2:-}"
    if [ -f "$file" ]; then
        local size lines words
        size=$(wc -c < "$file" | tr -d ' ')
        lines=$(wc -l < "$file" | tr -d ' ')
        words=$(wc -w < "$file" | tr -d ' ')
        log "  ${label}${CYAN}$(basename "$file")${NC}: ${words} words, ${lines} lines, ${size} bytes"
    else
        log "  ${label}${RED}$(basename "$file"): FILE NOT FOUND${NC}"
    fi
}

git_commit_push() {
    local msg="$1"
    log "${CYAN}[Git] Staging changes...${NC}"
    cd "$REPO_ROOT"
    local status_output
    status_output=$(git status --short 2>&1)
    if [ -z "$status_output" ]; then
        log "${DIM}[Git] No changes to commit.${NC}"
        return 0
    fi
    local changed_count
    changed_count=$(echo "$status_output" | wc -l | tr -d ' ')
    log "${DIM}[Git] ${changed_count} file(s) changed:${NC}"
    echo "$status_output" | head -20 | while read -r line; do
        log "  ${DIM}$line${NC}"
    done
    git add -A
    if git commit -m "$msg" > /dev/null 2>&1; then
        local sha
        sha=$(git rev-parse --short HEAD)
        log "${GREEN}[Git] Committed: ${sha} — ${msg}${NC}"
    else
        log "${YELLOW}[Git] Commit skipped.${NC}"
    fi
    log "${CYAN}[Git] Pushing...${NC}"
    if git push 2>&1 | tail -2 | while read -r line; do log "  ${DIM}$line${NC}"; done; then
        log "${GREEN}[Git] Push successful.${NC}"
    else
        log "${RED}[Git] Push failed!${NC}"
    fi
}

# ── Start ──────────────────────────────────────────────────────────────
echo "" > "$LOGFILE"
log "${BLUE}${BOLD}======================================================${NC}"
log "${BLUE}${BOLD}  RLM RESEARCH LOOP STARTED${NC}"
log "${BLUE}${BOLD}  Max iterations: $MAX_ITERATIONS | Model: $MODEL${NC}"
log "${BLUE}${BOLD}  Repo root: $REPO_ROOT${NC}"
log "${BLUE}${BOLD}  Log file: $LOGFILE${NC}"
if [ "$START_ITERATION" -gt 1 ] || [ "$RESUME_MODE" = "researcher" ]; then
    log "${GREEN}${BOLD}  Resuming from iteration $START_ITERATION ($RESUME_MODE phase)${NC}"
    log "${GREEN}${BOLD}  Iterations $((START_ITERATION - 1)) already complete in archive${NC}"
else
    log "${BLUE}${BOLD}  Starting fresh — no prior iterations found${NC}"
fi
log "${BLUE}${BOLD}======================================================${NC}"
echo ""

log "Pre-loop state:"
file_stats "$REPO_ROOT/docs/research_log.md" "Research log: "
log "  Git branch: $(git -C "$REPO_ROOT" branch --show-current 2>/dev/null || echo 'unknown')"
log "  Git HEAD: $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
log_separator
echo ""

FINAL_ITERATION=0
SKIP_CRITIQUER=false
if [ "$RESUME_MODE" = "researcher" ]; then
    SKIP_CRITIQUER=true
fi

for i in $(seq "$START_ITERATION" "$MAX_ITERATIONS"); do
    FINAL_ITERATION=$i
    ITER_START=$(date +%s)

    echo ""
    log "${YELLOW}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    log "${YELLOW}${BOLD}  ITERATION $i / $MAX_ITERATIONS${NC}"
    log "${YELLOW}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    # ── Phase 1: Critiquer ──────────────────────────────────────────────
    if [ "$SKIP_CRITIQUER" = true ]; then
        log "${YELLOW}[Phase 1: CRITIQUER] Skipped — resuming from researcher phase${NC}"
        log "  Using existing critique: $ARCHIVE/critique_${i}.md"
        SKIP_CRITIQUER=false
    else
    log "${RED}${BOLD}[Phase 1: CRITIQUER]${NC}"
    PHASE_START=$(date +%s)

    HISTORY_CONTEXT=""
    if [ "$i" -gt 1 ]; then
        HISTORY_CONTEXT="This is iteration $i. Prior critiques are in docs/exchanges/archive/. Read the researcher's latest response at docs/exchanges/researcher_response.md — pay close attention to pushbacks. Do NOT re-raise addressed points."
        log "  History context: referencing $((i-1)) prior critiques"
    else
        HISTORY_CONTEXT="This is the first iteration. Start by thoroughly exploring the codebase to understand the current architecture, then critique."
        log "  History context: first iteration"
    fi

    CRITIQUE_PROMPT="$(cat "$REPO_ROOT/docs/critique_prompt.md")

$HISTORY_CONTEXT

Read docs/research_log.md and review the actual code. Explore the full codebase — model architecture, training scripts, evaluation code, configs, benchmarks.

Your three priorities are NOVELTY, ROBUSTNESS, and STRENGTH OF CLAIM:
- NOVELTY: Is the Dynamic/Incremental RLM concept genuinely new? What would make it more novel? What experiments would increase the contribution?
- ROBUSTNESS: Are benchmarks testing the right things? Are there failure modes? Review actual code for correctness.
- STRENGTH: Are results as strong as evidence allows? Push for precise quantification.

MANDATORY: Apply the 3rd-Party Clarity Test to EVERY experiment. For each comparison, ask: would a skeptical engineer who has never seen this project instantly understand what's being compared, why the comparison is fair, and why the result matters? Flag any ambiguous or strawman comparisons as blocking issues.

CRITICAL CHECK: Does a head-to-head comparison of Incremental RLM vs Naive RLM (full recompute each turn) exist with real API numbers? If not, this is the #1 priority. Read the CRITICAL GAP section at the top of docs/research_log.md.

EXTERNAL REVIEWER CONCERNS (MANDATORY): An external reviewer raised three concerns that must be checked each iteration:
1. 'Caching is lossy compression' — Has losslessness been PROVEN with experiments (not just argued)?
2. 'Memory will blow up' — Has actual memory been PROFILED in bytes/KB/MB with scaling projections?
3. 'Only one benchmark' — Is there evidence on a second benchmark, or at least a rigorous characterization of the applicable problem class?
Read the EXTERNAL REVIEWER CONCERNS section in docs/critique_prompt.md and the NEXT CYCLE PRIORITIES in docs/research_log.md.

Write your critique to docs/exchanges/critique_latest.md. Use iteration number $i.

IMPORTANT: Do NOT set STATUS: ACCEPT. Always find concrete improvements — architectural changes, new experiments, robustness checks, code fixes. Be constructive but relentless."

    PROMPT_LEN=${#CRITIQUE_PROMPT}
    log "  Prompt length: ${PROMPT_LEN} chars"
    log "  Max turns: 25"
    log "  ${MAGENTA}Invoking claude (critiquer)...${NC}"

    cd "$REPO_ROOT"
    CLAUDE_START=$(date +%s)
    claude -p \
        --model "$MODEL" \
        --system-prompt "You are the critiquer agent for an ML architecture research project. You review code, experiments, and the research log. You may READ any file and run read-only bash commands, but ONLY write to docs/exchanges/critique_latest.md. Be technically rigorous. Suggest specific code changes and experiments." \
        --allowed-tools "Read,Write,Glob,Grep,Bash" \
        --max-turns 25 \
        --no-session-persistence \
        "$CRITIQUE_PROMPT" \
        > "$ARCHIVE/critique_${i}_log.txt" 2>&1
    CLAUDE_EXIT=$?
    CLAUDE_END=$(date +%s)
    CLAUDE_ELAPSED=$(( CLAUDE_END - CLAUDE_START ))

    log "  Claude exited with code ${BOLD}$CLAUDE_EXIT${NC} after ${BOLD}${CLAUDE_ELAPSED}s${NC} ($(( CLAUDE_ELAPSED / 60 ))m $(( CLAUDE_ELAPSED % 60 ))s)"
    file_stats "$ARCHIVE/critique_${i}_log.txt" "Agent log: "

    if [ -f "$EXCHANGES/critique_latest.md" ]; then
        cp "$EXCHANGES/critique_latest.md" "$ARCHIVE/critique_${i}.md"
        log "${GREEN}  Critique written.${NC}"
        file_stats "$EXCHANGES/critique_latest.md" "Critique: "

        log "  ${CYAN}Scores:${NC}"
        grep -E "^\|.*\|.*[0-9]+/10" "$EXCHANGES/critique_latest.md" 2>/dev/null | while read -r line; do
            log "    ${CYAN}$line${NC}"
        done

        PHASE_ELAPSED=$(( $(date +%s) - PHASE_START ))
        log "  Phase 1 total: ${PHASE_ELAPSED}s"
        log_separator

        # Git commit after critiquer
        git_commit_push "Iteration $i/$MAX_ITERATIONS: critiquer critique"
    else
        log "${RED}  FAILURE: No critique file produced!${NC}"
        tail -5 "$ARCHIVE/critique_${i}_log.txt" 2>/dev/null | while read -r line; do
            log "    ${DIM}$line${NC}"
        done
        log_separator
        continue
    fi
    fi  # end of critiquer skip/run block

    # ── Phase 2: Researcher ─────────────────────────────────────────────
    echo ""
    log "${GREEN}${BOLD}[Phase 2: RESEARCHER]${NC}"
    PHASE_START=$(date +%s)

    # Snapshot research log before revision
    if [ -f "$REPO_ROOT/docs/research_log.md" ]; then
        cp "$REPO_ROOT/docs/research_log.md" "$ARCHIVE/research_log_before_${i}.md"
        file_stats "$REPO_ROOT/docs/research_log.md" "Research log before: "
    fi

    RESEARCHER_PROMPT="$(cat "$REPO_ROOT/docs/researcher_prompt.md")

This is iteration $i of a maximum $MAX_ITERATIONS. Read the critique at docs/exchanges/critique_latest.md.

ITERATION PACING: You are on iteration $i of $MAX_ITERATIONS.
- If $i <= $(( MAX_ITERATIONS * 7 / 10 )): You are in EXPLORATION phase. Fix bugs, push new directions, address critique holes, run small/cheap experiments. Build infrastructure for the full experiment.
- If $i > $(( MAX_ITERATIONS * 7 / 10 )): You are in EXECUTION phase. Run the full production experiment with real API calls, full context, multiple tasks, multiple seeds. Produce the definitive paper-ready comparison table. No more exploration.

Your three priorities are NOVELTY, ROBUSTNESS, and STRENGTH OF CLAIM:
- NOVELTY: Implement architectural changes that make the Dynamic/Incremental RLM more novel. The prefix-sum analogy is the guiding principle.
- ROBUSTNESS: Write and run experiments. Every claim needs benchmark evidence. Test edge cases.
- STRENGTH: Quantify everything. When something works, measure how much. When it fails, diagnose why.

CRITICAL REQUIREMENT: The research MUST include a head-to-head comparison of Incremental RLM vs Naive RLM (full recompute each turn) on the same streaming task, measuring F1, tokens, pair checks, wall-clock time, and cost. This is the comparison that proves the system works. Read the CRITICAL GAP section at the top of docs/research_log.md.

EXTERNAL REVIEWER CONCERNS (MANDATORY): Three concerns from an external reviewer must be addressed with running code:
1. PROVE losslessness: Run with aggressive history pruning, show P=1.0. Add --verify-lossless mode.
2. PROFILE memory: tracemalloc on EntityCache/PairTracker at each turn. Report bytes. Extrapolate to 100K entities. Compare vs LLM context size.
3. CROSS-BENCHMARK: Characterize the problem class. Attempt a second benchmark if feasible.
Read the NEXT CYCLE PRIORITIES section at the bottom of docs/research_log.md for detailed specifications.

You have FULL access to the entire codebase. You can and should:
- Modify model architecture, training scripts, evaluation code, configs — ANYTHING
- Run training and evaluation (discover commands by reading Makefile, scripts, configs)
- Create new experiments, benchmarks, analysis scripts
- Generate plots and results
- Build entirely new components if that's what the critique calls for

Code changes are FIRST-CLASS outputs. A working prototype beats a paragraph of theory.

Write deliberation to docs/exchanges/researcher_response.md. Update docs/research_log.md with new results.

Do NOT set STATUS: CONVERGED. Always run at least one new experiment per iteration."

    PROMPT_LEN=${#RESEARCHER_PROMPT}
    log "  Prompt length: ${PROMPT_LEN} chars"
    log "  Max turns: 50"
    log "  ${MAGENTA}Invoking claude (researcher)...${NC}"

    cd "$REPO_ROOT"
    CLAUDE_START=$(date +%s)
    claude -p \
        --model "$MODEL" \
        --system-prompt "You are the researcher agent for an ML architecture project. You have FULL access to the entire codebase. Write code, run experiments, modify architecture, train models, run benchmarks. Code changes are first-class outputs. Update docs/research_log.md with results and docs/exchanges/researcher_response.md with deliberation." \
        --allowed-tools "Read,Write,Edit,Glob,Grep,Bash,NotebookEdit" \
        --max-turns 50 \
        --no-session-persistence \
        "$RESEARCHER_PROMPT" \
        > "$ARCHIVE/researcher_${i}_log.txt" 2>&1
    CLAUDE_EXIT=$?
    CLAUDE_END=$(date +%s)
    CLAUDE_ELAPSED=$(( CLAUDE_END - CLAUDE_START ))

    log "  Claude exited with code ${BOLD}$CLAUDE_EXIT${NC} after ${BOLD}${CLAUDE_ELAPSED}s${NC} ($(( CLAUDE_ELAPSED / 60 ))m $(( CLAUDE_ELAPSED % 60 ))s)"
    file_stats "$ARCHIVE/researcher_${i}_log.txt" "Agent log: "

    if [ -f "$EXCHANGES/researcher_response.md" ]; then
        cp "$EXCHANGES/researcher_response.md" "$ARCHIVE/researcher_response_${i}.md"
        if [ -f "$REPO_ROOT/docs/research_log.md" ]; then
            cp "$REPO_ROOT/docs/research_log.md" "$ARCHIVE/research_log_after_${i}.md"
        fi
        log "${GREEN}  Researcher response written.${NC}"
        file_stats "$EXCHANGES/researcher_response.md" "Response: "

        # Show benchmark results if any
        log "  ${CYAN}Benchmark results from response:${NC}"
        grep -E "^\|.*\|.*[0-9]" "$EXCHANGES/researcher_response.md" 2>/dev/null | head -10 | while read -r line; do
            log "    ${CYAN}$line${NC}"
        done

        PHASE_ELAPSED=$(( $(date +%s) - PHASE_START ))
        log "  Phase 2 total: ${PHASE_ELAPSED}s"
        log_separator
    else
        log "${RED}  FAILURE: No researcher response!${NC}"
        tail -5 "$ARCHIVE/researcher_${i}_log.txt" 2>/dev/null | while read -r line; do
            log "    ${DIM}$line${NC}"
        done
        log_separator
    fi

    # ── Git: commit each changed file individually, then push once ─────
    cd "$REPO_ROOT"
    CHANGED_FILES=$(git status --short 2>/dev/null | awk '{print $2}')
    if [ -n "$CHANGED_FILES" ]; then
        echo "$CHANGED_FILES" | while read -r filepath; do
            git add "$filepath"
            git commit -m "Iteration $i/$MAX_ITERATIONS [researcher]: $filepath" > /dev/null 2>&1 || true
        done
        FILE_COUNT=$(echo "$CHANGED_FILES" | wc -l | tr -d ' ')
        log "${GREEN}[Git] Committed ${FILE_COUNT} file(s) individually:${NC}"
        echo "$CHANGED_FILES" | while read -r filepath; do
            log "  ${DIM}$filepath${NC}"
        done
        log "${CYAN}[Git] Pushing all researcher commits...${NC}"
        if git push 2>&1 | tail -2 | while read -r line; do log "  ${DIM}$line${NC}"; done; then
            log "${GREEN}[Git] Push successful.${NC}"
        else
            log "${RED}[Git] Push failed!${NC}"
        fi
    else
        log "${DIM}[Git] No researcher changes to commit.${NC}"
    fi

    ITER_ELAPSED=$(( $(date +%s) - ITER_START ))
    echo ""
    log "${BLUE}${BOLD}━━━ Iteration $i complete in ${ITER_ELAPSED}s ($(( ITER_ELAPSED / 60 ))m $(( ITER_ELAPSED % 60 ))s) ━━━${NC}"
    echo ""
done

# ── Final Summary ───────────────────────────────────────────────────────
TOTAL_ELAPSED=$(( $(date +%s) - LOOP_START ))
TOTAL_MINS=$(( TOTAL_ELAPSED / 60 ))
TOTAL_SECS=$(( TOTAL_ELAPSED % 60 ))

echo ""
log "${BLUE}${BOLD}======================================================${NC}"
log "${BLUE}${BOLD}  RLM RESEARCH LOOP COMPLETE${NC}"
log "${BLUE}${BOLD}  Iterations: $FINAL_ITERATION | Total time: ${TOTAL_MINS}m ${TOTAL_SECS}s${NC}"
log "${BLUE}${BOLD}======================================================${NC}"
echo ""
log "${CYAN}${BOLD}Exit reason: All $MAX_ITERATIONS iterations completed.${NC}"

log "Outputs:"
log "  Research log: docs/research_log.md"
log "  Critique:     docs/exchanges/critique_latest.md"
log "  Response:     docs/exchanges/researcher_response.md"
log "  Full log:     $LOGFILE"

git_commit_push "Research loop complete after $FINAL_ITERATION iteration(s)"

if [ "$FINAL_ITERATION" -gt 1 ]; then
    log "${CYAN}${BOLD}Score progression:${NC}"
    for j in $(seq 1 "$FINAL_ITERATION"); do
        if [ -f "$ARCHIVE/critique_${j}.md" ]; then
            log "  ${BOLD}Iteration $j:${NC}"
            grep -E "^\|.*\|.*[0-9]+/10" "$ARCHIVE/critique_${j}.md" 2>/dev/null | while read -r line; do
                log "    $line"
            done
        fi
    done
fi

log ""
log "${CYAN}${BOLD}Timing summary:${NC}"
log "  Total elapsed: ${TOTAL_MINS}m ${TOTAL_SECS}s"
log "  Average per iteration: $(( TOTAL_ELAPSED / FINAL_ITERATION ))s"
log ""
log "${DIM}Full log saved to: $LOGFILE${NC}"
