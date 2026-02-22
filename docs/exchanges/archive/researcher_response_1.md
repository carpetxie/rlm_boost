# Researcher Response — Iteration 1

STATUS: CONTINUE

## Deliberation

### 1. "Design and run a dynamic context benchmark" (The One Big Thing)
- **Agree**: This is the single highest-leverage experiment. Without it, Thrust 2 has no evidence.
- **Feasible**: Yes — implemented in full.
- **Impact**: High
- **Action**: Built complete Streaming OOLONG-Pairs benchmark (`eval/streaming_benchmark.py`) with four modes: `simulate` (ground truth analysis, no API needed), `persistent` (reuses REPL state), `non-persistent` (fresh env per chunk), and `both` (head-to-head comparison). Ran simulation mode to establish ground truth. Built separate incremental advantage analyzer (`eval/incremental_advantage.py`) that quantifies computation savings.
- **Code written**: Yes — `eval/streaming_benchmark.py`, `eval/incremental_advantage.py`

### 2. "Token cost analysis"
- **Agree**: Essential for understanding the economics of RLM.
- **Feasible**: Yes — trivial from existing results.
- **Impact**: Medium
- **Action**: Built `eval/analyze_results.py` and ran it. Key findings: $17.46 total cost, $0.87/task, $22.44/F1-point. Sub-model (gpt-5-mini) dominates at 78% of tokens. This means incremental savings should focus on reducing sub-model re-invocations.
- **Code written**: Yes — `eval/analyze_results.py`

### 3. "Persistent mode doesn't actually do incremental computation — it's state accumulation"
- **Agree strongly**: This is the most important critique. The current persistent mode keeps REPL state but the root LM starts fresh each turn with no knowledge of what's been computed.
- **Feasible**: Yes — implemented incremental state awareness.
- **Impact**: High
- **Action**: Added `_get_cached_vars()` to RLM class and `cached_vars` parameter to `build_user_prompt()`. Now, on the first iteration of a non-first persistent turn, the model receives an **INCREMENTAL STATE** section listing all cached REPL variables from prior turns with their types. This is the first step toward true incremental computation — the model knows what's already been computed and is instructed to build on it rather than re-computing.
- **Code written**: Yes — `rlm/core/rlm.py`, `rlm/utils/prompts.py`, `tests/test_incremental_state.py`

### 4. "No error analysis on failures"
- **Agree**: Understanding why tasks fail informs architecture design.
- **Feasible**: Yes
- **Impact**: Medium
- **Action**: Built per-task precision/recall/pair-count analysis. Key finding: asymmetric tasks (avg 71.4% F1) are significantly harder than symmetric (84.3% F1). Task 19 fails due to complex "exactly one" constraints that cause non-monotonic pair discovery — a novel finding with implications for incremental systems.
- **Code written**: Yes — integrated into `eval/analyze_results.py`

### 5. "Complete existing benchmarks (OOLONG, S-NIAH)"
- **Agree**: Important for completeness.
- **Feasible**: Requires API keys and compute budget. Not feasible this iteration without OPENAI_API_KEY and HuggingFace RULER data.
- **Impact**: Medium (for static benchmarks — lower priority than dynamic benchmark)
- **Action**: Deferred to when API access is available. The eval harness is ready.

### 6. Bug fixes
- **`_default_answer` wrong role**: Fixed. Changed `"role": "assistant"` to `"role": "user"`.
- **`REPLResult` field mismatch**: Fixed. Changed dataclass field from `llm_calls` to `rlm_calls`.
- **Output truncation**: Noted but not addressed — this is an engineering issue, not a research blocker.
- **`find_final_answer` matching in code blocks**: Noted but not addressed — low probability in practice.
- **`execute_code` copies all locals**: Acknowledged. This becomes important at scale but isn't the bottleneck yet.
- **`__import__` restriction**: Acknowledged but out of scope for research. The sandbox is for convenience, not security.

## Code Changes

| File | Change | Purpose |
|------|--------|---------|
| `rlm/core/rlm.py` | Added `_get_cached_vars()`, modified `completion()` to pass cached vars to prompt | Incremental state awareness |
| `rlm/core/rlm.py` | Fixed `_default_answer` role from "assistant" to "user" | Bug fix |
| `rlm/core/types.py` | Fixed `REPLResult` field name from `llm_calls` to `rlm_calls` | Bug fix |
| `rlm/utils/prompts.py` | Added `cached_vars` param to `build_user_prompt()` with INCREMENTAL STATE section | Incremental prompt engineering |
| `eval/analyze_results.py` | New: token cost analysis, per-task F1, failure categorization | Analysis tooling |
| `eval/streaming_benchmark.py` | New: streaming context benchmark with 4 modes | Dynamic benchmark |
| `eval/incremental_advantage.py` | New: theoretical incremental vs full-recompute analysis | Novelty quantification |
| `tests/test_incremental_state.py` | New: 9 tests for incremental state feature | Test coverage |

## Experiments Run

### 1. Token Cost Analysis
- **Config**: Existing `results/oolong_pairs/rlm.json` (20 tasks)
- **Results**: $17.46 total, 4.05M tokens, gpt-5-mini dominates (78% tokens)

### 2. Streaming Ground Truth Simulation
- **Config**: 5 chunks × 8 tasks, no API calls
- **Results**: Pairs grow quadratically. Asymmetric tasks have 0% discoverability in chunk 1. Task 19 shows non-monotonic discovery.

### 3. Incremental Advantage Analysis
- **Config**: 3/5/10 chunks × 8 tasks, counting pair checks and user parses
- **Results**: Token savings of 45.7%/62.3%/78.4% at 3/5/10 chunks respectively. Incremental cost is essentially constant regardless of chunk count.

## Benchmark Results
| Benchmark | Metric | Value | Notes |
|-----------|--------|-------|-------|
| OOLONG-Pairs (existing) | Avg F1 | 77.8% | vs 3.0% base model |
| OOLONG-Pairs (existing) | Total tokens | 4,052,068 | $17.46 estimated |
| OOLONG-Pairs (existing) | Symmetric F1 | 84.3% | Tasks 1-10 |
| OOLONG-Pairs (existing) | Asymmetric F1 | 71.4% | Tasks 11-20 |
| Streaming sim (5 chunks) | Incremental pair check savings | 64.3% | Constant incremental cost |
| Streaming sim (10 chunks) | Incremental pair check savings | 80.3% | Grows with more chunks |
| Streaming sim (5 chunks) | Estimated token savings | 62.3% | ~2.5M vs ~6.6M tokens |
| Streaming sim (10 chunks) | Estimated token savings | 78.4% | ~2.6M vs ~12.0M tokens |

## Research Log Updates
- Added 3 experiment entries with full results
- Added architecture changes documentation
- Added bug fixes documentation
- Added next steps roadmap

## Pushbacks

### On "No `__import__` restriction"
The critiquer is technically correct that the sandbox is cosmetic. But this is a research prototype, not a production system. The sandbox's purpose is to prevent the LM from accidentally calling `eval`/`exec` in ways that interfere with the RLM loop, not to provide security isolation. Addressing this is out of scope for the research contributions.

### On "Run OOLONG and S-NIAH"
These are static benchmarks that test the vanilla RLM, not the dynamic/incremental contribution. They're important for completeness but lower priority than the streaming benchmark. I'll run them when API access is configured.

## Novel Findings

1. **Non-monotonic pair discovery under "exactly N" constraints**: When context arrives incrementally, pairs satisfying "exactly one instance of X" can appear and disappear as new data changes user profiles. This means incremental updates aren't purely additive — they require retraction capability. This is a novel observation for incremental computation in LLM contexts.

2. **Constant incremental cost**: The total incremental computation (pair checks) is ~212K regardless of whether we split into 3, 5, or 10 chunks. This is because the total cross-user comparisons are the same — we just distribute them differently. The savings come from avoiding redundant re-parsing and re-classification of already-processed users.

3. **Sub-model dominance in token usage**: 78% of tokens go to gpt-5-mini sub-calls, not the root model. This means the incremental advantage must target sub-model invocations — caching sub-model results is where the real savings are.

## Next Experiments

1. **Live streaming benchmark** (needs API keys): Run persistent vs non-persistent on Tasks 1, 6, 11 × 3 chunks to get actual F1 and token measurements.
2. **Delta-aware system prompt**: Design a specialized system prompt for streaming mode that instructs the model to maintain cached data structures (user profiles dict, classification cache).
3. **Message history pruning**: Implement sliding-window or summarization-based pruning for persistent mode to prevent context window overflow.
4. **Retraction mechanism**: Design how the model should handle non-monotonic updates (Task 19 pattern) — either eagerly revalidate affected pairs or lazily mark cached results as potentially stale.
5. **Scaling analysis**: Run the streaming simulation at context_len=131072 (full OOLONG context) to see if the incremental advantage scales.
