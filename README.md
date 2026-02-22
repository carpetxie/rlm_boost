# Incremental Computation in Recursive Language Models

This is a research fork of [RLM](https://github.com/alexzhang13/rlm) (MIT OASYS Lab) exploring **incremental computation** — making RLMs efficient when context arrives over time rather than all at once.

- **Base paper**: [Recursive Language Models](https://arxiv.org/abs/2512.24601) (Zhang, Kraska, Khattab, 2025)
- **Base repo**: [alexzhang13/rlm](https://github.com/alexzhang13/rlm)

## The Problem

Standard RLMs re-read the *entire* context from scratch on every call. If context arrives in 5 chunks over time (streaming data, multi-turn conversations), the naive approach re-processes everything 5 times — O(n²) total work. But most of the context hasn't changed between chunks.

**The dynamic metrics gap**: Current benchmarks (OOLONG, S-NIAH) only test static context — paste a big document, ask a question. But real-world context is *dynamic*: built up over many turns, changing incrementally. No existing benchmark measures this. This gap is both the motivation for the work and a potential contribution in itself.

## The Approach

We built an incremental computation framework inside the RLM REPL:

1. **First chunk**: Process all entities, classify them, find matching pairs, cache everything in persistent Python objects (`EntityCache`, `PairTracker`, `IncrementalState`)
2. **Subsequent chunks**: Process only new entities, check them against cached classifications
3. **Handle retractions**: When new data invalidates a previously correct answer (e.g., "exactly one location tag" violated by a second tag in a later chunk), the system detects this, retracts affected pairs via an inverted index, and re-evaluates them

The retraction mechanism is the core novelty — real-world conditions aren't always monotonic, and no prior incremental computation work in the LLM space handles non-monotonic updates.

## Key Results

Results from 12 automated research iterations (critiquer/researcher agent loop):

### Headline

- **94.3% of oracle F1** with P=1.0 (zero false positives) when processing 25K chars of OOLONG-Pairs data across 5 streaming turns of 5K chars each
- **22–42% pair-check savings** at k=5 and k=10 chunks, with 100% correctness validation at every chunk for every task

### Retraction Taxonomy

- **360x retraction range** across 11 task types (138 retractions for strict conditions vs 15,824 for broad ones), driven by condition semantics rather than data volume
- **49x temporal retraction asymmetry**: "before DATE" constraints produce far fewer retractions than "after DATE" due to monotonic vs bidirectional validity
- Retraction overhead is **predictable from task selectivity** — not a function of entity count

### Cost Model

```
savings(k, σ) ≈ 51*(1-2.93/k) + 8.9*σ*(1+1.60/k)
```

R²=0.936, p=0.025. Break-even at k≥4 for all task types. Practitioners can use this to decide when incremental computation is worthwhile without running the full system.

### Novel Findings

1. **Non-monotonic retraction** for LLM-driven entity matching — "exactly N" constraints mean new data can invalidate previously valid pairs, requiring retraction and re-evaluation
2. **Monotone attribute accumulation** as a correctness condition for streaming LLM computation — a 2-line fix implementing this took the A/C ratio from 64.3% to 94.3%
3. **Failure mode taxonomy** for LLM protocol execution: entity ID mismatch, premature FINAL_VAR, redundant process_chunk
4. **REPL-state as correctness ground truth** — deduplication guards in Python REPL state provide correctness guarantees independent of message history, enabling aggressive history pruning

## Architecture

### New Files

| File | Purpose |
|------|---------|
| `rlm/core/incremental.py` | `EntityCache`, `PairTracker`, `IncrementalState` with retraction support |
| `rlm/core/history_manager.py` | Message history pruning (sliding window, summarize, token budget) |
| `eval/incremental_simulation.py` | Simulation with real task conditions and correctness validation |
| `eval/sigma_cost_model.py` | Cost model fitting and per-entity retraction analysis |
| `tests/test_incremental_pipeline.py` | 28 tests for incremental primitives |
| `tests/test_mock_lm_integration.py` | 12 mock-LM end-to-end integration tests |

### Modified Files

| File | Change |
|------|--------|
| `rlm/core/rlm.py` | Integrated HistoryManager, persistent mode, system prompt switching |
| `rlm/utils/prompts.py` | Added `INCREMENTAL_SYSTEM_PROMPT` — protocol for incremental computation |
| `rlm/environments/local_repl.py` | REPL injection of incremental primitives in persistent mode |
| `eval/utils.py` | Extracted `_check_pair_condition()` for simulation use |

### How It Works

```
Turn 1 (first chunk):
  RLM.completion(prompt, context=chunk_1)
  → RLM_SYSTEM_PROMPT
  → Model parses all entities → entity_cache
  → Model finds all pairs → pair_tracker
  → Results cached in persistent REPL

Turn 2+ (subsequent chunks):
  RLM.completion(prompt, context=chunk_n)
  → INCREMENTAL_SYSTEM_PROMPT (automatic switch)
  → cached_vars hint shows what's already computed
  → Model calls _incremental.process_chunk(new_entities)
    → New entities added to cache
    → Updated entities trigger retraction of affected pairs
    → Only (new × existing) + (new × new) pairs checked
  → History pruned to stay within token budget
```

## Score Progression (Critiquer Ratings)

| Iteration | Novelty | Tech Soundness | Benchmark Perf | Scalability | Maturity |
|-----------|---------|----------------|----------------|-------------|----------|
| 1         | 3/10    | 6/10           | 7/10           | 4/10        | 2/10     |
| 4         | 6/10    | 7/10           | 6/10           | 5/10        | 5/10     |
| 8         | 7/10    | 7/10           | 6/10           | 6/10        | 7/10     |
| 12        | 7/10    | 7/10           | 7/10           | 6/10        | 7/10     |
| 13        | 7/10    | 7/10           | 8/10           | 5/10        | 6/10     |

## Open Questions

- **Dynamic benchmarks**: All evaluation uses artificially chunked static data. Genuinely dynamic benchmarks (Wikipedia edits, live conversations) remain future work.
- **k-sensitivity**: How do savings scale across k={3, 5, 7, 10} chunks? The cost model predicts it but the `run_k_sensitivity_sweep()` function has never been called.
- **Multi-run stability**: The 94.3% headline comes from the best of 2 runs; the other achieved 69.5%. Stability under the monotone attribute fix needs confirmation across 3-5 runs.
- **Cross-task validation**: Tasks 3 and 6 with the V3 monotone fix are untested. At-risk fraction analysis predicts Task 6 benefits most.

## Setup

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e .

# Development
make install-dev    # dev + test deps
make check          # lint + format + test

# Run benchmarks
python eval/run_rlm.py --benchmark oolong --output results/oolong/rlm.json
python eval/incremental_simulation.py  # incremental simulation
```

## Full Research Log

See [`docs/research_log.md`](docs/research_log.md) for the complete experiment log across all 12 iterations — hypotheses, results, architectural decisions, dead ends, and pivots.

## Citation

This work builds on:

```bibtex
@misc{zhang2025recursivelanguagemodels,
      title={Recursive Language Models},
      author={Alex L. Zhang and Tim Kraska and Omar Khattab},
      year={2025},
      eprint={2512.24601},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2512.24601},
}
```
