# Incremental Computation in Recursive Language Models

This is a research fork of [RLM](https://github.com/alexzhang13/rlm) (MIT OASYS Lab) exploring **incremental computation** — making RLMs efficient when context arrives over time rather than all at once.

- **Base paper**: [Recursive Language Models](https://arxiv.org/abs/2512.24601) (Zhang, Kraska, Khattab, 2025)
- **Base repo**: [alexzhang13/rlm](https://github.com/alexzhang13/rlm)

## The Problem

Standard RLMs re-read the *entire* context from scratch on every call. If context arrives in 5 chunks over time (streaming data, multi-turn conversations), the naive approach re-processes everything 5 times — O(k^2) total chunk-reads. But most of the context hasn't changed between chunks.

## The Approach

An incremental computation framework inside the RLM REPL:

1. **First chunk**: Process all entities, classify them, find matching pairs, cache everything in persistent Python objects (`EntityCache`, `PairTracker`, `IncrementalState`)
2. **Subsequent chunks**: Process only new entities, check them against cached classifications — O(k) total chunk-reads vs O(k^2)
3. **Handle retractions**: When new data invalidates a previously correct answer, the system detects this, retracts affected pairs via an inverted index, and re-evaluates them

## Key Results

Results from 23 automated research iterations (critiquer/researcher agent loop) on the OOLONG-Pairs benchmark:

### Headline: Same Quality, 71-91% Fewer Tokens

Head-to-head comparison of **Incremental RLM** vs **Full-Recompute RLM** (same framework, same context, same task — only difference is computational strategy):

| Approach | F1 | Precision | Tokens | Savings | Wall-Clock |
|----------|-----|-----------|--------|---------|------------|
| **Full-Recompute** (re-reads all chunks each turn) | 0.979 | 1.0 | 80K-246K | baseline | baseline |
| **Incremental** (processes only new chunks) | 0.979 | 1.0 | 18K-50K | **71-91%** | **2.7-4.9x faster** |

Validated across 3 tasks with **100% quality retention** (F1 identical in all cases) and **P=1.0** (zero false positives).

### Full-Corpus Performance

On the complete 96K-char OOLONG-Pairs corpus (k=5, ~19K chars/chunk):

| Task | F1 | Precision | Token Savings vs Full-Recompute |
|------|-----|-----------|-------------------------------|
| Task 1 (numeric/location) | **0.979 +/- 0.019** | 1.0 | 71-91% |
| Task 3 (description/abbreviation) | **0.993** | 1.0 | 77% |
| Task 6 (location/abbreviation) | **0.993** | 1.0 | 86% |

### Structural Savings Formula

```
savings = 1 - 2/(k+1)
```

At k=5: 66.7% structural bound. Empirical savings (71-91%) exceed this due to reduced per-turn prompt overhead. Deterministic and independent of LLM stochasticity.

### Dynamic Context (Entity Edits)

The retraction mechanism handles genuinely changing data — not just sequential chunks:

| Edits | Retractions Fired | Precision (all turns) | Post-Edit Continuation |
|-------|-------------------|----------------------|----------------------|
| 5 edits (2 down, 3 up) | 91 | 1.0 | works |
| 10 edits (5 down, 5 up) | 781 | 1.0 | works |

### Novel Contributions

1. **Non-monotonic retraction** for LLM-driven entity matching — 360x retraction range across task types
2. **Monotone attribute accumulation** as a correctness condition — 2-line library fix that raised A/C ratio from 64.3% to 94.3%
3. **Temporal retraction asymmetry** — 49x difference between "before" and "after" date constraints
4. **At-risk fraction diagnostic** — predicts monotone fix impact before running experiments (validated on 3 tasks)
5. **Library-vs-template principle** — moving invariants to the library eliminated stochastic compliance failures (60% to 100%)
6. **REPL-state as correctness ground truth** — deduplication guards provide correctness independent of message history
7. **Structural savings formula** — deterministic closed-form efficiency bound
8. **100% zero-shot protocol compliance** — LLMs follow the incremental protocol without fine-tuning

## Architecture

### Core Components

| Component | Purpose |
|-----------|---------|
| `rlm/core/incremental.py` | `EntityCache`, `PairTracker`, `IncrementalState` with retraction + monotone attribute support |
| `rlm/core/history_manager.py` | Message history pruning (sliding window, summarize, token budget) |
| `rlm/utils/prompts.py` | `RLM_SYSTEM_PROMPT` + `INCREMENTAL_SYSTEM_PROMPT` protocol |

### How It Works

```
Turn 1 (first chunk):
  RLM.completion(prompt, context=chunk_1)
  -> RLM_SYSTEM_PROMPT
  -> Model parses all entities -> entity_cache
  -> Model finds all pairs -> pair_tracker
  -> Results cached in persistent REPL

Turn 2+ (subsequent chunks):
  RLM.completion(prompt, context=chunk_n)
  -> INCREMENTAL_SYSTEM_PROMPT (automatic switch)
  -> Model calls _incremental.process_chunk(new_entities, monotone_attrs={"qualifying"})
    -> New entities added to cache
    -> Updated entities trigger retraction of affected pairs
    -> Only (new x existing) + (new x new) pairs checked
  -> History pruned to stay within token budget
```

## Score Progression (Critiquer Ratings)

| Iteration | Novelty | Tech Soundness | Benchmark Perf | Scalability | Maturity |
|-----------|---------|----------------|----------------|-------------|----------|
| 1         | 3/10    | 6/10           | 7/10           | 4/10        | 2/10     |
| 4         | 6/10    | 7/10           | 6/10           | 5/10        | 5/10     |
| 8         | 7/10    | 7/10           | 6/10           | 6/10        | 7/10     |
| 13        | 7/10    | 8/10           | 6.5/10         | 6/10        | 7.5/10   |
| 17        | 7/10    | 8.5/10         | 8.5/10         | 6.5/10      | 8/10     |

## Research Loop

This project uses an automated **critiquer/researcher agent loop** (`scripts/research_loop.sh`) that iteratively improves the architecture:

1. **Critiquer** (read-only): Reviews code + results, scores on 5 dimensions, identifies the single highest-impact improvement
2. **Researcher** (full access): Implements changes, runs experiments, updates the research log
3. **Repeat**: Each iteration ~30-60 minutes, fully automated

```bash
# Run 10 iterations
./scripts/research_loop.sh 10

# Resume from where you left off (auto-detects checkpoint)
./scripts/research_loop.sh 23
```

See [`docs/research_log.md`](docs/research_log.md) for the complete experiment log — hypotheses, results, architectural decisions, dead ends, and pivots.

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

## Open Questions

- **Cross-model validation**: All experiments use gpt-4o-mini. Cross-model generalization (gpt-4o, claude, gemini) is future work.
- **Non-monotone tasks**: Task 11 ("exactly N" constraints) shows F1=0.047 — the framework works but accuracy is poor. Principled scope boundary.
- **Dynamic benchmarks**: Entity edit experiments validate the mechanism, but genuinely dynamic benchmarks (Wikipedia edits, live conversations) remain future work.

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
