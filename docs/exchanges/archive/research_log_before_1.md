# RLM Research Log

## Status: Starting

No experiments run yet.

## Research Thrusts

### Thrust 1: Fine-Tuning Exploration
Explore fine-tuning open models to find ideal tuning methodologies and model choices.

### Thrust 2: Dynamic/Incremental RLM (Novel Contribution)
Architect an RLM that doesn't re-read the whole context every turn. Stateful Python objects, delta-based updates, incremental computation.

### Key Observation: The Dynamic Metrics Gap
Current benchmarks (OOLONG, S-NIAH) only test static context — paste a big document, ask a question. But the core thesis is that real-world context is *dynamic*: built up over many turns, changing incrementally. No existing benchmark tests this. This is both a problem (we can't measure our improvement) and an opportunity (filling this gap is itself a contribution). Exploring dynamic benchmarks is a high-priority research direction.

---

## Research Principles

1. **Kill dead ends fast.** If an approach hasn't shown progress after 2-3 iterations, stop. Analyze why it failed and pivot.
2. **Reason from results.** Failed experiments are data. Use them to infer what would work, not just try the next random thing.
3. **Extrapolate across experiments.** Look for patterns. If multiple approaches fail for the same reason, that reason is a finding.
4. **Every experiment needs a hypothesis.** "I'm trying X because results from Y suggest Z." Not "let's try X and see."

---

## Experiment Log

_Entries will be added as experiments are run._
