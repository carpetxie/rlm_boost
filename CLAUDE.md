# CLAUDE.md: Project LEARNED-STATE

## 0. Vision

**Problem**: The incremental RLM achieves P=1.0 on OOLONG-Pairs with >95% token savings, but the state schema (EntityCache, PairTracker, monotone_attrs) is hardcoded for one benchmark. This raises the question: **is incremental reasoning truly general, or have we just optimized for OOLONG?**

**Core Question**: Can models *learn* their own state management strategies across diverse tasks, rather than having state schemas imposed on them? If so, what do learned state representations look like?

**Research Direction**: Run two complementary experiments that let models discover efficient incremental state management through learning, not through engineered schemas. Evaluate both on OOLONG-Pairs (existing strength) and a second, structurally different benchmark (generality). Compare against full-context baselines to measure token savings and maintain F1/Precision.

---

## 1. Experiment A: Learned State Tokens

### 1.1 Core Idea
Fine-tune a model to learn special **state tokens** that compress information from chunk N and are prepended to chunk N+1.

**Mechanism**:
- At the end of each chunk, the model outputs a special token sequence `<STATE>...compressed info...</STATE>`
- These tokens are extracted, prepended to the next chunk, and the model continues
- The model learns (via fine-tuning) what information to compress and how to utilize it

**Why it's novel**: No prior work systematically teaches models to learn their own compression strategy at token level. Prior work either uses LLM-generated summaries (lossy, expensive) or system-defined summaries. This lets the model define its own state representation.

### 1.2 Training Setup
- **Base Model**: Llama 2 7B (or Qwen 7B as alternative)
- **Fine-tuning Method**: LoRA (4-8 ranks, 16 alpha) for speed. ~2-4 hours on 2x A100s.
- **Loss Function**:
  - Primary: Next-turn prediction loss (predict task output given state + new chunk)
  - Auxiliary: State reconstruction loss (from `<STATE>` tokens, can the model reconstruct key info from previous chunk? Measure via auxiliary binary classification heads for entity counts, yes/no answers, etc.)
- **Data**: Synthetic incremental chunks created from existing benchmarks:
  - **OOLONG-Pairs**: Tasks 1, 3, 6 (existing data, use current format)
  - **Second Benchmark**: SQuAD 2.0 with artificial context arrival (split long documents into 2-3 chunks, model answers questions incrementally)

### 1.3 Inference
- At test time, run the model chunk-by-chunk
- Extract `<STATE>` tokens after each chunk
- Inject them into the prompt for the next chunk
- Measure token count: full context × number of chunks vs. state token approach

### 1.4 Evaluation Metrics
- **Token Efficiency**: `(full_context_tokens - incremental_tokens) / full_context_tokens`
- **Task Performance**: F1 (OOLONG), EM+F1 (SQuAD), Precision (for OOLONG-Pairs)
- **Generalization**: Does token efficiency on SQuAD match OOLONG? If one benchmark shows worse efficiency, why?
- **State Interpretability**: Sample `<STATE>` tokens, decode them, see if they're coherent (e.g., entity mentions, counts, yes/no patterns)

### 1.5 Projected Results
- **Expected token savings**: 70-85% (vs. 95%+ for hardcoded EntityCache, but this is learned, not engineered)
- **Expected F1 on OOLONG**: 95-99% (some loss due to learned compression, but acceptable)
- **Expected F1 on SQuAD**: 85-92% (lower than single-pass, but proof-of-concept that state learning works across tasks)
- **Success Bar**: >70% token savings on both benchmarks, F1 drop <5% from baseline

---

## 2. Experiment D: State Prompt Evolution

### 2.1 Core Idea
Use vLLM's custom inference hooks to inject a **learnable "state prompt"** (plain text summary) between chunks. Fine-tune the model to learn what this state prompt should contain and how to utilize it.

**Mechanism**:
- Chunk 0 arrives. Model reads it, outputs the next chunk's state prompt: `[STATE: entity1 has attribute X, entity2 has attribute Y, ...]`
- Chunk 1 arrives. Model reads state prompt + new chunk, updates state prompt for chunk 2
- State prompt is learned via fine-tuning on the same loss function as Experiment A
- **Difference from A**: State is text-based (human-readable) rather than learned token embeddings

**Why it's novel**: This is closer to A-Mem's Zettelkasten approach, but learned end-to-end rather than using LLM function calls. The model discovers what template to use for state (structured bullet points? entity lists? claim-evidence pairs?).

### 2.2 Training Setup
- **Base Model**: Same as A (Llama 2 7B or Qwen 7B)
- **Fine-tuning Method**: LoRA, same hyperparams as A. ~2-3 hours on 1x A100.
- **Loss Function**: Next-turn prediction + state coherence (auxiliary head predicts next state prompt structure)
- **Data**: Same as Experiment A (OOLONG + SQuAD)
- **Prompt Format**:
  ```
  Previous State:
  [STATE: <model-generated summary>]

  New Chunk:
  <current context>

  Task: <question or label instruction>
  Answer:
  ```

### 2.3 Inference
- Run model chunk-by-chunk
- After each chunk, extract everything between `[STATE:` and `]` as the state for the next chunk
- Keep state prompt in the context window for subsequent chunks

### 2.4 Evaluation Metrics
- Same as Experiment A: token efficiency, F1, generalization
- **Additional metric**: State prompt quality (human-readable? Does it actually summarize the chunk? Measure via ROUGE-L against manual summaries)
- **Comparison with A**: Which learns more stable state? Which generalizes better?

### 2.5 Projected Results
- **Expected token savings**: 60-80% (state prompts are longer than learned tokens, so lower savings)
- **Expected F1 on OOLONG**: 92-98%
- **Expected F1 on SQuAD**: 82-90%
- **Success Bar**: >60% token savings on both benchmarks, human-readable state

---

## 3. Research Narrative (What This Proves)

### The Problem We're Solving
Current incremental RLM hardcodes state management:
- **OOLONG-Pairs**: Entity tracking with attributes, pair matching, monotone constraints
- **Other tasks**: No clear schema → fall back to full context

### The Hypothesis
If we fine-tune models to learn state management end-to-end, they will:
1. Discover task-specific state representations without engineering them
2. Achieve reasonable (60-85%) token savings across diverse tasks
3. Maintain F1/Precision within acceptable loss from baseline

This would prove that incremental reasoning is a learnable pattern, not a hardcoded trick for one benchmark.

### Experiments as Evidence
- **Experiment A (Learned Tokens)**: Can models learn compact state representations? Token-level view of what's essential.
- **Experiment D (State Prompts)**: Can models learn structured state? Natural language view of what's important.
- **Cross-benchmark validation**: If both approaches work on OOLONG + SQuAD, we've shown generalization.

### The Contribution
**Title**: "Learning to Maintain State: Fine-tuning LLMs for Incremental Reasoning Across Benchmarks"

**Core claim**: Models can learn efficient incremental reasoning strategies without hardcoded schemas. Token savings (60-85%) are not as aggressive as engineered approaches (95%+) but generalize to new task types.

**Novelty**:
- First systematic study of learned state tokens in LLMs
- Demonstrates that incremental computation is learnable, not just engineering
- Benchmark-agnostic approach (unlike previous work tied to specific tasks)

---

## 4. Implementation Checklist

### Pre-Training Data Prep (1 hour)
- [ ] Convert OOLONG-Pairs Tasks 1, 3, 6 into sequential chunk format
- [ ] Download SQuAD 2.0, split long passages into 2-3 chunks, create incremental QA instances
- [ ] Create training/eval split: 80/20

### Experiment A Setup (1.5 hours)
- [ ] Modify Llama 2 7B to output `<STATE>...</STATE>` tokens via special tokens
- [ ] Set up LoRA config for fine-tuning
- [ ] Implement next-turn prediction loss + state reconstruction auxiliary loss
- [ ] Start training on OOLONG (2 hrs on GPU)

### Experiment D Setup (1 hour)
- [ ] Set up vLLM inference endpoint
- [ ] Implement state prompt extraction hook
- [ ] Set up prompt formatting pipeline

### Parallel Training (2-3 hours)
- [ ] Run both A and D fine-tuning jobs in parallel on separate GPUs
- [ ] Monitor loss curves

### Evaluation (2 hours)
- [ ] Run inference on OOLONG test set with both approaches
- [ ] Run inference on SQuAD test set
- [ ] Compute token efficiency, F1, Precision
- [ ] Aggregate results and compare

### Analysis & Write-up (1 hour)
- [ ] Compare token savings across benchmarks
- [ ] Identify which approach generalizes better
- [ ] Sample state representations (tokens for A, prompts for D)
- [ ] Document results

---

## 5. Success Criteria

**Hard success** (publishable):
- Both A and D achieve >60% token savings on both benchmarks
- F1 drop <7% from baseline on at least one benchmark for each approach
- Clear evidence that learning generalizes (same relative performance on OOLONG and SQuAD)

**Soft success** (interesting findings):
- One approach (A vs D) generalizes significantly better → insight into what makes good state
- State representations are human-interpretable → can explain what models learn to track
- Token savings differ substantially between benchmarks → task-specific state complexity

**Hard failure** (go back to drawing board):
- Token savings <50% on either benchmark
- F1 drops >15% from baseline
- Approach only works on OOLONG, fails on SQuAD

---

## 6. Model & Infrastructure Choices

**Model**: Llama 2 7B or Qwen 7B (both have excellent LoRA support, inference is fast, fits on 2x GPUs)

**Alternatives considered**:
- GPT-3.5: Closed-source, can't modify for state token output. Rejected.
- Llama 3.1 70B: Powerful but slow fine-tuning. Use only if 7B results are weak.
- Mistral 7B: Good baseline, but Llama 2 is proven.

**Serving**: vLLM for Experiment D inference hooks. TorchServe or local inference for A.

**GPU**: Assume 2x A100 (40GB) available. If only 1 GPU, run A and D sequentially (~6 hours total).

---

## 7. Metrics Deep Dive

### Token Efficiency
```
Efficiency = (full_context_tokens - incremental_tokens) / full_context_tokens
```
- Full context = all prior chunks + current chunk (baseline)
- Incremental tokens = state (tokens A, or characters in prompt D) + current chunk

### F1 & Precision
- **OOLONG-Pairs**: Compute F1 and Precision per task, average across Tasks 1, 3, 6
- **SQuAD 2.0**: Compute EM (exact match) and F1 per question
- **Comparison baseline**: Single-pass model (no incremental, no state) to measure ceiling

### State Quality (Experiment D only)
- Sample 10 state prompts, have a human rate: "Does this summary capture the key info from the chunk?"
- ROUGE-L between model-generated and human-written summaries (if available)

---

## 8. Timeline & Parallelization

With 2+ GPUs and careful sequencing:

| Task | Duration | GPU | Start | End |
|------|----------|-----|-------|-----|
| Data prep | 1 hr | CPU | 0:00 | 1:00 |
| Setup A | 1 hr | CPU | 1:00 | 2:00 |
| Setup D | 1 hr | CPU | 1:00 | 2:00 |
| Train A | 2 hrs | GPU1 | 2:00 | 4:00 |
| Train D | 2 hrs | GPU2 | 2:00 | 4:00 |
| Eval A | 1.5 hrs | GPU1 | 4:00 | 5:30 |
| Eval D | 1.5 hrs | GPU2 | 4:00 | 5:30 |
| Analysis | 1 hr | CPU | 5:30 | 6:30 |

**Total elapsed time: ~6.5 hours** (within 1 day with buffer for iteration)

---

## 9. Future Directions (Not in Scope)

- Scaling to 13B or 70B models for stronger baselines
- Multi-benchmark evaluation (more than 2)
- Reinforcement learning on task performance to further optimize state learning
- Distillation of learned state representations into interpretable rules
- State memory banks (episodic memory for state rollback on errors)

---

## 10. Key Decision Points

**If A outperforms D on generalization**: Focus on learned tokens as the mechanism. Investigate what makes token representations stable across tasks.

**If D outperforms A on generalization**: Focus on structured prompts as state. Investigate whether there's a "universal" state template that works across tasks.

**If both fail to generalize**: Pivot to analyzing why state learning is task-specific. This is still a valuable finding (indicates state is inherently task-dependent, not learnable end-to-end).

**If both are too slow at inference**: Reduce state size, increase chunk size, or use distillation to compress learned state.
