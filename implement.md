# RLM Replication Guide: Base Model vs. RLM (GPT-5)
### Benchmarks: OOLONG · OOLONG-Pairs · S-NIAH

This guide covers everything needed to replicate the **Base Model vs. RLM(GPT-5)** rows
from Table 1 and Figure 1 of the paper *Recursive Language Models* (Zhang et al., 2026,
arXiv:2512.24601).

---

## Table of Contents
1. [Prerequisites](#1-prerequisites)
2. [Repository Setup](#2-repository-setup)
3. [Dataset Acquisition](#3-dataset-acquisition)
4. [OOLONG-Pairs: All 20 Task Definitions](#4-oolong-pairs-all-20-task-definitions)
5. [System Prompts](#5-system-prompts)
6. [Writing the Evaluation Harness](#6-writing-the-evaluation-harness)
7. [Scoring Functions](#7-scoring-functions)
8. [Running the Experiments](#8-running-the-experiments)
9. [Cost Estimate](#9-cost-estimate)
10. [Known Pitfalls](#10-known-pitfalls)

---

## 1. Prerequisites

### 1.1 API Keys

Set the following environment variables (add to `.env` in the repo root — it is
auto-loaded via `python-dotenv`):

```bash
OPENAI_API_KEY=sk-...        # Required: GPT-5 (root LM) and GPT-5-mini (sub-LM calls)
```

The `OpenAIClient` in `rlm/clients/openai.py:14` reads `OPENAI_API_KEY` automatically.
No other keys are needed for the GPT-5 experiments.

### 1.2 Model Names

Verify the exact API identifiers on the OpenAI platform before running. The paper uses:

| Role | Paper Name | Likely API ID |
|------|-----------|---------------|
| Root LM (RLM) | `gpt-5` | Check `platform.openai.com/docs/models` |
| Sub-LM (RLM recursive calls) | `gpt-5-mini` | Check same |
| Base Model | `gpt-5` | Same as root |

The paper specifies GPT-5 uses **"medium reasoning"** sampling parameters
(Singh et al., 2025). Pass any required reasoning/sampling params via
`backend_kwargs` if needed (e.g., `reasoning_effort="medium"`).

### 1.3 Python Environment

- Python 3.11+ required (3.12 recommended per README)
- `uv` package manager

---

## 2. Repository Setup

```bash
git clone https://github.com/alexzhang13/rlm.git
cd rlm

# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .

# Confirm install
python -c "from rlm import RLM; print('OK')"
```

Create a `.env` file in the repo root:
```
OPENAI_API_KEY=sk-...
```

Create output directories:
```bash
mkdir -p logs results
```

---

## 3. Dataset Acquisition

### 3.1 OOLONG

**Paper:** Bertsch et al., 2025 — *OOLONG: Evaluating Long Context Reasoning and
Aggregation Capabilities* — arXiv:2511.02817

**Split used:** `trec_coarse` — 50 tasks
**Context length:** 131K tokens (fixed)
**Scoring:** `score(ŷ) = 0.75^|y−ŷ|` for numeric answers; exact match otherwise
**Metric reported:** Average score across 50 tasks

```bash
pip install datasets
```

```python
from datasets import load_dataset
ds = load_dataset("abertsch/oolong", split="trec_coarse")
# Each example has: 'input' (context), 'question', 'answer'
```

> Check the OOLONG paper/repo (arXiv:2511.02817) for the exact HuggingFace dataset path
> and split name — the above is illustrative.

Each task is structured as:
- **context** (`str`): the long TREC document corpus (~131K tokens)
- **question** (`str`): the aggregation query
- **answer** (`str` or `float`): ground truth

The RLM `prompt` argument should be the context. The question is passed separately
as `root_prompt` in `rlm.completion(prompt=context, root_prompt=question)`.

### 3.2 OOLONG-Pairs

**Source:** Constructed by the RLM paper authors from OOLONG `trec_coarse` ground-truth labels.
**Tasks:** 20 (see Section 4 below for full definitions)
**Context lengths evaluated:** `[1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]` tokens
**Scoring:** F1 score over predicted vs. gold pairs of user IDs
**Metric reported:** Average F1 across 20 tasks

The OOLONG-Pairs contexts are the **same OOLONG `trec_coarse` dataset**, truncated/padded
to each target length. Each task asks for pairs of user IDs satisfying joint properties.

**The OOLONG-Pairs dataset is not separately released** — you construct it from the OOLONG
`trec_coarse` data plus the 20 task definitions in Section 4. The ground-truth answers
require computing pair memberships from the OOLONG label annotations.

### 3.3 S-NIAH (Single Needle in a Haystack)

**Paper:** Hsieh et al., 2024 — *RULER: What's the Real Context Size of Your Long-Context
Language Models?* — arXiv:2404.06654
**GitHub:** https://github.com/hsiehjackson/RULER
**Split used:** Single-NIAH variant — 50 tasks
**Context lengths:** Scale from 2^13 (8K) to 2^20 (1M) tokens — 8 data points
**Scoring:** Exact match (% correct) — the needle is a specific phrase or number
**Metric reported:** % correct per context-length bucket

```bash
git clone https://github.com/hsiehjackson/RULER.git
# Follow RULER README to generate S-NIAH tasks at each target length
```

The paper scales context lengths as: `8K, 16K, 32K, 65K, 131K, 262K, 524K, 1M`
(powers of 2 from 2^13 to 2^20). At each length, there are 50 tasks.

The RLM `prompt` is the haystack (long text). The needle phrase/number is embedded at a
random position. The `root_prompt` is the retrieval question.

---

## 4. OOLONG-Pairs: All 20 Task Definitions

These are reproduced verbatim from Appendix D.1 of the paper. Each task is prepended
with: *"In the above data, ..."* where "the above data" is the OOLONG `trec_coarse`
context. The label taxonomy is always:
**{description and abstract concept, entity, human being, numeric value, location, abbreviation}**

All tasks ask for output in the format: `(user_id_1, user_id_2)` per line, lower ID first,
no duplicates.

---

**Task 1:** List all pairs of user IDs (no duplicate pairs, list lower ID first) where both
users have at least one instance with a numeric value or location. Each of the questions can
be labelled as one of the labels (the data does not provide the labels, you need to figure out
the label from the semantics of the question): description and abstract concept, entity, human
being, numeric value, location, abbreviation. In your answer, list all pairs in the format
(user_id_1, user_id_2), separated by newlines.

**Task 2:** List all pairs of user IDs (no duplicate pairs, list lower ID first) where both
users have at least one instance with an entity or human being. Each of the questions can be
labelled as one of the labels (the data does not provide the labels, you need to figure out
the label from the semantics of the question): description and abstract concept, entity, human
being, numeric value, location, abbreviation. In your answer, list all pairs in the format
(user_id_1, user_id_2), separated by newlines.

**Task 3:** List all pairs of user IDs (no duplicate pairs, list lower ID first) where both
users have at least one instance with a description and abstract concept or abbreviation. Each
of the questions can be labelled as one of the labels (the data does not provide the labels,
you need to figure out the label from the semantics of the question): description and abstract
concept, entity, human being, numeric value, location, abbreviation. In your answer, list all
pairs in the format (user_id_1, user_id_2), separated by newlines.

**Task 4:** List all pairs of user IDs (no duplicate pairs, list lower ID first) where both
users have at least one instance with a human being or location, and all instances that are a
human being for both users must be after January 6, 2023. Each of the questions can be labelled
as one of the labels (the data does not provide the labels, you need to figure out the label
from the semantics of the question): description and abstract concept, entity, human being,
numeric value, location, abbreviation. In your answer, list all pairs in the format
(user_id_1, user_id_2), separated by newlines.

**Task 5:** List all pairs of user IDs (no duplicate pairs, list lower ID first) where both
users have at least one instance with an entity or numeric value, and all instances that are an
entity for both users must be before March 15, 2023. Each of the questions can be labelled as
one of the labels (the data does not provide the labels, you need to figure out the label from
the semantics of the question): description and abstract concept, entity, human being, numeric
value, location, abbreviation. In your answer, list all pairs in the format (user_id_1,
user_id_2), separated by newlines.

**Task 6:** List all pairs of user IDs (no duplicate pairs, list lower ID first) where both
users have at least one instance with a location or abbreviation. Each of the questions can be
labelled as one of the labels (the data does not provide the labels, you need to figure out
the label from the semantics of the question): description and abstract concept, entity, human
being, numeric value, location, abbreviation. In your answer, list all pairs in the format
(user_id_1, user_id_2), separated by newlines.

**Task 7:** List all pairs of user IDs (no duplicate pairs, list lower ID first) where both
users have at least one instance with a description and abstract concept or numeric value, and
all instances that are a numeric value for both users must be after February 1, 2023. Each of
the questions can be labelled as one of the labels (the data does not provide the labels, you
need to figure out the label from the semantics of the question): description and abstract
concept, entity, human being, numeric value, location, abbreviation. In your answer, list all
pairs in the format (user_id_1, user_id_2), separated by newlines.

**Task 8:** List all pairs of user IDs (no duplicate pairs, list lower ID first) where both
users have at least one instance with a human being or description and abstract concept. Each
of the questions can be labelled as one of the labels (the data does not provide the labels,
you need to figure out the label from the semantics of the question): description and abstract
concept, entity, human being, numeric value, location, abbreviation. In your answer, list all
pairs in the format (user_id_1, user_id_2), separated by newlines.

**Task 9:** List all pairs of user IDs (no duplicate pairs, list lower ID first) where both
users have at least one instance with an entity or location, and all instances that are a
location for both users must be after April 10, 2023. Each of the questions can be labelled as
one of the labels (the data does not provide the labels, you need to figure out the label from
the semantics of the question): description and abstract concept, entity, human being, numeric
value, location, abbreviation. In your answer, list all pairs in the format (user_id_1,
user_id_2), separated by newlines.

**Task 10:** List all pairs of user IDs (no duplicate pairs, list lower ID first) where both
users have at least one instance with a numeric value or abbreviation, and all instances that
are an abbreviation for both users must be before May 20, 2023. Each of the questions can be
labelled as one of the labels (the data does not provide the labels, you need to figure out
the label from the semantics of the question): description and abstract concept, entity, human
being, numeric value, location, abbreviation. In your answer, list all pairs in the format
(user_id_1, user_id_2), separated by newlines.

**Task 11:** List all pairs of user IDs (no duplicate pairs, list lower ID first) such that
one user has at least one instance with entity and one with abbreviation, and the other user
has exactly one instance with entity. Each of the questions can be labelled as one of the
labels (the data does not provide the labels, you need to figure out the label from the
semantics of the question): description and abstract concept, entity, human being, numeric
value, location, abbreviation. In your answer, list all pairs in the format (user_id_1,
user_id_2), separated by newlines.

**Task 12:** List all pairs of user IDs (no duplicate pairs, list lower ID first) such that
one user has at least two instances with numeric value, and the other user has at least one
instance with location and at least one instance with human being. Each of the questions can
be labelled as one of the labels (the data does not provide the labels, you need to figure out
the label from the semantics of the question): description and abstract concept, entity, human
being, numeric value, location, abbreviation. In your answer, list all pairs in the format
(user_id_1, user_id_2), separated by newlines.

**Task 13:** List all pairs of user IDs (no duplicate pairs, list lower ID first) such that
one user has exactly one instance with description and abstract concept, and the other user
has at least one instance with abbreviation and at least one instance with entity. Each of the
questions can be labelled as one of the labels (the data does not provide the labels, you need
to figure out the label from the semantics of the question): description and abstract concept,
entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in
the format (user_id_1, user_id_2), separated by newlines.

**Task 14:** List all pairs of user IDs (no duplicate pairs, list lower ID first) such that
one user has at least one instance with human being and at least one instance with numeric
value, and the other user has exactly two instances with location. Each of the questions can
be labelled as one of the labels (the data does not provide the labels, you need to figure out
the label from the semantics of the question): description and abstract concept, entity, human
being, numeric value, location, abbreviation. In your answer, list all pairs in the format
(user_id_1, user_id_2), separated by newlines.

**Task 15:** List all pairs of user IDs (no duplicate pairs, list lower ID first) such that
one user has at least one instance with entity, at least one instance with location, and at
least one instance with abbreviation, and the other user has exactly one instance with numeric
value. Each of the questions can be labelled as one of the labels (the data does not provide
the labels, you need to figure out the label from the semantics of the question): description
and abstract concept, entity, human being, numeric value, location, abbreviation. In your
answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.

**Task 16:** List all pairs of user IDs (no duplicate pairs, list lower ID first) such that
one user has at least one instance with description and abstract concept and at least one
instance with human being, and the other user has at least two instances with entity and
exactly one instance with abbreviation. Each of the questions can be labelled as one of the
labels (the data does not provide the labels, you need to figure out the label from the
semantics of the question): description and abstract concept, entity, human being, numeric
value, location, abbreviation. In your answer, list all pairs in the format (user_id_1,
user_id_2), separated by newlines.

**Task 17:** List all pairs of user IDs (no duplicate pairs, list lower ID first) such that
one user has exactly one instance with numeric value, and the other user has at least one
instance with location and at least one instance with description and abstract concept. Each
of the questions can be labelled as one of the labels (the data does not provide the labels,
you need to figure out the label from the semantics of the question): description and abstract
concept, entity, human being, numeric value, location, abbreviation. In your answer, list all
pairs in the format (user_id_1, user_id_2), separated by newlines.

**Task 18:** List all pairs of user IDs (no duplicate pairs, list lower ID first) such that
one user has at least one instance with abbreviation and exactly one instance with human being,
and the other user has at least one instance with entity and at least one instance with numeric
value. Each of the questions can be labelled as one of the labels (the data does not provide
the labels, you need to figure out the label from the semantics of the question): description
and abstract concept, entity, human being, numeric value, location, abbreviation. In your
answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.

**Task 19:** List all pairs of user IDs (no duplicate pairs, list lower ID first) such that
one user has at least two instances with location and at least one instance with entity, and
the other user has exactly one instance with description and abstract concept and exactly one
instance with abbreviation. Each of the questions can be labelled as one of the labels (the
data does not provide the labels, you need to figure out the label from the semantics of the
question): description and abstract concept, entity, human being, numeric value, location,
abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated
by newlines.

**Task 20:** List all pairs of user IDs (no duplicate pairs, list lower ID first) such that
one user has at least one instance with numeric value and at least one instance with human
being, and the other user has at least one instance with location, at least one instance with
entity, and exactly one instance with abbreviation. Each of the questions can be labelled as
one of the labels (the data does not provide the labels, you need to figure out the label from
the semantics of the question): description and abstract concept, entity, human being, numeric
value, location, abbreviation. In your answer, list all pairs in the format (user_id_1,
user_id_2), separated by newlines.

---

## 5. System Prompts

### 5.1 RLM with REPL (GPT-5)

The system prompt is already implemented in `rlm/utils/prompts.py:6` as `RLM_SYSTEM_PROMPT`.
**This is the exact prompt used in the paper for GPT-5** (Appendix C.1). No modifications
needed — it is loaded automatically by `RLM.__init__` when `custom_system_prompt=None`.

Key features of this prompt (do not modify):
- Instructs the model to use `context` variable in the REPL
- Exposes `llm_query(prompt)` for single sub-LM calls (~500K char context)
- Exposes `llm_query_batched(prompts)` for parallel sub-LM calls
- Provides chunking strategy examples
- Requires `FINAL(answer)` or `FINAL_VAR(varname)` to terminate

### 5.2 Base Model

The base model uses **no system prompt**. It receives the full context + question
concatenated directly as a user message. The paper notes the base model often hits
GPT-5's ~272K token context limit on longer tasks (marked with `*` in Table 1).

### 5.3 What NOT to Use for GPT-5

The paper notes (Appendix C.1, Appendix B) that:
- The Qwen3-Coder prompt adds an extra batching warning — **do not add this for GPT-5**
- The Qwen3-8B prompt adjusts context window references to ~32K — **do not use for GPT-5**
- The existing `RLM_SYSTEM_PROMPT` in the repo is the correct GPT-5 prompt as-is

---

## 6. Writing the Evaluation Harness

None of the benchmark evaluation scripts are in the repository. You need to create them.
Suggested structure:

```
eval/
  run_oolong.py            # OOLONG evaluation
  run_oolong_pairs.py      # OOLONG-Pairs evaluation
  run_sniah.py             # S-NIAH evaluation
  score.py                 # Scoring functions
  utils.py                 # Shared data loading helpers
results/
  oolong/
  oolong_pairs/
  sniah/
logs/                      # RLMLogger output (auto-created)
```

### 6.1 Base Model Evaluation

```python
# eval/run_base_model.py
import json
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = "gpt-5"  # verify exact name on platform.openai.com

def run_base_model(context: str, question: str) -> str:
    prompt = f"{context}\n\n{question}"
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        # Add reasoning_effort="medium" if required by GPT-5 API
    )
    return response.choices[0].message.content

def evaluate_oolong(dataset, output_path: str):
    results = []
    for i, example in enumerate(dataset):
        print(f"Running task {i+1}/{len(dataset)}")
        try:
            pred = run_base_model(example["input"], example["question"])
        except Exception as e:
            pred = ""
            print(f"  Error: {e}")
        results.append({"id": i, "prediction": pred, "answer": example["answer"]})
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
```

### 6.2 RLM Evaluation

```python
# eval/run_rlm.py
import json
from rlm import RLM
from rlm.logger import RLMLogger

ROOT_MODEL = "gpt-5"      # verify exact name
SUB_MODEL  = "gpt-5-mini" # verify exact name

def build_rlm(log_dir: str) -> RLM:
    logger = RLMLogger(log_dir=log_dir)
    return RLM(
        backend="openai",
        backend_kwargs={"model_name": ROOT_MODEL},
        other_backends=["openai"],
        other_backend_kwargs=[{"model_name": SUB_MODEL}],
        environment="local",
        max_iterations=30,
        logger=logger,
        verbose=False,  # set True to see live output
    )

def evaluate_oolong_rlm(dataset, output_path: str, log_dir: str):
    results = []
    for i, example in enumerate(dataset):
        print(f"Running task {i+1}/{len(dataset)}")
        rlm = build_rlm(log_dir=f"{log_dir}/task_{i}")
        try:
            completion = rlm.completion(
                prompt=example["input"],        # context goes here
                root_prompt=example["question"] # question goes here
            )
            pred = completion.response
        except Exception as e:
            pred = ""
            print(f"  Error: {e}")
        results.append({
            "id": i,
            "prediction": pred,
            "answer": example["answer"],
            "cost": completion.usage_summary.to_dict() if pred else None,
        })
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
```

> **Note on `prompt` vs `root_prompt`:**
> `prompt` is loaded as the `context` variable in the REPL — this should be the long
> document/corpus. `root_prompt` is shown to the model as a reminder of the original
> question. For all three benchmarks, pass the long text as `prompt` and the question
> as `root_prompt`.

### 6.3 S-NIAH Evaluation Loop

```python
# eval/run_sniah.py
CONTEXT_LENGTHS = [8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]

def evaluate_sniah(tasks_by_length: dict, method: str, output_path: str):
    """
    tasks_by_length: {length: [{"haystack": str, "needle": str, "question": str, "answer": str}]}
    """
    results = {}
    for length in CONTEXT_LENGTHS:
        results[length] = []
        tasks = tasks_by_length.get(length, [])
        for i, task in enumerate(tasks):
            if method == "base":
                pred = run_base_model(task["haystack"], task["question"])
            elif method == "rlm":
                rlm = build_rlm(log_dir=f"logs/sniah/{length}/task_{i}")
                completion = rlm.completion(
                    prompt=task["haystack"],
                    root_prompt=task["question"]
                )
                pred = completion.response
            results[length].append({
                "prediction": pred,
                "answer": task["answer"],
                "correct": task["answer"].lower() in pred.lower()
            })
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
```

---

## 7. Scoring Functions

### 7.1 OOLONG Score

From the paper (§3.1): numeric answers use `score(ŷ) = 0.75^|y−ŷ|`, all others use
exact match. Report the **average score** across 50 tasks.

```python
def score_oolong(prediction: str, answer: str) -> float:
    """Score a single OOLONG prediction."""
    pred = prediction.strip()
    ans = str(answer).strip()

    # Try numeric scoring first
    try:
        y_hat = float(pred.replace(",", ""))
        y = float(ans.replace(",", ""))
        return 0.75 ** abs(y - y_hat)
    except ValueError:
        pass

    # Exact match (case-insensitive)
    return 1.0 if pred.lower() == ans.lower() else 0.0

def evaluate_oolong_results(results: list) -> float:
    scores = [score_oolong(r["prediction"], r["answer"]) for r in results]
    return sum(scores) / len(scores) * 100  # report as percentage
```

### 7.2 OOLONG-Pairs F1 Score

From the paper (§3.1): report **F1 score** over predicted vs. gold pairs.

```python
def parse_pairs(text: str) -> set[tuple[int, int]]:
    """Parse (user_id_1, user_id_2) pairs from model output."""
    import re
    pairs = set()
    for match in re.finditer(r'\((\d+),\s*(\d+)\)', text):
        a, b = int(match.group(1)), int(match.group(2))
        pairs.add((min(a, b), max(a, b)))  # always lower ID first
    return pairs

def f1_pairs(predicted: str, gold: str) -> float:
    pred_set = parse_pairs(predicted)
    gold_set = parse_pairs(gold)
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def evaluate_oolong_pairs_results(results: list) -> float:
    scores = [f1_pairs(r["prediction"], r["answer"]) for r in results]
    return sum(scores) / len(scores) * 100  # report as percentage
```

### 7.3 S-NIAH Score

Exact match: the needle phrase/number must appear in the prediction.

```python
def score_sniah(prediction: str, answer: str) -> float:
    return 1.0 if answer.strip().lower() in prediction.strip().lower() else 0.0

def evaluate_sniah_results(results_by_length: dict) -> dict:
    """Returns % correct per context length."""
    scores = {}
    for length, results in results_by_length.items():
        correct = sum(score_sniah(r["prediction"], r["answer"]) for r in results)
        scores[length] = correct / len(results) * 100
    return scores
```

---

## 8. Running the Experiments

### 8.1 Recommended Execution Order

Run cheapest benchmarks first to validate the pipeline before committing to expensive ones:

```
1. OOLONG-Pairs  (~$10 total, fastest)
2. OOLONG        (~$29 total)
3. S-NIAH        (~$80 total, most runs)
```

### 8.2 OOLONG

```bash
# Base Model
python eval/run_base_model.py \
  --benchmark oolong \
  --output results/oolong/base_model.json

# RLM
python eval/run_rlm.py \
  --benchmark oolong \
  --output results/oolong/rlm.json \
  --log-dir logs/oolong/rlm
```

Expected scores (from paper Table 1, GPT-5):
- Base Model: **44.0**
- RLM: **56.5**

### 8.3 OOLONG-Pairs

```bash
# Base Model
python eval/run_base_model.py \
  --benchmark oolong_pairs \
  --output results/oolong_pairs/base_model.json

# RLM
python eval/run_rlm.py \
  --benchmark oolong_pairs \
  --output results/oolong_pairs/rlm.json \
  --log-dir logs/oolong_pairs/rlm
```

Expected scores (from paper Table 1, GPT-5):
- Base Model: **0.1** (F1)
- RLM: **58.0** (F1)

> The large gap here is the most dramatic result in the paper — OOLONG-Pairs requires
> quadratic reasoning over all pairs, which entirely breaks vanilla base model calls.

### 8.4 S-NIAH (Figure 1)

This requires 8 separate runs per method (one per context length), each over 50 tasks.

```bash
for LENGTH in 8192 16384 32768 65536 131072 262144 524288 1048576; do
  # Base Model (will fail/truncate at 524K+ due to GPT-5's ~272K context limit)
  python eval/run_sniah.py \
    --length $LENGTH \
    --method base \
    --output results/sniah/base_${LENGTH}.json

  # RLM
  python eval/run_sniah.py \
    --length $LENGTH \
    --method rlm \
    --output results/sniah/rlm_${LENGTH}.json \
    --log-dir logs/sniah/rlm_${LENGTH}
done
```

Expected pattern (from paper Figure 1):
- Base Model: strong at short lengths, degrades sharply — flat after 2^14
- RLM: maintains performance all the way to 1M tokens

### 8.5 Parallelization

The RLM codebase runs all LM calls **synchronously/blocking** (noted as a known
limitation in Appendix B). To speed up, run multiple tasks in parallel at the
process level using Python `multiprocessing` or `concurrent.futures.ProcessPoolExecutor`.
Do not use threading — each `RLM` instance spawns its own TCP socket server on a
unique port.

```python
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(run_single_task, task, i) for i, task in enumerate(dataset)]
    results = [f.result() for f in futures]
```

---

## 9. Cost Estimate

All costs are derived from per-task averages reported in Table 1 (GPT-5 section).
S-NIAH costs are estimated (not directly in Table 1).

| Benchmark | Method | Tasks | Cost/task | **Total** |
|-----------|--------|-------|-----------|-----------|
| OOLONG | Base Model | 50 | $0.14 | $7 |
| OOLONG | RLM | 50 | $0.43 | $22 |
| OOLONG-Pairs | Base Model | 20 | $0.16 | $3 |
| OOLONG-Pairs | RLM | 20 | $0.33 | $7 |
| S-NIAH | Base Model | 400 (50×8) | ~$0.05 avg | ~$21 |
| S-NIAH | RLM | 400 (50×8) | ~$0.15 avg | ~$60 |
| **Total** | | | | **~$120** |

Add ~25% buffer for retries/failures: **budget ~$150**

**Cost warnings:**
- RLM standard deviations are large. A single expensive OOLONG-Pairs RLM trajectory
  can reach $1–2 (mean is $0.33, but σ=$0.20). Budget for outliers.
- S-NIAH at 524K and 1M token lengths will be the most expensive RLM runs.
- The base model at 524K+ tokens will hit GPT-5's context window limit and fail fast
  (near-zero cost for those runs).

---

## 10. Known Pitfalls

From Appendix B of the paper (*Negative Results*) — things that did not work:

1. **FINAL/FINAL_VAR confusion:** The model sometimes outputs `FINAL(variable_in_repl)`
   instead of using `FINAL_VAR(variable_in_repl)`, or calls `FINAL_VAR` before assigning
   the variable. The RLM system prompt has safeguards for this, but monitor your logs.

2. **Max iterations hit:** If the RLM hits `max_iterations=30` without finding a final
   answer, `_default_answer()` is called as a fallback. This is a valid response but may
   be lower quality. Consider increasing to `max_iterations=50` for complex tasks.

3. **Too many sub-calls:** Without the Qwen3-specific batching warning, GPT-5 is
   generally conservative with sub-calls and should behave well. Monitor costs on
   OOLONG-Pairs specifically, as the quadratic complexity can trigger many sub-calls.

4. **S-NIAH at long contexts:** The base model simply cannot process 524K+ token inputs
   within GPT-5's context window (~272K tokens). Those runs will either be truncated
   by the API or return an error. This is expected and matches the red region in Figure 1.

5. **REPL security:** The default `LocalREPL` (`environment="local"`) runs generated
   Python code in the same process via `exec()`. Only run with trusted/benchmark inputs,
   not user-provided content.

6. **Socket port conflicts:** Each `RLM.completion()` call starts a TCP server on a
   random available port. Running many parallel processes on the same machine is fine,
   but ensure you are not hitting OS-level limits on open sockets.

7. **Reproducibility:** The paper uses GPT-5 with "medium reasoning" sampling. Results
   may vary slightly across runs due to model stochasticity. The paper reports averages
   over the full benchmark split (50 tasks for OOLONG, 20 for OOLONG-Pairs) — variance
   across tasks is captured in the ± std shown in Table 1.

---

## Quick Reference

| Item | Value |
|------|-------|
| Repo | https://github.com/alexzhang13/rlm |
| Paper | arXiv:2512.24601 |
| OOLONG paper | arXiv:2511.02817 |
| RULER/S-NIAH paper | arXiv:2404.06654 |
| RULER GitHub | https://github.com/hsiehjackson/RULER |
| Root model | `gpt-5` (verify on platform.openai.com) |
| Sub-call model | `gpt-5-mini` (verify on platform.openai.com) |
| Required env var | `OPENAI_API_KEY` |
| RLM system prompt | `rlm/utils/prompts.py:6` — use as-is for GPT-5 |
| OOLONG split | `trec_coarse`, 50 tasks, 131K tokens |
| OOLONG-Pairs | 20 tasks, lengths up to 1M tokens |
| S-NIAH | 50 tasks × 8 context lengths (8K–1M) |
| OOLONG scoring | `0.75^|y-ŷ|` numeric / exact match otherwise |
| OOLONG-Pairs scoring | F1 over predicted vs. gold pairs |
| S-NIAH scoring | % exact match |
| Estimated total cost | ~$120 (budget $150) |
