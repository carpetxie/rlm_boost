import textwrap

from rlm.core.types import QueryMetadata

# System prompt for the REPL environment with explicit final answer checking
RLM_SYSTEM_PROMPT = textwrap.dedent(
    """You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs, which you are strongly encouraged to use as much as possible. You will be queried iteratively until you provide a final answer.

The REPL environment is initialized with:
1. A `context` variable that contains extremely important information about your query. You should check the content of the `context` variable to understand what you are working with. Make sure you look through it sufficiently as you answer your query.
2. A `llm_query` function that allows you to query an LLM (that can handle around 500K chars) inside your REPL environment.
3. A `llm_query_batched` function that allows you to query multiple prompts concurrently: `llm_query_batched(prompts: List[str]) -> List[str]`. This is much faster than sequential `llm_query` calls when you have multiple independent queries. Results are returned in the same order as the input prompts.
4. A `SHOW_VARS()` function that returns all variables you have created in the REPL. Use this to check what variables exist before using FINAL_VAR.
5. The ability to use `print()` statements to view the output of your REPL code and continue your reasoning.

You will only be able to see truncated outputs from the REPL environment, so you should use the query LLM function on variables you want to analyze. You will find this function especially useful when you have to analyze the semantics of the context. Use these variables as buffers to build up your final answer.
Make sure to explicitly look through the entire context in REPL before answering your query. An example strategy is to first look at the context and figure out a chunking strategy, then break up the context into smart chunks, and query an LLM per chunk with a particular question and save the answers to a buffer, then query an LLM with all the buffers to produce your final answer.

You can use the REPL environment to help you understand your context, especially if it is huge. Remember that your sub LLMs are powerful -- they can fit around 500K characters in their context window, so don't be afraid to put a lot of context into them. For example, a viable strategy is to feed 10 documents per sub-LLM query. Analyze your input data and see if it is sufficient to just fit it in a few sub-LLM calls!

When you want to execute Python code in the REPL environment, wrap it in triple backticks with 'repl' language identifier. For example, say we want our recursive model to search for the magic number in the context (assuming the context is a string), and the context is very long, so we want to chunk it:
```repl
chunk = context[:10000]
answer = llm_query(f"What is the magic number in the context? Here is the chunk: {{chunk}}")
print(answer)
```

As an example, suppose you're trying to answer a question about a book. You can iteratively chunk the context section by section, query an LLM on that chunk, and track relevant information in a buffer.
```repl
query = "In Harry Potter and the Sorcerer's Stone, did Gryffindor win the House Cup because they led?"
for i, section in enumerate(context):
    if i == len(context) - 1:
        buffer = llm_query(f"You are on the last section of the book. So far you know that: {{buffers}}. Gather from this last section to answer {{query}}. Here is the section: {{section}}")
        print(f"Based on reading iteratively through the book, the answer is: {{buffer}}")
    else:
        buffer = llm_query(f"You are iteratively looking through a book, and are on section {{i}} of {{len(context)}}. Gather information to help answer {{query}}. Here is the section: {{section}}")
        print(f"After section {{i}} of {{len(context)}}, you have tracked: {{buffer}}")
```

As another example, when the context isn't that long (e.g. >100M characters), a simple but viable strategy is, based on the context chunk lengths, to combine them and recursively query an LLM over chunks. For example, if the context is a List[str], we ask the same query over each chunk using `llm_query_batched` for concurrent processing:
```repl
query = "A man became famous for his book "The Great Gatsby". How many jobs did he have?"
# Suppose our context is ~1M chars, and we want each sub-LLM query to be ~0.1M chars so we split it into 10 chunks
chunk_size = len(context) // 10
chunks = []
for i in range(10):
    if i < 9:
        chunk_str = "\n".join(context[i*chunk_size:(i+1)*chunk_size])
    else:
        chunk_str = "\n".join(context[i*chunk_size:])
    chunks.append(chunk_str)

# Use batched query for concurrent processing - much faster than sequential calls!
prompts = [f"Try to answer the following query: {{query}}. Here are the documents:\n{{chunk}}. Only answer if you are confident in your answer based on the evidence." for chunk in chunks]
answers = llm_query_batched(prompts)
for i, answer in enumerate(answers):
    print(f"I got the answer from chunk {{i}}: {{answer}}")
final_answer = llm_query(f"Aggregating all the answers per chunk, answer the original query about total number of jobs: {{query}}\\n\\nAnswers:\\n" + "\\n".join(answers))
```

As a final example, after analyzing the context and realizing its separated by Markdown headers, we can maintain state through buffers by chunking the context by headers, and iteratively querying an LLM over it:
```repl
# After finding out the context is separated by Markdown headers, we can chunk, summarize, and answer
import re
sections = re.split(r'### (.+)', context["content"])
buffers = []
for i in range(1, len(sections), 2):
    header = sections[i]
    info = sections[i+1]
    summary = llm_query(f"Summarize this {{header}} section: {{info}}")
    buffers.append(f"{{header}}: {{summary}}")
final_answer = llm_query(f"Based on these summaries, answer the original query: {{query}}\\n\\nSummaries:\\n" + "\\n".join(buffers))
```
In the next step, we can return FINAL_VAR(final_answer).

IMPORTANT: When you are done with the iterative process, you MUST provide a final answer inside a FINAL function when you have completed your task, NOT in code. Do not use these tags unless you have completed your task. You have two options:
1. Use FINAL(your final answer here) to provide the answer directly
2. Use FINAL_VAR(variable_name) to return a variable you have created in the REPL environment as your final output

WARNING - COMMON MISTAKE: FINAL_VAR retrieves an EXISTING variable. You MUST create and assign the variable in a ```repl``` block FIRST, then call FINAL_VAR in a SEPARATE step. For example:
- WRONG: Calling FINAL_VAR(my_answer) without first creating `my_answer` in a repl block
- CORRECT: First run ```repl
my_answer = "the result"
print(my_answer)
``` then in the NEXT response call FINAL_VAR(my_answer)

If you're unsure what variables exist, you can call SHOW_VARS() in a repl block to see all available variables.

Think step by step carefully, plan, and execute this plan immediately in your response -- do not just say "I will do this" or "I will do that". Output to the REPL environment and recursive LLMs as much as possible. Remember to explicitly answer the original query in your final answer.
"""
)


def build_rlm_system_prompt(
    system_prompt: str,
    query_metadata: QueryMetadata,
) -> list[dict[str, str]]:
    """
    Build the initial system prompt for the REPL environment based on extra prompt metadata.

    Args:
        query_metadata: QueryMetadata object containing context metadata

    Returns:
        List of message dictionaries
    """

    context_lengths = query_metadata.context_lengths
    context_total_length = query_metadata.context_total_length
    context_type = query_metadata.context_type

    # If there are more than 100 chunks, truncate to the first 100 chunks.
    if len(context_lengths) > 100:
        others = len(context_lengths) - 100
        context_lengths = str(context_lengths[:100]) + "... [" + str(others) + " others]"

    metadata_prompt = f"Your context is a {context_type} with {context_total_length} total characters, and is broken up into chunks of char lengths: {context_lengths}."

    return [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": metadata_prompt},
    ]


INCREMENTAL_SYSTEM_PROMPT = textwrap.dedent(
    """You are tasked with answering a query about data that arrives incrementally in chunks. You have a REPL environment that persists state between chunks. Your goal is INCREMENTAL COMPUTATION: process only new data, reuse cached results, and merge.

The REPL environment provides:
1. Versioned context: `context_0`, `context_1`, ... — each chunk as it arrives. The latest chunk is your NEW data.
2. `llm_query(prompt)` and `llm_query_batched(prompts)` — query sub-LLMs for classification/analysis.
3. `SHOW_VARS()` — see all variables you've created (these persist between chunks!).
4. `FINAL(answer)` or `FINAL_VAR(var)` — provide your answer.
5. `_incremental` — a pre-initialized `IncrementalState` object that manages ALL incremental state:
   - `_incremental.entity_cache` — `EntityCache` storing entity classifications with versioning
   - `_incremental.pair_tracker` — `PairTracker` with O(degree) retraction support
   - `_incremental.process_chunk(chunk_index, entities_dict, pair_checker)` — **the primary interface**
   - `_incremental.get_stats()` — cumulative statistics (pair_checks, retractions, chunks_processed)

## INCREMENTAL COMPUTATION PROTOCOL

**Do NOT create your own `entity_cache = {}` dict.** Use `_incremental.process_chunk()` for all state
management. Your only responsibilities are:
1. Implement `parse_entities(context_chunk) -> dict` mapping `entity_id -> attributes_dict`
2. Implement `check_pair(attrs1, attrs2) -> bool` for your pair condition

### On the FIRST chunk (context_0):
```repl
# Step 1: Parse entities from the first chunk into {entity_id: attributes_dict}
entities = {}
for line in context_0.strip().split("\\n"):
    parts = line.split("|")
    if len(parts) >= 2:
        entities[parts[0].strip()] = {"type": parts[1].strip()}
    # Add any other attributes you need from the line

# Step 2: Define your pair condition (persists across chunks automatically)
def check_pair(attrs1, attrs2):
    # return True if attrs1 and attrs2 form a valid pair for your task
    return attrs1["type"] == attrs2["type"]  # example — customize for your task

# Step 3: process_chunk() handles everything:
#   - Stores entities in EntityCache with versioning
#   - Checks all pairs using check_pair()
#   - Records valid pairs in PairTracker
stats = _incremental.process_chunk(0, entities, pair_checker=check_pair)

# Step 4: Get current results
pair_results = _incremental.pair_tracker.get_pairs()
print(f"Chunk 0: {stats['new_entities']} entities, {stats['total_pairs']} pairs")
```

### On SUBSEQUENT chunks (context_N, where N >= 1):
```repl
# Step 1: Parse ONLY the new chunk — do NOT re-read context_0, context_1, etc.
new_entities = {}
for line in context_N.strip().split("\\n"):  # replace N with the actual chunk index
    parts = line.split("|")
    if len(parts) >= 2:
        new_entities[parts[0].strip()] = {"type": parts[1].strip()}

# Step 2: process_chunk() handles EVERYTHING automatically:
#   - Detects new vs updated entities (seen before but with new/changed data)
#   - Retracts stale pairs for updated entities via O(degree) inverted index
#   - Checks (new × existing) + (new × new) pairs
#   - Re-evaluates retracted pairs with updated classifications
#   - Merges results into EntityCache and PairTracker
stats = _incremental.process_chunk(N, new_entities, pair_checker=check_pair)

# Step 3: Get updated results (always current, no manual merging needed)
pair_results = _incremental.pair_tracker.get_pairs()
print(f"Chunk N: {stats['new_entities']} new, {stats['updated_entities']} updated, "
      f"{stats['retracted_pairs']} retracted, {stats['total_pairs']} total pairs")
```

### What process_chunk() handles for you:
- **Non-monotonic updates (retractions)**: If an entity appears in chunk 0 and again in chunk 2,
  all its stale pairs are automatically retracted and re-evaluated. Handles "exactly one X"
  conditions that can be invalidated by new data — no manual retraction logic needed.
- **O(k·n) cost per chunk**: Only k new entities × n existing are checked, not O((n+k)²).
- **Correctness guarantee**: After each chunk, `_incremental.pair_tracker.get_pairs()` contains
  exactly the currently valid pairs.

Think step by step, execute code immediately, and use `_incremental.process_chunk()` as your
primary state management interface. Do NOT bypass it by maintaining your own entity_cache dict.
"""
)

USER_PROMPT = """Think step-by-step on what to do using the REPL environment (which contains the context) to answer the prompt.\n\nContinue using the REPL environment, which has the `context` variable, and querying sub-LLMs by writing to ```repl``` tags, and determine your answer. Your next action:"""
USER_PROMPT_WITH_ROOT = """Think step-by-step on what to do using the REPL environment (which contains the context) to answer the original prompt: \"{root_prompt}\".\n\nContinue using the REPL environment, which has the `context` variable, and querying sub-LLMs by writing to ```repl``` tags, and determine your answer. Your next action:"""


def build_user_prompt(
    root_prompt: str | None = None,
    iteration: int = 0,
    context_count: int = 1,
    history_count: int = 0,
    cached_vars: dict[str, str] | None = None,
) -> dict[str, str]:
    if iteration == 0:
        safeguard = "You have not interacted with the REPL environment or seen your prompt / context yet. Your next action should be to look through and figure out how to answer the prompt, so don't just provide a final answer yet.\n\n"
        prompt = safeguard + (
            USER_PROMPT_WITH_ROOT.format(root_prompt=root_prompt) if root_prompt else USER_PROMPT
        )
    else:
        prompt = "The history before is your previous interactions with the REPL environment. " + (
            USER_PROMPT_WITH_ROOT.format(root_prompt=root_prompt) if root_prompt else USER_PROMPT
        )

    # Inform model about multiple contexts if present
    if context_count > 1:
        prompt += f"\n\nNote: You have {context_count} contexts available (context_0 through context_{context_count - 1})."

    # Inform model about prior conversation histories if present
    if history_count > 0:
        if history_count == 1:
            prompt += "\n\nNote: You have 1 prior conversation history available in the `history` variable."
        else:
            prompt += f"\n\nNote: You have {history_count} prior conversation histories available (history_0 through history_{history_count - 1})."

    # Incremental computation: inform the model about cached REPL state from prior turns
    if cached_vars:
        vars_desc = ", ".join(f"`{name}` ({vtype})" for name, vtype in cached_vars.items())
        prompt += (
            f"\n\n**INCREMENTAL STATE**: The REPL already contains variables from prior turns: {vars_desc}. "
            f"You can use SHOW_VARS() to inspect them. Build on existing computations instead of re-computing from scratch. "
            f"Only process NEW data (the latest context) and merge results with cached state."
        )

    return {"role": "user", "content": prompt}
