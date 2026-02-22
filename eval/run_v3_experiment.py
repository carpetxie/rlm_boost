"""Run v3 RLM pipeline experiment with explicit code template in prompt."""
import os, json, sys, time
sys.path.insert(0, '.')
os.environ['OPENAI_API_KEY'] = open('.env').read().split('=',1)[1].strip()

from rlm.core.rlm import RLM
from rlm.utils.prompts import INCREMENTAL_SYSTEM_PROMPT
from eval.rlm_pipeline_experiment import (
    load_oolong_data, split_context_by_users, compute_gold_pairs,
    compute_f1, measure_execution_compliance, TASK_1_CHECKER_SETUP,
)

CHUNK_ROOT_PROMPT_V3 = """Task (OOLONG-Pairs Task 1): Find pairs of users BOTH with >= 1 instance.
Chunk {chunk_num} of {total_chunks}.

Context format: each line is "Date: [date] || User: [user_id] || Instance: [text] || Label: [cat]"
Entity ID = the numeric user_id (e.g. "44436"), NOT the date.

Run this code (using the actual chunk index {chunk_idx}):

```repl
import re
entities = {{}}
for line in context_{chunk_idx}.split('\\n'):
    m = re.search(r'User: (\\d+)', line)
    if m:
        uid = m.group(1)
        if uid not in entities:
            entities[uid] = {{"instances": []}}
        entities[uid]["instances"].append(line.strip())
stats = _incremental.process_chunk({chunk_idx}, entities, pair_checker=check_pair)
pair_results = list(_incremental.pair_tracker.get_pairs())
print(f"Chunk {chunk_num}: {{stats['new_entities']}} new entities, {{stats['total_pairs']}} pairs")
```

After the repl block runs successfully, return FINAL_VAR(pair_results).
Do NOT re-read context_0, context_1 etc. on this turn.
"""

print('Loading data...')
plain_context, labeled_context = load_oolong_data()
num_chunks = 3
max_chars = 5000
chunks = split_context_by_users(plain_context, num_chunks)
chunks = [c[:max_chars] for c in chunks]

gold_pairs = compute_gold_pairs(labeled_context, 1)
print(f'Gold pairs: {len(gold_pairs)}')

rlm = RLM(
    backend='openai',
    backend_kwargs={'model_name': 'gpt-4o-mini'},
    environment='local',
    environment_kwargs={'setup_code': TASK_1_CHECKER_SETUP},
    persistent=True,
    custom_system_prompt=INCREMENTAL_SYSTEM_PROMPT,
    max_iterations=6,
    verbose=False,
)

turn_results = []
prev_cp = 0
for chunk_i, chunk in enumerate(chunks):
    root_prompt = CHUNK_ROOT_PROMPT_V3.format(
        chunk_num=chunk_i+1, total_chunks=num_chunks, chunk_idx=chunk_i
    )
    print(f'\n--- Turn {chunk_i+1}/{num_chunks} ---')
    t0 = time.perf_counter()
    completion = rlm.completion(chunk, root_prompt=root_prompt)
    elapsed = time.perf_counter() - t0
    print(f'  Response: {str(completion.response)[:300]}')

    compliance = measure_execution_compliance(rlm._persistent_env, chunk_i, prev_cp)
    prev_cp = compliance.get('chunks_processed', prev_cp)
    print(f'  Compliant: {compliance["compliant"]}  chunks_processed: {compliance["chunks_processed"]}  pairs: {compliance["pair_results_count"]}')
    if compliance.get('error'):
        print(f'  ERROR: {compliance["error"]}')
    turn_results.append({'chunk_i': chunk_i, 'compliance': compliance, 'elapsed': round(elapsed, 2)})

# Check _incremental.pair_tracker directly
env = rlm._persistent_env
print('\n=== REPL State after all turns ===')
incr = env.locals.get('_incremental')
direct_pairs = []
entity_ids = []
stats = {'chunks_processed': 0, 'total_pairs': 0, 'total_entities': 0}
if incr:
    stats = incr.get_stats()
    print(f'chunks_processed: {stats["chunks_processed"]}')
    print(f'total_pairs: {stats["total_pairs"]}')
    print(f'total_entities: {stats["total_entities"]}')
    direct_pairs = list(incr.pair_tracker.get_pairs())
    entity_ids = list(incr.entity_cache.get_ids())
    print(f'Direct pair_tracker pairs: {len(direct_pairs)}')
    print(f'Entity IDs sample: {entity_ids[:10]}')
    if direct_pairs:
        sample = list(direct_pairs)[:5]
        print(f'Sample direct pairs: {sample}')

# F1 from direct pair_tracker
f1_direct = compute_f1(direct_pairs, gold_pairs)
print(f'\nF1 from pair_tracker: F1={f1_direct["f1"]}  P={f1_direct["precision"]}  R={f1_direct["recall"]}')
print(f'  Predicted={f1_direct.get("predicted_pairs")}  Gold={f1_direct.get("gold_pairs")}  TP={f1_direct.get("tp")}')

# F1 from pair_results var
pr = env.locals.get('pair_results')
f1_var = compute_f1(pr, gold_pairs)
print(f'F1 from pair_results var: F1={f1_var["f1"]}  P={f1_var["precision"]}  R={f1_var["recall"]}')

compliant = sum(1 for t in turn_results if t['compliance']['compliant'])
print(f'\nExecution compliance: {compliant}/{len(turn_results)} = {compliant/len(turn_results):.0%}')

rlm.close()

result = {
    'version': 'v3_explicit_code_template',
    'model': 'gpt-4o-mini', 'max_chunk_chars': max_chars,
    'execution_compliance_rate': compliant/len(turn_results),
    'f1_from_tracker': f1_direct,
    'f1_from_pair_results_var': f1_var,
    'chunks_processed': stats['chunks_processed'],
    'total_entities': stats['total_entities'],
    'total_pairs_in_tracker': stats['total_pairs'],
    'entity_id_sample': entity_ids[:10],
    'turn_results': [{'chunk_i': t['chunk_i'],
                      'compliant': t['compliance']['compliant'],
                      'chunks_processed': t['compliance']['chunks_processed'],
                      'pair_results_count': t['compliance']['pair_results_count'],
                      'error': t['compliance'].get('error')} for t in turn_results],
}
import pathlib
pathlib.Path('results/streaming').mkdir(parents=True, exist_ok=True)
with open('results/streaming/rlm_pipeline_v3_results.json', 'w') as f:
    json.dump(result, f, indent=2, default=str)
print('\nSaved to results/streaming/rlm_pipeline_v3_results.json')
