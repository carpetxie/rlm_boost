# CLAUDE.md

## Project Overview

RLM (Recursive Language Models) is an inference engine that lets LLMs programmatically decompose problems by interacting with a Python REPL and recursively calling themselves. Replace `llm.completion()` with `rlm.completion()` and the model can write/execute code, launch sub-LM calls, and reason through massive contexts.

- **Origin**: MIT OASYS Lab (Alex Zhang, Tim Kraska, Omar Khattab)
- **Paper**: https://arxiv.org/abs/2512.24601
- **License**: MIT

## Quick Reference

```bash
# Setup
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e .                # base install
uv pip install -e ".[modal]"       # + Modal sandboxes
uv pip install -e ".[prime]"       # + Prime Intellect sandboxes
uv pip install -e ".[daytona]"     # + Daytona sandboxes

# Development
make install-dev                   # install dev + test deps
make lint                          # ruff check .
make format                        # ruff format .
make test                          # pytest
make check                         # lint + format + test

# Pre-commit (runs ruff + ty)
uv run pre-commit install
uv run pre-commit run --all-files

# Examples
make quickstart                    # needs OPENAI_API_KEY
make docker-repl                   # needs Docker
make modal-repl                    # needs Modal
```

## Architecture

```
rlm/
├── core/           # RLM engine
│   ├── rlm.py          # Main RLM class — entry point for users
│   ├── lm_handler.py   # Multi-threaded TCP socket server for sub-LM routing
│   ├── types.py         # RLMIteration, REPLResult, UsageSummary, etc.
│   └── comms_utils.py   # Length-prefixed JSON socket protocol
├── clients/        # LM provider backends (OpenAI, Anthropic, Gemini, Portkey, etc.)
│   ├── base_lm.py      # Abstract base — all clients inherit from BaseLM
│   └── __init__.py      # get_client() router
├── environments/   # REPL sandbox implementations
│   ├── base_env.py      # NonIsolatedEnv (local/Docker) vs IsolatedEnv (cloud)
│   ├── local_repl.py    # Default: Python exec() with safe builtins
│   ├── docker_repl.py   # Docker containers
│   ├── modal_repl.py    # Modal Sandboxes (cloud, HTTP broker pattern)
│   ├── prime_repl.py    # Prime Intellect Sandboxes
│   └── daytona_repl.py  # Daytona sandboxes
├── logger/         # RLMLogger (.jsonl trajectories) + VerbosePrinter (rich output)
└── utils/          # Prompts, parsing (code extraction, FINAL answer), utilities
```

### Execution Flow

1. `RLM.completion(prompt, root_prompt)` spawns LMHandler + Environment
2. System prompt + context loaded into REPL
3. Loop: call root LM → extract code blocks → execute in REPL → collect sub-LM results
4. Terminates on `FINAL(answer)` or `FINAL_VAR(var_name)`
5. Returns result with full usage summary

### Communication

- **Non-isolated** (LocalREPL, DockerREPL): Direct TCP socket to LMHandler. Protocol: 4-byte big-endian length prefix + UTF-8 JSON.
- **Isolated** (Modal, Prime, Daytona): HTTP broker inside sandbox. Host polls `/pending`, forwards to LMHandler, POSTs response to `/respond`.

## Code Conventions

- **Formatter/Linter**: ruff (line-length=100, target=py311). All PRs must pass `ruff check --fix . && ruff format .`
- **Type checker**: ty (non-blocking, `--exit-zero`)
- **Naming**: `snake_case` methods/vars, `PascalCase` classes, `UPPER_CASE` constants
- **No `_` prefix** for private methods unless explicitly needed
- **Error handling**: Fail fast, fail loud. No silent fallbacks. Missing API key → immediate `ValueError`.
- **Minimize branching**: Every `if`/`try` needs justification
- **Dependencies**: Avoid new core deps. Use optional extras for non-essential features.
- **Scope**: Small focused diffs. One change per PR. Delete dead code.

## Testing

Tests live in `tests/`. Run with `uv run pytest` or `make test`.

- CI runs pytest on Python 3.11 and 3.12
- CI ignores Modal tests and client integration tests
- Use `tests/mock_lm.py` for mock LM in unit tests
- Write deterministic unit tests; mock external services for isolated environments

## Benchmarks

```bash
# Baseline (single LM call)
python eval/run_base_model.py --benchmark oolong --output results/oolong/base.json

# RLM evaluation
python eval/run_rlm.py --benchmark oolong --output results/oolong/rlm.json --log-dir logs/oolong/rlm

# S-NIAH (needle-in-haystack)
python eval/run_sniah.py --output results/sniah.json
```

Benchmarks: OOLONG (50 long-context QA, ~131K tokens), OOLONG-Pairs (20 diverse tasks, ~32K tokens), S-NIAH.

## Research Context

This repo has two research thrusts:

1. **Thrust 1 — Fine-tuning exploration**: Tuning open models (methodology, model selection)
2. **Thrust 2 — Dynamic/Incremental RLM** (novel contribution): Architect RLMs that don't re-read the whole context every turn. Stateful Python objects, delta-based updates, incremental computation (O(1)/O(n) vs O(n²)).

**Key insight — the dynamic metrics gap**: Current benchmarks (OOLONG, S-NIAH) only test static context. But the thesis is about dynamic context built up over turns. No existing benchmark measures this. Exploring dynamic benchmarks is both a measurement need and a potential paper contribution.

Research loop tooling: see `docs/research_loop_setup.md`.

## Key Files

| File | Purpose |
|------|---------|
| `rlm/core/rlm.py` | Main RLM class (~400 lines) |
| `rlm/core/lm_handler.py` | Socket server routing sub-LM calls |
| `rlm/environments/base_env.py` | Environment abstract base classes |
| `rlm/clients/base_lm.py` | Client abstract base class |
| `rlm/utils/prompts.py` | System prompt templates |
| `rlm/utils/parsing.py` | Code block extraction, FINAL answer parsing |
| `eval/run_rlm.py` | Main evaluation harness |
| `AGENTS.md` | Full contribution guide (client/environment development) |
| `CONTRIBUTING.md` | TODOs and project priorities |
