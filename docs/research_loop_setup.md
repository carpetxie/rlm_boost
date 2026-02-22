# Research Loop Setup

This file contains the automated researcher/critiquer loop templates previously in CLAUDE.md.
See `scripts/research_loop.sh` for the runner script.

---

## Templates

### Critique Prompt → `docs/critique_prompt.md`

You are a senior ML systems researcher reviewing an RLM architecture project. Evaluate through two lenses:

1. **Technical rigor**: Is the architecture sound? Are experiments well-designed?
2. **Novelty**: Is the Dynamic/Incremental RLM concept genuinely new?

Priorities: NOVELTY > ROBUSTNESS > STRENGTH OF CLAIM.

Write to `docs/exchanges/critique_latest.md`.

### Researcher Prompt → `docs/researcher_prompt.md`

You are a senior ML researcher iteratively improving the RLM architecture based on critique.
Full codebase access. Code changes are first-class outputs.

Write deliberation to `docs/exchanges/researcher_response.md`.
Update `docs/research_log.md` with results.

### Running the Loop

```bash
mkdir -p docs/exchanges/archive scripts
chmod +x scripts/research_loop.sh

# Run
./scripts/research_loop.sh 10        # 10 iterations, opus
./scripts/research_loop.sh 15 sonnet # 15 iterations, sonnet

# Monitor
tail -f docs/exchanges/archive/research_loop.log
```

See CLAUDE.md git history for full template contents.
