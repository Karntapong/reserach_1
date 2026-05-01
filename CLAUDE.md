# AlphaEvolve × Bayesian Experimental Design

## Objective
Research and prototype a hybrid system that accelerates AlphaEvolve prompt convergence for a **market conduct** use case by replacing random mutation selection with **Bayesian Experimental Design (BED)**.

## Hypothesis
The paper arxiv 2510.20886 shows BED with Sequential Monte Carlo (SMC) reduces evaluations needed by 31–45% in information-seeking tasks. Since AlphaEvolve's fitness function is binary (true/false), each evaluation is a Bernoulli observation — a natural fit for BED's EIG-maximizing acquisition strategy.

## Architecture

```
AlphaEvolve (evolutionary loop)
    ↓ mutation candidates from LLM
BED Selector (this project)
    ↓ picks highest-EIG candidate
Fitness Evaluator (binary true/false + LLM artifacts)
    ↓ observation
SMC Particle Update
    ↓ updated belief
Bayes-D: exploit vs. explore decision
```

## Key Components

| File | Role |
|------|------|
| `src/bed_selector.py` | EIG calculation, SMC particle updates, Bayes-Q/M/D logic |
| `src/alphaevolve_hybrid.py` | Hybrid evolutionary loop integrating BED selector |
| `experiments/baseline_run.py` | Pure evolutionary AlphaEvolve baseline |
| `experiments/bed_run.py` | BED-guided experiment |
| `research/bed_alphaevolve_design.md` | Theoretical mapping and design decisions |

## Paper Reference
- **arxiv 2510.20886** — Bayesian Experimental Design for agentic information-seeking
- Notebook: NotebookLM ID `50f203d4-a09c-4a1a-9ad3-f449e67d3d11`
- Key mechanism: Bayes-Q (EIG), Bayes-M (MAP action), Bayes-D (explore/exploit)

## How to Run

```bash
# Baseline (pure evolutionary)
python experiments/baseline_run.py

# BED-guided hybrid
python experiments/bed_run.py

# Run tests
python -m pytest tests/
```

## Agent Setup
When spawning worker agents, always:
1. Pass the notebook ID: `50f203d4-a09c-4a1a-9ad3-f449e67d3d11`
2. Read the agent's memory file from `memory/` before starting work
3. Write results back to the memory file when done
