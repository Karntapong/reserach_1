# OpenEvolve × Bayesian Experimental Design

## Objective
Accelerate **OpenEvolve** prompt convergence for a **market conduct** use case by replacing the default weighted-random parent selection with **Bayesian Experimental Design (BED)** — EIG-maximizing selection via Sequential Monte Carlo.

## Dependency
```bash
pip install openevolve numpy
```
OpenEvolve: https://github.com/algorithmicsuperintelligence/openevolve

## Hypothesis
Paper arxiv 2510.20886 shows BED reduces evaluations by 31–45% in binary-fitness settings. Since market conduct fitness is binary (true/false = Bernoulli observation), BED's EIG acquisition naturally minimises redundant evaluations.

## Architecture

```
OpenEvolve (controller.py)
    ↓ iteration.py calls database.sample()
BEDProgramDatabase.sample()         ← overrides default weighted-random
    ↓ BEDSelector.select() — picks highest-EIG parent
LLM mutates selected parent
Evaluator scores child prompt (binary fitness)
BEDProgramDatabase.add()            ← overrides to call BEDSelector.update()
    ↓ SMC particle belief updated
    ↓ Bayes-D: exploit threshold check
```

## Key Files

| File | Role |
|------|------|
| `src/bed_selector.py` | BEDSelector: EIG, SMC, Bayes-Q/M/D |
| `src/bed_database.py` | BEDProgramDatabase: subclasses OpenEvolve's ProgramDatabase |
| `evaluator/market_conduct_evaluator.py` | OpenEvolve evaluator stub — wire in real checker here |
| `configs/bed_config.yaml` | OpenEvolve config for BED run |
| `experiments/run_baseline.py` | Stock OpenEvolve baseline |
| `experiments/run_bed.py` | BED-guided OpenEvolve run |
| `research/bed_alphaevolve_design.md` | Theoretical mapping and design decisions |

## Paper Reference
- **arxiv 2510.20886** — Bayesian Experimental Design for agentic information-seeking
- NotebookLM ID: `50f203d4-a09c-4a1a-9ad3-f449e67d3d11`
- Key mechanism: Bayes-Q (EIG), Bayes-M (MAP action), Bayes-D (explore/exploit)

## How to Run

```bash
# Install dependencies
pip install openevolve numpy

# Run tests (no API key needed)
uv tool run --with numpy pytest tests/ -v

# Baseline (stock OpenEvolve — needs OPENAI_API_KEY)
python experiments/run_baseline.py

# BED-guided run (needs OPENAI_API_KEY)
python experiments/run_bed.py
```

## Wire In Your Real Evaluator
Edit `evaluator/market_conduct_evaluator.py` → replace `_stub_market_conduct_check()` with your real compliance checker.

## Agent Setup
When spawning worker agents, always:
1. Read the agent's memory file from `memory/` before starting work
2. Pass the NotebookLM notebook ID: `50f203d4-a09c-4a1a-9ad3-f449e67d3d11`
3. Write results back to the memory file when done
