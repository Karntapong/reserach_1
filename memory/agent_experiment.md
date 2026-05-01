# Agent Memory: Experiment Runner / Evaluation

## Role
Run and compare baseline (pure evolutionary AlphaEvolve) vs. BED-guided hybrid. Log convergence curves and produce comparison results.

## Project Context
- Working directory: `g:\claude_learning`
- Always read `CLAUDE.md` first for full project context
- NotebookLM notebook ID: `50f203d4-a09c-4a1a-9ad3-f449e67d3d11` (paper source)

## Your Scope
- `experiments/baseline_run.py` — pure evolutionary baseline
- `experiments/bed_run.py` — BED-guided experiment
- `experiments/compare_results.py` — convergence curve comparison

## Experiment Design

### Baseline (experiments/baseline_run.py)
- Standard AlphaEvolve: random mutation selection from LLM candidates
- Track: fitness score per evaluation count
- Run N trials, record mean + std of evaluations to reach fitness threshold

### BED-guided (experiments/bed_run.py)
- Same setup but mutation selection goes through `BEDSelector`
- Same fitness function, same LLM for candidates
- Track: same metrics for direct comparison

### Key Metrics
| Metric | Target |
|--------|--------|
| Evaluations to threshold | 31–45% fewer than baseline |
| Redundant mutations selected | <2% (vs. ~18% baseline) |
| Final fitness achieved | Equal or better than baseline |

### Fitness Threshold
- Define as: first time fitness reaches 0.8 (or as appropriate for market conduct task)

## Market Conduct Task
- Input: a prompt to be evolved
- Fitness: binary true/false (does this prompt comply with market conduct rules?)
- LLM artifact: generated alongside fitness, used in optimize step
- Simulate with a mock fitness function if real evaluator not available

## Convergence Curve Format
```python
# Output format for each run
{
    "run_id": int,
    "method": "baseline" | "bed",
    "evaluations": [int],       # eval count at each step
    "fitness": [float],         # fitness at each step
    "evals_to_threshold": int,  # primary metric
    "redundant_rate": float     # fraction of redundant mutations
}
```
