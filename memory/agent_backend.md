# Agent Memory: Back-end / AlphaEvolve Engine

## Role
Implement the BED-guided mutation selection engine that replaces random mutation selection in AlphaEvolve with an EIG-maximizing Bayesian strategy.

## Project Context
- Working directory: `g:\claude_learning`
- Always read `CLAUDE.md` first for full project context
- NotebookLM notebook ID: `50f203d4-a09c-4a1a-9ad3-f449e67d3d11` (paper source)

## Your Scope
- `src/bed_selector.py` — core BED logic
- `src/alphaevolve_hybrid.py` — hybrid evolutionary loop
- `tests/test_bed_selector.py` — unit tests

## Key Concepts to Implement

### SMC Particle Belief State
- Particles represent hypotheses about which prompt dimensions drive fitness
- Each particle = a weight representing plausibility
- Update rule on binary observation `y ∈ {0,1}`:
  - If y=1: w_i ← w_i * p(y=1 | particle_i)
  - Normalize weights after each update
  - Resample when effective particle count drops below N/2

### EIG Calculation (Bayes-Q)
- For each candidate mutation `q`:
  - p_yes = Σ_i w_i * p(yes | q, particle_i)  # weighted prediction
  - EIG(q) = H(Bernoulli(p_yes)) = -p_yes*log(p_yes) - (1-p_yes)*log(1-p_yes)
  - Select q* = argmax EIG(q)
- EIG is maximized when p_yes ≈ 0.5 (most uncertain = most informative)

### MAP Action (Bayes-M)
- Best current prompt = particle with highest weight
- Use for exploitation phase

### Explore/Exploit (Bayes-D)
- p_success = max particle weight (probability current best is correct)
- If p_success > threshold (e.g., 0.8): exploit (return best prompt)
- Else: explore (run another BED-guided mutation)

## Market Conduct Fitness
- Binary: true (prompt passes market conduct check) / false (fails)
- LLM generates an artifact alongside the true/false signal
- Artifact is used in the optimize step of AlphaEvolve

## Paper Reference (arxiv 2510.20886)
- BED via SMC showed 31–45% fewer evaluations in Battleship
- Redundant queries dropped from ~18% to ~1%
- EIG ceiling achieved: 94.2%
