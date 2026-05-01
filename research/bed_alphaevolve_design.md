# BED × AlphaEvolve: Theoretical Mapping and Design

## Problem Statement

AlphaEvolve uses evolutionary search to evolve prompts. Each generation:
1. LLM proposes K candidate mutations of the current best prompt
2. One (or a few) candidates are selected randomly or by heuristic
3. Selected candidates are evaluated by the fitness function
4. Best survivors propagate to next generation

**Inefficiency:** Step 2 ignores information about which parts of prompt space have already been explored. Mutations may be redundant or uninformative.

## Paper Summary (arxiv 2510.20886)

The paper frames agentic information-seeking as **Bayesian Experimental Design (BED)**:
- Agent maintains a **belief** over world states as weighted particles (SMC)
- At each step, selects the action that maximizes **Expected Information Gain (EIG)**
- EIG = entropy of predictive distribution = H(Bernoulli(p_yes))
- Binary fitness (yes/no) maps directly to Bernoulli observations

Results on Battleship game:
- 31–45% fewer moves to win
- Near-zero redundant questions (0.2–1.2% vs 14–18% baseline)
- 94.2% of theoretical EIG ceiling achieved

## Component Mapping

### 1. Belief State → Prompt Hypothesis Space

In the paper: particles = possible board configurations

In our system: particles = hypotheses about **which prompt dimensions drive fitness**

**Representation (Option B — structured dimensions):**
```python
# Each particle is a weight vector over prompt dimensions
# Dimensions: tone, specificity, constraint_type, length, formality, ...
particle = {
    "dimensions": np.array([d1, d2, ..., dn]),  # continuous [0,1]
    "weight": float
}
```

**Why structured over embeddings:**
- More interpretable
- Easier to compute p(fitness=1 | particle, mutation)
- Cheaper than embedding every candidate

### 2. Candidate Mutations → Questions

In the paper: LLM samples natural language questions, translates to Python programs

In our system: LLM samples candidate prompt mutations, each mutation targets a specific dimension

```python
# LLM proposes mutations with metadata
mutation = {
    "text": "revised prompt text",
    "targets": ["tone", "specificity"],  # which dimensions this mutation changes
    "direction": [+0.3, -0.1]            # estimated change in each dimension
}
```

### 3. EIG Calculation

For each candidate mutation q:
```
p_yes(q) = Σ_i  weight_i * sigmoid(particle_i.dimensions · q.targets)
EIG(q)   = -p_yes * log(p_yes) - (1-p_yes) * log(1-p_yes)
```

Select q* = argmax_q EIG(q)

**Key property:** EIG is maximized when p_yes ≈ 0.5 — the mutation most likely to halve our uncertainty.

### 4. SMC Belief Update

After observing fitness y ∈ {0, 1} for mutation q*:
```
# Update weights
for particle_i:
    p = sigmoid(particle_i.dimensions · q*.targets)
    likelihood = p if y==1 else (1-p)
    particle_i.weight *= likelihood

# Normalize
total = sum(p.weight for p in particles)
for p in particles: p.weight /= total

# Resample if effective N too low
N_eff = 1 / sum(p.weight**2 for p in particles)
if N_eff < N_particles / 2:
    resample(particles)  # systematic resampling
```

### 5. Exploit vs. Explore (Bayes-D)

```
p_success = max(p.weight for p in particles)

if p_success > EXPLOIT_THRESHOLD:  # e.g., 0.85
    return best_current_prompt      # Bayes-M: MAP action
else:
    run_another_BED_step()          # Bayes-Q: keep exploring
```

## Full Hybrid Loop

```
Initialize:
  - N particles with uniform weights
  - Current best prompt = seed prompt
  - Eval count = 0

Loop until convergence or budget exhausted:
  1. LLM generates K candidate mutations of current best prompt
  2. For each mutation: compute EIG against current particle belief
  3. Select mutation q* with highest EIG
  4. Evaluate q* fitness → y ∈ {0, 1}, artifact
  5. Update SMC particles with observation (q*, y)
  6. If y==1 and fitness(q*) > fitness(current best): update current best
  7. Bayes-D: if p_success > threshold → return current best (converged)
  8. Eval count += 1
```

## Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Particle representation | Structured dimensions | Interpretable, fast EIG computation |
| N particles | 50–200 | Balance accuracy vs. compute |
| Exploit threshold | 0.85 | Conservative; tune based on results |
| K candidates per step | 5–10 | Same as standard AlphaEvolve |
| LLM for candidates | Same model as AlphaEvolve | Consistent comparison |

## Open Questions

1. How to reliably extract `targets` and `direction` from LLM-proposed mutations?
   - Option: structured output with JSON schema
   - Option: embedding similarity to dimension prototypes

2. How to initialize particle dimensions for market conduct?
   - Need domain knowledge about which prompt axes matter for compliance

3. What is the real fitness function signature?
   - Needs to be confirmed with the actual codebase
