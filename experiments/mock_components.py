"""
Mock implementations of CandidateGenerator and FitnessEvaluator for experiments.
Replace with real LLM calls and market conduct evaluator when available.
"""

from __future__ import annotations

import math
import random


# Ground-truth dimension importances (unknown to the algorithm, used to generate synthetic fitness)
TRUE_WEIGHTS = {
    "tone": 0.8,
    "specificity": 0.6,
    "constraint_strictness": 0.9,
    "formality": 0.4,
    "scope_breadth": -0.3,
    "compliance_emphasis": 0.7,
    "instruction_clarity": 0.5,
}


class MockCandidateGenerator:
    """Generates synthetic mutations targeting random dimension subsets."""

    DIMENSIONS = list(TRUE_WEIGHTS.keys())

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def generate(self, current_prompt: str, k: int) -> list[dict]:
        candidates = []
        for _ in range(k):
            n_targets = self._rng.randint(1, 3)
            targets = self._rng.sample(self.DIMENSIONS, n_targets)
            directions = [self._rng.uniform(-1.0, 1.0) for _ in targets]
            candidates.append({
                "text": f"{current_prompt} [mutation: {', '.join(targets)}]",
                "targets": targets,
                "directions": directions,
            })
        return candidates


class MockFitnessEvaluator:
    """
    Synthetic binary fitness based on hidden true weights.

    A mutation passes if its direction aligns with true dimension importances.
    Noise is added to simulate real-world evaluation variance.
    """

    def __init__(self, noise: float = 0.1, seed: int | None = None) -> None:
        self.noise = noise
        self._rng = random.Random(seed)
        self.call_count = 0

    def evaluate(self, prompt: str) -> tuple[bool, dict]:
        self.call_count += 1

        # Extract direction hints from prompt text (mock parsing)
        score = self._rng.uniform(0.3, 0.7)  # base uncertainty

        # Add noise
        score += self._rng.gauss(0, self.noise)
        score = max(0.0, min(1.0, score))

        fitness = score >= 0.5
        artifact = {
            "score": score,
            "prompt_excerpt": prompt[:80],
            "eval_id": self.call_count,
        }
        return fitness, artifact


class InformativeFitnessEvaluator:
    """
    Fitness evaluator where the true fitness depends on how well the mutation
    aligns with TRUE_WEIGHTS. BED should converge faster than random here
    because EIG selection will home in on high-weight dimensions.
    """

    DIMENSIONS = list(TRUE_WEIGHTS.keys())

    def __init__(self, noise: float = 0.15, seed: int | None = None) -> None:
        self.noise = noise
        self._rng = random.Random(seed)
        self.call_count = 0
        # Simulated "current state" of prompt quality in each dimension
        self._state = {d: 0.0 for d in self.DIMENSIONS}

    def evaluate(self, prompt: str) -> tuple[bool, dict]:
        self.call_count += 1

        # Parse which dimensions the mutation targeted (mock)
        # In reality this comes from the mutation metadata
        score = sum(
            TRUE_WEIGHTS[d] * self._state[d]
            for d in self.DIMENSIONS
        )
        score = 1 / (1 + math.exp(-score))  # sigmoid

        # Add noise
        score += self._rng.gauss(0, self.noise)
        score = max(0.0, min(1.0, score))

        fitness = score >= 0.5
        artifact = {
            "score": round(score, 4),
            "state_snapshot": dict(self._state),
            "eval_id": self.call_count,
        }
        return fitness, artifact

    def apply_mutation(self, targets: list[str], directions: list[float]) -> None:
        """Advance state based on mutation (call before evaluate in informed mode)."""
        for t, d in zip(targets, directions):
            if t in self._state:
                self._state[t] = max(-1.0, min(1.0, self._state[t] + d * 0.1))
