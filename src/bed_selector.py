"""
BED Selector: Bayesian Experimental Design for AlphaEvolve mutation selection.

Implements Bayes-Q (EIG-based selection), SMC belief updates, and Bayes-D
(explore/exploit decision) from arxiv 2510.20886, adapted for prompt evolution
with binary fitness signals.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class Mutation:
    text: str
    targets: list[str]
    target_indices: list[int]
    direction: list[float]


@dataclass
class Particle:
    dimensions: np.ndarray  # shape (n_dims,), values in [0, 1]
    weight: float = 1.0

    def predict_yes(self, mutation: Mutation) -> float:
        """P(fitness=1 | this particle, this mutation)."""
        if not mutation.target_indices:
            return 0.5
        score = sum(
            self.dimensions[i] * d
            for i, d in zip(mutation.target_indices, mutation.direction)
        )
        return _sigmoid(score)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _bernoulli_entropy(p: float) -> float:
    """H(Bernoulli(p)) — zero at p=0 or p=1, max at p=0.5."""
    p = max(1e-10, min(1 - 1e-10, p))
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


class BEDSelector:
    """
    Bayesian Experimental Design mutation selector.

    Maintains a particle-based belief over prompt dimension importances and
    selects the mutation with the highest Expected Information Gain (EIG) at
    each step, matching the Bayes-Q strategy from the paper.
    """

    def __init__(
        self,
        dimension_names: list[str],
        n_particles: int = 100,
        exploit_threshold: float = 0.85,
        resample_threshold: float = 0.5,
        seed: int | None = None,
    ) -> None:
        self.dimension_names = dimension_names
        self.n_dims = len(dimension_names)
        self.exploit_threshold = exploit_threshold
        self.resample_threshold = resample_threshold
        self._rng = np.random.default_rng(seed)

        self.particles: list[Particle] = [
            Particle(
                dimensions=self._rng.uniform(-1.0, 1.0, self.n_dims),
                weight=1.0 / n_particles,
            )
            for _ in range(n_particles)
        ]

        self.eval_count = 0
        self.redundant_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(self, candidates: list[Mutation]) -> Mutation:
        """Bayes-Q: return the candidate with highest EIG."""
        scored = [(self._eig(c), c) for c in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
        best_eig, best = scored[0]

        # Track redundancy: EIG near 0 means mutation is uninformative
        if best_eig < 0.01:
            self.redundant_count += 1

        return best

    def update(self, mutation: Mutation, fitness: bool) -> None:
        """SMC update after observing binary fitness for a mutation."""
        y = 1 if fitness else 0
        for p in self.particles:
            prob = p.predict_yes(mutation)
            likelihood = prob if y == 1 else (1.0 - prob)
            p.weight *= likelihood

        self._normalize_weights()
        self._resample_if_needed()
        self.eval_count += 1

    def should_exploit(self) -> bool:
        """Bayes-D: return True if belief is confident enough to exploit."""
        return max(p.weight for p in self.particles) > self.exploit_threshold

    def best_dimension_belief(self) -> dict[str, float]:
        """Return MAP estimate of each dimension's importance."""
        weighted = np.zeros(self.n_dims)
        for p in self.particles:
            weighted += p.weight * p.dimensions
        return dict(zip(self.dimension_names, weighted.tolist()))

    @property
    def redundant_rate(self) -> float:
        if self.eval_count == 0:
            return 0.0
        return self.redundant_count / self.eval_count

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _eig(self, mutation: Mutation) -> float:
        """Expected Information Gain = H(Bernoulli(p_yes))."""
        p_yes = sum(p.weight * p.predict_yes(mutation) for p in self.particles)
        return _bernoulli_entropy(p_yes)

    def _normalize_weights(self) -> None:
        total = sum(p.weight for p in self.particles)
        if total < 1e-300:
            # Degenerate case: reset to uniform
            for p in self.particles:
                p.weight = 1.0 / len(self.particles)
        else:
            for p in self.particles:
                p.weight /= total

    def _resample_if_needed(self) -> None:
        n = len(self.particles)
        n_eff = 1.0 / sum(p.weight ** 2 for p in self.particles)
        if n_eff < n * self.resample_threshold:
            self._systematic_resample()

    def _systematic_resample(self) -> None:
        """Systematic resampling to avoid particle collapse."""
        n = len(self.particles)
        weights = np.array([p.weight for p in self.particles])
        cumsum = np.cumsum(weights)

        positions = (np.arange(n) + self._rng.uniform(0, 1)) / n
        indices = np.searchsorted(cumsum, positions)
        indices = np.clip(indices, 0, n - 1)

        resampled = [
            Particle(
                dimensions=self.particles[i].dimensions.copy(),
                weight=1.0 / n,
            )
            for i in indices
        ]
        self.particles = resampled


def make_mutation(
    text: str,
    targets: list[str],
    directions: list[float],
    dimension_names: list[str],
) -> Mutation:
    """Helper to build a Mutation with resolved dimension indices."""
    dim_index = {name: i for i, name in enumerate(dimension_names)}
    indices = [dim_index[t] for t in targets if t in dim_index]
    dirs = [d for t, d in zip(targets, directions) if t in dim_index]
    return Mutation(text=text, targets=targets, target_indices=indices, direction=dirs)
