"""
AlphaEvolve Hybrid: evolutionary prompt loop guided by BED mutation selection.

Drop-in replacement for random mutation selection. Plug in your real LLM
candidate generator and fitness function; the BED selector handles the rest.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Callable, Protocol

from bed_selector import BEDSelector, Mutation, make_mutation

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Protocols — implement these for your real system
# ------------------------------------------------------------------

class CandidateGenerator(Protocol):
    def generate(self, current_prompt: str, k: int) -> list[dict]:
        """
        Return K candidate mutations as dicts with keys:
          - text: str              (the mutated prompt)
          - targets: list[str]     (which dimensions this mutation targets)
          - directions: list[float] (estimated change per target, in [-1, 1])
        """
        ...


class FitnessEvaluator(Protocol):
    def evaluate(self, prompt: str) -> tuple[bool, dict]:
        """
        Evaluate a prompt. Returns:
          - fitness: bool          (true = passes market conduct check)
          - artifact: dict         (LLM-generated artifact for optimize step)
        """
        ...


# ------------------------------------------------------------------
# Result types
# ------------------------------------------------------------------

@dataclass
class StepResult:
    eval_number: int
    prompt: str
    mutation_text: str
    fitness: bool
    artifact: dict
    eig_selected: float
    best_prompt_so_far: str
    best_fitness_count: int  # cumulative true count


@dataclass
class RunResult:
    method: str
    steps: list[StepResult] = field(default_factory=list)
    evals_to_threshold: int | None = None
    final_best_prompt: str = ""
    redundant_rate: float = 0.0

    def fitness_curve(self) -> list[tuple[int, float]]:
        """Returns (eval_number, cumulative_true_rate) pairs."""
        return [
            (s.eval_number, s.best_fitness_count / s.eval_number)
            for s in self.steps
        ]


# ------------------------------------------------------------------
# Hybrid loop
# ------------------------------------------------------------------

class AlphaEvolveHybrid:
    """
    AlphaEvolve evolutionary loop with BED-guided mutation selection.

    Replaces random candidate selection with Bayes-Q (EIG maximization),
    maintains SMC particle belief, and uses Bayes-D for early stopping.
    """

    MARKET_CONDUCT_DIMENSIONS = [
        "tone",
        "specificity",
        "constraint_strictness",
        "formality",
        "scope_breadth",
        "compliance_emphasis",
        "instruction_clarity",
    ]

    def __init__(
        self,
        generator: CandidateGenerator,
        evaluator: FitnessEvaluator,
        dimension_names: list[str] | None = None,
        n_particles: int = 100,
        exploit_threshold: float = 0.85,
        k_candidates: int = 5,
        max_evals: int = 200,
        fitness_threshold: float = 0.8,
        seed: int | None = None,
    ) -> None:
        self.generator = generator
        self.evaluator = evaluator
        self.k_candidates = k_candidates
        self.max_evals = max_evals
        self.fitness_threshold = fitness_threshold

        dims = dimension_names or self.MARKET_CONDUCT_DIMENSIONS
        self.bed = BEDSelector(
            dimension_names=dims,
            n_particles=n_particles,
            exploit_threshold=exploit_threshold,
            seed=seed,
        )

    def run(self, seed_prompt: str) -> RunResult:
        result = RunResult(method="bed_guided")
        current_prompt = seed_prompt
        best_prompt = seed_prompt
        true_count = 0
        threshold_evals = int(self.max_evals * self.fitness_threshold)

        for eval_num in range(1, self.max_evals + 1):
            # Bayes-D: check if we should exploit
            if self.bed.should_exploit() and eval_num > 10:
                logger.info("Bayes-D: exploiting at eval %d", eval_num)
                result.evals_to_threshold = eval_num
                break

            # Generate K candidates from LLM
            raw_candidates = self.generator.generate(current_prompt, self.k_candidates)
            mutations = [
                make_mutation(
                    text=c["text"],
                    targets=c["targets"],
                    directions=c["directions"],
                    dimension_names=self.bed.dimension_names,
                )
                for c in raw_candidates
            ]

            # Bayes-Q: select highest-EIG mutation
            selected = self.bed.select(mutations)

            # Evaluate fitness
            fitness, artifact = self.evaluator.evaluate(selected.text)
            if fitness:
                true_count += 1
                best_prompt = selected.text
                current_prompt = selected.text

            # SMC update
            self.bed.update(selected, fitness)

            step = StepResult(
                eval_number=eval_num,
                prompt=selected.text,
                mutation_text=selected.text,
                fitness=fitness,
                artifact=artifact,
                eig_selected=self.bed._eig(selected),
                best_prompt_so_far=best_prompt,
                best_fitness_count=true_count,
            )
            result.steps.append(step)

            cumulative_rate = true_count / eval_num
            logger.debug("Eval %d: fitness=%s, cumulative_rate=%.3f", eval_num, fitness, cumulative_rate)

            if result.evals_to_threshold is None and cumulative_rate >= self.fitness_threshold:
                result.evals_to_threshold = eval_num

        result.final_best_prompt = best_prompt
        result.redundant_rate = self.bed.redundant_rate
        return result


# ------------------------------------------------------------------
# Baseline (pure evolutionary) for comparison
# ------------------------------------------------------------------

class AlphaEvolveBaseline:
    """
    Standard AlphaEvolve: random mutation selection (no BED).
    Same interface as AlphaEvolveHybrid for direct comparison.
    """

    def __init__(
        self,
        generator: CandidateGenerator,
        evaluator: FitnessEvaluator,
        k_candidates: int = 5,
        max_evals: int = 200,
        fitness_threshold: float = 0.8,
        seed: int | None = None,
    ) -> None:
        self.generator = generator
        self.evaluator = evaluator
        self.k_candidates = k_candidates
        self.max_evals = max_evals
        self.fitness_threshold = fitness_threshold
        self._rng = __import__("random").Random(seed)

    def run(self, seed_prompt: str) -> RunResult:
        import random as _random
        result = RunResult(method="baseline_evolutionary")
        current_prompt = seed_prompt
        best_prompt = seed_prompt
        true_count = 0

        for eval_num in range(1, self.max_evals + 1):
            raw_candidates = self.generator.generate(current_prompt, self.k_candidates)

            # Random selection (standard AlphaEvolve)
            chosen = self._rng.choice(raw_candidates)

            fitness, artifact = self.evaluator.evaluate(chosen["text"])
            if fitness:
                true_count += 1
                best_prompt = chosen["text"]
                current_prompt = chosen["text"]

            step = StepResult(
                eval_number=eval_num,
                prompt=chosen["text"],
                mutation_text=chosen["text"],
                fitness=fitness,
                artifact=artifact,
                eig_selected=0.0,
                best_prompt_so_far=best_prompt,
                best_fitness_count=true_count,
            )
            result.steps.append(step)

            cumulative_rate = true_count / eval_num
            if result.evals_to_threshold is None and cumulative_rate >= self.fitness_threshold:
                result.evals_to_threshold = eval_num

        result.final_best_prompt = best_prompt
        result.redundant_rate = 0.0
        return result
