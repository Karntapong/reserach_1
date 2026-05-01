"""
BED-enhanced ProgramDatabase for OpenEvolve.

Subclasses openevolve.database.ProgramDatabase and overrides sample() to use
EIG-maximizing Bayesian selection instead of OpenEvolve's default weighted-random
selection. Also overrides add() to feed fitness observations back to the BED
particle belief.

Integration point in OpenEvolve's loop:
    iteration.py → database.sample() → [overridden here] → BEDSelector.select()
    iteration.py → database.add()    → [overridden here] → BEDSelector.update()
"""

from __future__ import annotations

import threading
from typing import List, Optional, Tuple

from openevolve.database import ProgramDatabase, Program

from bed_selector import BEDSelector, Mutation, make_mutation


# Default market-conduct-relevant feature dimensions.
# These should match the MAP-Elites feature names in your OpenEvolve config.
DEFAULT_DIMENSIONS = [
    "fitness",
    "complexity",
    "diversity",
    "compliance_score",
    "clarity_score",
]

FITNESS_THRESHOLD = 0.5  # binary cutoff: score >= threshold → fitness=True


class BEDProgramDatabase(ProgramDatabase):
    """
    OpenEvolve ProgramDatabase with BED-guided parent selection.

    Usage:
        db = BEDProgramDatabase(config, bed_selector=BEDSelector(...))
        # Then pass db to OpenEvolve controller instead of the default database.
    """

    def __init__(
        self,
        *args,
        bed_selector: BEDSelector | None = None,
        k_candidates: int = 5,
        fitness_metric: str = "fitness",
        fitness_threshold: float = FITNESS_THRESHOLD,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.bed = bed_selector or BEDSelector(dimension_names=DEFAULT_DIMENSIONS)
        self.k_candidates = k_candidates
        self.fitness_metric = fitness_metric
        self.fitness_threshold = fitness_threshold

        # Track last selected mutation per thread for the update step in add()
        self._last_mutation: dict[int, Mutation] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def sample(
        self, num_inspirations: Optional[int] = None
    ) -> Tuple[Program, List[Program]]:
        """
        Bayes-Q: select the parent program that maximizes EIG.

        Calls the parent sample() K times to get K candidate parents,
        converts each to a BED Mutation, and returns the one with highest EIG.
        The inspiration programs come from the best-scoring candidate's sample call.
        """
        # Collect K candidate (parent, inspirations) pairs
        candidates: list[tuple[Program, list[Program]]] = []
        for _ in range(self.k_candidates):
            try:
                parent, inspirations = super().sample(num_inspirations)
                candidates.append((parent, inspirations))
            except Exception:
                break

        if not candidates:
            # Fallback: database may be empty at startup
            return super().sample(num_inspirations)

        # Convert parents to BED mutations
        mutations = [self._program_to_mutation(p) for p, _ in candidates]

        # Bayes-Q: select highest-EIG mutation
        with self._lock:
            selected_mutation = self.bed.select(mutations)
            thread_id = threading.get_ident()
            self._last_mutation[thread_id] = selected_mutation

        # Return the corresponding (parent, inspirations)
        idx = mutations.index(selected_mutation)
        return candidates[idx]

    def add(
        self,
        program: Program,
        iteration: int = None,
        target_island: Optional[int] = None,
    ) -> str:
        """
        After evaluation: update BED belief with the observed fitness.
        Then delegate to parent add() to update the MAP-Elites population.
        """
        thread_id = threading.get_ident()
        with self._lock:
            last_mutation = self._last_mutation.pop(thread_id, None)

        if last_mutation is not None:
            score = self._get_fitness_score(program)
            fitness = score >= self.fitness_threshold
            self.bed.update(last_mutation, fitness)

        return super().add(program, iteration=iteration, target_island=target_island)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _program_to_mutation(self, program: Program) -> Mutation:
        """Convert a Program's metrics into a BED Mutation representation."""
        metrics = getattr(program, "metrics", {}) or {}
        if not metrics and hasattr(program, "score"):
            metrics = {self.fitness_metric: getattr(program, "score", 0.0)}

        targets = [d for d in self.bed.dimension_names if d in metrics]
        directions = [float(metrics[d]) for d in targets]

        # Fallback: if no known dimension in metrics, use fitness_metric with 0.0
        if not targets:
            targets = [self.fitness_metric]
            directions = [float(metrics.get(self.fitness_metric, 0.0))]

        return make_mutation(
            text=getattr(program, "id", str(id(program))),
            targets=targets,
            directions=directions,
            dimension_names=self.bed.dimension_names,
        )

    def _get_fitness_score(self, program: Program) -> float:
        metrics = getattr(program, "metrics", {}) or {}
        if self.fitness_metric in metrics:
            return float(metrics[self.fitness_metric])
        if hasattr(program, "score"):
            return float(program.score)
        # Try the first numeric metric
        for v in metrics.values():
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
        return 0.0


def make_bed_database(
    config,
    dimension_names: list[str] | None = None,
    n_particles: int = 100,
    k_candidates: int = 5,
    exploit_threshold: float = 0.85,
    fitness_metric: str = "fitness",
    fitness_threshold: float = FITNESS_THRESHOLD,
    seed: int | None = None,
) -> BEDProgramDatabase:
    """Factory: build a BEDProgramDatabase ready to drop into OpenEvolve."""
    dims = dimension_names or DEFAULT_DIMENSIONS
    selector = BEDSelector(
        dimension_names=dims,
        n_particles=n_particles,
        exploit_threshold=exploit_threshold,
        seed=seed,
    )
    return BEDProgramDatabase(
        config,
        bed_selector=selector,
        k_candidates=k_candidates,
        fitness_metric=fitness_metric,
        fitness_threshold=fitness_threshold,
    )
