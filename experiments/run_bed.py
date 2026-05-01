"""
BED-guided run: OpenEvolve with BEDProgramDatabase replacing the default selector.

Usage:
    cd g:\\claude_learning
    python experiments/run_bed.py
"""

from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from openevolve import Config, OpenEvolve

from bed_database import make_bed_database

SEED_PROMPT = """\
You are a financial advisor. Evaluate whether this product recommendation
complies with market conduct regulations. Provide a clear pass/fail decision.
"""

EVALUATOR_PATH = os.path.join(os.path.dirname(__file__), "..", "evaluator", "market_conduct_evaluator.py")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "bed_config.yaml")

# BED hyperparameters
BED_DIMENSIONS = ["fitness", "prompt_length"]
N_PARTICLES = 100
K_CANDIDATES = 5
EXPLOIT_THRESHOLD = 0.85


def main() -> None:
    print("Running BED-guided OpenEvolve...")

    config = Config.from_file(CONFIG_PATH)
    config.output.log_dir = "outputs/bed_run"

    # Build BED-enhanced database and inject into OpenEvolve
    bed_db = make_bed_database(
        config=config,
        dimension_names=BED_DIMENSIONS,
        n_particles=N_PARTICLES,
        k_candidates=K_CANDIDATES,
        exploit_threshold=EXPLOIT_THRESHOLD,
        fitness_metric="fitness",
        fitness_threshold=0.5,
        seed=42,
    )

    controller = OpenEvolve(
        initial_program=SEED_PROMPT,
        evaluator=EVALUATOR_PATH,
        config=config,
        database=bed_db,          # inject BED database
    )

    result = controller.run()

    print(f"\nBED-guided run complete.")
    print(f"Best score:       {result.best_score:.4f}")
    print(f"BED eval count:   {bed_db.bed.eval_count}")
    print(f"Redundant rate:   {bed_db.bed.redundant_rate:.1%}")
    print(f"Output dir:       {result.output_dir}")
    print(f"\nBelief (top dimensions): {bed_db.bed.best_dimension_belief()}")


if __name__ == "__main__":
    main()
