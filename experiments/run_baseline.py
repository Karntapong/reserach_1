"""
Baseline run: stock OpenEvolve with default ProgramDatabase (no BED).

Usage:
    cd g:\\claude_learning
    python experiments/run_baseline.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openevolve import run_evolution

SEED_PROMPT = """\
You are a financial advisor. Evaluate whether this product recommendation
complies with market conduct regulations. Provide a clear pass/fail decision.
"""

EVALUATOR_PATH = os.path.join(os.path.dirname(__file__), "..", "evaluator", "market_conduct_evaluator.py")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "bed_config.yaml")


def main() -> None:
    print("Running baseline (stock OpenEvolve)...")

    result = run_evolution(
        initial_program=SEED_PROMPT,
        evaluator=EVALUATOR_PATH,
        config=CONFIG_PATH,
        output_dir="outputs/baseline_run",
    )

    print(f"\nBaseline complete.")
    print(f"Best score:   {result.best_score:.4f}")
    print(f"Output dir:   {result.output_dir}")


if __name__ == "__main__":
    main()
