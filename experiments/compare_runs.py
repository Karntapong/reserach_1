"""
Compare baseline AlphaEvolve vs BED-guided hybrid on a mock market conduct task.

Usage:
    cd g:\\claude_learning
    python experiments/compare_runs.py
"""

from __future__ import annotations

import sys
import os
import json
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from alphaevolve_hybrid import AlphaEvolveBaseline, AlphaEvolveHybrid, RunResult
from mock_components import MockCandidateGenerator, MockFitnessEvaluator

SEED_PROMPT = "Evaluate this financial product recommendation for compliance."
N_TRIALS = 20
MAX_EVALS = 100
FITNESS_THRESHOLD = 0.6
K_CANDIDATES = 5


def run_trials(method: str, n_trials: int) -> list[RunResult]:
    results = []
    for trial in range(n_trials):
        gen = MockCandidateGenerator(seed=trial * 100)
        evaluator = MockFitnessEvaluator(seed=trial * 100 + 1)

        if method == "baseline":
            runner = AlphaEvolveBaseline(
                generator=gen,
                evaluator=evaluator,
                k_candidates=K_CANDIDATES,
                max_evals=MAX_EVALS,
                fitness_threshold=FITNESS_THRESHOLD,
                seed=trial,
            )
        else:
            runner = AlphaEvolveHybrid(
                generator=gen,
                evaluator=evaluator,
                k_candidates=K_CANDIDATES,
                max_evals=MAX_EVALS,
                fitness_threshold=FITNESS_THRESHOLD,
                seed=trial,
            )

        result = runner.run(SEED_PROMPT)
        results.append(result)

    return results


def summarize(results: list[RunResult], method: str) -> dict:
    evals = [r.evals_to_threshold for r in results if r.evals_to_threshold is not None]
    redundant = [r.redundant_rate for r in results]

    return {
        "method": method,
        "n_trials": len(results),
        "converged": len(evals),
        "mean_evals_to_threshold": round(statistics.mean(evals), 1) if evals else None,
        "std_evals_to_threshold": round(statistics.stdev(evals), 1) if len(evals) > 1 else None,
        "mean_redundant_rate": round(statistics.mean(redundant), 4),
    }


def main() -> None:
    print(f"Running {N_TRIALS} trials each | max_evals={MAX_EVALS} | threshold={FITNESS_THRESHOLD}\n")

    print("Running baseline...")
    baseline_results = run_trials("baseline", N_TRIALS)
    baseline_summary = summarize(baseline_results, "baseline")

    print("Running BED-guided...")
    bed_results = run_trials("bed", N_TRIALS)
    bed_summary = summarize(bed_results, "bed_guided")

    print("\n=== Results ===")
    for s in [baseline_summary, bed_summary]:
        print(json.dumps(s, indent=2))

    if baseline_summary["mean_evals_to_threshold"] and bed_summary["mean_evals_to_threshold"]:
        reduction = (
            1 - bed_summary["mean_evals_to_threshold"] / baseline_summary["mean_evals_to_threshold"]
        ) * 100
        print(f"\nEval reduction: {reduction:.1f}%")
        print(f"Redundant rate: baseline={baseline_summary['mean_redundant_rate']:.1%}  bed={bed_summary['mean_redundant_rate']:.1%}")


if __name__ == "__main__":
    main()
