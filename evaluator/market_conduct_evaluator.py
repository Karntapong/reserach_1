"""
OpenEvolve evaluator for market conduct prompt evolution.

OpenEvolve calls evaluate(program_path) for each evolved candidate.
The function must return an EvaluationResult (or dict[str, float]).

Wire in your real market conduct checker where indicated below.
"""

from __future__ import annotations

import json
import os

from openevolve.evaluation_result import EvaluationResult


def evaluate(program_path: str) -> EvaluationResult:
    """
    Evaluate an evolved prompt for market conduct compliance.

    Args:
        program_path: Path to a file containing the evolved prompt text.

    Returns:
        EvaluationResult with at minimum a "fitness" metric (0.0 or 1.0).
    """
    with open(program_path, "r", encoding="utf-8") as f:
        prompt = f.read().strip()

    # ----------------------------------------------------------------
    # REPLACE THIS SECTION with your real market conduct evaluator.
    # Your evaluator should:
    #   1. Run the prompt through your compliance check (LLM-as-judge or rule-based)
    #   2. Return binary result: 1.0 (pass) or 0.0 (fail)
    #   3. Optionally return additional metrics for MAP-Elites features
    # ----------------------------------------------------------------
    fitness, artifact = _stub_market_conduct_check(prompt)
    # ----------------------------------------------------------------

    return EvaluationResult(
        metrics={
            "fitness": 1.0 if fitness else 0.0,
            # Add additional MAP-Elites feature metrics here, e.g.:
            # "prompt_length": len(prompt),
            # "compliance_score": artifact.get("compliance_score", 0.0),
        },
        artifacts={
            "prompt": prompt,
            "artifact": json.dumps(artifact),
        },
    )


def _stub_market_conduct_check(prompt: str) -> tuple[bool, dict]:
    """
    Stub evaluator — replace with real implementation.

    Returns a random-ish result based on prompt characteristics so the
    experiment runner can at least run end-to-end.
    """
    import hashlib
    score = int(hashlib.md5(prompt.encode()).hexdigest(), 16) % 100 / 100.0
    fitness = score >= 0.5
    return fitness, {"stub_score": score, "prompt_length": len(prompt)}
