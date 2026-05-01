"""Unit tests for BEDSelector."""

import sys
import os
import math
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bed_selector import BEDSelector, Mutation, make_mutation, _bernoulli_entropy


DIMS = ["tone", "specificity", "constraint_strictness", "formality"]


def make_selector(n_particles: int = 50, seed: int = 42) -> BEDSelector:
    return BEDSelector(dimension_names=DIMS, n_particles=n_particles, seed=seed)


def simple_mutation(targets: list[str], directions: list[float]) -> Mutation:
    return make_mutation("test prompt", targets, directions, DIMS)


# ------------------------------------------------------------------
# EIG tests
# ------------------------------------------------------------------

def test_eig_deterministic_zero():
    """EIG should be ~0 when all particles predict p_yes=0 or p_yes=1."""
    sel = make_selector()
    # Force all particles to have very high dimension values → p_yes ≈ 1
    for p in sel.particles:
        p.dimensions[:] = 10.0
    m = simple_mutation(["tone"], [1.0])
    eig = sel._eig(m)
    assert eig < 0.01, f"Expected near-zero EIG, got {eig}"


def test_eig_max_at_half():
    """EIG is maximized (=1.0 bits) when p_yes = 0.5."""
    assert abs(_bernoulli_entropy(0.5) - 1.0) < 1e-6


def test_eig_zero_at_extremes():
    """EIG is 0 at p=0 and p=1."""
    assert _bernoulli_entropy(0.0) == pytest.approx(0.0, abs=1e-6)
    assert _bernoulli_entropy(1.0) == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------------------------
# Select tests
# ------------------------------------------------------------------

def test_select_returns_one_candidate():
    sel = make_selector()
    candidates = [
        simple_mutation(["tone"], [0.5]),
        simple_mutation(["formality"], [-0.3]),
        simple_mutation(["specificity"], [0.8]),
    ]
    chosen = sel.select(candidates)
    assert chosen in candidates


def test_select_prefers_uncertain_mutation():
    """If all particles are neutral (weights uniform, dims≈0), all mutations have EIG≈1."""
    sel = make_selector(seed=0)
    for p in sel.particles:
        p.dimensions[:] = 0.0  # sigmoid(0) = 0.5 for any target

    candidates = [simple_mutation(["tone"], [d]) for d in [0.1, 0.5, 0.9]]
    eigs = [sel._eig(c) for c in candidates]
    # All should be near-maximal (p_yes ≈ 0.5)
    for e in eigs:
        assert e > 0.9, f"Expected high EIG near uniform belief, got {e}"


# ------------------------------------------------------------------
# Update / SMC tests
# ------------------------------------------------------------------

def test_weights_sum_to_one_after_update():
    sel = make_selector()
    m = simple_mutation(["tone"], [0.5])
    sel.update(m, fitness=True)
    total = sum(p.weight for p in sel.particles)
    assert abs(total - 1.0) < 1e-9


def test_eval_count_increments():
    sel = make_selector()
    m = simple_mutation(["tone"], [0.5])
    assert sel.eval_count == 0
    sel.update(m, True)
    assert sel.eval_count == 1
    sel.update(m, False)
    assert sel.eval_count == 2


def test_consistent_updates_shift_belief():
    """Repeated positive observations for 'tone' should increase tone's MAP estimate."""
    sel = make_selector(seed=99)
    m = simple_mutation(["tone"], [1.0])
    initial_belief = sel.best_dimension_belief()["tone"]
    for _ in range(20):
        sel.update(m, fitness=True)
    final_belief = sel.best_dimension_belief()["tone"]
    # After many positive observations aligned with tone, belief should shift
    assert final_belief != initial_belief  # belief has moved


# ------------------------------------------------------------------
# Exploit threshold
# ------------------------------------------------------------------

def test_no_exploit_at_start():
    sel = make_selector()
    # With uniform weights across 50 particles, max weight = 1/50 = 0.02 << 0.85
    assert not sel.should_exploit()


def test_exploit_when_one_particle_dominates():
    sel = make_selector(n_particles=10)
    # Force one particle to dominate
    for p in sel.particles:
        p.weight = 0.01
    sel.particles[0].weight = 0.90
    sel._normalize_weights()
    assert sel.should_exploit()
