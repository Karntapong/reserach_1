"""
Integration tests for BEDProgramDatabase.

Uses unittest.mock to stub out openevolve.database.ProgramDatabase so these
tests run without a live OpenEvolve install or LLM API key.
"""

from __future__ import annotations

import sys
import os
import types
from unittest.mock import MagicMock, patch, PropertyMock
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ---------------------------------------------------------------------------
# Stub the openevolve package so tests work without pip install openevolve
# ---------------------------------------------------------------------------

def _make_openevolve_stub():
    openevolve_mod = types.ModuleType("openevolve")
    db_mod = types.ModuleType("openevolve.database")

    class Program:
        def __init__(self, id="p1", metrics=None, score=0.0):
            self.id = id
            self.metrics = metrics or {}
            self.score = score

    class ProgramDatabase:
        def __init__(self, config=None, *args, **kwargs):
            self.config = config

        def sample(self, num_inspirations=None):
            return Program("default_parent"), []

        def add(self, program, iteration=None, target_island=None):
            return program.id

    db_mod.ProgramDatabase = ProgramDatabase
    db_mod.Program = Program
    openevolve_mod.database = db_mod
    sys.modules["openevolve"] = openevolve_mod
    sys.modules["openevolve.database"] = db_mod
    return Program, ProgramDatabase


Program, ProgramDatabase = _make_openevolve_stub()


from bed_selector import BEDSelector
from bed_database import BEDProgramDatabase, make_bed_database


DIMS = ["fitness", "complexity"]


def make_db(k=3, seed=0) -> BEDProgramDatabase:
    selector = BEDSelector(dimension_names=DIMS, n_particles=50, seed=seed)
    return BEDProgramDatabase(config=None, bed_selector=selector, k_candidates=k)


# ---------------------------------------------------------------------------
# sample() tests
# ---------------------------------------------------------------------------

def test_sample_returns_tuple():
    db = make_db()
    result = db.sample()
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_sample_returns_program_and_list():
    db = make_db()
    parent, inspirations = db.sample()
    assert isinstance(parent, Program)
    assert isinstance(inspirations, list)


def test_sample_calls_parent_k_times():
    db = make_db(k=4)
    call_count = []
    original_sample = ProgramDatabase.sample

    def counting_sample(self, num_inspirations=None):
        call_count.append(1)
        p = Program(id=f"p{len(call_count)}", metrics={"fitness": 0.5})
        return p, []

    with patch.object(ProgramDatabase, "sample", counting_sample):
        db.sample()

    assert len(call_count) == 4


# ---------------------------------------------------------------------------
# add() / BED update tests
# ---------------------------------------------------------------------------

def test_add_updates_bed_eval_count():
    db = make_db()
    # sample first to set _last_mutation
    db.sample()
    assert db.bed.eval_count == 0

    program = Program(id="child", metrics={"fitness": 1.0})
    db.add(program)
    assert db.bed.eval_count == 1


def test_add_without_prior_sample_does_not_crash():
    db = make_db()
    # No sample() called → _last_mutation is empty → should still call super().add()
    program = Program(id="child", metrics={"fitness": 0.0})
    result = db.add(program)
    assert result == "child"


def test_add_delegates_to_parent():
    db = make_db()
    db.sample()
    program = Program(id="xyz", metrics={"fitness": 1.0})
    result = db.add(program)
    assert result == "xyz"


# ---------------------------------------------------------------------------
# factory function
# ---------------------------------------------------------------------------

def test_make_bed_database_returns_instance():
    db = make_bed_database(config=None, dimension_names=DIMS, n_particles=20)
    assert isinstance(db, BEDProgramDatabase)
    assert db.bed.n_dims == 2


def test_make_bed_database_selector_configured():
    db = make_bed_database(config=None, dimension_names=DIMS, n_particles=30, k_candidates=7)
    assert db.k_candidates == 7
    assert len(db.bed.particles) == 30


# ---------------------------------------------------------------------------
# program_to_mutation helper
# ---------------------------------------------------------------------------

def test_program_to_mutation_uses_known_dimensions():
    db = make_db()
    p = Program(id="p1", metrics={"fitness": 0.9, "complexity": 0.3, "unknown_dim": 5.0})
    m = db._program_to_mutation(p)
    assert "fitness" in m.targets
    assert "complexity" in m.targets
    assert "unknown_dim" not in m.targets


def test_program_to_mutation_fallback_on_empty_metrics():
    db = make_db()
    p = Program(id="p1", metrics={}, score=0.7)
    m = db._program_to_mutation(p)
    assert len(m.targets) > 0
