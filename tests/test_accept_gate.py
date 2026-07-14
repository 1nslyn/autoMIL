"""Ladder keep-margin (``accept_margin`` δ) gate — val-firewall.

δ=0.0 (the default) reproduces strict composite dominance; δ>0 requires a child
to beat its parent's validation composite by more than the margin before it is
kept — a Ladder-style guard against promoting within-noise improvements over a
long agentic search. The gate shares one predicate (``_accept``) across every
keep/discard site (terminal writer, descendant re-eval, both reconcile paths).
"""
from __future__ import annotations

import pytest

from automil.graph import ExperimentGraph, _accept, _accept_margin


# --- pure predicate --------------------------------------------------------

def test_accept_strict_dominance_at_zero_margin():
    assert _accept(0.81, 0.80) is True
    assert _accept(0.80, 0.80) is False   # a tie is not kept
    assert _accept(0.79, 0.80) is False


def test_accept_margin_gates_within_noise_gain():
    assert _accept(0.81, 0.80, margin=0.02) is False   # +0.01 < δ
    assert _accept(0.82, 0.80, margin=0.02) is False   # exactly parent+δ, not strictly >
    assert _accept(0.83, 0.80, margin=0.02) is True    # +0.03 > δ


def test_accept_margin_reader_defaults_and_safety():
    assert _accept_margin(None) == 0.0
    assert _accept_margin({}) == 0.0
    assert _accept_margin({"scoring": {}}) == 0.0
    assert _accept_margin({"scoring": {"accept_margin": 0.05}}) == 0.05
    assert _accept_margin({"scoring": {"accept_margin": "bad"}}) == 0.0   # malformed → safe


# --- integration through keep/discard --------------------------------------

@pytest.fixture
def graph(tmp_path):
    return ExperimentGraph(tmp_path / "graph.json")


def test_scoring_default_accept_margin_is_zero(graph):
    assert graph.meta["scoring"]["accept_margin"] == 0.0


def test_default_margin_keeps_any_improvement(graph):
    parent = graph.add_executed(parent_id=None, description="p", techniques=[],
                                status="keep", metrics={"composite": 0.80})
    child = graph.add_executed(parent_id=parent, description="c", techniques=[],
                               status="keep", metrics={"composite": 0.81})
    graph._reevaluate_descendants(parent)
    assert graph.get_node(child)["status"] == "keep"   # +0.01, δ=0 → keep


def test_margin_discards_within_noise_child(graph):
    graph.meta["scoring"]["accept_margin"] = 0.05
    parent = graph.add_executed(parent_id=None, description="p", techniques=[],
                                status="keep", metrics={"composite": 0.80})
    child = graph.add_executed(parent_id=parent, description="c", techniques=[],
                               status="keep", metrics={"composite": 0.82})
    graph._reevaluate_descendants(parent)
    assert graph.get_node(child)["status"] == "discard"   # +0.02 < δ=0.05 → discard


# --- config wiring: predeclare δ in config.yaml -----------------------------

def test_accept_margin_seeded_from_sibling_config(tmp_path):
    """A predeclared scoring.accept_margin in config.yaml seeds a fresh graph."""
    (tmp_path / "config.yaml").write_text("scoring:\n  accept_margin: 0.03\n")
    g = ExperimentGraph(tmp_path / "graph.json")
    assert g.meta["scoring"]["accept_margin"] == 0.03


def test_accept_margin_defaults_zero_without_config(tmp_path):
    g = ExperimentGraph(tmp_path / "graph.json")   # no sibling config.yaml
    assert g.meta["scoring"]["accept_margin"] == 0.0


def test_persisted_margin_wins_over_config(tmp_path):
    """Once persisted in graph.json, meta.scoring.accept_margin is fixed."""
    import json
    (tmp_path / "config.yaml").write_text("scoring:\n  accept_margin: 0.03\n")
    (tmp_path / "graph.json").write_text(json.dumps({
        "schema_version": 2,
        "meta": {"scoring": {"exploration_weight": 0.005, "novelty_weight": 0.003,
                             "accept_margin": 0.09}},
        "nodes": {}, "technique_stats": {},
    }))
    g = ExperimentGraph(tmp_path / "graph.json")
    assert g.meta["scoring"]["accept_margin"] == 0.09


# --- robustness: negative / null margin (review M2, M3) ---------------------

def test_accept_margin_clamps_negative_and_handles_null_scoring():
    # M2: a negative margin would invert the gate (keep a worse child); clamp to 0.0.
    assert _accept_margin({"scoring": {"accept_margin": -0.05}}) == 0.0
    # M3: scoring present-but-null, or accept_margin null, must not crash.
    assert _accept_margin({"scoring": None}) == 0.0
    assert _accept_margin({"scoring": {"accept_margin": None}}) == 0.0


def test_graph_survives_null_scoring(tmp_path):
    """M3: a hand-corrupted meta.scoring=null must not crash __init__."""
    import json
    (tmp_path / "graph.json").write_text(json.dumps({
        "schema_version": 2, "meta": {"scoring": None}, "nodes": {}, "technique_stats": {},
    }))
    g = ExperimentGraph(tmp_path / "graph.json")   # must not raise
    assert g.meta["scoring"]["accept_margin"] == 0.0


def test_negative_config_margin_clamped(tmp_path):
    """M2: a negative scoring.accept_margin in config is clamped, never persisted negative."""
    (tmp_path / "config.yaml").write_text("scoring:\n  accept_margin: -0.1\n")
    g = ExperimentGraph(tmp_path / "graph.json")
    assert g.meta["scoring"]["accept_margin"] == 0.0
