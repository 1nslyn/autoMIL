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
