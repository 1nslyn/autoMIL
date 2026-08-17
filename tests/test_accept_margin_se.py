"""CR-4 (audit 2026-07-23): the Ladder keep-margin must be derived from measured
cross-fold noise, not left at a bare predeclared constant.

Before CR-4 the keep/discard gate was ``child > parent + δ`` with δ a global
config constant defaulting to 0.0, and **nothing anywhere measured the noise δ
was supposed to exceed**. On the ~10-val-patient cohorts (CPTAC-GBM n=99,
CPTAC-PDAC n=105) a discovery sweep screening ~60 candidates keeps the maximum
of 60 draws from that noise distribution — an apparent lift that does not
survive to the sealed test block.

CR-4 emits the cross-fold SE of the primary_value (``primary_se``) alongside it and
makes the margin actually applied ``max(predeclared δ, se_multiplier × parent SE)``.
Two invariants the tests below pin down:

  * **monotone** — the noise floor can only ever RAISE the bar. A campaign that
    predeclared δ=0.05 must never silently drop to 0.01 because one parent
    happened to have a tight CV.
  * **fail-closed on absence** — a parent with no SE (legacy node, or a partial
    run with <2 valid folds) falls back to the predeclared δ, never to 0.0.
"""
from __future__ import annotations

import json

import pytest

from automil.graph import (
    DEFAULT_SE_MULTIPLIER,
    ExperimentGraph,
    _accept,
    _accept_margin,
    _se_multiplier,
    effective_accept_margin,
    node_primary_se,
)
from automil.scoring import cross_fold_se


# --- cross_fold_se: the measurement ----------------------------------------

def test_cross_fold_se_matches_hand_computed_sem():
    # ddof=1 sample SD over {0.70,0.72,0.68,0.71,0.69} = 0.0158113883...
    # SE = SD / sqrt(5) = 0.00707106781...
    se = cross_fold_se([0.70, 0.72, 0.68, 0.71, 0.69])
    assert se == pytest.approx(0.0070710678, abs=1e-9)


def test_cross_fold_se_is_none_below_two_finite_folds():
    """Not estimable is None, never 0.0 — 0.0 reads as 'measured, noise-free'."""
    assert cross_fold_se([]) is None
    assert cross_fold_se([0.70]) is None
    assert cross_fold_se([0.70, float("nan"), float("inf")]) is None


def test_cross_fold_se_drops_non_finite_and_non_numeric_folds():
    assert cross_fold_se([0.70, float("nan"), 0.72]) == pytest.approx(0.01, abs=1e-9)
    assert cross_fold_se([0.70, None, "x", True, 0.72]) == pytest.approx(0.01, abs=1e-9)


def test_cross_fold_se_of_identical_folds_is_zero_not_none():
    """Zero spread IS a measurement (degenerate CV); distinct from unmeasurable."""
    assert cross_fold_se([0.70, 0.70, 0.70]) == 0.0


# --- node_primary_se / _se_multiplier: the readers ------------------------

def test_node_primary_value_se_reads_top_level_key():
    assert node_primary_se({"primary_se": 0.04}) == 0.04


def test_node_primary_value_se_is_none_for_legacy_or_degenerate_nodes():
    assert node_primary_se(None) is None
    assert node_primary_se({}) is None                       # legacy node
    assert node_primary_se({"primary_se": None}) is None   # <2 valid folds
    assert node_primary_se({"primary_se": "bad"}) is None
    assert node_primary_se({"primary_se": float("nan")}) is None
    assert node_primary_se({"primary_se": -0.01}) is None  # nonsense → unmeasured


def test_se_multiplier_defaults_and_safety():
    assert _se_multiplier(None) == DEFAULT_SE_MULTIPLIER
    assert _se_multiplier({}) == DEFAULT_SE_MULTIPLIER
    assert _se_multiplier({"scoring": None}) == DEFAULT_SE_MULTIPLIER
    assert _se_multiplier({"scoring": {"se_multiplier": None}}) == DEFAULT_SE_MULTIPLIER
    assert _se_multiplier({"scoring": {"se_multiplier": "bad"}}) == DEFAULT_SE_MULTIPLIER
    assert _se_multiplier({"scoring": {"se_multiplier": 2.0}}) == 2.0
    # A negative multiplier would turn the noise floor into a discount.
    assert _se_multiplier({"scoring": {"se_multiplier": -1.0}}) == 0.0


# --- effective_accept_margin: the derivation --------------------------------

def test_effective_margin_raises_predeclared_delta_to_the_noise_floor():
    meta = {"scoring": {"accept_margin": 0.015, "se_multiplier": 1.0}}
    assert effective_accept_margin(meta, {"primary_se": 0.04}) == pytest.approx(0.04)


def test_effective_margin_is_monotone_never_below_predeclared_delta():
    """A tight parent CV must NOT relax a predeclared campaign margin."""
    meta = {"scoring": {"accept_margin": 0.05, "se_multiplier": 1.0}}
    assert effective_accept_margin(meta, {"primary_se": 0.002}) == pytest.approx(0.05)
    assert effective_accept_margin(meta, {"primary_se": 0.0}) == pytest.approx(0.05)


def test_effective_margin_falls_back_to_delta_when_se_unmeasured():
    """Legacy / partial parent → predeclared δ, never 0.0-by-accident."""
    meta = {"scoring": {"accept_margin": 0.05, "se_multiplier": 1.0}}
    assert effective_accept_margin(meta, {}) == pytest.approx(0.05)
    assert effective_accept_margin(meta, {"primary_se": None}) == pytest.approx(0.05)
    assert effective_accept_margin(meta, None) == pytest.approx(0.05)


def test_effective_margin_scales_with_se_multiplier():
    meta = {"scoring": {"accept_margin": 0.0, "se_multiplier": 2.0}}
    assert effective_accept_margin(meta, {"primary_se": 0.03}) == pytest.approx(0.06)
    meta_off = {"scoring": {"accept_margin": 0.0, "se_multiplier": 0.0}}
    assert effective_accept_margin(meta_off, {"primary_se": 0.03}) == 0.0


def test_effective_margin_defaults_to_one_se_without_config():
    """se_multiplier absent → 1.0 (one SE), not 0 (feature silently off)."""
    assert effective_accept_margin({"scoring": {"accept_margin": 0.0}},
                                   {"primary_se": 0.03}) == pytest.approx(0.03)


# --- integration: keep/discard actually moves --------------------------------

@pytest.fixture
def graph(tmp_path):
    return ExperimentGraph(tmp_path / "graph.json")


def _executed(g, parent_id, primary_value, primary_se=None, status="keep"):
    metrics = {"primary_value": primary_value}
    if primary_se is not None:
        metrics["primary_se"] = primary_se
    return g.add_executed(parent_id=parent_id, description="x", techniques=[],
                          status=status, metrics=metrics)


def test_add_executed_records_primary_value_se_top_level(graph):
    nid = _executed(graph, None, 0.80, primary_se=0.04)
    assert graph.get_node(nid)["primary_se"] == 0.04


def test_add_executed_records_none_when_se_absent(graph):
    nid = _executed(graph, None, 0.80)
    assert graph.get_node(nid)["primary_se"] is None


def test_noise_floor_discards_a_within_se_child(graph):
    """THE DEFECT: at δ=0.0 a +0.02 child on a ±0.04-SE parent was kept."""
    parent = _executed(graph, None, 0.80, primary_se=0.04)
    child = _executed(graph, parent, 0.82, primary_se=0.04)
    graph._reevaluate_descendants(parent)
    assert graph.get_node(child)["status"] == "discard"   # +0.02 < 1 SE = 0.04


def test_noise_floor_still_keeps_a_beyond_se_child(graph):
    parent = _executed(graph, None, 0.80, primary_se=0.04)
    child = _executed(graph, parent, 0.86, primary_se=0.04)
    graph._reevaluate_descendants(parent)
    assert graph.get_node(child)["status"] == "keep"      # +0.06 > 1 SE = 0.04


def test_legacy_parent_without_se_keeps_prior_behaviour(graph):
    """Nodes already on disk carry no primary_se → gate is unchanged (δ)."""
    parent = _executed(graph, None, 0.80)
    child = _executed(graph, parent, 0.81)
    graph._reevaluate_descendants(parent)
    assert graph.get_node(child)["status"] == "keep"      # δ=0.0, as before CR-4


def test_child_se_cannot_lower_its_own_bar(graph):
    """The bar is a property of the INCUMBENT.

    If the margin were derived from the child's own SE, the argmax over ~60
    screened candidates would also be an argmin over their margins — selecting
    on the gate itself. Pin the bar to the parent.
    """
    parent = _executed(graph, None, 0.80, primary_se=0.04)
    child = _executed(graph, parent, 0.82, primary_se=0.0001)
    graph._reevaluate_descendants(parent)
    assert graph.get_node(child)["status"] == "discard"


def test_predeclared_delta_wins_over_a_tight_parent_cv(graph):
    graph.meta["scoring"]["accept_margin"] = 0.05
    parent = _executed(graph, None, 0.80, primary_se=0.001)
    child = _executed(graph, parent, 0.83, primary_se=0.001)
    graph._reevaluate_descendants(parent)
    assert graph.get_node(child)["status"] == "discard"   # +0.03 < δ=0.05


# --- config wiring: se_multiplier is predeclared, then frozen ----------------

def test_se_multiplier_seeded_from_sibling_config(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "scoring:\n  accept_margin: 0.015\n  se_multiplier: 2.0\n")
    g = ExperimentGraph(tmp_path / "graph.json")
    assert g.meta["scoring"]["se_multiplier"] == 2.0


def test_se_multiplier_defaults_to_one_without_config(tmp_path):
    g = ExperimentGraph(tmp_path / "graph.json")
    assert g.meta["scoring"]["se_multiplier"] == DEFAULT_SE_MULTIPLIER


def test_persisted_se_multiplier_wins_over_config(tmp_path):
    """Pre-registration: once in graph.json, the stored value is authoritative."""
    (tmp_path / "config.yaml").write_text("scoring:\n  se_multiplier: 2.0\n")
    (tmp_path / "graph.json").write_text(json.dumps({
        "schema_version": 2,
        "meta": {"scoring": {"exploration_weight": 0.005, "novelty_weight": 0.003,
                             "accept_margin": 0.05, "se_multiplier": 0.5}},
        "nodes": {}, "technique_stats": {},
    }))
    g = ExperimentGraph(tmp_path / "graph.json")
    assert g.meta["scoring"]["se_multiplier"] == 0.5


def test_negative_config_se_multiplier_clamped(tmp_path):
    (tmp_path / "config.yaml").write_text("scoring:\n  se_multiplier: -3\n")
    g = ExperimentGraph(tmp_path / "graph.json")
    assert g.meta["scoring"]["se_multiplier"] == 0.0


def test_legacy_graph_backfills_se_multiplier(tmp_path):
    """A graph written before CR-4 gains the key on load (M-1 backfill contract)."""
    (tmp_path / "graph.json").write_text(json.dumps({
        "schema_version": 2,
        "meta": {"scoring": {"exploration_weight": 0.005, "novelty_weight": 0.003,
                             "accept_margin": 0.05}},
        "nodes": {}, "technique_stats": {},
    }))
    g = ExperimentGraph(tmp_path / "graph.json")
    assert g.meta["scoring"]["se_multiplier"] == DEFAULT_SE_MULTIPLIER


# --- the predicate itself is untouched --------------------------------------

def test_accept_predicate_and_predeclared_reader_unchanged():
    """CR-4 changes WHICH margin is passed, not the predicate or the δ reader."""
    assert _accept(0.83, 0.80, margin=0.02) is True
    assert _accept_margin({"scoring": {"accept_margin": 0.05}}) == 0.05
