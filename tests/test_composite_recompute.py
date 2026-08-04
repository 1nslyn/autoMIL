"""CR-1b (audit 2026-07-23): the selection composite is derived from the declared
VALIDATION metrics, not trusted verbatim from the agent-written result.json.

Before this, a training script whose ``composite`` came from the sealed test block
produced a schema-valid result, the graph selected on it, and nothing detected the
leak — defeating the val-firewall's central claim.
"""
from __future__ import annotations

import pytest

from automil.graph import ExperimentGraph
from automil.scoring import (
    DEFAULT_FORMULA,
    composite_disagrees,
    recompute_composite,
)
from automil.terminal_writer import write_terminal_state


# --- reducer unit tests ---

def test_mean_reproduces_classification_composite():
    # The established formula is (val_auc + val_bacc) / 2 — 'mean' reproduces it.
    m = {"val_auc": 0.90, "val_bacc": 0.80}
    assert recompute_composite(m, "mean") == pytest.approx(0.85)


def test_mean_reproduces_survival_composite():
    # Survival composite is the lone val_c_index — mean of one value is itself.
    assert recompute_composite({"val_c_index": 0.62}, "mean") == pytest.approx(0.62)


def test_opt_out_returns_none():
    assert recompute_composite({"val_auc": 0.9}, "trust_reported") is None


def test_empty_metrics_returns_none():
    assert recompute_composite({}, DEFAULT_FORMULA) is None
    assert recompute_composite({"note": "text"}, DEFAULT_FORMULA) is None


def test_booleans_excluded():
    # bool is an int subclass — must not be averaged in as 1.0/0.0.
    assert recompute_composite({"val_auc": 0.8, "ok": True}, "mean") == pytest.approx(0.8)


def test_unknown_formula_raises():
    with pytest.raises(ValueError):
        recompute_composite({"val_auc": 0.8}, "bogus")


def test_disagreement_tolerates_4dp_rounding():
    # result.json rounds composite and metrics to 4dp; that must not trip the guard.
    assert not composite_disagrees(0.8500, 0.85004)
    assert composite_disagrees(0.8500, 0.9500)


# --- end-to-end through terminal_writer ---

def _run(tmp_path, result):
    adir = tmp_path / "automil"
    adir.mkdir()
    graph = ExperimentGraph(path=str(adir / "graph.json"))
    nid = graph.add_proposed("root", "exp", [], kind="hp")
    graph.save()

    completed = adir / "orchestrator" / "completed"
    archive = adir / "orchestrator" / "archive" / nid
    completed.mkdir(parents=True)
    archive.mkdir(parents=True)

    write_terminal_state(
        node_id=nid, result=result, graph=graph,
        completed_dir=completed, archive_dir=archive,
        results_tsv_writer=lambda *a, **k: None,
        spec={}, elapsed_s=1.0, gpu_id=0,
    )
    return nid, ExperimentGraph(path=str(adir / "graph.json"))


def test_test_derived_composite_is_overridden_by_val(tmp_path):
    """The exploit: composite reported from the sealed test block."""
    nid, g = _run(tmp_path, {
        "status": "completed",
        "composite": 0.99,                                  # from test — a lie
        "metrics": {"val_auc": 0.70, "val_bacc": 0.60},     # true val → 0.65
        "held_out": {"test_auc": 0.99},
    })
    node = g.get_node(nid)
    assert node["composite"] == pytest.approx(0.65)
    assert node["metadata"]["composite_disagreement"]["reported"] == pytest.approx(0.99)


def test_honest_composite_passes_through_unflagged(tmp_path):
    nid, g = _run(tmp_path, {
        "status": "completed",
        "composite": 0.85,
        "metrics": {"val_auc": 0.90, "val_bacc": 0.80},
        "held_out": {"test_auc": 0.5},
    })
    node = g.get_node(nid)
    assert node["composite"] == pytest.approx(0.85)
    assert "composite_disagreement" not in (node.get("metadata") or {})
