"""M-1 + M-4 (audit 2026-07-23) graph robustness:

- M-1: a legacy/hand-edited ``meta.scoring`` block missing exploration_weight /
  novelty_weight must NOT KeyError in recalculate_scores() (which silently turned
  every reconcile() into a no-op).
- M-4: ``lineage`` and ``_reevaluate_descendants`` must terminate on a parent/child
  cycle instead of hanging the daemon completion path.
"""
from __future__ import annotations

import contextlib
import json
import signal

import pytest

from automil.graph import ExperimentGraph


@contextlib.contextmanager
def _time_limit(seconds: float):
    """Fail fast (instead of hanging) if a cycle guard regresses."""
    def _handler(signum, frame):
        raise TimeoutError("operation exceeded time limit — cycle hang?")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _write_graph(path, nodes, scoring=None):
    meta = {
        "best_composite": 0.0, "best_node_id": None,
        "total_executed": len(nodes), "total_proposed": 0, "next_id": 99,
        "baseline_composite": 0.0,
    }
    if scoring is not None:
        meta["scoring"] = scoring
    path.write_text(json.dumps({
        "schema_version": 2, "meta": meta, "nodes": nodes, "technique_stats": {},
    }))
    return ExperimentGraph(path=str(path))


def test_recalculate_scores_backfills_missing_scoring_weights(tmp_path):
    # scoring block present but missing exploration_weight / novelty_weight.
    g = _write_graph(
        tmp_path / "graph.json",
        nodes={
            "node_0001": {"id": "node_0001", "type": "executed", "status": "keep",
                          "composite": 0.8, "parent_id": None, "metrics": {}, "techniques": []},
            "node_0002": {"id": "node_0002", "type": "proposed", "status": "pending",
                          "composite": 0.0, "parent_id": "node_0001", "metrics": {}, "techniques": []},
        },
        scoring={"accept_margin": 0.0},
    )
    assert "exploration_weight" in g.meta["scoring"]
    assert "novelty_weight" in g.meta["scoring"]
    g.recalculate_scores()  # previously raised KeyError -> silent reconcile no-op


def test_lineage_and_reeval_terminate_on_cycle(tmp_path):
    g = _write_graph(
        tmp_path / "graph.json",
        nodes={
            "A": {"id": "A", "type": "executed", "status": "keep", "composite": 0.8,
                  "parent_id": "B", "metrics": {}, "techniques": []},
            "B": {"id": "B", "type": "executed", "status": "keep", "composite": 0.7,
                  "parent_id": "A", "metrics": {}, "techniques": []},
        },
    )
    with _time_limit(3.0):
        lin = g.lineage("A")
        g._reevaluate_descendants("A")
    # Cycle is broken by the visited guard — every node appears at most once.
    ids = [n["id"] for n in lin]
    assert len(ids) == len(set(ids))
