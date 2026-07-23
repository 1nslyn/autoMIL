"""H-6 (audit 2026-07-23): meta.best_node_id must never point to a discarded
node. The old inline ``composite > best`` updates were status-agnostic and ran
before ``_reevaluate_descendants`` could flip a child to discard, so best could
be left on a discarded node (reported by status/viz; a certify target).

Best is now recomputed from keep nodes only after any status mutation.
"""
from __future__ import annotations

import pytest

from automil.graph import ExperimentGraph


def _fresh_graph(tmp_path, margin=0.0):
    g = ExperimentGraph(path=str(tmp_path / "graph.json"))
    g.meta.setdefault("scoring", {})["accept_margin"] = margin
    return g


def test_promote_discard_status_never_becomes_best(tmp_path):
    g = _fresh_graph(tmp_path)
    root = g.add_proposed("root", "root", [], kind="hp")
    g.promote(root, {"composite": 0.80, "status": "keep"})
    assert g.meta["best_node_id"] == root

    child = g.add_proposed(root, "higher-but-discarded", [], kind="hp")
    # Higher composite but explicitly discarded — must NOT capture best.
    g.promote(child, {"composite": 0.90, "status": "discard"})

    assert g.meta["best_node_id"] == root
    assert g.meta["best_composite"] == pytest.approx(0.80)


def test_descendant_flipped_to_discard_loses_best(tmp_path):
    """The reproduced trigger: a child promoted before its parent is kept and
    becomes best; when the parent later completes, the Ladder margin flips the
    child to discard — best must move off it."""
    g = _fresh_graph(tmp_path, margin=0.05)
    parent = g.add_proposed("root", "parent", [], kind="hp")
    child = g.add_proposed(parent, "within-margin child", [], kind="hp")

    # Child promoted first: parent composite defaults to 0, so 0.82 is kept and
    # becomes best.
    g.promote(child, {"composite": 0.82, "status": "keep"})
    assert g.meta["best_node_id"] == child
    assert g.meta["best_composite"] == pytest.approx(0.82)

    # Parent completes at 0.80. δ=0.05 → child (0.82) is within margin of parent
    # (needs > 0.85) → _reevaluate_descendants flips it to discard.
    g.promote(parent, {"composite": 0.80, "status": "keep"})

    assert g.nodes[child]["status"] == "discard"
    # Best must now be the parent (the only keep node), not the discarded child.
    assert g.meta["best_node_id"] == parent
    assert g.meta["best_composite"] == pytest.approx(0.80)
