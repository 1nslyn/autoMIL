"""End-to-end reconcile + descendant cascade tests (CAP-04 / D-123, D-124).

Uses a REAL ExperimentGraph backed by tmp_path/graph.json — NOT mocks —
to prove that partial primary values flow through the Pareto keep/discard logic
correctly (Fragile Invariant #6 defence).

Tests:
    test_reconcile_budget_kill_writes_partial_result_json
        reconcile_budget_kill with 3/5 folds → result.json w/ budget_killed
    test_reconcile_budget_kill_zero_folds_writes_crashed_result_json
        reconcile_budget_kill with 0 folds → status='crash', primary_value=0.0 (D-06)
    test_descendant_cascade_against_partial_primary_value_keeps_better_descendant
        descendant that beats partial primary_value 0.82 stays 'keep'
    test_descendant_cascade_against_partial_primary_value_discards_worse_descendant
        descendant that loses to partial primary_value 0.82 flips to 'discard'
    test_reconcile_logs_summary_at_info_level
        reconcile_budget_kill emits an INFO log with partial/2/ marker
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from automil.cells.reconcile import aggregate_folds, reconcile_budget_kill
from automil.graph import ExperimentGraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_fold(
    node_archive: Path,
    idx: int,
    primary_value: float = 0.80,
    fold_count: int = 5,
) -> None:
    """Write a well-formed fold_<idx>_result.json into node_archive/certify/ (Scope B).

    Born-sealed: reconcile_budget_kill aggregates from archive/<node>/certify/."""
    sealed = node_archive / "certify"
    sealed.mkdir(parents=True, exist_ok=True)
    payload = {
        "fold_index": idx,
        "fold_count": fold_count,
        "status": "completed",
        "metrics": {
            "val_auc": primary_value,
            "val_bacc": primary_value,
            "test_auc": primary_value,
            "test_bacc": primary_value,
        },
        "primary_value": primary_value,
        "elapsed_seconds": 1,
        "peak_vram_mb": 1000,
    }
    (sealed / f"fold_{idx}_result.json").write_text(json.dumps(payload))


# ---------------------------------------------------------------------------
# Test 1 — partial reconcile writes budget_killed result.json
# ---------------------------------------------------------------------------


def test_reconcile_budget_kill_writes_partial_result_json(tmp_path: Path):
    """3/5 folds present → result.json with status='partial', primary_value≈0.82,
    metadata.budget_killed=True.
    """
    node_id = "node_partial_test"
    archive_dir = tmp_path / "archive"
    node_archive = archive_dir / node_id
    node_archive.mkdir(parents=True)

    for i, comp in enumerate([0.80, 0.82, 0.84]):
        _write_fold(node_archive, i, primary_value=comp)

    mock_graph = MagicMock()
    payload = reconcile_budget_kill(
        node_id=node_id,
        archive_dir=archive_dir,
        graph=mock_graph,
        expected_fold_count=5,
    )

    # Verify payload returned
    assert payload["status"] == "partial"
    assert payload["partial_folds"] == 3
    assert abs(payload["primary_value"] - 0.82) < 0.01
    assert payload.get("metadata", {}).get("budget_killed") is True
    # D-10 (REC-02): archive result.json is written solely by terminal_writer, not here.
    # reconcile_budget_kill only returns the payload; the file is written downstream.


# ---------------------------------------------------------------------------
# Test 2 — zero folds → crash result.json with budget_killed (D-06: was 'crashed')
# ---------------------------------------------------------------------------


def test_reconcile_budget_kill_zero_folds_writes_crashed_result_json(tmp_path: Path):
    """Empty archive (0 fold files) → result.json status='crash', primary_value=0.0,
    metadata.budget_killed=True.

    D-06 (REC-03): canonical status is 'crash', not 'crashed' (drift value removed).
    """
    node_id = "node_zero_folds"
    archive_dir = tmp_path / "archive"
    node_archive = archive_dir / node_id
    node_archive.mkdir(parents=True)  # exists but empty

    mock_graph = MagicMock()
    payload = reconcile_budget_kill(
        node_id=node_id,
        archive_dir=archive_dir,
        graph=mock_graph,
        expected_fold_count=5,
    )

    assert payload["status"] == "crash"  # D-06: was 'crashed' pre-v1.1
    assert payload["primary_value"] == 0.0
    assert payload["partial_folds"] == 0
    assert payload.get("metadata", {}).get("budget_killed") is True
    # D-10 (REC-02): archive result.json is written solely by terminal_writer, not here.
    # reconcile_budget_kill only returns the payload; the file is written downstream.


# ---------------------------------------------------------------------------
# Test 3 — descendant cascade keeps better descendant
# ---------------------------------------------------------------------------


def test_descendant_cascade_against_partial_primary_value_keeps_better_descendant(
    tmp_path: Path,
):
    """Descendant with primary_value=0.85 > partial primary_value 0.82 stays 'keep'.

    Verifies Fragile Invariant #6: the cascade operates on the partial primary_value
    (0.82), not zero.  If a descendant beats the partial primary_value it should
    be promoted/kept, proving downstream experiments inherit a useful baseline.
    """
    eg = ExperimentGraph(tmp_path / "graph.json")

    parent_nid = eg.add_executed(
        parent_id=None, description="parent", techniques=[],
        metrics={"primary_value": 0.50, "test_auc": 0.50, "test_bacc": 0.50},
        status="keep",
    )
    capkill_nid = eg.add_executed(
        parent_id=parent_nid, description="cap-killed partial 0.82", techniques=[],
        metrics={"primary_value": 0.82, "test_auc": 0.82, "test_bacc": 0.82},
        status="keep",
    )
    # Descendant beats partial primary_value on all three axes: primary_value, AUC, BACC
    better_nid = eg.add_executed(
        parent_id=capkill_nid, description="better than partial 0.85", techniques=[],
        metrics={"primary_value": 0.85, "test_auc": 0.85, "test_bacc": 0.85},
        status="keep",
    )

    eg._reevaluate_descendants(capkill_nid)

    assert eg.get_node(better_nid)["status"] == "keep", (
        "Descendant with primary_value 0.85 > partial primary_value 0.82 must remain 'keep' "
        "after cascade. If it was discarded, the cascade did not use the partial primary_value."
    )


# ---------------------------------------------------------------------------
# Test 4 — descendant cascade discards worse descendant
# ---------------------------------------------------------------------------


def test_descendant_cascade_against_partial_primary_value_discards_worse_descendant(
    tmp_path: Path,
):
    """Descendant with primary_value=0.70 < partial primary_value 0.82 flips to 'discard'.

    This is the definitive Fragile Invariant #6 proof: 0.70 beats zero (so if
    the cascade ran against 0.0 it would falsely stay 'keep'), but loses to 0.82
    (so if the cascade ran against the correct partial primary_value it flips to
    'discard').  Any result other than 'discard' means the cascade is broken.
    """
    eg = ExperimentGraph(tmp_path / "graph.json")

    parent_nid = eg.add_executed(
        parent_id=None, description="parent", techniques=[],
        metrics={"primary_value": 0.50, "test_auc": 0.50, "test_bacc": 0.50},
        status="keep",
    )
    capkill_nid = eg.add_executed(
        parent_id=parent_nid, description="cap-killed partial 0.82", techniques=[],
        metrics={"primary_value": 0.82, "test_auc": 0.82, "test_bacc": 0.82},
        status="keep",
    )
    # 0.70 < 0.82 → should be discarded against partial primary_value
    # 0.70 > 0.00 → would be kept if cascade ran against zero (Fragile Invariant #6 break)
    worse_nid = eg.add_executed(
        parent_id=capkill_nid,
        description="worse than partial 0.70, beats zero",
        techniques=[],
        metrics={"primary_value": 0.70, "test_auc": 0.70, "test_bacc": 0.70},
        status="keep",
    )

    eg._reevaluate_descendants(capkill_nid)

    assert eg.get_node(worse_nid)["status"] == "discard", (
        "Descendant with primary_value 0.70 < partial primary_value 0.82 must be 'discard'. "
        "If it stayed 'keep', the cascade ran against zero — Fragile Invariant #6 broken."
    )


# ---------------------------------------------------------------------------
# Test 5 — reconcile logs at INFO with partial/fold-count marker
# ---------------------------------------------------------------------------


def test_reconcile_logs_summary_at_info_level(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """reconcile_budget_kill emits an INFO log containing 'partial' and '2/' (fold marker).

    Verifies that operators can trace budget-killed events in the daemon logs
    without having to grep result.json files manually.
    """
    node_id = "node_log_test"
    archive_dir = tmp_path / "archive"
    node_archive = archive_dir / node_id
    node_archive.mkdir(parents=True)

    # Write 2 fold files
    for i, comp in enumerate([0.80, 0.82]):
        _write_fold(node_archive, i, primary_value=comp)

    mock_graph = MagicMock()
    with caplog.at_level(logging.INFO, logger="automil.cells.reconcile"):
        reconcile_budget_kill(
            node_id=node_id,
            archive_dir=archive_dir,
            graph=mock_graph,
            expected_fold_count=5,
        )

    log_messages = [r.message for r in caplog.records]
    assert any("partial" in m.lower() for m in log_messages), (
        f"Expected a log record containing 'partial'. Got: {log_messages}"
    )
    assert any("2/" in m for m in log_messages), (
        f"Expected a log record containing '2/' (partial_folds=2 marker). "
        f"Got: {log_messages}"
    )
