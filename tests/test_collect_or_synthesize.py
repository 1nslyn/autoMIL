"""REC-01 D-03, REC-03 D-05/D-06: _collect_or_synthesize_result fold-first and status canonicalization.

D-03: _collect_or_synthesize_result must try archive fold aggregation BEFORE
      synthesizing a timeout/crash result from log heuristics.
D-05/D-06: OOM/timeout synthesis must produce status="crash" + termination_reason,
           not status="oom" or status="timeout".
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _write_fold(archive: Path, idx: int, composite: float = 0.80) -> None:
    """Write a well-formed fold result JSON into archive/certify/ (born-sealed, Scope B).

    The orchestrator points AUTOMIL_RESULTS_DIR at archive/<node>/certify/, so
    per-fold artifacts live there and _collect_or_synthesize_result reads them from
    certify/ (never the agent-visible node-archive root)."""
    sealed = archive / "certify"
    sealed.mkdir(parents=True, exist_ok=True)
    payload = {
        "fold_index": idx,
        "fold_count": 5,
        "status": "completed",
        "composite": composite,
        "metrics": {"val_auc": composite},
        "elapsed_seconds": 100,
        "peak_vram_mb": 4000,
    }
    (sealed / f"fold_{idx}_result.json").write_text(json.dumps(payload))


def _make_daemon_stub(tmp_path: Path, timed_out: dict | None = None):
    """Build a minimal ExperimentOrchestrator stub for unit-testing _collect_or_synthesize_result."""
    from automil.backends._orchestrator_daemon import ExperimentOrchestrator

    daemon = MagicMock(spec=ExperimentOrchestrator)
    daemon._timed_out = timed_out or {}
    daemon._read_fold_count_for_node = MagicMock(return_value=5)
    daemon.config = {}
    # runner.collect_result returns None to force the synthesis path
    daemon.runner = MagicMock()
    daemon.runner.collect_result = MagicMock(return_value=None)
    return daemon


def test_fold_aggregation_tried_before_synthesis(tmp_path: Path) -> None:
    """D-03: when archive has fold files but no result.json, _collect_or_synthesize_result
    returns a partial dict (not crash/0.0) — fold aggregation runs before log synthesis.

    RED until D-03 fold-first insertion ships in Plan 04.
    """
    archive = tmp_path / "archive" / "node_0001"
    archive.mkdir(parents=True)

    # Write 3 of 5 folds — no result.json, no run.log
    _write_fold(archive, 0, composite=0.80)
    _write_fold(archive, 1, composite=0.82)
    _write_fold(archive, 2, composite=0.84)

    from automil.backends._orchestrator_daemon import ExperimentOrchestrator

    daemon = _make_daemon_stub(tmp_path)
    wt_path = tmp_path / "wt"
    wt_path.mkdir()

    result = ExperimentOrchestrator._collect_or_synthesize_result(
        daemon, node_id="node_0001", archive=archive, returncode=-15, wt_path=wt_path
    )

    # After D-03 fix: must aggregate folds, not return 0.0 crash
    assert result is not None, "result must not be None"
    assert result.get("composite", 0.0) != 0.0, (
        "D-03 not implemented: fold aggregation not tried before synthesis. "
        "composite is 0.0 but 3 folds exist."
    )
    assert result.get("status") in ("partial", "completed"), (
        f"Expected status=partial, got {result.get('status')!r}. "
        "Fold aggregation should yield partial, not crash."
    )


def test_oom_synthesis_produces_crash_status_with_reason(tmp_path: Path) -> None:
    """D-05/D-06: OOM log signal produces status='crash' + termination_reason='oom'.

    NOT status='oom' — that is not in the tight enum.
    RED until D-05/D-06 canonicalization ships in Plan 04.
    """
    archive = tmp_path / "archive" / "node_0001"
    archive.mkdir(parents=True)
    # Write an OOM log — no fold files, no result.json to force synthesis
    (archive / "run.log").write_text(
        "Training...\nCUDA out of memory. Tried to allocate 2.00 GiB...\nTraceback..."
    )

    from automil.backends._orchestrator_daemon import ExperimentOrchestrator

    daemon = _make_daemon_stub(tmp_path)
    wt_path = tmp_path / "wt"
    wt_path.mkdir()

    result = ExperimentOrchestrator._collect_or_synthesize_result(
        daemon, node_id="node_0001", archive=archive, returncode=1, wt_path=wt_path
    )

    assert result is not None
    assert result.get("status") == "crash", (
        f"D-05/D-06 not implemented: OOM synthesis produced status={result.get('status')!r} "
        "instead of 'crash'. 'oom' is not in the tight enum."
    )
    assert result.get("termination_reason") == "oom", (
        f"D-05 not implemented: expected termination_reason='oom', "
        f"got {result.get('termination_reason')!r}."
    )


def test_timeout_synthesis_produces_crash_status_with_reason(tmp_path: Path) -> None:
    """D-05/D-06: timeout signal produces status='crash' + termination_reason='timeout'.

    NOT status='timeout' — that is not in the tight enum.
    RED until D-05/D-06 canonicalization ships in Plan 04.
    """
    archive = tmp_path / "archive" / "node_0001"
    archive.mkdir(parents=True)
    # No fold files — force synthesis path; timed_out flag set

    from automil.backends._orchestrator_daemon import ExperimentOrchestrator

    daemon = _make_daemon_stub(tmp_path, timed_out={"node_0001": True})
    wt_path = tmp_path / "wt"
    wt_path.mkdir()

    result = ExperimentOrchestrator._collect_or_synthesize_result(
        daemon, node_id="node_0001", archive=archive, returncode=-9, wt_path=wt_path
    )

    assert result is not None
    assert result.get("status") == "crash", (
        f"D-05/D-06 not implemented: timeout synthesis produced status={result.get('status')!r} "
        "instead of 'crash'. 'timeout' is not in the tight enum."
    )
    assert result.get("termination_reason") == "timeout", (
        f"D-05 not implemented: expected termination_reason='timeout', "
        f"got {result.get('termination_reason')!r}."
    )
