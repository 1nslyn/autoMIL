"""Wave 0 stubs for D-168/D-169 running/ namespace migration (BCK-05/06)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _build_minimal_automil(tmp_path: Path) -> Path:
    automil_dir = tmp_path / "automil"
    (automil_dir / "orchestrator" / "running").mkdir(parents=True)
    (automil_dir / "orchestrator" / "queue").mkdir(parents=True)
    (automil_dir / "orchestrator" / "archive").mkdir(parents=True)
    (automil_dir / "orchestrator" / "completed").mkdir(parents=True)
    (automil_dir / "config.yaml").write_text("backend:\n  name: local\n")
    (tmp_path / ".git").mkdir(exist_ok=True)
    return automil_dir


def _write_agent_active_cell(automil_dir: Path) -> None:
    from automil.cells.state import Cell, CellStatus, write_cell

    write_cell(
        Cell(
            cell_id="cell-1",
            dataset="dataset",
            encoder="encoder",
            mil_model="model",
            started_at=1.0,
            budget_seconds=100,
            safety_buffer_seconds=10,
            status=CellStatus.ACTIVE,
            mode="agent_active",
        ),
        automil_dir / "cells",
    )


def test_running_dir_per_backend(tmp_path):
    """D-169: daemon resolves running_dir per backend via _backend_running_dir(name)."""
    automil_dir = _build_minimal_automil(tmp_path)
    from automil.backends._orchestrator_daemon import ExperimentOrchestrator
    daemon = ExperimentOrchestrator(project_root=tmp_path, automil_dir=automil_dir)
    assert daemon._backend_running_dir("local") == automil_dir / "orchestrator" / "running" / "local"
    assert daemon._backend_running_dir("slurm") == automil_dir / "orchestrator" / "running" / "slurm"
    assert daemon._backend_running_dir("ray") == automil_dir / "orchestrator" / "running" / "ray"


def test_daemon_refuses_flat_running(tmp_path):
    """D-168: daemon.run() raises SystemExit if flat running/*.json exists with no namespaced subdirs."""
    automil_dir = _build_minimal_automil(tmp_path)
    flat_running = automil_dir / "orchestrator" / "running"
    (flat_running / "stale_node.json").write_text(json.dumps({"id": "stale_node"}))
    # Confirm precondition: no namespaced subdirs.
    assert not (flat_running / "local").exists()

    from automil.backends._orchestrator_daemon import ExperimentOrchestrator
    daemon = ExperimentOrchestrator(project_root=tmp_path, automil_dir=automil_dir)
    with pytest.raises(SystemExit, match="BREAKING CHANGE"):
        daemon.run()


def test_agent_active_tick_scrapes_claude_native_metric(tmp_path, monkeypatch):
    """A journaled session, not mutable cap config, enables the native scrape."""
    automil_dir = _build_minimal_automil(tmp_path)
    from automil.cells.activity import record_hook_event

    record_hook_event(
        automil_dir,
        "cell-1",
        {
            "hook_event_name": "SessionStart",
            "session_id": "session-1",
            "source": "startup",
        },
    )
    _write_agent_active_cell(automil_dir)

    from automil import activity_metrics
    from automil.backends._orchestrator_daemon import ExperimentOrchestrator

    events = []
    observation = object()
    monkeypatch.setattr(
        activity_metrics,
        "observe_activity_metrics",
        lambda observed_dir: events.append(observed_dir) or observation,
    )
    daemon = ExperimentOrchestrator(
        project_root=tmp_path, automil_dir=automil_dir
    )
    monkeypatch.setattr(daemon, "_reload_orchestrator_config", lambda: None)
    monkeypatch.setattr(daemon, "_check_running", lambda: None)
    tick_observations = []
    monkeypatch.setattr(
        daemon,
        "_tick_cells",
        lambda **kwargs: tick_observations.append(kwargs["activity_observation"]),
    )
    monkeypatch.setattr(daemon, "_get_pending", lambda: [])
    monkeypatch.setattr(daemon, "_save_state", lambda: None)

    daemon.tick()

    assert events == [automil_dir]
    assert tick_observations == [observation]
    assert not hasattr(daemon, "_activity_refresh_error")


def test_agent_active_tick_fails_closed_when_metric_endpoint_disappears(
    tmp_path, monkeypatch
):
    """Endpoint loss is passed to accounting and never stored as budget state."""
    automil_dir = _build_minimal_automil(tmp_path)
    from automil.cells.activity import record_hook_event

    record_hook_event(
        automil_dir,
        "cell-1",
        {
            "hook_event_name": "SessionStart",
            "session_id": "session-1",
            "source": "startup",
        },
    )
    _write_agent_active_cell(automil_dir)

    from automil import activity_metrics
    from automil.backends._orchestrator_daemon import ExperimentOrchestrator

    unavailable = object()
    monkeypatch.setattr(
        activity_metrics,
        "observe_activity_metrics",
        lambda _observed_dir: unavailable,
    )
    daemon = ExperimentOrchestrator(
        project_root=tmp_path, automil_dir=automil_dir
    )
    monkeypatch.setattr(daemon, "_reload_orchestrator_config", lambda: None)
    monkeypatch.setattr(daemon, "_check_running", lambda: None)
    tick_observations = []
    monkeypatch.setattr(
        daemon,
        "_tick_cells",
        lambda **kwargs: tick_observations.append(kwargs["activity_observation"]),
    )
    monkeypatch.setattr(daemon, "_get_pending", lambda: [])
    monkeypatch.setattr(daemon, "_save_state", lambda: None)

    daemon.tick()

    assert tick_observations == [unavailable]
    assert not hasattr(daemon, "_activity_refresh_error")


def test_completed_activity_session_does_not_probe_dead_endpoint(
    tmp_path, monkeypatch,
):
    """A durable final sample makes later daemon ticks network-independent."""
    automil_dir = _build_minimal_automil(tmp_path)
    from automil.cells.activity import (
        ingest_prometheus_metrics,
        record_hook_event,
    )

    record_hook_event(
        automil_dir,
        "cell-1",
        {
            "hook_event_name": "SessionStart",
            "session_id": "session-1",
            "source": "startup",
        },
        observed_at=1.0,
    )
    ingest_prometheus_metrics(
        automil_dir,
        'claude_code_active_time_total{session_id="session-1",type="cli"} 5\n',
        observed_at=3.0,
    )
    record_hook_event(
        automil_dir,
        "cell-1",
        {"hook_event_name": "SessionEnd", "session_id": "session-1"},
        observed_at=3.0,
        final_sample_observed_at=3.0,
    )
    _write_agent_active_cell(automil_dir)

    from automil import activity_metrics
    from automil.backends._orchestrator_daemon import ExperimentOrchestrator

    monkeypatch.setattr(
        activity_metrics,
        "observe_activity_metrics",
        lambda *_args, **_kwargs: pytest.fail("completed session was scraped"),
    )
    daemon = ExperimentOrchestrator(
        project_root=tmp_path, automil_dir=automil_dir,
    )

    assert daemon._observe_activity_for_tick() is None


def test_namespace_isolation(tmp_path):
    """D-169: backend A's running entries don't appear in backend B's list_running()."""
    automil_dir = _build_minimal_automil(tmp_path)
    # Drop a fake JSON file under running/slurm/ — local backend must NOT see it.
    slurm_running = automil_dir / "orchestrator" / "running" / "slurm"
    slurm_running.mkdir(parents=True)
    (slurm_running / "fake_slurm_node.json").write_text(json.dumps({
        "id": "fake_slurm_node",
        "backend": "slurm",
        "opaque_id": "12345",
        "submitted_at": 0.0,
    }))

    from automil.backends.local import LocalBackend
    backend = LocalBackend(project_root=tmp_path, automil_dir=automil_dir)
    handles = backend.list_running()
    assert all(h.node_id != "fake_slurm_node" for h in handles), \
        "LocalBackend.list_running leaked a SLURM-namespaced node"
