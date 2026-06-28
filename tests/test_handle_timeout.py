"""REC-01 / D-04: _handle_timeout main-PID-first signaling.

D-04 rewrites _handle_timeout to:
  1. SIGTERM the main PID first (so the flush handler can write partial result)
  2. SIGKILL the whole process group after a configurable grace window
     (orchestrator.timeout_grace_seconds, default 10s)

Current behavior: SIGTERM/SIGKILL the whole process group directly, never the
main PID first.
"""
from __future__ import annotations

import os
import signal
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_orchestrator_stub(config: dict | None = None):
    """Build a minimal ExperimentOrchestrator stub for testing _handle_timeout."""
    from automil.backends._orchestrator_daemon import ExperimentOrchestrator

    daemon = MagicMock(spec=ExperimentOrchestrator)
    daemon._timed_out = {}
    daemon.config = config or {}
    daemon._handle_completion = MagicMock()

    # Fake running experiment
    mock_process = MagicMock()
    mock_process.pid = 12345
    mock_process.poll = MagicMock(return_value=None)  # still running after SIGTERM
    mock_exp = MagicMock()
    mock_exp.process = mock_process
    daemon.running = {"node_0001": mock_exp}

    return daemon


def test_sigterm_sent_to_main_pid_first(tmp_path: Path) -> None:
    """D-04: SIGTERM must be sent to main PID before SIGKILL is sent to process group.

    RED until D-04 main-PID-first rewrite ships in Plan 04.
    """
    from automil.backends._orchestrator_daemon import ExperimentOrchestrator

    daemon = _make_orchestrator_stub()

    with (
        patch("os.kill") as mock_kill,
        patch("os.killpg") as mock_killpg,
        patch("os.getpgid", return_value=12345),
        patch("time.sleep"),
    ):
        ExperimentOrchestrator._handle_timeout(daemon, "node_0001")

    # D-04: SIGTERM to main PID must come BEFORE SIGKILL to process group
    # os.kill(pid, SIGTERM) must appear in the call list
    sigterm_to_pid_calls = [
        c for c in mock_kill.call_args_list
        if len(c.args) >= 2 and c.args[0] == 12345 and c.args[1] == signal.SIGTERM
    ]
    assert sigterm_to_pid_calls, (
        "D-04 not implemented: os.kill(pid, SIGTERM) was never called with the main PID. "
        "Main PID must be SIGTERM'd first so the flush handler can write partial result."
    )


def test_sigkill_sent_to_process_group_after_grace(tmp_path: Path) -> None:
    """D-04: SIGKILL must be sent to the process group after the grace window.

    RED until D-04 ships in Plan 04.
    """
    from automil.backends._orchestrator_daemon import ExperimentOrchestrator

    daemon = _make_orchestrator_stub()

    with (
        patch("os.kill") as mock_kill,
        patch("os.killpg") as mock_killpg,
        patch("os.getpgid", return_value=12345),
        patch("time.sleep") as mock_sleep,
    ):
        ExperimentOrchestrator._handle_timeout(daemon, "node_0001")

    # After grace window: SIGKILL to process group
    sigkill_pg_calls = [
        c for c in mock_killpg.call_args_list
        if len(c.args) >= 2 and c.args[1] == signal.SIGKILL
    ]
    assert sigkill_pg_calls, (
        "D-04 not implemented: os.killpg(pgid, SIGKILL) was never called after grace window. "
        "Process group must be SIGKILL'd after the grace period to reap VRAM."
    )

    # sleep must have been called (grace window)
    assert mock_sleep.called, (
        "D-04 not implemented: time.sleep not called — no grace window before SIGKILL."
    )


def test_grace_seconds_read_from_config(tmp_path: Path) -> None:
    """D-04: timeout_grace_seconds from config is used as the sleep duration.

    RED until D-04 configurable grace ships in Plan 04.
    """
    from automil.backends._orchestrator_daemon import ExperimentOrchestrator

    daemon = _make_orchestrator_stub(config={"orchestrator": {"timeout_grace_seconds": 3}})

    with (
        patch("os.kill"),
        patch("os.killpg"),
        patch("os.getpgid", return_value=12345),
        patch("time.sleep") as mock_sleep,
    ):
        ExperimentOrchestrator._handle_timeout(daemon, "node_0001")

    assert mock_sleep.called, "time.sleep must be called for the grace window"
    sleep_args = [c.args[0] for c in mock_sleep.call_args_list]
    assert any(abs(a - 3) < 0.1 for a in sleep_args), (
        f"D-04 not implemented: expected grace sleep of 3s from config, got {sleep_args}. "
        "timeout_grace_seconds=3 in config must be respected."
    )
