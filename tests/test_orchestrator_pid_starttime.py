"""Coverage for PID-file starttime cross-check (CLN-04 / D-17)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


def _make_orch(tmp_path):
    from automil.orchestrator import ExperimentOrchestrator
    automil_dir = tmp_path / "automil"
    automil_dir.mkdir()
    (automil_dir / "config.yaml").write_text("orchestrator: {}\n")
    (tmp_path / ".git").mkdir()
    return ExperimentOrchestrator(project_root=tmp_path, automil_dir=automil_dir)


def test_proc_starttime_parses_comm_with_spaces():
    from automil.orchestrator import _parse_starttime_from_stat_line
    # Synthetic /proc/<pid>/stat line.
    # pid=123, comm=(my (weird) name), state=R, ppid, pgrp, session, ...
    # Then 18 more fields before starttime which is field 22 (1-indexed).
    # Layout: "<pid> (<comm>) <fields 3..52>"
    # Fields 3..21 = 19 fields, then field 22 = starttime.
    # fields 3..21 = 19 fields; field 22 = starttime.
    # "R" covers field 3; range(2,20) covers fields 4..21 (18 items) → 19 items total.
    suffix_fields = ["R"] + [str(i) for i in range(2, 20)]  # state + 18 placeholders -> fields 3..21
    line = "123 (my (weird) name) " + " ".join(suffix_fields) + " 78901234 " + " ".join(["0"] * 30)
    ticks = _parse_starttime_from_stat_line(line)
    assert ticks == 78901234


@pytest.mark.skipif(not Path("/proc").exists(), reason="Linux /proc required (D-17 Linux-only)")
def test_is_pid_alive_for_current_process(tmp_path):
    """The current pytest process must be detectable via the helper."""
    from automil.orchestrator import _is_pid_alive_with_starttime, _read_proc_starttime
    my_pid = os.getpid()
    my_starttime = _read_proc_starttime(my_pid)
    assert my_starttime is not None
    assert _is_pid_alive_with_starttime(my_pid, my_starttime) is True


def test_is_pid_alive_for_nonexistent_pid(tmp_path):
    from automil.orchestrator import _is_pid_alive_with_starttime
    # PID 99999999 essentially never exists in any practical scenario.
    assert _is_pid_alive_with_starttime(99999999, 1234567) is False


@pytest.mark.skipif(not Path("/proc").exists(), reason="Linux /proc required (D-17 Linux-only)")
def test_is_pid_alive_with_wrong_starttime(tmp_path):
    """Headline CLN-04 scenario: PID alive but starttime mismatch (PID reuse)."""
    from automil.orchestrator import _is_pid_alive_with_starttime, _read_proc_starttime
    my_pid = os.getpid()
    actual_starttime = _read_proc_starttime(my_pid)
    # Pretend the recorded starttime was different (PID reuse simulation).
    wrong = (actual_starttime or 0) + 999_999
    assert _is_pid_alive_with_starttime(my_pid, wrong) is False


@pytest.mark.skipif(not Path("/proc").exists(), reason="Linux /proc required (D-17 Linux-only)")
def test_pid_file_written_as_json(tmp_path, monkeypatch):
    """After daemon prepares the pid file, json.loads must succeed."""
    from automil.orchestrator import _write_pid_file
    pid_file = tmp_path / "orchestrator.pid"
    _write_pid_file(pid_file)
    data = json.loads(pid_file.read_text())
    assert set(data.keys()) >= {"pid", "starttime_ticks", "starttime_iso"}
    assert isinstance(data["pid"], int)
    assert isinstance(data["starttime_ticks"], int)
    assert isinstance(data["starttime_iso"], str)
    assert data["pid"] == os.getpid()


def test_load_pid_file_handles_legacy_plain_int(tmp_path):
    """Legacy format (plain int + newline) is treated as stale, helper returns None."""
    from automil.orchestrator import _load_pid_file
    pid_file = tmp_path / "orchestrator.pid"
    pid_file.write_text("12345\n")
    assert _load_pid_file(pid_file) is None  # signal: stale, treat as no daemon


def test_load_pid_file_handles_invalid_json(tmp_path):
    from automil.orchestrator import _load_pid_file
    pid_file = tmp_path / "orchestrator.pid"
    pid_file.write_text("{not valid json")
    assert _load_pid_file(pid_file) is None


def test_load_pid_file_handles_missing_keys(tmp_path):
    from automil.orchestrator import _load_pid_file
    pid_file = tmp_path / "orchestrator.pid"
    pid_file.write_text(json.dumps({"pid": 123}))  # missing starttime_ticks
    assert _load_pid_file(pid_file) is None


def test_write_pid_file_carries_hostname_and_slurm_job_id(tmp_path, monkeypatch):
    """Provenance fields: which host, and (if any) which SLURM job wrote this."""
    import socket
    from automil.backends.pidfile import write_pid_file

    monkeypatch.setenv("SLURM_JOB_ID", "778899")
    pid_file = tmp_path / "orchestrator.pid"
    write_pid_file(pid_file)

    data = json.loads(pid_file.read_text())
    assert data["hostname"] == socket.gethostname()
    assert data["slurm_job_id"] == "778899"


def test_write_pid_file_slurm_job_id_is_none_outside_slurm(tmp_path, monkeypatch):
    from automil.backends.pidfile import write_pid_file

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    pid_file = tmp_path / "orchestrator.pid"
    write_pid_file(pid_file)

    data = json.loads(pid_file.read_text())
    assert data["slurm_job_id"] is None


def test_load_pid_file_reader_returns_hostname_and_slurm_job_id(tmp_path, monkeypatch):
    import socket
    from automil.backends.pidfile import write_pid_file, load_pid_file

    monkeypatch.setenv("SLURM_JOB_ID", "1234")
    pid_file = tmp_path / "orchestrator.pid"
    write_pid_file(pid_file)

    loaded = load_pid_file(pid_file)
    assert loaded is not None
    assert loaded["hostname"] == socket.gethostname()
    assert loaded["slurm_job_id"] == "1234"


def test_load_pid_file_tolerates_old_format_without_provenance_fields(tmp_path):
    """A pid file written before this change (no hostname/slurm_job_id) must
    still load -- load_pid_file's required-key check is a subset check, so
    it never depended on these new fields being present."""
    from automil.backends.pidfile import load_pid_file

    pid_file = tmp_path / "orchestrator.pid"
    pid_file.write_text(json.dumps({
        "pid": 4242,
        "starttime_ticks": 987654,
        "starttime_iso": "2026-01-01T00:00:00",
    }))

    loaded = load_pid_file(pid_file)
    assert loaded is not None
    assert loaded["pid"] == 4242
    assert loaded.get("hostname") is None
    assert loaded.get("slurm_job_id") is None
