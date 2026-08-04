"""Integration tests for `automil cancel` + `automil resubmit` CLI commands.

References: CLI-03, CLI-04, D-66, D-67, D-70 (point 5), D-76.

Test architecture:
- Uses MockSLURMBackend(poll_lag_seconds=0.05) for fast state transitions.
- Writes a synthetic graph.json + running/<id>.json fixture per test.
- Invokes commands through Click's CliRunner for black-box CLI testing.
- monkeypatch.chdir(tmp_path) so _find_automil_dir() resolves to the fixture dir.
- _isolated_backends autouse fixture clears BACKENDS between tests to prevent
  registration leakage from other test files (PATTERNS.md §11).
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def _isolated_backends():
    """Save + restore BACKENDS registry around every test.

    Uses save-restore (not clear-reimport) to correctly handle Python's module
    cache: @register decorators only run once at import time; a bare _clear_backends()
    cannot be undone by re-importing a cached module.  The save-restore pattern
    (from backends/conftest.py) is the canonical approach.
    """
    from automil.backends import BACKENDS  # noqa: PLC0415
    saved = dict(BACKENDS)
    yield
    BACKENDS.clear()
    BACKENDS.update(saved)


@pytest.fixture
def mock_backend():
    """A fresh MockSLURMBackend instance with fast poll lag for tests.

    Explicitly ensures MockSLURMBackend is in BACKENDS: @register only fires at
    class-definition time (first import), but _isolated_backends save-restore
    may un-register it when restoring a pre-import snapshot.  The explicit
    BACKENDS["mock_slurm"] = ... assignment sidesteps the Python module cache.
    """
    from automil.backends import BACKENDS  # noqa: PLC0415
    from automil.backends.mock_slurm import MockSLURMBackend  # noqa: PLC0415
    # Ensure MockSLURMBackend is registered for this test (handles the case
    # where _isolated_backends restored BACKENDS to a pre-import snapshot).
    if "mock_slurm" not in BACKENDS:
        BACKENDS["mock_slurm"] = MockSLURMBackend
    return MockSLURMBackend(poll_lag_seconds=0.05)


# ---------------------------------------------------------------------------
# Synthetic fixture helpers
# ---------------------------------------------------------------------------


def _make_adir(tmp_path: Path) -> Path:
    """Create a minimal automil/ directory structure under tmp_path.

    Also creates a .git marker so _find_git_root() can resolve the git root.
    D-169: running specs are stored per-backend under running/<backend>/;
    create running/local/ so the OPS-01 local-backend cancel path has a home.
    """
    (tmp_path / ".git").mkdir(exist_ok=True)
    adir = tmp_path / "automil"
    orch_dir = adir / "orchestrator"
    for sub in ("queue", "running", "archive"):
        (orch_dir / sub).mkdir(parents=True, exist_ok=True)
    # Per-backend running subdirectory for local jobs (D-169).
    (orch_dir / "running" / "local").mkdir(parents=True, exist_ok=True)
    # Minimal config.yaml so _find_automil_dir() resolves adir.
    (adir / "config.yaml").write_text("run:\n  script: train.py\n")
    return adir


def _write_graph(adir: Path, nodes: dict[str, Any]) -> None:
    """Write a graph.json with the given nodes dict."""
    graph = {
        "schema_version": 1,
        "meta": {
            "best_composite": 0.0,
            "best_node_id": None,
            "total_executed": 0,
            "total_proposed": 0,
            "next_id": 10,
            "baseline_composite": 0.0,
            "scoring": {
                "exploration_weight": 0.005,
                "novelty_weight": 0.003,
            },
        },
        "nodes": nodes,
        "technique_stats": {},
    }
    (adir / "graph.json").write_text(json.dumps(graph, indent=2))


def _write_running_spec(
    adir: Path,
    node_id: str,
    opaque_id: str,
    submitted_at: float | None = None,
    backend_name: str = "local",
) -> None:
    """Write running/<backend_name>/<node_id>.json with opaque_id + submitted_at.

    D-169: per-backend namespaced path (Phase 6). cancel.py reads from
    running/<backend_name>/<id>.json; backend_name defaults to 'local' for
    legacy nodes (D-76 fallback).

    This is the source of truth cancel.py reads (W-03 fix: opaque_id is NOT
    stored in graph.json metadata).
    """
    spec = {
        "id": node_id,
        "opaque_id": opaque_id,
        "submitted_at": submitted_at or time.time(),
        "base_commit": "abc1234",
        "command": ["python", "train.py"],
        "env": {},
    }
    running_dir = adir / "orchestrator" / "running" / backend_name
    running_dir.mkdir(parents=True, exist_ok=True)
    running_path = running_dir / f"{node_id}.json"
    running_path.write_text(json.dumps(spec, indent=2))


def _write_archive(adir: Path, node_id: str) -> None:
    """Write a minimal archive/<node_id>/ overlay for resubmit tests."""
    archive_dir = adir / "orchestrator" / "archive" / node_id
    archive_dir.mkdir(parents=True, exist_ok=True)
    # Write a minimal overlay file (not result.json/spec.json).
    (archive_dir / "train.py").write_text("# resubmit overlay\n")
    # Write a spec.json so resubmit.py can read command/base_commit/etc.
    spec = {
        "id": node_id,
        "base_commit": "abc1234",
        "command": ["python", "train.py"],
        "env": {},
        "estimated_vram_gb": 2.0,
        "timeout_min": 60,
        "working_subdir": "",
        "graph_metadata": {"parent_id": None, "techniques": []},
    }
    (archive_dir / "spec.json").write_text(json.dumps(spec, indent=2))


# ---------------------------------------------------------------------------
# T1: test_cancel_happy_path
# ---------------------------------------------------------------------------


def test_cancel_happy_path(
    cli_runner: CliRunner, tmp_path: Path, mock_backend, monkeypatch
) -> None:
    """Cancel a running node → exit 0, graph status = 'cancelled', running file archived.

    Uses a state_file so the MockSLURMBackend instance created by cancel.py
    can read the same in-flight job state as the test's mock_backend instance.
    """
    from automil.cli import main  # noqa: PLC0415
    from automil.backends import BACKENDS, JobSpec  # noqa: PLC0415
    from automil.backends.mock_slurm import MockSLURMBackend  # noqa: PLC0415

    adir = _make_adir(tmp_path)
    monkeypatch.chdir(tmp_path)

    running_node_id = "node_0001"
    state_file = tmp_path / "mock_state.json"

    # Create a state-file-backed MockSLURMBackend so the CLI can reopen the
    # same job state when it instantiates its own fresh backend object.
    stateful_backend = MockSLURMBackend(
        poll_lag_seconds=0.05,
        state_file=state_file,
    )

    # Patch BACKENDS so cancel.py picks up stateful_backend's class.
    # We override __init__ to return the stateful instance each time.
    class _StatefulFactory(MockSLURMBackend):
        def __init__(self, **_kw: object) -> None:  # type: ignore[override]
            # Share state: copy _jobs and _lock references from stateful_backend.
            self._poll_lag = stateful_backend._poll_lag
            self._jobs = stateful_backend._jobs
            self._counter = stateful_backend._counter
            self._lock = stateful_backend._lock
            self._state_file = stateful_backend._state_file

    BACKENDS["mock_slurm"] = _StatefulFactory

    # Submit a job so stateful_backend knows about opaque_id.
    handle = stateful_backend.submit(JobSpec(
        node_id=running_node_id,
        base_commit="abc1234",
        overlay_files=(),
        overlay_dir=adir / "orchestrator" / "archive" / running_node_id,
        command=("python", "train.py"),
        env=(),
        working_subdir="",
        gpu_estimate_gb=2.0,
        walltime_seconds=3600,
    ))
    opaque_id = handle.opaque_id

    # Write running spec (W-03: source of truth for cancel.py).
    # D-169: namespaced under running/mock_slurm/ since this test uses mock_slurm backend.
    _write_running_spec(adir, running_node_id, opaque_id, backend_name="mock_slurm")

    # Write graph with node in 'running' state + metadata.backend = "mock_slurm".
    _write_graph(adir, {
        running_node_id: {
            "id": running_node_id,
            "parent_id": None,
            "type": "proposed",
            "status": "running",
            "description": "test run",
            "techniques": [],
            "metadata": {"backend": "mock_slurm"},
        }
    })

    result = cli_runner.invoke(
        main, ["cancel", running_node_id, "--timeout", "10"], catch_exceptions=False
    )

    assert result.exit_code == 0, f"cancel failed: {result.output}"
    assert f"Cancelled {running_node_id}" in result.output

    # Graph node must now be 'cancelled'.
    graph = json.loads((adir / "graph.json").read_text())
    assert graph["nodes"][running_node_id]["status"] == "cancelled", (
        f"expected status='cancelled', got {graph['nodes'][running_node_id].get('status')!r}"
    )
    assert graph["nodes"][running_node_id].get("metadata", {}).get("cancel_reason") == "cli"

    # running/mock_slurm/<id>.json must have been moved to archive/<id>/.
    # D-169: namespaced path (Phase 6).
    running_path = adir / "orchestrator" / "running" / "mock_slurm" / f"{running_node_id}.json"
    assert not running_path.exists(), "running spec not archived after cancel"
    archive_path = adir / "orchestrator" / "archive" / running_node_id
    assert archive_path.exists(), "archive directory not created after cancel"


# ---------------------------------------------------------------------------
# T2: test_cancel_unknown_node
# ---------------------------------------------------------------------------


def test_cancel_unknown_node(
    cli_runner: CliRunner, tmp_path: Path, monkeypatch
) -> None:
    """Cancel a non-existent node → non-zero exit + 'not found' in output."""
    from automil.cli import main  # noqa: PLC0415

    adir = _make_adir(tmp_path)
    monkeypatch.chdir(tmp_path)
    _write_graph(adir, {})  # empty graph

    result = cli_runner.invoke(main, ["cancel", "nonexistent_xyz"])

    assert result.exit_code != 0, "expected non-zero exit for unknown node"
    assert "not found" in result.output.lower(), (
        f"expected 'not found' in output: {result.output!r}"
    )


# ---------------------------------------------------------------------------
# T3: test_cancel_terminal_node
# ---------------------------------------------------------------------------


def test_cancel_terminal_node(
    cli_runner: CliRunner, tmp_path: Path, monkeypatch
) -> None:
    """Cancel a completed node → non-zero exit + 'Refusing to cancel' in output."""
    from automil.cli import main  # noqa: PLC0415

    adir = _make_adir(tmp_path)
    monkeypatch.chdir(tmp_path)

    completed_id = "node_0002"
    _write_graph(adir, {
        completed_id: {
            "id": completed_id,
            "parent_id": None,
            "type": "executed",
            "status": "completed",
            "description": "done",
            "techniques": [],
            "metadata": {"backend": "local"},
        }
    })

    result = cli_runner.invoke(main, ["cancel", completed_id])

    assert result.exit_code != 0, "expected non-zero exit for terminal node"
    assert "Refusing to cancel" in result.output, (
        f"expected 'Refusing to cancel' in output: {result.output!r}"
    )


# ---------------------------------------------------------------------------
# T4: test_cancel_missing_running_spec
# ---------------------------------------------------------------------------


def test_cancel_missing_running_spec(
    cli_runner: CliRunner, tmp_path: Path, monkeypatch
) -> None:
    """Cancel when running/<id>.json is missing → non-zero + 'no running spec'."""
    from automil.cli import main  # noqa: PLC0415

    adir = _make_adir(tmp_path)
    monkeypatch.chdir(tmp_path)

    node_id = "node_0003"
    _write_graph(adir, {
        node_id: {
            "id": node_id,
            "parent_id": None,
            "type": "proposed",
            "status": "running",
            "description": "running but no spec file",
            "techniques": [],
            "metadata": {"backend": "local"},
        }
    })
    # Deliberately do NOT write running/<id>.json.

    result = cli_runner.invoke(main, ["cancel", node_id])

    assert result.exit_code != 0, "expected non-zero exit for missing running spec"
    assert "no running spec" in result.output.lower(), (
        f"expected 'no running spec' in output: {result.output!r}"
    )


# ---------------------------------------------------------------------------
# T5: test_cancel_timeout
# ---------------------------------------------------------------------------


def test_cancel_timeout(
    cli_runner: CliRunner, tmp_path: Path, mock_backend, monkeypatch
) -> None:
    """Cancel that doesn't transition → non-zero exit + timeout diagnostic."""
    from automil.cli import main  # noqa: PLC0415
    from automil.backends import JobSpec, JobState  # noqa: PLC0415

    adir = _make_adir(tmp_path)
    monkeypatch.chdir(tmp_path)

    node_id = "node_0004"

    # Submit a real job into MockSLURM.
    handle = mock_backend.submit(JobSpec(
        node_id=node_id,
        base_commit="abc1234",
        overlay_files=(),
        overlay_dir=adir / "orchestrator" / "archive" / node_id,
        command=("python", "train.py"),
        env=(),
        working_subdir="",
        gpu_estimate_gb=2.0,
        walltime_seconds=3600,
    ))
    opaque_id = handle.opaque_id

    # Patch the backend's poll to always return RUNNING (simulates stuck cancel).
    import automil.backends.mock_slurm as _ms  # noqa: PLC0415
    original_class = _ms.MockSLURMBackend

    class _StubbornBackend(original_class):
        def poll(self, h):
            return JobState.RUNNING  # never transitions

    # Patch BACKENDS to use our stubborn class.
    from automil.backends import BACKENDS  # noqa: PLC0415
    BACKENDS["mock_slurm"] = _StubbornBackend

    # D-169: running spec at namespaced path running/mock_slurm/<id>.json.
    _write_running_spec(adir, node_id, opaque_id, backend_name="mock_slurm")
    _write_graph(adir, {
        node_id: {
            "id": node_id,
            "parent_id": None,
            "type": "proposed",
            "status": "running",
            "description": "stubborn job",
            "techniques": [],
            "metadata": {"backend": "mock_slurm"},
        }
    })

    result = cli_runner.invoke(main, ["cancel", node_id, "--timeout", "1"])

    assert result.exit_code != 0, "expected non-zero exit for timeout"
    assert "cancel sent but state did not transition" in result.output.lower(), (
        f"expected timeout diagnostic in output: {result.output!r}"
    )


# ---------------------------------------------------------------------------
# T6: test_resubmit_happy_path
# ---------------------------------------------------------------------------


def test_resubmit_happy_path(
    cli_runner: CliRunner, tmp_path: Path, mock_backend, monkeypatch
) -> None:
    """Resubmit a crashed node → exit 0, new node_id printed, graph entry with resubmitted_from."""
    from automil.cli import main  # noqa: PLC0415

    adir = _make_adir(tmp_path)
    monkeypatch.chdir(tmp_path)

    crashed_id = "node_0005"

    # Write a crashed node + archive overlay.
    _write_graph(adir, {
        crashed_id: {
            "id": crashed_id,
            "parent_id": None,
            "type": "executed",
            "status": "crashed",
            "description": "original crashed experiment",
            "techniques": [],
            "metadata": {"backend": "mock_slurm"},
        }
    })
    _write_archive(adir, crashed_id)

    result = cli_runner.invoke(main, ["resubmit", crashed_id], catch_exceptions=False)

    assert result.exit_code == 0, f"resubmit failed: {result.output}"

    # New node_id must be printed to stdout (D-67 step 7).
    new_node_id = result.output.strip()
    assert new_node_id, "expected new node_id printed to stdout"
    assert new_node_id != crashed_id, "new node_id must differ from old (T-02-08-S02)"

    # Graph must contain the new node with resubmitted_from metadata.
    graph = json.loads((adir / "graph.json").read_text())
    assert new_node_id in graph["nodes"], (
        f"new node {new_node_id!r} not found in graph.json; nodes: {list(graph['nodes'].keys())}"
    )
    new_node = graph["nodes"][new_node_id]
    assert new_node.get("metadata", {}).get("resubmitted_from") == crashed_id, (
        f"expected metadata.resubmitted_from={crashed_id!r}, "
        f"got: {new_node.get('metadata', {})!r}"
    )


def test_resubmit_is_not_an_architecture_preserving_bypass(
    cli_runner: CliRunner, tmp_path: Path, monkeypatch,
) -> None:
    from automil.cli import main  # noqa: PLC0415

    adir = _make_adir(tmp_path)
    monkeypatch.chdir(tmp_path)
    (adir / "config.yaml").write_text(
        "registry:\n"
        "  mode: architecture-preserving\n"
        "  protected: [models/**]\n"
        "  allowed_override_options: [--hparams]\n"
        "  allowed_variant_kinds: [policy]\n"
        "files:\n"
        "  editable: [recipes/**]\n"
        "run:\n"
        "  script: train.py\n"
    )
    node_id = "node_0005"
    _write_graph(adir, {
        node_id: {
            "id": node_id,
            "parent_id": None,
            "type": "executed",
            "status": "crashed",
            "description": "old candidate",
            "techniques": [],
            "metadata": {"backend": "local"},
        }
    })
    _write_archive(adir, node_id)

    result = cli_runner.invoke(main, ["resubmit", node_id])
    assert result.exit_code != 0
    assert "architecture-preserving" in result.output
    assert "automil submit" in result.output


# ---------------------------------------------------------------------------
# OPS-01 RED stubs (Wave 0 — Nyquist compliance)
# These tests xfail until plan 13-02 implements the local direct-kill path.
# ---------------------------------------------------------------------------


def test_cancel_local_direct_kill(
    cli_runner: CliRunner, tmp_path: Path, monkeypatch
) -> None:
    """Cancel a daemon-launched local job via metadata.pid/pgid in the running spec.

    Anti-theater constraint (13-VALIDATION.md §Critical Anti-Theater Constraints):
    This test MUST spawn a real subprocess and assert that cancel kills it.
    Do NOT mock os.killpg — the test verifies real process termination.
    """
    from automil.cli import main  # noqa: PLC0415

    adir = _make_adir(tmp_path)
    monkeypatch.chdir(tmp_path)

    node_id = "node_0010"

    # Spawn a real child process with its own session (mirrors daemon behaviour).
    proc = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        pgid = os.getpgid(proc.pid)

        # Lazy import inside test body to avoid circular-import risk at collection time.
        from automil.orchestrator import _read_proc_starttime  # noqa: PLC0415
        starttime = _read_proc_starttime(proc.pid)  # None on non-Linux — that is fine.

        # Write running spec in the shape the daemon writes (D-169 local backend).
        running_spec: dict = {
            "id": node_id,
            "metadata": {
                "pid": proc.pid,
                "pgid": pgid,
            },
        }
        if starttime is not None:
            running_spec["metadata"]["starttime_ticks"] = starttime

        running_path = adir / "orchestrator" / "running" / "local" / f"{node_id}.json"
        running_path.write_text(json.dumps(running_spec, indent=2))

        # Write graph node as running with local backend.
        _write_graph(adir, {
            node_id: {
                "id": node_id,
                "parent_id": None,
                "type": "proposed",
                "status": "running",
                "description": "direct kill test",
                "techniques": [],
                "metadata": {"backend": "local"},
            }
        })

        result = cli_runner.invoke(main, ["cancel", node_id], catch_exceptions=False)
        assert result.exit_code == 0, result.output

        # Give the signal a moment to propagate.
        time.sleep(0.3)

        # Assert the process is dead (OPS-01 success criterion).
        with pytest.raises(ProcessLookupError):
            os.kill(proc.pid, 0)

        # Graph must be updated to cancelled.
        graph = json.loads((adir / "graph.json").read_text())
        assert graph["nodes"][node_id]["status"] == "cancelled"

    finally:
        # Teardown: kill the sleep process if still alive (e.g., while xfail).
        try:
            proc.kill()
            proc.wait()
        except Exception:
            pass


def test_cancel_missing_pid_metadata(
    cli_runner: CliRunner, tmp_path: Path, monkeypatch
) -> None:
    """Cancel when running spec has no opaque_id and no metadata.pid/pgid → hard fail.

    OPS-01: cancel must loud-fail (non-zero exit) with a message indicating
    the spec is corrupted or unmanageable when neither opaque_id nor metadata
    is present. This behavior already exists in the current implementation
    (the opaque_id hard-fail at cancel.py:100-105 catches this case);
    it must remain correct after OPS-01 adds the pid/pgid fallback path.
    NOT xfail — this specific sub-case already works correctly.
    """
    from automil.cli import main  # noqa: PLC0415

    adir = _make_adir(tmp_path)
    monkeypatch.chdir(tmp_path)

    node_id = "node_0011"

    # Running spec with neither opaque_id nor metadata (corrupted state).
    running_path = adir / "orchestrator" / "running" / "local" / f"{node_id}.json"
    running_path.write_text(json.dumps({"id": node_id}, indent=2))

    _write_graph(adir, {
        node_id: {
            "id": node_id,
            "parent_id": None,
            "type": "proposed",
            "status": "running",
            "description": "corrupted spec test",
            "techniques": [],
            "metadata": {"backend": "local"},
        }
    })

    result = cli_runner.invoke(main, ["cancel", node_id])

    assert result.exit_code != 0
    lower_out = result.output.lower()
    assert (
        "corrupted" in lower_out
        or "cannot cancel" in lower_out
        or "manage" in lower_out
    ), f"expected error message in output: {result.output!r}"


def test_cancel_no_starttime_ticks(
    cli_runner: CliRunner, tmp_path: Path, monkeypatch
) -> None:
    """Cancel succeeds using only pid/pgid when starttime_ticks is absent.

    OPS-01: starttime_ticks is optional (omitted on non-Linux). Cancellation
    must succeed via pid/pgid alone (with accepted PID-reuse risk on non-Linux).
    """
    from automil.cli import main  # noqa: PLC0415

    adir = _make_adir(tmp_path)
    monkeypatch.chdir(tmp_path)

    node_id = "node_0012"

    # Spawn a real child process (mirrors daemon behaviour).
    proc = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        pgid = os.getpgid(proc.pid)

        # Running spec WITHOUT starttime_ticks (simulates non-Linux environment).
        running_spec = {
            "id": node_id,
            "metadata": {
                "pid": proc.pid,
                "pgid": pgid,
                # starttime_ticks intentionally omitted
            },
        }
        running_path = adir / "orchestrator" / "running" / "local" / f"{node_id}.json"
        running_path.write_text(json.dumps(running_spec, indent=2))

        _write_graph(adir, {
            node_id: {
                "id": node_id,
                "parent_id": None,
                "type": "proposed",
                "status": "running",
                "description": "no starttime_ticks test",
                "techniques": [],
                "metadata": {"backend": "local"},
            }
        })

        result = cli_runner.invoke(main, ["cancel", node_id], catch_exceptions=False)
        assert result.exit_code == 0, result.output

        # Give the signal a moment to propagate.
        time.sleep(0.3)

        # Process must be dead even without starttime_ticks.
        with pytest.raises(ProcessLookupError):
            os.kill(proc.pid, 0)

        # Graph must be cancelled.
        graph = json.loads((adir / "graph.json").read_text())
        assert graph["nodes"][node_id]["status"] == "cancelled"

    finally:
        # Teardown: clean up if still alive while xfail.
        try:
            proc.kill()
            proc.wait()
        except Exception:
            pass
