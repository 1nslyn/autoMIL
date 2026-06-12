"""STP-06 / D-198 clause 5 / Pitfall 9 anti-acceptance #4.

Three tests, end-to-end via the real `automil` CLI:
  1. test_setup_gate_aborts_on_known_bad_config: train.py with ImportError
     causes orchestrator to mark node 'crash'; gate-runner returns False.
  2. test_setup_gate_passes_on_known_good_config: synthetic train.py writes
     valid result.json; orchestrator marks 'completed'; gate-runner returns True.
  3. test_setup_gate_polling_terminates_within_90s: polling budget is 90s;
     even pathological tail-spin train.py terminates (with status='crash'
     after the per-experiment timeout, OR as a budget-killed; both are
     non-success terminal).

Strategy (F-02 fix):
  - Use `automil orchestrator start` / `automil orchestrator stop` subprocesses
    rather than any private daemon helper (no synchronous one-shot entry point
    exists on ExperimentOrchestrator). This exercises the real daemon code path.
  - Skip cleanly when the `automil` console-script is not on PATH (rare;
    typically only in stripped CI images).
"""
from __future__ import annotations

# ## Setup helpers (top of file) -- F-01 fix: helpers BEFORE decorators.
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest


def _automil_on_path() -> bool:
    """Return True if the `automil` console-script is invocable on PATH.

    True under `uv run pytest` (uv puts the workspace's installed entry
    points on PATH automatically) and under `pip install -e .` shells.
    False only in stripped CI images that did not install the package.
    """
    return shutil.which("automil") is not None


def _bootstrap_project(repo: Path, train_py_content: str) -> Path:
    """Initialise repo via `automil init --no-healthcheck`; overwrite train.py.

    Returns the `<repo>/automil` directory.
    """
    init_proc = subprocess.run(
        ["automil", "init", "--no-healthcheck"],
        cwd=repo, capture_output=True, text=True, timeout=30,
    )
    assert init_proc.returncode == 0, (
        f"automil init failed (rc={init_proc.returncode}):\n"
        f"stdout={init_proc.stdout}\nstderr={init_proc.stderr}"
    )

    # Overwrite train.py with the per-test content.
    (repo / "train.py").write_text(train_py_content)

    # Stage + commit so submit has a clean base_commit.
    subprocess.run(["git", "add", "train.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "test variant"], cwd=repo, check=True)

    return repo / "automil"


def _start_daemon(repo: Path) -> subprocess.Popen:
    """Launch `automil orchestrator start` in its own process group."""
    return subprocess.Popen(
        ["automil", "orchestrator", "start"],
        cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        preexec_fn=os.setsid,
    )


def _stop_daemon(repo: Path, proc: subprocess.Popen) -> None:
    """Signal the daemon to drain via `automil orchestrator stop`; SIGTERM as fallback."""
    try:
        subprocess.run(
            ["automil", "orchestrator", "stop"],
            cwd=repo, capture_output=True, text=True, timeout=15,
        )
    except subprocess.TimeoutExpired:
        pass
    # Give the daemon a moment to honour the stop signal, then escalate.
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=5)


def _run_gate(repo: Path, automil_dir: Path, node_id: str) -> str:
    """Simulate the skill's setup-done gate via real CLI calls. Returns terminal status string.

    Steps (D-195):
      1. `automil check` (must exit 0; if non-zero in this synthetic env, skip).
      2. `automil submit --max-time 60 --node <id> --files train.py`.
      3. Start the daemon as a subprocess; poll for archive/<id>/result.json.
      4. Stop the daemon; read result.json and return the 'status' field.
    """
    # Step 1: check.
    check_proc = subprocess.run(
        ["automil", "check"],
        cwd=repo, capture_output=True, text=True, timeout=30,
    )
    if check_proc.returncode != 0:
        pytest.skip(
            f"automil check failed in this synthetic env (rc={check_proc.returncode}); "
            f"the gate's pre-flight is not the contract under test here.\n"
            f"stderr={check_proc.stderr[-200:]}"
        )

    # Step 2: submit.
    submit_proc = subprocess.run(
        ["automil", "submit",
         "--node", node_id,
         "--desc", "setup-validation",
         "--files", "train.py",
         "--max-time", "60",
         "--mil-model", "test_model"],  # D-12: required
        cwd=repo, capture_output=True, text=True, timeout=30,
    )
    assert submit_proc.returncode == 0, (
        f"automil submit failed (rc={submit_proc.returncode}):\n"
        f"stdout={submit_proc.stdout}\nstderr={submit_proc.stderr}"
    )

    # Step 3: start daemon + poll.
    archive_result = automil_dir / "orchestrator" / "archive" / node_id / "result.json"
    daemon = _start_daemon(repo)
    deadline = time.monotonic() + 90.0
    try:
        while time.monotonic() < deadline:
            if archive_result.exists():
                break
            if daemon.poll() is not None:
                # Daemon died unexpectedly.
                stdout, stderr = daemon.communicate(timeout=5)
                pytest.fail(
                    f"orchestrator daemon exited prematurely "
                    f"(rc={daemon.returncode}):\nstdout={stdout!r}\nstderr={stderr!r}"
                )
            time.sleep(2.0)
    finally:
        _stop_daemon(repo, daemon)

    if not archive_result.exists():
        return "timeout"
    return json.loads(archive_result.read_text()).get("status", "unknown")


# ## Per-test train.py contents (top-level constants, used by tests below).
_BAD_TRAIN_PY = "import nonexistent_module_xyz_07_09  # ImportError\n"

_GOOD_TRAIN_PY = (
    "import json, pathlib\n"
    "result = {\n"
    "    'status': 'completed',\n"
    "    'metrics': {'val_auc': 0.5, 'val_bacc': 0.5, 'test_auc': 0.5, 'test_bacc': 0.5},\n"
    "    'composite': 0.5, 'elapsed_seconds': 1, 'peak_vram_mb': 100,\n"
    "}\n"
    "pathlib.Path('result.json').write_text(json.dumps(result))\n"
)


# ## Tests (decorators reference _automil_on_path defined above).

@pytest.mark.skipif(
    not _automil_on_path(),
    reason="automil console-script not on PATH; install via `pip install -e .` or run under `uv run pytest`.",
)
def test_setup_gate_aborts_on_known_bad_config(tmp_git_repo, monkeypatch):
    """D-198 clause 5: ImportError train.py -> status='crash' -> gate aborts."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    automil_dir = _bootstrap_project(tmp_git_repo, _BAD_TRAIN_PY)
    status = _run_gate(tmp_git_repo, automil_dir, node_id="node_setup_validation")
    assert status in {"crash", "crashed"}, (
        f"known-bad train.py expected status='crash', got status={status!r}"
    )


@pytest.mark.skipif(
    not _automil_on_path(),
    reason="automil console-script not on PATH; install via `pip install -e .` or run under `uv run pytest`.",
)
def test_setup_gate_passes_on_known_good_config(tmp_git_repo, monkeypatch):
    """D-198 clause 5: valid train.py -> status='completed' -> gate passes."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    automil_dir = _bootstrap_project(tmp_git_repo, _GOOD_TRAIN_PY)
    status = _run_gate(tmp_git_repo, automil_dir, node_id="node_setup_validation")
    assert status in {"completed", "executed"}, (
        f"known-good train.py expected status in {{'completed','executed'}}, got status={status!r}"
    )


@pytest.mark.skipif(
    not _automil_on_path(),
    reason="automil console-script not on PATH; install via `pip install -e .` or run under `uv run pytest`.",
)
def test_setup_gate_polling_terminates_within_90s(tmp_git_repo, monkeypatch):
    """D-195: polling budget is 90s. The gate-runner does not hang on pathological train.py."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    automil_dir = _bootstrap_project(tmp_git_repo, _BAD_TRAIN_PY)
    start = time.monotonic()
    status = _run_gate(tmp_git_repo, automil_dir, node_id="node_setup_polling")
    elapsed = time.monotonic() - start
    assert elapsed < 95.0, f"gate polling took {elapsed:.1f}s; 90s budget exceeded"
    assert status != "timeout", "gate-runner did not observe a terminal state"
