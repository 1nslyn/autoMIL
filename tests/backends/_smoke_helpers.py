"""Helpers for tests/backends/test_node_0176_smoke.py (D-176 acceptance smoke).

Synthesises a CCRCC node_0176-equivalent 1-fold experiment via the named backend.
Returns the composite metric from result.json. Used by the parametrised
acceptance smoke test to verify D-179 clause 7 (composite within +-0.005 of
LocalBackend baseline across all three CI-runnable backends).
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from automil.backends.base import JobSpec, JobState


_VALID_BACKEND_NAMES = {"local", "slurm-debug", "ray-local"}


def _build_spec(node_id: str, project_root: Path, automil_dir: Path) -> JobSpec:
    """Build a minimal JobSpec for the synthetic train.py experiment."""
    archive_dir = automil_dir / "orchestrator" / "archive"
    overlay_dir = archive_dir / node_id
    overlay_dir.mkdir(parents=True, exist_ok=True)
    # Write a stub spec.json so overlay_dir is a valid archive entry.
    (overlay_dir / "spec.json").write_text(json.dumps({"id": node_id}))
    return JobSpec(
        node_id=node_id,
        base_commit="HEAD",
        overlay_files=(),
        overlay_dir=overlay_dir,
        command=("python", "train.py"),
        env=(),
        working_subdir="",
        gpu_estimate_gb=0.0,
        walltime_seconds=300,
    )


def _read_composite(workdir: Path) -> float:
    """Read result.json from workdir; return composite. Raise if missing/malformed."""
    rj = workdir / "result.json"
    if not rj.exists():
        raise FileNotFoundError(f"result.json not produced at {rj}")
    payload = json.loads(rj.read_text())
    return float(payload["composite"])


def _run_local(project_root: Path, automil_dir: Path) -> float:
    """LocalBackend path: run train.py directly via subprocess (no daemon in fixture).

    This is the W-8 acceptable shortcut documented in the plan: LocalBackend.submit
    writes a queue file that a live daemon would pick up, but the test fixture has no
    live daemon. Instead, we run train.py directly — which is exactly what the daemon
    would do — to verify the synthetic train.py produces composite=0.502 deterministically.
    """
    workdir = automil_dir / "smoke_local_workdir"
    workdir.mkdir(parents=True, exist_ok=True)
    # Copy train.py from project_root into workdir so it runs with workdir as CWD.
    train_src = project_root / "train.py"
    train_dst = workdir / "train.py"
    train_dst.write_text(train_src.read_text())
    subprocess.run(["python", "train.py"], cwd=str(workdir), check=True)
    return _read_composite(workdir)


def _run_slurm_debug(project_root: Path, automil_dir: Path) -> float:
    """SLURMBackend with cluster='debug' (submitit in-process DebugExecutor).

    DebugExecutor runs the submitted function synchronously, so submit() returns
    only after the job has completed. poll() should immediately return COMPLETED.
    result.json lands in the worktree created by Runner inside SLURMBackend.submit().
    """
    from automil.backends.slurm import SLURMBackend

    config = {
        "backend": {
            "slurm": {
                "debug_in_process": True,
                "walltime_seconds": 60,
                "directives": {
                    "partition": "debug",
                    "account": "test",
                    "cpus_per_task": 1,
                    "mem_gb": 1,
                },
            }
        },
    }
    backend = SLURMBackend(
        automil_dir=automil_dir,
        config=config,
        project_root=project_root,
    )
    spec = _build_spec("smoke_slurm_debug", project_root, automil_dir)
    handle = backend.submit(spec)

    # DebugExecutor is synchronous — the job runs inside submit() and result.json is
    # already written by the time submit() returns. Poll just to confirm state.
    deadline = time.monotonic() + 60.0
    final_state = None
    while time.monotonic() < deadline:
        s = backend.poll(handle)
        if s in {JobState.COMPLETED, JobState.CRASHED, JobState.CANCELLED, JobState.BUDGET_KILLED}:
            final_state = s
            break
        time.sleep(0.1)

    if final_state != JobState.COMPLETED:
        raise RuntimeError(
            f"SLURM-debug run did not complete: state={final_state}"
        )

    # Worktree path follows the Runner convention (scoped per automil project).
    from automil.runner import Runner
    worktree_path = Runner(project_root, automil_dir).worktree_path("smoke_slurm_debug")
    return _read_composite(worktree_path)


def _run_ray_local(project_root: Path, automil_dir: Path) -> float:
    """RayBackend with local cluster (ray.init no-args, NOT deprecated local_mode=True).

    result.json lands in the worktree created by Runner inside RayBackend.submit().
    """
    import ray
    from automil.backends.ray import RayBackend

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    config = {
        "backend": {
            "ray": {"allow_local_fallback": True},
        },
    }
    backend = RayBackend(
        automil_dir=automil_dir,
        config=config,
        project_root=project_root,
    )
    try:
        spec = _build_spec("smoke_ray_local", project_root, automil_dir)
        handle = backend.submit(spec)

        deadline = time.monotonic() + 60.0
        final_state = None
        while time.monotonic() < deadline:
            s = backend.poll(handle)
            if s in {JobState.COMPLETED, JobState.CRASHED, JobState.CANCELLED, JobState.BUDGET_KILLED}:
                final_state = s
                break
            time.sleep(0.1)

        if final_state != JobState.COMPLETED:
            raise RuntimeError(
                f"ray-local run did not complete: state={final_state}"
            )

        # Worktree path follows the Runner convention (scoped per automil project).
        from automil.runner import Runner
        worktree_path = Runner(project_root, automil_dir).worktree_path("smoke_ray_local")
        return _read_composite(worktree_path)
    finally:
        if backend._we_started_ray:
            ray.shutdown()


def run_node_0176_smoke(backend_name: str, project_root: Path, automil_dir: Path) -> float:
    """Run the synthetic node_0176-equivalent experiment; return composite (D-176).

    Args:
        backend_name: one of {"local", "slurm-debug", "ray-local"}.
        project_root: directory with train.py + .git (set up by the test).
        automil_dir: project_root / "automil" (orchestrator/ subdirs already created).

    Returns:
        The composite metric from result.json.

    Raises:
        ValueError on unknown backend_name; RuntimeError on dispatch failure;
        FileNotFoundError when result.json missing.
    """
    if backend_name not in _VALID_BACKEND_NAMES:
        raise ValueError(
            f"unknown backend_name {backend_name!r}; expected one of {sorted(_VALID_BACKEND_NAMES)}"
        )
    if backend_name == "local":
        return _run_local(project_root, automil_dir)
    if backend_name == "slurm-debug":
        return _run_slurm_debug(project_root, automil_dir)
    return _run_ray_local(project_root, automil_dir)
