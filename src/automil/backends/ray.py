"""RayBackend on ray>=2.55.1 raw @ray.remote (BCK-06 / D-161..D-167, D-179).

Opt-in via ``uv sync --extra ray``. Implements the Phase 2 Backend ABC by
dispatching jobs to a Ray cluster via raw ``@ray.remote`` functions (NOT Ray
Tune -- D-187 explicit non-goal). Hybrid init (D-161): try RAY_ADDRESS, fall
back to local cluster if allow_local_fallback=True.

Critical API decisions (RESEARCH.md OQ-2..4 corrections applied inline):
  - ``ray.init(ignore_reinit_error=True)`` -- local_mode is deprecated in Ray 2.55+
  - ``poll()`` catches WorkerCrashedError (force=True path) IN ADDITION to
    TaskCancelledError (force=False path) and RayTaskError (user exception)
  - ``@ray.remote`` on a FUNCTION (not Actor) -- force=True valid; D-162
  - worktree path passed explicitly to ``_run_experiment_ray`` (JobSpec is frozen)

BCK-04: zero ``os.kill | os.killpg | Popen | .pid`` -- Ray APIs sufficient.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Iterator, Optional

import ray
import ray.exceptions

from automil.backends import register
from automil.backends.base import Backend, HealthReport, JobHandle, JobSpec, JobState
from automil.backends.errors import BackendError, RayClusterUnreachableError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TERMINAL_STATES: frozenset[JobState] = frozenset({
    JobState.COMPLETED, JobState.CRASHED,
    JobState.CANCELLED, JobState.BUDGET_KILLED,
})

DEFAULT_GPU_VRAM_GB: float = 24.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _was_cap_cancel(handle: JobHandle, automil_dir: Path) -> bool:
    """Discriminate cap-kill vs operator-cancel by reading running/ray/<node>.json.

    The orchestrator sets ``cap_cancel_pending: true`` BEFORE calling
    ``backend.cancel(handle, signal=SIGTERM)`` per Phase 4 D-115 / Pitfall 4
    pattern (orchestrator daemon writes this annotation atomically before the
    cancel reaches the backend).
    """
    running_path = (
        automil_dir / "orchestrator" / "running" / "ray" / f"{handle.node_id}.json"
    )
    if not running_path.exists():
        return False
    try:
        payload = json.loads(running_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return bool(payload.get("cap_cancel_pending", False))


# ---------------------------------------------------------------------------
# Top-level @ray.remote function (must be module-level, not class method)
# ---------------------------------------------------------------------------

@ray.remote
def _run_experiment_ray(spec: JobSpec, worktree_path: Path, log_path: Path) -> int:
    """Inside the Ray worker process -- runs the experiment.

    RESEARCH.md OQ-4 / Pitfall 8: worktree_path passed explicitly because
    JobSpec is frozen (Phase 2 D-54).

    Steps:
      1. Build env from spec.env added onto current os.environ (no mutation of
         parent process env -- Ray local-cluster workers share the driver process
         under pytest; mutating os.environ pollutes subsequent tests).
      2. subprocess.run(spec.command, cwd=target_dir, env=sub_env, stdout=logf,
         stderr=STDOUT) writes both streams into log_path.
      3. Return retcode (ray.get(ref) raises if non-zero -- caller handles).

    Notes:
      - cwd= passes chdir to the child process only; the worker CWD is untouched.
        (W-4 fix: NOT os.chdir -- that would mutate the Ray worker's CWD globally.)
      - log_path lives at running/ray/<node_id>.log so log_iter knows where to tail.
      - BCK-04: subprocess.run (NOT Popen); no .pid access.
    """
    import subprocess  # noqa: PLC0415
    import os as _os  # noqa: PLC0415

    log_path.parent.mkdir(parents=True, exist_ok=True)
    target_dir = (
        worktree_path / spec.working_subdir if spec.working_subdir else worktree_path
    )

    # Build subprocess env without mutating os.environ.
    sub_env = dict(_os.environ)
    for k, v in spec.env:
        sub_env[k] = v

    with open(log_path, "w") as logf:
        # cwd= passes chdir to child only -- parent (Ray worker) CWD untouched.
        completed = subprocess.run(
            list(spec.command),
            cwd=str(target_dir),
            env=sub_env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return completed.returncode


# ---------------------------------------------------------------------------
# RayBackend
# ---------------------------------------------------------------------------

@register("ray")
class RayBackend(Backend):
    """Ray dispatch via raw @ray.remote (BCK-06 / D-161..D-167).

    Hybrid init (D-161):
      1. If RAY_ADDRESS is set (or "auto"), try to connect to that cluster.
      2. On ConnectionError, fall back to starting a local Ray cluster if
         ``allow_local_fallback`` is True (config: backend.ray.allow_local_fallback).
      3. If fallback disabled, raise RayClusterUnreachableError (D-178).

    ``_we_started_ray`` is a PUBLIC flag -- contract tests rely on it to drive
    teardown discipline (only call ray.shutdown() if we started the cluster).
    """

    _we_started_ray: bool  # public -- used by tests for ray.shutdown discipline

    def __init__(
        self,
        automil_dir: Path,
        config: dict,
        project_root: Optional[Path] = None,
    ) -> None:
        self._automil_dir = Path(automil_dir)
        self._config = config
        self._project_root = (
            Path(project_root) if project_root else self._automil_dir.parent
        )

        backend_cfg = config.get("backend", {}) or {}
        ray_cfg = backend_cfg.get("ray", {}) or {}
        allow_local_fallback = bool(ray_cfg.get("allow_local_fallback", True))

        self._jobs: dict[str, ray.ObjectRef] = {}
        self._we_started_ray = False
        self._running_dir = (
            self._automil_dir / "orchestrator" / "running" / "ray"
        )
        self._running_dir.mkdir(parents=True, exist_ok=True)

        # D-161 + RESEARCH.md OQ-2: hybrid init. local_mode is deprecated in Ray 2.55+.
        if not ray.is_initialized():
            ray_address = os.environ.get("RAY_ADDRESS", "auto")
            try:
                ray.init(
                    address=ray_address,
                    ignore_reinit_error=True,
                    log_to_driver=False,
                )
            except ConnectionError:
                if not allow_local_fallback:
                    raise RayClusterUnreachableError(ray_address)
                ray.init(ignore_reinit_error=True, log_to_driver=False)
                self._we_started_ray = True

        logger.info(
            "RayBackend initialised: we_started_ray=%s", self._we_started_ray
        )

    # ------------------------------------------------------------------
    # Backend.submit
    # ------------------------------------------------------------------

    def submit(self, spec: JobSpec) -> JobHandle:
        """Dispatch via @ray.remote function. Creates worktree first.

        B-1 fix: 2-positional create_worktree + separate apply_overlay (mirrors
        _orchestrator_daemon.py canonical two-step pattern -- NOT 4-kwarg call).
        """
        from automil.runner import Runner  # noqa: PLC0415

        runner = Runner(self._project_root, self._automil_dir)
        worktree_path = runner.create_worktree(spec.base_commit, spec.node_id)
        if spec.overlay_dir:
            runner.apply_overlay(
                worktree_path,
                spec.overlay_dir,
                deletions=getattr(spec, "deletions", None),
                manifest=getattr(spec, "overlay_manifest", None),  # HASH-0
            )
        log_path = self._running_dir / f"{spec.node_id}.log"

        # D-162: fractional GPU reservation from spec.gpu_estimate_gb.
        num_gpus = (
            spec.gpu_estimate_gb / DEFAULT_GPU_VRAM_GB
            if spec.gpu_estimate_gb > 0
            else 0
        )
        opts = {"num_gpus": num_gpus} if num_gpus > 0 else {}
        ref = _run_experiment_ray.options(**opts).remote(
            spec, worktree_path, log_path
        )

        opaque_id = ref.hex()
        handle = JobHandle(
            node_id=spec.node_id,
            backend="ray",
            opaque_id=opaque_id,
            submitted_at=time.time(),
        )
        self._jobs[opaque_id] = ref
        self._persist_running(handle, spec, worktree_path, log_path)
        return handle

    def _persist_running(
        self,
        handle: JobHandle,
        spec: JobSpec,
        worktree_path: Path,
        log_path: Path,
    ) -> None:
        """Write running/ray/<node_id>.json atomically (D-25 / Phase 0 pattern)."""
        payload = {
            "node_id": handle.node_id,
            "backend": "ray",
            "opaque_id": handle.opaque_id,
            "submitted_at": handle.submitted_at,
            "spec_path": str(spec.overlay_dir / "spec.json"),
            "worktree_path": str(worktree_path),
            "log_path": str(log_path),
            "cap_cancel_pending": False,
        }
        path = self._running_dir / f"{handle.node_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_path, str(path))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # Backend.poll
    # ------------------------------------------------------------------

    def poll(self, handle: JobHandle) -> JobState:
        """Non-blocking snapshot via ray.wait(timeout=0).

        RESEARCH.md OQ-3: catch WorkerCrashedError (force=True path) AND
        TaskCancelledError (force=False path) AND RayTaskError (user exception).
        Three SEPARATE except clauses so the exception map is explicit and
        grep-verifiable (plan requirement: 3 distinct except clauses).
        """
        ref = self._jobs.get(handle.opaque_id)
        if ref is None:
            # D-167: ObjectRef not restorable across daemon restart.
            return JobState.CRASHED

        ready, not_ready = ray.wait([ref], timeout=0)
        if not_ready:
            # Ray collapses PENDING+RUNNING -- D-164.
            return JobState.RUNNING

        # ref is in `ready`; ray.get surfaces terminal status or exception.
        try:
            ray.get(ref, timeout=0)
            return JobState.COMPLETED
        except ray.exceptions.RayTaskError:
            # Task raised a Python exception -- crashed.
            return JobState.CRASHED
        except ray.exceptions.WorkerCrashedError:
            # force=True cancel path (RESEARCH.md OQ-3 / Pitfall 4).
            # Discriminate cap-kill vs operator-cancel via running JSON flag.
            if _was_cap_cancel(handle, self._automil_dir):
                return JobState.BUDGET_KILLED
            return JobState.CANCELLED
        except ray.exceptions.TaskCancelledError:
            # force=False cancel path -- same discrimination.
            if _was_cap_cancel(handle, self._automil_dir):
                return JobState.BUDGET_KILLED
            return JobState.CANCELLED

    # ------------------------------------------------------------------
    # Backend.list_running
    # ------------------------------------------------------------------

    def list_running(self) -> list[JobHandle]:
        """Scan running/ray/*.json. ObjectRef restoration is process-local (D-167).

        Handles from a previous daemon process have no live ObjectRef in
        self._jobs; poll() will return CRASHED for them (D-167 documented
        limitation -- operator must resubmit).
        """
        handles: list[JobHandle] = []
        if not self._running_dir.exists():
            return handles
        for spec_file in sorted(self._running_dir.glob("*.json")):
            try:
                payload = json.loads(spec_file.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "RayBackend.list_running: skipping %s: %s",
                    spec_file.name, exc,
                )
                continue
            handles.append(JobHandle(
                node_id=payload.get("node_id", spec_file.stem),
                backend="ray",
                opaque_id=payload.get("opaque_id", ""),
                submitted_at=payload.get(
                    "submitted_at", spec_file.stat().st_mtime
                ),
            ))
        return handles

    # ------------------------------------------------------------------
    # Backend.cancel
    # ------------------------------------------------------------------

    def cancel(self, handle: JobHandle, signal: Optional[int] = None) -> None:
        """Fire-and-forget Ray cancel via ray.cancel(force=True).

        signal arg is ignored on Ray (no Unix-signal granularity per D-165);
        a logged warning fires when called with a non-None signal (Phase 2 D-57).

        force=True is valid for @ray.remote FUNCTIONS (NOT actors -- D-162
        design confirmation; RESEARCH.md Pitfall 3). recursive=True ensures
        chained remote calls are also cancelled.
        """
        if signal is not None:
            logger.warning(
                "RayBackend.cancel: signal=%d ignored; Ray uses force=True "
                "which terminates via SIGKILL after Ray's ~1s grace.",
                signal,
            )
        ref = self._jobs.get(handle.opaque_id)
        if ref is None:
            logger.info(
                "RayBackend.cancel: %s not in self._jobs "
                "(ref restoration not supported across daemon restart -- D-167)",
                handle.node_id,
            )
            return
        try:
            ray.cancel(ref, force=True, recursive=True)
        except Exception as exc:
            logger.warning(
                "RayBackend.cancel: ray.cancel failed for %s: %s",
                handle.node_id, exc,
            )

    # ------------------------------------------------------------------
    # Backend.log_iter
    # ------------------------------------------------------------------

    def log_iter(self, handle: JobHandle) -> Iterator[str]:
        """Tail running/ray/<node_id>.log written by _run_experiment_ray.

        D-166: per-actor log file populated by the wrapper before invoking
        spec.command; backend tails with 1s tick (matches SLURM tail pattern).
        Closes when poll() returns a terminal state.
        """
        log_path = self._running_dir / f"{handle.node_id}.log"
        offset = 0

        while True:
            if log_path.exists():
                try:
                    text = log_path.read_text()
                except OSError:
                    text = ""
                if len(text) > offset:
                    new_text = text[offset:]
                    offset = len(text)
                    for line in new_text.splitlines(keepends=True):
                        yield line

            state = self.poll(handle)
            if state in _TERMINAL_STATES:
                # Final drain after terminal state observed.
                if log_path.exists():
                    try:
                        text = log_path.read_text()
                    except OSError:
                        text = ""
                    if len(text) > offset:
                        for line in text[offset:].splitlines(keepends=True):
                            yield line
                return

            time.sleep(1.0)

    # ------------------------------------------------------------------
    # close() -- graceful shutdown (D-161)
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Shutdown Ray ONLY if WE started the local cluster (D-161).

        A shared/operator-managed cluster must NOT be shut down by the backend;
        only a local cluster that the backend's __init__ started is safe to stop.
        """
        if self._we_started_ray and ray.is_initialized():
            try:
                ray.shutdown()
            except Exception as exc:
                logger.warning(
                    "RayBackend.close: ray.shutdown raised: %s", exc
                )

    # ------------------------------------------------------------------
    # Backend ABC: healthcheck (D-189)
    # ------------------------------------------------------------------

    def healthcheck(self) -> HealthReport:
        """Hardware probe deferred for distributed backends (D-189 / D-196).

        Ray clusters surface resources via `ray status` cluster-resource map rather
        than per-node nvidia-smi. The healthcheck contract for Ray-shaped backends
        is deferred to a post-v1.0 phase; see 07-CONTEXT.md `<deferred>`.
        """
        raise NotImplementedError(
            "healthcheck deferred to Phase 7+ for distributed backends "
            "(use `salloc`/`ray status` directly)"
        )
