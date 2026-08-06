"""SLURMBackend on submitit>=1.5.3 (BCK-05 / D-152..D-160, D-179).

Opt-in via ``uv sync --extra slurm``. Implements the Phase 2 Backend ABC
(D-51..D-58) by dispatching jobs to a SLURM cluster through submitit's
AutoExecutor. The framework-mandated ``--signal=B:TERM@30`` SLURM directive
couples the Phase 4 D-115 cap contract into SLURM's native signal delivery.

Critical API decisions (RESEARCH.md OQ-1..4 corrections applied inline):
  - ``update_parameters(timeout_min=...)`` NOT ``time=`` (AutoExecutor uses the shared param name)
  - ``slurm_additional_parameters={"signal": "B:TERM@30"}`` NOT ``signal=`` kwarg
  - ``cluster="slurm"`` explicitly (not auto-detect; fails loudly if sbatch absent)
  - ``cluster="debug"`` when ``config['backend']['slurm']['debug_in_process']`` (CI)
  - ``job.paths.stdout`` for log-file path (NOT hardcoded ``{job_id}_log.out``)
  - worktree path passed explicitly to ``_run_experiment_subprocess`` (NOT a JobSpec field)

BCK-04: zero ``os.kill | os.killpg | Popen | .pid`` references — submitit APIs
are sufficient for the entire dispatch + state lifecycle.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Iterator, Optional

try:
    import submitit  # opt-in via [slurm] extra
    _SUBMITIT_AVAILABLE = True
except ImportError:
    submitit = None  # type: ignore[assignment]
    _SUBMITIT_AVAILABLE = False

from automil.backends import register
from automil.backends.base import Backend, HealthReport, JobHandle, JobSpec, JobState
from automil.backends.errors import BackendError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _walltime_to_timeout_min(walltime_seconds: int) -> int:
    """RESEARCH.md OQ-1 / D-155 corrected: convert walltime_seconds -> timeout_min.

    Pure function so Wave-0 ``test_walltime_seconds_to_timeout_min`` can exercise
    it without instantiating the backend or submitit.
    """
    return max(1, walltime_seconds // 60)


# ---------------------------------------------------------------------------
# State constants
# ---------------------------------------------------------------------------

_TERMINAL_STATES: frozenset[JobState] = frozenset({
    JobState.COMPLETED, JobState.CRASHED,
    JobState.CANCELLED, JobState.BUDGET_KILLED,
})


# D-157 + RESEARCH.md §"SLURM state map".
_SLURM_STATE_MAP: dict[str, JobState] = {
    "PENDING":        JobState.PENDING,
    "RUNNING":        JobState.RUNNING,
    "COMPLETED":      JobState.COMPLETED,
    "FAILED":         JobState.CRASHED,
    "CANCELLED":      JobState.CANCELLED,
    "TIMEOUT":        JobState.BUDGET_KILLED,
    "OUT_OF_MEMORY":  JobState.CRASHED,
    "NODE_FAIL":      JobState.CRASHED,
    "BOOT_FAIL":      JobState.CRASHED,
    "PREEMPTED":      JobState.CRASHED,
    "COMPLETING":     JobState.RUNNING,
    "REQUEUED":       JobState.PENDING,
    "UNKNOWN":        JobState.PENDING,  # fall-through; see _state_str_to_jobstate
}


# ---------------------------------------------------------------------------
# Top-level remote function (picklable; runs on the SLURM worker)
# ---------------------------------------------------------------------------

def _run_experiment_subprocess(spec: JobSpec, worktree_path: Path) -> int:
    """Inside the SLURM worker process — runs the experiment.

    RESEARCH.md OQ-4 / Pitfall 8: worktree_path is passed explicitly because
    JobSpec is frozen (Phase 2 D-54) and we cannot add a worktree-path field.

    Steps:
      1. Resolve target_dir = worktree_path / spec.working_subdir (if set).
      2. Build subprocess env from spec.env (whitelisted at orchestrator-side per CLN-02 D-04).
      3. ``subprocess.run(spec.command, cwd=str(target_dir), env=sub_env, check=False)``; return retcode.

    Notes:
      - We do NOT use ``Popen`` (BCK-04). ``subprocess.run`` is sufficient.
      - stdout/stderr land in submitit's ``{job_id}_0_log.out`` (D-159 corrected via
        ``job.paths.stdout``); log_iter tails that file.
      - SIGTERM (from ``--signal=B:TERM@30``) propagates to the subprocess; the
        user training script's ``register_sigterm_flush()`` (Phase 4 D-122)
        handles the per-fold partial-write.
      - We use cwd= kwarg (NOT os.chdir) to avoid mutating the parent process CWD
        under DebugExecutor / pytest (W-4 fix: avoids cross-test contamination).
      - We build sub_env from os.environ copy + spec.env additions (NOT mutating
        os.environ directly) to avoid cross-test contamination under DebugExecutor.
    """
    import subprocess  # noqa: PLC0415; needed inside remote
    import os as _os    # noqa: PLC0415

    target_dir = worktree_path / spec.working_subdir if spec.working_subdir else worktree_path
    # Build subprocess env without mutating os.environ (DebugExecutor runs in-process
    # under pytest; mutating shared state pollutes subsequent tests).
    sub_env = dict(_os.environ)
    for k, v in spec.env:
        sub_env[k] = v
    # cwd= passes the chdir down to the child only — does NOT mutate the parent CWD
    # (avoids cross-test contamination under DebugExecutor / Ray local cluster).
    completed = subprocess.run(list(spec.command), cwd=str(target_dir), env=sub_env, check=False)
    return completed.returncode


# ---------------------------------------------------------------------------
# SLURMBackend class
# ---------------------------------------------------------------------------

def _slurm_register(cls: type) -> type:
    """Conditionally register SLURMBackend: only when submitit is installed.

    D-153: guarded import in __init__.py prevents this module from being imported
    at all when the [slurm] extra is absent. However, in environments where the
    module IS imported (e.g. via direct ``import automil.backends.slurm``) without
    submitit, we must not register the backend — it cannot be used.
    """
    if _SUBMITIT_AVAILABLE:
        return register("slurm")(cls)
    return cls


@_slurm_register
class SLURMBackend(Backend):
    """SLURM dispatch via submitit AutoExecutor (BCK-05 / D-155..D-160).

    Constructor chooses between ``cluster="slurm"`` (production) and
    ``cluster="debug"`` (in-process CI) based on
    ``config['backend']['slurm']['debug_in_process']``.

    All five Backend ABC methods are implemented; ``_persist_running`` is a
    private helper for atomic JSON writes to ``running/slurm/<node_id>.json``.

    Requires the ``[slurm]`` extra (``uv sync --extra slurm``). Attempting
    to instantiate without submitit installed raises ``BackendNotInstalledError``.
    """

    def __init__(
        self,
        automil_dir: Path,
        config: dict,
        project_root: Optional[Path] = None,
    ) -> None:
        if not _SUBMITIT_AVAILABLE:
            from automil.backends.errors import BackendNotInstalledError  # noqa: PLC0415
            raise BackendNotInstalledError("slurm", "slurm")
        self._automil_dir = Path(automil_dir)
        self._config = config
        self._project_root = Path(project_root) if project_root else self._automil_dir.parent

        backend_cfg = config.get("backend", {}) or {}
        slurm_cfg = backend_cfg.get("slurm", {}) or {}
        directives = slurm_cfg.get("directives", {}) or {}
        debug_in_process = bool(slurm_cfg.get("debug_in_process", False))
        walltime_seconds = int(slurm_cfg.get("walltime_seconds", 21600))

        self._logs_dir = self._automil_dir / "orchestrator" / "running" / "slurm" / "submitit-logs"
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        self._running_dir = self._automil_dir / "orchestrator" / "running" / "slurm"
        self._running_dir.mkdir(parents=True, exist_ok=True)

        # D-155 + RESEARCH.md OQ-1: timeout_min, slurm_additional_parameters.
        # cluster="debug" uses submitit's in-process DebugExecutor (no sbatch needed).
        # cluster="slurm" uses the real AutoExecutor (fails loudly if sbatch absent).
        cluster = "debug" if debug_in_process else "slurm"
        self._executor = submitit.AutoExecutor(folder=str(self._logs_dir), cluster=cluster)

        # Build update_parameters kwargs — only include optional directives when present
        # to avoid sending None / 0 to submitit which may generate invalid sbatch lines.
        update_kwargs: dict = {
            # RESEARCH.md OQ-1: timeout_min (NOT time=) + slurm_additional_parameters (NOT signal=)
            "timeout_min": _walltime_to_timeout_min(walltime_seconds),
            "slurm_additional_parameters": {"signal": "B:TERM@30"},
        }
        if "cpus_per_task" in directives:
            update_kwargs["cpus_per_task"] = int(directives["cpus_per_task"])
        if "mem_gb" in directives:
            update_kwargs["mem_gb"] = int(directives["mem_gb"])
        if directives.get("gpus_per_node") is not None:
            update_kwargs["gpus_per_node"] = int(directives["gpus_per_node"])
        if directives.get("partition"):
            update_kwargs["slurm_partition"] = directives["partition"]
        if directives.get("account"):
            update_kwargs["slurm_account"] = directives["account"]
        if directives.get("qos"):
            update_kwargs["slurm_qos"] = directives["qos"]

        self._executor.update_parameters(**update_kwargs)
        logger.info(
            "SLURMBackend initialised: cluster=%s timeout_min=%d",
            cluster, update_kwargs["timeout_min"],
        )

    # ------------------------------------------------------------------
    # Backend ABC: submit (D-55)
    # ------------------------------------------------------------------

    def submit(self, spec: JobSpec) -> JobHandle:
        """Dispatch via submitit. Creates worktree first; passes path to remote function.

        RESEARCH.md OQ-4 / D-156 corrected: worktree path is an explicit argument
        to ``_run_experiment_subprocess`` — NOT a JobSpec field (JobSpec is frozen).

        Two-step worktree creation mirrors ``_orchestrator_daemon.py:629-642``:
          1. ``runner.create_worktree(spec.base_commit, spec.node_id)``
          2. ``runner.apply_overlay(worktree_path, spec.overlay_dir, deletions=...)``
        """
        from automil.runner import Runner  # noqa: PLC0415; lazy to avoid cycles
        runner = Runner(self._project_root)
        # Real Runner API: 2-positional create_worktree, then separate apply_overlay
        # (mirrors _orchestrator_daemon.py:629-642 — the canonical two-step pattern)
        worktree_path = runner.create_worktree(spec.base_commit, spec.node_id)
        if spec.overlay_dir:
            runner.apply_overlay(
                worktree_path,
                spec.overlay_dir,
                deletions=getattr(spec, "deletions", None),
                manifest=getattr(spec, "overlay_manifest", None),  # HASH-0
            )
        try:
            job = self._executor.submit(_run_experiment_subprocess, spec, worktree_path)
        except FileNotFoundError as exc:
            raise BackendError(
                "SLURM tools not found on PATH; this machine doesn't appear to have a "
                "SLURM installation. To use SLURMBackend, run on a SLURM-equipped node "
                "or set backend.slurm.debug_in_process=true for in-process testing."
            ) from exc

        opaque_id = str(job.job_id)
        handle = JobHandle(
            node_id=spec.node_id,
            backend="slurm",
            opaque_id=opaque_id,
            submitted_at=time.time(),
        )
        self._persist_running(handle, spec, worktree_path)
        return handle

    def _persist_running(self, handle: JobHandle, spec: JobSpec, worktree_path: Path) -> None:
        """Write running/slurm/<node_id>.json atomically (D-25 / Phase 0 atomic-write pattern)."""
        payload = {
            "node_id": handle.node_id,
            "backend": "slurm",
            "opaque_id": handle.opaque_id,
            "submitted_at": handle.submitted_at,
            "spec_path": str(spec.overlay_dir / "spec.json") if spec.overlay_dir else "",
            "worktree_path": str(worktree_path),
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
                os.unlink(tmp_path)  # path.unlink rollback per memory:feedback_never_blind_checkout
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # Backend ABC: poll (D-56)
    # ------------------------------------------------------------------

    def poll(self, handle: JobHandle) -> JobState:
        """Reconstruct submitit.Job; map state via _SLURM_STATE_MAP.

        D-157: ``job.state`` re-queries sacct on each access (with caching).
        Unknown SLURM states default to PENDING with a logged warning.
        """
        try:
            job = submitit.Job(folder=str(self._logs_dir), job_id=handle.opaque_id)
            state_str = (job.state or "UNKNOWN").upper()
        except Exception as exc:
            logger.warning(
                "SLURMBackend.poll: error querying SLURM for %s: %s",
                handle.node_id, exc,
            )
            return JobState.PENDING
        mapped = _SLURM_STATE_MAP.get(state_str)
        if mapped is None:
            logger.warning(
                "SLURMBackend.poll: unknown SLURM state %r for %s — defaulting to PENDING",
                state_str, handle.node_id,
            )
            return JobState.PENDING
        return mapped

    # ------------------------------------------------------------------
    # Backend ABC: list_running (D-59)
    # ------------------------------------------------------------------

    def list_running(self) -> list[JobHandle]:
        """Scan running/slurm/*.json (D-169 namespacing).

        Restart-safe (D-59 / D-160): a fresh SLURMBackend instance recovers
        the live set from disk. Stale handles (job no longer in SLURM) are
        returned as-is; the orchestrator transitions them to CRASHED on the
        next tick when poll() returns an unknown state.
        """
        handles: list[JobHandle] = []
        if not self._running_dir.exists():
            return handles
        for spec_file in sorted(self._running_dir.glob("*.json")):
            try:
                payload = json.loads(spec_file.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "SLURMBackend.list_running: skipping %s: %s",
                    spec_file.name, exc,
                )
                continue
            handles.append(JobHandle(
                node_id=payload.get("node_id", spec_file.stem),
                backend="slurm",
                opaque_id=payload.get("opaque_id", ""),
                submitted_at=payload.get("submitted_at", spec_file.stat().st_mtime),
            ))
        return handles

    # ------------------------------------------------------------------
    # Backend ABC: cancel (D-57)
    # ------------------------------------------------------------------

    def cancel(self, handle: JobHandle, signal: Optional[int] = None) -> None:
        """Fire-and-forget SLURM cancel via submitit.Job.cancel().

        Custom-signal warning per Phase 2 D-57: any signal other than SIGTERM
        bypasses the standard SIGTERM→scancel→TIMEOUT escalation.
        """
        if signal is not None:
            import signal as _sig  # noqa: PLC0415
            if signal != _sig.SIGTERM:
                logger.warning(
                    "SLURMBackend.cancel: custom signal %d; the standard "
                    "SIGTERM→scancel→TIMEOUT escalation is bypassed.",
                    signal,
                )
        try:
            job = submitit.Job(folder=str(self._logs_dir), job_id=handle.opaque_id)
            job.cancel()
        except Exception as exc:
            logger.warning(
                "SLURMBackend.cancel: scancel failed for %s: %s",
                handle.node_id, exc,
            )

    # ------------------------------------------------------------------
    # Backend ABC: log_iter (D-58)
    # ------------------------------------------------------------------

    def log_iter(self, handle: JobHandle) -> Iterator[str]:
        """Tail submitit's stdout file with 1s tick; closes on terminal state.

        RESEARCH.md OQ-2 / D-159 corrected: use ``job.paths.stdout`` (a Path),
        which resolves to ``{job_id}_0_log.out`` — NOT hardcoded ``{job_id}_log.out``.

        The iterator yields lines as they appear and closes when the SLURM job
        reaches a terminal state. The orchestrator enforces a 60s drain timeout
        (D-170); backends that block forever are a contract violation.
        """
        try:
            job = submitit.Job(folder=str(self._logs_dir), job_id=handle.opaque_id)
            log_path = Path(job.paths.stdout)
        except Exception as exc:
            logger.warning(
                "SLURMBackend.log_iter: cannot resolve stdout for %s: %s",
                handle.node_id, exc,
            )
            return

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
                # Final drain: yield any lines written after terminal state was set.
                if log_path.exists():
                    try:
                        text = log_path.read_text()
                    except OSError:
                        text = ""
                    if len(text) > offset:
                        for line in text[offset:].splitlines(keepends=True):
                            yield line
                return

            time.sleep(1.0)  # 1s tick (vs 0.1s for local -- SLURM polling is slower)

    # ------------------------------------------------------------------
    # Backend ABC: healthcheck (D-189)
    # ------------------------------------------------------------------

    def healthcheck(self) -> HealthReport:
        """Hardware probe deferred for distributed backends (D-189 / D-196).

        SLURM clusters surface hardware via `salloc` and per-partition `sinfo` queries
        rather than per-node nvidia-smi. The healthcheck contract for SLURM-shaped
        backends is deferred to a post-v1.0 phase; see 07-CONTEXT.md `<deferred>`.
        """
        raise NotImplementedError(
            "healthcheck deferred to Phase 7+ for distributed backends "
            "(use `salloc`/`ray status` directly)"
        )
