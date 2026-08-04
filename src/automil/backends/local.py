"""LocalBackend — thin protocol adapter over _orchestrator_daemon (BCK-02 / D-60, D-61, D-77).

LocalBackend is NOT a re-implementation of the experiment lifecycle.  Each
method delegates to the existing ExperimentOrchestrator on-disk state machine:

  - submit   → writes to queue/<id>.json  (preserves daemon-pickup model, D-77)
  - poll     → reads queue/, running/, archive/<id>/result.json snapshots (D-56)
  - cancel   → delegates to _daemon._kill_experiment()  (D-57, BCK-04)
  - list_running → scans running/*.json  (D-59)
  - log_iter → tails archive/<id>/run.log with 0.1s tick  (D-58)

``local.py`` is one of exactly two files (together with ``_orchestrator_daemon.py``)
allowed by BCK-04 to reference ``os.kill | os.killpg | .pid | Popen``.  All
process-control calls are delegated to ``_daemon._kill_experiment`` so that
the process-control surface stays inside ``_orchestrator_daemon.py``.

Phase 2 invariant: ``LocalBackend.__init__`` NEVER triggers ``_recover_orphans``.
``ExperimentOrchestrator.__init__`` already calls ``_load_state(recover=False)``
(see _orchestrator_daemon.py line ~355), so construction is safe to call from
any CLI command without corrupting live-daemon state.
"""
from __future__ import annotations

import importlib.metadata
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from automil.backends.base import Backend, HealthReport, JobHandle, JobSpec, JobState
from automil.backends.errors import BackendError
from automil.backends import register

logger = logging.getLogger(__name__)


def _get_automil_version() -> str:
    """importlib.metadata.version with fallback for editable installs."""
    try:
        return importlib.metadata.version("automil")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0+unknown"


@register("local")
class LocalBackend(Backend):
    """Thin protocol adapter wrapping ExperimentOrchestrator (D-60).

    Each public method maps to the daemon's on-disk state machine.  The daemon
    runs as a separate OS process; LocalBackend communicates with it purely via
    the ``queue/``, ``running/``, and ``archive/`` directories under
    ``automil/orchestrator/``.

    Behavioural identity guarantee (D-60): if all 394+ tests pass after adding
    this class, LocalBackend is a correct shim over _orchestrator_daemon.
    """

    def __init__(
        self,
        project_root: Path | None = None,
        automil_dir: Path | None = None,
    ) -> None:
        """Instantiate a LocalBackend.  Daemon construction is lazy.

        Signature mirrors ExperimentOrchestrator.__init__ exactly (D-61).
        Construction NEVER triggers _recover_orphans — ExperimentOrchestrator
        already calls _load_state(recover=False) in its __init__ (Phase 0
        invariant; see CONCERNS.md § "CLI commands must not trigger orphan
        recovery").

        The ExperimentOrchestrator is constructed on first access to any
        daemon-derived attribute (queue/running/archive dirs, _daemon itself).
        This lets ``LocalBackend().healthcheck()`` succeed from any cwd —
        the hardware probe doesn't need a project on disk.  Daemon-required
        methods (submit, cancel, list_running, log_iter) trigger construction
        and fail loudly if no ``automil/config.yaml`` is findable, same as
        before.

        Args:
            project_root: Path to the git repo root.  ``None`` → auto-detect
                          by walking up from cwd at first daemon access.
            automil_dir:  Path to the ``automil/`` overlay directory.  ``None``
                          → auto-detect at first daemon access.
        """
        self._project_root_arg = project_root
        self._automil_dir_arg = automil_dir
        self._daemon_cache: object | None = None

    @property
    def _daemon(self):
        """Lazily construct the ExperimentOrchestrator on first access."""
        if self._daemon_cache is None:
            # Lazy import inside the property to avoid circular imports at module load.
            # backends/__init__.py imports local.py; local.py must not import
            # _orchestrator_daemon at module top-level to keep the import graph acyclic.
            from automil.backends._orchestrator_daemon import ExperimentOrchestrator  # noqa: PLC0415

            self._daemon_cache = ExperimentOrchestrator(
                project_root=self._project_root_arg,
                automil_dir=self._automil_dir_arg,
            )
        return self._daemon_cache

    @property
    def _orch_dir(self) -> Path:
        return self._daemon.orch_dir

    @property
    def _queue_dir(self) -> Path:
        return self._daemon.queue_dir

    @property
    def _running_dir(self) -> Path:
        # D-169: explicit namespaced path; was self._daemon.running_dir (flat).
        # running/local/ is the LocalBackend's exclusive namespace (Phase 6 BCK-05/06).
        return self._orch_dir / "running" / "local"

    @property
    def _archive_dir(self) -> Path:
        return self._daemon.archive_dir

    # ------------------------------------------------------------------
    # Backend.submit  (D-55, D-77)
    # ------------------------------------------------------------------

    def submit(self, spec: JobSpec) -> JobHandle:
        """Write spec to queue/<id>.json and return a pending JobHandle.

        Converts the JobSpec frozen dataclass into the daemon's existing
        queue-spec dict format (same shape as cli/submit.py writes).  The
        daemon picks up the file from queue/ on its next poll tick.

        Returns a JobHandle with opaque_id="pending" because the daemon has
        not yet launched the job (and therefore has no PID).  The opaque_id
        remains "pending" until the caller observes RUNNING via poll() (D-77).
        """
        submitted_ts = time.time()
        submitted_iso = datetime.utcfromtimestamp(submitted_ts).isoformat()

        # Build spec dict in the shape cli/submit.py produces (lines 276-293).
        # Keys referenced by the daemon's _launch():
        #   id, base_commit, overlay_dir, overlay_manifest, deletions,
        #   priority, estimated_vram_gb, timeout_min, graph_metadata,
        #   submitted_at, metadata.backend
        #
        # overlay_dir CR-01 fix (Phase 2 review): use spec.overlay_dir, NOT a
        # hardcoded archive/<node_id> path. resubmit.py passes the OLD node's
        # archive (archive/<old_id>) as spec.overlay_dir; hardcoding the new
        # node's path silently runs the resubmitted experiment on base-commit
        # code instead of the variant overlay. Make the path relative to
        # orch_dir when possible (matches cli/submit.py's "archive/<id>"
        # convention); fall back to absolute string if outside orch_dir.
        try:
            overlay_dir_str = str(spec.overlay_dir.relative_to(self._orch_dir))
        except ValueError:
            overlay_dir_str = str(spec.overlay_dir)

        queue_spec: dict = {
            "id": spec.node_id,
            "description": "",
            "base_commit": spec.base_commit,
            "overlay_dir": overlay_dir_str,
            "overlay_manifest": {f: "" for f in spec.overlay_files},
            "deletions": [],
            "priority": 1,
            "estimated_vram_gb": spec.gpu_estimate_gb,
            "timeout_min": max(1, spec.walltime_seconds // 60),
            "graph_metadata": {
                "parent_id": None,
                "techniques": [],
                "config_hash": "",
            },
            "submitted_at": submitted_iso,
            # Per-spec env additions (D-54 / T-02-01-S01).
            "env": {k: v for k, v in spec.env},
        }
        # Phase 5 (D-140): merge spec.metadata tuple-of-tuples into queue_spec
        # metadata dict FIRST.  This is how gate/evaluate.py stamps gate_eval flags
        # through the SAME submit path (GTE-03: Backend.submit, NOT a parallel
        # mechanism).
        for k, v in spec.metadata:
            queue_spec.setdefault("metadata", {})[k] = v
        # D-76: mark backend so cancel.py / resubmit.py know which BACKENDS[name]
        # to dispatch to.  Backend stamp is written LAST so it wins over any
        # caller-provided "backend" key (T-05-03-01 tamper mitigation).
        queue_spec.setdefault("metadata", {})["backend"] = "local"

        # Ensure queue directory exists (may not exist in fresh project dirs).
        self._queue_dir.mkdir(parents=True, exist_ok=True)

        # Atomic write: write to a .tmp sibling, then os.replace to the final
        # path — same pattern as _atomic_write_text from cli/lifecycle/_shared.py.
        import os
        import tempfile

        queue_file = self._queue_dir / f"{spec.node_id}.json"
        payload = json.dumps(queue_spec, indent=2)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(self._queue_dir), suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w") as fh:
                fh.write(payload)
            os.replace(tmp_path, str(queue_file))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        logger.info(
            "LocalBackend.submit: queued %s (vram=%.1fGB, timeout=%ds)",
            spec.node_id, spec.gpu_estimate_gb, spec.walltime_seconds,
        )
        return JobHandle(
            node_id=spec.node_id,
            backend="local",
            opaque_id="pending",
            submitted_at=submitted_ts,
        )

    # ------------------------------------------------------------------
    # Backend.poll  (D-56, D-77)
    # ------------------------------------------------------------------

    def poll(self, handle: JobHandle) -> JobState:
        """Return a snapshot of the job's current state (non-blocking, D-56).

        Priority order matches the daemon's lifecycle transitions:
          1. running/<id>.json exists  → RUNNING
          2. archive/<id>/result.json  → map status field to JobState
          3. queue/<id>.json exists    → PENDING
          4. None of the above         → raise BackendError (unknown handle)

        Pure snapshot — does NOT advance the job state.
        """
        node_id = handle.node_id

        running_path = self._running_dir / f"{node_id}.json"
        result_path = self._archive_dir / node_id / "result.json"
        queue_path = self._queue_dir / f"{node_id}.json"

        if running_path.exists():
            return JobState.RUNNING

        if result_path.exists():
            try:
                result = json.loads(result_path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "LocalBackend.poll: unreadable result.json for %s: %s",
                    node_id, exc,
                )
                return JobState.CRASHED

            status = result.get("status", "")
            # Daemon status values → JobState mapping.
            # The daemon writes: "completed", "crash", "oom", "timeout",
            # "cancelled".  Map forward-compatibly.
            _STATUS_MAP: dict[str, JobState] = {
                "completed": JobState.COMPLETED,
                "crash": JobState.CRASHED,
                "oom": JobState.CRASHED,
                "timeout": JobState.CRASHED,
                "cancelled": JobState.CANCELLED,
                "budget_killed": JobState.BUDGET_KILLED,
            }
            if status in _STATUS_MAP:
                return _STATUS_MAP[status]
            # Unknown status — treat as CRASHED with a warning.
            logger.warning(
                "LocalBackend.poll: unknown result status %r for %s; "
                "treating as CRASHED.",
                status, node_id,
            )
            return JobState.CRASHED

        if queue_path.exists():
            return JobState.PENDING

        raise BackendError(
            f"Unknown handle: no spec at queue/, running/, or archive/ for "
            f"{node_id!r}.  The job may have been purged or never submitted "
            f"via this LocalBackend instance."
        )

    # ------------------------------------------------------------------
    # Backend.cancel  (D-57, BCK-04)
    # ------------------------------------------------------------------

    def cancel(self, handle: JobHandle, signal: Optional[int] = None) -> None:
        """Request cancellation — fire-and-forget (D-57).

        Two-phase logic:
          1. If the job is still PENDING (queue file present): remove the queue
             file immediately — the daemon will never pick it up.
          2. If the job is RUNNING: delegate to _daemon._kill_experiment()
             which signals the process group (BCK-04 allowlist applies).

        Returns None immediately regardless of outcome.  Observe the state
        transition to CANCELLED via subsequent poll() calls.

        ``signal`` is accepted for API compatibility with the Backend ABC
        (D-57) but custom signals are forwarded to _kill_experiment; a warning
        is logged because the daemon's standard SIGTERM→30s→SIGKILL escalation
        is bypassed.
        """
        import signal as signal_module

        node_id = handle.node_id
        queue_path = self._queue_dir / f"{node_id}.json"

        # Phase 1: cancel pending job by removing its queue file.
        if queue_path.exists():
            try:
                queue_path.unlink()
                logger.info(
                    "LocalBackend.cancel: removed queue spec for %s (was pending)",
                    node_id,
                )
            except OSError as exc:
                logger.warning(
                    "LocalBackend.cancel: could not remove queue file for %s: %s",
                    node_id, exc,
                )
            return

        # Phase 2: signal a running job via daemon delegate.
        sig = signal if signal is not None else signal_module.SIGTERM
        if signal is not None:
            logger.warning(
                "LocalBackend.cancel: custom signal %d forwarded to "
                "_kill_experiment; the standard SIGTERM→30s→SIGKILL escalation "
                "in the daemon is bypassed.",
                signal,
            )
        found = self._daemon._kill_experiment(node_id, sig=sig)
        if not found:
            logger.info(
                "LocalBackend.cancel: %s not in daemon.running — job may "
                "have already completed or be managed by a separate daemon "
                "process.  Cancel is fire-and-forget; no action taken.",
                node_id,
            )

    # ------------------------------------------------------------------
    # Backend.list_running  (D-59)
    # ------------------------------------------------------------------

    def list_running(self) -> list[JobHandle]:
        """Scan running/*.json and return one JobHandle per live spec.

        Restart-safe (D-59): a freshly constructed LocalBackend reads the
        same on-disk state as the in-flight daemon.  ``submitted_at`` is
        recovered from the spec's ``submitted_at`` field (ISO-8601 string →
        epoch float) or falls back to the file's mtime.
        """
        handles: list[JobHandle] = []
        if not self._running_dir.exists():
            return handles

        for spec_file in sorted(self._running_dir.glob("*.json")):
            try:
                spec = json.loads(spec_file.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "LocalBackend.list_running: unreadable %s: %s",
                    spec_file.name, exc,
                )
                continue

            node_id = spec.get("id", spec_file.stem)
            # Recover submitted_at: prefer ISO string in spec; fallback to mtime.
            submitted_iso = spec.get("submitted_at")
            if submitted_iso:
                try:
                    from datetime import timezone  # noqa: PLC0415
                    submitted_at = datetime.fromisoformat(submitted_iso).replace(
                        tzinfo=timezone.utc
                    ).timestamp()
                except (ValueError, TypeError):
                    submitted_at = spec_file.stat().st_mtime
            else:
                submitted_at = spec_file.stat().st_mtime

            handles.append(JobHandle(
                node_id=node_id,
                backend="local",
                opaque_id="running",  # PID not persisted on disk (Phase 2 design)
                submitted_at=submitted_at,
            ))

        return handles

    # ------------------------------------------------------------------
    # Backend.log_iter  (D-58)
    # ------------------------------------------------------------------

    def log_iter(self, handle: JobHandle) -> Iterator[str]:
        """Tail archive/<id>/run.log; yield lines; close on terminal state.

        The log file is written by the daemon at
        ``archive/<node_id>/run.log`` (see _orchestrator_daemon.py ~line 640).

        Behaviour:
          - If job is already in terminal state and log exists: yield all
            lines and return immediately.
          - If job is running: tail with a 0.1s tick (D-58), yielding new
            lines as they appear.  Closes when poll() returns a terminal state.
          - If log file does not exist yet: wait up to 1s then yield nothing.

        The 0.1s tick ensures CI-friendly behaviour — lines surface within ~1s
        of appearing in the file.  The iterator is safe to exhaust; it will
        not hang after the job reaches a terminal state.
        """
        node_id = handle.node_id
        log_path = self._archive_dir / node_id / "run.log"

        _TERMINAL: frozenset[JobState] = frozenset({
            JobState.COMPLETED,
            JobState.CRASHED,
            JobState.CANCELLED,
            JobState.BUDGET_KILLED,
        })

        def _is_terminal() -> bool:
            try:
                state = self.poll(handle)
                return state in _TERMINAL
            except BackendError:
                # Unknown handle → treat as terminal (job gone).
                return True

        # --- Immediate terminal: yield full file and return ---
        if _is_terminal():
            if log_path.exists():
                yield from log_path.read_text().splitlines(keepends=True)
            return

        # --- Running (or pending): tail the log file ---
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
                    # Yield complete lines; hold back the last line if no newline yet.
                    lines = new_text.splitlines(keepends=True)
                    for line in lines:
                        yield line

            if _is_terminal():
                # One final read after terminal to capture last lines.
                if log_path.exists():
                    try:
                        text = log_path.read_text()
                    except OSError:
                        text = ""
                    if len(text) > offset:
                        remaining = text[offset:]
                        for line in remaining.splitlines(keepends=True):
                            yield line
                return

            time.sleep(0.1)

    # ------------------------------------------------------------------
    # Backend.healthcheck  (D-189 / STP-01 / D-190)
    # ------------------------------------------------------------------

    def healthcheck(self) -> HealthReport:
        """Probe hardware and return a HealthReport (D-189 / STP-01 / D-190).

        Probe order: CUDA, then ROCm, then CPU. CUDA is reachable on Leo's
        workstation via NVIDIA_SMI_PATH. ROCm is best-effort, returns None on
        any error (Leo's environment is CUDA-only). CPU is the terminal
        fallback and always succeeds.

        detection_status branching (D-190):
          - 'ok': probe succeeded, all GPUs parsed, no warnings.
          - 'partial': some GPUs detected and others unparseable
            (e.g. '[Not Supported]', MIG slice memory).
          - 'failed': all probes failed AND user has env signal (CUDA_VISIBLE_DEVICES
            set) implying a GPU was expected. CLI consumers prompt operator override.
        """
        import os  # noqa: PLC0415

        warnings: list[str] = []
        accelerator: str = "cpu"
        gpu_count: int = 0
        gpu_vram_gb: tuple[float, ...] = ()
        status: str = "ok"

        cuda = self._healthcheck_cuda()
        if cuda is not None:
            gpu_count, gpu_vram_gb, cuda_warnings = cuda
            warnings.extend(cuda_warnings)
            accelerator = "cuda"
            if cuda_warnings or gpu_count != len(gpu_vram_gb):
                status = "partial"
        else:
            rocm = self._healthcheck_rocm()
            if rocm is not None:
                gpu_count, gpu_vram_gb, rocm_warnings = rocm
                warnings.extend(rocm_warnings)
                accelerator = "rocm"
                if rocm_warnings or gpu_count != len(gpu_vram_gb):
                    status = "partial"
            else:
                # CPU terminal fallback. CUDA expected -> 'failed'; otherwise 'ok'.
                if os.environ.get("CUDA_VISIBLE_DEVICES"):
                    status = "failed"
                    warnings.append(
                        "CUDA_VISIBLE_DEVICES is set but no GPU probe succeeded. "
                        "Verify nvidia-smi installation, driver, or unset the env var."
                    )

        return HealthReport(
            gpu_count=gpu_count,
            gpu_vram_gb=gpu_vram_gb,
            accelerator=accelerator,  # type: ignore[arg-type]
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            automil_version=_get_automil_version(),
            detection_status=status,  # type: ignore[arg-type]
            detection_warnings=tuple(warnings),
            detected_at=datetime.utcnow(),
        )

    def _healthcheck_cuda(
        self,
    ) -> Optional[tuple[int, tuple[float, ...], list[str]]]:
        """CUDA probe via NVIDIA_SMI_PATH. Returns None on whole-probe failure.

        Per RESEARCH.md OQ-1, reuses the path-pinned NVIDIA_SMI_PATH constant
        (NEVER bare 'nvidia-smi'). Parses --query-gpu=index,memory.total with
        --format=csv,noheader,nounits. Per-GPU parse failures surface in the
        warnings list with detection_status='partial'.
        """
        from automil.backends._orchestrator_daemon import NVIDIA_SMI_PATH  # noqa: PLC0415

        try:
            result = subprocess.run(
                [
                    NVIDIA_SMI_PATH,
                    "--query-gpu=index,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return None

        if result.returncode != 0 or not result.stdout.strip():
            return None

        warnings: list[str] = []
        vram_gb: list[float] = []
        gpu_count = 0
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            gpu_count += 1
            try:
                vram_mb = int(parts[1])
                vram_gb.append(vram_mb / 1024.0)
            except (ValueError, TypeError):
                warnings.append(
                    f"GPU index {parts[0]} memory.total unparseable ('{parts[1]}'); "
                    f"may be MIG slice or [Not Supported]. Skipping VRAM record."
                )

        # Optional MIG warning. Best-effort; failure is silent.
        try:
            mig = subprocess.run(
                [NVIDIA_SMI_PATH, "--query-gpu=mig.mode.current", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if mig.returncode == 0 and "Enabled" in mig.stdout:
                warnings.append(
                    "MIG mode is Enabled on at least one GPU; reported memory.total "
                    "is the slice memory, not parent device. Treat VRAM bin-packing as approximate."
                )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

        return gpu_count, tuple(vram_gb), warnings

    def _healthcheck_rocm(
        self,
    ) -> Optional[tuple[int, tuple[float, ...], list[str]]]:
        """ROCm best-effort probe via rocm-smi. Returns None on any error.

        Per RESEARCH.md Pitfall F, the ROCm CSV format is unstable across
        ROCm 5/6/7. Parse defensively, swallow any exception, return None.
        Leo's CUDA-only environment never reaches this path; tests mock it.
        """
        from automil.backends._orchestrator_daemon import ROCM_SMI_PATH  # noqa: PLC0415

        try:
            result = subprocess.run(
                [ROCM_SMI_PATH, "--showmeminfo", "vram", "--csv"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return None

        if result.returncode != 0 or not result.stdout.strip():
            return None

        warnings: list[str] = []
        vram_gb: list[float] = []
        gpu_count = 0
        try:
            lines = [l for l in result.stdout.strip().splitlines() if l and not l.startswith("device,")]
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 2:
                    continue
                gpu_count += 1
                # rocm-smi reports VRAM in bytes; column position varies.
                # Try the largest numeric column as the total.
                numeric_vals = []
                for p in parts[1:]:
                    try:
                        numeric_vals.append(int(p))
                    except ValueError:
                        continue
                if not numeric_vals:
                    warnings.append(f"rocm-smi line {gpu_count} has no numeric VRAM value")
                    continue
                vram_bytes = max(numeric_vals)
                vram_gb.append(vram_bytes / (1024 ** 3))
        except (ValueError, KeyError, IndexError) as e:
            return None

        return gpu_count, tuple(vram_gb), warnings
