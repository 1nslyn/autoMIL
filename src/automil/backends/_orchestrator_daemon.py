"""autoMIL Experiment Orchestrator.

Background daemon that watches a queue directory for experiment specs,
schedules them across GPUs using best-fit bin packing with priority,
manages process lifecycles via git worktree isolation, and archives results.

Usage:
    automil start    # Start daemon
    automil status   # Show status
    automil stop     # Graceful stop
    automil submit spec.json  # Submit experiment
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dotenv import dotenv_values

from automil.runner import Runner

# ---------------------------------------------------------------------------
# Defaults (overridden by config.yaml orchestrator section)
# ---------------------------------------------------------------------------
POLL_INTERVAL_SEC = 5
SAFETY_MARGIN_GB = 2.0
DEFAULT_TIMEOUT_MIN = 150
# Saturate GPUs by default: the orchestrator's job is to pack experiments
# until VRAM runs out, not to run them serially. Projects whose workloads
# are heavier should override via config.yaml → orchestrator.max_concurrent_per_gpu.
MAX_CONCURRENT_PER_GPU = 8
DEFAULT_VRAM_ESTIMATE_GB = 1.0

# ---------------------------------------------------------------------------
# Subprocess env whitelist (CLN-02 / D-04)
# ---------------------------------------------------------------------------
# Hardcoded system-minimal whitelist applied to os.environ when building the
# experiment subprocess env. Operator secrets (OPENAI_API_KEY, WANDB_API_KEY,
# GITHUB_TOKEN, AWS_SECRET_ACCESS_KEY, ...) are NOT inherited — closing the
# HIGH-severity exfiltration vector documented in
# CONCERNS.md §"Subprocess `env` inherits the full operator environment".
#
# Consumer-specific vars (e.g. AUTOBENCH_*_ROOT) are opted in per project via
# `automil/config.yaml: env.passthrough` — see _build_subprocess_env.
_SYSTEM_ENV_WHITELIST_LITERAL: frozenset[str] = frozenset({
    "PATH", "HOME", "USER", "SHELL", "LANG", "TZ", "TMPDIR",
    "LD_LIBRARY_PATH", "PYTHONPATH",
})
# Prefix-glob: matched via str.startswith on a tuple (Python idiom).
_SYSTEM_ENV_WHITELIST_PREFIX: tuple[str, ...] = (
    "LC_", "CUDA_", "NVIDIA_", "AUTOMIL_",
)
# Keys the orchestrator owns; per-spec env CANNOT override them
# (T-00-09 mitigation — prevents GPU-mask spoofing via spec.env).
_SPEC_ENV_BLOCKED: frozenset[str] = frozenset({"AUTOMIL_GPU", "CUDA_VISIBLE_DEVICES"})

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# nvidia-smi path pinning (CLN-05)
# ---------------------------------------------------------------------------
# Resolve nvidia-smi's absolute path once at module import. On a shared host a
# PATH-shim could otherwise return spoofed VRAM numbers and trick the
# bin-packer (CONCERNS.md §"nvidia-smi invocation has no path pinning"). If
# detection fails we fall back to bare PATH lookup with a WARN — never silent
# (D-18). Resolution happens here (module-level), not on every query_gpus
# call, so the cost is paid once and tests can re-resolve via importlib.reload.
_resolved_nvidia_smi = shutil.which("nvidia-smi")
NVIDIA_SMI_PATH = _resolved_nvidia_smi or "nvidia-smi"
if _resolved_nvidia_smi:
    logger.info("nvidia-smi resolved to %s", NVIDIA_SMI_PATH)
else:
    logger.warning(
        "nvidia-smi not found via shutil.which; falling back to bare PATH lookup. "
        "GPU state may be unreliable on hosts with shimmed PATH."
    )


def _find_automil_dir() -> Path:
    """Walk up from cwd to find automil/config.yaml. Returns the automil/ dir."""
    p = Path.cwd()
    while p != p.parent:
        if (p / "automil" / "config.yaml").exists():
            return p / "automil"
        p = p.parent
    raise RuntimeError(
        "No automil/config.yaml found. Run 'automil init' in your project root."
    )


def _find_git_root(start: Path | None = None) -> Path:
    """Walk up from *start* (default: cwd) to find the git repo root."""
    p = (start or Path.cwd()).resolve()
    while p != p.parent:
        if (p / ".git").exists():
            return p
        p = p.parent
    raise RuntimeError("Not inside a git repository.")


# ---------------------------------------------------------------------------
# PID-file starttime cross-check (CLN-04 / D-17)
# ---------------------------------------------------------------------------
# PID reuse on Linux can cause a stale PID file to claim ownership of an
# unrelated process. Compare both pid AND /proc/<pid>/stat starttime_ticks
# before signalling. Linux-only is acceptable per PROJECT.md Constraints.

def _parse_starttime_from_stat_line(line: str) -> int:
    """Parse field 22 (1-indexed) — process starttime in clock ticks — from a /proc/<pid>/stat line.

    The `comm` field (#2) is wrapped in parentheses and CAN contain spaces.
    Find the LAST ')' to skip past comm, then split the suffix on whitespace.
    """
    end_comm = line.rfind(")")
    if end_comm == -1:
        raise ValueError(f"Malformed /proc/<pid>/stat line: {line!r}")
    # After the ')' there's a space, then field 3 (state) onwards.
    suffix = line[end_comm + 1:].strip()
    fields = suffix.split()
    # suffix starts at field 3; starttime is field 22 (1-indexed) -> suffix index 22 - 3 = 19.
    if len(fields) < 20:
        raise ValueError(f"/proc/<pid>/stat has fewer fields than expected: {len(fields)}")
    return int(fields[19])


def _read_proc_starttime(pid: int) -> int | None:
    """Read /proc/<pid>/stat field 22 (starttime_ticks). Returns None if pid not found or /proc unavailable."""
    try:
        line = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, PermissionError, OSError):
        return None
    try:
        return _parse_starttime_from_stat_line(line)
    except ValueError as e:
        logger.warning("Could not parse /proc/%d/stat: %s", pid, e)
        return None


def _is_pid_alive_with_starttime(pid: int, expected_starttime_ticks: int) -> bool:
    """True iff the process at *pid* is running AND its starttime matches the recorded value.

    The starttime check defends against PID reuse: a previous daemon's PID
    could be reassigned to an unrelated process; signalling that PID would
    be wrong. See CONCERNS.md §"PID-file stale-detection uses os.kill(pid, 0)".
    """
    actual = _read_proc_starttime(pid)
    if actual is None:
        return False
    return actual == expected_starttime_ticks


def _write_pid_file(pid_file: Path) -> None:
    """Write PID file as JSON with pid + starttime_ticks + starttime_iso (D-17 shape)."""
    my_pid = os.getpid()
    starttime = _read_proc_starttime(my_pid)
    if starttime is None:
        # /proc unavailable (non-Linux test env); record what we can.
        starttime = 0
    payload = {
        "pid": my_pid,
        "starttime_ticks": starttime,
        "starttime_iso": datetime.now().isoformat(),
    }
    pid_file.write_text(json.dumps(payload) + "\n")


def _load_pid_file(pid_file: Path) -> dict | None:
    """Load pid_file as JSON. Returns None on legacy plain-int, invalid JSON, or missing keys.

    None means "treat as stale" — the caller should unlink and proceed as
    if no daemon were running. Documented for plain-int compat: an in-flight
    daemon started before this change uses the legacy format; on first
    post-upgrade cmd_start, the legacy file is treated as stale and
    unlinked, the operator restarts and gets the new format.
    """
    try:
        data = json.loads(pid_file.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    if not {"pid", "starttime_ticks", "starttime_iso"}.issubset(data.keys()):
        return None
    return data


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class GPUInfo:
    index: int
    total_mb: int
    free_mb: int
    utilization: int

    @property
    def free_gb(self) -> float:
        return self.free_mb / 1024


@dataclass
class RunningExperiment:
    id: str
    spec: dict
    gpu: int
    process: subprocess.Popen
    log_file: object  # file handle
    log_path: Path
    started_at: float
    timeout_at: float
    estimated_vram_gb: float


@dataclass(frozen=True)
class _NodeHandle:
    """Minimal handle carrying only node_id; used by _running_in_cell (CAP-02 / D-114).

    Lets _tick_cells pass handle.node_id to self.backend.cancel() without
    depending on the full backends.base.JobHandle (which carries opaque_id /
    submitted_at fields that the daemon doesn't track at this layer).  Tests
    may inject a real Backend whose cancel() receives this handle.
    """

    node_id: str


# ---------------------------------------------------------------------------
# GPU monitoring
# ---------------------------------------------------------------------------
def query_gpus() -> list[GPUInfo]:
    """Query nvidia-smi for GPU state.

    Uses the path resolved at module import (NVIDIA_SMI_PATH) to defend
    against PATH-shim spoofing on shared hosts (CLN-05).
    """
    try:
        result = subprocess.run(
            [
                NVIDIA_SMI_PATH,
                "--query-gpu=index,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpus.append(GPUInfo(
                    index=int(parts[0]),
                    total_mb=int(parts[1]),
                    free_mb=int(parts[2]),
                    utilization=int(parts[3]),
                ))
        return gpus
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        logger.warning(f"nvidia-smi failed: {e}")
        return []


# ---------------------------------------------------------------------------
# D-170 / D-171: Cross-backend log unification helpers
# ---------------------------------------------------------------------------

def _atomic_write_lines(path: Path, lines: list[str]) -> None:
    """Atomic write of log lines (D-170 / Phase 0 D-25 atomic-write pattern).

    Uses tempfile.mkstemp neighbour + os.replace (NOT git checkout — Leo memory
    feedback_never_blind_checkout: rollback uses os.unlink).

    Args:
        path: target path (parent dir created if absent).
        lines: list of strings; pre-existing newline characters are preserved.
            If callers provide raw lines without newlines, the writer does NOT
            add them — this matches LocalBackend's log_iter() semantics where
            splitlines(keepends=True) is used at yield time.
    """
    import os as _os                # noqa: PLC0415; module-scope helper
    import tempfile as _tempfile    # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = _tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with _os.fdopen(tmp_fd, "w") as f:
            f.writelines(lines)
        _os.replace(tmp_path, str(path))
    except Exception:
        try:
            _os.unlink(tmp_path)  # rollback per memory:feedback_never_blind_checkout
        except OSError:
            pass
        raise


def _drain_log_iter_with_timeout(backend, handle, timeout: float = 60.0) -> list[str]:
    """Drain backend.log_iter(handle) with a hard timeout (D-170).

    Spawns a daemon thread that consumes the iterator. After `timeout` seconds
    the wrapper returns the lines collected so far; backends whose log_iter
    doesn't close within the timeout are treated as a contract violation
    (logged warning). The thread is daemon=True so abandoned drainers are
    cleaned up on daemon exit.
    """
    import threading as _threading  # noqa: PLC0415

    lines: list[str] = []
    done = _threading.Event()

    def _drain() -> None:
        try:
            for line in backend.log_iter(handle):
                lines.append(line)
        except Exception as exc:
            logger.warning("_drain_log_iter_with_timeout: log_iter raised %s", exc)
        finally:
            done.set()

    t = _threading.Thread(target=_drain, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if not done.is_set():
        node_label = getattr(handle, "node_id", "<unknown>")
        logger.warning(
            "log_iter for %s did not close within %.1fs — force-closing (D-170 contract violation).",
            node_label, timeout,
        )
    return lines


def _symlink_slurm_logs(automil_dir: Path, archive_node_dir: Path, spec_data: dict) -> None:
    """D-171: symlink submitit's native stdout/stderr logs into archive/<id>/.

    Creates archive/<id>/slurm-stdout.out and archive/<id>/slurm-stderr.err as
    symlinks (NOT copies) pointing at submitit-logs/{opaque_id}_0_log.out/.err.
    Reduces disk usage — submitit already owns those files on disk.

    This is a module-level function (NOT a method) so it can be tested without
    instantiating ExperimentOrchestrator.
    """
    opaque_id = spec_data.get("opaque_id", "")
    if not opaque_id:
        return
    submitit_logs = (
        automil_dir / "orchestrator" / "running" / "slurm" / "submitit-logs"
    )
    stdout_src = submitit_logs / f"{opaque_id}_0_log.out"
    stderr_src = submitit_logs / f"{opaque_id}_0_log.err"
    stdout_dst = archive_node_dir / "slurm-stdout.out"
    stderr_dst = archive_node_dir / "slurm-stderr.err"
    if stdout_src.exists() and not stdout_dst.exists():
        try:
            stdout_dst.symlink_to(stdout_src.resolve())
        except OSError as exc:
            logger.warning("D-171 stdout symlink failed: %s", exc)
    if stderr_src.exists() and not stderr_dst.exists():
        try:
            stderr_dst.symlink_to(stderr_src.resolve())
        except OSError as exc:
            logger.warning("D-171 stderr symlink failed: %s", exc)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class ExperimentOrchestrator:
    """Schedules and manages experiment lifecycle across GPUs."""

    def __init__(self, project_root: Path | None = None,
                 automil_dir: Path | None = None):
        self.automil_dir = automil_dir or _find_automil_dir()
        self.project_root = project_root or _find_git_root()
        self.orch_dir = self.automil_dir / "orchestrator"
        self.queue_dir = self.orch_dir / "queue"
        # D-169: running_dir is no longer a single flat attribute — resolved per-backend
        # via _backend_running_dir(name). The base running root remains for the
        # startup guardrail check (D-168) and log unification (D-170).
        self.running_root = self.orch_dir / "running"
        # Backward alias: points at running/local/ so all existing internal
        # LocalBackend dispatch paths (lines 709, 771, 816, 852, 857, 917, 980)
        # resolve to the correct namespaced directory without further modification.
        # New code MUST call self._backend_running_dir(backend_name) instead.
        self.running_dir = self.running_root / "local"
        self.archive_dir = self.orch_dir / "archive"
        self.completed_dir = self.orch_dir / "completed"
        self.results_tsv = self.automil_dir / "results.tsv"
        self.pid_file = self.orch_dir / "orchestrator.pid"
        self.log_file = self.orch_dir / "orchestrator.log"
        self.gpu_state_file = self.orch_dir / "gpu_state.json"

        self.runner = Runner(self.project_root)

        # CR-01 fix: ExperimentGraph instance initialized here so _handle_completion
        # and _handle_cap_killed_completion both receive a valid graph object at
        # daemon runtime, without test injection. write_terminal_state uses
        # graph.path and graph._technique_map (loaded fresh per locked_update call),
        # so constructing the instance here is sufficient — no eager load of nodes.
        from automil.graph import ExperimentGraph
        self.graph = ExperimentGraph(
            path=self.automil_dir / "graph.json",
            technique_map=None,  # loaded fresh inside locked_update on each write
        )

        # Load config
        config_path = self.automil_dir / "config.yaml"
        if config_path.exists():
            try:
                import yaml
                self.config = yaml.safe_load(config_path.read_text())
            except ImportError:
                self.config = self._parse_yaml_fallback(config_path)
        else:
            self.config = {}

        # Run script config
        run_config = self.config.get("run", {}) if self.config else {}
        self.run_script = run_config.get("script", "train.py")
        self.run_command = run_config.get("command")

        orch_cfg = self.config.get("orchestrator", {}) if self.config else {}
        self.poll_interval = orch_cfg.get("poll_interval_sec", POLL_INTERVAL_SEC)
        self.safety_margin_gb = orch_cfg.get("safety_margin_gb", SAFETY_MARGIN_GB)
        self.default_timeout = orch_cfg.get("default_timeout_min", DEFAULT_TIMEOUT_MIN)
        self.max_per_gpu = orch_cfg.get("max_concurrent_per_gpu", MAX_CONCURRENT_PER_GPU)
        self.default_vram = orch_cfg.get("default_vram_estimate_gb", DEFAULT_VRAM_ESTIMATE_GB)

        # CLN-02 / D-04: env.passthrough — literal var names the operator
        # explicitly opts in to forward into experiment subprocesses. The
        # config layer accepts only a list of strings (no globs — globs live
        # in the hardcoded system whitelist so the operator cannot widen the
        # surface from config). Missing vars WARN once at startup and never
        # block scheduling.
        env_cfg = self.config.get("env", {}) if self.config else {}
        raw_passthrough = env_cfg.get("passthrough", []) or []
        if not isinstance(raw_passthrough, list):
            logger.warning(
                "env.passthrough must be a list of var names; got %r — ignoring.",
                type(raw_passthrough).__name__,
            )
            raw_passthrough = []
        self._env_passthrough: list[str] = [str(k) for k in raw_passthrough]
        for key in self._env_passthrough:
            if key not in os.environ:
                logger.warning(
                    "env.passthrough declares %s but it is not set in the orchestrator's "
                    "environment — the var will be unavailable to experiment subprocesses.",
                    key,
                )

        # Runtime state
        self.running: dict[str, RunningExperiment] = {}
        self.gpu_allocations: dict[int, list[str]] = {}
        self.counter = 0
        self.draining = False
        self._shutdown = False
        self._timed_out: dict[str, bool] = {}
        # node_id -> wall-clock deadline at which an outstanding SIGTERM
        # should escalate to SIGKILL. Set by _kill_experiment; consumed by
        # _check_running. Grace period is 5s, matching _handle_timeout.
        self._pending_sigkill_at: dict[str, float] = {}
        # Phase 4 (CAP-02): optional Backend instance for cancel dispatch.
        # Injected by tests (or future Backend integration) to receive
        # cancel(handle, signal=SIGTERM) calls from _tick_cells.  When None,
        # _tick_cells falls back to _kill_experiment (direct os.killpg path).
        self.backend: object | None = None

        # Detect GPUs
        gpus = query_gpus()
        for g in gpus:
            self.gpu_allocations[g.index] = []
        if not self.gpu_allocations:
            logger.warning("No GPUs detected, using GPU 0 as fallback")
            self.gpu_allocations[0] = []

        # Load .env from project root so worktree processes inherit env vars
        # (worktrees don't contain .env since it's typically gitignored)
        self._load_dotenv()

        # Ensure directories.
        # NOTE: we create running_root (the parent running/ dir) but NOT the
        # per-backend running/local/ subdirectory here. The _backend_running_dir
        # helper creates backend subdirs on demand, and the D-168 guardrail in
        # run() checks whether running/local/ (or /slurm/ or /ray/) EXISTS as a
        # signal that this installation is already on the 6.x namespaced layout.
        # Creating running/local/ in __init__ would defeat that guardrail by
        # making every fresh daemon startup look "already migrated".
        for d in (self.queue_dir, self.running_root, self.archive_dir, self.completed_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Load persisted state (don't recover orphans until run() is called)
        self._load_state(recover=False)

    def _backend_running_dir(self, backend_name: str) -> Path:
        """Return orch_dir / 'running' / <backend_name>; create on demand (D-169).

        Per-backend namespacing was introduced in Phase 6 (BCK-05/06). Default
        fallback is 'local' for legacy nodes without metadata.backend (Phase 2 D-76).
        New code (cancel.py, reconcile.py, cell.py, log unification) MUST call
        this helper instead of accessing self.running_dir directly.
        """
        if not backend_name:
            backend_name = "local"
        path = self.running_root / backend_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _parse_yaml_fallback(config_path: Path) -> dict:
        """Minimal YAML parsing when PyYAML is not installed."""
        lines = config_path.read_text().splitlines()
        orch: dict = {}
        in_orch = False
        for line in lines:
            if line.strip() == "orchestrator:":
                in_orch = True
                continue
            if in_orch:
                if line and not line[0].isspace():
                    break
                parts = line.strip().split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = parts[1].strip()
                    try:
                        orch[key] = float(val) if "." in val else int(val)
                    except ValueError:
                        orch[key] = val
        return {"orchestrator": orch}

    def _load_dotenv(self) -> None:
        """Load .env files from the project root into os.environ.

        Worktrees are detached git checkouts and don't contain .env
        (which is typically gitignored). Loading here ensures the
        orchestrator's child processes inherit the variables.

        The set of files is configurable via ``env.dotenv_files`` in
        ``automil/config.yaml`` (list of paths relative to project_root).
        ``.env`` at the project root is always loaded if it exists; the
        config field is additive and consumer-specific (consumer projects
        whose .env lives outside the project root add the relative path
        here).

        Uses python-dotenv so quoted values, the ``export`` prefix, and
        inline ``# comments`` after unquoted values are handled
        correctly (CLN-03; see CONCERNS.md §"Naive .env parser").
        Pre-existing entries in ``os.environ`` are preserved
        (``setdefault`` semantic) — the shell wins over the file.
        """
        env_cfg = (self.config or {}).get("env", {}) if isinstance(self.config, dict) else {}
        extra_files = env_cfg.get("dotenv_files", []) or []

        candidates: list[Path] = [self.project_root / ".env"]
        for rel in extra_files:
            rel_str = str(rel).strip()
            if not rel_str:
                continue
            p = Path(rel_str)
            if p.is_absolute():
                logger.warning(
                    "env.dotenv_files entry %r is absolute; ignored "
                    "(paths must be relative to project root)", rel_str,
                )
                continue
            candidates.append(self.project_root / p)

        for env_file in candidates:
            if not env_file.is_file():
                continue
            parsed = dotenv_values(env_file)
            for key, value in parsed.items():
                if value is None:
                    continue
                # Don't override existing env vars (preserves prior semantic).
                if key not in os.environ:
                    os.environ[key] = value
                    logger.debug("Loaded env var %s from %s", key, env_file)

    # --- State persistence ---

    def _load_state(self, recover: bool = True):
        """Load counter from persisted state. Only recover orphans if requested."""
        if self.gpu_state_file.exists():
            try:
                state = json.loads(self.gpu_state_file.read_text())
                self.counter = state.get("counter", 0)
            except (json.JSONDecodeError, KeyError):
                pass

        if recover:
            self._recover_orphans()

    def _save_state(self):
        """Persist GPU state and counters to disk."""
        gpus = query_gpus()
        gpu_data = {}
        for g in gpus:
            running_on = self.gpu_allocations.get(g.index, [])
            alloc_vram = sum(
                self.running[eid].estimated_vram_gb
                for eid in running_on
                if eid in self.running
            )
            gpu_data[str(g.index)] = {
                "total_mb": g.total_mb,
                "free_mb": g.free_mb,
                "schedulable_free_gb": round(g.free_gb - self.safety_margin_gb - alloc_vram, 1),
                "running": running_on,
                "utilization_pct": g.utilization,
            }

        state = {
            "counter": self.counter,
            "last_updated": datetime.now().isoformat(),
            "gpus": gpu_data,
            "queue_depth": len(list(self.queue_dir.glob("*.json"))),
            "total_running": len(self.running),
            "total_completed": len(list(self.completed_dir.glob("*.json"))),
        }
        self.gpu_state_file.write_text(json.dumps(state, indent=2) + "\n")

    def _recover_orphans(self):
        """Mark orphaned running experiments as crashed and clean up worktrees.

        SIGKILLs the recorded process group before marking the node crashed:
        when the daemon restarts after a mid-run crash, any pending SIGKILL
        escalation from a prior ``_kill_experiment`` SIGTERM was lost with the
        previous process's memory. The training subprocess may still be alive
        and holding VRAM. Reading ``metadata.pid`` + ``metadata.starttime_ticks``
        from the running spec lets us reap the orphan before marking it crashed
        (D-17 starttime cross-check defends against PID reuse).
        """
        if not self.running_dir.exists():
            return
        for f in self.running_dir.glob("*.json"):
            try:
                spec = json.loads(f.read_text())
                node_id = spec.get("id", f.stem)
                logger.info(f"Orphaned experiment {node_id} found, marking as crashed")

                self._sigkill_orphan_pg(node_id, spec)

                archive = self.archive_dir / node_id
                archive.mkdir(parents=True, exist_ok=True)
                result = {"status": "crash", "error": "Orchestrator restarted while running"}
                (archive / "result.json").write_text(json.dumps(result, indent=2))
                (self.completed_dir / f"{node_id}.json").write_text(json.dumps({
                    "id": node_id,
                    "status": "crash",
                    "completed_at": datetime.now().isoformat(),
                }, indent=2))

                f.unlink()

                wt = self.runner.worktree_path(node_id)
                if wt.exists():
                    self.runner.cleanup_worktree(wt)
            except Exception:
                continue

    def _sigkill_orphan_pg(self, node_id: str, spec: dict) -> None:
        """SIGKILL an orphaned process group recorded in the running spec.

        Reads ``spec['metadata']['pid']`` + ``spec['metadata']['starttime_ticks']``
        (written by ``_launch``). Verifies the PID's starttime matches before
        signalling (PID reuse defence, CLN-04 / D-17). Soft-fails: logs and
        continues if the spec lacks PID metadata (legacy specs from pre-fix
        launches) or the process has already exited.
        """
        meta = spec.get("metadata") or {}
        pid_raw = meta.get("pid")
        starttime_recorded = meta.get("starttime_ticks")
        if pid_raw is None:
            # Legacy spec (pre-PID-persistence) or backend that doesn't track PIDs.
            return
        try:
            pid = int(pid_raw)
        except (TypeError, ValueError):
            logger.warning("orphan %s has unparseable pid %r; skipping SIGKILL", node_id, pid_raw)
            return
        if starttime_recorded is not None and not _is_pid_alive_with_starttime(pid, int(starttime_recorded)):
            logger.info("orphan %s PID %d is gone or reused; no SIGKILL needed", node_id, pid)
            return
        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError:
            return
        except OSError as exc:
            logger.warning("orphan %s: getpgid(%d) failed: %s", node_id, pid, exc)
            return
        try:
            os.killpg(pgid, signal.SIGKILL)
            logger.warning(
                "orphan %s: SIGKILLed pgid %d (daemon restart recovery; freeing VRAM)",
                node_id, pgid,
            )
        except ProcessLookupError:
            pass
        except OSError as exc:
            logger.warning("orphan %s: killpg(%d, SIGKILL) failed: %s", node_id, pgid, exc)

    # --- Scheduling ---

    def _get_pending(self) -> list[dict]:
        """Read and sort pending experiments from queue."""
        pending = []
        for f in sorted(self.queue_dir.glob("*.json")):
            try:
                spec = json.loads(f.read_text())
                spec["_file"] = f
                pending.append(spec)
            except (json.JSONDecodeError, Exception) as e:
                logger.error(f"Bad spec {f}: {e}")
        # Sort by priority ASC, then submitted_at ASC
        pending.sort(key=lambda s: (s.get("priority", 2), s.get("submitted_at", "")))
        return pending

    def _find_best_gpu(self, needed_gb: float) -> int | None:
        """Best-fit bin packing: find GPU with least free VRAM that still fits."""
        gpus = query_gpus()
        candidates = []

        for g in gpus:
            running_on = self.gpu_allocations.get(g.index, [])
            if len(running_on) >= self.max_per_gpu:
                continue
            alloc_vram = sum(
                self.running[eid].estimated_vram_gb
                for eid in running_on
                if eid in self.running
            )
            schedulable = g.free_gb - self.safety_margin_gb - alloc_vram
            if schedulable >= needed_gb:
                candidates.append((g.index, schedulable))

        if not candidates:
            return None

        # Best-fit: pick GPU with LEAST schedulable free (tightest fit)
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]

    def _pre_launch_check(self, gpu_id: int, needed_gb: float) -> bool:
        """Final VRAM check right before launch."""
        gpus = query_gpus()
        for g in gpus:
            if g.index == gpu_id:
                return g.free_gb >= needed_gb + self.safety_margin_gb
        return False

    # --- Experiment lifecycle ---

    def _build_subprocess_env(
        self,
        *,
        gpu_id: int,
        node_id: str,
        archive: Path,
        spec: dict,
    ) -> dict[str, str]:
        """Build the subprocess environment from a hardcoded whitelist + config passthrough.

        Replaces the previous ``env = {**os.environ, ...}`` leak (CLN-02 / D-04;
        see CONCERNS.md §"Subprocess `env` inherits the full operator environment").

        Layering (highest precedence wins):
          1. System whitelist (literal + prefix-glob match against ``os.environ``).
          2. Config passthrough (literal names from ``automil/config.yaml: env.passthrough``).
          3. Orchestrator-injected fixed keys (always overrides 1 + 2).
          4. Per-spec ``spec.env`` (last-write-wins, except ``_SPEC_ENV_BLOCKED``).
        """
        # D-199 / DEC-01: Consumer-specific env vars and PYTHONPATH overlay
        # (formerly injected here in Phase 0) are removed; consumers wire
        # them via env.passthrough in automil/config.yaml (D-202).
        env: dict[str, str] = {}

        # 1. System whitelist (literal + prefix-glob).
        for key, value in os.environ.items():
            if key in _SYSTEM_ENV_WHITELIST_LITERAL or key.startswith(_SYSTEM_ENV_WHITELIST_PREFIX):
                env[key] = value

        # 2. Config-driven passthrough (literal names only).
        for key in self._env_passthrough:
            if key in os.environ:
                env[key] = os.environ[key]

        # 3. Orchestrator-injected (always overrides 1 + 2).
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["AUTOMIL_GPU"] = "0"
        env["AUTOMIL_DESC"] = spec.get("description", "")
        env["AUTOMIL_NODE_ID"] = node_id
        env["AUTOMIL_RESULTS_DIR"] = str(archive.resolve())

        # Phase 4 (D-120): inject fold count so SIGTERM handler in the training
        # script can read it via automil.runtime_helpers.get_fold_count().
        # Resolved from automil/config.yaml: training.fold_count; fallback 5.
        try:
            import yaml as _yaml
            _cfg = _yaml.safe_load((self.automil_dir / "config.yaml").read_text()) or {}
            _fold_count = int((_cfg.get("training") or {}).get("fold_count", 5))
        except Exception:
            _fold_count = 5
        env["AUTOMIL_FOLD_COUNT"] = str(_fold_count)

        # 4. Per-spec env (last-write-wins, except blocked keys).
        for k, v in spec.get("env", {}).items():
            if k not in _SPEC_ENV_BLOCKED:
                env[k] = str(v)

        return env

    def _launch(self, spec: dict, gpu_id: int):
        """Launch an experiment in an isolated git worktree."""
        node_id = spec["id"]
        archive = self.archive_dir / node_id
        archive.mkdir(parents=True, exist_ok=True)

        # Save spec (without internal keys)
        spec_clean = {k: v for k, v in spec.items() if k not in ("_file",)}
        (archive / "spec.json").write_text(json.dumps(spec_clean, indent=2))

        # Remove from queue before attempting launch (prevents infinite retry)
        src_file = spec.get("_file")
        if src_file and Path(src_file).exists():
            Path(src_file).unlink()

        # Reject specs without an explicit base_commit. submit always pins
        # the parent SHA at queue-time; any path that bypasses this would
        # silently resolve "HEAD" at run time (parent drift between submit
        # and launch). A NULL/missing base_commit is a programmer error,
        # not something to paper over with a runtime default.
        base_commit = spec.get("base_commit")
        if not base_commit:
            logger.error(
                "Spec for %s has no base_commit; refusing to launch. "
                "All queued specs must pin the parent SHA at submit time.",
                node_id,
            )
            self._mark_crashed(
                node_id, spec,
                "spec missing required 'base_commit' field — submit must "
                "pin the parent SHA at queue-time to prevent runtime HEAD "
                "drift.",
            )
            return
        try:
            wt_path = self.runner.create_worktree(base_commit, node_id)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create worktree for {node_id}: {e}")
            self._mark_crashed(node_id, spec, f"Worktree creation failed: {e}")
            return

        overlay_dir = spec.get("overlay_dir")
        deletions = spec.get("deletions")
        if overlay_dir:
            self.runner.apply_overlay(
                wt_path, self.orch_dir / overlay_dir, deletions=deletions
            )

        # CLN-02 / D-04 + DEC-01 / D-199: build env from explicit whitelist +
        # config passthrough. Consumer-specific vars (formerly auto-injected
        # by this block in Phase 0) are now opted in per project via
        # automil/config.yaml: env.passthrough (D-202).
        env = self._build_subprocess_env(
            gpu_id=gpu_id,
            node_id=node_id,
            archive=archive,
            spec=spec,
        )

        log_path = archive / "run.log"
        log_fh = open(log_path, "w")
        try:
            if self.run_command:
                cmd = shlex.split(self.run_command)
            else:
                cmd = [sys.executable, self.run_script]
            process = subprocess.Popen(
                cmd,
                cwd=str(wt_path),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        except Exception as e:
            log_fh.close()
            logger.error(f"Failed to launch {node_id}: {e}")
            self._mark_crashed(node_id, spec, str(e))
            self.runner.cleanup_worktree(wt_path)
            return

        timeout_min = spec.get("timeout_min", self.default_timeout)
        estimated_vram = spec.get("estimated_vram_gb", self.default_vram)

        self.running[node_id] = RunningExperiment(
            id=node_id,
            spec=spec,
            gpu=gpu_id,
            process=process,
            log_file=log_fh,
            log_path=log_path,
            started_at=time.time(),
            timeout_at=time.time() + timeout_min * 60,
            estimated_vram_gb=estimated_vram,
        )
        self.gpu_allocations.setdefault(gpu_id, []).append(node_id)

        # Copy spec to running dir for orphan recovery.
        # Use _backend_running_dir to ensure running/local/ exists (created on demand
        # per D-169; __init__ no longer pre-creates the backend subdir).
        # The running spec embeds the launched PID (+ starttime_ticks when
        # readable from /proc) so a daemon-restart orphan recovery can SIGKILL
        # the leaked pgrp before marking the node crashed. CLN-04 starttime
        # cross-check defends against PID reuse; if starttime is unreadable at
        # launch (non-Linux test env, /proc unavailable, fork race), we OMIT
        # the field — recovery then skips the cross-check and signals anyway,
        # accepting the small reuse risk over the larger VRAM-leak risk.
        running_spec_path = self._backend_running_dir("local") / f"{node_id}.json"
        running_spec_payload = dict(spec_clean)
        running_spec_meta = dict(running_spec_payload.get("metadata") or {})
        try:
            recorded_pgid = os.getpgid(process.pid)
        except OSError:
            recorded_pgid = process.pid
        running_spec_meta["pid"] = process.pid
        running_spec_meta["pgid"] = recorded_pgid
        recorded_starttime = _read_proc_starttime(process.pid)
        if recorded_starttime is not None:
            running_spec_meta["starttime_ticks"] = recorded_starttime
        running_spec_payload["metadata"] = running_spec_meta
        running_spec_path.write_text(json.dumps(running_spec_payload, indent=2))

        logger.info(
            f"Launched {node_id} on GPU {gpu_id} "
            f"(PID {process.pid}, est. {estimated_vram}GB, timeout {timeout_min}min)"
        )

    def _running_in_cell(self, cell_id: str) -> list:
        """Return _NodeHandle list for in-self.running experiments tagged with cell_id.

        cell_id matching uses spec["metadata"]["cell_id"] (set by submit, Plan 04-06).
        Legacy nodes without a cell_id never match — they are immune to cap
        enforcement (D-117 backward compat).

        Returns:
            List of _NodeHandle(node_id=...) for matching experiments.
        """
        result = []
        for node_id, exp in self.running.items():
            spec_meta = (exp.spec or {}).get("metadata", {}) or {}
            if spec_meta.get("cell_id") == cell_id:
                result.append(_NodeHandle(node_id=node_id))
        return result

    def _tick_cells(self) -> None:
        """Advance cap state machine for all cells (CAP-02 / D-114).

        Idempotent: re-running on an already-transitioned cell is a no-op
        because next_status returns the same value when consumed/running counts
        are stable. TERMINATING fires backend.cancel(SIGTERM) on all running
        in-cell experiments AFTER annotating their running/<node>.json with
        metadata.cancel_reason='cap' so reconcile_budget_kill can distinguish
        cap kills from operator cancels (Pitfall 4).

        Process-group kill is the backend's responsibility (D-115).
        """
        import signal as _sig
        from dataclasses import replace
        from automil.cells import (
            accrue_active,
            list_cells,
            next_status,
            read_last_action_at,
            write_cell,
            CellStatus,
        )

        now = time.time()
        # Resolve cells_dir from the orchestrator's explicit automil_dir, not
        # via the cwd-walking _find_automil_dir() fallback. Tests construct
        # the orchestrator with a tmp_path automil_dir; the global fallback
        # would find the host project's untracked `automil/` instead.
        cells_dir = self.automil_dir / "cells"
        if not cells_dir.exists():
            logger.debug("_tick_cells: no cells dir at %s; skipping", cells_dir)
            return
        # P2.2: project-level agent-activity marker drives agent_active billing.
        last_action_at = read_last_action_at(self.automil_dir)
        for cell in list_cells(cells_dir):
            # Advance the agent_active budget BEFORE the state machine so
            # next_status sees this tick's consumed time. No-op for wall_clock.
            cell = accrue_active(cell, now, last_action_at)
            running = self._running_in_cell(cell.cell_id)
            new_status = next_status(cell, now, len(running))
            if new_status == cell.status:
                # Persist the accrual (consumed_active_seconds / last_tick_at)
                # even when the status is unchanged, else it never advances.
                if cell.mode == "agent_active":
                    write_cell(cell, cells_dir)
                continue
            if new_status == CellStatus.TERMINATING:
                for handle in running:
                    # D-124 / Pitfall 4: write cancel_reason='cap' BEFORE
                    # calling cancel so reconcile_budget_kill can detect cap kills
                    # even if the SIGTERM handler races the annotation write.
                    running_spec_path = self.running_dir / f"{handle.node_id}.json"
                    if running_spec_path.exists():
                        try:
                            spec_data = json.loads(running_spec_path.read_text())
                            spec_data.setdefault("metadata", {})["cancel_reason"] = "cap"
                            running_spec_path.write_text(json.dumps(spec_data, indent=2))
                        except (json.JSONDecodeError, OSError) as exc:
                            logger.warning(
                                "Could not annotate cancel_reason for %s: %s",
                                handle.node_id, exc,
                            )
                    if self.backend is not None:
                        try:
                            self.backend.cancel(handle, signal=_sig.SIGTERM)
                        except Exception as exc:
                            logger.warning(
                                "backend.cancel failed for %s: %s", handle.node_id, exc
                            )
                    else:
                        # Fallback: direct process-group kill (production path)
                        self._kill_experiment(handle.node_id, _sig.SIGTERM)
            write_cell(replace(cell, status=new_status), cells_dir)
            logger.info(
                "_tick_cells: %s transitioned %s -> %s (running=%d)",
                cell.cell_id[:8], cell.status.value, new_status.value, len(running),
            )

    def _check_running(self):
        """Poll running experiments for completion or timeout.

        Also escalates pending SIGTERM cancels (recorded in
        ``self._pending_sigkill_at``) to SIGKILL once their grace period
        has elapsed. Without this, a trainer that ignores SIGTERM (raw C
        extension, blocked CUDA kernel) would sit in VRAM indefinitely
        after a cap-driven or operator cancel.
        """
        # Drop pending-sigkill entries whose experiment is no longer
        # tracked here (completion or timeout path popped self.running
        # without clearing the deadline). Otherwise the dict grows
        # unbounded across the daemon's lifetime.
        orphans = [
            eid for eid in self._pending_sigkill_at if eid not in self.running
        ]
        for eid in orphans:
            self._pending_sigkill_at.pop(eid, None)

        now = time.time()
        for exp_id, exp in list(self.running.items()):
            retcode = exp.process.poll()
            if retcode is not None:
                self._pending_sigkill_at.pop(exp_id, None)
                self._handle_completion(exp_id, retcode)
                continue
            if now > exp.timeout_at:
                self._handle_timeout(exp_id)
                continue
            deadline = self._pending_sigkill_at.get(exp_id)
            if deadline is not None and now >= deadline:
                self._escalate_to_sigkill(exp_id)

    def _escalate_to_sigkill(self, node_id: str) -> None:
        """SIGKILL the process group when a previously-issued SIGTERM
        wasn't enough. Removes the pending-deadline entry whether or not
        the kill succeeded; the next poll will reap the exit code."""
        self._pending_sigkill_at.pop(node_id, None)
        exp = self.running.get(node_id)
        if exp is None:
            return
        pid = exp.process.pid
        if exp.process.poll() is not None:
            return  # already exited between deadline check and now
        logger.warning(
            "_escalate_to_sigkill: SIGTERM grace expired for %s (PID %d); "
            "sending SIGKILL", node_id, pid,
        )
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as exc:
            logger.warning("_escalate_to_sigkill: killpg failed: %s", exc)

    def _read_fold_count_for_node(self, node_id: str) -> int:
        """Read AUTOMIL_FOLD_COUNT from the node spec env, or fall back to config.

        Priority:
            1. spec.env["AUTOMIL_FOLD_COUNT"] (set by _build_subprocess_env at launch)
            2. automil/config.yaml: training.fold_count
            3. Hard fallback: 5 (Leo's paper-campaign default)
        """
        for path in (
            self.running_dir / f"{node_id}.json",
            self.archive_dir / node_id / "spec.json",
        ):
            if path.exists():
                try:
                    spec = json.loads(path.read_text())
                    env = (spec.get("env") or {}) if isinstance(spec, dict) else {}
                    if "AUTOMIL_FOLD_COUNT" in env:
                        return int(env["AUTOMIL_FOLD_COUNT"])
                except (json.JSONDecodeError, OSError, ValueError, TypeError):
                    continue
        # Fall back: read automil/config.yaml training.fold_count
        try:
            import yaml as _yaml
            cfg = _yaml.safe_load((self.automil_dir / "config.yaml").read_text()) or {}
            return int((cfg.get("training") or {}).get("fold_count", 5))
        except Exception:
            return 5

    def _read_backend_name_for_node(self, node_id: str) -> str:
        """Read metadata.backend from the running spec (any backend subdir) or archive spec.

        Returns 'local' as fallback (Phase 2 D-76 legacy compatibility).
        """
        for backend_subdir in ("local", "slurm", "ray"):
            candidate = self.running_root / backend_subdir / f"{node_id}.json"
            if candidate.exists():
                try:
                    payload = json.loads(candidate.read_text())
                    return payload.get("backend") or backend_subdir
                except (json.JSONDecodeError, OSError):
                    continue
        archive_spec = self.archive_dir / node_id / "spec.json"
        if archive_spec.exists():
            try:
                payload = json.loads(archive_spec.read_text())
                return payload.get("metadata", {}).get("backend", "local")
            except (json.JSONDecodeError, OSError):
                pass
        return "local"

    def _read_running_spec(self, node_id: str, backend_name: str) -> dict:
        """Read running/<backend>/<node>.json; return {} if absent."""
        path = self.running_root / backend_name / f"{node_id}.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def _handle_completion(self, node_id: str, returncode: int):
        """Process a completed experiment: collect results, write TSV, clean up.

        Orchestrates the per-completion lifecycle by delegating to three
        single-purpose helpers (each independently testable):

          - ``_was_cap_killed_completion``  → was this a cap-driven cancel?
          - ``_handle_cap_killed_completion`` → reconcile + cleanup (early return)
          - ``_collect_or_synthesize_result`` → validated result dict, never None
          - ``_drain_remote_backend_log``  → cross-backend log unification
        """
        exp = self.running.pop(node_id)
        if exp.gpu in self.gpu_allocations:
            try:
                self.gpu_allocations[exp.gpu].remove(node_id)
            except ValueError:
                pass

        exp.log_file.close()
        elapsed_s = time.time() - exp.started_at
        archive = self.archive_dir / node_id
        wt_path = self.runner.worktree_path(node_id)
        gpu_id = exp.gpu
        spec = exp.spec

        # CAP-04 cap-driven cancel branch — never falls through to the standard path.
        if self._was_cap_killed_completion(node_id):
            self._handle_cap_killed_completion(
                node_id, wt_path, elapsed_s=elapsed_s, gpu_id=gpu_id, spec=spec
            )
            return

        result = self._collect_or_synthesize_result(node_id, archive, returncode, wt_path)

        # Include error details in result for better agent visibility before terminal write
        status = result.get("status", "completed")
        if status in ("crash", "oom", "timeout"):
            log_path = archive / "run.log"
            error_tail = ""
            if log_path.exists():
                lines = log_path.read_text().splitlines()
                error_tail = "\n".join(lines[-20:])
            result = dict(result)
            result.setdefault("error", error_tail)
            result.setdefault("log_location", str(log_path))

        # REC-02 / D-09, D-10: delegate all four artifact writes to terminal_writer.
        # Fixed write order: graph → completed/<node>.json → archive result.json → results.tsv.
        from automil.terminal_writer import write_terminal_state
        write_terminal_state(
            node_id=node_id,
            result=result,
            graph=self.graph,
            completed_dir=self.completed_dir,
            archive_dir=archive,
            results_tsv_writer=self._append_results_tsv,
            spec=spec,
            elapsed_s=elapsed_s,
            gpu_id=gpu_id,
        )

        # D-170: cross-backend log unification (no-op for local backend).
        self._drain_remote_backend_log(node_id, archive)

        # Clean running spec
        running_spec = self.running_dir / f"{node_id}.json"
        if running_spec.exists():
            running_spec.unlink()

        # Cleanup worktree
        if wt_path.exists():
            self.runner.cleanup_worktree(wt_path)

        # Clear timeout flag
        self._timed_out.pop(node_id, None)

        status_str = result.get("status", "unknown")
        composite = result.get("composite", 0)
        logger.info(
            f"Completed {node_id}: status={status_str}, "
            f"composite={composite:.4f}, elapsed={elapsed_s / 60:.1f}min, GPU {gpu_id}"
        )

    # --- _handle_completion helpers (each independently testable) ---

    def _was_cap_killed_completion(self, node_id: str) -> bool:
        """True iff the running or archive spec has metadata.cancel_reason == 'cap'.

        Reads running/<node>.json first (annotation written by _tick_cells
        BEFORE backend.cancel() is called — Pitfall 4 ordering guarantee).
        Falls back to archive/<node>/spec.json if running/ was already cleaned.
        """
        for _spec_path in (
            self.running_dir / f"{node_id}.json",
            self.archive_dir / node_id / "spec.json",
        ):
            if _spec_path.exists():
                try:
                    _raw = json.loads(_spec_path.read_text())
                    if _raw.get("metadata", {}).get("cancel_reason") == "cap":
                        return True
                except (json.JSONDecodeError, OSError):
                    pass
        return False

    def _handle_cap_killed_completion(
        self,
        node_id: str,
        wt_path: Path,
        *,
        elapsed_s: float = 0.0,
        gpu_id: int | str = -1,
        spec: dict | None = None,
    ) -> None:
        """Cap-driven cancel reconcile + cleanup (CAP-04 / D-123, D-124).

        Aggregates per-fold partial results, then delegates all four terminal
        artifact writes to write_terminal_state (REC-02 / D-09, D-10).
        Cleans the running spec + worktree. Never throws; soft-fails to logged
        warnings on graph access errors.
        """
        if spec is None:
            spec = {}
        from automil.cells.reconcile import reconcile_budget_kill

        expected_folds = self._read_fold_count_for_node(node_id)
        payload = reconcile_budget_kill(
            node_id=node_id,
            archive_dir=self.archive_dir,
            graph=self.graph,
            expected_fold_count=expected_folds,
        )
        # REC-02 / D-09, D-10: delegate all four artifact writes to terminal_writer.
        # graph node promotion (running→executed or crash) + archive result.json +
        # completed/<node>.json + results.tsv are all written by write_terminal_state.
        # D-01: partial results get status="partial" in the graph (quarantined).
        # self.graph is guaranteed by __init__ (CR-01 fix) — no hasattr guard needed.
        from automil.terminal_writer import write_terminal_state
        write_terminal_state(
            node_id=node_id,
            result=payload,
            graph=self.graph,
            completed_dir=self.completed_dir,
            archive_dir=self.archive_dir / node_id,
            results_tsv_writer=self._append_results_tsv,
            spec=spec,
            elapsed_s=elapsed_s,
            gpu_id=gpu_id,
        )
        logger.info(
            "Cap-driven cancel reconciled for %s: status=%s composite=%.4f "
            "partial_folds=%d/%d",
            node_id, payload["status"], payload["composite"],
            payload.get("partial_folds", 0), payload.get("expected_folds", 0),
        )
        # Clean running spec and worktree
        running_spec = self.running_dir / f"{node_id}.json"
        if running_spec.exists():
            running_spec.unlink()
        if wt_path.exists():
            self.runner.cleanup_worktree(wt_path)
        self._timed_out.pop(node_id, None)

    def _collect_or_synthesize_result(
        self, node_id: str, archive: Path, returncode: int, wt_path: Path,
    ) -> dict:
        """Return a valid result dict for the experiment, never None.

        Three branches:
          1. result.json exists in the worktree → schema-validate via D-201;
             on schema failure synthesize a crash payload citing the schema
             pointer so the consumer can self-correct.
          2. result.json absent → synthesize from log heuristics
             (CUDA OOM / timeout / nonzero returncode / clean exit), persist
             the synthesised payload to archive/result.json.
          3. Either branch — backfill ``status`` if the result was schema-
             valid but missing the top-level status hint.
        """
        result = self.runner.collect_result(wt_path, archive)

        # D-201 / DEC-03: validate result.json against
        # automil/schemas/result.schema.json. Malformed payloads transition
        # the node to crashed with a schema-pointer error so the consumer
        # can self-correct. The fall-through path (result is None) skips
        # validation; the synthesised minimal payload below is constructed
        # by the orchestrator and is contract-compliant by construction.
        if result is not None:
            try:
                from automil.schemas import validate_result, ValidationError
                validate_result(result)
            except ValidationError as exc:
                logger.warning(
                    "result.json schema validation failed for %s: %s; "
                    "see automil/schemas/result.schema.json",
                    node_id, exc.message,
                )
                result = {
                    "status": "crash",
                    "composite": 0.0,
                    "metrics": {},
                    "error": (
                        f"result.json failed schema validation: {exc.message} "
                        f"(json_path={exc.json_path}); "
                        f"see automil/schemas/result.schema.json"
                    ),
                }

        if result is None:
            # D-03 (REC-01): try fold aggregation from archive BEFORE synthesising
            # from log heuristics. If the process was SIGKILLed before the flush
            # handler could run, fold files may still exist in the archive even
            # without a result.json.
            fold_files = list(archive.glob("fold_*_result.json"))
            if fold_files:
                from automil.cells.reconcile import aggregate_folds
                expected = self._read_fold_count_for_node(node_id)
                result = aggregate_folds(archive, expected)
                reason = "timeout" if self._timed_out.get(node_id) else "sigkill"
                result["termination_reason"] = reason   # D-05 (REC-03)
                # Write atomically to archive so _collect_result finds it on retry.
                import tempfile as _tempfile
                _tmp_fd, _tmp_path = _tempfile.mkstemp(
                    dir=str(archive), suffix=".tmp"
                )
                try:
                    with os.fdopen(_tmp_fd, "w") as _fh:
                        _fh.write(json.dumps(result, indent=2) + "\n")
                    os.replace(_tmp_path, str(archive / "result.json"))
                except Exception:
                    try:
                        os.unlink(_tmp_path)
                    except OSError:
                        pass
            else:
                # Log-heuristic synthesis — no fold files exist.
                log_text = (archive / "run.log").read_text() if (archive / "run.log").exists() else ""
                # D-05/D-06 (REC-03): canonicalize status — "oom" and "timeout" are
                # not in the tight enum; move them to termination_reason with
                # status="crash". The tight enum is:
                # [completed, crash, budget_killed, cancelled, partial].
                termination_reason: str | None = None
                if "CUDA out of memory" in log_text or "OutOfMemoryError" in log_text:
                    status = "crash"
                    termination_reason = "oom"
                elif self._timed_out.get(node_id):
                    status = "crash"
                    termination_reason = "timeout"
                elif returncode != 0:
                    status = "crash"
                else:
                    status = "completed"

                error_tail = log_text[-2000:] if status != "completed" else ""
                result = {"status": status}
                if termination_reason:
                    result["termination_reason"] = termination_reason
                if error_tail:
                    result["error"] = error_tail
                (archive / "result.json").write_text(json.dumps(result, indent=2))

        if "status" not in result:
            result["status"] = "completed" if returncode == 0 else "crash"

        return result

    def _drain_remote_backend_log(self, node_id: str, archive: Path) -> None:
        """D-170: cross-backend log unification.

        For non-local backends, the orchestrator drains backend.log_iter()
        into archive/<id>/run.log. The local backend already writes this file
        inline (via log_fh in _launch); we only add the cross-backend drain
        for SLURM/Ray cases where archive run.log doesn't yet exist.
        For SLURM nodes, also symlinks submitit's native log files (D-171).
        Soft-fails to a logged warning on any backend error.
        """
        archive_run_log = archive / "run.log"
        backend_name_for_node = self._read_backend_name_for_node(node_id)
        if not (backend_name_for_node and backend_name_for_node != "local" and not archive_run_log.exists()):
            return
        try:
            from automil.backends import BACKENDS  # noqa: PLC0415
            from automil.backends.base import JobHandle  # noqa: PLC0415
            BackendCls = BACKENDS.get(backend_name_for_node)
            if BackendCls is None:
                return
            spec_data = self._read_running_spec(node_id, backend_name_for_node)
            drain_handle = JobHandle(
                node_id=node_id,
                backend=backend_name_for_node,
                opaque_id=spec_data.get("opaque_id", ""),
                submitted_at=spec_data.get("submitted_at", 0.0),
            )
            # Reuse the configured backend instance if it matches; else instantiate.
            _backend = (
                self.backend
                if (
                    self.backend is not None
                    and getattr(self.backend, "_backend_name", None) == backend_name_for_node
                )
                else BackendCls(self.automil_dir, self.config)
            )
            drain_lines = _drain_log_iter_with_timeout(_backend, drain_handle, timeout=60.0)
            _atomic_write_lines(archive_run_log, drain_lines)
            # D-171: for SLURM nodes, symlink submitit's native log files into archive/.
            if backend_name_for_node == "slurm":
                _symlink_slurm_logs(self.automil_dir, archive, spec_data)
        except Exception as exc:
            logger.warning(
                "D-170 cross-backend log unification failed for %s: %s", node_id, exc
            )

    def _handle_timeout(self, exp_id: str) -> None:
        """D-04 (REC-01): SIGTERM main PID first (flush handler runs), then SIGKILL
        process group after a configurable grace window. LOCAL BACKEND ONLY —
        SLURM and Ray backends handle their own timeout signals (SLURM via
        --signal=B:TERM@30; Ray via ray.cancel) and never call this method.

        Rationale: sending SIGTERM to the whole process group immediately hits
        DataLoader workers before the main process's handler runs. The main-PID-
        first approach lets the Python signal handler (register_sigterm_flush)
        complete and write fold-aggregated result.json before SIGKILL reclaims VRAM.

        T-09-05: os.kill raises ProcessLookupError if the process has already
        exited; ProcessLookupError is caught on both kill calls. The Popen
        reference keeps the PID alive until poll() confirms exit.
        """
        exp = self.running[exp_id]
        pid = exp.process.pid
        grace = int((self.config.get("orchestrator") or {}).get("timeout_grace_seconds", 10))
        logger.warning(
            "Timeout for %s: SIGTERMing main PID %d (grace=%ds before SIGKILL group)",
            exp_id, pid, grace,
        )
        try:
            os.kill(pid, signal.SIGTERM)    # main PID only — lets flush handler run
        except ProcessLookupError:
            pass
        time.sleep(grace)
        if exp.process.poll() is None:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as exc:
                # CR-02 fix: os.getpgid() raises OSError(EPERM) when the caller
                # lacks permission to query the PID's process group (different
                # session, container boundary, or kernel hardening). Without this
                # catch the exception propagated before the two lines below,
                # leaving the experiment stuck in self.running forever and causing
                # an infinite blocking loop on every subsequent tick. Mirror the
                # pattern already used by _escalate_to_sigkill and _kill_experiment.
                logger.warning(
                    "_handle_timeout: killpg(%d, SIGKILL) failed: %s — "
                    "process group may still be alive; continuing cleanup.",
                    pid, exc,
                )
        # These two lines MUST run regardless of whether killpg succeeded or
        # raised OSError: marking _timed_out and calling _handle_completion
        # removes the experiment from self.running, preventing the re-fire loop.
        self._timed_out[exp_id] = True
        self._handle_completion(exp_id, returncode=-9)

    def _kill_experiment(self, node_id: str, sig: int = signal.SIGTERM) -> bool:
        """Send *sig* to the process group of *node_id* and return True if found.

        Called by LocalBackend.cancel (BCK-02 / D-57).  Returns immediately
        after the signal goes out — state transition to ``cancelled`` is
        observed via subsequent poll() calls against the on-disk state.
        Uses the same starttime cross-check as ``_handle_timeout``
        (CLN-04 / D-17) to guard against PID reuse.

        For SIGTERM-style cancels, also records a SIGKILL escalation
        deadline in ``self._pending_sigkill_at``. The next
        ``_check_running`` tick after the grace period (5s) will SIGKILL
        the process group if it hasn't exited. SIGKILL itself is not
        escalated (already maximum force).

        Returns:
            True  — signal was delivered to the process group.
            False — node not found in self.running (already finished, or this
                    LocalBackend instance was freshly constructed and the daemon
                    is the process that holds the live Popen handle).
        """
        exp = self.running.get(node_id)
        if exp is None:
            logger.warning(
                "_kill_experiment: %s not in self.running (daemon may hold the "
                "live handle — cancel via sentinel file is not implemented in "
                "Phase 2; the daemon's _handle_timeout will handle timeouts).",
                node_id,
            )
            return False
        pid = exp.process.pid
        logger.info(
            "_kill_experiment: sending signal %d to PID %d (node %s)",
            sig, pid, node_id,
        )
        try:
            os.killpg(os.getpgid(pid), sig)
        except ProcessLookupError:
            logger.info("_kill_experiment: PID %d already gone", pid)
        except OSError as e:
            logger.warning("_kill_experiment: os.killpg failed: %s", e)
        if sig == signal.SIGTERM:
            self._pending_sigkill_at[node_id] = time.time() + 5.0
        return True

    def _mark_crashed(self, node_id: str, spec: dict, error: str):
        """Mark an experiment as crashed without a running process."""
        archive = self.archive_dir / node_id
        archive.mkdir(parents=True, exist_ok=True)

        result = {
            "id": node_id,
            "description": spec.get("description", ""),
            "status": "crash",
            "error": error,
            "completed_at": datetime.now().isoformat(),
        }
        if "graph_metadata" in spec:
            result["graph_metadata"] = spec["graph_metadata"]

        (archive / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        spec_clean = {k: v for k, v in spec.items() if k not in ("_file",)}
        (archive / "spec.json").write_text(json.dumps(spec_clean, indent=2) + "\n")
        (self.completed_dir / f"{node_id}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )

    def _append_results_tsv(self, node_id: str, result: dict, description: str = ""):
        """Append a row to results.tsv (sole writer, no locking needed).

        Metric columns are derived from the keys of ``result["metrics"]``
        on first write — no hardcoded MIL-vocabulary. The header is
        ``node_id, <metric_keys sorted>, composite, vram_gb, elapsed_min,
        status, description``. Subsequent rows are aligned to the
        existing header by parsing it from disk; missing keys are filled
        with empty strings so the file remains a valid TSV across
        schema drift.
        """
        metrics = result.get("metrics", {})
        composite = result.get("composite", 0.0)
        status = result.get("status", "completed")
        elapsed_s = result.get("elapsed_seconds", 0)
        vram_mb = result.get("peak_vram_mb", 0)

        if not self.results_tsv.exists() or self.results_tsv.stat().st_size == 0:
            metric_cols = sorted(metrics.keys())
            header_cols = ["node_id"] + metric_cols + [
                "composite", "vram_gb", "elapsed_min", "status", "description",
            ]
            self.results_tsv.write_text("\t".join(header_cols) + "\n")
        else:
            first_line = self.results_tsv.read_text().split("\n", 1)[0]
            header_cols = first_line.split("\t")
            metric_cols = [
                c for c in header_cols
                if c not in {"node_id", "composite", "vram_gb",
                             "elapsed_min", "status", "description"}
            ]

        def _fmt(v) -> str:
            if v is None or v == "":
                return ""
            try:
                return f"{float(v):.4f}"
            except (TypeError, ValueError):
                return str(v)

        cells: list[str] = [node_id]
        cells.extend(_fmt(metrics.get(c, "")) for c in metric_cols)
        cells.extend([
            f"{composite:.6f}",
            f"{vram_mb / 1024:.1f}",
            f"{elapsed_s / 60:.1f}",
            status,
            description or node_id,
        ])
        with open(self.results_tsv, "a") as f:
            f.write("\t".join(cells) + "\n")

    # --- Main loop ---

    def _reload_orchestrator_config(self) -> None:
        """Hot-reload the orchestrator section of config.yaml each tick.

        Lets an operator raise/lower concurrency and VRAM estimates live
        without restarting the daemon (which would orphan running jobs).
        Only the orchestrator.* section is reloaded; other sections are
        not used after construction.
        """
        config_path = self.automil_dir / "config.yaml"
        if not config_path.exists():
            return
        try:
            import yaml
            cfg = yaml.safe_load(config_path.read_text()) or {}
        except Exception as e:
            logger.warning(
                f"Config reload skipped: {config_path.name} parse failed ({e}); "
                f"keeping previous values (max_per_gpu={self.max_per_gpu}, "
                f"default_vram={self.default_vram}, safety_margin={self.safety_margin_gb})"
            )
            return
        orch_cfg = (cfg.get("orchestrator") or {}) if isinstance(cfg, dict) else {}
        new_max = orch_cfg.get("max_concurrent_per_gpu", self.max_per_gpu)
        new_vram = orch_cfg.get("default_vram_estimate_gb", self.default_vram)
        new_safety = orch_cfg.get("safety_margin_gb", self.safety_margin_gb)
        if new_max != self.max_per_gpu:
            logger.info(
                f"Config reload: max_concurrent_per_gpu {self.max_per_gpu} -> {new_max}"
            )
            self.max_per_gpu = new_max
        if new_vram != self.default_vram:
            logger.info(
                f"Config reload: default_vram_estimate_gb {self.default_vram} -> {new_vram}"
            )
            self.default_vram = new_vram
        if new_safety != self.safety_margin_gb:
            logger.info(
                f"Config reload: safety_margin_gb {self.safety_margin_gb} -> {new_safety}"
            )
            self.safety_margin_gb = new_safety

    def tick(self):
        """Single scheduling cycle."""
        # 0. Hot-reload config so concurrency bumps take effect live
        self._reload_orchestrator_config()

        # 1. Check running experiments
        self._check_running()

        # Phase 4 step 1.5: cap state machine (D-114).
        self._tick_cells()

        # 2. Schedule pending experiments (skip if draining)
        if not self.draining:
            pending = self._get_pending()
            for spec in pending:
                if not spec.get("id"):
                    self.counter += 1
                    spec["id"] = f"{self.counter:04d}"

                needed_gb = spec.get("estimated_vram_gb", self.default_vram)
                gpu = self._find_best_gpu(needed_gb)

                if gpu is not None and self._pre_launch_check(gpu, needed_gb):
                    self._launch(spec, gpu)

        # 3. Save state
        self._save_state()

    def run(self):
        """Main daemon loop."""
        # D-168 (BREAKING in 6.0.0): refuse to start if flat running/*.json files
        # exist AND no namespaced subdirectory exists. autoMIL 6.x does NOT
        # auto-migrate; operators must drain via `automil orchestrator stop` and
        # confirm running/ is empty before upgrading. See CHANGELOG.md 6.0.0.
        if self.running_root.exists():
            flat_jsons = list(self.running_root.glob("*.json"))  # top-level only
            namespaced = [
                name for name in ("local", "slurm", "ray")
                if (self.running_root / name).is_dir()
            ]
            if flat_jsons and not namespaced:
                raise SystemExit(
                    "BREAKING CHANGE: flat orchestrator/running/*.json files detected. "
                    "autoMIL 6.x uses per-backend namespacing "
                    "(running/<backend>/<id>.json). "
                    f"Found {len(flat_jsons)} flat file(s) in {self.running_root}. "
                    "Drain in-flight runs with `automil orchestrator stop`, confirm "
                    "orchestrator/running/ contains no top-level *.json files, then "
                    "restart the daemon. See CHANGELOG.md 6.0.0 for full recovery steps."
                )

        self.runner.prune_stale_worktrees()
        self._recover_orphans()

        logger.info(
            f"Orchestrator started. GPUs: {list(self.gpu_allocations.keys())}, "
            f"poll={self.poll_interval}s, safety={self.safety_margin_gb}GB"
        )

        # Signal handlers
        def handle_signal(signum, frame):
            sig_name = signal.Signals(signum).name
            logger.info(f"Received {sig_name}, shutting down gracefully...")
            self._shutdown = True

        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)

        # Write PID file
        _write_pid_file(self.pid_file)

        try:
            while not self._shutdown:
                try:
                    self.tick()
                except Exception as e:
                    logger.error(f"Tick error: {e}", exc_info=True)
                time.sleep(self.poll_interval)

            # Graceful shutdown: wait for running experiments
            if self.running:
                logger.info(f"Waiting for {len(self.running)} running experiments...")
                self.draining = True
                while self.running:
                    self._check_running()
                    time.sleep(5)
                logger.info("All experiments completed, exiting.")
        finally:
            if self.pid_file.exists():
                self.pid_file.unlink()
            self._save_state()
            logger.info("Orchestrator stopped.")

    # --- CLI commands (instance methods) ---

    def cmd_start(self):
        """Start the orchestrator daemon."""
        if self.pid_file.exists():
            loaded = _load_pid_file(self.pid_file)
            if loaded and _is_pid_alive_with_starttime(loaded["pid"], loaded["starttime_ticks"]):
                print(f"Orchestrator already running (PID {loaded['pid']})")
                return
            # Legacy plain-int OR stale (PID reused / daemon dead). Unlink and proceed.
            logger.info("Removing stale PID file at %s", self.pid_file)
            self.pid_file.unlink()

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler(),
            ],
        )

        self.run()

    def cmd_status(self):
        """Print orchestrator status."""
        loaded = _load_pid_file(self.pid_file) if self.pid_file.exists() else None
        if loaded and _is_pid_alive_with_starttime(loaded["pid"], loaded["starttime_ticks"]):
            print(f"Status: running (PID {loaded['pid']})")
        elif self.pid_file.exists():
            print("Status: stale or no PID file")
        else:
            print("Orchestrator: NOT RUNNING")

        # GPU state
        if self.gpu_state_file.exists():
            state = json.loads(self.gpu_state_file.read_text())
            print(f"\nLast updated: {state.get('last_updated', 'unknown')}")
            print(f"Queue depth: {state.get('queue_depth', 0)}")
            print(f"Running: {state.get('total_running', 0)}")
            print(f"Completed: {state.get('total_completed', 0)}")
            print(f"Counter: {state.get('counter', 0)}")
            print("\nGPUs:")
            for idx, gpu in sorted(state.get("gpus", {}).items()):
                running_ids = gpu.get("running", [])
                sched = gpu.get("schedulable_free_gb", 0)
                util = gpu.get("utilization_pct", 0)
                print(f"  GPU {idx}: {sched:.1f}GB schedulable, {util}% util, running={running_ids}")
        else:
            gpus = query_gpus()
            print("\nGPUs (live):")
            for g in gpus:
                print(f"  GPU {g.index}: {g.free_gb:.1f}GB free, {g.utilization}% util")

        # Queue
        pending = list(self.queue_dir.glob("*.json"))
        if pending:
            print(f"\nPending ({len(pending)}):")
            for f in sorted(pending):
                try:
                    spec = json.loads(f.read_text())
                    print(f"  {f.name}: {spec.get('description', '?')[:60]} (P{spec.get('priority', '?')})")
                except Exception:
                    print(f"  {f.name}: (unreadable)")

    def cmd_stop(self):
        """Stop the orchestrator gracefully."""
        if not self.pid_file.exists():
            print("Orchestrator not running")
            return
        loaded = _load_pid_file(self.pid_file)
        if not loaded:
            print("Orchestrator PID file is stale or malformed; removing.")
            self.pid_file.unlink()
            return
        if not _is_pid_alive_with_starttime(loaded["pid"], loaded["starttime_ticks"]):
            print(f"Recorded PID {loaded['pid']} is not our daemon (PID reused or dead). Removing stale file.")
            self.pid_file.unlink()
            return
        try:
            os.kill(loaded["pid"], signal.SIGTERM)
            print(f"Sent SIGTERM to PID {loaded['pid']}")
        except OSError as e:
            print(f"Failed to stop: {e}")
            self.pid_file.unlink()

    def cmd_submit(self, spec_path: str):
        """Submit an experiment spec to the queue."""
        src = Path(spec_path)
        if not src.exists():
            print(f"File not found: {spec_path}")
            sys.exit(1)

        spec = json.loads(src.read_text())
        if not spec.get("id"):
            # Auto-assign ID from counter
            counter = 0
            if self.gpu_state_file.exists():
                try:
                    counter = json.loads(self.gpu_state_file.read_text()).get("counter", 0)
                except Exception:
                    pass
            counter += 1
            spec["id"] = f"{counter:04d}"

        if not spec.get("submitted_at"):
            spec["submitted_at"] = datetime.now().isoformat()

        dst = self.queue_dir / f"{spec['id']}.json"
        dst.write_text(json.dumps(spec, indent=2) + "\n")
        print(f"Submitted experiment {spec['id']}: {spec.get('description', '?')}")


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    # For status/stop/submit, we just need path resolution, not GPU init.
    # Construct orchestrator instance.
    orch = ExperimentOrchestrator()

    if cmd == "start":
        orch.cmd_start()
    elif cmd == "status":
        orch.cmd_status()
    elif cmd == "stop":
        orch.cmd_stop()
    elif cmd == "submit":
        if len(sys.argv) < 3:
            print("Usage: automil submit <spec.json>")
            sys.exit(1)
        orch.cmd_submit(sys.argv[2])
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
