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

import fcntl
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from dotenv import dotenv_values

# NOTE (IN-01): _collect_editable_source_roots lives in automil.cli.check but has
# no CLI dependency (only uses site + pathlib). The import works today because no
# CLI submodule imports the daemon (no circular import). A future refactor could
# move it to automil.utils or automil.editable to resolve the layering concern.
from automil.cli.check import _collect_editable_source_roots
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
SCHEDULING_POLICY = "best_fit"

# ---------------------------------------------------------------------------
# Subprocess env whitelist (CLN-02 / D-04)
# ---------------------------------------------------------------------------
# Hardcoded system-minimal whitelist applied to os.environ when building the
# experiment subprocess env. Operator secrets (OPENAI_API_KEY, WANDB_API_KEY,
# GITHUB_TOKEN, AWS_SECRET_ACCESS_KEY, ...) are NOT inherited -- closing the
# HIGH-severity exfiltration vector documented in
# CONCERNS.md §"Subprocess `env` inherits the full operator environment".
#
# Consumer-specific vars (e.g. AUTOBENCH_*_ROOT) are opted in per project via
# `automil/config.yaml: env.passthrough` -- see _build_subprocess_env.
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
# A5 (claims-alignment): the val-firewall keys are as orchestrator-owned as the
# GPU masks — a spec.env could otherwise retarget the born-sealing directory
# (AUTOMIL_RESULTS_DIR), repoint policy resolution (AUTOMIL_DIR_REL), corrupt
# partial/complete discrimination (AUTOMIL_FOLD_COUNT), or flip the consumer
# test-print gates (AUTOMIL_CERTIFY) so test metrics stream into the
# agent-visible run.log during search. Enumerated, not prefix-wide:
# AUTOMIL_VARIANT_* is legitimate spec env (cli/lifecycle/apply.py).
_SPEC_ENV_BLOCKED: frozenset[str] = frozenset({
    "AUTOMIL_ACCELERATOR",
    "AUTOMIL_CERTIFY",
    "AUTOMIL_DIR_REL",
    "AUTOMIL_FOLD_COUNT",
    "AUTOMIL_GPU",
    "AUTOMIL_NODE_ID",
    "AUTOMIL_RESULTS_DIR",
    "CUDA_DEVICE_ORDER",
    "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
})

logger = logging.getLogger(__name__)


class _CellAdmission(str, Enum):
    """Launch decision kept separate from the monotone cap state."""

    ALLOW = "allow"
    HOLD_TELEMETRY = "hold-telemetry"
    REFUSE_CAP = "refuse-cap"

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
_resolved_rocm_smi = shutil.which("rocm-smi")
ROCM_SMI_PATH = _resolved_rocm_smi or "rocm-smi"
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
def visible_gpu_ids() -> frozenset[int] | None:
    """Parse the operator's host-local GPU partition, or None for all GPUs.

    ``AUTOMIL_VISIBLE_GPUS`` restricts this daemon to a comma-separated set
    of physical GPU indexes so several projects can schedule concurrently on
    one host without double-booking VRAM. It is runtime host configuration,
    deliberately outside any frozen project config. A malformed value raises
    rather than silently scheduling on every GPU.
    """
    raw = os.environ.get("AUTOMIL_VISIBLE_GPUS", "").strip()
    if not raw:
        return None
    ids: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token.isdecimal():
            raise ValueError(
                f"AUTOMIL_VISIBLE_GPUS must be comma-separated GPU indexes; "
                f"got {raw!r}"
            )
        ids.add(int(token))
    return frozenset(ids)


def _apply_partition(
    gpus: list[GPUInfo], visible: frozenset[int] | None,
) -> list[GPUInfo]:
    if visible is None:
        return gpus
    return [gpu for gpu in gpus if gpu.index in visible]


def query_gpus(*, apply_partition: bool = True) -> list[GPUInfo]:
    """Query nvidia-smi for GPU state.

    Uses the path resolved at module import (NVIDIA_SMI_PATH) to defend
    against PATH-shim spoofing on shared hosts (CLN-05). Results are
    restricted to the operator's ``AUTOMIL_VISIBLE_GPUS`` partition when
    one is declared; a malformed partition raises here, outside the smi
    error handling, so it can never be misreported as an smi failure.
    """
    visible = visible_gpu_ids() if apply_partition else None
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
        return _apply_partition(gpus, visible)
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        logger.warning(f"nvidia-smi failed: {e}")
        return []


def query_rocm_gpus() -> list[GPUInfo]:
    """Query live ROCm device count and VRAM via pinned ``rocm-smi``.

    ``--showmeminfo vram --json`` reports total and currently used bytes for
    every device. Any missing, malformed, duplicate, or internally inconsistent
    record rejects the whole snapshot: partial telemetry must never become
    permission to launch on an unverified device.
    """
    visible = visible_gpu_ids()
    try:
        result = subprocess.run(
            [ROCM_SMI_PATH, "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0 or not result.stdout.strip():
            logger.warning(
                "rocm-smi failed with return code %s: %s",
                result.returncode,
                result.stderr.strip(),
            )
            return []
        payload = json.loads(result.stdout)
        if not isinstance(payload, dict):
            raise ValueError("top-level payload is not a mapping")

        devices: list[GPUInfo] = []
        seen_indices: set[int] = set()
        for raw_name, raw_metrics in payload.items():
            if not isinstance(raw_metrics, dict):
                continue
            total_raw = raw_metrics.get("VRAM Total Memory (B)")
            used_raw = raw_metrics.get("VRAM Total Used Memory (B)")
            if total_raw is None and used_raw is None:
                continue
            if total_raw is None or used_raw is None:
                raise ValueError(f"{raw_name!r} has partial VRAM telemetry")

            match = re.search(r"(\d+)$", str(raw_name))
            if match is None:
                raise ValueError(f"cannot derive device index from {raw_name!r}")
            index = int(match.group(1))
            if index in seen_indices:
                raise ValueError(f"duplicate device index {index}")

            total_bytes = int(str(total_raw).strip())
            used_bytes = int(str(used_raw).strip())
            if total_bytes <= 0 or used_bytes < 0 or used_bytes > total_bytes:
                raise ValueError(
                    f"invalid VRAM values for device {index}: "
                    f"total={total_bytes}, used={used_bytes}"
                )
            devices.append(GPUInfo(
                index=index,
                total_mb=total_bytes // (1024 * 1024),
                free_mb=(total_bytes - used_bytes) // (1024 * 1024),
                utilization=0,
            ))
            seen_indices.add(index)

        if not devices:
            raise ValueError("no complete device records")
        devices.sort(key=lambda device: device.index)
        return _apply_partition(devices, visible)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError,
            json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("rocm-smi failed: %s", exc)
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
    """D-171 / B3: copy submitit's native stdout/stderr logs into archive/<id>/,
    with held-out redaction applied to the copies.

    Previously these were symlinks to submitit's raw files — an unredacted
    agent-visible view of the same stdout H-1 redacts in run.log (redacting a
    symlink target would corrupt submitit's own file). Copies cost disk but
    keep the firewall closed; redaction uses the generic held-out markers
    (the result's own held_out keys are not known at drain time).

    This is a module-level function (NOT a method) so it can be tested without
    instantiating ExperimentOrchestrator.
    """
    from automil.firewall import redact_log_file

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
            shutil.copyfile(stdout_src, stdout_dst)
            redact_log_file(stdout_dst)
        except OSError as exc:
            logger.warning("D-171 stdout copy failed: %s", exc)
    if stderr_src.exists() and not stderr_dst.exists():
        try:
            shutil.copyfile(stderr_src, stderr_dst)
            redact_log_file(stderr_dst)
        except OSError as exc:
            logger.warning("D-171 stderr copy failed: %s", exc)


def _submitted_at_key(spec: dict) -> str:
    """Stable, always-comparable sort key for a queue spec's ``submitted_at`` (L-7).

    ``submitted_at`` is normally an ISO-8601 string written by ``automil
    submit``, but ``_get_pending`` also has to sort specs where it is absent
    (older or hand-written specs), explicitly ``null`` (a present key reads
    back as ``None``, which ``dict.get(..., default)`` does NOT paper over —
    the default only applies when the key is missing), or — from a malformed
    producer — some other JSON type. Python 3 raises ``TypeError`` comparing
    ``None``/``str`` or ``int``/``str``, and that exception previously
    propagated out of ``list.sort()`` inside ``_get_pending``, which is
    called every ``tick()`` — one bad queue file stalled scheduling for
    every pending spec, not just its own.

    Coercing to ``str`` makes every comparison well-defined. Specs with no
    usable timestamp (falsy: absent, ``None``, ``""``) collapse to ``""``,
    which sorts before every real ISO-8601 timestamp — so they are treated
    as "submitted before anything timestamped" and fall back to queue
    (filename) order among themselves via Python's stable sort.
    """
    raw = spec.get("submitted_at")
    return str(raw) if raw else ""


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
        # Diagnostic state only.  It is never consulted for billing or cap
        # transitions; it merely suppresses repeated outage messages.
        self._activity_health_by_cell: dict[str, str] = {}
        # Log-dedup for declared-but-unreadable cell holds, separate from
        # activity health so a file blip never fakes a telemetry transition.
        self._unreadable_cell_logged: set[str] = set()

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
        self.scheduling_policy: str = orch_cfg.get("scheduling_policy", SCHEDULING_POLICY)
        self._rr_cursor: int = 0
        # SCH-02 / D-03: opt-in editable-install overlay guard (default OFF).
        # When True, _launch prepends the worktree's editable src root to PYTHONPATH.
        # Default is False to preserve D-199/DEC-01 invariant (no auto-injection).
        self.editable_overlay_guard: bool = bool(
            orch_cfg.get("editable_overlay_guard", False)
        )

        # Generic CPU consumers (for example examples/sklearn-iris) declare the
        # execution substrate explicitly.  A CPU-only project still needs one
        # local scheduling slot, but it must not be represented as a physical
        # GPU or subjected to VRAM admission checks.
        hardware_cfg = self.config.get("hardware", {}) if self.config else {}
        configured_accelerator = str(
            hardware_cfg.get("accelerator", "")
        ).strip().lower()
        try:
            configured_gpu_count = int(hardware_cfg.get("gpu_count", 0))
        except (TypeError, ValueError):
            configured_gpu_count = -1
        try:
            configured_min_vram_gb = float(hardware_cfg.get("min_vram_gb", 0.0))
        except (TypeError, ValueError):
            configured_min_vram_gb = 0.0
        self._accelerator = configured_accelerator
        self._configured_gpu_count = configured_gpu_count
        self._configured_min_vram_gb = configured_min_vram_gb
        self._cpu_only = (
            configured_accelerator == "cpu" and configured_gpu_count == 0
        )

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

        # Detect typed execution slots. ``gpu_allocations`` is the legacy
        # internal accounting map; its integer keys are scheduler slot IDs, not
        # hardware provenance. CPU slot 0 therefore never becomes ``gpu: 0`` in
        # an artifact. ROCm has no nvidia-smi-compatible dynamic query, so its
        # slots come from the hardware report stamped by ``automil init`` and
        # are admitted against that report's conservative minimum VRAM.
        if self._cpu_only:
            self.gpu_allocations[0] = []
            logger.info("CPU-only execution configured; using local slot 0")
        elif configured_accelerator == "cpu":
            logger.warning(
                "CPU execution requires hardware.gpu_count: 0; "
                "queued jobs will remain pending"
            )
        elif configured_accelerator == "rocm":
            if configured_gpu_count > 0 and configured_min_vram_gb > 0:
                live_rocm = query_rocm_gpus()
                live_indices = [device.index for device in live_rocm]
                # A declared host partition restricts which of the configured
                # devices this daemon owns; the live (already filtered) view
                # must then equal exactly the partition members that exist
                # under the configured count, not the full range.
                _partition = visible_gpu_ids()
                if _partition is None:
                    expected_indices = list(range(configured_gpu_count))
                else:
                    expected_indices = sorted(
                        index for index in _partition
                        if index < configured_gpu_count
                    )
                live_min_vram_gb = min(
                    (device.total_mb / 1024 for device in live_rocm),
                    default=0.0,
                )
                if (
                    expected_indices
                    and live_indices == expected_indices
                    and live_min_vram_gb >= configured_min_vram_gb
                ):
                    for device_index in live_indices:
                        self.gpu_allocations[device_index] = []
                    logger.info(
                        "ROCm execution configured; verified %d live device slot(s)",
                        configured_gpu_count,
                    )
                else:
                    logger.warning(
                        "Live ROCm hardware does not match config "
                        "(expected indices=%s, min_vram_gb>=%.1f; "
                        "detected indices=%s, min_vram_gb=%.1f); "
                        "queued jobs will remain pending",
                        expected_indices,
                        configured_min_vram_gb,
                        live_indices,
                        live_min_vram_gb,
                    )
            else:
                logger.warning(
                    "ROCm configuration requires positive hardware.gpu_count and "
                    "hardware.min_vram_gb; queued jobs will remain pending"
                )
        elif configured_accelerator in {"", "cuda"}:
            for gpu in query_gpus():
                self.gpu_allocations[gpu.index] = []
            if self.gpu_allocations and not self._accelerator:
                self._accelerator = "cuda"
            if not self.gpu_allocations:
                _partition = visible_gpu_ids()
                _detected = (
                    query_gpus(apply_partition=False)
                    if _partition is not None else []
                )
                if _detected:
                    logger.warning(
                        "AUTOMIL_VISIBLE_GPUS=%s selects none of the %d "
                        "detected GPU(s) (indexes %s); queued jobs will "
                        "remain pending",
                        ",".join(str(i) for i in sorted(_partition)),
                        len(_detected),
                        [gpu.index for gpu in _detected],
                    )
                else:
                    logger.warning(
                        "No CUDA GPUs detected for a GPU-targeted "
                        "configuration; queued jobs will remain pending"
                    )
        else:
            logger.warning(
                "Unsupported hardware.accelerator %r; queued jobs will remain pending",
                configured_accelerator,
            )

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
        """Persist typed execution-slot state and legacy CUDA telemetry."""
        gpus = query_gpus() if self._accelerator == "cuda" else []
        rocm_gpus = query_rocm_gpus() if self._accelerator == "rocm" else []
        gpu_data = {}
        execution_slots = {}
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
            execution_slots[f"cuda:{g.index}"] = {
                "accelerator": "cuda",
                "device_index": g.index,
                "running": running_on,
                "capacity": self.max_per_gpu,
                "schedulable_free_gb": gpu_data[str(g.index)]["schedulable_free_gb"],
            }

        if self._cpu_only:
            execution_slots["cpu:0"] = {
                "accelerator": "cpu",
                "device_index": None,
                "running": self.gpu_allocations.get(0, []),
                "capacity": self.max_per_gpu,
            }
        elif self._accelerator == "rocm":
            live_by_index = {device.index: device for device in rocm_gpus}
            for device_index, running_on in sorted(self.gpu_allocations.items()):
                alloc_vram = sum(
                    self.running[eid].estimated_vram_gb
                    for eid in running_on
                    if eid in self.running
                )
                slot_state = {
                    "accelerator": "rocm",
                    "device_index": device_index,
                    "running": running_on,
                    "capacity": self.max_per_gpu,
                    "telemetry_available": device_index in live_by_index,
                }
                live_device = live_by_index.get(device_index)
                if live_device is not None:
                    slot_state["schedulable_free_gb"] = round(
                        live_device.free_gb - self.safety_margin_gb - alloc_vram,
                        1,
                    )
                execution_slots[f"rocm:{device_index}"] = slot_state

        state = {
            "counter": self.counter,
            "last_updated": datetime.now().isoformat(),
            "gpus": gpu_data,
            "execution_slots": execution_slots,
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
        # WR-02 fix: scan all per-backend subdirs (local, slurm, ray), not just
        # running/local/. SLURM and Ray running specs live in running/slurm/ and
        # running/ray/; the original code only looked at self.running_dir (= local).
        # Mirror the pattern already used in _read_backend_name_for_node (lines
        # that iterate ("local", "slurm", "ray")). Gracefully skips backends that
        # don't have a subdirectory yet (fresh installs, single-backend deployments).
        _backend_subdirs = [
            self.running_root / name
            for name in ("local", "slurm", "ray")
            if (self.running_root / name).exists()
        ]
        if not _backend_subdirs:
            return
        import itertools
        for f in itertools.chain.from_iterable(d.glob("*.json") for d in _backend_subdirs):
            try:
                spec = json.loads(f.read_text())
                node_id = spec.get("id", f.stem)
                logger.info(f"Orphaned experiment {node_id} found, marking as crashed")

                _meta = spec.get("metadata") or {}
                if _meta.get("launch_phase") == "launching":
                    # M-6: the daemon died between Popen and the running-spec
                    # write, so no pid was ever recorded. The node is recoverable;
                    # the process is not. Name the worktree so it can be found.
                    logger.warning(
                        "Orphan %s died mid-launch with no pid recorded: a training "
                        "process may still be alive on %s. Worktree: %s. "
                        "Check for it manually — recovery cannot signal an "
                        "unrecorded process group.",
                        node_id,
                        self._execution_label_from_metadata(_meta),
                        _meta.get("worktree"),
                    )
                else:
                    self._sigkill_orphan_pg(node_id, spec)

                archive = self.archive_dir / node_id
                archive.mkdir(parents=True, exist_ok=True)
                result = {"status": "crash", "error": "Orchestrator restarted while running"}
                (archive / "result.json").write_text(json.dumps(result, indent=2))
                (self.completed_dir / f"{node_id}.json").write_text(json.dumps({
                    "id": node_id,
                    "status": "crash",
                    "accelerator": _meta.get("accelerator"),
                    "gpu": _meta.get("gpu"),
                    "completed_at": datetime.now().isoformat(),
                }, indent=2))
                # M-5: same reasoning as _mark_crashed — the graph must not be
                # left describing a node that is still "running".
                self._mark_node_terminal_in_graph(
                    node_id, "crash", "Orchestrator restarted while running",
                )

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
        # Sort by priority ASC, then submitted_at ASC. submitted_at is coerced
        # via _submitted_at_key (L-7) so a mix of string / absent / explicit-None
        # / malformed-type values across the queue can never raise TypeError here.
        pending.sort(key=lambda s: (s.get("priority", 2), _submitted_at_key(s)))
        return pending

    def _find_best_gpu(self, needed_gb: float) -> int | None:
        """Find a GPU for the pending job according to self.scheduling_policy (best_fit | round_robin | least_loaded)."""
        if getattr(self, "_cpu_only", False):
            running_on = self.gpu_allocations.get(0, [])
            return 0 if len(running_on) < self.max_per_gpu else None

        accelerator = getattr(self, "_accelerator", "")
        if accelerator == "cpu" or accelerator not in {"", "cuda", "rocm"}:
            return None

        candidates: list[tuple[int, float]] = []
        if accelerator == "rocm":
            for device in query_rocm_gpus():
                running_on = self.gpu_allocations.get(device.index)
                if running_on is None:
                    continue
                if len(running_on) >= self.max_per_gpu:
                    continue
                alloc_vram = sum(
                    self.running[eid].estimated_vram_gb
                    for eid in running_on
                    if eid in self.running
                )
                schedulable = device.free_gb - self.safety_margin_gb - alloc_vram
                if schedulable >= needed_gb:
                    candidates.append((device.index, schedulable))
        else:
            for g in query_gpus():
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

        policy = self.scheduling_policy
        if policy == "least_loaded":
            # Most schedulable free VRAM = least loaded GPU
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        elif policy == "round_robin":
            # Cycle through eligible GPUs in stable index order
            candidates.sort(key=lambda x: x[0])
            chosen = candidates[self._rr_cursor % len(candidates)][0]
            self._rr_cursor += 1
            return chosen
        else:
            if policy != "best_fit":
                logger.warning(
                    "Unknown scheduling_policy %r; falling back to best_fit", policy
                )
            # best_fit (default): tightest fit — preserves current behavior
            candidates.sort(key=lambda x: x[1])
            return candidates[0][0]

    def _pre_launch_check(self, gpu_id: int, needed_gb: float) -> bool:
        """Final VRAM check right before launch."""
        if getattr(self, "_cpu_only", False):
            return gpu_id == 0
        accelerator = getattr(self, "_accelerator", "")
        if accelerator == "cpu" or accelerator not in {"", "cuda", "rocm"}:
            return False
        if accelerator == "rocm":
            running_on = self.gpu_allocations.get(gpu_id)
            if running_on is None or len(running_on) >= self.max_per_gpu:
                return False
            for device in query_rocm_gpus():
                if device.index == gpu_id:
                    return device.free_gb >= needed_gb + self.safety_margin_gb
            return False

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
        accelerator = getattr(self, "_accelerator", "cuda") or "cuda"
        # On Linux ROCm, ROCR_VISIBLE_DEVICES selects the host device first;
        # HIP/CUDA/GPU_DEVICE_ORDINAL then refer to logical device 0 inside that
        # restricted view (the same layering used by AMD AgentKernelArena).
        if accelerator == "rocm":
            env["ROCR_VISIBLE_DEVICES"] = str(gpu_id)
            env["HIP_VISIBLE_DEVICES"] = "0"
            env["CUDA_VISIBLE_DEVICES"] = "0"
            env["GPU_DEVICE_ORDINAL"] = "0"
        elif accelerator == "cuda":
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            # gpu_id is an nvidia-smi (PCI-order) index; pin CUDA's
            # enumeration to the same order so disjoint host partitions can
            # never land two jobs on one physical device.
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            env["HIP_VISIBLE_DEVICES"] = ""
            env["ROCR_VISIBLE_DEVICES"] = ""
            env["GPU_DEVICE_ORDINAL"] = ""
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["HIP_VISIBLE_DEVICES"] = ""
            env["ROCR_VISIBLE_DEVICES"] = ""
            env["GPU_DEVICE_ORDINAL"] = ""
        # Backward-compatible logical slot. Consumers must use
        # AUTOMIL_ACCELERATOR to distinguish CPU from accelerator execution.
        env["AUTOMIL_GPU"] = "0"
        env["AUTOMIL_ACCELERATOR"] = accelerator
        env["AUTOMIL_DESC"] = spec.get("description", "")
        env["AUTOMIL_NODE_ID"] = node_id
        try:
            _automil_rel = self.automil_dir.resolve().relative_to(
                self.project_root.resolve()
            )
        except ValueError as exc:
            raise RuntimeError(
                "automil_dir must live under project_root for an isolated launch"
            ) from exc
        env["AUTOMIL_DIR_REL"] = _automil_rel.as_posix()
        # Val-firewall (Scope B): born-seal every test-bearing training artifact.
        # AUTOMIL_RESULTS_DIR is the sealed subdir, so fold_*_result.json (the
        # per-fold writer), results/ (framework detail tree), and the SIGTERM
        # flush all write directly into archive/<node>/certify/ — never the
        # agent-visible node-archive root. Daemon-side readers repoint to
        # certify/ in step. Off-limits to the agent; read once by `automil certify`.
        _sealed = (archive / "certify").resolve()
        _sealed.mkdir(parents=True, exist_ok=True)
        env["AUTOMIL_RESULTS_DIR"] = str(_sealed)

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

        # A5: the AUTOMIL_ whitelist prefix forwards the operator's ambient
        # environment, so a leftover shell `export AUTOMIL_CERTIFY=1` (from a
        # past certification run) would flip the consumer test-print gates in
        # every search child. The daemon never sets it during search — drop it.
        env.pop("AUTOMIL_CERTIFY", None)

        # 4. Per-spec env (last-write-wins, except blocked keys).
        for k, v in spec.get("env", {}).items():
            if k not in _SPEC_ENV_BLOCKED:
                env[k] = str(v)

        return env

    def _apply_editable_overlay_guard(
        self, env: dict[str, str], wt_path: Path
    ) -> None:
        """SCH-02 / D-03: opt-in editable-install worktree PYTHONPATH guard.

        When self.editable_overlay_guard is True, prepends the worktree-relative
        equivalent of each editable source root to env["PYTHONPATH"] so that
        Python resolves imports from the worktree overlay first.

        Called from _launch AFTER _build_subprocess_env returns.
        Default OFF (editable_overlay_guard: false) preserves D-199/DEC-01
        invariant: PYTHONPATH is NOT force-set unless the operator opts in.

        Roots outside self.project_root are silently skipped (ValueError).
        Only roots whose worktree counterpart actually exists are prepended.
        """
        if not self.editable_overlay_guard:
            return
        prepends: list[str] = []
        for editable_root in _collect_editable_source_roots():
            root_p = Path(editable_root)
            try:
                rel = root_p.relative_to(self.project_root)
            except ValueError:
                continue  # editable root not under project_root; skip
            wt_candidate = wt_path / rel
            if wt_candidate.is_dir():
                prepends.append(str(wt_candidate))
        if prepends:
            existing_pp = env.get("PYTHONPATH", "")
            existing_parts = existing_pp.split(":") if existing_pp else []
            # Only prepend paths not already present; dedup prevents misleading
            # "prepended N path(s)" log when the daemon inherited PYTHONPATH
            # that already includes the worktree src (common in dev environments).
            new_parts = [p for p in prepends if p not in existing_parts]
            if new_parts:
                env["PYTHONPATH"] = ":".join(new_parts + existing_parts)
                logger.debug(
                    "editable_overlay_guard: prepended %d path(s) to PYTHONPATH for wt=%s",
                    len(new_parts), wt_path,
                )

    # --- Cell cap enforcement at the launch path (CAP-1 / H-2) ---

    # Sentinel: the spec declares a cell whose file is currently missing or
    # unreadable. Distinct from None (no declared cell) because the two have
    # opposite admission outcomes: no-cell specs launch uncapped by design,
    # unreadable-cell specs are HELD — never launched unmetered, never
    # canceled by a possibly transient read failure.
    _CELL_UNREADABLE = object()

    def _cell_for_spec(self, spec: dict, *, warn: bool = True):
        """Return the spec's Cell, None (no identity), or ``_CELL_UNREADABLE``.

        Resolves ``cells/`` from ``self.automil_dir`` rather than through
        ``cells.get_cell`` — the registry's cwd-walking ``_find_automil_dir()``
        fallback would find the host project's overlay when the daemon runs from
        another cwd (same reasoning as ``_tick_cells``).

        ``warn=False`` suppresses the unresolvable-cell warning for repeat
        lookups within one launch, so the operator sees it once, not per call.
        """
        from automil.cells import read_cell

        cell_id = (spec.get("metadata") or {}).get("cell_id")
        if not cell_id:
            return None
        path = self.automil_dir / "cells" / f"{cell_id}.json"
        if not path.exists():
            if warn:
                logger.error(
                    "Spec %s references cell %s which has no cells/<id>.json; "
                    "the launch is held until the cell file is readable.",
                    spec.get("id"), str(cell_id)[:8],
                )
            return self._CELL_UNREADABLE
        try:
            return read_cell(path)
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError) as exc:
            if warn:
                logger.error(
                    "Could not read cell %s for spec %s (%s); the launch is "
                    "held until the cell file is readable.",
                    str(cell_id)[:8], spec.get("id"), exc,
                )
            return self._CELL_UNREADABLE

    def _observe_activity_for_tick(self):
        """Return one shared native-meter observation for this daemon tick.

        Open journaled sessions, rather than mutable ``config.yaml``, decide
        whether the exporter is relevant. Completed sessions use their durable
        final sample and never probe an endpoint that has legitimately exited.
        A missing endpoint is represented as data and never converted into
        consumed budget.
        """
        from automil.activity_metrics import observe_activity_metrics
        from automil.cells import list_cells
        from automil.cells.activity import (
            ACTIVITY_JOURNAL_FILENAME,
            ActivityError,
            read_activity_report,
            read_unbound_activity_report,
        )

        if not (self.automil_dir / ACTIVITY_JOURNAL_FILENAME).exists():
            return None
        try:
            if read_unbound_activity_report(self.automil_dir).open_sessions:
                return observe_activity_metrics(self.automil_dir)
            for cell in list_cells(self.automil_dir / "cells"):
                if (
                    cell.mode == "agent_active"
                    and read_activity_report(
                        self.automil_dir, cell.cell_id,
                    ).open_sessions
                ):
                    return observe_activity_metrics(self.automil_dir)
        except ActivityError:
            # The per-cell assessment path reports the durable journal error.
            # A scrape cannot repair corrupt lifecycle evidence.
            return None
        return None

    def _note_activity_health(self, cell, assessment) -> None:
        """Log activity degradation and recovery once per state change."""
        if cell.status.value == "finalized" or assessment.complete:
            self._activity_health_by_cell.pop(cell.cell_id, None)
            return
        reason = None if assessment.admissible else assessment.reason
        previous = self._activity_health_by_cell.get(cell.cell_id)
        if reason is None:
            if previous is not None:
                logger.info(
                    "Claude active-time telemetry recovered for cell %s",
                    cell.cell_id[:8],
                )
                self._activity_health_by_cell.pop(cell.cell_id, None)
            return
        if previous != reason:
            logger.warning(
                "Holding new work for cell %s until active-time telemetry "
                "recovers: %s",
                cell.cell_id[:8], reason,
            )
            self._activity_health_by_cell[cell.cell_id] = reason

    def _cell_admission(self, spec: dict, *, activity_observation=None):
        """Classify a queued cell spec without mutating queue or cap state."""
        from automil.cells import blocks_new_work
        from automil.cells.activity import (
            ActivityError,
            assess_activity,
            read_activity_report,
        )

        cell_id = (spec.get("metadata") or {}).get("cell_id")
        if not cell_id:
            return _CellAdmission.ALLOW
        cell = self._cell_for_spec(spec, warn=False)
        if cell is self._CELL_UNREADABLE:
            # Fail closed without destruction: a declared-but-unreadable cell
            # blocks the launch but keeps queue and graph state intact. A
            # transient read error (fd pressure, filesystem blip) recovers on
            # a later tick; refusal here would cancel the node irreversibly.
            if str(cell_id) not in self._unreadable_cell_logged:
                logger.warning(
                    "Holding new work for cell %s: declared cell file is "
                    "missing or unreadable.",
                    str(cell_id)[:8],
                )
                self._unreadable_cell_logged.add(str(cell_id))
            return _CellAdmission.HOLD_TELEMETRY
        self._unreadable_cell_logged.discard(str(cell_id))
        if blocks_new_work(cell):
            return _CellAdmission.REFUSE_CAP
        if cell.mode != "agent_active":
            return _CellAdmission.ALLOW
        try:
            report = read_activity_report(self.automil_dir, cell.cell_id)
            assessment = assess_activity(report, activity_observation)
        except ActivityError as exc:
            previous = self._activity_health_by_cell.get(cell.cell_id)
            reason = str(exc)
            if previous != reason:
                logger.warning(
                    "Holding new work for cell %s: invalid activity evidence: %s",
                    cell.cell_id[:8], reason,
                )
                self._activity_health_by_cell[cell.cell_id] = reason
            return _CellAdmission.HOLD_TELEMETRY
        self._note_activity_health(cell, assessment)
        return (
            _CellAdmission.ALLOW
            if assessment.admissible
            else _CellAdmission.HOLD_TELEMETRY
        )

    def _block_cell_spec(self, spec: dict, *, activity_observation=None) -> bool:
        """Apply the tri-state launch gate; true means do not launch now."""
        admission = self._cell_admission(
            spec, activity_observation=activity_observation,
        )
        if admission is _CellAdmission.ALLOW:
            return False
        if admission is _CellAdmission.HOLD_TELEMETRY:
            # The exporter can recover.  Keep both queue and graph state intact.
            return True
        return self._refuse_closed_cell_spec(spec)

    def _refuse_closed_cell_spec(self, spec: dict) -> bool:
        """Refuse a queued spec whose budget cell has closed. True iff refused (CAP-1).

        Until this existed, only ``automil submit`` was gated: any spec already
        sitting in ``queue/`` when its cell flipped to REFUSING_NEW still
        launched, so the cap bounded the front door and nothing else.

        Refusal mirrors ``automil dequeue`` — unlink ``queue/<node>.json`` and
        ``graph.cancel()`` the node. The spec is NOT left queued: the cap state
        machine is monotone, so a closed cell never re-opens and the spec would
        strand its node id forever (submit refuses a duplicate queue entry, and
        children refuse a still-running parent). It is not marked crashed either
        — nothing ran, so a crash row would poison the failure statistics and,
        via ``completed/``, would have ``reconcile`` promote a phantom executed
        node with composite 0.0.

        Specs with no ``metadata.cell_id`` remain valid because ``Backend.submit``
        is a first-class non-cell submission path. A spec that *declares* a cell
        but references missing or malformed state is HELD by ``_cell_admission``,
        not refused: it never launches unmetered, and it is never canceled by a
        possibly transient read failure either — refusal is irreversible and
        must be reserved for a readable cell that has genuinely closed.

        This method is called on every poll for every pending spec, so it stays
        silent unless it actually refuses.
        """
        from automil.cells import blocks_new_work
        from automil.cells.state import CellStatus

        node_id = spec.get("id")
        cell_id = (spec.get("metadata") or {}).get("cell_id")
        if not cell_id:
            return False
        cell = self._cell_for_spec(spec, warn=False)
        if cell is None or cell is self._CELL_UNREADABLE:
            # Held by _cell_admission, never refused: refusal cancels the node
            # irreversibly, and a missing/unreadable cell file may be a
            # transient condition. (None is unreachable here — no-cell specs
            # never route to refusal — but is treated identically for safety.)
            return False
        if not blocks_new_work(cell):
            return False
        if (
            node_id in cell.billed_node_ids
            and cell.status not in (CellStatus.TERMINATING, CellStatus.FINALIZED)
        ):
            # A9 exactly-once, final-attempt corner: an attempt that was already
            # BILLED (crash/unlink-abort inside its launch window, now retrying)
            # is paid-for work being completed, not new work — and on the last
            # budgeted attempt the bill itself flips evals_exhausted, so refusing
            # here would stamp the archived spec cap_refused and leave the freeze
            # census permanently one short of consumed_evals. Exempt it, unless
            # the TIME axis has already escalated to terminating/finalized (the
            # hard wall may not be crossed to start new processes).
            logger.info(
                "Launching already-billed %s despite %s cell %s: a charged "
                "attempt being retried is not new work.",
                node_id, cell.status.value, cell.cell_id[:8],
            )
            return False
        logger.warning(
            "Refusing to launch %s: cell %s is %s "
            "(consumed_evals=%d/%s). Dequeuing the spec and cancelling "
            "the node — a closed cell never re-opens.",
            node_id, cell.cell_id[:8], cell.status.value,
            cell.consumed_evals,
            cell.eval_budget if cell.eval_budget is not None else "-",
        )

        src_file = spec.get("_file")
        if src_file and Path(src_file).exists():
            try:
                Path(src_file).unlink()
            except OSError as exc:
                logger.warning("Could not remove refused queue spec %s: %s", src_file, exc)

        # Audit trail: the archived spec records why this node never ran.
        try:
            archive = self.archive_dir / node_id
            archive.mkdir(parents=True, exist_ok=True)
            spec_clean = {k: v for k, v in spec.items() if k != "_file"}
            spec_clean["metadata"] = {
                **(spec_clean.get("metadata") or {}),
                "cancel_reason": "cap",
                "cap_refused": True,
            }
            (archive / "spec.json").write_text(json.dumps(spec_clean, indent=2))
        except OSError:
            logger.exception("Could not archive refused spec for %s", node_id)

        self._cancel_node_for_cap_refusal(node_id, str(cell_id))
        return True

    def _cancel_node_for_cap_refusal(self, node_id: str, cell_id: str) -> None:
        """Mark a cap-refused node cancelled in graph.json (same shape as dequeue)."""
        from automil.graph import locked_update, merged_metadata

        try:
            with locked_update(
                str(self.graph.path),
                technique_map=getattr(self.graph, "_technique_map", None),
            ) as g:
                node = g.get_node(node_id)
                if node is None:
                    logger.debug(
                        "cap refusal: %s is not in the graph; nothing to cancel", node_id
                    )
                    return
                g.cancel(node_id)
                node["cancel_reason"] = "cap"
                # L-8a: copy-on-write (graph.merged_metadata) — node["metadata"]
                # can be aliased with another node's dict (gate/evaluate.py
                # creates gate-eval children via a shallow dict(node) copy).
                node["metadata"] = merged_metadata(node, {"cap_refused": True})
                node.setdefault("cell_id", cell_id)
        except Exception:  # noqa: BLE001 — a graph failure must not wedge the loop
            logger.exception("cap refusal: could not cancel graph node %s", node_id)

    def _record_cell_launch(self, spec: dict) -> None:
        """Bill one evaluation to the dispatching cell (H-2, A9).

        Called when the attempt's spec is archived — the moment it becomes a
        countable attempt — not after Popen. Everything past the cap-refusal
        gate counts: crashed, partial, budget-killed, and pre-spawn failures
        (admissibility, base_commit, worktree, Popen) alike, because equal
        effort means equal attempts, not equal successes, and because the
        campaign freeze requires archived attempts == billed attempts exactly
        (a pre-spawn failure that archived without billing deadlocked the cell
        permanently).

        Also the single place an unbillable launch is reported: exactly once per
        dispatch, so the operator can find (and the paper can quantify) every
        evaluation that no cell budget paid for.
        """
        if not (spec.get("metadata") or {}).get("cell_id"):
            logger.warning(
                "Launched %s with no metadata.cell_id: this evaluation is billed to "
                "no cell budget and is invisible to per-cell effort accounting "
                "(legacy spec, or a submission path other than `automil submit`).",
                spec.get("id"),
            )
            return
        cell = self._cell_for_spec(spec, warn=True)
        if cell is None or cell is self._CELL_UNREADABLE:
            return  # unresolvable cell — already reported by _cell_for_spec
        self._bump_cell_counters(spec, consumed_delta=1)

    def _record_cell_completion(self, spec: dict, status: str) -> None:
        """Record a usable result against the cell (H-2, reported secondary).

        Only ``completed`` / ``partial`` count. This is never the cap — if
        crashes were free retries the budget would stop being a budget — it
        exists so per-cell effort can be quoted as both attempts and usable
        results.
        """
        if status not in ("completed", "partial"):
            return
        self._bump_cell_counters(spec, completed_delta=1)

    def _bump_cell_counters(self, spec: dict, *, consumed_delta: int = 0,
                            completed_delta: int = 0) -> None:
        """Immutable read-modify-write of a cell's eval counters.

        A9 exactly-once: the consumed (billing) axis is keyed on the node id —
        ``_launch`` can re-process one node after a crash inside its
        archive→queue-unlink window or a failed queue unlink, and a re-bill
        would strand the cell with ``consumed_evals`` above the archived-spec
        census (the mirror image of the under-billing deadlock A9 fixed). The
        membership check and the counter advance land in one atomic
        ``write_cell``.
        """
        from dataclasses import replace

        from automil.cells import write_cell

        cell = self._cell_for_spec(spec, warn=False)
        if cell is None or cell is self._CELL_UNREADABLE:
            return
        node_id = str(spec.get("id") or "")
        billed = list(cell.billed_node_ids)
        if consumed_delta and node_id:
            if node_id in billed:
                consumed_delta = 0
            else:
                billed.append(node_id)
        try:
            write_cell(
                replace(
                    cell,
                    consumed_evals=cell.consumed_evals + consumed_delta,
                    completed_evals=cell.completed_evals + completed_delta,
                    billed_node_ids=billed,
                ),
                self.automil_dir / "cells",
            )
        except OSError:
            logger.exception(
                "Could not update eval counters for cell %s (node %s)",
                cell.cell_id[:8], spec.get("id"),
            )

    def _mark_node_terminal_in_graph(self, node_id: str, status: str,
                                     error: str = "") -> None:
        """Record a terminal status on the graph node (M-5, M-7).

        Both crash paths used to write ``archive/result.json`` and
        ``completed/<id>.json`` and stop there, leaving ``graph.json`` describing
        a node that is still queued or running. ``reconcile`` would eventually
        repair it — but only if somebody ran it, which on a daemon-driven
        campaign may be never, so ``automil rank`` and ``automil status`` showed
        a node that never resolves.

        Routed through ``mark_failed`` rather than assigning fields here, because
        that is also where ``meta.total_executed`` / ``total_proposed`` are
        maintained (M-7): ``total_executed`` is the UCB exploration denominator
        (``graph.py``: ``sqrt(log(total) / (1 + child_count))``), so a daemon that
        never incremented it explored against a frozen count.

        Idempotent. Orphan recovery and ``_mark_crashed`` can both fire for one
        node, and double-counting the denominator would be its own distortion, so
        a node already marked executed is left alone.
        """
        from automil.graph import locked_update

        try:
            with locked_update(
                str(self.graph.path),
                technique_map=getattr(self.graph, "_technique_map", None),
            ) as g:
                node = g.get_node(node_id)
                if node is None:
                    return          # Backend.submit path the graph never saw
                if node.get("type") == "executed":
                    return          # already terminal — do not re-bill the counters
                g.mark_failed(node_id, status, error)
        except Exception:  # noqa: BLE001 — a graph failure must not wedge the daemon
            logger.exception("could not mark %s as %s in the graph", node_id, status)

    def _device_provenance(self, slot_id: int) -> dict[str, str | int | None]:
        """Map an internal scheduler slot to truthful hardware provenance."""
        accelerator = getattr(self, "_accelerator", "") or "cuda"
        return {
            "accelerator": accelerator,
            "gpu": None if accelerator == "cpu" else slot_id,
        }

    @staticmethod
    def _execution_label_from_metadata(metadata: dict) -> str:
        """Human-readable device label without describing CPU as a GPU."""
        accelerator = str(metadata.get("accelerator") or "cuda").lower()
        gpu = metadata.get("gpu")
        if accelerator == "cpu":
            return "CPU"
        return f"{accelerator.upper()} GPU {gpu}"

    def _execution_label(self, slot_id: int) -> str:
        return self._execution_label_from_metadata(self._device_provenance(slot_id))

    def _write_launch_intent(self, spec: dict, gpu_id: int, worktree) -> Path:
        """Record the intent to launch BEFORE spawning the process (M-6).

        The running spec proper cannot be written before ``Popen`` — it carries
        the pid. So the window between the spawn and that write was one in which
        a daemon death left the node queued forever AND the training process
        alive, holding its GPU, invisible to orphan recovery.

        Writing an intent record first cannot recover the pid; nothing can. What
        it does is split the failure in two and fix the half that is fixable: the
        node is now correctly marked crashed on the next start, and the leak is
        REPORTED with the worktree that identifies it, instead of being silent.
        ``_launch`` overwrites this file with the real running spec moments later.
        """
        path = self._backend_running_dir("local") / f"{spec.get('id')}.json"
        payload = {k: v for k, v in spec.items() if k != "_file"}
        meta = dict(payload.get("metadata") or {})
        meta.update({
            "launch_phase": "launching",
            "pid": None,
            **self._device_provenance(gpu_id),
            "worktree": str(worktree),
            "intent_at": datetime.now().isoformat(),
        })
        payload["metadata"] = meta
        path.write_text(json.dumps(payload, indent=2))
        return path

    def _launch(self, spec: dict, gpu_id: int, *, activity_observation=None):
        """Launch an experiment in an isolated git worktree."""
        node_id = spec["id"]
        # The cap and telemetry gate must bind here, not only at submit time.
        # This reloads persisted cell state immediately before worktree creation.
        if self._block_cell_spec(
            spec, activity_observation=activity_observation,
        ):
            return
        archive = self.archive_dir / node_id
        archive.mkdir(parents=True, exist_ok=True)

        # Save spec (without internal keys)
        spec_clean = {k: v for k, v in spec.items() if k not in ("_file",)}
        (archive / "spec.json").write_text(json.dumps(spec_clean, indent=2))

        # A9 (claims-alignment): bill at archive time, not after Popen. The
        # freeze census counts archived non-cap-refused specs and requires it
        # to equal the cell's consumed evals exactly — billing only spawned
        # processes let every pre-spawn failure (admissibility, base_commit,
        # worktree, Popen) archive an unbilled spec and deadlock the cell
        # permanently. Billing is exactly-once per node id (the cell's
        # billed_node_ids key), so a crash inside this window or a failed
        # queue unlink cannot re-bill on retry. Residual fail-open paths
        # (cell file missing/unreadable, counter write failure) remain
        # deliberate and are loudly logged — they are I/O faults, not the
        # per-launch failure modes that used to deadlock every cell.
        self._record_cell_launch(spec)

        # Remove from queue before attempting launch (prevents infinite retry).
        # A9: on unlink failure, abort THIS attempt instead of launching with
        # the queue file still present — the next tick retries cleanly (the
        # bill is idempotent, the archive write is an overwrite), whereas
        # launching now would let the still-queued spec double-dispatch.
        src_file = spec.get("_file")
        if src_file and Path(src_file).exists():
            try:
                Path(src_file).unlink()
            except OSError:
                logger.exception(
                    "Could not unlink queue file for %s; aborting this launch "
                    "attempt (billed once; retried next tick)", node_id,
                )
                return

        # LCH-1/LCH-3: submit-time admissibility is evidence, not authority.
        # Recompute it from the live policy and exact archived overlay before a
        # worktree is created. Architecture-preserving legacy specs, policy
        # drift, unmanifested files, and changed variant selections fail closed.
        try:
            from automil.admissibility import (
                load_candidate_policy,
                revalidate_candidate_spec,
                validate_campaign_binding,
            )

            _candidate_policy = load_candidate_policy(self.automil_dir)
            _overlay_rel = spec.get("overlay_dir")
            _overlay_path = (
                self.orch_dir / str(_overlay_rel)
                if _overlay_rel
                else archive
            )
            revalidate_candidate_spec(_candidate_policy, spec, _overlay_path)

            # The candidate overlay is only half the launched identity. Pin the
            # config-owned base command as well, and for campaign cells recheck
            # the immutable manifest bytes immediately before worktree creation.
            _expected_command_hash = spec.get("base_run_command_sha256")
            _live_command_hash = hashlib.sha256(
                (self.run_command or "").encode()
            ).hexdigest()
            if _expected_command_hash is None:
                if _candidate_policy.mode == "architecture-preserving":
                    raise ValueError(
                        "architecture-preserving spec is missing base_run_command_sha256"
                    )
            elif _expected_command_hash != _live_command_hash:
                raise ValueError(
                    "base run command changed between submit and launch"
                )

            _campaign = ((spec.get("metadata") or {}).get("campaign"))
            if _campaign is not None:
                if not isinstance(_campaign, dict):
                    raise ValueError("spec metadata.campaign must be a mapping")
                _manifest_rel = Path(str(_campaign.get("manifest", "")))
                if (
                    not _manifest_rel.parts
                    or _manifest_rel.is_absolute()
                    or ".." in _manifest_rel.parts
                ):
                    raise ValueError("campaign manifest path is unsafe")
                _manifest_path = self.project_root / _manifest_rel
                if not _manifest_path.is_file():
                    raise ValueError(f"campaign manifest missing: {_manifest_rel}")
                _actual_manifest_hash = hashlib.sha256(
                    _manifest_path.read_bytes()
                ).hexdigest()
                if _actual_manifest_hash != _campaign.get("manifest_sha256"):
                    raise ValueError("campaign manifest changed between submit and launch")
                validate_campaign_binding(
                    _manifest_path,
                    _campaign,
                    base_run_command=self.run_command,
                    budget_cell_id=str((spec.get("metadata") or {}).get("cell_id", "")),
                )
        except Exception as exc:  # fail closed at the last pre-launch seam
            logger.error(
                "Spec for %s failed launch-time admissibility: %s",
                node_id,
                exc,
            )
            self._mark_crashed(
                node_id,
                spec,
                f"launch-time admissibility failed: {exc}",
            )
            return

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
            # HASH-0: verify the digests `automil submit` recorded before any
            # file lands. Until now the manifest was written into every spec and
            # checked by nothing, so an archive edited between submit and launch
            # would run under the original node's label.
            self.runner.apply_overlay(
                wt_path, self.orch_dir / overlay_dir, deletions=deletions,
                manifest=spec.get("overlay_manifest"),
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

        # SCH-02 / D-03: opt-in editable-install worktree PYTHONPATH guard.
        # Post-processing step after _build_subprocess_env — signature unchanged.
        self._apply_editable_overlay_guard(env=env, wt_path=wt_path)

        log_path = archive / "run.log"
        log_fh = open(log_path, "w")
        try:
            if self.run_command:
                cmd = shlex.split(self.run_command)
            else:
                cmd = [sys.executable, self.run_script]
            # D-04 (CFG-03): append per-node override args after base run.command.
            # Must be list append (not string concat) and Popen must not use shell=True
            # so shlex.split tokenizes metacharacters as literal tokens (T-11-03-01).
            override_str = spec.get("run_command_override")
            if override_str:
                cmd = cmd + shlex.split(override_str)
            # M-6: intent BEFORE the side effect. Overwritten with the real
            # running spec (pid/pgid/starttime) a few lines below.
            self._write_launch_intent(spec, gpu_id, wt_path)
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

        # H-2/A9: billing happened at archive time (top of _launch), so crashed,
        # budget-killed, and pre-spawn-failed attempts all cost the same as
        # successful ones and the archive census equals the billed count.

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
        running_spec_meta.update(self._device_provenance(gpu_id))
        recorded_starttime = _read_proc_starttime(process.pid)
        if recorded_starttime is not None:
            running_spec_meta["starttime_ticks"] = recorded_starttime
        running_spec_payload["metadata"] = running_spec_meta
        running_spec_path.write_text(json.dumps(running_spec_payload, indent=2))

        logger.info(
            "Launched %s on %s (PID %d, est. %sGB, timeout %smin)",
            node_id,
            self._execution_label(gpu_id),
            process.pid,
            estimated_vram,
            timeout_min,
        )

    def _running_in_cell(self, cell_id: str) -> list:
        """Return _NodeHandle list for in-self.running experiments tagged with cell_id.

        cell_id matching uses spec["metadata"]["cell_id"] (set by submit, Plan 04-06).
        Specs submitted directly through a backend without a cell identity do
        not belong to a budget cell and therefore never match.

        Returns:
            List of _NodeHandle(node_id=...) for matching experiments.
        """
        result = []
        for node_id, exp in self.running.items():
            spec_meta = (exp.spec or {}).get("metadata", {}) or {}
            if spec_meta.get("cell_id") == cell_id:
                result.append(_NodeHandle(node_id=node_id))
        return result

    def _tick_cells(self, *, activity_observation=None) -> None:
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
            CellStatus,
            list_cells,
            next_status,
            write_cell,
        )
        from automil.cells.activity import (
            ActivityError,
            assess_activity,
            read_activity_report,
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
        for cell in list_cells(cells_dir):
            agent_active_seconds = None
            if cell.mode == "agent_active":
                try:
                    activity = read_activity_report(
                        self.automil_dir, cell.cell_id,
                    )
                    assessment = assess_activity(
                        activity, activity_observation,
                    )
                    self._note_activity_health(cell, assessment)
                    # Even while live telemetry is degraded, the last authentic
                    # cumulative sample remains valid input to the pure cap
                    # reducer.  No unavailable observation can add seconds.
                    agent_active_seconds = assessment.active_seconds
                except ActivityError as exc:
                    reason = str(exc)
                    if self._activity_health_by_cell.get(cell.cell_id) != reason:
                        logger.warning(
                            "_tick_cells: preserving cell %s because its activity "
                            "evidence is invalid: %s",
                            cell.cell_id[:8], reason,
                        )
                        self._activity_health_by_cell[cell.cell_id] = reason
                    # Invalid evidence cannot safely drive any time transition.
                    # Evaluation-count transitions are handled on later valid
                    # ticks; never synthesize the budget to force closure.
                    continue
            running = self._running_in_cell(cell.cell_id)
            new_status = next_status(
                cell,
                now,
                len(running),
                agent_active_seconds=agent_active_seconds,
            )
            if new_status == cell.status:
                continue
            if new_status == CellStatus.TERMINATING:
                for handle in running:
                    # D-124 / Pitfall 4: write cancel_reason='cap' BEFORE
                    # calling cancel so reconcile_budget_kill can detect cap kills
                    # even if the SIGTERM handler races the annotation write.
                    # WR-04 fix: use _read_backend_name_for_node + _backend_running_dir
                    # so SLURM/Ray running specs are found under running/slurm/ and
                    # running/ray/. The original self.running_dir (= running/local/)
                    # always missed non-local backends, silently skipping the annotation
                    # and causing _was_cap_killed_completion to return False for all
                    # SLURM/Ray cap-triggered cancels.
                    _backend_name = self._read_backend_name_for_node(handle.node_id)
                    running_spec_path = self._backend_running_dir(_backend_name) / f"{handle.node_id}.json"
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

        Uses backend-aware running spec path (IN-01 fix / D-169) so SLURM/Ray
        nodes don't silently fall through to the archive-spec fallback.
        """
        _backend_fc = self._read_backend_name_for_node(node_id)
        for path in (
            self._backend_running_dir(_backend_fc) / f"{node_id}.json",
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

        # B3 (claims-alignment): drain the remote backend log BEFORE collection
        # and redaction — it used to run last, so on SLURM/Ray the H-1
        # redaction below operated on a file that did not exist yet (a no-op),
        # and the crash-synthesis path classified OOM/timeout against an empty
        # log. One moved call fixes all three. No-op for the local backend.
        self._drain_remote_backend_log(node_id, archive)

        result = self._collect_or_synthesize_result(node_id, archive, returncode, wt_path)

        # H-1 (audit 2026-07-23): run.log is the raw training stdout at the
        # AGENT-VISIBLE archive root, so a script that prints a test metric leaks
        # the sealed quantity. Redact held-out lines at the orchestrator boundary
        # (defence-in-depth over per-script gating, which a re-vendored upstream
        # can silently regress).
        from automil.firewall import held_out_keys, redact_held_out, redact_log_file
        _ho_keys = held_out_keys(result)
        try:
            _n_redacted = redact_log_file(archive / "run.log", _ho_keys)
            if _n_redacted:
                logger.warning(
                    "val-firewall: redacted %d held-out line(s) from %s/run.log — "
                    "the training script printed test metrics to stdout",
                    _n_redacted, node_id,
                )
        except Exception:  # noqa: BLE001 — log hygiene must never fail a completion
            logger.exception("firewall: run.log redaction failed for %s", node_id)

        # Include error details in result for better agent visibility before terminal write
        status = result.get("status", "completed")
        if status in ("crash", "oom", "timeout"):
            log_path = archive / "run.log"
            error_tail = ""
            if log_path.exists():
                lines = log_path.read_text().splitlines()
                error_tail = "\n".join(lines[-20:])
            result = dict(result)
            # The tail lands in the agent-facing result.json — redact it too.
            result.setdefault("error", redact_held_out(error_tail, _ho_keys))
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
            gpu_id=self._device_provenance(gpu_id)["gpu"],
            accelerator=self._device_provenance(gpu_id)["accelerator"],
        )

        # H-2: record a usable result against the cell (reported secondary; the
        # attempt was already billed at launch).
        self._record_cell_completion(spec, result.get("status", ""))

        # (B3: the D-170 cross-backend log drain moved above collection so the
        # drained log is covered by redaction and crash classification.)

        # Clean running spec — use backend-aware path (WR-02 fix: D-169).
        # self.running_dir is the alias for running/local/ only; SLURM/Ray specs
        # live under running/<backend>/ and would never be found or cleaned here.
        _backend_name_cleanup = self._read_backend_name_for_node(node_id)
        running_spec = self._backend_running_dir(_backend_name_cleanup) / f"{node_id}.json"
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
            "Completed %s: status=%s, composite=%.4f, elapsed=%.1fmin, %s",
            node_id,
            status_str,
            composite,
            elapsed_s / 60,
            self._execution_label(gpu_id),
        )

    # --- _handle_completion helpers (each independently testable) ---

    def _recorded_cancel_reason(self, node_id: str) -> str | None:
        """The ``metadata.cancel_reason`` recorded for this node, if any (H-7).

        ``'cap'`` is stamped by ``_tick_cells`` and ``_refuse_closed_cell_spec``;
        ``'cli'`` by ``automil cancel``. Both are written into the running spec
        BEFORE the kill, so a completion that finds one knows the process did not
        die of its own accord.

        This exists because a deliberate stop and a real failure are the same
        observation from the daemon's side — a dead process and no result.json.
        Without the annotation both became ``crash``, which poisons the failure
        statistics the gate's health diagnostic reads and makes an operator's
        cancel indistinguishable from a bug in the training code.
        """
        _backend = self._read_backend_name_for_node(node_id)
        for _spec_path in (
            self._backend_running_dir(_backend) / f"{node_id}.json",
            self.archive_dir / node_id / "spec.json",
        ):
            if _spec_path.exists():
                try:
                    _raw = json.loads(_spec_path.read_text())
                    reason = (_raw.get("metadata") or {}).get("cancel_reason")
                    if reason:
                        return str(reason)
                except (json.JSONDecodeError, OSError):
                    pass
        return None

    def _was_cap_killed_completion(self, node_id: str) -> bool:
        """True iff the running or archive spec has metadata.cancel_reason == 'cap'.

        Reads running/<backend>/<node>.json first (annotation written by
        _tick_cells BEFORE backend.cancel() is called — Pitfall 4 ordering
        guarantee). Uses the backend-aware path so SLURM/Ray annotations in
        running/slurm/ or running/ray/ are found correctly (WR-02 fix / D-169).
        Falls back to archive/<node>/spec.json if running/ was already cleaned.
        """
        return self._recorded_cancel_reason(node_id) == "cap"

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
            gpu_id=(
                self._device_provenance(int(gpu_id))["gpu"]
                if isinstance(gpu_id, int) and gpu_id >= 0
                else gpu_id
            ),
            accelerator=(
                self._device_provenance(int(gpu_id))["accelerator"]
                if isinstance(gpu_id, int) and gpu_id >= 0
                else getattr(self, "_accelerator", "") or "cuda"
            ),
        )
        logger.info(
            "Cap-driven cancel reconciled for %s: status=%s composite=%.4f "
            "partial_folds=%d/%d",
            node_id, payload["status"], payload["composite"],
            payload.get("partial_folds", 0), payload.get("expected_folds", 0),
        )
        # H-2: a budget-killed run that produced folds still yielded a usable
        # result. The discriminator is the terminal status, not the cause.
        self._record_cell_completion(spec, payload.get("status", ""))
        # Clean running spec and worktree — use backend-aware path (WR-02 fix / D-169).
        _backend_name_cap = self._read_backend_name_for_node(node_id)
        running_spec = self._backend_running_dir(_backend_name_cap) / f"{node_id}.json"
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
            # D-03 (REC-01): try fold aggregation BEFORE synthesising from log
            # heuristics. If the process was SIGKILLed before the flush handler
            # could run, fold files may still exist even without a result.json.
            # Val-firewall (Scope B): fold_*_result.json are born-sealed under
            # archive/<node>/certify/ (AUTOMIL_RESULTS_DIR), so recovery both
            # reads and re-writes there — the raw aggregated result.json (which
            # carries held_out) never lands in the agent-visible node-archive root.
            sealed = archive / "certify"
            fold_files = list(sealed.glob("fold_*_result.json"))
            if fold_files:
                from automil.cells.reconcile import aggregate_folds
                expected = self._read_fold_count_for_node(node_id)
                result = aggregate_folds(sealed, expected)
                reason = "timeout" if self._timed_out.get(node_id) else "sigkill"
                result["termination_reason"] = reason   # D-05 (REC-03)
                # Write atomically into certify/ (sealed) so a retry finds it.
                import tempfile as _tempfile
                sealed.mkdir(parents=True, exist_ok=True)
                _tmp_fd, _tmp_path = _tempfile.mkstemp(
                    dir=str(sealed), suffix=".tmp"
                )
                try:
                    with os.fdopen(_tmp_fd, "w") as _fh:
                        _fh.write(json.dumps(result, indent=2) + "\n")
                    os.replace(_tmp_path, str(sealed / "result.json"))
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
                # H-7: a deliberate stop looks exactly like a failure from here —
                # dead process, no result.json. `automil cancel` records its
                # intent in the running spec before the kill, so honour it rather
                # than overwriting `cancelled` with `crash`: a cancel counted as a
                # failure poisons the crash statistics the gate's health
                # diagnostic reads, and makes an operator stop indistinguishable
                # from a bug in the training code.
                _cancel_reason = self._recorded_cancel_reason(node_id)
                if _cancel_reason == "cli":
                    status = "cancelled"
                    termination_reason = "cancelled_by_operator"
                elif "CUDA out of memory" in log_text or "OutOfMemoryError" in log_text:
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
                    # H-1: this tail lands in the agent-facing result.json.
                    from automil.firewall import redact_held_out
                    result["error"] = redact_held_out(error_tail)
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
        # WR-01 mitigation: cap grace to MAX_GRACE so a misconfigured
        # timeout_grace_seconds cannot block the tick loop for an arbitrary
        # duration. With max_concurrent_per_gpu=8 and multiple GPUs, N
        # simultaneous timeouts each sleep `grace` seconds serially inside
        # _check_running, stacking to N×grace seconds of total blocking.
        # Known limitation: the sleep is still synchronous (a proper fix
        # requires async deadline tracking like _pending_sigkill_at). Capped
        # at 30s as a safe maximum; warn operators when configured above 15s.
        _MAX_GRACE = 30
        grace = min(
            int((self.config.get("orchestrator") or {}).get("timeout_grace_seconds", 10)),
            _MAX_GRACE,
        )
        if grace > 15:
            logger.warning(
                "timeout_grace_seconds=%ds will block the tick loop for that duration "
                "(capped at %ds); consider lowering orchestrator.timeout_grace_seconds",
                grace, _MAX_GRACE,
            )
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

        # B6 symmetry guard: this payload is locally constructed and carries no
        # sealed keys today; the strip keeps that an invariant rather than an
        # accident if the dict ever grows (this path bypasses write_terminal_state).
        result = {k: v for k, v in result.items() if k not in ("held_out", "summary")}

        (archive / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        spec_clean = {k: v for k, v in spec.items() if k not in ("_file",)}
        (archive / "spec.json").write_text(json.dumps(spec_clean, indent=2) + "\n")
        (self.completed_dir / f"{node_id}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        # M-5 / M-7: the artifacts above are what `reconcile` rebuilds from, but
        # the graph must not wait for a reconcile that may never be run.
        self._mark_node_terminal_in_graph(node_id, "crash", error)

    _TSV_TRAILING = ("composite", "vram_gb", "elapsed_min", "status", "description")

    def _append_results_tsv(self, node_id: str, result: dict, description: str = ""):
        """Append a row to results.tsv (sole writer, no locking needed).

        Metric columns come from the keys of ``result["metrics"]`` — no hardcoded
        MIL vocabulary. Header shape is
        ``node_id, <metric keys sorted>, composite, vram_gb, elapsed_min, status,
        description``.

        TSV-1: the header used to be locked by the FIRST row, and any later key it
        did not already carry was silently dropped. The preprint campaign is
        precisely the breaking case — 65 classification experiments emit
        ``val_auc``/``val_bacc``, 100 survival experiments emit ``val_c_index``,
        and whichever finished first decided which group lost its only metric.
        ``composite`` still landed, so the file looked populated.

        A genuinely new key now WIDENS the header and rewrites the file,
        backfilling earlier rows with blanks (they really had no value for that
        column). The rewrite is atomic and only happens on a schema change; a row
        whose keys the header already covers is a plain append.
        """
        metrics = result.get("metrics", {})
        composite = result.get("composite", 0.0)
        status = result.get("status", "completed")
        elapsed_s = result.get("elapsed_seconds", 0)
        vram_mb = result.get("peak_vram_mb", 0)

        bad = [k for k in metrics if "\t" in str(k) or "\n" in str(k)]
        if bad:
            raise ValueError(
                f"metric name(s) {bad} contain a tab or newline; that would shift "
                f"every column to their right in results.tsv"
            )

        trailing = list(self._TSV_TRAILING)
        existing_rows: list[list[str]] = []
        if self.results_tsv.exists() and self.results_tsv.stat().st_size:
            lines = self.results_tsv.read_text().splitlines()
            header_cols = lines[0].split("\t")
            metric_cols = [c for c in header_cols
                           if c != "node_id" and c not in set(trailing)]
            existing_rows = [ln.split("\t") for ln in lines[1:] if ln]
        else:
            header_cols, metric_cols = [], []

        new_metrics = [k for k in sorted(metrics) if k not in metric_cols]
        if new_metrics or not header_cols:
            # Schema change (or first write): widen and rewrite, backfilling the
            # rows that predate the new column(s) with blanks.
            old_metric_cols = list(metric_cols)
            metric_cols = sorted(set(metric_cols) | set(metrics))
            header_cols = ["node_id"] + metric_cols + trailing
            rebuilt = ["\t".join(header_cols)]
            for row in existing_rows:
                old_header = ["node_id"] + old_metric_cols + trailing
                by_name = dict(zip(old_header, row))
                rebuilt.append("\t".join(by_name.get(c, "") for c in header_cols))
            self._write_tsv_atomic("\n".join(rebuilt) + "\n")

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
            # A newline or tab in a free-text description would forge a row or a
            # column; the agent writes this text, so flatten rather than trust it.
            (description or node_id).replace("\n", " ").replace("\r", " ").replace("\t", " "),
        ])
        with open(self.results_tsv, "a") as f:
            f.write("\t".join(cells) + "\n")

    def _write_tsv_atomic(self, text: str) -> None:
        """Replace results.tsv in one step.

        The schema-widening rewrite is the only path that touches bytes already
        on disk, and the viz dashboard reads this file unlocked (L-8), so a
        partial write would be observable as a truncated table.
        """
        import tempfile

        directory = self.results_tsv.parent
        directory.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".results-", suffix=".tsv")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(text)
            os.replace(tmp, self.results_tsv)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

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
        new_policy = orch_cfg.get("scheduling_policy", self.scheduling_policy)
        if new_policy != self.scheduling_policy:
            logger.info(
                "Config reload: scheduling_policy %r -> %r",
                self.scheduling_policy,
                new_policy,
            )
            self.scheduling_policy = new_policy
        # _rr_cursor is intentionally NOT reset on policy change or on topology change.
        # Rationale: (1) resetting on policy change would re-visit recently-used GPUs
        # when an operator briefly switches policy and reverts; (2) cursor % len(candidates)
        # is always a valid index regardless of how many candidates are currently eligible,
        # so correctness is preserved across topology changes (e.g. a GPU going offline).
        # The counter grows unbounded but Python int has no overflow.
        new_guard = bool(orch_cfg.get("editable_overlay_guard", self.editable_overlay_guard))
        if new_guard != self.editable_overlay_guard:
            logger.info(
                "Config reload: editable_overlay_guard %r -> %r",
                self.editable_overlay_guard, new_guard,
            )
            self.editable_overlay_guard = new_guard

    def tick(self):
        """Single scheduling cycle."""
        # 0. Hot-reload config so concurrency bumps take effect live
        self._reload_orchestrator_config()

        activity_observation = self._observe_activity_for_tick()

        # 1. Check running experiments
        self._check_running()

        # Phase 4 step 1.5: cap state machine (D-114).
        self._tick_cells(activity_observation=activity_observation)

        # 2. Schedule pending experiments (skip if draining)
        if not self.draining:
            pending = self._get_pending()
            for spec in pending:
                if not spec.get("id"):
                    self.counter += 1
                    spec["id"] = f"{self.counter:04d}"

                # CAP-1: decide admission BEFORE the GPU search, so a spec whose
                # cell has closed is withdrawn immediately instead of lingering
                # in the queue for as long as the cluster stays busy.
                if self._block_cell_spec(
                    spec, activity_observation=activity_observation,
                ):
                    continue

                needed_gb = spec.get("estimated_vram_gb", self.default_vram)
                gpu = self._find_best_gpu(needed_gb)

                if gpu is not None and self._pre_launch_check(gpu, needed_gb):
                    self._launch(
                        spec, gpu, activity_observation=activity_observation,
                    )

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

        try:
            logger.info(
                "Orchestrator started. Accelerator: %s, slots: %s, poll=%ss, safety=%sGB",
                self._accelerator or "cuda",
                list(self.gpu_allocations.keys()),
                self.poll_interval,
                self.safety_margin_gb,
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
        # A malformed GPU partition must refuse startup, never silently
        # schedule on every GPU of a shared host — and a well-formed
        # partition that selects no schedulable device must refuse rather
        # than daemonize into a permanently idle scheduler.
        partition = visible_gpu_ids()
        if partition is not None:
            print(
                "GPU partition (AUTOMIL_VISIBLE_GPUS): "
                + ",".join(str(index) for index in sorted(partition))
            )
            if not self._cpu_only and not self.gpu_allocations:
                raise ValueError(
                    "AUTOMIL_VISIBLE_GPUS="
                    + ",".join(str(index) for index in sorted(partition))
                    + " selects no schedulable GPU on this host; fix the "
                    "partition before starting the orchestrator"
                )
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

        # Typed execution state (with legacy GPU-only fallback).
        if self.gpu_state_file.exists():
            state = json.loads(self.gpu_state_file.read_text())
            print(f"\nLast updated: {state.get('last_updated', 'unknown')}")
            print(f"Queue depth: {state.get('queue_depth', 0)}")
            print(f"Running: {state.get('total_running', 0)}")
            print(f"Completed: {state.get('total_completed', 0)}")
            print(f"Counter: {state.get('counter', 0)}")
            slots = state.get("execution_slots")
            if isinstance(slots, dict):
                print("\nExecution slots:")
                for slot_name, slot in sorted(slots.items()):
                    running_ids = slot.get("running", [])
                    capacity = slot.get("capacity", "?")
                    sched = slot.get("schedulable_free_gb")
                    sched_text = (
                        f", {sched:.1f}GB schedulable"
                        if isinstance(sched, (int, float))
                        else ""
                    )
                    print(
                        f"  {slot_name}: running={running_ids}, "
                        f"capacity={capacity}{sched_text}"
                    )
            else:
                print("\nGPUs:")
                for idx, gpu in sorted(state.get("gpus", {}).items()):
                    running_ids = gpu.get("running", [])
                    sched = gpu.get("schedulable_free_gb", 0)
                    util = gpu.get("utilization_pct", 0)
                    print(
                        f"  GPU {idx}: {sched:.1f}GB schedulable, "
                        f"{util}% util, running={running_ids}"
                    )
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
        """Submit an experiment spec to the queue.

        L-6 (audit 2026-07-23): this is the legacy in-daemon submit path
        (``automil.backends._orchestrator_daemon.main()``'s ``submit``
        subcommand). The supported path is ``automil submit`` ->
        ``cli/submit.py``, which allocates ids from graph.json under
        ``locked_update``. This path instead derives the next id from
        ``gpu_state.json``'s ``"counter"`` field, which only the daemon's
        main tick loop (``_save_state``) persists -- ``cmd_submit`` never
        advances it itself. Two near-simultaneous calls on this path
        (including one racing a daemon tick) could therefore read the same
        stale counter, compute the same id, and the second call's
        unconditional write would silently overwrite the first's queue
        spec -- the first submission lost with no error.

        Fixed with two independent guards:
          1. The id read + write is serialized under the SAME lock
             graph.json writers use (``locked_update``'s
             ``<graph_path>.lock`` sidecar). This path never touches
             graph.json's contents, but sharing its lock file makes id
             allocation mutually exclusive with every other allocator in
             the process, including a concurrent legacy submit.
          2. The write itself refuses (does not overwrite) when the target
             queue file already exists -- so any collision that still
             slips through (e.g. a caller-supplied id, or a lock-holder
             that crashed mid-write on a prior run) fails loudly instead
             of silently discarding the earlier submission.
        """
        src = Path(spec_path)
        if not src.exists():
            print(f"File not found: {spec_path}")
            sys.exit(1)

        spec = json.loads(src.read_text())

        lock_path = self.graph.path.with_suffix(self.graph.path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_f = open(lock_path, "w")
        try:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)

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
            if dst.exists():
                print(
                    f"Refusing to submit: a queue spec already exists for id "
                    f"{spec['id']!r} ({dst}). Not overwriting it."
                )
                sys.exit(1)
            dst.write_text(json.dumps(spec, indent=2) + "\n")
            print(f"Submitted experiment {spec['id']}: {spec.get('description', '?')}")
        finally:
            try:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
            finally:
                lock_f.close()


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
