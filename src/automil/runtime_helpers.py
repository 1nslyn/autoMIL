"""SIGTERM flush helper + fold count accessor (CAP-03 / D-121, D-122).

register_sigterm_flush() MUST be called from the training script's main()
BEFORE any DataLoader / multiprocessing initialisation. signal.signal()
only works in the main thread of the main interpreter — calling it from
a DataLoader worker raises ValueError (RESEARCH §Pitfall 1). [VERIFIED]

The handler exits with sys.exit(0) — NOT 130 — so the daemon's
_handle_completion treats the SIGTERM-flushed run as a graceful
completion. reconcile_budget_kill (Plan 04-05) then upgrades it to
status='executed' with metadata.budget_killed=True.
"""
from __future__ import annotations

import json
import logging
import math
import numbers
import os
import signal
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_SIGTERM_REGISTERED: bool = False  # module-level idempotent guard

#: Keys stripped from the worktree-visible result.json (L-3 / val-firewall).
#: Mirrors terminal_writer._seal_node_archive's sealed-key set exactly.
#: Duplicated rather than imported: importing from automil.terminal_writer here
#: would make it a hard dependency of runtime_helpers (imported by training
#: scripts), the same circular-import concern _handler's atomic-write already
#: works around by inlining rather than importing (see below).
_SEALED_RESULT_KEYS = ("held_out", "summary")


def get_fold_count() -> int:
    """Read AUTOMIL_FOLD_COUNT env var (injected by orchestrator). Default 5."""
    return int(os.environ.get("AUTOMIL_FOLD_COUNT", "5"))


def json_safe(value: object) -> object:
    """Return a copy of ``value`` with every non-finite float replaced by ``None``.

    ``json.dumps`` defaults to ``allow_nan=True`` and emits the bare tokens
    ``NaN`` / ``Infinity`` / ``-Infinity``. None of the three is valid JSON
    (RFC 8259), so the file they land in is unreadable to `jq`, serde, the viz
    SSE stream — and to autoMIL itself: ``Runner.collect_result`` parses
    result.json with a ``parse_constant`` hook that rejects those tokens (CR-1a),
    and rewrites the whole node as a crash.

    That guard is right about ``composite`` — a non-finite selection signal would
    rig keep/discard — but a result also carries honestly-unestimable
    *diagnostics*: multi-class sensitivity, a cross-fold CI over zero finite
    folds. Those were killing valid runs. ``null`` is JSON's way of saying "no
    value available", which is precisely what an unestimable metric is, so that
    is what gets written.

    This does not weaken CR-1a. A hand-written ``NaN`` token still fails at the
    parse boundary, and a ``null`` where a number belongs (``composite``, any
    ``metrics`` entry) still fails the schema's ``{"type": "number"}`` — the
    difference is that the failure is now scoped to the field that is actually
    broken instead of condemning the file.

    Numpy scalars are covered, via ``numbers.Real`` rather than ``float``.
    ``np.float64`` subclasses ``float`` but ``np.float32`` does NOT, so an
    ``isinstance(value, float)`` test silently skips a ``np.float32`` NaN and
    lets it reach ``json.dumps``, which then raises ``TypeError`` (numpy scalars
    are not JSON-serializable at all) — inside the SIGTERM handler, that costs
    the whole partial flush. Finite numpy scalars are narrowed to their Python
    equivalent for the same reason: this function's contract is that whatever it
    returns can actually be serialized. ``bool`` is checked first because it is
    an ``Integral`` subclass and must stay JSON ``true``/``false`` rather than
    being narrowed to 1/0. ``numbers`` is stdlib, so this costs the framework no
    numpy dependency.

    Never mutates the input: containers are rebuilt, not edited in place.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Real):
        if not math.isfinite(value):
            return None
        if isinstance(value, float):
            return value
        return int(value) if isinstance(value, numbers.Integral) else float(value)
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON atomically via tempfile + os.replace (CR-04 pattern).

    A SIGKILL (e.g. the daemon's grace-timer expiry) can arrive between a
    plain write_text's open() and close() syscalls, leaving a torn/zero-byte
    result.json on disk. The daemon's ingestion path reads that file first,
    and a partial JSON silently falls through to log-heuristic synthesis,
    discarding every real result. Shared by register_sigterm_flush's handler
    and write_result_json so both get the same crash-safety.

    Non-finite floats are written as ``null`` (see :func:`json_safe`); with them
    gone, ``allow_nan=False`` can never fire and stands as an assertion that no
    invalid JSON token reaches disk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path_str = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as fh:
            fh.write(json.dumps(json_safe(payload), indent=2, allow_nan=False) + "\n")
        os.replace(tmp_path_str, str(path))
    except Exception:
        try:
            os.unlink(tmp_path_str)
        except OSError:
            pass
        raise


def _resolve_sealed_results_dir() -> Path | None:
    """Return AUTOMIL_RESULTS_DIR as an absolute Path, or None if unusable.

    T-09-06: a relative value is rejected rather than resolved against an
    unknown base (same validation register_sigterm_flush's handler applies).
    None means "no sealed location available" -- either the var is unset (a
    manual, non-orchestrated run) or malformed.
    """
    results_dir_env = os.environ.get("AUTOMIL_RESULTS_DIR")
    if not results_dir_env:
        return None
    candidate = Path(results_dir_env)
    return candidate if candidate.is_absolute() else None


def write_result_json(payload: dict, *, worktree_dir: str | Path | None = None) -> None:
    """Split-write a training script's final result across the val-firewall boundary (L-3).

    Training scripts should call this to report their final result instead of
    writing ``result.json`` directly. It writes to up to two places:

      - The FULL payload (``held_out`` / ``summary`` included, if present) goes
        to ``AUTOMIL_RESULTS_DIR`` -- the orchestrator-injected, sealed
        ``archive/<node>/certify/`` directory (see
        ``_orchestrator_daemon._build_subprocess_env``). ``runner.collect_result``
        treats a result.json already present there as authoritative and reads
        it back as-is (see its docstring), so this is the durable, test-bearing
        copy.
      - A STRIPPED payload (``held_out`` and ``summary`` removed) goes to
        ``result.json`` under ``worktree_dir`` (default: ``Path.cwd()``, which
        is the worktree while a training script runs under the orchestrator --
        the daemon launches it with ``cwd=`` the worktree path).

    The sealed copy is written FIRST, deliberately: if the process is killed
    between the two writes, the surviving copy is the more valuable one, and
    ``collect_result`` already prefers the sealed copy when present, so
    recovery is correct regardless of whether the worktree write ever ran.

    Before this existed, a script that wrote result.json straight into the
    worktree left the FULL payload -- test metrics included -- sitting in
    ``.automil_worktrees/<node>/result.json`` for the whole run (worktree
    creation to cleanup), a location with no access control of its own.
    Anything that can read the project directory during search, including the
    coding agent driving it, could read the sealed held_out block straight off
    disk without waiting for ``automil certify``. After this, the worktree
    copy carries at most the same validation-only view already shown in the
    final, agent-facing ``archive/<node>/result.json`` -- nothing an agent
    could not already see.

    If ``AUTOMIL_RESULTS_DIR`` is unset or malformed (a manual, unorchestrated
    run -- no sealed location exists to split into), the FULL payload is
    written straight to the worktree copy instead: there is nowhere sealed to
    put ``held_out``, so keeping it in the one file that exists is strictly
    better than silently dropping it.

    Never mutates ``payload`` -- both copies are freshly built dicts.
    """
    target_worktree = Path(worktree_dir) if worktree_dir is not None else Path.cwd()
    sealed_dir = _resolve_sealed_results_dir()

    if sealed_dir is None:
        _atomic_write_json(target_worktree / "result.json", dict(payload))
        return

    stripped = {k: v for k, v in payload.items() if k not in _SEALED_RESULT_KEYS}
    _atomic_write_json(sealed_dir / "result.json", dict(payload))    # full, sealed -- written first
    _atomic_write_json(target_worktree / "result.json", stripped)    # val-only, agent-visible


def register_sigterm_flush(*, fold_count_env: str = "AUTOMIL_FOLD_COUNT") -> None:
    """Install SIGTERM handler that flushes partial fold results and exits 0.

    Idempotent — calling twice is a no-op (module-level _SIGTERM_REGISTERED guard).
    Handler aggregates fold_*_result.json from CWD, writes result.json, sys.exit(0).

    Call BEFORE creating any DataLoader or threading.Thread. signal.signal()
    raises ValueError if called from a non-main thread.

    sys.exit(0) — NOT sys.exit(130) — returncode 0 lets the daemon distinguish
    graceful flush from process death before flush (D-121). [VERIFIED]
    """
    global _SIGTERM_REGISTERED
    if _SIGTERM_REGISTERED:
        return

    def _handler(signum: int, frame: object) -> None:
        # Lazy import — automil.cells.reconcile lands in Plan 04-05 (same wave 2).
        # Lazy keeps runtime_helpers importable in Wave 1 before reconcile exists.
        from automil.cells.reconcile import aggregate_folds
        n = int(os.environ.get(fold_count_env, "5"))
        # D-02 (REC-01): write to AUTOMIL_RESULTS_DIR (the archive dir set by
        # orchestrator), not Path.cwd() (the worktree). Falls back to cwd only
        # when running outside the orchestrator (e.g. manual run). T-09-06:
        # _resolve_sealed_results_dir validates the env-var path is absolute
        # before use; a relative/malformed value is treated as unset here too.
        target = _resolve_sealed_results_dir() or Path.cwd()
        payload = aggregate_folds(target, n)
        payload["termination_reason"] = "sigterm"   # D-05 (REC-03): annotate reason
        # CR-04 fix: write result.json atomically (tempfile + os.replace) via
        # the shared _atomic_write_json helper. SIGKILL from the daemon's
        # grace-timer expiry can arrive between a plain write_text's open()
        # and close() syscalls, leaving a torn zero-byte result.json.
        # _collect_or_synthesize_result reads this file first; a partial JSON
        # causes a parse error and falls through to log-heuristic synthesis,
        # silently discarding all fold results aggregated above.
        _atomic_write_json(target / "result.json", payload)
        sys.exit(0)  # NOT sys.exit(130) — clean exit signals graceful flush to daemon

    signal.signal(signal.SIGTERM, _handler)
    _SIGTERM_REGISTERED = True
    logger.info("register_sigterm_flush: SIGTERM handler installed")
