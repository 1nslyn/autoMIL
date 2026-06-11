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
import os
import signal
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_SIGTERM_REGISTERED: bool = False  # module-level idempotent guard


def get_fold_count() -> int:
    """Read AUTOMIL_FOLD_COUNT env var (injected by orchestrator). Default 5."""
    return int(os.environ.get("AUTOMIL_FOLD_COUNT", "5"))


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
        # when running outside the orchestrator (e.g. manual run).
        # T-09-06: validate the env-var path is absolute before use; relative
        # values are rejected and fall back to cwd for safety.
        results_dir_env = os.environ.get("AUTOMIL_RESULTS_DIR")
        if results_dir_env:
            target = Path(results_dir_env)
            if not target.is_absolute():
                target = Path.cwd()  # safe fallback for relative/malformed env value
        else:
            target = Path.cwd()
        payload = aggregate_folds(target, n)
        payload["termination_reason"] = "sigterm"   # D-05 (REC-03): annotate reason
        (target / "result.json").write_text(json.dumps(payload, indent=2))
        sys.exit(0)  # NOT sys.exit(130) — clean exit signals graceful flush to daemon

    signal.signal(signal.SIGTERM, _handler)
    _SIGTERM_REGISTERED = True
    logger.info("register_sigterm_flush: SIGTERM handler installed")
