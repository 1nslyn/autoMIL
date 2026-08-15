"""DEPRECATED: this module is now a re-export shim.

The orchestrator code lives at automil.backends._orchestrator_daemon as of
Phase 2 (D-60). This shim preserves all `from automil.orchestrator import X`
call sites — both public (ExperimentOrchestrator, NVIDIA_SMI_PATH) and the
private PID/starttime helpers used by tests.

New code should import from automil.backends instead.

Reload transparency note: when this shim is reloaded via importlib.reload(),
it also reloads automil.backends._orchestrator_daemon so that module-level
resolution code (e.g. shutil.which for NVIDIA_SMI_PATH) re-runs. This
preserves the behaviour of tests that patch shutil.which and then reload
automil.orchestrator expecting fresh path resolution.
"""
# DEPRECATED: This module is a re-export shim (Phase 2 / D-60).
# Import from automil.backends instead:
#   ExperimentOrchestrator  -> automil.backends._orchestrator_daemon.ExperimentOrchestrator
# Will be removed in v2.0. See automil.compat for migration table.
import importlib as _importlib
import sys as _sys
import warnings as _warnings

# Reload the underlying daemon module so that module-level resolution code
# (e.g. shutil.which("nvidia-smi") -> NVIDIA_SMI_PATH) re-runs when this shim
# is reloaded. Without this, importlib.reload(automil.orchestrator) would
# re-execute only the shim's import statements while _orchestrator_daemon stays
# cached in sys.modules with its old values.
_daemon_name = "automil.backends._orchestrator_daemon"
if _daemon_name in _sys.modules:
    _importlib.reload(_sys.modules[_daemon_name])

from automil.backends._orchestrator_daemon import (  # noqa: F401, F403
    ExperimentOrchestrator,
    NVIDIA_SMI_PATH,
    _SYSTEM_ENV_WHITELIST_LITERAL,
    _SYSTEM_ENV_WHITELIST_PREFIX,
    _is_pid_alive_with_starttime,
    _read_proc_starttime,
    _write_pid_file,
    _load_pid_file,
)
from automil.backends.pidfile import (  # noqa: F401 — test-facing legacy alias
    parse_starttime_from_stat_line as _parse_starttime_from_stat_line,
)
from automil.backends._orchestrator_daemon import *  # noqa: F401, F403


def __getattr__(name: str):
    # WR-01 fix (Phase 2 review): short-circuit Python-internal dunder probes
    # (__path__, __bases__, __test__, __wrapped__, etc.) BEFORE issuing a
    # DeprecationWarning. The import machinery and pytest collection probe
    # these on every module access; warning on each one floods the test
    # output with 14+ spurious "automil.orchestrator.__path__ moved..."
    # messages. Real callers that hit __getattr__ for a renamed name still
    # see the deprecation warning.
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    _warnings.warn(
        f"automil.orchestrator.{name} moved to automil.backends._orchestrator_daemon "
        f"in Phase 2 (D-60). Update imports by 2027-01.",
        DeprecationWarning,
        stacklevel=2,
    )
    from automil.backends import _orchestrator_daemon as _mod  # noqa: F401
    return getattr(_mod, name)
