"""Backend abstraction (BCK-01..04 / D-51..D-77).

Plan 02-01 ships the public re-export surface (Backend ABC + dataclasses +
BackendError).  Plan 02-02 extends this with BACKENDS registry dict,
``register`` decorator, and ``_clear_backends`` test utility.

Registry shape (D-68):
  BACKENDS: dict[str, type[Backend]] = {}
  populated via @register("local") / @register("mock_slurm") decorators.

D-69: ``mock_slurm`` is NOT auto-imported here — tests import it explicitly
to avoid leaking a test fixture into production config selection.
"""
from __future__ import annotations

import logging

from automil.backends.base import Backend, JobHandle, JobSpec, JobState
from automil.backends.errors import (
    BackendError,
    BackendNotInstalledError,
    SlurmDirectivesIncompleteError,
    RayClusterUnreachableError,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend registry (D-68)
# ---------------------------------------------------------------------------

BACKENDS: dict[str, type[Backend]] = {}


def register(name: str):
    """Class decorator: register a Backend subclass under ``name``.

    Usage::

        @register("local")
        class LocalBackend(Backend): ...

    D-68: Backends are discovered at import time.
    D-69: ``mock_slurm`` is NOT auto-imported here; tests import it explicitly.

    Raises:
        BackendError: if ``cls`` does not subclass ``Backend``, or if
                      ``name`` is already registered.
    """
    def _decorator(cls: type) -> type:
        if not (isinstance(cls, type) and issubclass(cls, Backend)):
            raise BackendError(
                f"{cls.__name__} must subclass Backend to be registered. "
                f"Ensure the class inherits from automil.backends.Backend."
            )
        if name in BACKENDS:
            raise BackendError(
                f"Backend {name!r} is already registered as "
                f"{BACKENDS[name].__name__}. Duplicate registration rejected."
            )
        BACKENDS[name] = cls
        logger.info("Registered backend %r -> %s", name, cls.__name__)
        return cls

    return _decorator


def _clear_backends() -> None:
    """Test-only: clear BACKENDS registry for isolation.

    Never call in production — variant registration is import-time and
    un-registration is not a supported v1 operation.
    """
    BACKENDS.clear()


from automil.backends import local as _local_backend  # noqa: F401  # D-68: auto-register LocalBackend
from automil.backends.local import LocalBackend  # noqa: F401  # re-export for public surface
# mock_slurm NOT auto-imported here — tests import it explicitly (D-69)

# Opt-in distributed backends (Phase 6 / D-153): guarded imports so
# `uv sync` (no extras) never fails. When the extra is missing the
# backend is simply unavailable at runtime; callers attempting to dispatch
# through an unavailable backend get BackendNotInstalledError (raised by the
# config-resolution path, not by import).
try:
    from automil.backends import slurm as _slurm_backend  # noqa: F401  # registers SLURMBackend
except ImportError:
    pass  # [slurm] extra not installed — backend unavailable

try:
    from automil.backends import ray as _ray_backend  # noqa: F401  # registers RayBackend
except ImportError:
    pass  # [ray] extra not installed — backend unavailable

__all__ = [
    # Plan 02-01 surface
    "Backend",
    "JobHandle",
    "JobSpec",
    "JobState",
    "BackendError",
    # Plan 06-02: Phase 6 typed error subclasses (D-178)
    "BackendNotInstalledError",
    "RayClusterUnreachableError",
    "SlurmDirectivesIncompleteError",
    # Plan 02-02 registry surface (D-68)
    "BACKENDS",
    "register",
    "_clear_backends",
    # Plan 02-05: LocalBackend (auto-registered as "local" via D-68)
    "LocalBackend",
]
