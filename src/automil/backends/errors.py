"""Backend error types (BCK-01 / D-68)."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class BackendError(Exception):
    """Raised when backend registration or dispatch fails."""


class BackendNotInstalledError(BackendError):
    """Raised when the selected backend's extra is not installed.

    Carries ``extra_name`` attribute so callers can surface the pip hint.
    """
    def __init__(self, backend_name: str, extra_name: str) -> None:
        self.extra_name = extra_name
        super().__init__(
            f"Backend {backend_name!r} requires the [{extra_name}] extra. "
            f"Install it with: uv sync --extra {extra_name}"
        )


class SlurmDirectivesIncompleteError(BackendError):
    """Raised by automil check when required SLURM directives are missing
    or contain the TODO_FILL_IN sentinel.

    Carries ``missing_keys`` list for structured error reporting.
    """
    def __init__(self, missing_keys: list[str]) -> None:
        self.missing_keys = missing_keys
        super().__init__(
            f"SLURM directives incomplete — missing or TODO-sentinel values "
            f"for required keys: {missing_keys}. "
            f"Edit automil/config.yaml: backend.slurm.directives"
        )


class RayClusterUnreachableError(BackendError):
    """Raised when RAY_ADDRESS is set but the cluster is unreachable AND
    allow_local_fallback is False (config: backend.ray.allow_local_fallback).
    """
    def __init__(self, address: str) -> None:
        self.address = address
        super().__init__(
            f"Ray cluster at {address!r} is unreachable and "
            f"backend.ray.allow_local_fallback is False. "
            f"Check RAY_ADDRESS and cluster health."
        )
