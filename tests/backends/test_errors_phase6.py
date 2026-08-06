"""Phase 6 D-178 error type tests."""
from __future__ import annotations

import pytest

from automil.backends.errors import (
    BackendError,
    BackendNotInstalledError,
    SlurmDirectivesIncompleteError,
    RayClusterUnreachableError,
)


def test_backend_not_installed_error_carries_extra_name():
    exc = BackendNotInstalledError("slurm", "slurm")
    assert exc.extra_name == "slurm"
    assert "uv sync --extra slurm" in str(exc)
    assert isinstance(exc, BackendError)


def test_slurm_directives_incomplete_carries_missing_keys():
    exc = SlurmDirectivesIncompleteError(["partition", "account"])
    assert exc.missing_keys == ["partition", "account"]
    assert "partition" in str(exc)
    assert isinstance(exc, BackendError)


def test_ray_cluster_unreachable_carries_address():
    exc = RayClusterUnreachableError("ray://head:10001")
    assert exc.address == "ray://head:10001"
    assert "ray://head:10001" in str(exc)
    assert isinstance(exc, BackendError)
