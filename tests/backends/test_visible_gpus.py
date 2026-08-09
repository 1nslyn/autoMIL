"""Host-local GPU partitioning for concurrent orchestrators (AUTOMIL_VISIBLE_GPUS)."""
from __future__ import annotations

import pytest

from automil.backends._orchestrator_daemon import (
    GPUInfo,
    _filter_visible,
    visible_gpu_ids,
)


def _gpus(count: int) -> list[GPUInfo]:
    return [
        GPUInfo(index=index, total_mb=81920, free_mb=81920, utilization=0)
        for index in range(count)
    ]


def test_absent_partition_means_every_gpu(monkeypatch):
    monkeypatch.delenv("AUTOMIL_VISIBLE_GPUS", raising=False)
    assert visible_gpu_ids() is None
    assert _filter_visible(_gpus(4)) == _gpus(4)
    monkeypatch.setenv("AUTOMIL_VISIBLE_GPUS", "   ")
    assert visible_gpu_ids() is None


def test_partition_restricts_to_declared_physical_indexes(monkeypatch):
    monkeypatch.setenv("AUTOMIL_VISIBLE_GPUS", "1, 3")
    assert visible_gpu_ids() == frozenset({1, 3})
    assert [gpu.index for gpu in _filter_visible(_gpus(4))] == [1, 3]


def test_malformed_partition_raises_instead_of_scheduling_everywhere(
    monkeypatch,
):
    for bad in ("1,x", "0;1", "-1", "0,,2"):
        monkeypatch.setenv("AUTOMIL_VISIBLE_GPUS", bad)
        with pytest.raises(ValueError, match="AUTOMIL_VISIBLE_GPUS"):
            visible_gpu_ids()
