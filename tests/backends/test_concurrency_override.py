"""Host-local per-GPU concurrency for the daemon (AUTOMIL_MAX_CONCURRENT_PER_GPU).

The frozen project config carries ``orchestrator.max_concurrent_per_gpu`` as
a default; the host that runs the daemon may hold more (or fewer) attempts
per GPU than the config's author assumed, so the launcher sets the cap for
the allocation it actually got, the way AUTOMIL_VISIBLE_GPUS sets the GPUs.
"""
from __future__ import annotations

import pytest

from automil.backends import _orchestrator_daemon as daemon_module
from automil.backends._orchestrator_daemon import (
    ExperimentOrchestrator,
    max_concurrent_per_gpu_override,
)


def _orchestrator(tmp_path, monkeypatch):
    automil_dir = tmp_path / "automil"
    automil_dir.mkdir()
    (automil_dir / "config.yaml").write_text(
        "run:\n"
        "  script: train.py\n"
        "orchestrator:\n"
        "  max_concurrent_per_gpu: 4\n"
    )
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(daemon_module, "query_gpus", lambda: [])
    monkeypatch.setattr(daemon_module, "query_rocm_gpus", lambda: [])
    return ExperimentOrchestrator(project_root=tmp_path, automil_dir=automil_dir)


def test_absent_override_means_the_config_value(monkeypatch, tmp_path):
    monkeypatch.delenv("AUTOMIL_MAX_CONCURRENT_PER_GPU", raising=False)
    assert max_concurrent_per_gpu_override() is None
    assert _orchestrator(tmp_path, monkeypatch).max_per_gpu == 4
    monkeypatch.setenv("AUTOMIL_MAX_CONCURRENT_PER_GPU", "   ")
    assert max_concurrent_per_gpu_override() is None


def test_override_replaces_the_config_value(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOMIL_MAX_CONCURRENT_PER_GPU", " 8 ")
    assert max_concurrent_per_gpu_override() == 8
    assert _orchestrator(tmp_path, monkeypatch).max_per_gpu == 8


def test_override_survives_the_per_tick_config_reload(monkeypatch, tmp_path):
    # tick() hot-reloads orchestrator.* from config.yaml; the host's cap must
    # keep winning over the frozen config's 4 on every tick, not just at start.
    monkeypatch.setenv("AUTOMIL_MAX_CONCURRENT_PER_GPU", "8")
    orch = _orchestrator(tmp_path, monkeypatch)
    orch._reload_orchestrator_config()
    assert orch.max_per_gpu == 8


def test_without_override_the_reload_follows_the_config(monkeypatch, tmp_path):
    monkeypatch.delenv("AUTOMIL_MAX_CONCURRENT_PER_GPU", raising=False)
    orch = _orchestrator(tmp_path, monkeypatch)
    (tmp_path / "automil" / "config.yaml").write_text(
        "run:\n  script: train.py\norchestrator:\n  max_concurrent_per_gpu: 6\n"
    )
    orch._reload_orchestrator_config()
    assert orch.max_per_gpu == 6


def test_malformed_override_raises_instead_of_falling_back(monkeypatch):
    for bad in ("x", "0", "-1", "2.5", "8,8"):
        monkeypatch.setenv("AUTOMIL_MAX_CONCURRENT_PER_GPU", bad)
        with pytest.raises(ValueError, match="AUTOMIL_MAX_CONCURRENT_PER_GPU"):
            max_concurrent_per_gpu_override()
