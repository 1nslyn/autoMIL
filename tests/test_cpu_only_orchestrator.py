"""CPU-only local scheduling must work without inventing a physical GPU."""
from __future__ import annotations

from automil.backends import _orchestrator_daemon as daemon_module
from automil.orchestrator import ExperimentOrchestrator


def _orchestrator(tmp_path, monkeypatch, *, accelerator: str, gpu_count: int):
    automil_dir = tmp_path / "automil"
    automil_dir.mkdir()
    (automil_dir / "config.yaml").write_text(
        "run:\n"
        "  script: train.py\n"
        "orchestrator:\n"
        "  max_concurrent_per_gpu: 1\n"
        "hardware:\n"
        f"  accelerator: {accelerator}\n"
        f"  gpu_count: {gpu_count}\n"
    )
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(daemon_module, "query_gpus", lambda: [])
    return ExperimentOrchestrator(project_root=tmp_path, automil_dir=automil_dir)


def test_cpu_only_config_gets_one_logical_execution_slot(tmp_path, monkeypatch):
    orch = _orchestrator(
        tmp_path, monkeypatch, accelerator="cpu", gpu_count=0,
    )

    assert orch._cpu_only is True
    assert orch._find_best_gpu(needed_gb=1000.0) == 0
    assert orch._pre_launch_check(gpu_id=0, needed_gb=1000.0) is True


def test_cpu_slot_respects_local_concurrency_limit(tmp_path, monkeypatch):
    orch = _orchestrator(
        tmp_path, monkeypatch, accelerator="cpu", gpu_count=0,
    )
    orch.gpu_allocations[0] = ["already-running"]

    assert orch._find_best_gpu(needed_gb=0.0) is None


def test_missing_gpu_does_not_fall_back_for_gpu_config(tmp_path, monkeypatch):
    orch = _orchestrator(
        tmp_path, monkeypatch, accelerator="cuda", gpu_count=1,
    )

    assert orch._cpu_only is False
    assert orch._find_best_gpu(needed_gb=0.0) is None
    assert orch._pre_launch_check(gpu_id=0, needed_gb=0.0) is False


def test_cpu_subprocess_hides_cuda_devices(tmp_path, monkeypatch):
    orch = _orchestrator(
        tmp_path, monkeypatch, accelerator="cpu", gpu_count=0,
    )

    env = orch._build_subprocess_env(
        gpu_id=0,
        node_id="cpu-node",
        archive=tmp_path / "archive",
        spec={"description": "cpu-only", "env": {}},
    )

    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["AUTOMIL_GPU"] == "0"

