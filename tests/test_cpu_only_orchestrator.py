"""Typed local execution slots must preserve truthful device provenance."""
from __future__ import annotations

import json

from automil.backends import _orchestrator_daemon as daemon_module
from automil.orchestrator import ExperimentOrchestrator


class _DetectedGpu:
    index = 0
    free_gb = 64.0
    total_mb = 65536
    free_mb = 65536
    utilization = 0


def _orchestrator(
    tmp_path, monkeypatch, *, accelerator: str, gpu_count: int, detected=(),
):
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
        "  min_vram_gb: 64\n"
    )
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(daemon_module, "query_gpus", lambda: list(detected))
    return ExperimentOrchestrator(project_root=tmp_path, automil_dir=automil_dir)


def test_cpu_only_config_gets_one_logical_execution_slot(tmp_path, monkeypatch):
    orch = _orchestrator(
        tmp_path, monkeypatch, accelerator="cpu", gpu_count=0,
    )

    assert orch._cpu_only is True
    assert orch._find_best_gpu(needed_gb=1000.0) == 0
    assert orch._pre_launch_check(gpu_id=0, needed_gb=1000.0) is True


def test_rocm_config_gets_one_slot_per_declared_device(tmp_path, monkeypatch):
    orch = _orchestrator(
        tmp_path, monkeypatch, accelerator="rocm", gpu_count=2,
    )

    assert orch._cpu_only is False
    assert sorted(orch.gpu_allocations) == [0, 1]
    assert orch._find_best_gpu(needed_gb=1.0) == 0
    assert orch._pre_launch_check(gpu_id=1, needed_gb=1.0) is True


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


def test_inconsistent_cpu_config_fails_closed(tmp_path, monkeypatch):
    orch = _orchestrator(
        tmp_path, monkeypatch, accelerator="cpu", gpu_count=1,
        detected=(_DetectedGpu(),),
    )

    assert orch.gpu_allocations == {}
    assert orch._find_best_gpu(needed_gb=0.0) is None


def test_unknown_accelerator_fails_closed(tmp_path, monkeypatch):
    orch = _orchestrator(
        tmp_path, monkeypatch, accelerator="mps", gpu_count=1,
        detected=(_DetectedGpu(),),
    )

    assert orch.gpu_allocations == {}
    assert orch._find_best_gpu(needed_gb=0.0) is None


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
    assert env["HIP_VISIBLE_DEVICES"] == ""
    assert env["ROCR_VISIBLE_DEVICES"] == ""
    assert env["AUTOMIL_GPU"] == ""
    assert env["AUTOMIL_ACCELERATOR"] == "cpu"


def test_rocm_subprocess_masks_one_physical_device_and_blocks_spoofing(
    tmp_path, monkeypatch,
):
    orch = _orchestrator(
        tmp_path, monkeypatch, accelerator="rocm", gpu_count=2,
    )

    env = orch._build_subprocess_env(
        gpu_id=1,
        node_id="rocm-node",
        archive=tmp_path / "archive",
        spec={
            "description": "rocm",
            "env": {
                "CUDA_VISIBLE_DEVICES": "99",
                "HIP_VISIBLE_DEVICES": "99",
                "ROCR_VISIBLE_DEVICES": "99",
                "AUTOMIL_GPU": "99",
                "AUTOMIL_ACCELERATOR": "cpu",
            },
        },
    )

    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["HIP_VISIBLE_DEVICES"] == "1"
    assert env["ROCR_VISIBLE_DEVICES"] == "1"
    assert env["AUTOMIL_GPU"] == "0"
    assert env["AUTOMIL_ACCELERATOR"] == "rocm"


def test_cpu_launch_intent_does_not_invent_a_gpu(tmp_path, monkeypatch):
    orch = _orchestrator(
        tmp_path, monkeypatch, accelerator="cpu", gpu_count=0,
    )

    path = orch._write_launch_intent(
        {"id": "cpu-node"}, gpu_id=0, worktree=tmp_path / "wt",
    )
    metadata = json.loads(path.read_text())["metadata"]

    assert metadata["accelerator"] == "cpu"
    assert metadata["gpu"] is None


def test_cpu_running_state_has_a_typed_execution_slot(tmp_path, monkeypatch):
    orch = _orchestrator(
        tmp_path, monkeypatch, accelerator="cpu", gpu_count=0,
    )
    orch.gpu_allocations[0] = ["cpu-node"]

    orch._save_state()
    state = json.loads(orch.gpu_state_file.read_text())

    assert state["gpus"] == {}
    assert state["execution_slots"] == {
        "cpu:0": {
            "accelerator": "cpu",
            "device_index": None,
            "running": ["cpu-node"],
            "capacity": 1,
        }
    }
