"""result.json names the accelerator the run trained on.

Campaign runs must all train on one GPU type (a full H100 and an H100 MIG
slice disagree by up to 0.045 validation AUC per fold), so every result
carries the device it came from as evidence. The record never changes a
number: it sits beside peak_vram_mb, outside ``metrics`` and ``held_out``.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_experiment.py"


@pytest.fixture(scope="module")
def run_experiment():
    name = "run_experiment_device_record"
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _summary() -> dict:
    return {
        "test": {"auc_roc": {"mean": 0.70}, "balanced_accuracy": {"mean": 0.60}},
        "val": {"auc_roc": {"mean": 0.70}, "balanced_accuracy": {"mean": 0.60}},
        "per_fold_val": [
            {"auc_roc": auc, "balanced_accuracy": 0.60}
            for auc in (0.70, 0.72, 0.68)
        ],
        "per_fold_test": [],
        "n_folds": 3,
    }


def _fake_cuda(monkeypatch, module, *, available: bool) -> None:
    cuda = module.torch.cuda
    monkeypatch.setattr(cuda, "is_available", lambda: available)
    monkeypatch.setattr(cuda, "max_memory_allocated", lambda *a, **k: 0)
    monkeypatch.setattr(cuda, "current_device", lambda: 0)
    monkeypatch.setattr(cuda, "get_device_name", lambda index=0: "NVIDIA H100 80GB HBM3")
    monkeypatch.setattr(
        cuda, "get_device_properties",
        lambda index=0: SimpleNamespace(
            multi_processor_count=132, total_memory=81559 * 1024 * 1024,
        ),
    )
    # The real call loads cuDNN, which a CUDA wheel on a GPU-less runner lacks.
    monkeypatch.setattr(module.torch.backends.cudnn, "version", lambda: 91002)


def test_result_names_the_gpu_it_trained_on(run_experiment, monkeypatch):
    _fake_cuda(monkeypatch, run_experiment, available=True)
    result = run_experiment.summary_to_result_json(_summary(), 10.0)
    device = result["device"]
    assert device["name"] == "NVIDIA H100 80GB HBM3"
    assert device["multiprocessors"] == 132
    assert device["memory_mb"] == 81559
    assert device["torch"] == run_experiment.torch.__version__
    assert device["cudnn"] == 91002
    assert set(device) == {
        "name", "multiprocessors", "memory_mb", "torch", "cuda", "cudnn",
    }


def test_cpu_run_records_no_device(run_experiment, monkeypatch):
    _fake_cuda(monkeypatch, run_experiment, available=False)
    result = run_experiment.summary_to_result_json(_summary(), 10.0)
    assert result["device"] is None


def test_device_record_keeps_the_result_schema_valid(run_experiment, monkeypatch):
    from automil.schemas import validate_result

    _fake_cuda(monkeypatch, run_experiment, available=True)
    result = run_experiment.summary_to_result_json(_summary(), 10.0)
    validate_result(result)
    assert "device" not in result["metrics"]
