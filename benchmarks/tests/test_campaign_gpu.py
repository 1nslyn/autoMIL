"""Contracts for the campaign GPU rule: every training run executes on the
GPU type reproduction_policy.json declares.

A re-run on the same GPU type reproduces a baseline bit for bit; a full H100
and an H100 MIG slice disagree by up to 0.045 validation AUC per fold (fir
jobs 61495558 / 61495560). Every refusal below is exercised by a forged
listing that tries to slip past it.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from autobench.campaign import REPRODUCTION_POLICY_PATH
from autobench.campaign_gpu import (
    CampaignGpuError,
    load_declared_gpu,
    require_declared_gpu,
)

H100 = "NVIDIA H100 80GB HBM3"
DECLARED = {"name": H100, "mig": False}


def _declare(repo_root: Path, gpu=DECLARED) -> None:
    path = repo_root / REPRODUCTION_POLICY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"epsilon": 0.025}
    if gpu is not None:
        payload["gpu"] = gpu
    path.write_text(json.dumps(payload))


def _nvidia_smi(monkeypatch, stdout: str = "", returncode: int = 0,
                missing: bool = False) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        if missing:
            raise FileNotFoundError("nvidia-smi")
        return SimpleNamespace(returncode=returncode, stdout=stdout,
                               stderr="driver not loaded")

    monkeypatch.setattr("autobench.campaign_gpu.subprocess.run", fake_run)
    return calls


def test_committed_policy_declares_a_full_h100():
    repo_root = Path(__file__).resolve().parents[2]
    assert load_declared_gpu(repo_root) == DECLARED


def test_full_h100_passes(tmp_path, monkeypatch):
    _declare(tmp_path)
    calls = _nvidia_smi(
        monkeypatch,
        f"0, {H100}, Disabled\n1, {H100}, Disabled\n",
    )
    require_declared_gpu(tmp_path, [0, 1])
    assert calls[0][0] == "nvidia-smi"


def test_mig_slice_is_refused(tmp_path, monkeypatch):
    """The slice reports the parent's name; only the MIG mode tells it apart."""
    _declare(tmp_path)
    _nvidia_smi(monkeypatch, f"0, {H100}, Enabled\n")
    with pytest.raises(CampaignGpuError, match="MIG"):
        require_declared_gpu(tmp_path, [0])


def test_another_gpu_model_is_refused(tmp_path, monkeypatch):
    _declare(tmp_path)
    _nvidia_smi(monkeypatch, "0, NVIDIA RTX 6000 Ada Generation, [N/A]\n")
    with pytest.raises(CampaignGpuError, match="RTX 6000"):
        require_declared_gpu(tmp_path, [0])


def test_every_requested_index_is_checked(tmp_path, monkeypatch):
    _declare(tmp_path)
    _nvidia_smi(monkeypatch, f"0, {H100}, Disabled\n1, {H100}, Enabled\n")
    with pytest.raises(CampaignGpuError, match="GPU 1"):
        require_declared_gpu(tmp_path, [0, 1])


def test_missing_index_is_refused(tmp_path, monkeypatch):
    _declare(tmp_path)
    _nvidia_smi(monkeypatch, f"0, {H100}, Disabled\n")
    with pytest.raises(CampaignGpuError, match="GPU 3"):
        require_declared_gpu(tmp_path, [0, 3])


def test_failing_nvidia_smi_is_refused(tmp_path, monkeypatch):
    _declare(tmp_path)
    _nvidia_smi(monkeypatch, returncode=9)
    with pytest.raises(CampaignGpuError, match="nvidia-smi"):
        require_declared_gpu(tmp_path, [0])


def test_absent_nvidia_smi_is_refused(tmp_path, monkeypatch):
    _declare(tmp_path)
    _nvidia_smi(monkeypatch, missing=True)
    with pytest.raises(CampaignGpuError, match="nvidia-smi"):
        require_declared_gpu(tmp_path, [0])


def test_empty_request_is_refused(tmp_path, monkeypatch):
    _declare(tmp_path)
    _nvidia_smi(monkeypatch, f"0, {H100}, Disabled\n")
    with pytest.raises(CampaignGpuError, match="no GPU"):
        require_declared_gpu(tmp_path, [])


def test_policy_without_a_gpu_declaration_is_refused(tmp_path, monkeypatch):
    _declare(tmp_path, gpu=None)
    calls = _nvidia_smi(monkeypatch, f"0, {H100}, Disabled\n")
    with pytest.raises(CampaignGpuError, match="declare"):
        require_declared_gpu(tmp_path, [0])
    assert calls == []


def test_absent_policy_is_refused(tmp_path):
    with pytest.raises(CampaignGpuError, match="reproduction_policy.json"):
        load_declared_gpu(tmp_path)


@pytest.mark.parametrize("gpu", [
    {"name": H100},
    {"name": H100, "mig": "false"},
    {"name": "", "mig": False},
    {"name": H100, "mig": False, "count": 2},
    [H100, False],
])
def test_malformed_gpu_declaration_is_refused(tmp_path, gpu):
    _declare(tmp_path, gpu=gpu)
    with pytest.raises(CampaignGpuError, match="gpu"):
        load_declared_gpu(tmp_path)
