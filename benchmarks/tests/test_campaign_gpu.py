"""Contracts for the campaign GPU rule: every training run executes on the
GPU type reproduction_policy.json declares for the runtime set holding its
cell.

A re-run on the same GPU type reproduces a baseline bit for bit, while two
GPU types disagree: a full H100 and an H100 MIG slice by up to 0.045
validation AUC per fold (fir jobs 61495558 / 61495560), an RTX 6000 Ada and
a full H100 by up to 0.043 (aihub, 2026-09-30). Every refusal below is
exercised by a forged listing that tries to slip past it.
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
RTX = "NVIDIA RTX 6000 Ada Generation"
FULL_H100 = {"name": H100, "mig": False}
RTX_ADA = {"name": RTX, "mig": False}
WORKSTATION_SET = "runtime-aihub-hnsc-a"
DECLARED = {"runtime": FULL_H100, WORKSTATION_SET: RTX_ADA}


def _declare(repo_root: Path, gpu=DECLARED) -> None:
    path = repo_root / REPRODUCTION_POLICY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"epsilon": 0.025}
    if gpu is not None:
        payload["gpu"] = gpu
    path.write_text(json.dumps(payload))


def _cell(repo_root: Path, runtime: str = "runtime") -> Path:
    cell_root = repo_root / "campaign" / runtime / "dataset__task__arm"
    cell_root.mkdir(parents=True)
    return cell_root


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


def test_committed_policy_declares_each_runtime_set():
    """The final grid trains on fir's full H100s; each of the three aihub
    protocol-v5 trial sets trains on aihub's RTX 6000 Ada."""
    repo_root = Path(__file__).resolve().parents[2]
    assert load_declared_gpu(repo_root, "runtime") == FULL_H100
    for name in ("runtime-aihub-hnsc-a", "runtime-aihub-hnsc-b", "runtime-aihub-hnsc-c"):
        assert load_declared_gpu(repo_root, name) == RTX_ADA, name


def test_every_committed_runtime_set_is_declared_and_counted():
    """The final grid (``runtime``) and every set with a committed roster
    declare their GPU, or none of their cells could train. A roster's count
    and cohorts must match its cell list, or the scan refuses the set."""
    repo_root = Path(__file__).resolve().parents[2]
    rosters = sorted(
        (repo_root / "benchmarks/campaigns/preprint_130").glob("*.roster.json")
    )
    assert rosters
    for path in rosters:
        roster = json.loads(path.read_text())
        cell_ids = roster["cell_ids"]
        assert roster["cells"] == len(cell_ids) == len(set(cell_ids)), path.name
        assert {cell.split("__")[0] for cell in cell_ids} == set(roster["cohorts"])
        load_declared_gpu(repo_root, path.name.removesuffix(".roster.json"))
    load_declared_gpu(repo_root, "runtime")


def test_full_h100_passes(tmp_path, monkeypatch):
    _declare(tmp_path)
    calls = _nvidia_smi(
        monkeypatch,
        f"0, {H100}, Disabled\n1, {H100}, Disabled\n",
    )
    require_declared_gpu(tmp_path, cell_root=_cell(tmp_path), gpu_ids=[0, 1])
    assert calls[0][0] == "nvidia-smi"


def test_mig_slice_is_refused(tmp_path, monkeypatch):
    """The slice reports the parent's name; only the MIG mode tells it apart."""
    _declare(tmp_path)
    _nvidia_smi(monkeypatch, f"0, {H100}, Enabled\n")
    with pytest.raises(CampaignGpuError, match="MIG enabled"):
        require_declared_gpu(tmp_path, cell_root=_cell(tmp_path), gpu_ids=[0])


def test_rtx_is_refused_for_an_h100_set(tmp_path, monkeypatch):
    _declare(tmp_path)
    _nvidia_smi(monkeypatch, f"0, {RTX}, [N/A]\n")
    with pytest.raises(CampaignGpuError, match="RTX 6000.*runtime set runtime"):
        require_declared_gpu(tmp_path, cell_root=_cell(tmp_path), gpu_ids=[0])


def test_workstation_set_trains_on_its_declared_rtx(tmp_path, monkeypatch):
    """An RTX reports MIG as [N/A]: it has no MIG mode, so it is not a slice."""
    _declare(tmp_path)
    _nvidia_smi(monkeypatch, "".join(f"{i}, {RTX}, [N/A]\n" for i in range(3)))
    require_declared_gpu(
        tmp_path, cell_root=_cell(tmp_path, WORKSTATION_SET), gpu_ids=[0, 1, 2],
    )


def test_h100_is_refused_for_the_workstation_set(tmp_path, monkeypatch):
    _declare(tmp_path)
    _nvidia_smi(monkeypatch, f"0, {H100}, Disabled\n")
    with pytest.raises(CampaignGpuError, match=f"runtime set {WORKSTATION_SET}"):
        require_declared_gpu(
            tmp_path, cell_root=_cell(tmp_path, WORKSTATION_SET), gpu_ids=[0],
        )


def test_every_requested_index_is_checked(tmp_path, monkeypatch):
    _declare(tmp_path)
    _nvidia_smi(monkeypatch, f"0, {H100}, Disabled\n1, {H100}, Enabled\n")
    with pytest.raises(CampaignGpuError, match="GPU 1"):
        require_declared_gpu(tmp_path, cell_root=_cell(tmp_path), gpu_ids=[0, 1])


def test_missing_index_is_refused(tmp_path, monkeypatch):
    _declare(tmp_path)
    _nvidia_smi(monkeypatch, f"0, {H100}, Disabled\n")
    with pytest.raises(CampaignGpuError, match="GPU 3"):
        require_declared_gpu(tmp_path, cell_root=_cell(tmp_path), gpu_ids=[0, 3])


def test_failing_nvidia_smi_is_refused(tmp_path, monkeypatch):
    _declare(tmp_path)
    _nvidia_smi(monkeypatch, returncode=9)
    with pytest.raises(CampaignGpuError, match="nvidia-smi"):
        require_declared_gpu(tmp_path, cell_root=_cell(tmp_path), gpu_ids=[0])


def test_absent_nvidia_smi_is_refused(tmp_path, monkeypatch):
    _declare(tmp_path)
    _nvidia_smi(monkeypatch, missing=True)
    with pytest.raises(CampaignGpuError, match="nvidia-smi"):
        require_declared_gpu(tmp_path, cell_root=_cell(tmp_path), gpu_ids=[0])


def test_empty_request_is_refused(tmp_path, monkeypatch):
    _declare(tmp_path)
    _nvidia_smi(monkeypatch, f"0, {H100}, Disabled\n")
    with pytest.raises(CampaignGpuError, match="no GPU"):
        require_declared_gpu(tmp_path, cell_root=_cell(tmp_path), gpu_ids=[])


@pytest.mark.parametrize("cell_root", [
    Path("campaign/archive/dataset__task__arm"),
    Path("campaign/runtime/dataset__task__arm/automil"),
], ids=["undeclared-set", "inside-a-cell"])
def test_cell_outside_a_declared_set_is_refused_before_nvidia_smi(
    tmp_path, monkeypatch, cell_root,
):
    """A path inside a cell has the cell directory for a parent, never a
    runtime set: a caller must pass the cell root itself."""
    _declare(tmp_path)
    calls = _nvidia_smi(monkeypatch, f"0, {H100}, Disabled\n")
    with pytest.raises(CampaignGpuError, match="declares no GPU"):
        require_declared_gpu(tmp_path, cell_root=tmp_path / cell_root, gpu_ids=[0])
    assert calls == []


def test_policy_without_a_gpu_declaration_is_refused(tmp_path, monkeypatch):
    _declare(tmp_path, gpu=None)
    calls = _nvidia_smi(monkeypatch, f"0, {H100}, Disabled\n")
    with pytest.raises(CampaignGpuError, match="declare"):
        require_declared_gpu(tmp_path, cell_root=_cell(tmp_path), gpu_ids=[0])
    assert calls == []


def test_absent_policy_is_refused(tmp_path):
    with pytest.raises(CampaignGpuError, match="reproduction_policy.json"):
        load_declared_gpu(tmp_path, "runtime")


@pytest.mark.parametrize("gpu", [
    FULL_H100,
    [FULL_H100],
    {},
    {"runtime": {"name": H100}},
    {"runtime": {"name": H100, "mig": "false"}},
    {"runtime": {"name": "", "mig": False}},
    {"runtime": {"name": H100, "mig": False, "count": 2}},
    {"runtime": [H100, False]},
    {"runtime": FULL_H100, WORKSTATION_SET: {"name": RTX}},
])
def test_malformed_gpu_declaration_is_refused(tmp_path, gpu):
    """The whole map is checked: a typo in another set's entry fails every
    set, and the old single-type form is refused outright."""
    _declare(tmp_path, gpu=gpu)
    with pytest.raises(CampaignGpuError, match="gpu"):
        load_declared_gpu(tmp_path, "runtime")
