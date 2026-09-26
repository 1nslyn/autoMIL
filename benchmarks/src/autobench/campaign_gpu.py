"""Every campaign training run executes on the GPU type the reproduction
policy declares.

A re-run on the same GPU type reproduces a baseline bit for bit, while a full
H100 and an H100 MIG slice disagree by up to 0.045 validation AUC per fold
(fir jobs 61495558 / 61495560, 2026-09-25). A cell compares every attempt
against its baseline, so both must come from the same GPU type. The two
places that hand GPUs to training call ``require_declared_gpu`` before any
work starts: ``campaign_stages._execute_frozen_command`` (baselines and the
reproduction gate) and ``campaign_operate`` before it starts an orchestrator
daemon (discovery and promotion).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

from autobench.campaign import REPRODUCTION_POLICY_PATH

#: A MIG slice reports its parent GPU's name; only this column tells it apart.
NVIDIA_SMI_QUERY = (
    "nvidia-smi", "--query-gpu=index,name,mig.mode.current",
    "--format=csv,noheader",
)


class CampaignGpuError(RuntimeError):
    """The GPU a run was handed is not the campaign's declared GPU type."""


def load_declared_gpu(repo_root: Path) -> dict[str, Any]:
    """The ``gpu`` block of reproduction_policy.json: ``{name, mig}``."""
    path = repo_root / REPRODUCTION_POLICY_PATH
    try:
        policy = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignGpuError(f"cannot read {REPRODUCTION_POLICY_PATH}: {exc}") from exc
    gpu = policy.get("gpu") if isinstance(policy, dict) else None
    if gpu is None:
        raise CampaignGpuError(
            f"{REPRODUCTION_POLICY_PATH} does not declare the campaign gpu; "
            'add "gpu": {"name": <nvidia-smi name>, "mig": false}'
        )
    if (
        not isinstance(gpu, dict)
        or set(gpu) != {"name", "mig"}
        or not isinstance(gpu["name"], str)
        or not gpu["name"]
        or not isinstance(gpu["mig"], bool)
    ):
        raise CampaignGpuError(
            f"{REPRODUCTION_POLICY_PATH} gpu must be exactly "
            '{"name": <non-empty string>, "mig": <true|false>}'
        )
    return {"name": gpu["name"], "mig": gpu["mig"]}


def _listed_gpus() -> dict[int, tuple[str, bool]]:
    """``{index: (name, mig_enabled)}`` for every GPU this host exposes."""
    try:
        completed = subprocess.run(
            list(NVIDIA_SMI_QUERY), capture_output=True, text=True, check=False,
        )
    except OSError as exc:
        raise CampaignGpuError(f"cannot run nvidia-smi: {exc}") from exc
    if completed.returncode != 0:
        raise CampaignGpuError(
            f"nvidia-smi exited {completed.returncode}: "
            f"{(completed.stderr or completed.stdout).strip()}"
        )
    listed: dict[int, tuple[str, bool]] = {}
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3 and parts[0].isdecimal():
            listed[int(parts[0])] = (parts[1], parts[2] == "Enabled")
    return listed


def require_declared_gpu(repo_root: Path, gpu_ids: Sequence[int]) -> None:
    """Refuse unless every requested GPU index is the declared GPU type."""
    declared = load_declared_gpu(repo_root)
    if not gpu_ids:
        raise CampaignGpuError("no GPU was requested for a campaign run")
    listed = _listed_gpus()
    wanted = f"{declared['name']} with MIG {'enabled' if declared['mig'] else 'disabled'}"
    for index in gpu_ids:
        if index not in listed:
            raise CampaignGpuError(
                f"GPU {index} is not present on this host "
                f"(nvidia-smi lists {sorted(listed)})"
            )
        name, mig = listed[index]
        if name != declared["name"] or mig != declared["mig"]:
            observed = f"{name} with MIG {'enabled' if mig else 'disabled'}"
            raise CampaignGpuError(
                f"GPU {index} is {observed}; every campaign run must train on "
                f"{wanted} ({REPRODUCTION_POLICY_PATH}). Request full GPUs "
                "(--gpus=h100:N), never a MIG slice."
            )
