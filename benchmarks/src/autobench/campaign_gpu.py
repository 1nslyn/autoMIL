"""Every campaign training run executes on the GPU type the reproduction
policy declares for the runtime set holding its cell.

A re-run on the same GPU type reproduces a baseline bit for bit, while two
GPU types disagree: a full H100 and an H100 MIG slice by up to 0.045
validation AUC per fold (fir jobs 61495558 / 61495560, 2026-09-25), an RTX
6000 Ada and a full H100 by up to 0.043 (aihub, 2026-09-30). A cell compares
every attempt against its baseline, so both must come from the same GPU type.
reproduction_policy.json therefore maps each runtime set (the directory the
cells sit in) to one GPU type. The two places that hand GPUs to training
call ``require_declared_gpu`` before any work starts:
``campaign_stages._execute_frozen_command`` (baselines and the reproduction
gate) and ``campaign_operate`` before it starts an orchestrator daemon
(discovery and promotion).
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


def _is_gpu_type(gpu: object) -> bool:
    return (
        isinstance(gpu, dict)
        and set(gpu) == {"name", "mig"}
        and isinstance(gpu["name"], str)
        and bool(gpu["name"])
        and isinstance(gpu["mig"], bool)
    )


def load_declared_gpu(repo_root: Path, runtime: str) -> dict[str, Any]:
    """The GPU type reproduction_policy.json declares for one runtime set:
    ``{name, mig}``. Every entry of the map is checked on every load."""
    path = repo_root / REPRODUCTION_POLICY_PATH
    try:
        policy = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignGpuError(f"cannot read {REPRODUCTION_POLICY_PATH}: {exc}") from exc
    declared = policy.get("gpu") if isinstance(policy, dict) else None
    if not isinstance(declared, dict) or not declared:
        raise CampaignGpuError(
            f"{REPRODUCTION_POLICY_PATH} does not declare the GPU of each "
            'runtime set; add "gpu": {"<runtime set>": {"name": <nvidia-smi '
            'name>, "mig": false}}'
        )
    for name, gpu in declared.items():
        if not _is_gpu_type(gpu):
            raise CampaignGpuError(
                f"{REPRODUCTION_POLICY_PATH} gpu[{name!r}] must be exactly "
                '{"name": <non-empty string>, "mig": <true|false>}'
            )
    if runtime not in declared:
        raise CampaignGpuError(
            f"{REPRODUCTION_POLICY_PATH} declares no GPU for runtime set "
            f"{runtime!r} (declared: {', '.join(sorted(declared))})"
        )
    return {"name": declared[runtime]["name"], "mig": declared[runtime]["mig"]}


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


def _describe(name: str, mig: bool) -> str:
    return f"{name} with MIG {'enabled' if mig else 'disabled'}"


def require_declared_gpu(
    repo_root: Path, *, cell_root: Path, gpu_ids: Sequence[int],
) -> None:
    """Refuse unless every requested GPU index is the type declared for the
    runtime set holding ``cell_root`` (the cell's parent directory)."""
    runtime = Path(cell_root).resolve().parent.name
    declared = load_declared_gpu(repo_root, runtime)
    if not gpu_ids:
        raise CampaignGpuError("no GPU was requested for a campaign run")
    listed = _listed_gpus()
    for index in gpu_ids:
        if index not in listed:
            raise CampaignGpuError(
                f"GPU {index} is not present on this host "
                f"(nvidia-smi lists {sorted(listed)})"
            )
        name, mig = listed[index]
        if name != declared["name"] or mig != declared["mig"]:
            raise CampaignGpuError(
                f"GPU {index} is {_describe(name, mig)}; every run of runtime "
                f"set {runtime} must train on "
                f"{_describe(declared['name'], declared['mig'])} "
                f"({REPRODUCTION_POLICY_PATH})"
            )
