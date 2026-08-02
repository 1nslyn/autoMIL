"""Immutable campaign manifests and isolated per-cell runtime materialization.

The preprint campaign must not derive its command from one source and its
budget identity from another.  This module makes one checked-in manifest the
source for both, then materializes one independent ``automil/`` state root per
cell.  It contains no scheduler or ranking policy; stage transitions live in
``campaign_stages.py``.
"""
from __future__ import annotations

import copy
import hashlib
import json
import shlex
from pathlib import Path
from typing import Any, Mapping

import yaml

from automil.cells.state import make_cell_id, normalize_mil_model

SCHEMA_VERSION = 2
CAMPAIGN_ID = "automil-preprint-130-v2"
DATASETS = (
    "tcga_luad",
    "tcga_lgg",
    "cptac_gbm",
    "cptac_pdac",
    "tcga_hnsc",
)
ENCODERS = ("uni_v2", "virchow2", "hoptimus1")
TILE_ARMS = (
    ("clam", "clam_models"),
    ("nnmil", "nnmil_models"),
    ("abmil", "abmil_models"),
    ("dtfd", "dtfd_models"),
)
STAGE_FOLDS = {
    "discovery": (0, 1, 2),
    "promotion": (3, 4),
}
CERTIFICATION_FOLDS = (0, 1, 2, 3, 4)
DISCOVERY_ATTEMPTS = 60
PROMOTION_CANDIDATES = 10
FOLD_TRAININGS_PER_CELL = (
    DISCOVERY_ATTEMPTS * len(STAGE_FOLDS["discovery"])
    + PROMOTION_CANDIDATES * len(STAGE_FOLDS["promotion"])
)
PROTOCOL = {
    "seed": 42,
    "split_folds": 5,
    "discovery_attempts": DISCOVERY_ATTEMPTS,
    "promotion_candidates": PROMOTION_CANDIDATES,
    "frozen_winners": 1,
    "stage_folds": {key: list(value) for key, value in STAGE_FOLDS.items()},
    "winner_selection": {
        "metric_source": "validation",
        "aggregation": "mean",
        "folds": list(CERTIFICATION_FOLDS),
    },
    "certification": {
        "mode": "unseal-existing-held-out",
        "folds": list(CERTIFICATION_FOLDS),
        "retrain": False,
    },
    "fold_trainings_per_cell": FOLD_TRAININGS_PER_CELL,
}


class CampaignManifestError(ValueError):
    """A campaign artifact is malformed or has drifted from its lock."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")


def content_sha256(value: object) -> str:
    """Hash a JSON-compatible value independently of whitespace/key order."""
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset_config_path(repo_root: Path, dataset: str) -> Path:
    source = "cptac" if dataset.startswith("cptac_") else "tcga"
    return repo_root / "benchmarks" / "datasets" / source / f"{dataset}.yaml"


def _policy_template_path(repo_root: Path, dataset: str) -> Path:
    return (
        repo_root / "benchmarks" / "experiments" / dataset
        / "automil" / "config.yaml"
    )


def _run_command(cell: Mapping[str, Any], stage: str) -> str:
    if stage not in STAGE_FOLDS:
        raise CampaignManifestError(f"unknown campaign stage {stage!r}")
    tokens = [
        "python", "benchmarks/scripts/run_experiment.py",
        "--dataset", str(cell["dataset"]),
        "--task", str(cell["task"]),
        "--encoder", str(cell["encoder"]),
        "--model", str(cell["model"]),
        "--framework", str(cell["framework"]),
        "--strategy", "standard",
        "--seed", str(PROTOCOL["seed"]),
        "--n_folds", str(PROTOCOL["split_folds"]),
        "--folds", ",".join(str(i) for i in STAGE_FOLDS[stage]),
    ]
    if cell["survival_loss"] is not None:
        tokens.extend(["--survival_loss", str(cell["survival_loss"])])
    tokens.append("--no_wandb")
    return shlex.join(tokens)


def _cell_record(
    *,
    dataset: str,
    task: str,
    task_type: str,
    encoder: str,
    framework: str,
    model: str,
    survival_loss: str | None,
    dataset_config: str,
    dataset_config_sha256: str,
    policy_template: str,
    policy_template_sha256: str,
) -> dict[str, Any]:
    suffix = f"__{survival_loss}" if survival_loss else ""
    experiment_id = (
        f"{dataset}__{framework}__standard__{task}__{encoder}__{model}"
        f"__s{PROTOCOL['seed']}{suffix}"
    )
    normalized_model = normalize_mil_model(model)
    cell = {
        "cell_id": experiment_id,
        "dataset": dataset,
        "task": task,
        "task_type": task_type,
        "encoder": encoder,
        "framework": framework,
        "model": model,
        "survival_loss": survival_loss,
        "regime": "slide" if framework == "titan" else "tile",
        "strategy": "standard",
        "seed": PROTOCOL["seed"],
        "dataset_config": dataset_config,
        "dataset_config_sha256": dataset_config_sha256,
        "policy_template": policy_template,
        "policy_template_sha256": policy_template_sha256,
        "budget_identity": {
            "dataset": dataset,
            "task": task,
            "encoder": encoder,
            "mil_model": normalized_model,
            "cell_id": make_cell_id(dataset, encoder, normalized_model, task),
        },
    }
    # There is deliberately no ``final`` training command.  The frozen winner
    # already owns folds 0-2 from discovery and folds 3-4 from promotion; final
    # reporting unseals only that candidate's existing five-fold held-out data.
    cell["commands"] = {
        stage: _run_command(cell, stage) for stage in STAGE_FOLDS
    }
    cell["cell_sha256"] = content_sha256(cell)
    return cell


def build_preprint_manifest(repo_root: Path) -> dict[str, Any]:
    """Build the exact 130-cell manifest from the five pinned dataset YAMLs."""
    repo_root = repo_root.resolve()
    cells: list[dict[str, Any]] = []
    sources: dict[str, str] = {}
    policy_sources: dict[str, str] = {}
    for dataset in DATASETS:
        config_path = _dataset_config_path(repo_root, dataset)
        policy_path = _policy_template_path(repo_root, dataset)
        raw = yaml.safe_load(config_path.read_text()) or {}
        tasks = raw.get("tasks") or {}
        classification = [
            name for name, spec in tasks.items()
            if (spec or {}).get("task_type", "classification") != "survival"
        ]
        survival = [
            name for name, spec in tasks.items()
            if (spec or {}).get("task_type", "classification") == "survival"
        ]
        if len(classification) != 1 or survival != ["os"]:
            raise CampaignManifestError(
                f"{dataset}: expected one classification task plus os, got "
                f"classification={classification}, survival={survival}"
            )
        losses = list((tasks["os"] or {}).get("survival_losses") or [])
        if losses != ["nllsurv"]:
            raise CampaignManifestError(
                f"{dataset}: preprint survival_losses must be exactly ['nllsurv'], "
                f"got {losses}"
            )
        encoder_dims = ((raw.get("encoders") or {}).get("dims") or {})
        if not set(ENCODERS).issubset(encoder_dims):
            raise CampaignManifestError(
                f"{dataset}: missing roster encoder(s) "
                f"{sorted(set(ENCODERS) - set(encoder_dims))}"
            )
        config_rel = config_path.relative_to(repo_root).as_posix()
        config_hash = file_sha256(config_path)
        sources[config_rel] = config_hash
        policy_rel = policy_path.relative_to(repo_root).as_posix()
        policy_hash = file_sha256(policy_path)
        policy_sources[policy_rel] = policy_hash
        task_pairs = ((classification[0], "classification", None),
                      ("os", "survival", "nllsurv"))
        for task, task_type, loss in task_pairs:
            for framework, roster_key in TILE_ARMS:
                models = list(raw.get(roster_key) or [])
                if len(models) != 1:
                    raise CampaignManifestError(
                        f"{dataset}: {roster_key} must pin exactly one model, got {models}"
                    )
                for encoder in ENCODERS:
                    cells.append(_cell_record(
                        dataset=dataset, task=task, task_type=task_type,
                        encoder=encoder, framework=framework, model=models[0],
                        survival_loss=loss, dataset_config=config_rel,
                        dataset_config_sha256=config_hash,
                        policy_template=policy_rel,
                        policy_template_sha256=policy_hash,
                    ))
            cells.append(_cell_record(
                dataset=dataset, task=task, task_type=task_type,
                encoder="titan", framework="titan", model="titan",
                survival_loss=loss, dataset_config=config_rel,
                dataset_config_sha256=config_hash,
                policy_template=policy_rel,
                policy_template_sha256=policy_hash,
            ))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "protocol": PROTOCOL,
        "dataset_sources": sources,
        "policy_sources": policy_sources,
        "cells": cells,
    }
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Fail closed on roster, identity, hash, or command drift."""
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise CampaignManifestError("unsupported campaign manifest schema")
    if manifest.get("campaign_id") != CAMPAIGN_ID:
        raise CampaignManifestError("unexpected campaign_id")
    if manifest.get("protocol") != PROTOCOL:
        raise CampaignManifestError("campaign protocol differs from the frozen contract")
    cells = manifest.get("cells")
    if not isinstance(cells, list) or len(cells) != 130:
        raise CampaignManifestError(f"campaign must contain exactly 130 cells, got {len(cells or [])}")
    ids: set[str] = set()
    budgets: set[str] = set()
    dataset_sources = manifest.get("dataset_sources")
    policy_sources = manifest.get("policy_sources")
    if not isinstance(dataset_sources, dict) or not isinstance(policy_sources, dict):
        raise CampaignManifestError("campaign source locks must be objects")
    per_dataset: dict[str, int] = {}
    per_task_type: dict[str, int] = {}
    for raw in cells:
        if not isinstance(raw, dict):
            raise CampaignManifestError("every campaign cell must be an object")
        cell = dict(raw)
        recorded_hash = cell.pop("cell_sha256", None)
        if recorded_hash != content_sha256(cell):
            raise CampaignManifestError(f"cell hash mismatch for {cell.get('cell_id')}")
        cell_id = str(cell["cell_id"])
        budget_id = str((cell.get("budget_identity") or {})["cell_id"])
        if cell_id in ids or budget_id in budgets:
            raise CampaignManifestError(f"duplicate cell or budget identity: {cell_id}")
        ids.add(cell_id)
        budgets.add(budget_id)
        dataset_config = cell.get("dataset_config")
        policy_template = cell.get("policy_template")
        if dataset_sources.get(dataset_config) != cell.get("dataset_config_sha256"):
            raise CampaignManifestError(f"dataset source lock mismatch for {cell_id}")
        if policy_sources.get(policy_template) != cell.get("policy_template_sha256"):
            raise CampaignManifestError(f"policy source lock mismatch for {cell_id}")
        per_dataset[cell["dataset"]] = per_dataset.get(cell["dataset"], 0) + 1
        per_task_type[cell["task_type"]] = per_task_type.get(cell["task_type"], 0) + 1
        expected_commands = {
            stage: _run_command(cell, stage) for stage in STAGE_FOLDS
        }
        if cell.get("commands") != expected_commands:
            raise CampaignManifestError(f"command drift for {cell_id}")
    if per_dataset != {dataset: 26 for dataset in DATASETS}:
        raise CampaignManifestError(f"per-dataset census mismatch: {per_dataset}")
    if per_task_type != {"classification": 65, "survival": 65}:
        raise CampaignManifestError(f"task-axis census mismatch: {per_task_type}")


def write_manifest(manifest: Mapping[str, Any], path: Path) -> str:
    """Write a deterministic manifest plus adjacent SHA-256 lock."""
    validate_manifest(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.write_text(payload)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n"
    )
    return digest


def load_manifest(path: Path, *, verify_lock: bool = True) -> dict[str, Any]:
    """Load, schema-check, and optionally verify the adjacent byte lock."""
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignManifestError(f"cannot read campaign manifest {path}: {exc}") from exc
    validate_manifest(manifest)
    if verify_lock:
        lock_path = path.with_suffix(path.suffix + ".sha256")
        try:
            expected = lock_path.read_text().split()[0]
        except (OSError, IndexError) as exc:
            raise CampaignManifestError(f"cannot read manifest lock {lock_path}") from exc
        actual = file_sha256(path)
        if actual != expected:
            raise CampaignManifestError(
                f"manifest byte hash mismatch ({actual} != {expected})"
            )
    return manifest


def _task_block(cell: Mapping[str, Any], dataset_raw: Mapping[str, Any]) -> dict[str, Any]:
    source = dict((dataset_raw.get("tasks") or {})[cell["task"]])
    if cell["task_type"] == "survival":
        return {
            "name": cell["task"], "type": "survival",
            "event_column": source["event_col"],
            "time_column": source["time_col"],
            "survival_loss": "nllsurv", "nll_bins": source.get("nll_bins", 4),
        }
    return {
        "name": cell["task"], "type": "classification",
        "num_classes": source["n_classes"], "label_column": source["label_col"],
    }


def materialize_discovery_cells(
    manifest_path: Path,
    output_root: Path,
    repo_root: Path,
) -> list[Path]:
    """Create 130 isolated discovery roots from the immutable manifest.

    Each root has its own graph/plan/learnings/orchestrator namespace.  The
    generated config's run command, budget identity, fold subset, and source
    hashes all come from the same cell record.
    """
    manifest = load_manifest(manifest_path)
    manifest_hash = file_sha256(manifest_path)
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    try:
        output_root.relative_to(repo_root)
    except ValueError as exc:
        raise CampaignManifestError("campaign output_root must live inside the git repo") from exc
    written: list[Path] = []
    for cell in manifest["cells"]:
        adir = output_root / cell["cell_id"] / "automil"
        adir.mkdir(parents=True, exist_ok=True)
        template_path = repo_root / cell["policy_template"]
        if not template_path.exists():
            raise CampaignManifestError(f"missing cohort template {template_path}")
        if file_sha256(template_path) != cell["policy_template_sha256"]:
            raise CampaignManifestError(
                f"policy template drift for {cell['dataset']}; regenerate the manifest"
            )
        config = copy.deepcopy(yaml.safe_load(template_path.read_text()) or {})
        dataset_path = repo_root / cell["dataset_config"]
        if file_sha256(dataset_path) != cell["dataset_config_sha256"]:
            raise CampaignManifestError(
                f"dataset config drift for {cell['dataset']}; regenerate the manifest"
            )
        dataset_raw = yaml.safe_load(dataset_path.read_text()) or {}
        config["project"] = {
            "name": cell["dataset"],
            "description": f"{CAMPAIGN_ID}: {cell['cell_id']}",
        }
        config["task"] = _task_block(cell, dataset_raw)
        config.setdefault("data", {})["num_folds"] = PROTOCOL["split_folds"]
        config["data"]["seed"] = PROTOCOL["seed"]
        config.setdefault("encoders", {})["primary"] = cell["encoder"]
        if cell["task_type"] == "survival":
            config["metrics"] = {
                "primary": "val_c_index", "composite_formula": "val_c_index",
                "track": ["val_c_index"],
            }
        else:
            config["metrics"] = {
                "primary": "composite",
                "composite_formula": "(val_auc + val_bacc) / 2",
                "track": ["val_auc", "val_bacc"],
            }
        adir_rel = adir.relative_to(repo_root).as_posix()
        config["files"] = {
            "editable": [f"{adir_rel}/variants/_policies/*.py"],
        }
        config["run"] = {
            "script": None,
            "command": cell["commands"]["discovery"],
            "mil_model": cell["model"],
        }
        config.setdefault("cap", {})["eval_budget"] = PROTOCOL["discovery_attempts"]
        config["training"] = {"fold_count": len(STAGE_FOLDS["discovery"])}
        config["campaign"] = {
            "campaign_id": CAMPAIGN_ID,
            "manifest": manifest_path.relative_to(repo_root).as_posix(),
            "manifest_sha256": manifest_hash,
            "cell_id": cell["cell_id"],
            "cell_sha256": cell["cell_sha256"],
            "budget_cell_id": cell["budget_identity"]["cell_id"],
            "stage": "discovery",
        }
        (adir / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
        )
        (adir / "campaign_cell.json").write_text(
            json.dumps(cell, indent=2, sort_keys=True) + "\n"
        )
        (adir / ".gitignore").write_text(
            "graph.json\nresults.tsv\nresult.json\norchestrator/\ncells/\n"
            ".automil_active\n.automil_worktrees/\n*.log\n*.pid\n"
        )
        (adir / "plan.md").write_text(
            f"# Discovery plan — {cell['cell_id']}\n\nNo proposals queued yet.\n"
        )
        (adir / "learnings.md").write_text(
            f"# Cell-local learnings — {cell['cell_id']}\n"
        )
        policy_dir = adir / "variants" / "_policies"
        policy_dir.mkdir(parents=True, exist_ok=True)
        (policy_dir / ".gitkeep").touch()
        from autobench.campaign_stages import initialize_stage_state

        initialize_stage_state(
            adir.parent,
            cell=cell,
            manifest_sha256=manifest_hash,
        )
        written.append(adir)
    return written
