#!/usr/bin/env python
"""One-off migration and repair runner for the preprint-130 baselines.

This file is intentionally campaign-local and temporary.  It remains in the
tree until the shared dataset-root results pass the final 130-cell audit, then
is deleted.  Every legacy ``benchmark_5fold`` tree is treated as read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping

from automil.scoring import cross_fold_se
from autobench.campaign import (
    CAMPAIGN_ID,
    content_sha256,
    file_sha256,
    load_manifest,
)
from autobench.pipeline.abmil.config import ABMILConfig
from autobench.pipeline.config import TrainConfig


HISTORICAL_REUSABLE_FRAMEWORKS = frozenset({"nnmil", "dtfd", "titan"})
HISTORICAL_STALE_FRAMEWORKS = frozenset({"clam", "abmil"})
_FOLDS = tuple(range(5))
_PREP_COMPONENTS = ("dataset_csv", "splits", "features", "nnmil", "titan")
_RERUN_FRAMEWORKS = ("clam", "abmil")


class HistoricalBaselineError(ValueError):
    """A historical result cannot be proven reusable for the current cell."""


def _selected_datasets(
    manifest: Mapping[str, Any], datasets: tuple[str, ...] | None,
) -> tuple[str, ...]:
    available = {str(cell["dataset"]) for cell in manifest["cells"]}
    selected = tuple(sorted(set(datasets or available)))
    unknown = set(selected) - available
    if unknown:
        raise HistoricalBaselineError(
            f"dataset selection is outside the manifest: {sorted(unknown)}"
        )
    if not selected:
        raise HistoricalBaselineError("dataset selection must not be empty")
    return selected


def historical_result_dir(legacy_root: Path, cell: Mapping[str, Any]) -> Path:
    """Resolve one manifest cell to its canonical historical result directory."""
    path = (
        legacy_root
        / str(cell["dataset"])
        / "benchmark_5fold"
        / "results"
        / str(cell["framework"])
        / "standard"
        / str(cell["task"])
        / str(cell["encoder"])
        / str(cell["model"])
    )
    if cell.get("task_type") == "survival":
        path /= "nllsurv"
    return path


def canonical_result_dir(phase_root: Path, cell: Mapping[str, Any]) -> Path:
    """Resolve a cell to the current seed-aware dataset-root result path."""
    path = (
        phase_root
        / str(cell["dataset"])
        / "results"
        / str(cell["framework"])
        / "standard"
        / str(cell["task"])
        / str(cell["encoder"])
        / str(cell["model"])
    )
    if cell.get("task_type") == "survival":
        path /= "nllsurv"
    return path / f"s{int(cell['seed'])}"


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise HistoricalBaselineError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise HistoricalBaselineError(f"{label} must be a JSON object: {path}")
    return value


def _number(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise HistoricalBaselineError(f"{label} is not finite")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise HistoricalBaselineError(f"{label} is outside [0, 1]")
    return number


def _equivalent(left: object, right: object) -> bool:
    """Compare legacy JSON recursively while treating paired NaNs as equal."""
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _equivalent(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _equivalent(a, b) for a, b in zip(left, right)
        )
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        a, b = float(left), float(right)
        return (math.isnan(a) and math.isnan(b)) or math.isclose(
            a, b, rel_tol=0.0, abs_tol=1e-12,
        )
    return left == right


def _validate_identity(
    cell: Mapping[str, Any], config: Mapping[str, Any], summary: Mapping[str, Any],
) -> None:
    task = config.get("task")
    model = config.get("model")
    train = config.get("train")
    if not all(isinstance(value, Mapping) for value in (task, model, train)):
        raise HistoricalBaselineError("historical config lacks task/model/train blocks")
    expected_loss = "nllsurv" if cell.get("task_type") == "survival" else None
    checks = {
        "config task": (task.get("name"), cell.get("task")),
        "config task type": (task.get("task_type"), cell.get("task_type")),
        "config encoder": (config.get("encoder_key"), cell.get("encoder")),
        "config model": (model.get("model_type"), cell.get("model")),
        "config framework": (config.get("framework"), cell.get("framework")),
        "config strategy": (config.get("strategy"), "standard"),
        "config folds": (config.get("n_folds"), 5),
        "config seed": (train.get("seed"), cell.get("seed")),
        "config survival loss": (config.get("survival_loss"), expected_loss),
        "summary task": (summary.get("task"), cell.get("task")),
        "summary encoder": (summary.get("encoder"), cell.get("encoder")),
        "summary model": (summary.get("model_type"), cell.get("model")),
        "summary framework": (summary.get("framework"), cell.get("framework")),
        "summary strategy": (summary.get("strategy"), "standard"),
        "summary folds": (summary.get("n_folds"), 5),
        "summary seed": (summary.get("seed"), cell.get("seed")),
        "summary survival loss": (summary.get("survival_loss"), expected_loss),
    }
    for label, (actual, expected) in checks.items():
        if actual != expected:
            raise HistoricalBaselineError(
                f"{label} mismatch: expected {expected!r}, got {actual!r}"
            )


def validate_historical_baseline(
    cell: Mapping[str, Any], source: Path,
) -> dict[str, Any]:
    """Validate an unchanged legacy result and its five-fold evidence."""
    framework = str(cell.get("framework"))
    if framework in HISTORICAL_STALE_FRAMEWORKS:
        raise HistoricalBaselineError(
            f"{framework} native recipe changed after the historical run"
        )
    if framework not in HISTORICAL_REUSABLE_FRAMEWORKS:
        raise HistoricalBaselineError(f"framework {framework!r} has no reuse rule")

    return _validate_result_bundle(cell, source)


def _validate_result_bundle(
    cell: Mapping[str, Any], source: Path,
) -> dict[str, Any]:
    """Validate identity and all five result/summary relationships."""

    config_path = source / "config.json"
    summary_path = source / "summary.json"
    config = _load_object(config_path, "historical config")
    summary = _load_object(summary_path, "historical summary")
    _validate_identity(cell, config, summary)

    observed_fold_dirs = {
        path.name for path in source.glob("fold_*") if path.is_dir()
    }
    expected_fold_dirs = {f"fold_{fold}" for fold in _FOLDS}
    if observed_fold_dirs != expected_fold_dirs:
        raise HistoricalBaselineError(
            f"historical folds must be exactly {sorted(expected_fold_dirs)}"
        )
    per_fold_val = summary.get("per_fold_val")
    per_fold_test = summary.get("per_fold_test")
    if (
        not isinstance(per_fold_val, list)
        or not isinstance(per_fold_test, list)
        or len(per_fold_val) != len(_FOLDS)
        or len(per_fold_test) != len(_FOLDS)
    ):
        raise HistoricalBaselineError("summary does not contain exactly five folds")

    fold_records: list[dict[str, Any]] = []
    source_hashes = {
        "config.json": file_sha256(config_path),
        "summary.json": file_sha256(summary_path),
    }
    primary = (
        ("c_index",)
        if cell.get("task_type") == "survival"
        else ("auc_roc", "balanced_accuracy")
    )
    for fold in _FOLDS:
        metrics_path = source / f"fold_{fold}" / "metrics.json"
        metrics = _load_object(metrics_path, f"fold {fold} metrics")
        if metrics.get("fold") != fold:
            raise HistoricalBaselineError(f"fold {fold} identity mismatch")
        val_metrics = metrics.get("val_metrics")
        test_metrics = metrics.get("test_metrics")
        if not isinstance(val_metrics, Mapping) or not isinstance(test_metrics, Mapping):
            raise HistoricalBaselineError(f"fold {fold} lacks val/test metrics")
        if not _equivalent(val_metrics, per_fold_val[fold]):
            raise HistoricalBaselineError(f"fold {fold} validation differs from summary")
        if not _equivalent(test_metrics, per_fold_test[fold]):
            raise HistoricalBaselineError(f"fold {fold} held-out differs from summary")
        for key in primary:
            _number(val_metrics.get(key), f"fold {fold} val {key}")
            _number(test_metrics.get(key), f"fold {fold} test {key}")
        elapsed = metrics.get("elapsed_seconds", 0)
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or float(elapsed) < 0
        ):
            raise HistoricalBaselineError(f"fold {fold} elapsed_seconds is invalid")
        source_hashes[f"fold_{fold}/metrics.json"] = file_sha256(metrics_path)
        fold_records.append({
            "fold_index": fold,
            "val_metrics": dict(val_metrics),
            "test_metrics": dict(test_metrics),
            "elapsed_seconds": float(elapsed),
        })

    for block_name, records in (
        ("val", per_fold_val), ("test", per_fold_test),
    ):
        aggregate = summary.get(block_name)
        if not isinstance(aggregate, Mapping):
            raise HistoricalBaselineError(f"summary {block_name} block is missing")
        for key in primary:
            values = [
                _number(record.get(key), f"summary {block_name} fold {key}")
                for record in records
            ]
            reported = aggregate.get(key)
            if not isinstance(reported, Mapping):
                raise HistoricalBaselineError(
                    f"summary {block_name} aggregate {key} is missing"
                )
            mean = _number(reported.get("mean"), f"summary {block_name} mean {key}")
            if not math.isclose(
                mean, math.fsum(values) / len(values),
                rel_tol=0.0, abs_tol=1e-12,
            ):
                raise HistoricalBaselineError(
                    f"summary {block_name} mean {key} differs from its folds"
                )

    return {
        "cell_id": cell["cell_id"],
        "source": str(source.resolve()),
        "source_sha256": source_hashes,
        "folds": fold_records,
        "config": config,
        "summary": summary,
    }


def validate_current_baseline(
    cell: Mapping[str, Any], source: Path,
) -> dict[str, Any]:
    """Validate a dataset-root result, including corrected native recipes."""
    validated = _validate_result_bundle(cell, source)
    config = validated["config"]
    summary = validated["summary"]
    framework = str(cell.get("framework"))
    if framework in HISTORICAL_STALE_FRAMEWORKS:
        for label, actual in (
            ("config dataset", config.get("dataset")),
            ("summary dataset", summary.get("dataset")),
        ):
            if actual != cell.get("dataset"):
                raise HistoricalBaselineError(
                    f"{label} mismatch: expected {cell.get('dataset')!r}, "
                    f"got {actual!r}"
                )
    if framework == "clam":
        expected_train = asdict(TrainConfig())
        if config.get("train") != expected_train:
            raise HistoricalBaselineError("fresh CLAM result does not use TrainConfig defaults")
        if config.get("arm") is not None:
            raise HistoricalBaselineError("fresh CLAM result has an unexpected arm block")
        if config.get("train_fields_superseded_by_arm") != []:
            raise HistoricalBaselineError("fresh CLAM result supersedes shared train fields")
    elif framework == "abmil":
        if config.get("arm") != asdict(ABMILConfig()):
            raise HistoricalBaselineError("fresh ABMIL result does not use upstream defaults")
        expected_superseded = [
            "early_stopping", "lr", "max_epochs", "patience", "weight_decay",
        ]
        if config.get("train_fields_superseded_by_arm") != expected_superseded:
            raise HistoricalBaselineError("fresh ABMIL provenance is incomplete")
    elif framework not in HISTORICAL_REUSABLE_FRAMEWORKS:
        raise HistoricalBaselineError(f"unknown framework {framework!r}")

    if framework in HISTORICAL_STALE_FRAMEWORKS:
        fingerprint = _load_object(
            source / "config_fingerprint.json", "fresh result fingerprint",
        )
        payload = fingerprint.get("config")
        if not isinstance(payload, Mapping):
            raise HistoricalBaselineError("fresh result fingerprint has no config")
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()
        if fingerprint.get("digest") != digest:
            raise HistoricalBaselineError("fresh result fingerprint digest mismatch")
    return validated


def _converted_artifacts(validated: Mapping[str, Any]) -> tuple[dict, list[dict]]:
    folds = list(validated["folds"])
    is_survival = "c_index" in folds[0]["val_metrics"]
    validation_folds: list[dict[str, Any]] = []
    sealed_folds: list[dict[str, Any]] = []
    composites: list[float] = []
    for row in folds:
        fold = int(row["fold_index"])
        val = row["val_metrics"]
        test = row["test_metrics"]
        if is_survival:
            val_metrics = {"val_c_index": float(val["c_index"])}
            held_out = {"test_c_index": float(test["c_index"])}
        else:
            val_metrics = {
                "val_auc": float(val["auc_roc"]),
                "val_bacc": float(val["balanced_accuracy"]),
            }
            held_out = {
                "test_auc": float(test["auc_roc"]),
                "test_bacc": float(test["balanced_accuracy"]),
            }
        composite = math.fsum(val_metrics.values()) / len(val_metrics)
        composites.append(composite)
        validation_folds.append({
            "fold_index": fold,
            "metrics": val_metrics,
            "composite": composite,
        })
        sealed_folds.append({
            "fold_index": fold,
            "fold_count": len(_FOLDS),
            "status": "completed",
            "metrics": val_metrics,
            "held_out": held_out,
            "composite": composite,
            "elapsed_seconds": int(float(row["elapsed_seconds"])),
            "peak_vram_mb": 0,
        })
    composite = math.fsum(composites) / len(composites)
    metrics = (
        {"val_c_index": composite}
        if is_survival else {
            "val_auc": math.fsum(
                fold["metrics"]["val_auc"] for fold in validation_folds
            ) / len(validation_folds),
            "val_bacc": math.fsum(
                fold["metrics"]["val_bacc"] for fold in validation_folds
            ) / len(validation_folds),
        }
    )
    public_result = {
        "status": "completed",
        "metrics": metrics,
        "composite": composite,
        "composite_se": cross_fold_se(composites),
        "elapsed_seconds": round(math.fsum(
            float(row["elapsed_seconds"]) for row in folds
        ), 1),
        "peak_vram_mb": 0,
        "n_valid_folds": len(_FOLDS),
        "n_folds": len(_FOLDS),
        "validation_folds": validation_folds,
    }
    return public_result, sealed_folds


def convert_and_register_historical_baseline(
    cell_root: Path, cell: Mapping[str, Any], source: Path,
    *, register: Callable[[Path, Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Convert one validated legacy result and register it without GPU work."""
    if register is None:
        from autobench.campaign_stages import attest_and_register_baseline

        register = attest_and_register_baseline
    validated = validate_historical_baseline(cell, source)
    public_result, sealed_folds = _converted_artifacts(validated)
    temporary = Path(tempfile.mkdtemp(prefix=".baseline-reuse-", dir=cell_root))
    archive = temporary / "archive"
    try:
        certify = archive / "certify"
        certify.mkdir(parents=True)
        (archive / "result.json").write_text(
            json.dumps(public_result, indent=2, sort_keys=True) + "\n"
        )
        for payload in sealed_folds:
            path = certify / f"fold_{payload['fold_index']}_result.json"
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        state = register(cell_root, archive)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return {
        "cell_id": cell["cell_id"],
        "disposition": "registered-reuse",
        "source": validated["source"],
        "source_sha256": validated["source_sha256"],
        "baseline_candidate_sha256": state["baseline"]["candidate_sha256"],
        "baseline_attestation_sha256": state["baseline"]["attestation_sha256"],
    }


def audit_historical_baselines(
    manifest_path: Path, legacy_root: Path,
) -> dict[str, Any]:
    """Classify every manifest cell as reusable or requiring a fresh run."""
    manifest = load_manifest(manifest_path)
    rows: list[dict[str, Any]] = []
    for cell in manifest["cells"]:
        source = historical_result_dir(legacy_root, cell)
        if cell["framework"] in HISTORICAL_STALE_FRAMEWORKS:
            rows.append({
                "cell_id": cell["cell_id"],
                "framework": cell["framework"],
                "disposition": "rerun",
                "reason": "native recipe changed after the historical run",
                "source": str(source.resolve()),
                "source_sha256": None,
            })
            continue
        try:
            validated = validate_historical_baseline(cell, source)
            row = {
                "cell_id": cell["cell_id"],
                "framework": cell["framework"],
                "disposition": "reuse",
                "reason": "unchanged native recipe and exact five-fold evidence",
                "source": validated["source"],
                "source_sha256": validated["source_sha256"],
            }
        except HistoricalBaselineError as exc:
            row = {
                "cell_id": cell["cell_id"],
                "framework": cell["framework"],
                "disposition": "invalid-reuse",
                "reason": str(exc),
                "source": str(source.resolve()),
                "source_sha256": None,
            }
        rows.append(row)
    counts = {
        disposition: sum(row["disposition"] == disposition for row in rows)
        for disposition in ("reuse", "rerun", "invalid-reuse")
    }
    payload = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "manifest_sha256": file_sha256(manifest_path),
        "legacy_root": str(legacy_root.resolve()),
        "counts": counts,
        "cells": rows,
    }
    payload["audit_sha256"] = content_sha256(payload)
    return payload


def import_reusable_baselines(
    manifest_path: Path, runtime_root: Path, legacy_root: Path,
) -> dict[str, Any]:
    """Import every reusable cell and persist a campaign-wide source audit."""
    manifest = load_manifest(manifest_path)
    audit = audit_historical_baselines(manifest_path, legacy_root)
    by_id = {cell["cell_id"]: cell for cell in manifest["cells"]}
    registrations: list[dict[str, Any]] = []
    for row in audit["cells"]:
        if row["disposition"] != "reuse":
            continue
        cell = by_id[row["cell_id"]]
        cell_root = runtime_root / row["cell_id"]
        if not cell_root.is_dir():
            raise HistoricalBaselineError(
                f"materialized campaign cell is missing: {cell_root}"
            )
        registrations.append(convert_and_register_historical_baseline(
            cell_root, cell, Path(row["source"]),
        ))
    result = {
        **{key: value for key, value in audit.items() if key != "audit_sha256"},
        "registrations": registrations,
    }
    result["audit_sha256"] = content_sha256(result)
    target = runtime_root / "baseline_reuse_audit.json"
    fd, temporary = tempfile.mkstemp(
        dir=str(runtime_root), prefix=".baseline-reuse-audit-", suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return result


def _tree_inventory(root: Path) -> dict[str, dict[str, Any]]:
    """Hash every regular file in a result tree and reject special entries."""
    if not root.is_dir() or root.is_symlink():
        raise HistoricalBaselineError(f"result tree is missing or not a directory: {root}")
    inventory: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise HistoricalBaselineError(f"result tree contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise HistoricalBaselineError(f"result tree contains a special entry: {path}")
        inventory[relative] = {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    if not inventory:
        raise HistoricalBaselineError(f"result tree is empty: {root}")
    return inventory


def _inventory_summary(inventory: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "file_count": len(inventory),
        "bytes": sum(int(row["bytes"]) for row in inventory.values()),
        "tree_sha256": content_sha256(inventory),
    }


def ensure_prepared_links(
    phase_root: Path, datasets: list[str] | tuple[str, ...], *, create: bool,
) -> list[dict[str, str]]:
    """Create or verify dataset-root links to immutable legacy prep artifacts."""
    rows: list[dict[str, str]] = []
    for dataset in sorted(set(datasets)):
        dataset_root = phase_root / dataset
        legacy = dataset_root / "benchmark_5fold"
        if not dataset_root.is_dir() or not legacy.is_dir():
            raise HistoricalBaselineError(
                f"dataset or legacy benchmark root is missing: {dataset_root}"
            )
        results = dataset_root / "results"
        if results.is_symlink():
            raise HistoricalBaselineError(
                f"canonical results must be a real directory, not a symlink: {results}"
            )
        for name in _PREP_COMPONENTS:
            source = legacy / name
            destination = dataset_root / name
            if not source.is_dir() or source.is_symlink():
                raise HistoricalBaselineError(f"legacy prep source is invalid: {source}")
            if os.path.lexists(destination):
                if not destination.is_symlink():
                    raise HistoricalBaselineError(
                        f"prepared destination exists but is not the managed link: {destination}"
                    )
                try:
                    same = destination.resolve(strict=True) == source.resolve(strict=True)
                except OSError as exc:
                    raise HistoricalBaselineError(
                        f"prepared link is broken: {destination}"
                    ) from exc
                if not same:
                    raise HistoricalBaselineError(
                        f"prepared link targets the wrong source: {destination}"
                    )
                disposition = "verified"
            elif create:
                temporary = dataset_root / f".{name}.baseline-repair-{uuid.uuid4().hex}"
                try:
                    temporary.symlink_to(
                        Path("benchmark_5fold") / name, target_is_directory=True,
                    )
                    os.replace(temporary, destination)
                finally:
                    if os.path.lexists(temporary):
                        temporary.unlink()
                if destination.resolve(strict=True) != source.resolve(strict=True):
                    raise HistoricalBaselineError(
                        f"new prepared link failed verification: {destination}"
                    )
                disposition = "created"
            else:
                raise HistoricalBaselineError(
                    f"prepared dataset-root link is missing: {destination}"
                )
            rows.append({
                "dataset": dataset,
                "component": name,
                "source": str(source.resolve()),
                "destination": str(destination),
                "disposition": disposition,
            })
    return rows


def _copy_result_tree_atomic(source: Path, destination: Path) -> dict[str, Any]:
    """Copy, hash-verify, then atomically publish one immutable result tree."""
    before = _tree_inventory(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination):
        if destination.is_symlink() or not destination.is_dir():
            raise HistoricalBaselineError(f"result destination is not a real directory: {destination}")
        current = _tree_inventory(destination)
        if current != before:
            raise HistoricalBaselineError(
                f"refusing to overwrite non-identical result destination: {destination}"
            )
        return {"disposition": "already-present", **_inventory_summary(current)}

    staging = destination.parent / (
        f".{destination.name}.baseline-repair-{uuid.uuid4().hex}"
    )
    try:
        shutil.copytree(source, staging, copy_function=shutil.copy2)
        copied = _tree_inventory(staging)
        after = _tree_inventory(source)
        if before != after:
            raise HistoricalBaselineError(f"legacy source changed while copying: {source}")
        if copied != before:
            raise HistoricalBaselineError(f"staged copy differs from legacy source: {source}")
        os.replace(staging, destination)
    finally:
        if os.path.lexists(staging):
            shutil.rmtree(staging)
    published = _tree_inventory(destination)
    if published != before:
        raise HistoricalBaselineError(f"published result differs from legacy source: {destination}")
    return {"disposition": "copied", **_inventory_summary(published)}


def migrate_reusable_results(
    manifest_path: Path, phase_root: Path, *, datasets: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Safely publish selected reusable cells into dataset-root results."""
    preflight = preflight_migration(
        manifest_path, phase_root, datasets=datasets,
    )
    if preflight["counts"]["invalid"]:
        raise HistoricalBaselineError("migration preflight contains invalid cells")
    manifest = load_manifest(manifest_path)
    selected = _selected_datasets(manifest, datasets)
    cells = [
        cell for cell in manifest["cells"] if str(cell["dataset"]) in selected
    ]
    links = ensure_prepared_links(
        phase_root, [str(cell["dataset"]) for cell in cells], create=True,
    )
    rows: list[dict[str, Any]] = []
    for cell in cells:
        if cell["framework"] not in HISTORICAL_REUSABLE_FRAMEWORKS:
            continue
        source = historical_result_dir(phase_root, cell)
        validated = validate_historical_baseline(cell, source)
        destination = canonical_result_dir(phase_root, cell)
        copy = _copy_result_tree_atomic(source, destination)
        validate_current_baseline(cell, destination)
        rows.append({
            "cell_id": cell["cell_id"],
            "dataset": cell["dataset"],
            "framework": cell["framework"],
            "source": validated["source"],
            "destination": str(destination.resolve()),
            **copy,
        })
    expected_reusable = sum(
        cell["framework"] in HISTORICAL_REUSABLE_FRAMEWORKS for cell in cells
    )
    if len(rows) != expected_reusable:
        raise HistoricalBaselineError(
            f"expected exactly {expected_reusable} reusable cells, found {len(rows)}"
        )
    report = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "manifest_sha256": file_sha256(manifest_path),
        "phase_root": str(phase_root.resolve()),
        "datasets": list(selected),
        "legacy_policy": "read-only",
        "prepared_links": links,
        "counts": {
            "reusable": len(rows),
            "copied": sum(row["disposition"] == "copied" for row in rows),
            "already_present": sum(
                row["disposition"] == "already-present" for row in rows
            ),
        },
        "cells": rows,
    }
    report["audit_sha256"] = content_sha256(report)
    for dataset in selected:
        dataset_report = {
            **{key: value for key, value in report.items()
               if key not in {"cells", "prepared_links", "counts", "audit_sha256"}},
            "datasets": [dataset],
            "prepared_links": [
                row for row in links if row["dataset"] == dataset
            ],
            "cells": [
                row for row in rows if row["dataset"] == dataset
            ],
        }
        dataset_report["counts"] = {
            "reusable": len(dataset_report["cells"]),
            "copied": sum(
                row["disposition"] == "copied" for row in dataset_report["cells"]
            ),
            "already_present": sum(
                row["disposition"] == "already-present"
                for row in dataset_report["cells"]
            ),
        }
        dataset_report["audit_sha256"] = content_sha256(dataset_report)
        _write_json_atomic(
            phase_root / dataset / "baseline_repair_migration.json",
            dataset_report,
        )
    return report


def preflight_migration(
    manifest_path: Path, phase_root: Path, *, datasets: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Read-only all-cell conflict, inventory, and capacity check before migration."""
    manifest = load_manifest(manifest_path)
    selected = _selected_datasets(manifest, datasets)
    link_rows: list[dict[str, str]] = []
    for dataset in selected:
        dataset_root = phase_root / dataset
        legacy = dataset_root / "benchmark_5fold"
        if not dataset_root.is_dir() or not legacy.is_dir():
            raise HistoricalBaselineError(
                f"dataset or legacy benchmark root is missing: {dataset_root}"
            )
        if (dataset_root / "results").is_symlink():
            raise HistoricalBaselineError(
                f"canonical results is unexpectedly a symlink: {dataset_root / 'results'}"
            )
        results_root = dataset_root / "results"
        needs_dataset_root_write = not results_root.exists()
        for component in _PREP_COMPONENTS:
            source = legacy / component
            destination = dataset_root / component
            if not source.is_dir() or source.is_symlink():
                raise HistoricalBaselineError(f"legacy prep source is invalid: {source}")
            if not os.path.lexists(destination):
                status = "will-create"
                needs_dataset_root_write = True
            elif destination.is_symlink() and (
                destination.resolve(strict=True) == source.resolve(strict=True)
            ):
                status = "verified"
            else:
                raise HistoricalBaselineError(
                    f"prepared destination conflicts with migration: {destination}"
                )
            link_rows.append({
                "dataset": dataset,
                "component": component,
                "status": status,
            })
        if needs_dataset_root_write and not os.access(
            dataset_root, os.W_OK | os.X_OK,
        ):
            raise HistoricalBaselineError(
                f"dataset root is not writable for canonical publication: {dataset_root}"
            )
        if results_root.is_dir() and not os.access(
            results_root, os.W_OK | os.X_OK,
        ):
            raise HistoricalBaselineError(
                f"canonical results root is not writable: {results_root}"
            )

    rows: list[dict[str, Any]] = []
    bytes_to_copy = 0
    for cell in manifest["cells"]:
        if str(cell["dataset"]) not in selected:
            continue
        if cell["framework"] not in HISTORICAL_REUSABLE_FRAMEWORKS:
            continue
        source = historical_result_dir(phase_root, cell)
        validate_historical_baseline(cell, source)
        source_inventory = _tree_inventory(source)
        summary = _inventory_summary(source_inventory)
        destination = canonical_result_dir(phase_root, cell)
        if not os.path.lexists(destination):
            status = "will-copy"
            bytes_to_copy += int(summary["bytes"])
        elif (
            destination.is_dir()
            and not destination.is_symlink()
            and _tree_inventory(destination) == source_inventory
        ):
            status = "verified-existing"
        else:
            raise HistoricalBaselineError(
                f"canonical result conflicts with legacy source: {destination}"
            )
        rows.append({
            "cell_id": cell["cell_id"],
            "source": str(source.resolve()),
            "destination": str(destination),
            "status": status,
            **summary,
        })
    expected_reusable = sum(
        cell["framework"] in HISTORICAL_REUSABLE_FRAMEWORKS
        for cell in manifest["cells"] if str(cell["dataset"]) in selected
    )
    if len(rows) != expected_reusable:
        raise HistoricalBaselineError(
            f"preflight expected {expected_reusable} reusable cells, found {len(rows)}"
        )
    free_bytes = shutil.disk_usage(phase_root).free
    if bytes_to_copy > free_bytes:
        raise HistoricalBaselineError(
            f"migration needs {bytes_to_copy} bytes but filesystem reports "
            f"only {free_bytes} free"
        )
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "manifest_sha256": file_sha256(manifest_path),
        "phase_root": str(phase_root.resolve()),
        "datasets": list(selected),
        "counts": {
            "reusable": len(rows),
            "will_copy": sum(row["status"] == "will-copy" for row in rows),
            "verified_existing": sum(
                row["status"] == "verified-existing" for row in rows
            ),
            "invalid": 0,
        },
        "bytes_to_copy": bytes_to_copy,
        "filesystem_free_bytes": free_bytes,
        "prepared_links": link_rows,
        "cells": rows,
    }


def rerun_plan(
    manifest_path: Path, phase_root: Path, *, datasets: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Build the stable 60-cell corrected CLAM/ABMIL array plan."""
    manifest = load_manifest(manifest_path)
    selected = _selected_datasets(manifest, datasets)
    rows: list[dict[str, Any]] = []
    for cell in manifest["cells"]:
        if str(cell["dataset"]) not in selected:
            continue
        if cell["framework"] not in HISTORICAL_STALE_FRAMEWORKS:
            continue
        tokens = shlex.split(str(cell["commands"]["baseline"]))
        if tokens[:2] != ["python", "benchmarks/scripts/run_experiment.py"]:
            raise HistoricalBaselineError(
                f"unexpected baseline entrypoint for {cell['cell_id']}"
            )
        if "--benchmark-dir" in tokens or "--benchmark_dir" in tokens:
            raise HistoricalBaselineError(
                f"manifest command already overrides benchmark root: {cell['cell_id']}"
            )
        dataset_root = phase_root / str(cell["dataset"])
        command = [
            *tokens,
            "--benchmark-dir", str(dataset_root),
            "--skip-prep",
            "--gpu", "0",
        ]
        rows.append({
            "array_index": len(rows),
            "cell_id": cell["cell_id"],
            "dataset": cell["dataset"],
            "task_type": cell["task_type"],
            "framework": cell["framework"],
            "destination": str(canonical_result_dir(phase_root, cell)),
            "command": command,
        })
    expected_count = sum(
        cell["framework"] in HISTORICAL_STALE_FRAMEWORKS
        for cell in manifest["cells"] if str(cell["dataset"]) in selected
    )
    if len(rows) != expected_count:
        raise HistoricalBaselineError(
            f"expected {expected_count} reruns, found {len(rows)}"
        )
    regimes = {(row["framework"], row["task_type"]) for row in rows}
    expected = {(arm, task) for arm in _RERUN_FRAMEWORKS
                for task in ("classification", "survival")}
    if regimes != expected:
        raise HistoricalBaselineError(f"rerun regimes differ: {sorted(regimes)}")
    canaries = []
    for regime in sorted(expected):
        canaries.append(next(
            row["array_index"] for row in rows
            if (row["framework"], row["task_type"]) == regime
        ))
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "manifest_sha256": file_sha256(manifest_path),
        "phase_root": str(phase_root.resolve()),
        "datasets": list(selected),
        "count": len(rows),
        "canary_indices": canaries,
        "cells": rows,
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def write_slurm_runner(
    manifest_path: Path, phase_root: Path, repo_root: Path, ops_root: Path,
    *, datasets: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Write the temporary array plan and one-cell SLURM entrypoint."""
    plan = rerun_plan(manifest_path, phase_root, datasets=datasets)
    ops_root.mkdir(parents=True, exist_ok=True)
    logs = ops_root / "logs"
    logs.mkdir(exist_ok=True)
    plan_path = ops_root / "rerun_plan.json"
    _write_json_atomic(plan_path, plan)
    runner_path = ops_root / "rerun_baselines.sbatch"
    script_path = repo_root / "benchmarks/campaigns/preprint_130/repair_baselines.py"
    command_parts = [
        "python", str(script_path), "run-cell",
        "--manifest", str(manifest_path),
        "--phase-root", str(phase_root),
        "--repo-root", str(repo_root),
        "--ops-root", str(ops_root),
    ]
    if datasets:
        command_parts.extend(["--datasets", ",".join(sorted(set(datasets)))])
    command_parts.extend(["--index", "${SLURM_ARRAY_TASK_ID}"])
    command = " ".join(shlex.quote(part) for part in command_parts)
    command = command.replace("'${SLURM_ARRAY_TASK_ID}'", '"${SLURM_ARRAY_TASK_ID}"')
    body = f"""#!/bin/bash
#SBATCH --job-name=baseline_fix
#SBATCH --account=rrg-jma
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=96G
#SBATCH --output={logs}/%A_%a.out
#SBATCH --error={logs}/%A_%a.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
module load cuda/12.2 2>/dev/null || true
cd {shlex.quote(str(repo_root))}
source .venv/bin/activate
set -a
source benchmarks/.env
set +a
{command}
"""
    fd, temporary = tempfile.mkstemp(
        dir=str(ops_root), prefix=".rerun-baselines-", suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(body)
        os.chmod(temporary, 0o750)
        os.replace(temporary, runner_path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
    return {
        "plan": str(plan_path),
        "runner": str(runner_path),
        "count": plan["count"],
        "canary_indices": plan["canary_indices"],
    }


def run_rerun_cell(
    manifest_path: Path, phase_root: Path, repo_root: Path, ops_root: Path,
    index: int, *, datasets: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Execute exactly one stable rerun-plan entry and validate its result."""
    import fcntl

    plan = rerun_plan(manifest_path, phase_root, datasets=datasets)
    if index < 0 or index >= len(plan["cells"]):
        raise HistoricalBaselineError(
            f"array index outside [0, {len(plan['cells'])}): {index}"
        )
    row = plan["cells"][index]
    manifest = load_manifest(manifest_path)
    cell = next(cell for cell in manifest["cells"] if cell["cell_id"] == row["cell_id"])
    ensure_prepared_links(phase_root, [str(cell["dataset"])], create=False)

    lock_dir = ops_root / "locks"
    work_dir = ops_root / "work" / str(cell["cell_id"])
    completion_dir = ops_root / "completed"
    lock_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    completion_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{cell['cell_id']}.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise HistoricalBaselineError(
                f"cell already has a running repair job: {cell['cell_id']}"
            ) from exc
        tokens = list(row["command"])
        command = [
            sys.executable,
            str(repo_root / "benchmarks/scripts/run_experiment.py"),
            *tokens[2:],
        ]
        env = os.environ.copy()
        env.pop("AUTOMIL_RESULTS_DIR", None)
        env.pop("AUTOMIL_NODE_ID", None)
        python_paths = [str(repo_root / "src"), str(repo_root / "benchmarks/src")]
        if env.get("PYTHONPATH"):
            python_paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(python_paths)
        completed = subprocess.run(command, cwd=work_dir, env=env, check=False)
        if completed.returncode != 0:
            raise HistoricalBaselineError(
                f"baseline rerun failed for {cell['cell_id']} with exit "
                f"{completed.returncode}; partial results were preserved"
            )
        destination = canonical_result_dir(phase_root, cell)
        validated = validate_current_baseline(cell, destination)
        inventory = _tree_inventory(destination)
        report = {
            "schema_version": 1,
            "cell_id": cell["cell_id"],
            "array_index": index,
            "destination": str(destination.resolve()),
            "result_sha256": validated["source_sha256"],
            **_inventory_summary(inventory),
        }
        report["audit_sha256"] = content_sha256(report)
        _write_json_atomic(completion_dir / f"{cell['cell_id']}.json", report)
        return report


def audit_canonical_results(
    manifest_path: Path, phase_root: Path,
) -> dict[str, Any]:
    """Validate exact manifest coverage in canonical dataset-root storage."""
    manifest = load_manifest(manifest_path)
    rows: list[dict[str, Any]] = []
    for cell in manifest["cells"]:
        destination = canonical_result_dir(phase_root, cell)
        try:
            validated = validate_current_baseline(cell, destination)
            if cell["framework"] in HISTORICAL_REUSABLE_FRAMEWORKS:
                source = historical_result_dir(phase_root, cell)
                validate_historical_baseline(cell, source)
                if _tree_inventory(source) != _tree_inventory(destination):
                    raise HistoricalBaselineError(
                        "canonical reusable result differs from its legacy source"
                    )
            inventory = _tree_inventory(destination)
            rows.append({
                "cell_id": cell["cell_id"],
                "framework": cell["framework"],
                "status": "complete",
                "destination": validated["source"],
                **_inventory_summary(inventory),
            })
        except HistoricalBaselineError as exc:
            rows.append({
                "cell_id": cell["cell_id"],
                "framework": cell["framework"],
                "status": "pending",
                "destination": str(destination),
                "reason": str(exc),
            })
    counts = {
        "complete": sum(row["status"] == "complete" for row in rows),
        "pending": sum(row["status"] == "pending" for row in rows),
    }
    report = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "manifest_sha256": file_sha256(manifest_path),
        "phase_root": str(phase_root.resolve()),
        "counts": counts,
        "cells": rows,
    }
    report["audit_sha256"] = content_sha256(report)
    return report


def _paths(args: argparse.Namespace) -> tuple[Path, Path]:
    manifest = Path(args.manifest).expanduser().resolve()
    phase_root = Path(args.phase_root).expanduser().resolve()
    if not manifest.is_file():
        raise HistoricalBaselineError(f"manifest is missing: {manifest}")
    if not phase_root.is_dir():
        raise HistoricalBaselineError(f"phase root is missing: {phase_root}")
    return manifest, phase_root


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=(
            "audit", "preflight", "migrate", "write-slurm", "run-cell", "verify",
        ),
    )
    parser.add_argument(
        "--manifest", default=str(Path(__file__).with_name("manifest.json")),
    )
    parser.add_argument("--phase-root", required=True)
    parser.add_argument(
        "--datasets",
        help="Optional comma-separated manifest dataset subset for recoverable batches.",
    )
    parser.add_argument("--repo-root", default=str(Path(__file__).parents[3]))
    parser.add_argument(
        "--ops-root",
        default=str(Path(__file__).with_name("runtime") / "baseline-repair"),
    )
    parser.add_argument("--index", type=int)
    args = parser.parse_args(argv)
    try:
        manifest, phase_root = _paths(args)
        datasets = (
            tuple(part.strip() for part in args.datasets.split(",") if part.strip())
            if args.datasets else None
        )
        if args.action == "audit":
            result = audit_historical_baselines(manifest, phase_root)
        elif args.action == "preflight":
            result = preflight_migration(
                manifest, phase_root, datasets=datasets,
            )
        elif args.action == "migrate":
            result = migrate_reusable_results(
                manifest, phase_root, datasets=datasets,
            )
        elif args.action == "write-slurm":
            result = write_slurm_runner(
                manifest, phase_root, Path(args.repo_root).resolve(),
                Path(args.ops_root).resolve(), datasets=datasets,
            )
        elif args.action == "run-cell":
            if args.index is None:
                raise HistoricalBaselineError("run-cell requires --index")
            result = run_rerun_cell(
                manifest, phase_root, Path(args.repo_root).resolve(),
                Path(args.ops_root).resolve(), args.index, datasets=datasets,
            )
        else:
            result = audit_canonical_results(manifest, phase_root)
    except HistoricalBaselineError as exc:
        parser.exit(2, f"baseline-repair error: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
