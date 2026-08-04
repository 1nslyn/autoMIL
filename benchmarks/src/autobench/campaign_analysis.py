"""Fail-closed publication artifact for the frozen 130-cell campaign."""
from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from autobench.campaign import (
    ANALYSIS_PLAN_PATH,
    CAMPAIGN_ID,
    CERTIFICATION_FOLDS,
    TILE_ARMS,
    content_sha256,
    file_sha256,
    load_manifest,
)
from autobench.campaign_stages import (
    CAMPAIGN_CERTIFICATION_FILE,
    CAMPAIGN_CELL_COUNT,
    SELECTION_FREEZE_FILE,
)

PUBLICATION_REPORT_FILE = "publication_report.json"


class CampaignAnalysisError(ValueError):
    """The frozen campaign cannot support its predeclared publication report."""


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignAnalysisError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CampaignAnalysisError(f"{label} must be a JSON object")
    return payload


def _file_hash(path: Path, label: str) -> str:
    try:
        return file_sha256(path)
    except OSError as exc:
        raise CampaignAnalysisError(f"cannot hash {label} {path}: {exc}") from exc


def _finite(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise CampaignAnalysisError(f"{label} must be finite")
    return float(value)


def _safe_relative(root: Path, raw: object, label: str) -> Path:
    if not isinstance(raw, str):
        raise CampaignAnalysisError(f"{label} path must be a string")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise CampaignAnalysisError(f"{label} path escapes the campaign root")
    path = (root / Path(*relative.parts)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise CampaignAnalysisError(f"{label} path escapes the campaign root") from exc
    return path


def _metric_mean(folds: list[dict[str, Any]], key: str, label: str) -> float:
    return math.fsum(
        _finite(row["held_out"].get(key), f"{label}.{key}") for row in folds
    ) / len(folds)


def _ordered_folds(raw: object, label: str) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or len(raw) != len(CERTIFICATION_FOLDS):
        raise CampaignAnalysisError(f"{label} must contain exactly five folds")
    by_fold: dict[int, dict[str, Any]] = {}
    for row in raw:
        if not isinstance(row, dict) or not isinstance(row.get("held_out"), dict):
            raise CampaignAnalysisError(f"{label} contains a malformed fold")
        fold = row.get("fold_index")
        if isinstance(fold, bool) or not isinstance(fold, int) or fold in by_fold:
            raise CampaignAnalysisError(f"{label} fold identity is invalid")
        by_fold[fold] = row
    if set(by_fold) != set(CERTIFICATION_FOLDS):
        raise CampaignAnalysisError(f"{label} fold roster differs from 0..4")
    return [by_fold[fold] for fold in CERTIFICATION_FOLDS]


def _primary_values(
    bundle: Mapping[str, Any], task_type: str, cell_id: str,
) -> tuple[list[float], list[float]]:
    winner_folds = _ordered_folds(bundle.get("held_out_folds"), f"{cell_id}.winner")
    baseline_folds = _ordered_folds(
        bundle.get("baseline_held_out_folds"), f"{cell_id}.baseline",
    )
    required = (
        ("test_auc", "test_bacc")
        if task_type == "classification"
        else ("test_c_index",)
    )
    values: list[list[float]] = [[], []]
    for output, folds, label in (
        (values[0], baseline_folds, "baseline"),
        (values[1], winner_folds, "winner"),
    ):
        for row in folds:
            metrics = row["held_out"]
            missing = set(required) - set(metrics)
            if missing:
                raise CampaignAnalysisError(
                    f"{cell_id}.{label} lacks {sorted(missing)}"
                )
            primary = (
                math.fsum(_finite(metrics[key], f"{cell_id}.{label}.{key}")
                          for key in required) / len(required)
            )
            output.append(primary)

    for aggregate_key, folds, aggregate in (
        ("baseline_held_out", baseline_folds, bundle.get("baseline_held_out")),
        ("held_out", winner_folds, bundle.get("held_out")),
    ):
        if not isinstance(aggregate, dict):
            raise CampaignAnalysisError(f"{cell_id}.{aggregate_key} is malformed")
        for key in required:
            observed = _finite(aggregate.get(key), f"{cell_id}.{aggregate_key}.{key}")
            expected = _metric_mean(folds, key, f"{cell_id}.{aggregate_key}")
            if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
                raise CampaignAnalysisError(
                    f"{cell_id}.{aggregate_key}.{key} disagrees with its folds"
                )
    return values[0], values[1]


def _average_ranks_desc(values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    ranks: dict[str, float] = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average = ((start + 1) + end) / 2
        for arm, _ in ordered[start:end]:
            ranks[arm] = average
        start = end
    return ranks


def _kendall_tau_b(x: Mapping[str, float], y: Mapping[str, float]) -> float | None:
    keys = sorted(x)
    concordant = discordant = ties_x = ties_y = 0
    for left_index, left in enumerate(keys):
        for right in keys[left_index + 1:]:
            dx = (x[left] > x[right]) - (x[left] < x[right])
            dy = (y[left] > y[right]) - (y[left] < y[right])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_x)
        * (concordant + discordant + ties_y)
    )
    if denominator == 0:
        return None
    return (concordant - discordant) / denominator


def _summary(values: Iterable[float]) -> dict[str, Any]:
    observed = [float(value) for value in values]
    if not observed:
        raise CampaignAnalysisError("cannot summarize an empty estimand")
    tolerance = 1e-12
    return {
        "n": len(observed),
        "mean": math.fsum(observed) / len(observed),
        "median": statistics.median(observed),
        "min": min(observed),
        "max": max(observed),
        "positive": sum(value > tolerance for value in observed),
        "zero": sum(abs(value) <= tolerance for value in observed),
        "negative": sum(value < -tolerance for value in observed),
    }


def _grouped_lift(cells: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for cell in cells:
        grouped[str(cell[key])].append(cell["primary_lift"])
    return {name: _summary(values) for name, values in sorted(grouped.items())}


def _resource_summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    statuses: dict[str, int] = defaultdict(int)
    numeric = (
        "input_tokens", "output_tokens", "cached_input_tokens", "cost_usd",
    )
    observed: dict[str, list[float]] = {key: [] for key in numeric}
    for cell in cells:
        usage = cell["agent_usage"]
        statuses[str(usage["status"])] += 1
        for key in numeric:
            value = usage.get(key)
            if value is not None:
                observed[key].append(float(value))
    return {
        "usage_status": dict(sorted(statuses.items())),
        **{
            key: {
                "reported_cells": len(values),
                "total": math.fsum(values) if values else None,
            }
            for key, values in observed.items()
        },
    }


def _ranking_blocks(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expected_arms = {framework for framework, _ in TILE_ARMS}
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        if cell["regime"] == "tile":
            grouped[(
                cell["dataset"], cell["task"], cell["task_type"], cell["encoder"],
            )].append(cell)
    blocks: list[dict[str, Any]] = []
    for (dataset, task, task_type, encoder), rows in sorted(grouped.items()):
        if len(rows) != len(expected_arms) or {row["framework"] for row in rows} != expected_arms:
            raise CampaignAnalysisError(
                f"incomplete tile ranking block {dataset}/{task}/{encoder}"
            )
        baseline = {row["framework"]: row["baseline_primary"] for row in rows}
        automil = {row["framework"]: row["winner_primary"] for row in rows}
        baseline_ranks = _average_ranks_desc(baseline)
        automil_ranks = _average_ranks_desc(automil)
        baseline_top = sorted(
            arm for arm, value in baseline.items() if value == max(baseline.values())
        )
        automil_top = sorted(
            arm for arm, value in automil.items() if value == max(automil.values())
        )
        blocks.append({
            "dataset": dataset,
            "task": task,
            "task_type": task_type,
            "encoder": encoder,
            "baseline_primary": baseline,
            "winner_primary": automil,
            "baseline_rank": baseline_ranks,
            "automil_rank": automil_ranks,
            "rank_shift": {
                arm: automil_ranks[arm] - baseline_ranks[arm]
                for arm in sorted(expected_arms)
            },
            "kendall_tau_b": _kendall_tau_b(baseline, automil),
            "baseline_top_arms": baseline_top,
            "automil_top_arms": automil_top,
            "top_arm_set_changed": baseline_top != automil_top,
        })
    if len(blocks) != 30:
        raise CampaignAnalysisError(f"expected 30 tile ranking blocks, got {len(blocks)}")
    return blocks


def _ranking_summary(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for task_type in ("classification", "survival"):
        selected = [row for row in blocks if row["task_type"] == task_type]
        taus = [row["kendall_tau_b"] for row in selected
                if row["kendall_tau_b"] is not None]
        output[task_type] = {
            "blocks": len(selected),
            "top_arm_set_changed": sum(row["top_arm_set_changed"] for row in selected),
            "top_arm_set_changed_fraction": (
                sum(row["top_arm_set_changed"] for row in selected) / len(selected)
            ),
            "mean_kendall_tau_b": (
                math.fsum(taus) / len(taus) if taus else None
            ),
        }
    return output


def build_publication_report(
    *, runtime_root: Path, manifest_path: Path, repo_root: Path,
) -> dict[str, Any]:
    """Validate the complete certification census and derive locked estimands."""
    runtime_root = runtime_root.resolve()
    manifest_path = manifest_path.resolve()
    repo_root = repo_root.resolve()
    manifest = load_manifest(manifest_path)
    plan_path = repo_root / ANALYSIS_PLAN_PATH
    if _file_hash(plan_path, "analysis plan") != manifest["analysis_plan"]["sha256"]:
        raise CampaignAnalysisError("analysis plan differs from the manifest lock")
    plan = _read_json(plan_path, "analysis plan")
    if (
        plan.get("campaign_id") != CAMPAIGN_ID
        or plan.get("status") != "frozen-before-held-out-certification"
    ):
        raise CampaignAnalysisError("analysis plan is not frozen for this campaign")

    index_path = runtime_root / CAMPAIGN_CERTIFICATION_FILE
    index = _read_json(index_path, "campaign certification index")
    recorded_index_hash = index.get("certification_sha256")
    if (
        index.get("campaign_id") != CAMPAIGN_ID
        or index.get("manifest_sha256") != _file_hash(manifest_path, "manifest")
        or index.get("cell_count") != CAMPAIGN_CELL_COUNT
        or recorded_index_hash != content_sha256({
            key: value for key, value in index.items()
            if key != "certification_sha256"
        })
    ):
        raise CampaignAnalysisError("campaign certification index integrity mismatch")
    raw_entries = index.get("cells")
    if not isinstance(raw_entries, list):
        raise CampaignAnalysisError("campaign certification roster is malformed")
    entries = {
        entry.get("cell_id"): entry for entry in raw_entries
        if isinstance(entry, dict) and isinstance(entry.get("cell_id"), str)
    }
    manifest_cells = {cell["cell_id"]: cell for cell in manifest["cells"]}
    if len(entries) != CAMPAIGN_CELL_COUNT or set(entries) != set(manifest_cells):
        raise CampaignAnalysisError("campaign certification roster is incomplete")
    freeze = _read_json(
        runtime_root / SELECTION_FREEZE_FILE, "campaign selection freeze",
    )
    freeze_hash = freeze.get("freeze_sha256")
    if (
        freeze_hash != index.get("selection_freeze_sha256")
        or freeze_hash != content_sha256({
            key: value for key, value in freeze.items() if key != "freeze_sha256"
        })
        or freeze.get("cell_count") != CAMPAIGN_CELL_COUNT
    ):
        raise CampaignAnalysisError("campaign selection freeze integrity mismatch")
    freeze_entries = {
        row.get("cell_id"): row for row in freeze.get("cells", [])
        if isinstance(row, dict) and isinstance(row.get("cell_id"), str)
    }
    if set(freeze_entries) != set(manifest_cells):
        raise CampaignAnalysisError("campaign selection freeze roster is incomplete")

    cells: list[dict[str, Any]] = []
    for cell_id in sorted(manifest_cells):
        cell = manifest_cells[cell_id]
        entry = entries[cell_id]
        freeze_entry = freeze_entries[cell_id]
        agent_usage = freeze_entry.get("agent_usage")
        if not isinstance(agent_usage, dict):
            raise CampaignAnalysisError(f"{cell_id}: agent usage is missing")
        bundle_path = _safe_relative(runtime_root, entry.get("bundle"), cell_id)
        if _file_hash(bundle_path, f"{cell_id} certification bundle") != entry.get("file_sha256"):
            raise CampaignAnalysisError(f"{cell_id}: certification file hash mismatch")
        bundle = _read_json(bundle_path, f"{cell_id} certification bundle")
        recorded_bundle_hash = bundle.get("bundle_sha256")
        if (
            bundle.get("schema_version") != 2
            or bundle.get("campaign_id") != CAMPAIGN_ID
            or bundle.get("cell_id") != cell_id
            or bundle.get("selection_freeze_sha256")
            != index.get("selection_freeze_sha256")
            or recorded_bundle_hash != entry.get("bundle_sha256")
            or recorded_bundle_hash != content_sha256({
                key: value for key, value in bundle.items()
                if key != "bundle_sha256"
            })
        ):
            raise CampaignAnalysisError(f"{cell_id}: certification bundle mismatch")
        baseline_folds, winner_folds = _primary_values(
            bundle, cell["task_type"], cell_id,
        )
        baseline_primary = math.fsum(baseline_folds) / len(baseline_folds)
        winner_primary = math.fsum(winner_folds) / len(winner_folds)
        cells.append({
            "cell_id": cell_id,
            "dataset": cell["dataset"],
            "task": cell["task"],
            "task_type": cell["task_type"],
            "encoder": cell["encoder"],
            "framework": cell["framework"],
            "model": cell["model"],
            "regime": cell["regime"],
            "winner_kind": (bundle.get("winner") or {}).get("kind"),
            "baseline_primary_folds": baseline_folds,
            "winner_primary_folds": winner_folds,
            "paired_primary_deltas": [
                winner - baseline
                for baseline, winner in zip(baseline_folds, winner_folds, strict=True)
            ],
            "baseline_primary": baseline_primary,
            "winner_primary": winner_primary,
            "primary_lift": winner_primary - baseline_primary,
            "bundle_sha256": recorded_bundle_hash,
            "agent_usage": agent_usage,
        })

    blocks = _ranking_blocks(cells)
    report: dict[str, Any] = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "manifest_sha256": _file_hash(manifest_path, "manifest"),
        "analysis_plan_sha256": _file_hash(plan_path, "analysis plan"),
        "certification_sha256": recorded_index_hash,
        "selection_freeze_sha256": index["selection_freeze_sha256"],
        "cell_count": len(cells),
        "cells": cells,
        "tile_ranking_blocks": blocks,
        "summaries": {
            "all_cells": _summary(cell["primary_lift"] for cell in cells),
            "by_task_type": _grouped_lift(cells, "task_type"),
            "by_regime": _grouped_lift(cells, "regime"),
            "by_dataset": _grouped_lift(cells, "dataset"),
            "by_framework": _grouped_lift(cells, "framework"),
            "tile_ranking_response": _ranking_summary(blocks),
            "agent_resources": _resource_summary(cells),
            "titan": _summary(
                cell["primary_lift"] for cell in cells if cell["regime"] == "slide"
            ),
        },
        "source_certified_at": index.get("certified_at"),
    }
    report["report_sha256"] = content_sha256(report)
    return report


def write_publication_report(
    *, runtime_root: Path, manifest_path: Path, repo_root: Path,
) -> dict[str, Any]:
    """Write the deterministic report only after the full census validates."""
    report = build_publication_report(
        runtime_root=runtime_root,
        manifest_path=manifest_path,
        repo_root=repo_root,
    )
    target = runtime_root.resolve() / PUBLICATION_REPORT_FILE
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)
    return report
