"""Fail-closed publication artifact for the frozen 130-cell campaign."""
from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from autobench.campaign import (
    AGENT_PROTOCOL_FILE,
    ANALYSIS_PLAN_PATH,
    ATTEMPT_OUTCOME_CLASSES,
    CAMPAIGN_ID,
    CERTIFICATION_FOLDS,
    DISCOVERY_ATTEMPTS,
    HELD_OUT_SCHEMA_BY_FAMILY,
    PROTOCOL_VERSION,
    TILE_ARMS,
    content_sha256,
    file_sha256,
    load_manifest,
    validate_agent_protocol,
)
from autobench.campaign_stages import (
    CAMPAIGN_CERTIFICATION_FILE,
    CAMPAIGN_CELL_COUNT,
    CampaignStageError,
    SELECTION_FREEZE_FILE,
    validate_agent_usage_artifact,
    validate_certification_bundle_binding,
    validate_certification_source_bindings,
    validate_certification_timestamp_order,
    validate_certified_runtime_binding,
    validate_process_evidence_artifact,
    validate_selection_freeze_artifact,
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


def _unit_interval(value: object, label: str) -> float:
    observed = _finite(value, label)
    if not 0 <= observed <= 1:
        raise CampaignAnalysisError(f"{label} must be in [0, 1]")
    return observed


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
    bundle: Mapping[str, Any], task_family: str, cell_id: str,
) -> tuple[list[float], list[float]]:
    winner_folds = _ordered_folds(bundle.get("held_out_folds"), f"{cell_id}.winner")
    baseline_folds = _ordered_folds(
        bundle.get("baseline_held_out_folds"), f"{cell_id}.baseline",
    )
    # The sealed evidence schema is exact-key-locked PER TASK FAMILY
    # (HELD_OUT_SCHEMA_BY_FAMILY, frozen into the manifest cell record), and
    # the PRIMARY estimand is that family's first key — the analysis plan's
    # ``primary_by_task_family``: AUROC for binary and nominal multiclass,
    # quadratic-weighted kappa for ordinal grading, concordance index for
    # survival. Selection stays the primary VALIDATION metric everywhere
    # (scoring.formula: val_auc / val_c_index); companions are recorded in
    # the sealed evidence but never rank the campaign.
    required = HELD_OUT_SCHEMA_BY_FAMILY.get(task_family)
    if required is None:
        raise CampaignAnalysisError(
            f"{cell_id}: unknown task family {task_family!r}"
        )
    primary_key = required[0]
    values: list[list[float]] = [[], []]
    for output, folds, label in (
        (values[0], baseline_folds, "baseline"),
        (values[1], winner_folds, "winner"),
    ):
        for row in folds:
            metrics = row["held_out"]
            if set(metrics) != set(required):
                raise CampaignAnalysisError(
                    f"{cell_id}.{label} metric schema differs from {sorted(required)}"
                )
            for key in required:
                _unit_interval(metrics[key], f"{cell_id}.{label}.{key}")
            output.append(
                _unit_interval(
                    metrics[primary_key], f"{cell_id}.{label}.{primary_key}",
                )
            )

    for aggregate_key, folds, aggregate in (
        ("baseline_held_out", baseline_folds, bundle.get("baseline_held_out")),
        ("held_out", winner_folds, bundle.get("held_out")),
    ):
        if not isinstance(aggregate, dict):
            raise CampaignAnalysisError(f"{cell_id}.{aggregate_key} is malformed")
        if set(aggregate) != set(required):
            raise CampaignAnalysisError(
                f"{cell_id}.{aggregate_key} metric schema is not locked"
            )
        for key in required:
            observed = _unit_interval(
                aggregate.get(key), f"{cell_id}.{aggregate_key}.{key}",
            )
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


def _family_stratified_lift(
    cells: list[dict[str, Any]], key: str,
) -> dict[str, Any]:
    # Magnitude summaries pool only WITHIN a task family — each family has
    # exactly one reporting metric and metrics are not interchangeable (the
    # analysis plan's scale_rule); only sign counts may pool across families.
    return {
        family: _grouped_lift(
            [cell for cell in cells if cell["task_family"] == family], key,
        )
        for family in HELD_OUT_SCHEMA_BY_FAMILY
    }


def _direction_only(values: Iterable[float]) -> dict[str, int]:
    observed = [float(value) for value in values]
    tolerance = 1e-12
    return {
        "n": len(observed),
        "positive": sum(value > tolerance for value in observed),
        "zero": sum(abs(value) <= tolerance for value in observed),
        "negative": sum(value < -tolerance for value in observed),
    }


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


def _validated_agent_usage(raw: object, cell_id: str) -> dict[str, Any]:
    try:
        return validate_agent_usage_artifact(raw)
    except CampaignStageError as exc:
        raise CampaignAnalysisError(f"{cell_id}: {exc}") from exc


def _validated_process_evidence(
    raw: object, expected_sha256: object, cell_id: str,
    expected_session_id: str | None = None,
    expected_session_binding: str | None = None,
) -> dict[str, Any]:
    if expected_session_id is None or expected_session_binding is None:
        attempts = (
            (raw.get("discovery") or {}).get("attempts")
            if isinstance(raw, dict) else None
        )
        first = attempts[0] if isinstance(attempts, list) and attempts else None
        if not isinstance(first, dict):
            raise CampaignAnalysisError(
                f"{cell_id}: process evidence requires an exact session binding"
            )
        expected_session_id = first.get("agent_session_id")
        expected_session_binding = first.get("agent_session_binding_sha256")
        if (
            not isinstance(expected_session_id, str)
            or not isinstance(expected_session_binding, str)
        ):
            raise CampaignAnalysisError(
                f"{cell_id}: process evidence requires an exact session binding"
            )
    try:
        return validate_process_evidence_artifact(
            raw,
            expected_sha256,
            cell_id=cell_id,
            expected_session_id=expected_session_id,
            expected_session_binding=expected_session_binding,
        )
    except CampaignStageError as exc:
        raise CampaignAnalysisError(f"{cell_id}: {exc}") from exc


def _search_process_summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    class_counts: dict[str, int] = defaultdict(int)
    result_counts: dict[str, int] = defaultdict(int)
    promotion_counts: dict[str, int] = defaultdict(int)
    discovery_outcomes: dict[str, int] = defaultdict(int)
    promotion_outcomes: dict[str, int] = defaultdict(int)
    stage_resources: dict[str, dict[str, Any]] = {}
    for cell in cells:
        process = cell["search_process"]
        for key, value in process["discovery"]["candidate_class_counts"].items():
            class_counts[key] += int(value)
        for key, value in process["discovery"]["result_status_counts"].items():
            result_counts[key] += int(value)
        for key, value in process["promotion"]["status_counts"].items():
            promotion_counts[key] += int(value)
        for key, value in process["discovery"]["outcome_class_counts"].items():
            discovery_outcomes[key] += int(value)
        for key, value in process["promotion"]["outcome_class_counts"].items():
            promotion_outcomes[key] += int(value)
    for stage in ("baseline", "discovery", "promotion"):
        resources = [cell["search_process"][stage]["resources"] for cell in cells]
        elapsed_values = [
            float(row["elapsed_seconds"]["total"])
            for row in resources if row["elapsed_seconds"]["total"] is not None
        ]
        vram_values = [
            float(row["peak_vram_mb"]["maximum"])
            for row in resources if row["peak_vram_mb"]["maximum"] is not None
        ]
        elapsed_reported = sum(int(row["elapsed_seconds"]["reported"]) for row in resources)
        elapsed_missing = sum(int(row["elapsed_seconds"]["missing"]) for row in resources)
        observed_total = math.fsum(elapsed_values) if elapsed_values else None
        stage_resources[stage] = {
            "elapsed_seconds": {
                "reported": elapsed_reported,
                "missing": elapsed_missing,
                "observed_total": observed_total,
                "observed_gpu_attached_job_hours": (
                    observed_total / 3600 if observed_total is not None else None
                ),
                "lower_bound": elapsed_missing > 0 and elapsed_reported > 0,
            },
            "peak_vram_mb": {
                "reported": sum(
                    int(row["peak_vram_mb"]["reported"]) for row in resources
                ),
                "missing": sum(
                    int(row["peak_vram_mb"]["missing"]) for row in resources
                ),
                "maximum": max(vram_values) if vram_values else None,
            },
        }
    search_elapsed_values = [
        stage_resources[stage]["elapsed_seconds"]["observed_total"]
        for stage in ("discovery", "promotion")
        if stage_resources[stage]["elapsed_seconds"]["observed_total"] is not None
    ]
    search_elapsed = (
        math.fsum(search_elapsed_values) if search_elapsed_values else None
    )
    search_missing = sum(
        stage_resources[stage]["elapsed_seconds"]["missing"]
        for stage in ("discovery", "promotion")
    )
    return {
        "discovery_attempts": DISCOVERY_ATTEMPTS * len(cells),
        "candidate_class_counts": dict(sorted(class_counts.items())),
        "result_status_counts": dict(sorted(result_counts.items())),
        "promotion_status_counts": dict(sorted(promotion_counts.items())),
        "discovery_outcome_class_counts": dict(sorted(discovery_outcomes.items())),
        "promotion_outcome_class_counts": dict(sorted(promotion_outcomes.items())),
        "resources_by_stage": stage_resources,
        "agentic_search_total": {
            "elapsed_seconds": {
                "reported": sum(
                    stage_resources[stage]["elapsed_seconds"]["reported"]
                    for stage in ("discovery", "promotion")
                ),
                "missing": search_missing,
                "observed_total": search_elapsed,
                "observed_gpu_attached_job_hours": (
                    search_elapsed / 3600 if search_elapsed is not None else None
                ),
                "lower_bound": search_missing > 0 and search_elapsed is not None,
            },
        },
    }


def _ranking_blocks(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expected_arms = {framework for framework, _ in TILE_ARMS}
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        if cell["regime"] == "tile":
            grouped[(
                cell["dataset"], cell["task"], cell["task_type"],
                cell["task_family"], cell["encoder"],
            )].append(cell)
    blocks: list[dict[str, Any]] = []
    for (dataset, task, task_type, task_family, encoder), rows in sorted(
        grouped.items()
    ):
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
            # Rank statistics are scale-free, so the classification summary
            # deliberately pools blocks ranked on test_auc with the grade
            # blocks ranked on test_qwk (declared in the plan's
            # tile_ranking_response); the family here labels each block's
            # actual ranking metric so the mixture stays visible.
            "task_family": task_family,
            "primary_metric": HELD_OUT_SCHEMA_BY_FAMILY[task_family][0],
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
        or (((plan.get("estimands") or {}).get("search_process") or {}).get(
            "outcome_classes"
        ) != list(ATTEMPT_OUTCOME_CLASSES))
        or (plan.get("missingness") or {}).get(
            "expected_discovery_attempts_per_cell"
        ) != DISCOVERY_ATTEMPTS
    ):
        raise CampaignAnalysisError("analysis plan is not frozen for this campaign")
    # The plan's declared per-family primary must match the code authority
    # the report actually ranks on — the manifest hash pins the plan bytes,
    # but only this check pins the plan AGAINST the code that outlived it.
    if (plan.get("aggregation") or {}).get("primary_by_task_family") != {
        family: keys[0] for family, keys in HELD_OUT_SCHEMA_BY_FAMILY.items()
    }:
        raise CampaignAnalysisError(
            "analysis plan primary_by_task_family disagrees with "
            "HELD_OUT_SCHEMA_BY_FAMILY"
        )
    agent_protocol_path = runtime_root / AGENT_PROTOCOL_FILE
    try:
        agent_protocol = validate_agent_protocol(
            _read_json(agent_protocol_path, "agent protocol")
        )
    except ValueError as exc:
        raise CampaignAnalysisError(f"locked agent protocol is invalid: {exc}") from exc
    agent_protocol_sha256 = content_sha256(agent_protocol)

    index_path = runtime_root / CAMPAIGN_CERTIFICATION_FILE
    index = _read_json(index_path, "campaign certification index")
    recorded_index_hash = index.get("certification_sha256")
    index_fields = {
        "schema_version", "campaign_id", "manifest_sha256",
        "selection_freeze_sha256", "cell_count", "cells", "certified_at",
        "certification_sha256",
    }
    try:
        index_certified_at = datetime.fromisoformat(str(index.get("certified_at")))
    except ValueError:
        index_certified_at = None
    if (
        set(index) != index_fields
        or index.get("schema_version") != 1
        or index.get("campaign_id") != CAMPAIGN_ID
        or index.get("manifest_sha256") != _file_hash(manifest_path, "manifest")
        or index.get("cell_count") != CAMPAIGN_CELL_COUNT
        or index_certified_at is None
        or index_certified_at.tzinfo is None
        or recorded_index_hash != content_sha256({
            key: value for key, value in index.items()
            if key != "certification_sha256"
        })
    ):
        raise CampaignAnalysisError("campaign certification index integrity mismatch")
    raw_entries = index.get("cells")
    if not isinstance(raw_entries, list) or len(raw_entries) != CAMPAIGN_CELL_COUNT:
        raise CampaignAnalysisError("campaign certification roster is malformed")
    entries = {
        entry.get("cell_id"): entry for entry in raw_entries
        if isinstance(entry, dict) and isinstance(entry.get("cell_id"), str)
    }
    manifest_cells = {cell["cell_id"]: cell for cell in manifest["cells"]}
    if (
        len(entries) != CAMPAIGN_CELL_COUNT
        or set(entries) != set(manifest_cells)
        or any(
            not isinstance(entry, dict)
            or set(entry) != {
                "cell_id", "bundle", "bundle_sha256", "file_sha256",
            }
            or not isinstance(entry.get("bundle_sha256"), str)
            or len(entry["bundle_sha256"]) != 64
            or any(
                char not in "0123456789abcdef"
                for char in entry["bundle_sha256"]
            )
            or not isinstance(entry.get("file_sha256"), str)
            or len(entry["file_sha256"]) != 64
            or any(
                char not in "0123456789abcdef" for char in entry["file_sha256"]
            )
            for entry in raw_entries
        )
    ):
        raise CampaignAnalysisError("campaign certification roster is incomplete")
    try:
        freeze = validate_selection_freeze_artifact(_read_json(
            runtime_root / SELECTION_FREEZE_FILE, "campaign selection freeze",
        ))
    except CampaignStageError as exc:
        raise CampaignAnalysisError(
            f"campaign selection freeze is invalid: {exc}"
        ) from exc
    freeze_hash = freeze.get("freeze_sha256")
    manifest_sha256 = _file_hash(manifest_path, "manifest")
    if (
        freeze_hash != index.get("selection_freeze_sha256")
        or freeze_hash != content_sha256({
            key: value for key, value in freeze.items() if key != "freeze_sha256"
        })
        or freeze.get("campaign_id") != CAMPAIGN_ID
        or freeze.get("manifest_sha256") != manifest_sha256
        or freeze.get("protocol_version") != PROTOCOL_VERSION
        or freeze.get("agent_protocol_sha256") != agent_protocol_sha256
        or freeze.get("cell_count") != CAMPAIGN_CELL_COUNT
    ):
        raise CampaignAnalysisError("campaign selection freeze integrity mismatch")
    raw_freeze_entries = freeze.get("cells")
    if (
        not isinstance(raw_freeze_entries, list)
        or len(raw_freeze_entries) != CAMPAIGN_CELL_COUNT
    ):
        raise CampaignAnalysisError("campaign selection freeze roster is malformed")
    freeze_entries = {
        row.get("cell_id"): row for row in raw_freeze_entries
        if isinstance(row, dict) and isinstance(row.get("cell_id"), str)
    }
    manifest_roster = {
        cell["cell_id"]: cell["cell_sha256"] for cell in manifest["cells"]
    }
    freeze_roster = {
        row.get("cell_id"): row.get("cell_sha256")
        for row in raw_freeze_entries if isinstance(row, dict)
    }
    session_ids = [
        row.get("agent_session_id") if isinstance(row, dict) else None
        for row in raw_freeze_entries
    ]
    session_bindings = [
        row.get("agent_session_binding_sha256") if isinstance(row, dict) else None
        for row in raw_freeze_entries
    ]
    if (
        len(freeze_entries) != CAMPAIGN_CELL_COUNT
        or set(freeze_entries) != set(manifest_cells)
        or freeze_roster != manifest_roster
        or freeze.get("roster_sha256") != content_sha256(
            dict(sorted(manifest_roster.items()))
        )
        or any(not isinstance(session_id, str) or not session_id for session_id in session_ids)
        or len(set(session_ids)) != CAMPAIGN_CELL_COUNT
        or any(
            not isinstance(binding, str)
            or len(binding) != 64
            or any(char not in "0123456789abcdef" for char in binding)
            for binding in session_bindings
        )
    ):
        raise CampaignAnalysisError("campaign selection freeze roster is incomplete")

    cells: list[dict[str, Any]] = []
    for cell_id in sorted(manifest_cells):
        cell = manifest_cells[cell_id]
        entry = entries[cell_id]
        freeze_entry = freeze_entries[cell_id]
        if freeze_entry.get("cell_sha256") != cell.get("cell_sha256"):
            raise CampaignAnalysisError(f"{cell_id}: freeze cell binding mismatch")
        agent_usage = _validated_agent_usage(
            freeze_entry.get("agent_usage"), cell_id,
        )
        search_process = _validated_process_evidence(
            freeze_entry.get("process_evidence"),
            freeze_entry.get("process_sha256"),
            cell_id,
            str(freeze_entry["agent_session_id"]),
            str(freeze_entry.get("agent_session_binding_sha256")),
        )
        expected_bundle = f"{cell_id}/certification/certify.json"
        if entry.get("bundle") != expected_bundle:
            raise CampaignAnalysisError(
                f"{cell_id}: certification bundle path is not canonical"
            )
        bundle_path = _safe_relative(runtime_root, entry.get("bundle"), cell_id)
        if _file_hash(bundle_path, f"{cell_id} certification bundle") != entry.get("file_sha256"):
            raise CampaignAnalysisError(f"{cell_id}: certification file hash mismatch")
        try:
            bundle = validate_certification_bundle_binding(
                _read_json(bundle_path, f"{cell_id} certification bundle"),
                freeze_entry,
                selection_freeze_sha256=str(freeze["freeze_sha256"]),
            )
        except CampaignStageError as exc:
            raise CampaignAnalysisError(
                f"{cell_id}: certification bundle is invalid: {exc}"
            ) from exc
        try:
            validate_certification_source_bindings(
                runtime_root, bundle, freeze_entry,
            )
            validate_certification_timestamp_order(
                freeze["frozen_at"], bundle, index["certified_at"],
            )
            validate_certified_runtime_binding(
                runtime_root, freeze, freeze_entry, bundle,
            )
        except CampaignStageError as exc:
            raise CampaignAnalysisError(
                f"{cell_id}: certification source evidence is invalid: {exc}"
            ) from exc
        recorded_bundle_hash = bundle["bundle_sha256"]
        if (
            recorded_bundle_hash != entry.get("bundle_sha256")
            or bundle.get("selection_freeze_sha256")
            != index.get("selection_freeze_sha256")
        ):
            raise CampaignAnalysisError(f"{cell_id}: certification bundle mismatch")
        baseline_folds, winner_folds = _primary_values(
            bundle, cell["task_family"], cell_id,
        )
        baseline_primary = math.fsum(baseline_folds) / len(baseline_folds)
        winner_primary = math.fsum(winner_folds) / len(winner_folds)
        cells.append({
            "cell_id": cell_id,
            "dataset": cell["dataset"],
            "task": cell["task"],
            "task_type": cell["task_type"],
            "task_family": cell["task_family"],
            "primary_metric": HELD_OUT_SCHEMA_BY_FAMILY[cell["task_family"]][0],
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
            "search_process": search_process,
        })

    blocks = _ranking_blocks(cells)
    report: dict[str, Any] = {
        "schema_version": 2,
        "campaign_id": CAMPAIGN_ID,
        "manifest_sha256": _file_hash(manifest_path, "manifest"),
        "analysis_plan_sha256": _file_hash(plan_path, "analysis plan"),
        "certification_sha256": recorded_index_hash,
        "selection_freeze_sha256": index["selection_freeze_sha256"],
        "selection_roster_sha256": freeze["roster_sha256"],
        "protocol_version": freeze["protocol_version"],
        "agent_protocol_sha256": freeze["agent_protocol_sha256"],
        "cell_count": len(cells),
        "cells": cells,
        "tile_ranking_blocks": blocks,
        "summaries": {
            "overall_lift_direction_only": _direction_only(
                cell["primary_lift"] for cell in cells
            ),
            "agentic_lift": {
                "by_task_family": _grouped_lift(cells, "task_family"),
                "by_regime_within_task_family": _family_stratified_lift(
                    cells, "regime",
                ),
                "by_dataset_within_task_family": _family_stratified_lift(
                    cells, "dataset",
                ),
                "by_framework_within_task_family": _family_stratified_lift(
                    cells, "framework",
                ),
            },
            "tile_ranking_response": _ranking_summary(blocks),
            "agent_resources": _resource_summary(cells),
            "search_process": _search_process_summary(cells),
            "titan_by_task_family": {
                family: _summary(
                    cell["primary_lift"] for cell in cells
                    if cell["regime"] == "slide"
                    and cell["task_family"] == family
                )
                for family in HELD_OUT_SCHEMA_BY_FAMILY
            },
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
