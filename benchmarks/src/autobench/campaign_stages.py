"""Restart-safe stage ledger for the frozen preprint campaign.

This module is the trusted consumer-side controller.  It never imports or
opens held-out values during search: baseline registration hashes sealed files
without parsing them, while discovery freeze reads only agent-facing
``result.json`` files containing validation-only fold evidence.
"""
from __future__ import annotations

import fcntl
import copy
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Iterator, Mapping

import yaml

from automil.admissibility import (
    AdmissibilityError,
    load_candidate_policy,
    revalidate_candidate_spec,
)
from automil.cells.state import read_cell
from automil.cells.state import Cell, CellStatus, write_cell

from autobench.campaign import (
    CAMPAIGN_ID,
    CERTIFICATION_FOLDS,
    DISCOVERY_ATTEMPTS,
    PROMOTION_CANDIDATES,
    PROTOCOL,
    STAGE_FOLDS,
    content_sha256,
    file_sha256,
)

STATE_SCHEMA_VERSION = 1
STATE_FILE = "campaign_state.json"


class CampaignStageError(ValueError):
    """A campaign stage transition cannot be proven safe or complete."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state_digest(state: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in state.items() if key != "state_sha256"}
    return content_sha256(payload)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _commit_state(cell_root: Path, state: dict[str, Any]) -> dict[str, Any]:
    committed = json.loads(json.dumps(state))
    committed["state_sha256"] = _state_digest(committed)
    _atomic_write_json(cell_root / STATE_FILE, committed)
    return committed


@contextmanager
def _stage_lock(cell_root: Path) -> Iterator[None]:
    """Serialize controller transitions while keeping state writes atomic."""
    cell_root.mkdir(parents=True, exist_ok=True)
    lock_path = cell_root / ".campaign_state.lock"
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def load_stage_state(cell_root: Path) -> dict[str, Any]:
    """Load and integrity-check one cell's controller state."""
    path = cell_root / STATE_FILE
    try:
        state = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignStageError(f"cannot read stage state {path}: {exc}") from exc
    if not isinstance(state, dict):
        raise CampaignStageError("campaign stage state must be a JSON object")
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        raise CampaignStageError("unsupported campaign stage-state schema")
    if state.get("campaign_id") != CAMPAIGN_ID:
        raise CampaignStageError("stage state belongs to a different campaign")
    if state.get("protocol_sha256") != content_sha256(PROTOCOL):
        raise CampaignStageError("stage state protocol differs from the frozen contract")
    if state.get("state_sha256") != _state_digest(state):
        raise CampaignStageError("campaign stage-state integrity hash mismatch")
    return state


def initialize_stage_state(
    cell_root: Path,
    *,
    cell: Mapping[str, Any],
    manifest_sha256: str,
) -> dict[str, Any]:
    """Create the immutable discovery ledger, or verify an identical restart."""
    with _stage_lock(cell_root):
        return _initialize_stage_state_unlocked(
            cell_root, cell=cell, manifest_sha256=manifest_sha256,
        )


def _initialize_stage_state_unlocked(
    cell_root: Path,
    *,
    cell: Mapping[str, Any],
    manifest_sha256: str,
) -> dict[str, Any]:
    path = cell_root / STATE_FILE
    if path.exists():
        state = load_stage_state(cell_root)
        expected = (cell["cell_id"], cell["cell_sha256"], manifest_sha256)
        actual = (
            state.get("cell_id"), state.get("cell_sha256"),
            state.get("manifest_sha256"),
        )
        if actual != expected:
            raise CampaignStageError(
                "existing stage state is bound to different campaign inputs"
            )
        return state

    now = _utc_now()
    state: dict[str, Any] = {
        "schema_version": STATE_SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "manifest_sha256": manifest_sha256,
        "cell_id": cell["cell_id"],
        "cell_sha256": cell["cell_sha256"],
        "protocol_sha256": content_sha256(PROTOCOL),
        "phase": "discovery",
        "revision": 0,
        "created_at": now,
        "updated_at": now,
        "baseline": None,
        "discovery": {
            "attempt_budget": DISCOVERY_ATTEMPTS,
            "attempts_charged": 0,
            "complete_candidates": 0,
            "unique_complete_candidates": 0,
            "frozen": False,
            "frozen_at": None,
            "attempt_audit": [],
            "promoted_candidates": [],
        },
        "promotion": {
            "candidate_budget": PROMOTION_CANDIDATES,
            "jobs": [],
            "frozen": False,
            "materialized": False,
            "materialized_at": None,
        },
        "winner": None,
        "certification": None,
        "history": [{"event": "initialized", "at": now}],
    }
    return _commit_state(cell_root, state)


def _import_baseline_archive(cell_root: Path, source: Path) -> Path:
    """Atomically import only the baseline artifacts the campaign consumes."""
    source = source.resolve()
    target_root = cell_root / "baseline"
    target = target_root / "archive"
    if source == target.resolve():
        return target

    required = [source / "result.json"] + [
        source / "certify" / f"fold_{fold}_result.json"
        for fold in CERTIFICATION_FOLDS
    ]
    if not all(path.is_file() for path in required):
        raise CampaignStageError("external baseline archive is incomplete")
    expected = {path.relative_to(source).as_posix(): file_sha256(path) for path in required}
    if target_root.exists():
        actual = {
            relative: file_sha256(target / relative)
            for relative in expected
            if (target / relative).is_file()
        }
        if actual != expected:
            raise CampaignStageError(
                "cell-local baseline import exists with different artifact bytes"
            )
        return target

    temporary = Path(tempfile.mkdtemp(prefix=".baseline-", dir=str(cell_root)))
    temporary_archive = temporary / "archive"
    try:
        for source_file in required:
            relative = source_file.relative_to(source)
            destination = temporary_archive / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, destination)
        os.replace(temporary, target_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def _validation_folds(
    result: Mapping[str, Any], expected_folds: tuple[int, ...],
) -> list[dict[str, Any]]:
    """Validate and normalize public fold evidence without touching test."""
    if "held_out" in result or "summary" in result:
        raise CampaignStageError(
            "controller received a test-bearing result; use agent-facing result.json"
        )
    if result.get("status") != "completed":
        raise CampaignStageError(
            f"result status must be completed, got {result.get('status')!r}"
        )
    raw_folds = result.get("validation_folds")
    if not isinstance(raw_folds, list):
        raise CampaignStageError("result is missing validation_folds")
    normalized: list[dict[str, Any]] = []
    seen: set[int] = set()
    for raw in raw_folds:
        if not isinstance(raw, dict):
            raise CampaignStageError("validation_folds entries must be objects")
        fold_index = raw.get("fold_index")
        composite = raw.get("composite")
        if type(fold_index) is not int or fold_index in seen:
            raise CampaignStageError("validation fold indices must be unique integers")
        if (
            isinstance(composite, bool)
            or not isinstance(composite, (int, float))
            or not math.isfinite(float(composite))
        ):
            raise CampaignStageError(
                f"fold {fold_index} has no finite validation composite"
            )
        metrics = raw.get("metrics")
        if not isinstance(metrics, dict) or any("test" in str(key).lower() for key in metrics):
            raise CampaignStageError(
                f"fold {fold_index} metrics are not validation-only"
            )
        seen.add(fold_index)
        normalized.append({
            "fold_index": fold_index,
            "metrics": metrics,
            "composite": float(composite),
        })
    if seen != set(expected_folds):
        raise CampaignStageError(
            f"validation folds must be exactly {list(expected_folds)}, got {sorted(seen)}"
        )
    normalized.sort(key=lambda fold: fold["fold_index"])
    return normalized


def _mean(folds: list[Mapping[str, Any]]) -> float:
    return math.fsum(float(fold["composite"]) for fold in folds) / len(folds)


def _ensure_discovery_baseline_root(
    cell_root: Path,
    state: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> str:
    """Create or verify the discovery graph's validation-only incumbent.

    Discovery compares candidates on folds 0/1/2, so its graph root must use
    the same evidence.  The campaign's final baseline incumbent remains the
    separately frozen five-fold mean in ``campaign_state.json``.
    """
    from automil.graph import locked_update, merged_metadata
    from automil.scoring import cross_fold_se

    discovery_folds = [
        fold for fold in baseline["validation_folds"]
        if fold["fold_index"] in STAGE_FOLDS["discovery"]
    ]
    if [fold["fold_index"] for fold in discovery_folds] != list(
        STAGE_FOLDS["discovery"]
    ):
        raise CampaignStageError("baseline lacks exact discovery-fold evidence")
    discovery_mean = _mean(discovery_folds)
    discovery_se = cross_fold_se(
        [float(fold["composite"]) for fold in discovery_folds]
    )
    try:
        cell = json.loads(
            (cell_root / "automil" / "campaign_cell.json").read_text()
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignStageError(f"cannot read campaign cell identity: {exc}") from exc
    if cell.get("cell_id") != state["cell_id"]:
        raise CampaignStageError("campaign cell identity differs from stage state")
    budget_cell_id = cell["budget_identity"]["cell_id"]
    graph_path = cell_root / "automil" / "graph.json"
    with locked_update(graph_path) as graph:
        matches = [
            node for node in graph.nodes.values()
            if (isinstance(node.get("metadata"), dict)
                and node["metadata"].get("campaign_baseline_sha256")
                == baseline["candidate_sha256"])
        ]
        if len(matches) > 1:
            raise CampaignStageError("discovery graph contains duplicate baseline roots")
        if matches:
            node = matches[0]
            metadata = node.get("metadata") or {}
            recorded_composite = node.get("composite")
            recorded_baseline = graph.meta.get("baseline_composite")
            valid = (
                node.get("parent_id") is None
                and node.get("type") == "executed"
                and node.get("status") == "keep"
                and not isinstance(recorded_composite, bool)
                and isinstance(recorded_composite, (int, float))
                and math.isclose(
                    float(recorded_composite), discovery_mean,
                    rel_tol=0.0, abs_tol=1e-12,
                )
                and metadata.get("cell_id") == budget_cell_id
                and isinstance(metadata.get("validation_folds"), list)
                and content_sha256(metadata.get("validation_folds"))
                == content_sha256(discovery_folds)
                and not isinstance(recorded_baseline, bool)
                and isinstance(recorded_baseline, (int, float))
                and math.isclose(
                    float(recorded_baseline),
                    discovery_mean, rel_tol=0.0, abs_tol=1e-12,
                )
            )
            if not valid:
                raise CampaignStageError("discovery baseline graph root drifted")
            recorded_root = baseline.get("discovery_root_node_id")
            if recorded_root is not None and recorded_root != node["id"]:
                raise CampaignStageError("baseline ledger points to a different graph root")
            return str(node["id"])
        if graph.nodes:
            raise CampaignStageError(
                "discovery graph already has nodes but no registered baseline root"
            )

        node_id = graph.add_executed(
            parent_id=None,
            description="native upstream baseline (discovery folds 0/1/2)",
            techniques=[],
            metrics={
                "composite": discovery_mean,
                "composite_se": discovery_se,
            },
            status="keep",
            config_hash=baseline["candidate_sha256"],
            bootstrapped=True,
        )
        node = graph.get_node(node_id)
        node["cell_id"] = budget_cell_id
        node["metadata"] = merged_metadata(node, {
            "cell_id": budget_cell_id,
            "mil_model": cell["budget_identity"]["mil_model"],
            "campaign_baseline_sha256": baseline["candidate_sha256"],
            "validation_folds": discovery_folds,
        })
        graph.meta["baseline_composite"] = discovery_mean
        graph.recalculate_scores()
        return node_id


def register_baseline(cell_root: Path, baseline_archive: Path) -> dict[str, Any]:
    """Register the five-fold native baseline as the immutable incumbent.

    Sealed fold files are hashed as opaque bytes here.  Their JSON is not parsed
    until winner-only certification, after validation has frozen the winner.
    """
    with _stage_lock(cell_root):
        return _register_baseline_unlocked(cell_root, baseline_archive)


def _register_baseline_unlocked(
    cell_root: Path, baseline_archive: Path,
) -> dict[str, Any]:
    state = load_stage_state(cell_root)
    if state["phase"] != "discovery":
        raise CampaignStageError("baseline must be registered before discovery freeze")
    baseline_archive = baseline_archive.resolve()
    result_path = baseline_archive / "result.json"
    try:
        result = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignStageError(f"cannot read baseline result.json: {exc}") from exc
    folds = _validation_folds(result, CERTIFICATION_FOLDS)
    sealed_hashes: dict[str, str] = {}
    for fold_index in CERTIFICATION_FOLDS:
        sealed = baseline_archive / "certify" / f"fold_{fold_index}_result.json"
        if not sealed.is_file():
            raise CampaignStageError(f"baseline is missing sealed fold {fold_index}")
        sealed_hashes[f"fold_{fold_index}_result.json"] = file_sha256(sealed)
    imported_archive = _import_baseline_archive(cell_root, baseline_archive)
    imported_result = imported_archive / "result.json"
    if file_sha256(imported_result) != file_sha256(result_path):
        raise CampaignStageError("cell-local baseline import changed result bytes")
    identity_payload = {
        "result_sha256": file_sha256(result_path),
        "sealed_fold_sha256": sealed_hashes,
        "validation_folds": folds,
    }
    baseline = {
        "candidate_id": "baseline",
        "candidate_sha256": content_sha256(identity_payload),
        "archive": "baseline/archive",
        "result_sha256": identity_payload["result_sha256"],
        "sealed_fold_sha256": sealed_hashes,
        "validation_folds": folds,
        "validation_mean": _mean(folds),
        "registered_at": _utc_now(),
    }
    current = state.get("baseline")
    if current is not None:
        if current.get("candidate_sha256") != baseline["candidate_sha256"]:
            raise CampaignStageError("a different baseline is already registered")
        _ensure_discovery_baseline_root(cell_root, state, current)
        return state
    baseline["discovery_root_node_id"] = _ensure_discovery_baseline_root(
        cell_root, state, baseline,
    )
    state["baseline"] = baseline
    state["revision"] += 1
    state["updated_at"] = _utc_now()
    state["history"].append({
        "event": "baseline-registered",
        "candidate_sha256": baseline["candidate_sha256"],
        "at": state["updated_at"],
    })
    return _commit_state(cell_root, state)


def _candidate_identity(
    spec: Mapping[str, Any], verdict: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    payload = {
        "base_commit": spec.get("base_commit"),
        "run_command_override": spec.get("run_command_override"),
        "overlay_manifest": spec.get("overlay_manifest") or {},
        "deletions": spec.get("deletions") or [],
        "candidate_class": verdict.get("candidate_class"),
        "policy_hash": verdict.get("policy_hash"),
        "variant_selection_hash": verdict.get("variant_selection_hash"),
        "override_hash": verdict.get("override_hash"),
    }
    return content_sha256(payload), payload


def _pending_stage_work(adir: Path) -> list[str]:
    pending: list[str] = []
    queue = adir / "orchestrator" / "queue"
    if queue.exists():
        pending.extend(path.name for path in queue.glob("*.json"))
    running = adir / "orchestrator" / "running"
    if running.exists():
        pending.extend(path.name for path in running.rglob("*.json"))
    return sorted(pending)


def _discovery_cell(adir: Path, budget_cell_id: str):
    path = adir / "cells" / f"{budget_cell_id}.json"
    if not path.is_file():
        raise CampaignStageError("discovery budget cell has not been opened")
    return read_cell(path)


def freeze_discovery(cell_root: Path) -> dict[str, Any]:
    """Freeze up to ten complete candidates after exactly 60 charged attempts."""
    with _stage_lock(cell_root):
        return _freeze_discovery_unlocked(cell_root)


def _freeze_discovery_unlocked(cell_root: Path) -> dict[str, Any]:
    state = load_stage_state(cell_root)
    if state["discovery"]["frozen"]:
        return state
    if state["phase"] != "discovery":
        raise CampaignStageError(f"cannot freeze discovery from phase {state['phase']!r}")
    if state.get("baseline") is None:
        raise CampaignStageError("register the native five-fold baseline first")

    adir = cell_root / "automil"
    try:
        config = json.loads((adir / "campaign_cell.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignStageError(f"cannot read campaign_cell.json: {exc}") from exc
    if config.get("cell_id") != state["cell_id"]:
        raise CampaignStageError("discovery root belongs to a different cell")
    pending = _pending_stage_work(adir)
    if pending:
        raise CampaignStageError(f"discovery still has queued/running work: {pending}")
    budget_cell = _discovery_cell(adir, config["budget_identity"]["cell_id"])
    if budget_cell.eval_budget != DISCOVERY_ATTEMPTS:
        raise CampaignStageError("discovery cell does not carry the frozen 60-attempt cap")
    if budget_cell.consumed_evals != DISCOVERY_ATTEMPTS:
        raise CampaignStageError(
            f"discovery requires exactly {DISCOVERY_ATTEMPTS} charged attempts; "
            f"found {budget_cell.consumed_evals}"
        )

    policy = load_candidate_policy(adir)
    archive_root = adir / "orchestrator" / "archive"
    attempt_audit: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    launched = 0
    for archive in sorted(archive_root.iterdir() if archive_root.exists() else []):
        if not archive.is_dir() or not (archive / "spec.json").is_file():
            continue
        try:
            spec = json.loads((archive / "spec.json").read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise CampaignStageError(f"cannot read {archive.name}/spec.json: {exc}") from exc
        campaign = (spec.get("metadata") or {}).get("campaign") or {}
        if campaign.get("cell_id") != state["cell_id"] or campaign.get("stage") != "discovery":
            continue
        if (spec.get("metadata") or {}).get("cap_refused"):
            continue
        launched += 1
        audit: dict[str, Any] = {
            "node_id": archive.name,
            "eligible": False,
            "reason": "missing result.json",
        }
        result_path = archive / "result.json"
        if result_path.is_file():
            try:
                result = json.loads(result_path.read_text())
                folds = _validation_folds(result, STAGE_FOLDS["discovery"])
                verdict = revalidate_candidate_spec(policy, spec, archive).to_dict()
                candidate_sha, identity = _candidate_identity(spec, verdict)
                candidate = {
                    "candidate_id": archive.name,
                    "candidate_sha256": candidate_sha,
                    "source_spec_sha256": file_sha256(archive / "spec.json"),
                    "identity": identity,
                    "validation_folds": folds,
                    "discovery_mean": _mean(folds),
                }
                audit.update({
                    "eligible": True,
                    "reason": "complete",
                    "candidate_sha256": candidate_sha,
                    "validation_mean": candidate["discovery_mean"],
                })
                eligible.append(candidate)
            except (
                CampaignStageError, AdmissibilityError, OSError,
                json.JSONDecodeError, KeyError, TypeError, ValueError,
            ) as exc:
                audit["reason"] = str(exc)
        attempt_audit.append(audit)
    if launched != DISCOVERY_ATTEMPTS:
        raise CampaignStageError(
            f"charged-attempt ledger says {DISCOVERY_ATTEMPTS}, but exactly "
            f"{launched} launched discovery specs were archived"
        )

    eligible.sort(key=lambda item: (-item["discovery_mean"], item["candidate_id"]))
    unique_eligible: list[dict[str, Any]] = []
    seen_identities: set[str] = set()
    for candidate in eligible:
        identity = candidate["candidate_sha256"]
        if identity in seen_identities:
            continue
        seen_identities.add(identity)
        unique_eligible.append(candidate)
    promoted = unique_eligible[:PROMOTION_CANDIDATES]
    frozen_at = _utc_now()
    state["phase"] = "promotion-ready" if promoted else "selection-ready"
    state["discovery"].update({
        "attempts_charged": DISCOVERY_ATTEMPTS,
        "complete_candidates": len(eligible),
        "unique_complete_candidates": len(unique_eligible),
        "frozen": True,
        "frozen_at": frozen_at,
        "attempt_audit": attempt_audit,
        "promoted_candidates": promoted,
    })
    state["revision"] += 1
    state["updated_at"] = frozen_at
    state["history"].append({
        "event": "discovery-frozen",
        "attempts_charged": DISCOVERY_ATTEMPTS,
        "complete_candidates": len(eligible),
        "unique_complete_candidates": len(unique_eligible),
        "promoted_candidates": len(promoted),
        "at": frozen_at,
    })
    return _commit_state(cell_root, state)


def _map_overlay_path(path: str, source_adir_rel: str, target_adir_rel: str) -> str:
    candidate = PurePosixPath(path)
    source_root = PurePosixPath(source_adir_rel)
    try:
        suffix = candidate.relative_to(source_root)
    except ValueError as exc:
        raise CampaignStageError(
            f"candidate overlay path {path!r} escapes its discovery automil root"
        ) from exc
    return (PurePosixPath(target_adir_rel) / suffix).as_posix()


def _copy_exact_overlay(
    *,
    source_archive: Path,
    target_archive: Path,
    source_spec: Mapping[str, Any],
    source_adir_rel: str,
    target_adir_rel: str,
) -> tuple[dict[str, str], list[str], list[str]]:
    source_manifest = source_spec.get("overlay_manifest") or {}
    if not isinstance(source_manifest, dict):
        raise CampaignStageError("source overlay_manifest must be an object")
    target_manifest: dict[str, str] = {}
    path_map: dict[str, str] = {}
    for source_path, recorded_hash in sorted(source_manifest.items()):
        if not isinstance(source_path, str) or not isinstance(recorded_hash, str):
            raise CampaignStageError("source overlay manifest is not string-to-string")
        target_path = _map_overlay_path(
            source_path, source_adir_rel, target_adir_rel,
        )
        source_file = source_archive / source_path
        if not source_file.is_file():
            raise CampaignStageError(f"source overlay file is missing: {source_path}")
        actual = f"sha256:{file_sha256(source_file)}"
        if actual != recorded_hash:
            raise CampaignStageError(f"source overlay hash drift: {source_path}")
        destination = target_archive / target_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, destination)
        target_manifest[target_path] = actual
        path_map[source_path] = target_path

    source_deletions = source_spec.get("deletions") or []
    if not isinstance(source_deletions, list) or not all(
        isinstance(path, str) for path in source_deletions
    ):
        raise CampaignStageError("source deletions must be a list of paths")
    target_deletions = [
        _map_overlay_path(path, source_adir_rel, target_adir_rel)
        for path in source_deletions
    ]
    source_framework = source_spec.get("framework_overlay_files") or []
    if not isinstance(source_framework, list) or not all(
        isinstance(path, str) for path in source_framework
    ):
        raise CampaignStageError("source framework_overlay_files must be paths")
    target_framework = []
    for path in source_framework:
        mapped = path_map.get(path)
        if mapped is None:
            raise CampaignStageError(
                f"framework overlay path {path!r} is absent from source manifest"
            )
        target_framework.append(mapped)
    return target_manifest, target_deletions, target_framework


def _selection_from_overlay(
    archive: Path, framework_files: list[str],
) -> Mapping[str, Any] | None:
    candidates = [
        archive / path for path in framework_files
        if PurePosixPath(path).name == "applied_variant.json"
    ]
    if not candidates:
        return None
    if len(candidates) != 1:
        raise CampaignStageError("promotion overlay has multiple variant selections")
    try:
        selection = json.loads(candidates[0].read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignStageError(f"invalid promotion variant selection: {exc}") from exc
    if not isinstance(selection, dict):
        raise CampaignStageError("promotion variant selection must be an object")
    return selection


def _promotion_cell(
    adir: Path, *, cell: Mapping[str, Any], config: Mapping[str, Any], budget: int,
) -> None:
    from automil.cells.capconfig import resolve_cap_config
    from automil.cells.state import normalize_mil_model

    cap = resolve_cap_config(dict(config), eval_budget_override=budget)
    budget_identity = cell["budget_identity"]
    expected_id = budget_identity["cell_id"]
    created = Cell(
        cell_id=expected_id,
        dataset=str(budget_identity["dataset"]),
        encoder=str(budget_identity["encoder"]),
        mil_model=normalize_mil_model(str(budget_identity["mil_model"])),
        started_at=time.time(),
        budget_seconds=cap.budget_seconds,
        safety_buffer_seconds=cap.safety_buffer_seconds,
        status=CellStatus.ACTIVE,
        mode=cap.mode,
        idle_grace_seconds=cap.idle_grace_seconds,
        eval_budget=budget,
    )
    write_cell(created, adir / "cells")


def materialize_promotion(
    cell_root: Path, *, repo_root: Path,
) -> dict[str, Any]:
    """Atomically create exact promotion jobs for the frozen top candidates."""
    with _stage_lock(cell_root):
        return _materialize_promotion_unlocked(cell_root, repo_root=repo_root)


def _materialize_promotion_unlocked(
    cell_root: Path, *, repo_root: Path,
) -> dict[str, Any]:
    state = load_stage_state(cell_root)
    if state["promotion"]["materialized"]:
        return state
    if state["phase"] != "promotion-ready":
        raise CampaignStageError(
            f"promotion can materialize only from promotion-ready, got {state['phase']!r}"
        )
    candidates = state["discovery"]["promoted_candidates"]
    if not candidates or len(candidates) > PROMOTION_CANDIDATES:
        raise CampaignStageError("frozen promotion candidate count is invalid")
    repo_root = repo_root.resolve()
    cell_root = cell_root.resolve()
    try:
        cell_root.relative_to(repo_root)
    except ValueError as exc:
        raise CampaignStageError("campaign cell root must live inside repo_root") from exc

    source_adir = cell_root / "automil"
    source_adir_rel = source_adir.relative_to(repo_root).as_posix()
    cell = json.loads((source_adir / "campaign_cell.json").read_text())
    target_dir = cell_root / "promotion"
    target_adir = target_dir / "automil"
    target_adir_rel = target_adir.relative_to(repo_root).as_posix()
    if target_dir.exists():
        jobs = _recover_promotion_plan(
            state, target_adir=target_adir, source_adir=source_adir,
        )
        return _finalize_promotion_state(cell_root, state, jobs)

    temporary = Path(tempfile.mkdtemp(prefix=".promotion-", dir=str(cell_root)))
    temporary_adir = temporary / "automil"
    try:
        source_config = yaml.safe_load((source_adir / "config.yaml").read_text()) or {}
        config = copy.deepcopy(source_config)
        config["files"] = {
            "editable": [f"{target_adir_rel}/variants/_policies/*.py"],
        }
        config.setdefault("run", {})["command"] = cell["commands"]["promotion"]
        config["run"]["mil_model"] = cell["model"]
        config.setdefault("cap", {})["eval_budget"] = len(candidates)
        config["training"] = {"fold_count": len(STAGE_FOLDS["promotion"])}
        config.setdefault("campaign", {})["stage"] = "promotion"
        temporary_adir.mkdir(parents=True)
        (temporary_adir / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
        )
        (temporary_adir / "campaign_cell.json").write_text(
            json.dumps(cell, indent=2, sort_keys=True) + "\n"
        )
        (temporary_adir / ".gitignore").write_text(
            "graph.json\nresults.tsv\nresult.json\norchestrator/\ncells/\n"
            ".automil_active\n.automil_worktrees/\n*.log\n*.pid\n"
        )
        (temporary_adir / "plan.md").write_text(
            f"# Frozen promotion — {state['cell_id']}\n\n"
            f"{len(candidates)} exact candidates; no agent proposals permitted.\n"
        )
        (temporary_adir / "learnings.md").write_text(
            f"# Promotion ledger — {state['cell_id']}\n"
        )
        (temporary_adir / "variants" / "_policies").mkdir(parents=True)
        _promotion_cell(
            temporary_adir, cell=cell, config=config, budget=len(candidates),
        )

        policy = load_candidate_policy(temporary_adir)
        target_command = config["run"]["command"]
        command_hash = hashlib.sha256(target_command.encode()).hexdigest()
        campaign_binding = dict(config["campaign"])
        from automil.admissibility import validate_campaign_binding

        manifest_path = repo_root / campaign_binding["manifest"]
        validate_campaign_binding(
            manifest_path,
            campaign_binding,
            base_run_command=target_command,
            budget_cell_id=cell["budget_identity"]["cell_id"],
        )

        jobs: list[dict[str, Any]] = []
        queue_dir = temporary_adir / "orchestrator" / "queue"
        archive_root = temporary_adir / "orchestrator" / "archive"
        queue_dir.mkdir(parents=True)
        from automil.graph import locked_update, merged_metadata

        with locked_update(temporary_adir / "graph.json") as graph:
            for rank, candidate in enumerate(candidates, 1):
                source_node = candidate["candidate_id"]
                source_archive = source_adir / "orchestrator" / "archive" / source_node
                source_spec_path = source_archive / "spec.json"
                if file_sha256(source_spec_path) != candidate["source_spec_sha256"]:
                    raise CampaignStageError(
                        f"source spec changed after discovery freeze: {source_node}"
                    )
                source_spec = json.loads(source_spec_path.read_text())
                source_verdict = revalidate_candidate_spec(
                    load_candidate_policy(source_adir), source_spec, source_archive,
                ).to_dict()
                source_sha, _ = _candidate_identity(source_spec, source_verdict)
                if source_sha != candidate["candidate_sha256"]:
                    raise CampaignStageError(
                        f"source candidate identity changed after freeze: {source_node}"
                    )

                target_node = graph.add_proposed(
                    parent_id="campaign_root",
                    description=f"promotion rank {rank}: exact {source_node}",
                    techniques=[],
                    kind="hp" if source_verdict["candidate_class"] == "config-only"
                    else "regularization",
                )
                graph.mark_running(target_node)
                graph_node = graph.get_node(target_node)
                graph_node["cell_id"] = cell["budget_identity"]["cell_id"]
                graph_node["metadata"] = merged_metadata(graph_node, {
                    "source_node_id": source_node,
                    "source_candidate_sha256": source_sha,
                })

                target_archive = archive_root / target_node
                target_archive.mkdir(parents=True)
                manifest, deletions, framework_files = _copy_exact_overlay(
                    source_archive=source_archive,
                    target_archive=target_archive,
                    source_spec=source_spec,
                    source_adir_rel=source_adir_rel,
                    target_adir_rel=target_adir_rel,
                )
                selection = _selection_from_overlay(target_archive, framework_files)
                candidate_paths = sorted(
                    (set(manifest) - set(framework_files)) | set(deletions)
                )
                override = source_spec.get("run_command_override")
                verdict = policy.classify(
                    candidate_paths,
                    override=str(override) if override is not None else None,
                    variant_selection=selection,
                )
                if not verdict.accepted:
                    raise CampaignStageError(
                        f"promotion mapping made {source_node} inadmissible: {verdict.reason}"
                    )
                config_parts = [
                    f"{path}:{digest}" for path, digest in sorted(manifest.items())
                ] + [f"DELETE:{path}" for path in sorted(deletions)]
                if override is not None:
                    config_parts.append(f"OVERRIDE:{override}")
                config_hash = hashlib.sha256(
                    (str(source_spec["base_commit"]) + "\n" + "\n".join(config_parts)).encode()
                ).hexdigest()[:16]
                target_spec: dict[str, Any] = {
                    "id": target_node,
                    "description": f"promotion rank {rank}: exact {source_node}",
                    "base_commit": source_spec["base_commit"],
                    "overlay_dir": f"archive/{target_node}",
                    "overlay_manifest": manifest,
                    "deletions": deletions,
                    "framework_overlay_files": framework_files,
                    "admissibility": verdict.to_dict(),
                    "base_run_command_sha256": command_hash,
                    "priority": 0,
                    "estimated_vram_gb": source_spec.get("estimated_vram_gb", 0),
                    "graph_metadata": {
                        "parent_id": "campaign_root",
                        "techniques": [],
                        "config_hash": config_hash,
                    },
                    "submitted_at": _utc_now(),
                    "metadata": {
                        "backend": (config.get("backend") or {}).get("name", "local"),
                        "runtime": "campaign-controller",
                        "cell_id": cell["budget_identity"]["cell_id"],
                        "campaign": campaign_binding,
                        "promotion": {
                            "source_node_id": source_node,
                            "source_candidate_sha256": source_sha,
                            "source_spec_sha256": candidate["source_spec_sha256"],
                            "expected_folds": list(STAGE_FOLDS["promotion"]),
                        },
                    },
                }
                if override is not None:
                    target_spec["run_command_override"] = override
                revalidate_candidate_spec(policy, target_spec, target_archive)
                (queue_dir / f"{target_node}.json").write_text(
                    json.dumps(target_spec, indent=2, sort_keys=True) + "\n"
                )
                target_sha, target_identity = _candidate_identity(
                    target_spec, verdict.to_dict(),
                )
                jobs.append({
                    "rank": rank,
                    "source_node_id": source_node,
                    "source_candidate_sha256": source_sha,
                    "promotion_node_id": target_node,
                    "promotion_candidate_sha256": target_sha,
                    "promotion_identity": target_identity,
                    "status": "queued",
                })

        plan = {
            "campaign_id": CAMPAIGN_ID,
            "cell_id": state["cell_id"],
            "source_state_sha256": state["state_sha256"],
            "jobs": jobs,
        }
        plan["plan_sha256"] = content_sha256(plan)
        (temporary_adir / "promotion_plan.json").write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary, target_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return _finalize_promotion_state(cell_root, state, jobs)


def _recover_promotion_plan(
    state: Mapping[str, Any], *, target_adir: Path, source_adir: Path,
) -> list[dict[str, Any]]:
    """Adopt an atomically published plan if state commit was interrupted."""
    plan_path = target_adir / "promotion_plan.json"
    try:
        plan = json.loads(plan_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignStageError(
            "promotion directory exists without a recoverable immutable plan"
        ) from exc
    recorded_hash = plan.pop("plan_sha256", None)
    if recorded_hash != content_sha256(plan):
        raise CampaignStageError("promotion plan integrity hash mismatch")
    if (
        plan.get("campaign_id") != CAMPAIGN_ID
        or plan.get("cell_id") != state["cell_id"]
        or plan.get("source_state_sha256") != state["state_sha256"]
    ):
        raise CampaignStageError("promotion plan is not bound to the frozen state")
    jobs = plan.get("jobs")
    if not isinstance(jobs, list):
        raise CampaignStageError("promotion plan jobs must be a list")
    expected = [
        (candidate["candidate_id"], candidate["candidate_sha256"])
        for candidate in state["discovery"]["promoted_candidates"]
    ]
    actual = [
        (job.get("source_node_id"), job.get("source_candidate_sha256"))
        for job in jobs if isinstance(job, dict)
    ]
    if actual != expected:
        raise CampaignStageError("promotion plan candidate order/identity drifted")
    if not (target_adir / "graph.json").is_file():
        raise CampaignStageError("promotion plan is missing graph.json")
    for job in jobs:
        node_id = job["promotion_node_id"]
        queue_spec = target_adir / "orchestrator" / "queue" / f"{node_id}.json"
        archived_spec = target_adir / "orchestrator" / "archive" / node_id / "spec.json"
        running_specs = list(
            (target_adir / "orchestrator" / "running").glob(f"*/{node_id}.json")
        )
        if not queue_spec.is_file() and not archived_spec.is_file() and not running_specs:
            raise CampaignStageError(f"promotion job {node_id} has no durable spec")
    # Re-read the source specs named by the plan. This does not inspect results
    # or any held-out file; it proves the frozen source identity still exists.
    for job in jobs:
        source_spec = (
            source_adir / "orchestrator" / "archive"
            / job["source_node_id"] / "spec.json"
        )
        if not source_spec.is_file():
            raise CampaignStageError(
                f"promotion source disappeared: {job['source_node_id']}"
            )
    return jobs


def _finalize_promotion_state(
    cell_root: Path, state: dict[str, Any], jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    now = _utc_now()
    state["phase"] = "promotion"
    state["promotion"].update({
        "jobs": jobs,
        "materialized": True,
        "materialized_at": now,
    })
    state["revision"] += 1
    state["updated_at"] = now
    state["history"].append({
        "event": "promotion-materialized",
        "jobs": len(jobs),
        "at": now,
    })
    return _commit_state(cell_root, state)


def freeze_promotion(cell_root: Path) -> dict[str, Any]:
    """Reconcile every promotion job and freeze the five-fold eligible pool."""
    with _stage_lock(cell_root):
        return _freeze_promotion_unlocked(cell_root)


def _freeze_promotion_unlocked(cell_root: Path) -> dict[str, Any]:
    state = load_stage_state(cell_root)
    if state["promotion"]["frozen"]:
        return state
    if state["phase"] != "promotion":
        raise CampaignStageError(
            f"promotion can freeze only from promotion phase, got {state['phase']!r}"
        )
    jobs = state["promotion"]["jobs"]
    if not jobs:
        raise CampaignStageError("promotion ledger contains no jobs")
    adir = cell_root / "promotion" / "automil"
    pending = _pending_stage_work(adir)
    if pending:
        raise CampaignStageError(f"promotion still has queued/running work: {pending}")
    cell = json.loads((adir / "campaign_cell.json").read_text())
    budget_cell = _discovery_cell(adir, cell["budget_identity"]["cell_id"])
    if budget_cell.eval_budget != len(jobs):
        raise CampaignStageError("promotion budget differs from frozen job count")
    if budget_cell.consumed_evals != len(jobs):
        raise CampaignStageError(
            f"promotion requires {len(jobs)} charged attempts; "
            f"found {budget_cell.consumed_evals}"
        )

    policy = load_candidate_policy(adir)
    discovery_by_id = {
        candidate["candidate_id"]: candidate
        for candidate in state["discovery"]["promoted_candidates"]
    }
    frozen_jobs: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for job in jobs:
        job = dict(job)
        source_node = job["source_node_id"]
        source = discovery_by_id.get(source_node)
        if source is None or source["candidate_sha256"] != job["source_candidate_sha256"]:
            raise CampaignStageError(
                f"promotion job {job['promotion_node_id']} lost its frozen source identity"
            )
        node_id = job["promotion_node_id"]
        archive = adir / "orchestrator" / "archive" / node_id
        spec_path = archive / "spec.json"
        result_path = archive / "result.json"
        if not spec_path.is_file() or not result_path.is_file():
            raise CampaignStageError(
                f"promotion job {node_id} is terminal but lacks spec/result artifact"
            )
        try:
            spec = json.loads(spec_path.read_text())
            result = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise CampaignStageError(f"cannot read promotion job {node_id}: {exc}") from exc
        link = (spec.get("metadata") or {}).get("promotion") or {}
        if (
            link.get("source_node_id") != source_node
            or link.get("source_candidate_sha256") != source["candidate_sha256"]
            or link.get("source_spec_sha256") != source["source_spec_sha256"]
            or link.get("expected_folds") != list(STAGE_FOLDS["promotion"])
        ):
            raise CampaignStageError(f"promotion source link drifted for {node_id}")
        verdict = revalidate_candidate_spec(policy, spec, archive).to_dict()
        promotion_sha, _ = _candidate_identity(spec, verdict)
        if promotion_sha != job["promotion_candidate_sha256"]:
            raise CampaignStageError(f"promotion candidate identity drifted for {node_id}")

        if result.get("status") != "completed":
            job.update({
                "status": "ineligible",
                "reason": f"promotion status {result.get('status')!r}",
            })
            frozen_jobs.append(job)
            continue
        try:
            promotion_folds = _validation_folds(
                result, STAGE_FOLDS["promotion"],
            )
        except CampaignStageError as exc:
            job.update({"status": "ineligible", "reason": str(exc)})
            frozen_jobs.append(job)
            continue
        five_folds = sorted(
            [*source["validation_folds"], *promotion_folds],
            key=lambda fold: fold["fold_index"],
        )
        if [fold["fold_index"] for fold in five_folds] != list(CERTIFICATION_FOLDS):
            raise CampaignStageError(f"five-fold coverage is not exact for {node_id}")
        selection_candidate = {
            "candidate_id": source_node,
            "candidate_sha256": source["candidate_sha256"],
            "promotion_node_id": node_id,
            "promotion_candidate_sha256": promotion_sha,
            "validation_folds": five_folds,
            "validation_mean": _mean(five_folds),
        }
        eligible.append(selection_candidate)
        job.update({
            "status": "eligible",
            "reason": "complete five-fold validation",
            "validation_mean": selection_candidate["validation_mean"],
        })
        frozen_jobs.append(job)

    frozen_at = _utc_now()
    state["phase"] = "selection-ready"
    state["promotion"].update({
        "jobs": frozen_jobs,
        "attempts_charged": len(jobs),
        "eligible_candidates": eligible,
        "frozen": True,
        "frozen_at": frozen_at,
    })
    state["revision"] += 1
    state["updated_at"] = frozen_at
    state["history"].append({
        "event": "promotion-frozen",
        "attempts_charged": len(jobs),
        "eligible_candidates": len(eligible),
        "at": frozen_at,
    })
    return _commit_state(cell_root, state)


def _verify_baseline_unchanged(cell_root: Path, baseline: Mapping[str, Any]) -> None:
    archive = (cell_root / str(baseline["archive"])).resolve()
    result = archive / "result.json"
    if not result.is_file() or file_sha256(result) != baseline["result_sha256"]:
        raise CampaignStageError("registered baseline validation artifact changed")
    for filename, expected in baseline["sealed_fold_sha256"].items():
        path = archive / "certify" / filename
        if not path.is_file() or file_sha256(path) != expected:
            raise CampaignStageError(
                f"registered baseline sealed artifact changed: {filename}"
            )


def select_winner(cell_root: Path) -> dict[str, Any]:
    """Freeze the deterministic five-fold validation winner, test-blindly."""
    with _stage_lock(cell_root):
        return _select_winner_unlocked(cell_root)


def _select_winner_unlocked(cell_root: Path) -> dict[str, Any]:
    state = load_stage_state(cell_root)
    if state.get("winner") is not None:
        return state
    if state["phase"] != "selection-ready":
        raise CampaignStageError(
            f"winner can freeze only from selection-ready, got {state['phase']!r}"
        )
    baseline = state.get("baseline")
    if not isinstance(baseline, dict):
        raise CampaignStageError("native baseline is not registered")
    _verify_baseline_unchanged(cell_root, baseline)

    pool: list[dict[str, Any]] = [{
        "kind": "baseline",
        "candidate_id": "baseline",
        "candidate_sha256": baseline["candidate_sha256"],
        "validation_folds": baseline["validation_folds"],
        "validation_mean": float(baseline["validation_mean"]),
        "promotion_node_id": None,
    }]
    for candidate in state["promotion"].get("eligible_candidates", []):
        if [fold["fold_index"] for fold in candidate["validation_folds"]] != list(
            CERTIFICATION_FOLDS
        ):
            raise CampaignStageError(
                f"selection candidate {candidate['candidate_id']} lacks exact five-fold evidence"
            )
        recomputed = _mean(candidate["validation_folds"])
        if not math.isclose(
            recomputed, float(candidate["validation_mean"]), rel_tol=0.0, abs_tol=1e-12,
        ):
            raise CampaignStageError(
                f"selection candidate {candidate['candidate_id']} mean drifted"
            )
        pool.append({
            "kind": "searched",
            **candidate,
            "validation_mean": recomputed,
        })

    # Exact ties prefer the native baseline. Remaining ties use the stable
    # discovery node id, never filesystem iteration or completion order.
    pool.sort(key=lambda candidate: (
        -candidate["validation_mean"],
        0 if candidate["kind"] == "baseline" else 1,
        candidate["candidate_id"],
    ))
    selected = pool[0]
    selected_at = _utc_now()
    audit = [{
        "rank": rank,
        "kind": candidate["kind"],
        "candidate_id": candidate["candidate_id"],
        "candidate_sha256": candidate["candidate_sha256"],
        "promotion_node_id": candidate.get("promotion_node_id"),
        "validation_mean": candidate["validation_mean"],
    } for rank, candidate in enumerate(pool, 1)]
    winner = {
        "kind": selected["kind"],
        "candidate_id": selected["candidate_id"],
        "candidate_sha256": selected["candidate_sha256"],
        "promotion_node_id": selected.get("promotion_node_id"),
        "validation_folds": selected["validation_folds"],
        "validation_mean": selected["validation_mean"],
        "baseline_validation_mean": baseline["validation_mean"],
        "lift_over_baseline": (
            selected["validation_mean"] - float(baseline["validation_mean"])
        ),
        "selection_audit": audit,
        "selection_sha256": content_sha256(audit),
        "selected_at": selected_at,
    }
    state["winner"] = winner
    state["phase"] = "winner-frozen"
    state["revision"] += 1
    state["updated_at"] = selected_at
    state["history"].append({
        "event": "winner-frozen",
        "kind": winner["kind"],
        "candidate_id": winner["candidate_id"],
        "validation_mean": winner["validation_mean"],
        "selection_sha256": winner["selection_sha256"],
        "at": selected_at,
    })
    return _commit_state(cell_root, state)


def _same_fold_evidence(left: list[Mapping[str, Any]], right: list[Mapping[str, Any]]) -> bool:
    return content_sha256(left) == content_sha256(right)


def _searched_winner_sources(
    cell_root: Path, state: Mapping[str, Any], winner: Mapping[str, Any],
) -> dict[int, Path]:
    source = next((
        candidate for candidate in state["discovery"]["promoted_candidates"]
        if candidate["candidate_id"] == winner["candidate_id"]
    ), None)
    promoted = next((
        candidate for candidate in state["promotion"].get("eligible_candidates", [])
        if candidate["candidate_id"] == winner["candidate_id"]
    ), None)
    if source is None or promoted is None:
        raise CampaignStageError("frozen searched winner is absent from its stage ledgers")
    if (
        source["candidate_sha256"] != winner["candidate_sha256"]
        or promoted["promotion_node_id"] != winner["promotion_node_id"]
        or not _same_fold_evidence(
            promoted["validation_folds"], winner["validation_folds"],
        )
    ):
        raise CampaignStageError("frozen searched winner identity/evidence drifted")

    discovery_adir = cell_root / "automil"
    discovery_archive = (
        discovery_adir / "orchestrator" / "archive" / winner["candidate_id"]
    )
    discovery_spec_path = discovery_archive / "spec.json"
    if file_sha256(discovery_spec_path) != source["source_spec_sha256"]:
        raise CampaignStageError("winner discovery spec changed after selection")
    discovery_spec = json.loads(discovery_spec_path.read_text())
    discovery_verdict = revalidate_candidate_spec(
        load_candidate_policy(discovery_adir), discovery_spec, discovery_archive,
    ).to_dict()
    discovery_sha, _ = _candidate_identity(discovery_spec, discovery_verdict)
    if discovery_sha != winner["candidate_sha256"]:
        raise CampaignStageError("winner discovery candidate changed after selection")
    discovery_result = json.loads((discovery_archive / "result.json").read_text())
    discovery_folds = _validation_folds(
        discovery_result, STAGE_FOLDS["discovery"],
    )
    if not _same_fold_evidence(discovery_folds, source["validation_folds"]):
        raise CampaignStageError("winner discovery validation evidence changed")

    promotion_adir = cell_root / "promotion" / "automil"
    promotion_archive = (
        promotion_adir / "orchestrator" / "archive"
        / winner["promotion_node_id"]
    )
    promotion_spec = json.loads((promotion_archive / "spec.json").read_text())
    promotion_verdict = revalidate_candidate_spec(
        load_candidate_policy(promotion_adir), promotion_spec, promotion_archive,
    ).to_dict()
    promotion_sha, _ = _candidate_identity(promotion_spec, promotion_verdict)
    if promotion_sha != promoted["promotion_candidate_sha256"]:
        raise CampaignStageError("winner promotion candidate changed after selection")
    promotion_result = json.loads((promotion_archive / "result.json").read_text())
    promotion_folds = _validation_folds(
        promotion_result, STAGE_FOLDS["promotion"],
    )
    five_folds = sorted(
        [*discovery_folds, *promotion_folds], key=lambda fold: fold["fold_index"],
    )
    if not _same_fold_evidence(five_folds, winner["validation_folds"]):
        raise CampaignStageError("winner five-fold validation evidence changed")
    return {
        fold: (
            discovery_archive if fold in STAGE_FOLDS["discovery"]
            else promotion_archive
        ) / "certify" / f"fold_{fold}_result.json"
        for fold in CERTIFICATION_FOLDS
    }


def _winner_sealed_sources(
    cell_root: Path, state: Mapping[str, Any], winner: Mapping[str, Any],
) -> dict[int, Path]:
    if winner["kind"] == "baseline":
        baseline = state["baseline"]
        _verify_baseline_unchanged(cell_root, baseline)
        archive = (cell_root / baseline["archive"]).resolve()
        return {
            fold: archive / "certify" / f"fold_{fold}_result.json"
            for fold in CERTIFICATION_FOLDS
        }
    return _searched_winner_sources(cell_root, state, winner)


def _read_held_out_fold(path: Path, expected_fold: int) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignStageError(
            f"cannot read selected winner sealed fold {expected_fold}: {exc}"
        ) from exc
    if payload.get("fold_index") != expected_fold:
        raise CampaignStageError(
            f"selected sealed fold identity mismatch: expected {expected_fold}"
        )
    held_out = payload.get("held_out")
    if not isinstance(held_out, dict) or not held_out:
        raise CampaignStageError(f"sealed fold {expected_fold} has no held_out metrics")
    normalized: dict[str, float] = {}
    for key, value in held_out.items():
        if "test" not in str(key).lower():
            raise CampaignStageError(
                f"sealed fold {expected_fold} contains a non-test metric {key!r}"
            )
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise CampaignStageError(
                f"sealed fold {expected_fold} metric {key!r} is not finite"
            )
        normalized[str(key)] = float(value)
    return normalized


def certify_winner(cell_root: Path) -> dict[str, Any]:
    """Reveal exactly the already-frozen winner's existing five sealed folds."""
    with _stage_lock(cell_root):
        return _certify_winner_unlocked(cell_root)


def _certify_winner_unlocked(cell_root: Path) -> dict[str, Any]:
    state = load_stage_state(cell_root)
    certification = state.get("certification")
    if certification is not None:
        bundle_path = cell_root / certification["bundle"]
        try:
            bundle = json.loads(bundle_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise CampaignStageError(f"cannot verify existing certification: {exc}") from exc
        recorded = bundle.pop("bundle_sha256", None)
        if (
            recorded != content_sha256(bundle)
            or recorded != certification["bundle_sha256"]
        ):
            raise CampaignStageError("existing certification bundle hash mismatch")
        return state
    if state["phase"] != "winner-frozen" or not isinstance(state.get("winner"), dict):
        raise CampaignStageError("certification requires an immutable validation winner")
    winner = state["winner"]
    target = cell_root / "certification"
    if target.exists():
        try:
            recovered = json.loads((target / "certify.json").read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise CampaignStageError(
                "certification directory exists without a recoverable bundle"
            ) from exc
        recorded = recovered.pop("bundle_sha256", None)
        if recorded != content_sha256(recovered):
            raise CampaignStageError("recoverable certification bundle hash mismatch")
        if (
            recovered.get("campaign_id") != CAMPAIGN_ID
            or recovered.get("cell_id") != state["cell_id"]
            or recovered.get("selection_sha256") != winner["selection_sha256"]
            or (recovered.get("winner") or {}).get("candidate_sha256")
            != winner["candidate_sha256"]
        ):
            raise CampaignStageError("certification bundle is not bound to the winner")
        recovery_sources = _winner_sealed_sources(cell_root, state, winner)
        recovery_hashes = {
            f"fold_{fold}_result.json": file_sha256(recovery_sources[fold])
            for fold in CERTIFICATION_FOLDS
        }
        if recovered.get("source_fold_sha256") != recovery_hashes:
            raise CampaignStageError(
                "certification bundle source hashes differ from the frozen winner"
            )
        return _finalize_certification_state(
            cell_root, state, bundle_sha256=recorded,
            certified_at=recovered["certified_at"],
        )
    sources = _winner_sealed_sources(cell_root, state, winner)

    held_out_folds: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    metric_keys: set[str] | None = None
    for fold in CERTIFICATION_FOLDS:
        path = sources[fold]
        metrics = _read_held_out_fold(path, fold)
        keys = set(metrics)
        if metric_keys is None:
            metric_keys = keys
        elif keys != metric_keys:
            raise CampaignStageError("held-out metric keys differ across winner folds")
        held_out_folds.append({"fold_index": fold, "held_out": metrics})
        source_hashes[f"fold_{fold}_result.json"] = file_sha256(path)
    aggregate = {
        key: math.fsum(fold["held_out"][key] for fold in held_out_folds)
        / len(held_out_folds)
        for key in sorted(metric_keys or set())
    }
    certified_at = _utc_now()
    bundle: dict[str, Any] = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "cell_id": state["cell_id"],
        "winner": {
            "kind": winner["kind"],
            "candidate_id": winner["candidate_id"],
            "candidate_sha256": winner["candidate_sha256"],
            "promotion_node_id": winner.get("promotion_node_id"),
        },
        "selection_sha256": winner["selection_sha256"],
        "validation_mean": winner["validation_mean"],
        "held_out_folds": held_out_folds,
        "held_out": aggregate,
        "source_fold_sha256": source_hashes,
        "retrained": False,
        "certified_at": certified_at,
    }
    bundle["bundle_sha256"] = content_sha256(bundle)
    temporary = Path(tempfile.mkdtemp(prefix=".certification-", dir=str(cell_root)))
    try:
        (temporary / "certify.json").write_text(
            json.dumps(bundle, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return _finalize_certification_state(
        cell_root, state, bundle_sha256=bundle["bundle_sha256"],
        certified_at=certified_at,
    )


def _finalize_certification_state(
    cell_root: Path,
    state: dict[str, Any],
    *,
    bundle_sha256: str,
    certified_at: str,
) -> dict[str, Any]:
    winner = state["winner"]
    state["certification"] = {
        "bundle": "certification/certify.json",
        "bundle_sha256": bundle_sha256,
        "certified_at": certified_at,
    }
    state["phase"] = "certified"
    state["revision"] += 1
    state["updated_at"] = certified_at
    state["history"].append({
        "event": "winner-certified",
        "candidate_id": winner["candidate_id"],
        "bundle_sha256": bundle_sha256,
        "at": certified_at,
    })
    return _commit_state(cell_root, state)
