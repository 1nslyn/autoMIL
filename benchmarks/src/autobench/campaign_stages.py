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
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Iterator, Mapping

import yaml
from dotenv import dotenv_values

from automil.admissibility import (
    AdmissibilityError,
    load_candidate_policy,
    revalidate_candidate_spec,
)
from automil.cells import (
    ActivityError,
    bind_activity_session,
    get_or_create_cell,
    read_activity_report,
    read_unbound_activity_report,
    resolve_cap_config,
    resolve_cell_identity,
)
from automil.cells.state import Cell, CellStatus, read_cell, write_cell
from automil.launch_binding import LaunchBindingError, validate_launch_binding

from autobench.campaign import (
    AGENT_PROTOCOL_FILE,
    ATTEMPT_OUTCOME_CLASSES,
    CAMPAIGN_ID,
    CERTIFICATION_FOLDS,
    DATASETS,
    DISCOVERY_ATTEMPTS,
    PROMOTION_CANDIDATES,
    PROMOTION_WALL_CLOCK_CONTAINMENT,
    PROTOCOL,
    PROTOCOL_VERSION,
    STAGE_FOLDS,
    SUBMIT_CLOCK_SKEW_TOLERANCE_SECONDS,
    classify_attempt_outcome,
    content_sha256,
    expected_promotion_sources,
    file_sha256,
    load_manifest,
    validate_agent_protocol,
)

STATE_SCHEMA_VERSION = 3
STATE_FILE = "campaign_state.json"
BASELINE_ATTESTATION_FILE = "baseline_attestation.json"
SELECTION_FREEZE_FILE = "selection_freeze.json"
CAMPAIGN_CERTIFICATION_FILE = "campaign_certification.json"
AGENT_SESSION_FILE = "agent_session.json"
CAMPAIGN_CELL_COUNT = len(DATASETS) * 26
SELECTION_FREEZE_SCHEMA_VERSION = 4


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


@contextmanager
def _campaign_lock(runtime_root: Path) -> Iterator[None]:
    """Serialize the one campaign-wide transition into held-out certification."""
    runtime_root.mkdir(parents=True, exist_ok=True)
    lock_path = runtime_root / ".selection_freeze.lock"
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
    if state.get("protocol_version") != PROTOCOL_VERSION:
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
        expected = (
            cell["cell_id"], cell["cell_sha256"], manifest_sha256,
            PROTOCOL_VERSION,
        )
        actual = (
            state.get("cell_id"), state.get("cell_sha256"),
            state.get("manifest_sha256"), state.get("protocol_version"),
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
        "protocol_version": PROTOCOL_VERSION,
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


def _import_baseline_archive(
    cell_root: Path,
    source: Path,
    expected_sha256: Mapping[str, str],
) -> Path:
    """Verify, then atomically import the exact attested baseline artifacts."""
    source = source.resolve()
    target_root = cell_root / "baseline"
    target = target_root / "archive"
    required = {
        relative: source / relative
        for relative in expected_sha256
    }
    if set(required) != {
        "result.json",
        BASELINE_ATTESTATION_FILE,
        *(f"certify/fold_{fold}_result.json" for fold in CERTIFICATION_FOLDS),
    }:
        raise CampaignStageError("baseline import artifact set is not exact")
    if not all(path.is_file() for path in required.values()):
        raise CampaignStageError("external baseline archive is incomplete")
    actual_source = {
        relative: file_sha256(path) for relative, path in required.items()
    }
    if actual_source != dict(expected_sha256):
        raise CampaignStageError("baseline artifacts differ from their attestation")
    if source == target.resolve():
        return target

    if target_root.exists():
        actual = {
            relative: file_sha256(target / relative)
            for relative in expected_sha256
            if (target / relative).is_file()
        }
        if actual != dict(expected_sha256):
            raise CampaignStageError(
                "cell-local baseline import exists with different artifact bytes"
            )
        return target

    temporary = Path(tempfile.mkdtemp(prefix=".baseline-", dir=str(cell_root)))
    temporary_archive = temporary / "archive"
    try:
        for relative, source_file in required.items():
            destination = temporary_archive / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, destination)
        copied = {
            relative: file_sha256(temporary_archive / relative)
            for relative in expected_sha256
        }
        if copied != dict(expected_sha256):
            raise CampaignStageError("baseline artifacts changed during import")
        os.replace(temporary, target_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def _expected_baseline_attestation(
    cell_root: Path,
    state: Mapping[str, Any],
    identity_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a portable baseline archive to one frozen cell and recipe."""
    try:
        cell = json.loads(
            (cell_root / "automil" / "campaign_cell.json").read_text()
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignStageError(f"cannot read baseline campaign cell: {exc}") from exc
    recorded_cell_sha256 = cell.get("cell_sha256")
    unhashed_cell = {
        key: value for key, value in cell.items() if key != "cell_sha256"
    }
    if (
        cell.get("cell_id") != state["cell_id"]
        or recorded_cell_sha256 != state["cell_sha256"]
        or recorded_cell_sha256 != content_sha256(unhashed_cell)
    ):
        raise CampaignStageError("baseline campaign cell differs from stage state")
    identity = cell.get("identity")
    expected_identity = {
        "dataset": cell.get("dataset"),
        "task": cell.get("task"),
        "encoder": cell.get("encoder"),
        "arm": cell.get("framework"),
        "seed": cell.get("seed"),
        "protocol_version": state["protocol_version"],
    }
    if (
        not isinstance(identity, dict)
        or set(identity) != set(expected_identity)
        or identity != expected_identity
    ):
        raise CampaignStageError("baseline campaign identity is invalid")
    payload: dict[str, Any] = {
        "schema_version": 2,
        "cell_id": state["cell_id"],
        "identity": identity,
        "result_sha256": identity_payload["result_sha256"],
        "sealed_fold_sha256": identity_payload["sealed_fold_sha256"],
    }
    payload["attestation_sha256"] = content_sha256(payload)
    return payload


def _write_baseline_attestation(
    cell_root: Path, state: Mapping[str, Any], baseline_archive: Path,
) -> Path:
    """Write the deterministic provenance binding for a local baseline run."""
    result = baseline_archive / "result.json"
    if not result.is_file():
        raise CampaignStageError("cannot attest a baseline without result.json")
    sealed_hashes = _sealed_fold_hashes(baseline_archive, CERTIFICATION_FOLDS)
    expected = _expected_baseline_attestation(cell_root, state, {
        "result_sha256": file_sha256(result),
        "sealed_fold_sha256": sealed_hashes,
    })
    path = baseline_archive / BASELINE_ATTESTATION_FILE
    if path.exists():
        try:
            current = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise CampaignStageError(f"cannot read baseline attestation: {exc}") from exc
        if current != expected:
            raise CampaignStageError("existing baseline attestation differs from frozen inputs")
        return path
    _atomic_write_json(path, expected)
    return path


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
            or not 0 <= float(composite) <= 1
        ):
            raise CampaignStageError(
                f"fold {fold_index} validation composite is outside [0, 1]"
            )
        metrics = raw.get("metrics")
        if (
            not isinstance(metrics, dict)
            or not metrics
            or any("test" in str(key).lower() for key in metrics)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0 <= float(value) <= 1
                for value in metrics.values()
            )
        ):
            raise CampaignStageError(
                f"fold {fold_index} metrics are not validation-only unit-interval values"
            )
        # Ordinal tasks (TCGA-HNSC grade) carry a third component. Locked as its
        # own exact key set rather than relaxed to "any subset", so a fold that
        # merely LOST a component still fails closed.
        if set(metrics) == {"val_auc", "val_bacc", "val_qwk"}:
            expected_composite = (
                float(metrics["val_auc"])
                + float(metrics["val_bacc"])
                + float(metrics["val_qwk"])
            ) / 3
        elif set(metrics) == {"val_auc", "val_bacc"}:
            expected_composite = (
                float(metrics["val_auc"]) + float(metrics["val_bacc"])
            ) / 2
        elif set(metrics) == {"val_c_index"}:
            expected_composite = float(metrics["val_c_index"])
        else:
            raise CampaignStageError(
                f"fold {fold_index} validation metric schema is not campaign-locked"
            )
        if not math.isclose(
            float(composite), expected_composite, rel_tol=0.0, abs_tol=1e-12,
        ):
            raise CampaignStageError(
                f"fold {fold_index} composite disagrees with validation metrics"
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


def _sealed_fold_hashes(
    archive: Path, expected_folds: tuple[int, ...],
) -> dict[str, str]:
    """Hash sealed folds opaquely before validation can select a candidate."""
    certify_dir = archive / "certify"
    expected_names = {
        f"fold_{fold}_result.json" for fold in expected_folds
    }
    observed_names = {
        path.name for path in certify_dir.glob("fold_*_result.json")
        if path.is_file()
    }
    if observed_names != expected_names:
        raise CampaignStageError(
            "sealed folds must be exactly "
            f"{sorted(expected_names)}, got {sorted(observed_names)}"
        )
    hashes: dict[str, str] = {}
    for fold in expected_folds:
        filename = f"fold_{fold}_result.json"
        path = certify_dir / filename
        if not path.is_file():
            raise CampaignStageError(f"candidate is missing sealed fold {fold}")
        hashes[filename] = file_sha256(path)
    return hashes


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


def attest_and_register_baseline(
    cell_root: Path, baseline_archive: Path,
) -> dict[str, Any]:
    """Attest a locally converted archive and register it under one lock.

    Historical results may be reusable even when they predate the campaign's
    portable attestation format.  Conversion code must first reconstruct the
    current validation-only/public and sealed-fold artifact contract; this
    entry point then binds those exact bytes to the current six-field cell
    identity before importing them.  It deliberately does not relax any of
    :func:`register_baseline`'s validation.
    """
    with _stage_lock(cell_root):
        state = load_stage_state(cell_root)
        _write_baseline_attestation(cell_root, state, baseline_archive)
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
    baseline_resources = _process_resource_summary([result])
    folds = _validation_folds(result, CERTIFICATION_FOLDS)
    sealed_hashes = _sealed_fold_hashes(baseline_archive, CERTIFICATION_FOLDS)
    identity_payload = {
        "result_sha256": file_sha256(result_path),
        "sealed_fold_sha256": sealed_hashes,
        "validation_folds": folds,
    }
    expected_attestation = _expected_baseline_attestation(
        cell_root, state, identity_payload,
    )
    attestation_path = baseline_archive / BASELINE_ATTESTATION_FILE
    try:
        attestation = json.loads(attestation_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignStageError(f"cannot read baseline attestation: {exc}") from exc
    if attestation != expected_attestation:
        raise CampaignStageError(
            "baseline attestation is not bound to this cell/protocol/artifact set"
        )
    import_hashes = {
        "result.json": identity_payload["result_sha256"],
        BASELINE_ATTESTATION_FILE: file_sha256(attestation_path),
        **{
            f"certify/{filename}": digest
            for filename, digest in sealed_hashes.items()
        },
    }
    _import_baseline_archive(cell_root, baseline_archive, import_hashes)
    baseline = {
        "candidate_id": "baseline",
        "candidate_sha256": content_sha256(identity_payload),
        "archive": "baseline/archive",
        "result_sha256": identity_payload["result_sha256"],
        "sealed_fold_sha256": sealed_hashes,
        "attestation_sha256": expected_attestation["attestation_sha256"],
        "validation_folds": folds,
        "validation_mean": _mean(folds),
        "result_status": result.get("status"),
        "resources": baseline_resources,
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


def run_native_baseline(
    cell_root: Path,
    *,
    repo_root: Path,
    gpu_id: int = 0,
) -> dict[str, Any]:
    """Run and register the frozen five-fold baseline outside agentic budget.

    Training executes in a detached worktree at the repository's current HEAD.
    That commit is execution metadata only and is not part of campaign identity
    or baseline reuse.
    Its public result is validation-only; full and per-fold held-out artifacts
    are born under ``baseline-execution/archive/certify`` and are parsed only
    after validation selection freezes a winner.
    """
    if gpu_id < 0:
        raise CampaignStageError("baseline gpu_id must be non-negative")
    cell_root = cell_root.resolve()
    repo_root = repo_root.resolve()
    try:
        cell_root.relative_to(repo_root)
    except ValueError as exc:
        raise CampaignStageError("campaign cell root must live inside repo_root") from exc

    run_lock_path = cell_root / ".baseline_execution.lock"
    run_lock_path.parent.mkdir(parents=True, exist_ok=True)
    with run_lock_path.open("a+") as run_lock:
        try:
            fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CampaignStageError("native baseline is already running") from exc
        try:
            state = load_stage_state(cell_root)
            if state.get("baseline") is not None:
                baseline = state["baseline"]
                if not isinstance(baseline, dict):
                    raise CampaignStageError("registered baseline state is invalid")
                _verify_baseline_unchanged(cell_root, state, baseline)
                return state
            if state["phase"] != "discovery":
                raise CampaignStageError("native baseline must run before discovery freeze")
            try:
                cell = json.loads(
                    (cell_root / "automil" / "campaign_cell.json").read_text()
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise CampaignStageError(f"cannot read campaign cell: {exc}") from exc
            if cell.get("cell_id") != state["cell_id"]:
                raise CampaignStageError("baseline cell identity differs from stage state")

            execution_archive = cell_root / "baseline-execution" / "archive"
            sealed_dir = execution_archive / "certify"
            required = [execution_archive / "result.json"] + [
                sealed_dir / f"fold_{fold}_result.json"
                for fold in CERTIFICATION_FOLDS
            ]
            if all(path.is_file() for path in required):
                _write_baseline_attestation(cell_root, state, execution_archive)
                return register_baseline(cell_root, execution_archive)
            sealed_dir.mkdir(parents=True, exist_ok=True)

            tokens = shlex.split(str(cell["commands"]["baseline"]))
            if len(tokens) < 2 or tokens[1] != "benchmarks/scripts/run_experiment.py":
                raise CampaignStageError("manifest baseline command has an invalid entrypoint")
            worktree_parent = Path(tempfile.mkdtemp(
                prefix=".baseline-worktree-", dir=str(cell_root),
            ))
            worktree = worktree_parent / "repo"
            worktree_added = False
            returncode: int | None = None
            try:
                subprocess.run(
                    [
                        "git", "worktree", "add", "--detach", str(worktree),
                        "HEAD",
                    ],
                    cwd=repo_root,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                worktree_added = True
                command = [
                    sys.executable,
                    str(worktree / "benchmarks/scripts/run_experiment.py"),
                    *tokens[2:],
                ]
                env = os.environ.copy()
                for key, value in dotenv_values(repo_root / "benchmarks/.env").items():
                    if value is not None:
                        env.setdefault(str(key), str(value))
                python_paths = [
                    str(worktree / "src"),
                    str(worktree / "benchmarks/src"),
                ]
                if env.get("PYTHONPATH"):
                    python_paths.append(env["PYTHONPATH"])
                env.update({
                    "PYTHONPATH": os.pathsep.join(python_paths),
                    "CUDA_VISIBLE_DEVICES": str(gpu_id),
                    "AUTOMIL_GPU": "0",
                    "AUTOMIL_NODE_ID": "native-baseline",
                    "AUTOMIL_RESULTS_DIR": str(sealed_dir.resolve()),
                    "AUTOMIL_FOLD_COUNT": str(len(CERTIFICATION_FOLDS)),
                })
                log_path = execution_archive / "run.log"
                with log_path.open("a") as log:
                    completed = subprocess.run(
                        command,
                        cwd=worktree,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                returncode = completed.returncode
                public_result = worktree / "result.json"
                if returncode == 0 and public_result.is_file():
                    shutil.copy2(public_result, execution_archive / "result.json")
            except (OSError, subprocess.CalledProcessError) as exc:
                raise CampaignStageError(f"cannot execute native baseline: {exc}") from exc
            finally:
                if worktree_added:
                    subprocess.run(
                        ["git", "worktree", "remove", "--force", str(worktree)],
                        cwd=repo_root,
                        check=False,
                        capture_output=True,
                    )
                shutil.rmtree(worktree_parent, ignore_errors=True)

            if returncode != 0:
                raise CampaignStageError(
                    f"native baseline exited with code {returncode}; see "
                    f"{execution_archive / 'run.log'}"
                )
            _write_baseline_attestation(cell_root, state, execution_archive)
            return register_baseline(cell_root, execution_archive)
        finally:
            fcntl.flock(run_lock.fileno(), fcntl.LOCK_UN)


def _candidate_identity(
    spec: Mapping[str, Any], verdict: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    payload = {
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


def _terminal_evidence(
    adir: Path, node_id: str, result: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Join the archived result with the trusted daemon completion record."""
    completion_path = adir / "orchestrator" / "completed" / f"{node_id}.json"
    completion: dict[str, Any] = {}
    if completion_path.is_file():
        try:
            loaded = json.loads(completion_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise CampaignStageError(
                f"terminal record for {node_id} is unreadable"
            ) from exc
        if not isinstance(loaded, dict) or loaded.get("id") != node_id:
            raise CampaignStageError(f"terminal record for {node_id} is misbound")
        completion = loaded
    result = result if isinstance(result, Mapping) else {}
    result_status = result.get("status")
    completion_status = completion.get("status")
    if (
        isinstance(result_status, str)
        and isinstance(completion_status, str)
        and result_status != completion_status
    ):
        raise CampaignStageError(f"terminal status disagrees for {node_id}")
    status = (
        result_status if isinstance(result_status, str) and result_status
        else completion_status if isinstance(completion_status, str) and completion_status
        else "missing"
    )
    raw_reason = result.get("termination_reason")
    if not isinstance(raw_reason, str) or not raw_reason.strip():
        raw_reason = completion.get("termination_reason")
    termination_reason = (
        raw_reason if isinstance(raw_reason, str) and raw_reason.strip()
        else "unspecified"
    )
    metadata = result.get("metadata")
    budget_killed = (
        status == "budget_killed"
        or isinstance(metadata, Mapping) and metadata.get("budget_killed") is True
        or completion.get("budget_killed") is True
    )
    outcome_class = classify_attempt_outcome(
        status, termination_reason, budget_killed,
    )

    evidence: dict[str, Any] = {
        "result_status": status,
        "termination_reason": termination_reason,
        "budget_killed": budget_killed,
        "outcome_class": outcome_class,
        "elapsed_seconds": None,
        "peak_vram_mb": None,
    }
    for key in ("elapsed_seconds", "peak_vram_mb"):
        candidates = [result.get(key)]
        if key == "elapsed_seconds":
            candidates.append(completion.get(key))
        else:
            completion_vram = completion.get(key)
            candidates.append(
                completion_vram
                if isinstance(completion_vram, (int, float))
                and not isinstance(completion_vram, bool)
                and float(completion_vram) > 0
                else None
            )
        for value in candidates:
            if (
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(float(value))
                and float(value) >= 0
            ):
                evidence[key] = float(value)
                break
    return evidence


def _discovery_cell(adir: Path, budget_cell_id: str):
    path = adir / "cells" / f"{budget_cell_id}.json"
    if not path.is_file():
        raise CampaignStageError("discovery budget cell has not been opened")
    return read_cell(path)


def freeze_discovery(cell_root: Path) -> dict[str, Any]:
    """Freeze up to ten complete candidates after the charged-attempt budget."""
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
    protocol_sha256 = _cell_agent_protocol_sha256(cell_root, state)
    agent_session = _agent_session_for_discovery(
        cell_root, state, protocol_sha256,
    )
    pending = _pending_stage_work(adir)
    if pending:
        raise CampaignStageError(f"discovery still has queued/running work: {pending}")
    budget_cell = _discovery_cell(adir, config["budget_identity"]["cell_id"])
    if budget_cell.eval_budget != DISCOVERY_ATTEMPTS:
        raise CampaignStageError(
            f"discovery cell does not carry the frozen {DISCOVERY_ATTEMPTS}-attempt cap"
        )
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
        submitted_at = spec.get("submitted_at")
        try:
            submitted = datetime.fromisoformat(str(submitted_at))
        except ValueError as exc:
            raise CampaignStageError(
                f"discovery spec {archive.name} has an invalid submitted_at"
            ) from exc
        if submitted.tzinfo is None:
            raise CampaignStageError(
                f"discovery spec {archive.name} submitted_at lacks a timezone"
            )
        session_bound = datetime.fromisoformat(
            agent_session["session"]["bound_at"]
        )
        # C-j (claims-alignment): submit host != controller host; NTP-level
        # skew around the first submit must not brick the cell forever. The
        # tolerance is declared in PROTOCOL (hash-locked); beyond it, fail
        # closed as before.
        skew = timedelta(seconds=SUBMIT_CLOCK_SKEW_TOLERANCE_SECONDS)
        if submitted < session_bound - skew:
            raise CampaignStageError(
                f"discovery spec {archive.name} predates controller session "
                f"binding by more than the declared "
                f"{SUBMIT_CLOCK_SKEW_TOLERANCE_SECONDS}s clock-skew tolerance"
            )
        spec_session = (spec.get("metadata") or {}).get("agent_session")
        expected_session = {
            "session_id": agent_session["session"]["session_id"],
            "agent_protocol_sha256": protocol_sha256,
            "binding_sha256": agent_session["binding_sha256"],
        }
        if spec_session != expected_session:
            raise CampaignStageError(
                f"discovery spec {archive.name} is outside the pre-bound agent session"
            )
        audit: dict[str, Any] = {
            "node_id": archive.name,
            "source_spec_sha256": file_sha256(archive / "spec.json"),
            "submitted_at": submitted_at,
            "agent_session_id": expected_session["session_id"],
            "agent_session_binding_sha256": expected_session["binding_sha256"],
            "candidate_class": "inadmissible",
            "policy_hash": None,
            "result_status": "missing",
            "termination_reason": "unspecified",
            "budget_killed": False,
            "outcome_class": "missing-result",
            "elapsed_seconds": None,
            "peak_vram_mb": None,
            "eligible": False,
            "reason": "missing result.json",
            "candidate_sha256": None,
            "validation_mean": None,
        }
        audit.update(_terminal_evidence(adir, archive.name, None))
        verdict: dict[str, Any] | None = None
        try:
            verdict = revalidate_candidate_spec(policy, spec, archive).to_dict()
            audit["candidate_class"] = verdict["candidate_class"]
            audit["policy_hash"] = verdict["policy_hash"]
        except (AdmissibilityError, OSError, KeyError, TypeError, ValueError) as exc:
            audit["reason"] = str(exc)
        result_path = archive / "result.json"
        if result_path.is_file():
            try:
                result = json.loads(result_path.read_text())
                audit.update(_terminal_evidence(adir, archive.name, result))
                if verdict is None:
                    raise CampaignStageError(audit["reason"])
                folds = _validation_folds(result, STAGE_FOLDS["discovery"])
                candidate_sha, identity = _candidate_identity(spec, verdict)
                # Per-fold prediction hashes are the byte discriminator the
                # outcome dedup below keys on. They live beside — never
                # inside — the normalized fold entries: winner verification
                # re-derives those entries from the archive and compares
                # them by content hash, so their shape must stay stable.
                raw_hash_by_fold = {
                    raw_fold["fold_index"]: (
                        raw_fold.get("val_predictions_sha256")
                        if _is_sha256(raw_fold.get("val_predictions_sha256"))
                        else None
                    )
                    for raw_fold in result["validation_folds"]
                }
                candidate = {
                    "candidate_id": archive.name,
                    "candidate_sha256": candidate_sha,
                    "source_spec_sha256": file_sha256(archive / "spec.json"),
                    "identity": identity,
                    "validation_folds": folds,
                    "val_predictions_sha256": [
                        raw_hash_by_fold.get(fold["fold_index"])
                        for fold in folds
                    ],
                    "discovery_mean": _mean(folds),
                    "sealed_fold_sha256": _sealed_fold_hashes(
                        archive, STAGE_FOLDS["discovery"],
                    ),
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
    seen_hash_vectors: set[tuple[str, ...]] = set()
    seen_outcomes: set[tuple] = set()
    for candidate in eligible:
        identity = candidate["candidate_sha256"]
        if identity in seen_identities:
            continue
        # Outcome identity: runs are bit-deterministic under the locked seed,
        # so two candidates that produced the SAME validation predictions are
        # the same measurement wearing different configs (uni_v2 canary: a
        # weight-decay value inside the logit-scaling invariant regime
        # reproduced its parent to 16 digits and occupied a second promotion
        # slot). Promotion re-runs byte-copies on folds 3/4; re-measuring an
        # identical run twice buys zero information, so the slot goes to the
        # next distinct config. The discriminator is the per-fold
        # val_predictions_sha256 vector when both candidates carry a complete
        # one — quantized composites can tie for genuinely different configs,
        # and dropping those would silently lose a distinct candidate. Only
        # hashless artifacts (pre-hash cells, e.g. the live canaries) fall
        # back to the legacy composite-tuple rule, and a hash-bearing
        # candidate never dedups against a hashless one.
        hashes = candidate["val_predictions_sha256"]
        if hashes and all(isinstance(value, str) for value in hashes):
            vector = tuple(hashes)
            if vector in seen_hash_vectors:
                continue
            seen_hash_vectors.add(vector)
        else:
            outcome = tuple(
                (fold["fold_index"], fold["composite"])
                for fold in candidate["validation_folds"]
            )
            if outcome in seen_outcomes:
                continue
            seen_outcomes.add(outcome)
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
        # Promotion has no coding-agent session. Its time wall therefore uses
        # exact elapsed wall time while the eval axis controls candidate count.
        mode="wall_clock",
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
    if (
        not state["promotion"]["materialized"]
        and state["phase"] != "promotion-ready"
    ):
        raise CampaignStageError(
            f"promotion can materialize only from promotion-ready, got {state['phase']!r}"
        )
    _require_closed_discovery_activity(cell_root, state)
    if state["promotion"]["materialized"]:
        return state
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
        # The deep-copied discovery config carries the 12h agent-active
        # budget; promotion has no agent, so that number would be a wall-clock
        # kill switch mid-evaluation. Containment-size the time wall instead.
        config["cap"]["budget"] = PROMOTION_WALL_CLOCK_CONTAINMENT
        config["cap"]["mode"] = "wall_clock"
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
            ".activity.jsonl\n.activity.samples.json\n.activity.lock\n"
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
                    "\n".join(config_parts).encode()
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


def _require_closed_discovery_activity(
    cell_root: Path, state: Mapping[str, Any],
) -> None:
    """Require the one bound discovery session to have durable end evidence."""
    adir = cell_root / "automil"
    protocol_sha256 = _cell_agent_protocol_sha256(cell_root, state)
    try:
        session_payload = json.loads((cell_root / AGENT_SESSION_FILE).read_text())
        session = _validate_agent_session(
            session_payload,
            state=state,
            agent_protocol_sha256=protocol_sha256,
        )
        config = yaml.safe_load((adir / "config.yaml").read_text()) or {}
        identity = resolve_cell_identity(config)
        campaign_cell = json.loads((adir / "campaign_cell.json").read_text())
        budget_cell_id = str(campaign_cell["budget_identity"]["cell_id"])
        if (
            identity.cell_id != budget_cell_id
            or (config.get("campaign") or {}).get("budget_cell_id")
            != budget_cell_id
        ):
            raise CampaignStageError(
                "discovery activity budget identity differs from the frozen cell"
            )
        activity = read_activity_report(adir, budget_cell_id)
    except CampaignStageError:
        raise
    except ActivityError as exc:
        raise CampaignStageError(
            "SessionEnd and a durable final Claude active-time sample are "
            f"required before leaving discovery: {exc}"
        ) from exc
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        yaml.YAMLError,
    ) as exc:
        raise CampaignStageError(
            "discovery activity accounting is invalid: " + str(exc)
        ) from exc

    expected_session_id = session["session"]["session_id"]
    if activity.sessions != (expected_session_id,):
        raise CampaignStageError(
            "discovery activity journal is not exclusive to the bound agent session"
        )
    if activity.bindings != (
        (expected_session_id, session["binding_sha256"]),
    ):
        raise CampaignStageError(
            "discovery activity binding differs from agent_session.json"
        )
    if (
        activity.open_sessions
        or activity.ended_sessions != (expected_session_id,)
        or not activity.complete
    ):
        raise CampaignStageError(
            "SessionEnd and a durable final Claude active-time sample are required "
            "before promotion or winner selection"
        )


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
        job.update({
            "candidate_class": source["identity"]["candidate_class"],
            "policy_hash": source["identity"]["policy_hash"],
            "result_status": "missing",
            "source_spec_sha256": source["source_spec_sha256"],
            "promotion_spec_sha256": None,
            "submitted_at": None,
            "elapsed_seconds": None,
            "peak_vram_mb": None,
            "validation_mean": None,
        })
        node_id = job["promotion_node_id"]
        archive = adir / "orchestrator" / "archive" / node_id
        job.update(_terminal_evidence(adir, node_id, None))
        spec_path = archive / "spec.json"
        result_path = archive / "result.json"
        if not spec_path.is_file():
            raise CampaignStageError(
                f"promotion job {node_id} lost its durable spec"
            )
        try:
            spec = json.loads(spec_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise CampaignStageError(
                f"promotion spec is unreadable for {node_id}: {exc}"
            ) from exc
        job["promotion_spec_sha256"] = file_sha256(spec_path)
        job["submitted_at"] = spec.get("submitted_at")
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
        if not result_path.is_file():
            job.update({
                "status": "ineligible",
                "reason": "terminal promotion is missing result.json",
            })
            frozen_jobs.append(job)
            continue
        try:
            result = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            job.update({
                "status": "ineligible",
                "reason": f"terminal promotion result is unreadable: {exc}",
            })
            frozen_jobs.append(job)
            continue
        job.update(_terminal_evidence(adir, node_id, result))
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
            promotion_sealed = _sealed_fold_hashes(
                archive, STAGE_FOLDS["promotion"],
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
            "sealed_fold_sha256": {
                **source["sealed_fold_sha256"],
                **promotion_sealed,
            },
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


def _verify_baseline_unchanged(
    cell_root: Path,
    state: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> None:
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
    observed = {
        path.name for path in (archive / "certify").glob("fold_*_result.json")
        if path.is_file()
    }
    if observed != set(baseline["sealed_fold_sha256"]):
        raise CampaignStageError("registered baseline sealed fold set changed")
    attestation_path = archive / BASELINE_ATTESTATION_FILE
    try:
        attestation = json.loads(attestation_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignStageError("registered baseline attestation changed") from exc
    recorded = attestation.get("attestation_sha256")
    payload = {
        key: value for key, value in attestation.items()
        if key != "attestation_sha256"
    }
    if (
        recorded != baseline.get("attestation_sha256")
        or recorded != content_sha256(payload)
    ):
        raise CampaignStageError("registered baseline attestation changed")
    expected = _expected_baseline_attestation(cell_root, state, {
        "result_sha256": baseline["result_sha256"],
        "sealed_fold_sha256": baseline["sealed_fold_sha256"],
    })
    if attestation != expected:
        raise CampaignStageError(
            "registered baseline attestation is not bound to the current cell"
        )


def select_winner(cell_root: Path) -> dict[str, Any]:
    """Freeze the deterministic five-fold validation winner, test-blindly."""
    with _stage_lock(cell_root):
        return _select_winner_unlocked(cell_root)


def _select_winner_unlocked(cell_root: Path) -> dict[str, Any]:
    state = load_stage_state(cell_root)
    if state.get("winner") is None and state["phase"] != "selection-ready":
        raise CampaignStageError(
            f"winner can freeze only from selection-ready, got {state['phase']!r}"
        )
    _require_closed_discovery_activity(cell_root, state)
    if state.get("winner") is not None:
        return state
    baseline = state.get("baseline")
    if not isinstance(baseline, dict):
        raise CampaignStageError("native baseline is not registered")
    _verify_baseline_unchanged(cell_root, state, baseline)

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
        "sealed_fold_sha256": selected.get("sealed_fold_sha256"),
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
    sources = {
        fold: (
            discovery_archive if fold in STAGE_FOLDS["discovery"]
            else promotion_archive
        ) / "certify" / f"fold_{fold}_result.json"
        for fold in CERTIFICATION_FOLDS
    }
    expected_hashes = winner.get("sealed_fold_sha256")
    if not isinstance(expected_hashes, dict):
        raise CampaignStageError("frozen searched winner lacks sealed-fold hashes")
    for fold, path in sources.items():
        filename = f"fold_{fold}_result.json"
        if (
            not path.is_file()
            or expected_hashes.get(filename) != file_sha256(path)
        ):
            raise CampaignStageError(
                f"frozen searched winner sealed artifact changed: {filename}"
            )
    return sources


def _winner_sealed_sources(
    cell_root: Path, state: Mapping[str, Any], winner: Mapping[str, Any],
) -> dict[int, Path]:
    if winner["kind"] == "baseline":
        baseline = state.get("baseline")
        if not isinstance(baseline, dict):
            raise CampaignStageError("native baseline is not registered")
        _verify_baseline_unchanged(cell_root, state, baseline)
        archive = (cell_root / baseline["archive"]).resolve()
        return {
            fold: archive / "certify" / f"fold_{fold}_result.json"
            for fold in CERTIFICATION_FOLDS
        }
    return _searched_winner_sources(cell_root, state, winner)


def _baseline_sealed_sources(
    cell_root: Path, state: Mapping[str, Any],
) -> dict[int, Path]:
    baseline = state.get("baseline")
    if not isinstance(baseline, dict):
        raise CampaignStageError("native baseline is not registered")
    _verify_baseline_unchanged(cell_root, state, baseline)
    archive = (cell_root / str(baseline["archive"])).resolve()
    return {
        fold: archive / "certify" / f"fold_{fold}_result.json"
        for fold in CERTIFICATION_FOLDS
    }


def _source_fold_anchors(
    runtime_root: Path, sources: Mapping[int, Path],
) -> dict[str, dict[str, str]]:
    root = runtime_root.resolve()
    if set(sources) != set(CERTIFICATION_FOLDS):
        raise CampaignStageError("source fold roster differs from 0..4")
    anchors: dict[str, dict[str, str]] = {}
    for fold in CERTIFICATION_FOLDS:
        path = sources[fold].resolve()
        if not path.is_file() or not path.is_relative_to(root):
            raise CampaignStageError("source fold path escapes the campaign runtime")
        filename = f"fold_{fold}_result.json"
        anchors[filename] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": file_sha256(path),
        }
    return anchors


def _validate_source_fold_anchors_artifact(
    raw: object, *, label: str, expected_cell_id: str | None = None,
) -> dict[str, dict[str, str]]:
    expected = {f"fold_{fold}_result.json" for fold in CERTIFICATION_FOLDS}
    if not isinstance(raw, dict) or set(raw) != expected:
        raise CampaignStageError(f"{label} source-fold roster is invalid")
    normalized: dict[str, dict[str, str]] = {}
    paths: set[str] = set()
    for filename in sorted(expected):
        record = raw[filename]
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise CampaignStageError(f"{label} source-fold record is invalid")
        path = record.get("path")
        relative = PurePosixPath(str(path))
        if (
            not isinstance(path, str)
            or not path
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != path
            or relative.name != filename
            or (
                expected_cell_id is not None
                and (
                    not relative.parts
                    or relative.parts[0] != expected_cell_id
                )
            )
            or path in paths
            or not _is_sha256(record.get("sha256"))
        ):
            raise CampaignStageError(f"{label} source-fold anchor is invalid")
        paths.add(path)
        normalized[filename] = {
            "path": path,
            "sha256": str(record["sha256"]),
        }
    return normalized


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


def _read_certification_evidence(
    sources: Mapping[int, Path],
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, str]]:
    """Read one pre-frozen five-fold sealed evidence set exactly once."""
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
            raise CampaignStageError("held-out metric keys differ across folds")
        held_out_folds.append({"fold_index": fold, "held_out": metrics})
        source_hashes[f"fold_{fold}_result.json"] = file_sha256(path)
    aggregate = {
        key: math.fsum(fold["held_out"][key] for fold in held_out_folds)
        / len(held_out_folds)
        for key in sorted(metric_keys or set())
    }
    return held_out_folds, aggregate, source_hashes


def _locked_agent_protocol(runtime_root: Path) -> tuple[dict[str, Any], str]:
    path = runtime_root / AGENT_PROTOCOL_FILE
    try:
        raw = json.loads(path.read_text())
        protocol = validate_agent_protocol(raw)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise CampaignStageError(f"cannot verify locked agent protocol: {exc}") from exc
    return protocol, content_sha256(protocol)


def validate_agent_usage_artifact(usage: object) -> dict[str, Any]:
    """Validate the exact agent-usage schema shared by freeze and analysis."""
    usage_fields = {
        "status", "input_tokens", "output_tokens", "cached_input_tokens",
        "cost_usd", "basis",
    }
    if not isinstance(usage, dict) or set(usage) != usage_fields:
        raise CampaignStageError("agent session usage field set is not exact")
    if usage.get("status") not in {"exact", "estimated", "unavailable"}:
        raise CampaignStageError("agent session usage status is invalid")
    if not isinstance(usage.get("basis"), str) or not usage["basis"].strip():
        raise CampaignStageError("agent session usage basis is required")
    numeric_fields = ("input_tokens", "output_tokens", "cached_input_tokens")
    for key in numeric_fields:
        value = usage.get(key)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise CampaignStageError(f"agent session usage {key} is invalid")
    cost = usage.get("cost_usd")
    if cost is not None and (
        isinstance(cost, bool)
        or not isinstance(cost, (int, float))
        or not math.isfinite(float(cost))
        or float(cost) < 0
    ):
        raise CampaignStageError("agent session usage cost_usd is invalid")
    if usage["status"] == "unavailable":
        if any(usage[key] is not None for key in (*numeric_fields, "cost_usd")):
            raise CampaignStageError("unavailable usage must use null numeric fields")
    elif usage["input_tokens"] is None or usage["output_tokens"] is None:
        raise CampaignStageError("reported usage requires input and output tokens")
    return json.loads(json.dumps(usage))


def _session_binding_payload(
    *, state: Mapping[str, Any], protocol_sha256: str,
    session_id: str, started_at: str, bound_at: str,
) -> dict[str, Any]:
    return {
        "campaign_id": CAMPAIGN_ID,
        "cell_id": state.get("cell_id"),
        "agent_protocol_sha256": protocol_sha256,
        "session_id": session_id,
        "started_at": started_at,
        "bound_at": bound_at,
    }


def _validate_agent_session(
    payload: object, *, state: Mapping[str, Any],
    agent_protocol_sha256: str, require_finalized: bool = False,
) -> dict[str, Any]:
    try:
        launch = validate_launch_binding(
            payload,
            campaign_id=CAMPAIGN_ID,
            cell_id=str(state.get("cell_id")),
            agent_protocol_sha256=agent_protocol_sha256,
            require_open=(
                isinstance(payload, Mapping) and payload.get("status") == "open"
            ),
        )
    except LaunchBindingError as exc:
        raise CampaignStageError(str(exc)) from exc
    if not isinstance(payload, Mapping):
        raise CampaignStageError("agent session record is invalid")
    session = payload.get("session")
    bound = datetime.fromisoformat(launch["bound_at"])
    if not isinstance(session, Mapping):
        raise CampaignStageError("agent session record is invalid")
    status = payload.get("status")
    if status == "open":
        if require_finalized:
            raise CampaignStageError("agent session has not been finalized")
        if (
            session.get("ended_at") is not None
            or session.get("termination_reason") is not None
            or session.get("usage") is not None
            or session.get("activity") is not None
            or payload.get("attestation_sha256") is not None
        ):
            raise CampaignStageError("open agent session contains post-session fields")
    elif status == "finalized":
        reason = session.get("termination_reason")
        if not isinstance(reason, str) or not reason.strip():
            raise CampaignStageError("agent session termination_reason is invalid")
        try:
            ended = datetime.fromisoformat(str(session.get("ended_at")))
        except ValueError as exc:
            raise CampaignStageError("agent session ended_at is invalid") from exc
        if ended.tzinfo is None or ended < bound:
            raise CampaignStageError("agent session time interval is invalid")
        validate_agent_usage_artifact(session.get("usage"))
        activity = session.get("activity")
        if not isinstance(activity, Mapping) or set(activity) != {
            "source", "active_seconds", "event_count", "sha256",
        }:
            raise CampaignStageError("agent session activity attestation is invalid")
        if activity.get("source") != "claude-native-active-time-v1":
            raise CampaignStageError("agent session activity source is invalid")
        if (
            not isinstance(activity.get("active_seconds"), (int, float))
            or isinstance(activity.get("active_seconds"), bool)
            or not math.isfinite(float(activity["active_seconds"]))
            or float(activity["active_seconds"]) < 0
            or not isinstance(activity.get("event_count"), int)
            or isinstance(activity.get("event_count"), bool)
            or int(activity["event_count"]) < 3
            or not isinstance(activity.get("sha256"), str)
            or len(str(activity["sha256"])) != 64
            or any(char not in "0123456789abcdef" for char in str(activity["sha256"]))
        ):
            raise CampaignStageError("agent session activity attestation is invalid")
        canonical = {
            key: value for key, value in payload.items()
            if key != "attestation_sha256"
        }
        if payload.get("attestation_sha256") != content_sha256(canonical):
            raise CampaignStageError("agent session attestation hash mismatch")
    else:
        raise CampaignStageError("agent session status is invalid")
    return json.loads(json.dumps(payload))


def _cell_agent_protocol_sha256(
    cell_root: Path, state: Mapping[str, Any],
) -> str:
    _, protocol_sha256 = _locked_agent_protocol(cell_root.parent)
    try:
        config = yaml.safe_load((cell_root / "automil/config.yaml").read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise CampaignStageError("cell config is unreadable") from exc
    if (config.get("campaign") or {}).get(
        "agent_protocol_sha256"
    ) != protocol_sha256:
        raise CampaignStageError("cell config agent protocol binding mismatch")
    if state.get("campaign_id") != CAMPAIGN_ID:
        raise CampaignStageError("cell campaign identity mismatch")
    return protocol_sha256


def _bind_campaign_activity_and_open_cell(
    cell_root: Path,
    *,
    cell: Mapping[str, Any],
    session_id: str,
    binding_sha256: str,
) -> None:
    """Idempotently join the runtime session, journal, and budget cell."""
    adir = cell_root / "automil"
    try:
        config = yaml.safe_load((adir / "config.yaml").read_text()) or {}
        identity = resolve_cell_identity(config)
        cap = resolve_cap_config(config)
        budget_identity = cell["budget_identity"]
        expected_identity = {
            "dataset": identity.dataset,
            "task": identity.task,
            "encoder": identity.encoder,
            "mil_model": identity.mil_model,
            "cell_id": identity.cell_id,
        }
        if budget_identity != expected_identity:
            raise CampaignStageError(
                "campaign budget identity differs from config.yaml"
            )
        report = read_activity_report(adir, identity.cell_id)
        unbound_report = read_unbound_activity_report(adir)
        if report.sessions == (session_id,):
            # Idempotent retry after the immutable binding was persisted.
            if unbound_report.sessions:
                raise CampaignStageError(
                    "another unbound SessionStart exists for this project"
                )
        elif not report.sessions:
            if unbound_report.sessions != (session_id,):
                raise CampaignStageError(
                    "SessionStart was not recorded exclusively for this campaign cell"
                )
            if unbound_report.metered_sessions != (session_id,):
                raise CampaignStageError(
                    "Claude active-time metrics were not recorded for this session"
                )
        else:
            raise CampaignStageError(
                "SessionStart was not recorded exclusively for this campaign cell"
            )
        if report.sessions and report.metered_sessions != (session_id,):
            raise CampaignStageError(
                "Claude active-time metrics were not recorded for this session"
            )
        allowed_bindings = ((), ((session_id, binding_sha256),))
        if report.bindings not in allowed_bindings:
            raise CampaignStageError(
                "activity journal has a conflicting launch binding"
            )
        bind_activity_session(
            adir, identity.cell_id, session_id, binding_sha256,
        )
        bound_report = read_activity_report(adir, identity.cell_id)
        if bound_report.bindings != ((session_id, binding_sha256),):
            raise CampaignStageError("activity journal binding was not persisted")
        budget_cell = get_or_create_cell(
            dataset=identity.dataset,
            encoder=identity.encoder,
            mil_model=identity.mil_model,
            task=identity.task,
            budget_seconds=cap.budget_seconds,
            safety_buffer_seconds=cap.safety_buffer_seconds,
            mode=cap.mode,
            eval_budget=cap.eval_budget,
            cells_dir=adir / "cells",
        )
    except CampaignStageError:
        raise
    except (ActivityError, KeyError, OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise CampaignStageError(
            f"cannot bind campaign activity accounting: {exc}"
        ) from exc

    expected_cap = (
        identity.cell_id,
        cap.budget_seconds,
        cap.safety_buffer_seconds,
        cap.mode,
        cap.eval_budget,
    )
    actual_cap = (
        budget_cell.cell_id,
        budget_cell.budget_seconds,
        budget_cell.safety_buffer_seconds,
        budget_cell.mode,
        budget_cell.eval_budget,
    )
    if actual_cap != expected_cap:
        raise CampaignStageError("persisted campaign budget cell has drifted")


def open_agent_session(
    cell_root: Path, session_start: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the only coding-agent session before its first proposal."""
    with _campaign_lock(cell_root.parent), _stage_lock(cell_root):
        state = load_stage_state(cell_root)
        if state.get("phase") != "discovery":
            raise CampaignStageError("agent session must open during discovery")
        # B7 (claims-alignment): the baseline is the incumbent AND the only
        # fail-closed data preflight (TITAN's missing features surface here,
        # not as 30 charged crashes). Ordering was runbook-only: a session
        # could open, burn attempts with no incumbent, and a reconcile-created
        # graph would then brick baseline registration permanently.
        if not state.get("baseline"):
            raise CampaignStageError(
                "agent session requires a registered native baseline — run "
                "`campaign_stage.py run-baseline` (or register-baseline) first"
            )
        target = cell_root / AGENT_SESSION_FILE
        adir = cell_root / "automil"
        if not isinstance(session_start, Mapping) or set(session_start) != {
            "session_id", "started_at",
        }:
            raise CampaignStageError("agent session start field set is not exact")
        session_id = session_start.get("session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            raise CampaignStageError("agent session session_id is invalid")
        protocol_sha256 = _cell_agent_protocol_sha256(cell_root, state)
        cell = json.loads((adir / "campaign_cell.json").read_text())
        if target.exists():
            try:
                current = _validate_agent_session(
                    json.loads(target.read_text()),
                    state=state,
                    agent_protocol_sha256=protocol_sha256,
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise CampaignStageError("agent session record is unreadable") from exc
            expected_start = {
                "session_id": current["session"]["session_id"],
                "started_at": current["session"]["started_at"],
            }
            if json.loads(json.dumps(session_start)) != expected_start:
                raise CampaignStageError("agent session opening is immutable")
            _bind_campaign_activity_and_open_cell(
                cell_root,
                cell=cell,
                session_id=session_id,
                binding_sha256=current["binding_sha256"],
            )
            return current
        durable_specs = [
            *adir.glob("orchestrator/queue/*.json"),
            *adir.glob("orchestrator/running/**/*.json"),
            *adir.glob("orchestrator/archive/*/spec.json"),
        ]
        if durable_specs:
            raise CampaignStageError("agent session must precede the first proposal")
        graph_path = adir / "graph.json"
        if graph_path.is_file():
            try:
                graph = json.loads(graph_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise CampaignStageError("discovery graph is unreadable") from exc
            nodes = graph.get("nodes")
            allowed = {
                (state.get("baseline") or {}).get("discovery_root_node_id")
            } - {None}
            if not isinstance(nodes, dict) or set(nodes) != allowed:
                raise CampaignStageError(
                    "agent session must precede every non-baseline proposal"
                )
        policy_dir = adir / "variants" / "_policies"
        if policy_dir.exists() and any(
            path.name != ".gitkeep" for path in policy_dir.iterdir()
        ):
            raise CampaignStageError(
                "agent session must precede every candidate policy file"
            )
        expected_plan = f"# Discovery plan — {state['cell_id']}\n\nNo proposals queued yet.\n"
        expected_learnings = f"# Cell-local learnings — {state['cell_id']}\n"
        for path, expected in (
            (adir / "plan.md", expected_plan),
            (adir / "learnings.md", expected_learnings),
        ):
            if path.is_file() and path.read_text() != expected:
                raise CampaignStageError(
                    "agent session must precede proposal planning and learnings"
                )
        budget_path = adir / "cells" / f"{cell['budget_identity']['cell_id']}.json"
        if budget_path.is_file() and read_cell(budget_path).consumed_evals != 0:
            raise CampaignStageError("agent session must precede budget consumption")
        for sibling in cell_root.parent.iterdir():
            sibling_session = sibling / AGENT_SESSION_FILE
            if sibling == cell_root or not sibling_session.is_file():
                continue
            try:
                existing = json.loads(sibling_session.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise CampaignStageError(
                    f"cannot verify session reservation under {sibling.name}"
                ) from exc
            if not isinstance(existing, Mapping):
                raise CampaignStageError(
                    f"cannot verify session reservation under {sibling.name}"
                )
            existing_session = existing.get("session")
            if not isinstance(existing_session, Mapping):
                raise CampaignStageError(
                    f"cannot verify session reservation under {sibling.name}"
                )
            if existing_session.get("session_id") == session_id:
                raise CampaignStageError(
                    "agent session_id is already reserved by another cell"
                )
        started_at = session_start.get("started_at")
        bound_at = _utc_now()
        payload: dict[str, Any] = {
            "schema_version": 3,
            "campaign_id": CAMPAIGN_ID,
            "cell_id": state["cell_id"],
            "agent_protocol_sha256": protocol_sha256,
            "status": "open",
            "session": {
                "session_id": session_id,
                "started_at": started_at,
                "bound_at": bound_at,
                "ended_at": None,
                "termination_reason": None,
                "usage": None,
                "activity": None,
            },
            "binding_sha256": content_sha256(_session_binding_payload(
                state=state, protocol_sha256=protocol_sha256,
                session_id=str(session_id), started_at=str(started_at),
                bound_at=bound_at,
            )),
            "attestation_sha256": None,
        }
        validated = _validate_agent_session(
            payload, state=state, agent_protocol_sha256=protocol_sha256,
        )
        _atomic_write_json(target, validated)
        _bind_campaign_activity_and_open_cell(
            cell_root,
            cell=cell,
            session_id=session_id,
            binding_sha256=validated["binding_sha256"],
        )
        return validated


def finalize_agent_session(
    cell_root: Path, session_end: Mapping[str, Any],
) -> dict[str, Any]:
    """Close the pre-bound session and attach its resource attestation."""
    with _stage_lock(cell_root):
        state = load_stage_state(cell_root)
        if state.get("phase") != "winner-frozen":
            raise CampaignStageError("agent session finalization requires a frozen winner")
        protocol_sha256 = _cell_agent_protocol_sha256(cell_root, state)
        target = cell_root / AGENT_SESSION_FILE
        try:
            current = json.loads(target.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise CampaignStageError("pre-bound agent session is missing") from exc
        current = _validate_agent_session(
            current, state=state, agent_protocol_sha256=protocol_sha256,
        )
        required = {"session_id", "ended_at", "termination_reason", "usage"}
        if not isinstance(session_end, Mapping) or set(session_end) != required:
            raise CampaignStageError("agent session end field set is not exact")
        if session_end.get("session_id") != current["session"]["session_id"]:
            raise CampaignStageError("agent session end uses a different session_id")
        if current["status"] == "finalized":
            expected = {
                "session_id": current["session"]["session_id"],
                "ended_at": current["session"]["ended_at"],
                "termination_reason": current["session"]["termination_reason"],
                "usage": current["session"]["usage"],
            }
            if json.loads(json.dumps(session_end)) != expected:
                raise CampaignStageError("agent session finalization is immutable")
            return current
        try:
            cell = json.loads(
                (cell_root / "automil/campaign_cell.json").read_text()
            )
            budget_cell_id = str(cell["budget_identity"]["cell_id"])
            activity = read_activity_report(
                cell_root / "automil", budget_cell_id,
            )
        except (ActivityError, KeyError, OSError, TypeError, json.JSONDecodeError) as exc:
            raise CampaignStageError(
                f"agent session activity accounting is invalid: {exc}"
            ) from exc
        expected_session_id = current["session"]["session_id"]
        if activity.sessions != (expected_session_id,):
            raise CampaignStageError(
                "agent session activity journal is not exclusive to this session"
            )
        if activity.bindings != (
            (expected_session_id, current["binding_sha256"]),
        ):
            raise CampaignStageError(
                "agent session activity binding differs from agent_session.json"
            )
        if not activity.complete:
            raise CampaignStageError(
                "SessionEnd and a final Claude active-time sample must be recorded "
                "before finalization"
            )
        audits = state.get("discovery", {}).get("attempt_audit")
        if not isinstance(audits, list) or len(audits) != DISCOVERY_ATTEMPTS:
            raise CampaignStageError(
                f"agent session lacks the exact {DISCOVERY_ATTEMPTS}-proposal audit"
            )
        if any(
            not isinstance(row, dict)
            or row.get("agent_session_binding_sha256") != current["binding_sha256"]
            or row.get("agent_session_id") != current["session"]["session_id"]
            for row in audits
        ):
            raise CampaignStageError("a discovery proposal is outside the bound session")
        try:
            ended = datetime.fromisoformat(str(session_end.get("ended_at")))
        except ValueError as exc:
            raise CampaignStageError("agent session ended_at is invalid") from exc
        if ended.tzinfo is None:
            raise CampaignStageError("agent session ended_at must include a timezone")
        for row in audits:
            try:
                submitted = datetime.fromisoformat(str(row["submitted_at"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise CampaignStageError(
                    "agent session audit contains an invalid submission timestamp"
                ) from exc
            # (No skew tolerance on the upper bound, deliberately: ended_at is
            # operator-supplied at finalization time and can simply be set at
            # or after the last proposal; only the controller-stamped bound_at
            # races the first submit across hosts — C-j.)
            if submitted.tzinfo is None or submitted > ended:
                raise CampaignStageError(
                    "a discovery proposal falls outside the agent session interval"
                )
        prepared = json.loads(json.dumps(current))
        prepared["status"] = "finalized"
        prepared["session"].update({
            "ended_at": session_end["ended_at"],
            "termination_reason": session_end["termination_reason"],
            "usage": session_end["usage"],
            "activity": {
                "source": "claude-native-active-time-v1",
                "active_seconds": activity.active_seconds,
                "event_count": activity.event_count,
                "sha256": activity.sha256,
            },
        })
        prepared["attestation_sha256"] = content_sha256({
            key: value for key, value in prepared.items()
            if key != "attestation_sha256"
        })
        validated = _validate_agent_session(
            prepared, state=state, agent_protocol_sha256=protocol_sha256,
            require_finalized=True,
        )
        _atomic_write_json(target, validated)
        return validated


def _agent_session_for_discovery(
    cell_root: Path, state: Mapping[str, Any], protocol_sha256: str,
) -> dict[str, Any]:
    try:
        payload = json.loads((cell_root / AGENT_SESSION_FILE).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignStageError(
            "open the agent session before freezing discovery"
        ) from exc
    validated = _validate_agent_session(
        payload, state=state, agent_protocol_sha256=protocol_sha256,
    )
    if validated["status"] != "open":
        raise CampaignStageError("discovery proposals require the open agent session")
    return validated


def _agent_session_for_freeze(
    cell_root: Path, state: Mapping[str, Any], protocol_sha256: str,
) -> dict[str, Any]:
    try:
        config = yaml.safe_load((cell_root / "automil/config.yaml").read_text()) or {}
        payload = json.loads((cell_root / AGENT_SESSION_FILE).read_text())
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise CampaignStageError(
            f"{state.get('cell_id')}: agent session attestation is missing"
        ) from exc
    if (config.get("campaign") or {}).get(
        "agent_protocol_sha256"
    ) != protocol_sha256:
        raise CampaignStageError(
            f"{state.get('cell_id')}: cell agent protocol binding drift"
        )
    return _validate_agent_session(
        payload, state=state, agent_protocol_sha256=protocol_sha256,
        require_finalized=True,
    )


def _process_resource_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key in ("elapsed_seconds", "peak_vram_mb"):
        values: list[float] = []
        for row in rows:
            value = row.get(key)
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise CampaignStageError(f"process resource {key} is invalid")
            values.append(float(value))
        summary = {
            "reported": len(values),
            "missing": len(rows) - len(values),
            "maximum": max(values) if values else None,
        }
        if key == "elapsed_seconds":
            total = math.fsum(values) if values else None
            summary["total"] = total
            summary["gpu_attached_job_hours"] = (
                total / 3600 if total is not None else None
            )
        output[key] = summary
    return output


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _process_evidence(state: Mapping[str, Any]) -> dict[str, Any]:
    """Return the complete validation-only search record frozen before test."""
    discovery = state.get("discovery")
    baseline = state.get("baseline")
    if not isinstance(discovery, dict) or not isinstance(baseline, dict):
        raise CampaignStageError("process evidence requires discovery and baseline")
    attempts = discovery.get("attempt_audit")
    if (
        discovery.get("attempt_budget") != DISCOVERY_ATTEMPTS
        or discovery.get("attempts_charged") != DISCOVERY_ATTEMPTS
        or not isinstance(attempts, list)
        or len(attempts) != DISCOVERY_ATTEMPTS
    ):
        raise CampaignStageError(
            f"process evidence requires exactly {DISCOVERY_ATTEMPTS} discovery attempts"
        )
    audit_fields = {
        "node_id", "source_spec_sha256", "submitted_at", "agent_session_id",
        "agent_session_binding_sha256", "candidate_class", "policy_hash",
        "result_status", "termination_reason", "budget_killed", "outcome_class",
        "elapsed_seconds", "peak_vram_mb", "eligible", "reason",
        "candidate_sha256", "validation_mean",
    }
    classes = ("config-only", "train-only-source", "inadmissible")
    ordered: list[dict[str, Any]] = []
    for row in attempts:
        if not isinstance(row, dict) or set(row) != audit_fields:
            raise CampaignStageError("discovery process audit field set is not exact")
        if row.get("candidate_class") not in classes:
            raise CampaignStageError("discovery candidate class is invalid")
        if row.get("outcome_class") not in ATTEMPT_OUTCOME_CLASSES:
            raise CampaignStageError("discovery outcome class is invalid")
        if (
            not isinstance(row.get("budget_killed"), bool)
            or row.get("outcome_class") != classify_attempt_outcome(
                row.get("result_status"), row.get("termination_reason"),
                row.get("budget_killed"),
            )
        ):
            raise CampaignStageError("discovery terminal outcome is inconsistent")
        if not isinstance(row.get("eligible"), bool):
            raise CampaignStageError("discovery eligibility must be boolean")
        for key in ("node_id", "source_spec_sha256", "submitted_at",
                    "agent_session_id", "agent_session_binding_sha256",
                    "result_status", "termination_reason", "reason"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise CampaignStageError(f"discovery process {key} is invalid")
        try:
            submitted = datetime.fromisoformat(row["submitted_at"])
        except ValueError as exc:
            raise CampaignStageError("discovery process timestamp is invalid") from exc
        if submitted.tzinfo is None:
            raise CampaignStageError("discovery process timestamp lacks timezone")
        for key in ("elapsed_seconds", "peak_vram_mb"):
            value = row.get(key)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise CampaignStageError(f"discovery process {key} is invalid")
        if row["eligible"] and (
            row["candidate_class"] == "inadmissible"
            or row.get("result_status") != "completed"
            or row.get("outcome_class") != "completed"
            or row.get("reason") != "complete"
            or not _is_sha256(row.get("candidate_sha256"))
            or not _is_sha256(row.get("policy_hash"))
            or not _is_sha256(row.get("source_spec_sha256"))
            or not isinstance(row.get("validation_mean"), (int, float))
            or isinstance(row.get("validation_mean"), bool)
            or not math.isfinite(float(row["validation_mean"]))
            or not 0 <= float(row["validation_mean"]) <= 1
        ):
            raise CampaignStageError("eligible discovery process row is incomplete")
        ordered.append(json.loads(json.dumps(row)))
    ordered.sort(key=lambda row: (row["submitted_at"], row["node_id"]))
    discovery_baseline_folds = [
        fold for fold in baseline.get("validation_folds", [])
        if fold.get("fold_index") in STAGE_FOLDS["discovery"]
    ]
    if len(discovery_baseline_folds) != len(STAGE_FOLDS["discovery"]):
        raise CampaignStageError("baseline lacks the discovery-fold incumbent")
    baseline_value = _mean(discovery_baseline_folds)
    best_value = baseline_value
    best_candidate = "baseline"
    trajectory: list[dict[str, Any]] = []
    for attempt_index, row in enumerate(ordered, 1):
        value = row.get("validation_mean")
        if row["eligible"] and float(value) > best_value:
            best_value = float(value)
            best_candidate = str(row["node_id"])
        trajectory.append({
            "attempt_index": attempt_index,
            "node_id": row["node_id"],
            "result_status": row["result_status"],
            "outcome_class": row["outcome_class"],
            "eligible": row["eligible"],
            "validation_mean": value,
            "running_best_candidate_id": best_candidate,
            "running_best_validation_mean": best_value,
        })
    class_counts = {
        candidate_class: sum(
            row["candidate_class"] == candidate_class for row in ordered
        )
        for candidate_class in classes
    }
    result_counts = {
        status: sum(row["result_status"] == status for row in ordered)
        for status in sorted({row["result_status"] for row in ordered})
    }
    outcome_counts = {
        outcome: sum(row["outcome_class"] == outcome for row in ordered)
        for outcome in ATTEMPT_OUTCOME_CLASSES
    }
    complete_candidates = sum(row["eligible"] for row in ordered)
    unique_complete_candidates = len({
        row["candidate_sha256"] for row in ordered if row["eligible"]
    })
    expected_sources = expected_promotion_sources(ordered)
    expected_promoted = len(expected_sources)
    promoted = discovery.get("promoted_candidates")
    actual_sources = [
        {
            "source_node_id": str(candidate.get("candidate_id")),
            "source_candidate_sha256": str(candidate.get("candidate_sha256")),
            "source_spec_sha256": str(candidate.get("source_spec_sha256")),
            "candidate_class": str((candidate.get("identity") or {}).get(
                "candidate_class"
            )),
            "policy_hash": str((candidate.get("identity") or {}).get("policy_hash")),
        }
        for candidate in promoted
    ] if isinstance(promoted, list) else []
    if (
        discovery.get("complete_candidates") != complete_candidates
        or discovery.get("unique_complete_candidates") != unique_complete_candidates
        or actual_sources != expected_sources
    ):
        raise CampaignStageError("discovery process counts do not reconcile")

    promotion = state.get("promotion")
    if not isinstance(promotion, dict):
        raise CampaignStageError("promotion process evidence is missing")
    jobs = promotion.get("jobs")
    if not isinstance(jobs, list) or len(jobs) > PROMOTION_CANDIDATES:
        raise CampaignStageError("promotion process roster is invalid")
    if not isinstance(promoted, list) or len(promoted) != len(jobs):
        raise CampaignStageError("promotion process differs from the discovery freeze")
    attempts_charged = promotion.get("attempts_charged", 0)
    if attempts_charged != len(jobs):
        raise CampaignStageError("promotion charged-attempt count is inconsistent")
    if jobs and not promotion.get("frozen"):
        raise CampaignStageError("promotion process is not frozen")
    frozen_jobs = json.loads(json.dumps(jobs))
    job_fields = {
        "rank", "source_node_id", "source_candidate_sha256",
        "promotion_node_id", "promotion_candidate_sha256",
        "promotion_identity", "status", "candidate_class", "policy_hash",
        "result_status", "source_spec_sha256", "promotion_spec_sha256",
        "submitted_at", "elapsed_seconds", "peak_vram_mb", "validation_mean",
        "termination_reason", "budget_killed", "outcome_class", "reason",
    }
    for rank, (job, source) in enumerate(
        zip(frozen_jobs, expected_sources, strict=True), 1,
    ):
        if (
            not isinstance(job, dict)
            or set(job) != job_fields
            or job.get("rank") != rank
            or any(job.get(key) != value for key, value in source.items())
            or not _is_sha256(job.get("promotion_candidate_sha256"))
            or job.get("status") not in {"eligible", "ineligible"}
            or not isinstance(job.get("reason"), str)
            or not job["reason"]
            or job.get("candidate_class") not in {"config-only", "train-only-source"}
            or job.get("outcome_class") not in ATTEMPT_OUTCOME_CLASSES
            or not isinstance(job.get("termination_reason"), str)
            or not isinstance(job.get("budget_killed"), bool)
            or job.get("outcome_class") != classify_attempt_outcome(
                job.get("result_status"), job.get("termination_reason"),
                job.get("budget_killed"),
            )
        ):
            raise CampaignStageError("promotion process job is incomplete")
        if job["status"] == "eligible" and (
            job.get("result_status") != "completed"
            or job.get("outcome_class") != "completed"
            or job.get("reason") != "complete five-fold validation"
            or not _is_sha256(job.get("promotion_spec_sha256"))
            or isinstance(job.get("validation_mean"), bool)
            or not isinstance(job.get("validation_mean"), (int, float))
            or not math.isfinite(float(job["validation_mean"]))
            or not 0 <= float(job["validation_mean"]) <= 1
        ):
            raise CampaignStageError("eligible promotion process job is incomplete")
    promotion_status_counts = {
        status: sum(job["status"] == status for job in frozen_jobs)
        for status in ("eligible", "ineligible")
    }
    promotion_outcome_counts = {
        outcome: sum(job["outcome_class"] == outcome for job in frozen_jobs)
        for outcome in ATTEMPT_OUTCOME_CLASSES
    }
    return {
        "schema_version": 1,
        "baseline": {
            "folds": list(CERTIFICATION_FOLDS),
            "result_status": baseline.get("result_status"),
            "resources": baseline.get("resources"),
        },
        "discovery": {
            "attempt_budget": DISCOVERY_ATTEMPTS,
            "attempts_charged": DISCOVERY_ATTEMPTS,
            "baseline_validation_mean": baseline_value,
            "complete_candidates": complete_candidates,
            "unique_complete_candidates": unique_complete_candidates,
            "promoted_candidates": len(promoted),
            "candidate_class_counts": class_counts,
            "result_status_counts": result_counts,
            "outcome_class_counts": outcome_counts,
            "attempts": ordered,
            "validation_anytime": trajectory,
            "resources": _process_resource_summary(ordered),
        },
        "promotion": {
            "candidate_budget": PROMOTION_CANDIDATES,
            "attempts_charged": attempts_charged,
            "status_counts": promotion_status_counts,
            "outcome_class_counts": promotion_outcome_counts,
            "yield": (
                promotion_status_counts["eligible"] / len(frozen_jobs)
                if frozen_jobs else None
            ),
            "jobs": frozen_jobs,
            "resources": _process_resource_summary(frozen_jobs),
        },
    }


def _validate_process_resource_summary(
    raw: object, *, count: int, label: str,
) -> None:
    if not isinstance(raw, dict) or set(raw) != {
        "elapsed_seconds", "peak_vram_mb",
    }:
        raise CampaignStageError(f"{label} resource summary is malformed")
    elapsed = raw["elapsed_seconds"]
    vram = raw["peak_vram_mb"]
    if (
        not isinstance(elapsed, dict)
        or set(elapsed) != {
            "reported", "missing", "maximum", "total",
            "gpu_attached_job_hours",
        }
        or not isinstance(vram, dict)
        or set(vram) != {"reported", "missing", "maximum"}
    ):
        raise CampaignStageError(f"{label} resource fields are malformed")
    for name, record in (("elapsed_seconds", elapsed), ("peak_vram_mb", vram)):
        reported = record.get("reported")
        missing = record.get("missing")
        if (
            isinstance(reported, bool) or not isinstance(reported, int)
            or isinstance(missing, bool) or not isinstance(missing, int)
            or reported < 0 or missing < 0 or reported + missing != count
        ):
            raise CampaignStageError(
                f"{label} {name} missingness is inconsistent"
            )
    if elapsed["reported"] == 0:
        if any(elapsed[key] is not None for key in (
            "maximum", "total", "gpu_attached_job_hours",
        )):
            raise CampaignStageError(f"{label} missing elapsed time was imputed")
    else:
        values = [elapsed[key] for key in (
            "maximum", "total", "gpu_attached_job_hours",
        )]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
            for value in values
        ):
            raise CampaignStageError(f"{label} elapsed summary is invalid")
        maximum, total, gpu_hours = map(float, values)
        if total < maximum or not math.isclose(
            gpu_hours, total / 3600, rel_tol=0.0, abs_tol=1e-12,
        ):
            raise CampaignStageError(f"{label} elapsed summary is inconsistent")
    maximum_vram = vram["maximum"]
    if vram["reported"] == 0:
        if maximum_vram is not None:
            raise CampaignStageError(f"{label} missing VRAM was imputed")
    elif (
        isinstance(maximum_vram, bool)
        or not isinstance(maximum_vram, (int, float))
        or not math.isfinite(float(maximum_vram))
        or float(maximum_vram) < 0
    ):
        raise CampaignStageError(f"{label} VRAM maximum is invalid")


def _process_unit_interval(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 <= float(value) <= 1
    ):
        raise CampaignStageError(f"{label} must be finite and in [0, 1]")
    return float(value)


def validate_process_evidence_artifact(
    raw: object,
    expected_sha256: object,
    *,
    cell_id: str,
    expected_session_id: str,
    expected_session_binding: str,
) -> dict[str, Any]:
    """Validate one complete pre-unblinding search-process census."""
    if not isinstance(cell_id, str) or not cell_id:
        raise CampaignStageError("process evidence cell_id is invalid")
    if not isinstance(expected_session_id, str) or not expected_session_id:
        raise CampaignStageError(f"{cell_id}: process session id is invalid")
    if not _is_sha256(expected_session_binding):
        raise CampaignStageError(f"{cell_id}: process session binding is invalid")
    if (
        not isinstance(raw, dict)
        or expected_sha256 != content_sha256(raw)
        or set(raw) != {"schema_version", "baseline", "discovery", "promotion"}
        or raw.get("schema_version") != 1
    ):
        raise CampaignStageError(f"{cell_id}: process evidence schema/hash mismatch")

    baseline = raw.get("baseline")
    if (
        not isinstance(baseline, dict)
        or set(baseline) != {"folds", "result_status", "resources"}
        or baseline.get("folds") != list(CERTIFICATION_FOLDS)
        or baseline.get("result_status") != "completed"
    ):
        raise CampaignStageError(f"{cell_id}: baseline process evidence is invalid")
    _validate_process_resource_summary(
        baseline.get("resources"), count=1, label=f"{cell_id}.baseline",
    )

    discovery = raw.get("discovery")
    discovery_fields = {
        "attempt_budget", "attempts_charged", "baseline_validation_mean",
        "complete_candidates", "unique_complete_candidates",
        "promoted_candidates", "candidate_class_counts",
        "result_status_counts", "outcome_class_counts", "attempts",
        "validation_anytime", "resources",
    }
    if not isinstance(discovery, dict) or set(discovery) != discovery_fields:
        raise CampaignStageError(f"{cell_id}: discovery process schema is invalid")
    attempts = discovery.get("attempts")
    anytime = discovery.get("validation_anytime")
    if (
        discovery.get("attempt_budget") != DISCOVERY_ATTEMPTS
        or discovery.get("attempts_charged") != DISCOVERY_ATTEMPTS
        or not isinstance(attempts, list)
        or len(attempts) != DISCOVERY_ATTEMPTS
        or not isinstance(anytime, list)
        or len(anytime) != DISCOVERY_ATTEMPTS
    ):
        raise CampaignStageError(f"{cell_id}: discovery census is not exact")
    audit_fields = {
        "node_id", "source_spec_sha256", "submitted_at", "agent_session_id",
        "agent_session_binding_sha256", "candidate_class", "policy_hash",
        "result_status", "termination_reason", "budget_killed",
        "outcome_class", "elapsed_seconds", "peak_vram_mb", "eligible",
        "reason", "candidate_sha256", "validation_mean",
    }
    classes = ("config-only", "train-only-source", "inadmissible")
    for row in attempts:
        if not isinstance(row, dict) or set(row) != audit_fields:
            raise CampaignStageError(f"{cell_id}: discovery attempt schema drift")
        for key in (
            "node_id", "submitted_at", "agent_session_id", "result_status",
            "termination_reason", "reason",
        ):
            if not isinstance(row.get(key), str) or not row[key]:
                raise CampaignStageError(
                    f"{cell_id}: discovery attempt {key} is invalid"
                )
        try:
            submitted = datetime.fromisoformat(row["submitted_at"])
        except ValueError as exc:
            raise CampaignStageError(
                f"{cell_id}: discovery timestamp is invalid"
            ) from exc
        if submitted.tzinfo is None:
            raise CampaignStageError(f"{cell_id}: discovery timestamp lacks timezone")
        if (
            not _is_sha256(row.get("source_spec_sha256"))
            or row.get("agent_session_id") != expected_session_id
            or row.get("agent_session_binding_sha256") != expected_session_binding
            or row.get("candidate_class") not in classes
            or not isinstance(row.get("budget_killed"), bool)
            or row.get("outcome_class") not in ATTEMPT_OUTCOME_CLASSES
            or row.get("outcome_class") != classify_attempt_outcome(
                row.get("result_status"), row.get("termination_reason"),
                row.get("budget_killed"),
            )
            or not isinstance(row.get("eligible"), bool)
        ):
            raise CampaignStageError(f"{cell_id}: discovery attempt value drift")
        for key in ("elapsed_seconds", "peak_vram_mb"):
            value = row.get(key)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise CampaignStageError(
                    f"{cell_id}: discovery attempt {key} is invalid"
                )
        for key in ("policy_hash", "candidate_sha256"):
            value = row.get(key)
            if value is not None and not _is_sha256(value):
                raise CampaignStageError(
                    f"{cell_id}: discovery attempt {key} is invalid"
                )
        value = row.get("validation_mean")
        if value is not None:
            _process_unit_interval(value, f"{cell_id}.attempt.validation_mean")
        if row["eligible"] and (
            row["candidate_class"] == "inadmissible"
            or row["result_status"] != "completed"
            or row["outcome_class"] != "completed"
            or row["reason"] != "complete"
            or not _is_sha256(row.get("policy_hash"))
            or not _is_sha256(row.get("candidate_sha256"))
            or value is None
        ):
            raise CampaignStageError(
                f"{cell_id}: eligible discovery attempt is incomplete"
            )
    if attempts != sorted(
        attempts, key=lambda row: (row["submitted_at"], row["node_id"]),
    ):
        raise CampaignStageError(f"{cell_id}: discovery attempt order drift")
    if (
        len({row["node_id"] for row in attempts}) != DISCOVERY_ATTEMPTS
        or len({row["source_spec_sha256"] for row in attempts})
        != DISCOVERY_ATTEMPTS
    ):
        raise CampaignStageError(
            f"{cell_id}: discovery attempt identities are not unique"
        )

    class_counts = {
        candidate_class: sum(
            row["candidate_class"] == candidate_class for row in attempts
        )
        for candidate_class in classes
    }
    result_counts = {
        status: sum(row["result_status"] == status for row in attempts)
        for status in sorted({row["result_status"] for row in attempts})
    }
    outcome_counts = {
        outcome: sum(row["outcome_class"] == outcome for row in attempts)
        for outcome in ATTEMPT_OUTCOME_CLASSES
    }
    complete_candidates = sum(row["eligible"] for row in attempts)
    unique_complete_candidates = len({
        row["candidate_sha256"] for row in attempts if row["eligible"]
    })
    expected_sources = expected_promotion_sources(attempts)
    if (
        discovery.get("candidate_class_counts") != class_counts
        or discovery.get("result_status_counts") != result_counts
        or discovery.get("outcome_class_counts") != outcome_counts
        or discovery.get("complete_candidates") != complete_candidates
        or discovery.get("unique_complete_candidates") != unique_complete_candidates
        or discovery.get("promoted_candidates") != len(expected_sources)
        or discovery.get("resources") != _process_resource_summary(attempts)
    ):
        raise CampaignStageError(f"{cell_id}: discovery summaries do not reconcile")
    _validate_process_resource_summary(
        discovery.get("resources"), count=len(attempts),
        label=f"{cell_id}.discovery",
    )
    best = _process_unit_interval(
        discovery.get("baseline_validation_mean"),
        f"{cell_id}.baseline_validation_mean",
    )
    best_id = "baseline"
    for index, (attempt, point) in enumerate(
        zip(attempts, anytime, strict=True), 1,
    ):
        if attempt["eligible"]:
            candidate = _process_unit_interval(
                attempt["validation_mean"],
                f"{cell_id}.attempt.{index}.validation_mean",
            )
            if candidate > best:
                best = candidate
                best_id = attempt["node_id"]
        expected_point = {
            "attempt_index": index,
            "node_id": attempt["node_id"],
            "result_status": attempt["result_status"],
            "outcome_class": attempt["outcome_class"],
            "eligible": attempt["eligible"],
            "validation_mean": attempt["validation_mean"],
            "running_best_candidate_id": best_id,
            "running_best_validation_mean": best,
        }
        if point != expected_point:
            raise CampaignStageError(
                f"{cell_id}: validation-anytime trajectory is inconsistent"
            )

    promotion = raw.get("promotion")
    promotion_fields = {
        "candidate_budget", "attempts_charged", "status_counts",
        "outcome_class_counts", "yield", "jobs", "resources",
    }
    if not isinstance(promotion, dict) or set(promotion) != promotion_fields:
        raise CampaignStageError(f"{cell_id}: promotion process schema is invalid")
    jobs = promotion.get("jobs")
    if (
        promotion.get("candidate_budget") != PROMOTION_CANDIDATES
        or not isinstance(jobs, list)
        or len(jobs) != len(expected_sources)
        or len(jobs) > PROMOTION_CANDIDATES
        or promotion.get("attempts_charged") != len(jobs)
    ):
        raise CampaignStageError(f"{cell_id}: promotion census is inconsistent")
    job_fields = {
        "rank", "source_node_id", "source_candidate_sha256",
        "promotion_node_id", "promotion_candidate_sha256",
        "promotion_identity", "status", "candidate_class", "policy_hash",
        "result_status", "source_spec_sha256", "promotion_spec_sha256",
        "submitted_at", "elapsed_seconds", "peak_vram_mb", "validation_mean",
        "termination_reason", "budget_killed", "outcome_class", "reason",
    }
    for rank, (job, source) in enumerate(
        zip(jobs, expected_sources, strict=True), 1,
    ):
        if not isinstance(job, dict) or set(job) != job_fields:
            raise CampaignStageError(f"{cell_id}: promotion job schema drift")
        identity = job.get("promotion_identity")
        if (
            job.get("rank") != rank
            or any(job.get(key) != value for key, value in source.items())
            or not isinstance(job.get("promotion_node_id"), str)
            or not job["promotion_node_id"]
            or not _is_sha256(job.get("promotion_candidate_sha256"))
            or not isinstance(identity, dict)
            or set(identity) != {
                "overlay_manifest", "deletions",
                "candidate_class", "policy_hash", "variant_selection_hash",
                "override_hash",
            }
            or content_sha256(identity) != job["promotion_candidate_sha256"]
            or identity.get("candidate_class") != job.get("candidate_class")
            or job.get("status") not in {"eligible", "ineligible"}
            or not isinstance(job.get("result_status"), str)
            or not job["result_status"]
            or not isinstance(job.get("termination_reason"), str)
            or not job["termination_reason"]
            or not isinstance(job.get("budget_killed"), bool)
            or job.get("outcome_class") not in ATTEMPT_OUTCOME_CLASSES
            or job.get("outcome_class") != classify_attempt_outcome(
                job.get("result_status"), job.get("termination_reason"),
                job.get("budget_killed"),
            )
            or not isinstance(job.get("reason"), str)
            or not job["reason"]
        ):
            raise CampaignStageError(f"{cell_id}: promotion job value drift")
        try:
            submitted = datetime.fromisoformat(str(job.get("submitted_at")))
        except ValueError as exc:
            raise CampaignStageError(
                f"{cell_id}: promotion timestamp is invalid"
            ) from exc
        if submitted.tzinfo is None:
            raise CampaignStageError(f"{cell_id}: promotion timestamp lacks timezone")
        for key in ("elapsed_seconds", "peak_vram_mb"):
            value = job.get(key)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise CampaignStageError(
                    f"{cell_id}: promotion job {key} is invalid"
                )
        promotion_spec = job.get("promotion_spec_sha256")
        if not _is_sha256(promotion_spec):
            raise CampaignStageError(
                f"{cell_id}: promotion spec hash is invalid"
            )
        validation_mean = job.get("validation_mean")
        if validation_mean is not None:
            _process_unit_interval(
                validation_mean, f"{cell_id}.promotion.validation_mean",
            )
        if job["status"] == "eligible" and (
            job["result_status"] != "completed"
            or job["outcome_class"] != "completed"
            or job["reason"] != "complete five-fold validation"
            or not _is_sha256(promotion_spec)
            or validation_mean is None
        ):
            raise CampaignStageError(
                f"{cell_id}: eligible promotion job is incomplete"
            )
    if (
        len({job["promotion_node_id"] for job in jobs}) != len(jobs)
        or len({job["promotion_spec_sha256"] for job in jobs}) != len(jobs)
    ):
        raise CampaignStageError(
            f"{cell_id}: promotion job identities are not unique"
        )
    status_counts = {
        status: sum(job["status"] == status for job in jobs)
        for status in ("eligible", "ineligible")
    }
    promotion_outcomes = {
        outcome: sum(job["outcome_class"] == outcome for job in jobs)
        for outcome in ATTEMPT_OUTCOME_CLASSES
    }
    expected_yield = status_counts["eligible"] / len(jobs) if jobs else None
    if (
        promotion.get("status_counts") != status_counts
        or promotion.get("outcome_class_counts") != promotion_outcomes
        or promotion.get("yield") != expected_yield
        or promotion.get("resources") != _process_resource_summary(jobs)
    ):
        raise CampaignStageError(f"{cell_id}: promotion summaries do not reconcile")
    _validate_process_resource_summary(
        promotion.get("resources"), count=len(jobs),
        label=f"{cell_id}.promotion",
    )
    return json.loads(json.dumps(raw))


def _process_matches_session(
    process: Mapping[str, Any], session: Mapping[str, Any],
) -> bool:
    attempts = (process.get("discovery") or {}).get("attempts")
    if not isinstance(attempts, list):
        return False
    expected_id = (session.get("session") or {}).get("session_id")
    expected_binding = session.get("binding_sha256")
    return all(
        isinstance(row, dict)
        and row.get("agent_session_id") == expected_id
        and row.get("agent_session_binding_sha256") == expected_binding
        for row in attempts
    )


def _roster_payload(cells: object) -> dict[str, str]:
    if not isinstance(cells, list):
        raise CampaignStageError("campaign roster must be a list")
    roster: dict[str, str] = {}
    for row in cells:
        if not isinstance(row, Mapping):
            raise CampaignStageError("campaign roster row is invalid")
        cell_id = row.get("cell_id")
        cell_sha256 = row.get("cell_sha256")
        if (
            not isinstance(cell_id, str) or not cell_id
            or not isinstance(cell_sha256, str) or len(cell_sha256) != 64
            or any(char not in "0123456789abcdef" for char in cell_sha256)
            or cell_id in roster
        ):
            raise CampaignStageError("campaign roster row is invalid")
        roster[cell_id] = cell_sha256
    return dict(sorted(roster.items()))


def _locked_manifest_roster(
    cell_root: Path, state: Mapping[str, Any],
) -> dict[str, str]:
    try:
        config = yaml.safe_load((cell_root / "automil/config.yaml").read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise CampaignStageError("cell config is unreadable") from exc
    raw_manifest = (config.get("campaign") or {}).get("manifest")
    relative = PurePosixPath(str(raw_manifest))
    if relative.is_absolute() or ".." in relative.parts:
        raise CampaignStageError("cell manifest path is unsafe")
    matches: list[Path] = []
    for ancestor in (cell_root, *cell_root.parents):
        candidate = ancestor / Path(*relative.parts)
        if candidate.is_file() and file_sha256(candidate) == state.get("manifest_sha256"):
            matches.append(candidate)
    if len(matches) != 1:
        raise CampaignStageError("cannot resolve the cell's uniquely locked manifest")
    try:
        manifest = json.loads(matches[0].read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignStageError("locked campaign manifest is unreadable") from exc
    if not isinstance(manifest, dict):
        raise CampaignStageError("locked campaign manifest must be a JSON object")
    cells = manifest.get("cells")
    if (
        manifest.get("campaign_id") != CAMPAIGN_ID
        or not isinstance(cells, list)
        or len(cells) != CAMPAIGN_CELL_COUNT
    ):
        raise CampaignStageError("locked campaign manifest roster is incomplete")
    return _roster_payload(cells)


def validate_selection_freeze_artifact(artifact: object) -> dict[str, Any]:
    """Validate the intrinsic, versioned selection-freeze contract."""
    if not isinstance(artifact, dict):
        raise CampaignStageError("campaign selection freeze must be a JSON object")
    recorded = artifact.get("freeze_sha256")
    payload = {
        key: value for key, value in artifact.items() if key != "freeze_sha256"
    }
    cells = artifact.get("cells")
    roster = _roster_payload(cells)
    expected_fields = {
        "schema_version", "campaign_id", "manifest_sha256",
        "protocol_version", "agent_protocol_sha256",
        "roster_sha256", "cell_count", "cells", "frozen_at",
        "freeze_sha256",
    }
    entry_fields = {
        "cell_id", "cell_sha256", "state_sha256", "selection_sha256",
        "winner_kind", "winner_candidate_id", "winner_candidate_sha256",
        "winner_promotion_node_id",
        "winner_validation_mean", "baseline_validation_mean",
        "baseline_candidate_sha256", "winner_source_folds",
        "baseline_source_folds",
        "agent_session_sha256", "agent_session_id",
        "agent_session_binding_sha256", "agent_usage", "process_sha256",
        "process_evidence",
    }
    def valid_entry(row: object) -> bool:
        if not isinstance(row, Mapping) or set(row) != entry_fields:
            return False
        if (
            not isinstance(row.get("cell_id"), str)
            or not row["cell_id"]
            or row.get("winner_kind") not in {"baseline", "searched"}
            or not isinstance(row.get("winner_candidate_id"), str)
            or not row["winner_candidate_id"]
            or not isinstance(row.get("agent_session_id"), str)
            or not row["agent_session_id"]
            or not all(_is_sha256(row.get(key)) for key in (
                "cell_sha256", "state_sha256", "selection_sha256",
                "winner_candidate_sha256", "baseline_candidate_sha256",
                "agent_session_sha256",
                "agent_session_binding_sha256", "process_sha256",
            ))
            or not isinstance(row.get("process_evidence"), dict)
            or row.get("process_sha256")
            != content_sha256(row["process_evidence"])
        ):
            return False
        for key in ("winner_validation_mean", "baseline_validation_mean"):
            value = row.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0 <= float(value) <= 1
            ):
                return False
        if row["winner_kind"] == "baseline" and (
            row["winner_candidate_id"] != "baseline"
            or row["winner_promotion_node_id"] is not None
            or row["winner_candidate_sha256"]
            != row["baseline_candidate_sha256"]
            or row["winner_validation_mean"] != row["baseline_validation_mean"]
            or row["winner_source_folds"] != row["baseline_source_folds"]
        ):
            return False
        if (
            row["winner_kind"] == "searched"
            and (
                row["winner_candidate_id"] == "baseline"
                or not isinstance(row.get("winner_promotion_node_id"), str)
                or not row["winner_promotion_node_id"]
            )
        ):
            return False
        try:
            _validate_source_fold_anchors_artifact(
                row.get("winner_source_folds"), label="winner",
                expected_cell_id=row["cell_id"],
            )
            _validate_source_fold_anchors_artifact(
                row.get("baseline_source_folds"), label="baseline",
                expected_cell_id=row["cell_id"],
            )
            validate_agent_usage_artifact(row.get("agent_usage"))
            process = validate_process_evidence_artifact(
                row.get("process_evidence"),
                row.get("process_sha256"),
                cell_id=row["cell_id"],
                expected_session_id=row["agent_session_id"],
                expected_session_binding=row["agent_session_binding_sha256"],
            )
            if row["winner_kind"] == "searched":
                matches = [
                    job for job in process["promotion"]["jobs"]
                    if job["promotion_node_id"]
                    == row["winner_promotion_node_id"]
                ]
                if len(matches) != 1 or (
                    matches[0]["status"] != "eligible"
                    or matches[0]["source_node_id"]
                    != row["winner_candidate_id"]
                    or matches[0]["source_candidate_sha256"]
                    != row["winner_candidate_sha256"]
                    or matches[0]["validation_mean"]
                    != row["winner_validation_mean"]
                ):
                    raise CampaignStageError(
                        "searched winner differs from process evidence"
                    )
        except CampaignStageError as exc:
            raise CampaignStageError(
                f"selection freeze cell {row['cell_id']} is invalid: {exc}"
            ) from exc
        return True

    try:
        frozen_at = datetime.fromisoformat(str(artifact.get("frozen_at")))
    except ValueError:
        frozen_at = None

    if (
        set(artifact) != expected_fields
        or artifact.get("schema_version") != SELECTION_FREEZE_SCHEMA_VERSION
        or artifact.get("campaign_id") != CAMPAIGN_ID
        or artifact.get("protocol_version") != PROTOCOL_VERSION
        or not all(_is_sha256(artifact.get(key)) for key in (
            "manifest_sha256", "agent_protocol_sha256",
            "roster_sha256", "freeze_sha256",
        ))
        or frozen_at is None
        or frozen_at.tzinfo is None
        or artifact.get("cell_count") != CAMPAIGN_CELL_COUNT
        or not isinstance(cells, list)
        or len(cells) != CAMPAIGN_CELL_COUNT
        or any(not valid_entry(row) for row in cells)
        or len({row.get("cell_id") for row in cells if isinstance(row, dict)})
        != CAMPAIGN_CELL_COUNT
        or len({
            row.get("agent_session_id") for row in cells
            if isinstance(row, dict)
        }) != CAMPAIGN_CELL_COUNT
        or artifact.get("roster_sha256") != content_sha256(roster)
        or not isinstance(recorded, str)
        or recorded != content_sha256(payload)
    ):
        raise CampaignStageError("campaign selection freeze integrity mismatch")
    return json.loads(json.dumps(artifact))


def _validated_selection_freeze(runtime_root: Path) -> dict[str, Any]:
    path = runtime_root / SELECTION_FREEZE_FILE
    try:
        artifact = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignStageError(
            "held-out certification requires the global 130-cell selection freeze"
        ) from exc
    return validate_selection_freeze_artifact(artifact)


def validate_certification_bundle_artifact(artifact: object) -> dict[str, Any]:
    """Validate the exact self-consistent five-fold certification bundle."""
    top_fields = {
        "schema_version", "campaign_id", "cell_id", "winner",
        "selection_sha256", "selection_freeze_sha256",
        "selection_state_sha256", "validation_mean", "baseline",
        "held_out_folds", "held_out", "source_fold_sha256",
        "baseline_held_out_folds", "baseline_held_out",
        "baseline_source_fold_sha256", "paired_fold_deltas",
        "held_out_lift", "retrained", "certified_at", "bundle_sha256",
    }
    if not isinstance(artifact, dict) or set(artifact) != top_fields:
        raise CampaignStageError("certification bundle field set is not exact")
    recorded = artifact.get("bundle_sha256")
    payload = {
        key: value for key, value in artifact.items() if key != "bundle_sha256"
    }
    winner = artifact.get("winner")
    baseline = artifact.get("baseline")
    if (
        artifact.get("schema_version") != 2
        or artifact.get("campaign_id") != CAMPAIGN_ID
        or not isinstance(artifact.get("cell_id"), str)
        or not artifact["cell_id"]
        or not all(_is_sha256(artifact.get(key)) for key in (
            "selection_sha256", "selection_freeze_sha256",
            "selection_state_sha256",
        ))
        or recorded != content_sha256(payload)
        or artifact.get("retrained") is not False
        or not isinstance(winner, dict)
        or set(winner) != {
            "kind", "candidate_id", "candidate_sha256", "promotion_node_id",
        }
        or winner.get("kind") not in {"baseline", "searched"}
        or not isinstance(winner.get("candidate_id"), str)
        or not winner["candidate_id"]
        or not _is_sha256(winner.get("candidate_sha256"))
        or (
            winner["kind"] == "baseline"
            and winner.get("promotion_node_id") is not None
        )
        or (
            winner["kind"] == "searched"
            and (
                not isinstance(winner.get("promotion_node_id"), str)
                or not winner["promotion_node_id"]
            )
        )
        or not isinstance(baseline, dict)
        or set(baseline) != {
            "candidate_id", "candidate_sha256", "validation_mean",
        }
        or baseline.get("candidate_id") != "baseline"
        or not _is_sha256(baseline.get("candidate_sha256"))
        or (
            winner["kind"] == "baseline"
            and (
                winner.get("candidate_id") != "baseline"
                or winner.get("candidate_sha256")
                != baseline.get("candidate_sha256")
            )
        )
        or (
            winner["kind"] == "searched"
            and winner.get("candidate_id") == "baseline"
        )
    ):
        raise CampaignStageError("certification bundle identity is invalid")
    for label, value in (
        ("winner validation_mean", artifact.get("validation_mean")),
        ("baseline validation_mean", baseline.get("validation_mean")),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0 <= float(value) <= 1
        ):
            raise CampaignStageError(f"certification {label} is invalid")
    try:
        certified_at = datetime.fromisoformat(str(artifact.get("certified_at")))
    except ValueError as exc:
        raise CampaignStageError("certification timestamp is invalid") from exc
    if certified_at.tzinfo is None:
        raise CampaignStageError("certification timestamp lacks a timezone")

    def validated_folds(raw: object, label: str) -> tuple[list[dict[str, Any]], set[str]]:
        if not isinstance(raw, list) or len(raw) != len(CERTIFICATION_FOLDS):
            raise CampaignStageError(f"{label} must contain exactly five folds")
        by_fold: dict[int, dict[str, Any]] = {}
        metric_keys: set[str] | None = None
        for row in raw:
            if (
                not isinstance(row, dict)
                or set(row) != {"fold_index", "held_out"}
                or not isinstance(row.get("fold_index"), int)
                or isinstance(row.get("fold_index"), bool)
                or row["fold_index"] in by_fold
                or not isinstance(row.get("held_out"), dict)
            ):
                raise CampaignStageError(f"{label} fold schema is invalid")
            metrics = row["held_out"]
            keys = set(metrics)
            if keys not in ({"test_auc", "test_bacc"}, {"test_c_index"}):
                raise CampaignStageError(f"{label} metric schema is not locked")
            if metric_keys is None:
                metric_keys = keys
            elif keys != metric_keys:
                raise CampaignStageError(f"{label} metric keys differ across folds")
            for key, value in metrics.items():
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or not 0 <= float(value) <= 1
                ):
                    raise CampaignStageError(f"{label}.{key} is outside [0, 1]")
            by_fold[row["fold_index"]] = row
        if set(by_fold) != set(CERTIFICATION_FOLDS):
            raise CampaignStageError(f"{label} fold roster differs from 0..4")
        return [by_fold[fold] for fold in CERTIFICATION_FOLDS], metric_keys or set()

    baseline_folds, baseline_keys = validated_folds(
        artifact.get("baseline_held_out_folds"), "baseline held-out",
    )
    winner_folds, winner_keys = validated_folds(
        artifact.get("held_out_folds"), "winner held-out",
    )
    if baseline_keys != winner_keys:
        raise CampaignStageError("winner and baseline metric schemas differ")

    def validate_aggregate(
        raw: object, folds: list[dict[str, Any]], label: str,
    ) -> dict[str, float]:
        if not isinstance(raw, dict) or set(raw) != baseline_keys:
            raise CampaignStageError(f"{label} aggregate schema is invalid")
        normalized: dict[str, float] = {}
        for key in sorted(baseline_keys):
            value = raw.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0 <= float(value) <= 1
            ):
                raise CampaignStageError(f"{label}.{key} is invalid")
            expected = math.fsum(
                float(row["held_out"][key]) for row in folds
            ) / len(folds)
            if not math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12):
                raise CampaignStageError(f"{label}.{key} disagrees with its folds")
            normalized[key] = float(value)
        return normalized

    baseline_aggregate = validate_aggregate(
        artifact.get("baseline_held_out"), baseline_folds, "baseline held-out",
    )
    winner_aggregate = validate_aggregate(
        artifact.get("held_out"), winner_folds, "winner held-out",
    )
    hash_keys = {f"fold_{fold}_result.json" for fold in CERTIFICATION_FOLDS}
    for label in ("source_fold_sha256", "baseline_source_fold_sha256"):
        hashes = artifact.get(label)
        if (
            not isinstance(hashes, dict)
            or set(hashes) != hash_keys
            or any(not _is_sha256(value) for value in hashes.values())
        ):
            raise CampaignStageError(f"certification {label} is invalid")
    if winner["kind"] == "baseline" and (
        not math.isclose(
            float(artifact["validation_mean"]),
            float(baseline["validation_mean"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or winner_folds != baseline_folds
        or winner_aggregate != baseline_aggregate
        or artifact["source_fold_sha256"]
        != artifact["baseline_source_fold_sha256"]
    ):
        raise CampaignStageError(
            "baseline winner evidence differs from its baseline comparator"
        )
    paired = artifact.get("paired_fold_deltas")
    if not isinstance(paired, list) or len(paired) != len(CERTIFICATION_FOLDS):
        raise CampaignStageError("certification paired deltas are incomplete")
    for fold, row in enumerate(paired):
        expected_delta = {
            key: float(winner_folds[fold]["held_out"][key])
            - float(baseline_folds[fold]["held_out"][key])
            for key in sorted(baseline_keys)
        }
        observed_delta = row.get("held_out_delta") if isinstance(row, dict) else None
        if (
            not isinstance(row, dict)
            or set(row) != {"fold_index", "held_out_delta"}
            or row.get("fold_index") != fold
            or not isinstance(observed_delta, dict)
            or set(observed_delta) != baseline_keys
            or any(
                isinstance(observed_delta.get(key), bool)
                or not isinstance(observed_delta.get(key), (int, float))
                or not math.isfinite(float(observed_delta[key]))
                or not math.isclose(
                    float(observed_delta[key]), value, rel_tol=0.0, abs_tol=1e-12,
                )
                for key, value in expected_delta.items()
            )
        ):
            raise CampaignStageError("certification paired deltas are inconsistent")
    lift = artifact.get("held_out_lift")
    if not isinstance(lift, dict) or set(lift) != baseline_keys:
        raise CampaignStageError("certification held-out lift schema is invalid")
    for key in sorted(baseline_keys):
        value = lift.get(key)
        expected = winner_aggregate[key] - baseline_aggregate[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise CampaignStageError(f"certification held-out lift {key} is inconsistent")
    return json.loads(json.dumps(artifact))


def validate_certification_bundle_binding(
    bundle: object,
    freeze_entry: object,
    *,
    selection_freeze_sha256: str,
) -> dict[str, Any]:
    """Bind a strict certification bundle to its frozen validation winner."""
    normalized = validate_certification_bundle_artifact(bundle)
    if not isinstance(freeze_entry, Mapping):
        raise CampaignStageError("certification freeze entry is invalid")
    winner = normalized.get("winner")
    baseline = normalized.get("baseline")
    expected_cell_id = freeze_entry.get("cell_id")
    winner_sources = _validate_source_fold_anchors_artifact(
        freeze_entry.get("winner_source_folds"), label="winner",
        expected_cell_id=(
            expected_cell_id if isinstance(expected_cell_id, str) else None
        ),
    )
    baseline_sources = _validate_source_fold_anchors_artifact(
        freeze_entry.get("baseline_source_folds"), label="baseline",
        expected_cell_id=(
            expected_cell_id if isinstance(expected_cell_id, str) else None
        ),
    )
    if (
        not _is_sha256(selection_freeze_sha256)
        or normalized.get("campaign_id") != CAMPAIGN_ID
        or normalized.get("cell_id") != freeze_entry.get("cell_id")
        or normalized.get("selection_freeze_sha256")
        != selection_freeze_sha256
        or normalized.get("selection_state_sha256")
        != freeze_entry.get("state_sha256")
        or normalized.get("selection_sha256")
        != freeze_entry.get("selection_sha256")
        or normalized.get("validation_mean")
        != freeze_entry.get("winner_validation_mean")
        or not isinstance(baseline, Mapping)
        or baseline.get("validation_mean")
        != freeze_entry.get("baseline_validation_mean")
        or baseline.get("candidate_sha256")
        != freeze_entry.get("baseline_candidate_sha256")
        or normalized.get("source_fold_sha256") != {
            filename: record["sha256"]
            for filename, record in winner_sources.items()
        }
        or normalized.get("baseline_source_fold_sha256") != {
            filename: record["sha256"]
            for filename, record in baseline_sources.items()
        }
        or not isinstance(winner, Mapping)
        or winner.get("kind") != freeze_entry.get("winner_kind")
        or winner.get("candidate_id")
        != freeze_entry.get("winner_candidate_id")
        or winner.get("candidate_sha256")
        != freeze_entry.get("winner_candidate_sha256")
        or winner.get("promotion_node_id")
        != freeze_entry.get("winner_promotion_node_id")
    ):
        raise CampaignStageError(
            "certification bundle differs from the frozen validation winner"
        )
    return normalized


def validate_certification_timestamp_order(
    selection_frozen_at: object,
    bundle: Mapping[str, Any],
    index_certified_at: object | None = None,
) -> None:
    """Require freeze <= bundle <= campaign index certification times."""
    try:
        freeze_time = datetime.fromisoformat(str(selection_frozen_at))
        bundle_time = datetime.fromisoformat(str(bundle.get("certified_at")))
        index_time = (
            datetime.fromisoformat(str(index_certified_at))
            if index_certified_at is not None else None
        )
    except ValueError as exc:
        raise CampaignStageError(
            "campaign certification timestamp is invalid"
        ) from exc
    if (
        freeze_time.tzinfo is None
        or bundle_time.tzinfo is None
        or freeze_time > bundle_time
        or (
            index_time is not None
            and (index_time.tzinfo is None or index_time < bundle_time)
        )
    ):
        raise CampaignStageError(
            "campaign certification timestamps violate freeze/bundle/index order"
        )


def validate_certification_source_bindings(
    runtime_root: Path,
    bundle: Mapping[str, Any],
    freeze_entry: Mapping[str, Any],
) -> None:
    root = runtime_root.resolve()

    def read_anchored(
        field: str, label: str,
    ) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, str]]:
        expected_cell_id = freeze_entry.get("cell_id")
        anchors = _validate_source_fold_anchors_artifact(
            freeze_entry.get(field), label=label,
            expected_cell_id=(
                expected_cell_id if isinstance(expected_cell_id, str) else None
            ),
        )
        sources: dict[int, Path] = {}
        for fold in CERTIFICATION_FOLDS:
            filename = f"fold_{fold}_result.json"
            record = anchors[filename]
            path = (root / Path(*PurePosixPath(record["path"]).parts)).resolve()
            if (
                not path.is_relative_to(root)
                or not path.is_file()
                or file_sha256(path) != record["sha256"]
            ):
                raise CampaignStageError(
                    f"{label} source fold differs from the selection freeze"
                )
            sources[fold] = path
        folds, aggregate, hashes = _read_certification_evidence(sources)
        expected_hashes = {
            filename: record["sha256"]
            for filename, record in anchors.items()
        }
        if hashes != expected_hashes:
            raise CampaignStageError(
                f"{label} source hashes differ from the selection freeze"
            )
        return folds, aggregate, hashes

    winner_folds, winner_aggregate, winner_hashes = read_anchored(
        "winner_source_folds", "winner",
    )
    baseline_folds, baseline_aggregate, baseline_hashes = read_anchored(
        "baseline_source_folds", "baseline",
    )
    if (
        bundle.get("held_out_folds") != winner_folds
        or bundle.get("held_out") != winner_aggregate
        or bundle.get("source_fold_sha256") != winner_hashes
        or bundle.get("baseline_held_out_folds") != baseline_folds
        or bundle.get("baseline_held_out") != baseline_aggregate
        or bundle.get("baseline_source_fold_sha256") != baseline_hashes
    ):
        raise CampaignStageError(
            "certification metrics differ from the anchored source folds"
        )


def validate_certified_runtime_binding(
    runtime_root: Path,
    selection_freeze: Mapping[str, Any],
    freeze_entry: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconcile published evidence with the pre-unblinding cell ledgers."""
    root = runtime_root.resolve()
    cell_id = freeze_entry.get("cell_id")
    if not isinstance(cell_id, str) or not cell_id:
        raise CampaignStageError("certified runtime cell identity is invalid")
    cell_root = (root / cell_id).resolve()
    if not cell_root.is_relative_to(root) or cell_root.parent != root:
        raise CampaignStageError("certified runtime cell path is invalid")
    state = load_stage_state(cell_root)
    winner = state.get("winner")
    baseline = state.get("baseline")
    certification = state.get("certification")
    if (
        state.get("phase") != "certified"
        or not isinstance(winner, Mapping)
        or not isinstance(baseline, Mapping)
        or not isinstance(certification, Mapping)
        or set(certification) != {
            "bundle", "bundle_sha256", "certified_at",
            "selection_state_sha256",
        }
    ):
        raise CampaignStageError(f"{cell_id}: certified stage state is incomplete")

    session = _agent_session_for_freeze(
        cell_root, state, str(selection_freeze.get("agent_protocol_sha256")),
    )
    process = _process_evidence(state)
    winner_hashes = {
        filename: record["sha256"]
        for filename, record in _validate_source_fold_anchors_artifact(
            freeze_entry.get("winner_source_folds"),
            label="winner", expected_cell_id=cell_id,
        ).items()
    }
    baseline_hashes = {
        filename: record["sha256"]
        for filename, record in _validate_source_fold_anchors_artifact(
            freeze_entry.get("baseline_source_folds"),
            label="baseline", expected_cell_id=cell_id,
        ).items()
    }
    state_winner_hashes = (
        baseline.get("sealed_fold_sha256")
        if winner.get("kind") == "baseline"
        else winner.get("sealed_fold_sha256")
    )
    session_record = session.get("session")
    if (
        state.get("campaign_id") != CAMPAIGN_ID
        or state.get("cell_id") != cell_id
        or state.get("cell_sha256") != freeze_entry.get("cell_sha256")
        or state.get("manifest_sha256")
        != selection_freeze.get("manifest_sha256")
        or state.get("protocol_version")
        != selection_freeze.get("protocol_version")
        or certification.get("bundle") != "certification/certify.json"
        or certification.get("bundle_sha256") != bundle.get("bundle_sha256")
        or certification.get("certified_at") != bundle.get("certified_at")
        or certification.get("selection_state_sha256")
        != freeze_entry.get("state_sha256")
        or winner.get("kind") != freeze_entry.get("winner_kind")
        or winner.get("candidate_id")
        != freeze_entry.get("winner_candidate_id")
        or winner.get("candidate_sha256")
        != freeze_entry.get("winner_candidate_sha256")
        or winner.get("promotion_node_id")
        != freeze_entry.get("winner_promotion_node_id")
        or winner.get("validation_mean")
        != freeze_entry.get("winner_validation_mean")
        or baseline.get("candidate_sha256")
        != freeze_entry.get("baseline_candidate_sha256")
        or baseline.get("validation_mean")
        != freeze_entry.get("baseline_validation_mean")
        or state_winner_hashes != winner_hashes
        or baseline.get("sealed_fold_sha256") != baseline_hashes
        or session.get("attestation_sha256")
        != freeze_entry.get("agent_session_sha256")
        or session.get("binding_sha256")
        != freeze_entry.get("agent_session_binding_sha256")
        or not isinstance(session_record, Mapping)
        or session_record.get("session_id")
        != freeze_entry.get("agent_session_id")
        or session_record.get("usage") != freeze_entry.get("agent_usage")
        or process != freeze_entry.get("process_evidence")
        or content_sha256(process) != freeze_entry.get("process_sha256")
    ):
        raise CampaignStageError(
            f"{cell_id}: published evidence differs from the certified cell state"
        )
    return state


def _verify_selection_freeze_for_cell(
    cell_root: Path, state: Mapping[str, Any],
) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    artifact = _validated_selection_freeze(cell_root.parent)
    _, agent_protocol_sha256 = _locked_agent_protocol(cell_root.parent)
    if (
        artifact.get("manifest_sha256") != state.get("manifest_sha256")
        or artifact.get("protocol_version") != state.get("protocol_version")
        or artifact.get("agent_protocol_sha256") != agent_protocol_sha256
    ):
        raise CampaignStageError("campaign selection freeze binding mismatch")
    if _roster_payload(artifact["cells"]) != _locked_manifest_roster(
        cell_root, state,
    ):
        raise CampaignStageError(
            "campaign selection freeze roster differs from the locked manifest"
        )
    entries = {
        row["cell_id"]: row
        for row in artifact["cells"]
        if isinstance(row, dict) and isinstance(row.get("cell_id"), str)
    }
    entry = entries.get(str(state.get("cell_id")))
    winner = state.get("winner")
    if not isinstance(entry, dict) or not isinstance(winner, dict):
        raise CampaignStageError("cell is absent from the global selection freeze")
    session = _agent_session_for_freeze(
        cell_root, state, agent_protocol_sha256,
    )
    process_evidence = _process_evidence(state)
    certification = state.get("certification")
    selection_state_sha256 = (
        certification.get("selection_state_sha256")
        if isinstance(certification, Mapping)
        else state.get("state_sha256")
    )
    winner_source_folds = _source_fold_anchors(
        cell_root.parent,
        _winner_sealed_sources(cell_root, state, winner),
    )
    baseline_source_folds = _source_fold_anchors(
        cell_root.parent,
        _baseline_sealed_sources(cell_root, state),
    )
    if (
        entry.get("cell_sha256") != state.get("cell_sha256")
        or entry.get("state_sha256") != selection_state_sha256
        or entry.get("selection_sha256") != winner.get("selection_sha256")
        or entry.get("winner_candidate_sha256") != winner.get("candidate_sha256")
        or entry.get("winner_candidate_id") != winner.get("candidate_id")
        or entry.get("winner_kind") != winner.get("kind")
        or entry.get("winner_promotion_node_id")
        != winner.get("promotion_node_id")
        or entry.get("winner_validation_mean") != winner.get("validation_mean")
        or entry.get("baseline_validation_mean")
        != (state.get("baseline") or {}).get("validation_mean")
        or entry.get("baseline_candidate_sha256")
        != (state.get("baseline") or {}).get("candidate_sha256")
        or entry.get("winner_source_folds") != winner_source_folds
        or entry.get("baseline_source_folds") != baseline_source_folds
        or entry.get("agent_session_sha256")
        != session.get("attestation_sha256")
        or entry.get("agent_session_id") != session["session"]["session_id"]
        or entry.get("agent_session_binding_sha256") != session["binding_sha256"]
        or entry.get("agent_usage") != session["session"]["usage"]
        or entry.get("process_sha256") != content_sha256(process_evidence)
        or entry.get("process_evidence") != process_evidence
        or not _process_matches_session(process_evidence, session)
    ):
        raise CampaignStageError("cell winner differs from the global selection freeze")
    return (
        str(artifact["freeze_sha256"]),
        str(entry["state_sha256"]),
        json.loads(json.dumps(entry)),
        artifact,
    )


def freeze_campaign_selections(
    runtime_root: Path, manifest_path: Path,
) -> dict[str, Any]:
    """Freeze all 130 validation winners before any held-out value is opened."""
    runtime_root = runtime_root.resolve()
    manifest_path = manifest_path.resolve()
    manifest = load_manifest(manifest_path)
    manifest_sha256 = file_sha256(manifest_path)
    _, agent_protocol_sha256 = _locked_agent_protocol(runtime_root)
    expected = {cell["cell_id"]: cell for cell in manifest["cells"]}
    expected_roster = _roster_payload(manifest["cells"])
    roster_sha256 = content_sha256(expected_roster)
    if len(expected) != CAMPAIGN_CELL_COUNT:
        raise CampaignStageError(
            f"selection freeze requires exactly {CAMPAIGN_CELL_COUNT} manifest cells"
        )
    with _campaign_lock(runtime_root):
        path = runtime_root / SELECTION_FREEZE_FILE
        if path.exists():
            artifact = _validated_selection_freeze(runtime_root)
            entries = {row["cell_id"]: row for row in artifact["cells"]}
            if (
                set(entries) != set(expected)
                or artifact.get("manifest_sha256") != manifest_sha256
                or artifact.get("roster_sha256") != roster_sha256
                or artifact.get("agent_protocol_sha256")
                != agent_protocol_sha256
            ):
                raise CampaignStageError("selection freeze cell roster mismatch")
            for cell_id, cell in expected.items():
                state = load_stage_state(runtime_root / cell_id)
                winner = state.get("winner") or {}
                entry = entries[cell_id]
                session = _agent_session_for_freeze(
                    runtime_root / cell_id, state, agent_protocol_sha256,
                )
                process_evidence = _process_evidence(state)
                winner_source_folds = _source_fold_anchors(
                    runtime_root,
                    _winner_sealed_sources(runtime_root / cell_id, state, winner),
                )
                baseline_source_folds = _source_fold_anchors(
                    runtime_root,
                    _baseline_sealed_sources(runtime_root / cell_id, state),
                )
                if (
                    state.get("cell_sha256") != cell["cell_sha256"]
                    or entry.get("state_sha256") != state.get("state_sha256")
                    or artifact.get("protocol_version")
                    != state.get("protocol_version")
                    or entry.get("selection_sha256")
                    != winner.get("selection_sha256")
                    or entry.get("winner_kind") != winner.get("kind")
                    or entry.get("winner_candidate_id")
                    != winner.get("candidate_id")
                    or entry.get("winner_promotion_node_id")
                    != winner.get("promotion_node_id")
                    or entry.get("winner_validation_mean")
                    != winner.get("validation_mean")
                    or entry.get("baseline_validation_mean")
                    != (state.get("baseline") or {}).get("validation_mean")
                    or entry.get("baseline_candidate_sha256")
                    != (state.get("baseline") or {}).get("candidate_sha256")
                    or entry.get("winner_source_folds") != winner_source_folds
                    or entry.get("baseline_source_folds")
                    != baseline_source_folds
                    or entry.get("winner_candidate_sha256")
                    != winner.get("candidate_sha256")
                    or entry.get("agent_session_sha256")
                    != session.get("attestation_sha256")
                    or entry.get("agent_session_id")
                    != session["session"]["session_id"]
                    or entry.get("agent_session_binding_sha256")
                    != session["binding_sha256"]
                    or entry.get("agent_usage")
                    != session["session"]["usage"]
                    or entry.get("process_sha256")
                    != content_sha256(process_evidence)
                    or entry.get("process_evidence") != process_evidence
                    or not _process_matches_session(process_evidence, session)
                ):
                    raise CampaignStageError(
                        f"{cell_id}: winner drift after campaign selection freeze"
                    )
            session_ids = [entries[cell_id].get("agent_session_id") for cell_id in expected]
            if (
                any(not isinstance(session_id, str) or not session_id for session_id in session_ids)
                or len(set(session_ids)) != CAMPAIGN_CELL_COUNT
            ):
                raise CampaignStageError("campaign cells do not use distinct agent sessions")
            return artifact

        actual = {path.name for path in runtime_root.iterdir() if path.is_dir()}
        if actual != set(expected):
            missing = sorted(set(expected) - actual)
            unexpected = sorted(actual - set(expected))
            raise CampaignStageError(
                "runtime roster differs from manifest "
                f"(missing={missing[:3]}, unexpected={unexpected[:3]})"
            )
        entries: list[dict[str, Any]] = []
        for cell_id in sorted(expected):
            cell = expected[cell_id]
            state = load_stage_state(runtime_root / cell_id)
            winner = state.get("winner")
            if (
                state.get("phase") != "winner-frozen"
                or state.get("certification") is not None
                or not isinstance(winner, dict)
            ):
                raise CampaignStageError(
                    f"{cell_id}: all winners must be frozen before certification"
                )
            if (
                state.get("manifest_sha256") != manifest_sha256
                or state.get("cell_sha256") != cell["cell_sha256"]
            ):
                raise CampaignStageError(f"{cell_id}: campaign binding drift")
            if state.get("protocol_version") != PROTOCOL_VERSION:
                raise CampaignStageError(f"{cell_id}: protocol version drift")
            session = _agent_session_for_freeze(
                runtime_root / cell_id, state, agent_protocol_sha256,
            )
            process_evidence = _process_evidence(state)
            if not _process_matches_session(process_evidence, session):
                raise CampaignStageError(
                    f"{cell_id}: process evidence belongs to another agent session"
                )
            winner_sources = _source_fold_anchors(
                runtime_root,
                _winner_sealed_sources(runtime_root / cell_id, state, winner),
            )
            baseline_sources = _source_fold_anchors(
                runtime_root,
                _baseline_sealed_sources(runtime_root / cell_id, state),
            )
            entries.append({
                "cell_id": cell_id,
                "cell_sha256": state["cell_sha256"],
                "state_sha256": state["state_sha256"],
                "selection_sha256": winner["selection_sha256"],
                "winner_kind": winner["kind"],
                "winner_candidate_id": winner["candidate_id"],
                "winner_candidate_sha256": winner["candidate_sha256"],
                "winner_promotion_node_id": winner.get("promotion_node_id"),
                "winner_validation_mean": winner["validation_mean"],
                "baseline_validation_mean": state["baseline"]["validation_mean"],
                "baseline_candidate_sha256": state["baseline"]["candidate_sha256"],
                "winner_source_folds": winner_sources,
                "baseline_source_folds": baseline_sources,
                "agent_session_sha256": session["attestation_sha256"],
                "agent_session_id": session["session"]["session_id"],
                "agent_session_binding_sha256": session["binding_sha256"],
                "agent_usage": session["session"]["usage"],
                "process_sha256": content_sha256(process_evidence),
                "process_evidence": process_evidence,
            })
        session_ids = [entry["agent_session_id"] for entry in entries]
        if len(set(session_ids)) != CAMPAIGN_CELL_COUNT:
            raise CampaignStageError("campaign cells do not use distinct agent sessions")
        artifact: dict[str, Any] = {
            "schema_version": SELECTION_FREEZE_SCHEMA_VERSION,
            "campaign_id": CAMPAIGN_ID,
            "manifest_sha256": manifest_sha256,
            "protocol_version": PROTOCOL_VERSION,
            "agent_protocol_sha256": agent_protocol_sha256,
            "roster_sha256": roster_sha256,
            "cell_count": len(entries),
            "cells": entries,
            "frozen_at": _utc_now(),
        }
        artifact["freeze_sha256"] = content_sha256(artifact)
        validated = validate_selection_freeze_artifact(artifact)
        _atomic_write_json(path, validated)
        return validated


def _validated_campaign_certification_index(
    runtime_root: Path,
    *,
    expected_ids: list[str],
    manifest_sha256: str,
    selection_freeze: Mapping[str, Any],
    freeze_entries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate and return an already-published immutable campaign index."""
    path = runtime_root / CAMPAIGN_CERTIFICATION_FILE
    try:
        index = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignStageError(
            "existing campaign certification index is unreadable"
        ) from exc
    expected_fields = {
        "schema_version", "campaign_id", "manifest_sha256",
        "selection_freeze_sha256", "cell_count", "cells", "certified_at",
        "certification_sha256",
    }
    if not isinstance(index, dict) or set(index) != expected_fields:
        raise CampaignStageError("campaign certification index schema mismatch")
    recorded = index.get("certification_sha256")
    payload = {
        key: value for key, value in index.items()
        if key != "certification_sha256"
    }
    cells = index.get("cells")
    try:
        certified_at = datetime.fromisoformat(str(index.get("certified_at")))
    except ValueError:
        certified_at = None
    if (
        index.get("schema_version") != 1
        or index.get("campaign_id") != CAMPAIGN_ID
        or index.get("manifest_sha256") != manifest_sha256
        or index.get("selection_freeze_sha256")
        != selection_freeze.get("freeze_sha256")
        or index.get("cell_count") != CAMPAIGN_CELL_COUNT
        or not isinstance(cells, list)
        or len(cells) != CAMPAIGN_CELL_COUNT
        or certified_at is None
        or certified_at.tzinfo is None
        or recorded != content_sha256(payload)
    ):
        raise CampaignStageError("campaign certification index integrity mismatch")
    entries: dict[str, Mapping[str, Any]] = {}
    for entry in cells:
        if not isinstance(entry, Mapping) or set(entry) != {
            "cell_id", "bundle", "bundle_sha256", "file_sha256",
        }:
            raise CampaignStageError("campaign certification entry is malformed")
        cell_id = entry.get("cell_id")
        if not isinstance(cell_id, str) or cell_id in entries:
            raise CampaignStageError("campaign certification cell identity is invalid")
        entries[cell_id] = entry
    if sorted(entries) != expected_ids:
        raise CampaignStageError("campaign certification roster mismatch")
    for cell_id in expected_ids:
        entry = entries[cell_id]
        expected_bundle = f"{cell_id}/certification/certify.json"
        if entry.get("bundle") != expected_bundle:
            raise CampaignStageError(
                f"{cell_id}: certification bundle path is not canonical"
            )
        bundle_path = runtime_root / Path(*PurePosixPath(expected_bundle).parts)
        try:
            bundle = json.loads(bundle_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise CampaignStageError(
                f"{cell_id}: indexed certification bundle is unreadable"
            ) from exc
        bundle = validate_certification_bundle_binding(
            bundle,
            freeze_entries.get(cell_id),
            selection_freeze_sha256=str(selection_freeze["freeze_sha256"]),
        )
        validate_certification_timestamp_order(
            selection_freeze["frozen_at"], bundle, index["certified_at"],
        )
        validate_certification_source_bindings(
            runtime_root, bundle, freeze_entries[cell_id],
        )
        validate_certified_runtime_binding(
            runtime_root, selection_freeze, freeze_entries[cell_id], bundle,
        )
        bundle_recorded = bundle["bundle_sha256"]
        if (
            file_sha256(bundle_path) != entry.get("file_sha256")
            or bundle_recorded != entry.get("bundle_sha256")
        ):
            raise CampaignStageError(
                f"{cell_id}: indexed certification bundle integrity mismatch"
            )
    return index


def certify_campaign(
    runtime_root: Path, manifest_path: Path,
) -> dict[str, Any]:
    """Certify every frozen cell and publish one complete, hashed bundle index."""
    runtime_root = runtime_root.resolve()
    manifest_path = manifest_path.resolve()
    manifest = load_manifest(manifest_path)
    expected_ids = sorted(cell["cell_id"] for cell in manifest["cells"])
    if len(expected_ids) != CAMPAIGN_CELL_COUNT:
        raise CampaignStageError(
            f"campaign certification requires {CAMPAIGN_CELL_COUNT} cells"
        )
    with _campaign_lock(runtime_root):
        manifest_sha256 = file_sha256(manifest_path)
        freeze = _validated_selection_freeze(runtime_root)
        if (
            freeze.get("manifest_sha256") != manifest_sha256
            or freeze.get("roster_sha256")
            != content_sha256(_roster_payload(manifest["cells"]))
            or sorted(row["cell_id"] for row in freeze["cells"]) != expected_ids
        ):
            raise CampaignStageError("campaign certification freeze roster mismatch")
        index_path = runtime_root / CAMPAIGN_CERTIFICATION_FILE
        if index_path.exists():
            freeze_entries = {
                str(row["cell_id"]): row for row in freeze["cells"]
            }
            return _validated_campaign_certification_index(
                runtime_root,
                expected_ids=expected_ids,
                manifest_sha256=manifest_sha256,
                selection_freeze=freeze,
                freeze_entries=freeze_entries,
            )
        freeze_entries = {
            str(row["cell_id"]): row for row in freeze["cells"]
        }
        entries: list[dict[str, Any]] = []
        certified_bundles: list[dict[str, Any]] = []
        for cell_id in expected_ids:
            cell_root = runtime_root / cell_id
            state = certify_winner(cell_root)
            certification = state.get("certification")
            if state.get("phase") != "certified" or not isinstance(
                certification, dict
            ):
                raise CampaignStageError(f"{cell_id}: certification did not complete")
            bundle_path = cell_root / str(certification["bundle"])
            try:
                bundle = json.loads(bundle_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise CampaignStageError(
                    f"{cell_id}: cannot index certification bundle: {exc}"
                ) from exc
            bundle = validate_certification_bundle_binding(
                bundle,
                freeze_entries[cell_id],
                selection_freeze_sha256=str(freeze["freeze_sha256"]),
            )
            validate_certification_source_bindings(
                runtime_root, bundle, freeze_entries[cell_id],
            )
            validate_certified_runtime_binding(
                runtime_root, freeze, freeze_entries[cell_id], bundle,
            )
            recorded = bundle["bundle_sha256"]
            if (
                recorded != certification.get("bundle_sha256")
                or bundle.get("selection_freeze_sha256")
                != freeze["freeze_sha256"]
            ):
                raise CampaignStageError(
                    f"{cell_id}: certification bundle integrity mismatch"
                )
            entries.append({
                "cell_id": cell_id,
                "bundle": bundle_path.relative_to(runtime_root).as_posix(),
                "bundle_sha256": recorded,
                "file_sha256": file_sha256(bundle_path),
            })
            certified_bundles.append(bundle)
        index_certified_at = _utc_now()
        for bundle in certified_bundles:
            validate_certification_timestamp_order(
                freeze["frozen_at"], bundle, index_certified_at,
            )
        index: dict[str, Any] = {
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "manifest_sha256": manifest_sha256,
            "selection_freeze_sha256": freeze["freeze_sha256"],
            "cell_count": len(entries),
            "cells": entries,
            "certified_at": index_certified_at,
        }
        index["certification_sha256"] = content_sha256(index)
        _atomic_write_json(runtime_root / CAMPAIGN_CERTIFICATION_FILE, index)
        return index


def certify_winner(cell_root: Path) -> dict[str, Any]:
    """Reveal exactly the already-frozen winner's existing five sealed folds."""
    with _stage_lock(cell_root):
        return _certify_winner_unlocked(cell_root)


def _certify_winner_unlocked(cell_root: Path) -> dict[str, Any]:
    state = load_stage_state(cell_root)
    (
        selection_freeze_sha256,
        selection_state_sha256,
        freeze_entry,
        selection_freeze,
    ) = (
        _verify_selection_freeze_for_cell(cell_root, state)
    )
    certification = state.get("certification")
    if certification is not None:
        if (
            not isinstance(certification, Mapping)
            or set(certification) != {
                "bundle", "bundle_sha256", "certified_at",
                "selection_state_sha256",
            }
            or certification.get("bundle") != "certification/certify.json"
        ):
            raise CampaignStageError("existing certification state is malformed")
        bundle_path = cell_root / certification["bundle"]
        try:
            bundle = json.loads(bundle_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise CampaignStageError(f"cannot verify existing certification: {exc}") from exc
        bundle = validate_certification_bundle_binding(
            bundle,
            freeze_entry,
            selection_freeze_sha256=selection_freeze_sha256,
        )
        recorded = bundle["bundle_sha256"]
        if (
            recorded != certification["bundle_sha256"]
        ):
            raise CampaignStageError("existing certification bundle hash mismatch")
        validate_certification_source_bindings(
            cell_root.parent, bundle, freeze_entry,
        )
        validate_certification_timestamp_order(
            selection_freeze["frozen_at"], bundle,
        )
        validate_certified_runtime_binding(
            cell_root.parent, selection_freeze, freeze_entry, bundle,
        )
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
        recovered = validate_certification_bundle_binding(
            recovered,
            freeze_entry,
            selection_freeze_sha256=selection_freeze_sha256,
        )
        recorded = recovered["bundle_sha256"]
        validate_certification_source_bindings(
            cell_root.parent, recovered, freeze_entry,
        )
        validate_certification_timestamp_order(
            selection_freeze["frozen_at"], recovered,
        )
        return _finalize_certification_state(
            cell_root, state, bundle_sha256=recorded,
            certified_at=recovered["certified_at"],
            selection_state_sha256=selection_state_sha256,
        )
    winner_sources = _winner_sealed_sources(cell_root, state, winner)
    baseline_sources = _baseline_sealed_sources(cell_root, state)
    held_out_folds, aggregate, source_hashes = _read_certification_evidence(
        winner_sources
    )
    baseline_folds, baseline_aggregate, baseline_source_hashes = (
        _read_certification_evidence(baseline_sources)
    )
    if set(aggregate) != set(baseline_aggregate):
        raise CampaignStageError(
            "winner and baseline held-out metric keys differ"
        )
    paired_fold_deltas = []
    for winner_fold, baseline_fold in zip(
        held_out_folds, baseline_folds, strict=True,
    ):
        if winner_fold["fold_index"] != baseline_fold["fold_index"]:
            raise CampaignStageError("winner and baseline held-out folds are not paired")
        paired_fold_deltas.append({
            "fold_index": winner_fold["fold_index"],
            "held_out_delta": {
                key: winner_fold["held_out"][key] - baseline_fold["held_out"][key]
                for key in sorted(aggregate)
            },
        })
    held_out_lift = {
        key: math.fsum(
            fold["held_out_delta"][key] for fold in paired_fold_deltas
        ) / len(paired_fold_deltas)
        for key in sorted(aggregate)
    }
    certified_at = _utc_now()
    bundle: dict[str, Any] = {
        "schema_version": 2,
        "campaign_id": CAMPAIGN_ID,
        "cell_id": state["cell_id"],
        "winner": {
            "kind": winner["kind"],
            "candidate_id": winner["candidate_id"],
            "candidate_sha256": winner["candidate_sha256"],
            "promotion_node_id": winner.get("promotion_node_id"),
        },
        "selection_sha256": winner["selection_sha256"],
        "selection_freeze_sha256": selection_freeze_sha256,
        "selection_state_sha256": selection_state_sha256,
        "validation_mean": winner["validation_mean"],
        "baseline": {
            "candidate_id": "baseline",
            "candidate_sha256": state["baseline"]["candidate_sha256"],
            "validation_mean": state["baseline"]["validation_mean"],
        },
        "held_out_folds": held_out_folds,
        "held_out": aggregate,
        "source_fold_sha256": source_hashes,
        "baseline_held_out_folds": baseline_folds,
        "baseline_held_out": baseline_aggregate,
        "baseline_source_fold_sha256": baseline_source_hashes,
        "paired_fold_deltas": paired_fold_deltas,
        "held_out_lift": held_out_lift,
        "retrained": False,
        "certified_at": certified_at,
    }
    bundle["bundle_sha256"] = content_sha256(bundle)
    bundle = validate_certification_bundle_binding(
        bundle,
        freeze_entry,
        selection_freeze_sha256=selection_freeze_sha256,
    )
    validate_certification_source_bindings(
        cell_root.parent, bundle, freeze_entry,
    )
    validate_certification_timestamp_order(
        selection_freeze["frozen_at"], bundle,
    )
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
        selection_state_sha256=selection_state_sha256,
    )


def _finalize_certification_state(
    cell_root: Path,
    state: dict[str, Any],
    *,
    bundle_sha256: str,
    certified_at: str,
    selection_state_sha256: str,
) -> dict[str, Any]:
    winner = state["winner"]
    state["certification"] = {
        "bundle": "certification/certify.json",
        "bundle_sha256": bundle_sha256,
        "certified_at": certified_at,
        "selection_state_sha256": selection_state_sha256,
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
