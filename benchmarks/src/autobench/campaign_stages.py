"""Restart-safe stage ledger for the frozen preprint campaign.

This module is the trusted consumer-side controller.  It never imports or
opens held-out values during search: baseline registration hashes sealed files
without parsing them, while discovery freeze reads only agent-facing
``result.json`` files containing validation-only fold evidence.
"""
from __future__ import annotations

import fcntl
import json
import math
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from automil.admissibility import (
    AdmissibilityError,
    load_candidate_policy,
    revalidate_candidate_spec,
)
from automil.cells.state import read_cell

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
        },
        "winner": None,
        "certification": None,
        "history": [{"event": "initialized", "at": now}],
    }
    return _commit_state(cell_root, state)


def _relative_source(cell_root: Path, source: Path) -> str:
    source = source.resolve()
    try:
        return source.relative_to(cell_root.resolve()).as_posix()
    except ValueError:
        relative = os.path.relpath(source, cell_root.resolve())
        if relative.startswith("../"):
            raise CampaignStageError(
                "baseline archive must live inside the campaign cell root"
            )
        return Path(relative).as_posix()


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
    return sum(float(fold["composite"]) for fold in folds) / len(folds)


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
    identity_payload = {
        "result_sha256": file_sha256(result_path),
        "sealed_fold_sha256": sealed_hashes,
        "validation_folds": folds,
    }
    baseline = {
        "candidate_id": "baseline",
        "candidate_sha256": content_sha256(identity_payload),
        "archive": _relative_source(cell_root, baseline_archive),
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
        return state
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
