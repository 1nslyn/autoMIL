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
from datetime import datetime, timezone
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
from automil.cells.state import read_cell
from automil.cells.state import Cell, CellStatus, write_cell

from autobench.campaign import (
    AGENT_PROTOCOL_FILE,
    CAMPAIGN_ID,
    CERTIFICATION_FOLDS,
    DATASETS,
    DISCOVERY_ATTEMPTS,
    PROMOTION_CANDIDATES,
    PROTOCOL,
    STAGE_FOLDS,
    content_sha256,
    file_sha256,
    load_manifest,
    validate_agent_protocol,
)

STATE_SCHEMA_VERSION = 2
STATE_FILE = "campaign_state.json"
BASELINE_ATTESTATION_FILE = "baseline_attestation.json"
SELECTION_FREEZE_FILE = "selection_freeze.json"
CAMPAIGN_CERTIFICATION_FILE = "campaign_certification.json"
AGENT_SESSION_FILE = "agent_session.json"
CAMPAIGN_CELL_COUNT = len(DATASETS) * 26


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
    base_commit: str,
) -> dict[str, Any]:
    """Create the immutable discovery ledger, or verify an identical restart."""
    with _stage_lock(cell_root):
        return _initialize_stage_state_unlocked(
            cell_root, cell=cell, manifest_sha256=manifest_sha256,
            base_commit=base_commit,
        )


def _initialize_stage_state_unlocked(
    cell_root: Path,
    *,
    cell: Mapping[str, Any],
    manifest_sha256: str,
    base_commit: str,
) -> dict[str, Any]:
    path = cell_root / STATE_FILE
    if path.exists():
        state = load_stage_state(cell_root)
        expected = (
            cell["cell_id"], cell["cell_sha256"], manifest_sha256, base_commit,
        )
        actual = (
            state.get("cell_id"), state.get("cell_sha256"),
            state.get("manifest_sha256"), state.get("base_commit"),
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
        "base_commit": base_commit,
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
    if cell.get("cell_id") != state["cell_id"]:
        raise CampaignStageError("baseline campaign cell differs from stage state")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "manifest_sha256": state["manifest_sha256"],
        "cell_id": state["cell_id"],
        "cell_sha256": state["cell_sha256"],
        "base_commit": state["base_commit"],
        "baseline_command": cell["commands"]["baseline"],
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


def _sealed_fold_hashes(
    archive: Path, expected_folds: tuple[int, ...],
) -> dict[str, str]:
    """Hash sealed folds opaquely before validation can select a candidate."""
    hashes: dict[str, str] = {}
    for fold in expected_folds:
        filename = f"fold_{fold}_result.json"
        path = archive / "certify" / filename
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
            "baseline attestation is not bound to this cell/base/command/artifact set"
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

    Training executes in a detached worktree at the materialized base commit.
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
                _ensure_discovery_baseline_root(cell_root, state, state["baseline"])
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
                        state["base_commit"],
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
        "base_commit": spec.get("base_commit"),
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
        if spec.get("base_commit") != state["base_commit"]:
            raise CampaignStageError(
                f"discovery spec {archive.name} differs from the frozen base commit"
            )
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
            base_commit=state["base_commit"],
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
                if source_spec.get("base_commit") != state["base_commit"]:
                    raise CampaignStageError(
                        f"source candidate base commit drifted: {source_node}"
                    )
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
        missing = [
            name for name, path in (("spec.json", spec_path), ("result.json", result_path))
            if not path.is_file()
        ]
        if missing:
            job.update({
                "status": "ineligible",
                "reason": f"terminal promotion is missing {', '.join(missing)}",
            })
            frozen_jobs.append(job)
            continue
        try:
            spec = json.loads(spec_path.read_text())
            result = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            job.update({
                "status": "ineligible",
                "reason": f"terminal promotion artifact is unreadable: {exc}",
            })
            frozen_jobs.append(job)
            continue
        if spec.get("base_commit") != state["base_commit"]:
            raise CampaignStageError(f"promotion base commit drifted for {node_id}")
        link = (spec.get("metadata") or {}).get("promotion") or {}
        if (
            link.get("source_node_id") != source_node
            or link.get("source_candidate_sha256") != source["candidate_sha256"]
            or link.get("source_spec_sha256") != source["source_spec_sha256"]
            or link.get("expected_folds") != list(STAGE_FOLDS["promotion"])
        ):
            raise CampaignStageError(f"promotion source link drifted for {node_id}")
        if result.get("status") != "completed":
            job.update({
                "status": "ineligible",
                "reason": f"promotion status {result.get('status')!r}",
            })
            frozen_jobs.append(job)
            continue
        verdict = revalidate_candidate_spec(policy, spec, archive).to_dict()
        promotion_sha, _ = _candidate_identity(spec, verdict)
        if promotion_sha != job["promotion_candidate_sha256"]:
            raise CampaignStageError(f"promotion candidate identity drifted for {node_id}")
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
    if discovery_spec.get("base_commit") != state["base_commit"]:
        raise CampaignStageError("winner discovery base commit changed")
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
    if promotion_spec.get("base_commit") != state["base_commit"]:
        raise CampaignStageError("winner promotion base commit changed")
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
        _verify_baseline_unchanged(cell_root, baseline)
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
    _verify_baseline_unchanged(cell_root, baseline)
    archive = (cell_root / str(baseline["archive"])).resolve()
    return {
        fold: archive / "certify" / f"fold_{fold}_result.json"
        for fold in CERTIFICATION_FOLDS
    }


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
        protocol = validate_agent_protocol(raw, allow_canary=True)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise CampaignStageError(f"cannot verify locked agent protocol: {exc}") from exc
    return protocol, content_sha256(protocol)


def _validate_agent_session(
    payload: Mapping[str, Any], *, state: Mapping[str, Any],
    agent_protocol_sha256: str,
) -> dict[str, Any]:
    required = {
        "schema_version", "campaign_id", "cell_id", "agent_protocol_sha256",
        "sessions", "attestation_sha256",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise CampaignStageError("agent session attestation field set is not exact")
    recorded = payload.get("attestation_sha256")
    canonical = {
        key: value for key, value in payload.items()
        if key != "attestation_sha256"
    }
    if (
        payload.get("schema_version") != 1
        or payload.get("campaign_id") != CAMPAIGN_ID
        or payload.get("cell_id") != state.get("cell_id")
        or payload.get("agent_protocol_sha256") != agent_protocol_sha256
        or recorded != content_sha256(canonical)
    ):
        raise CampaignStageError("agent session attestation binding mismatch")
    sessions = payload.get("sessions")
    if not isinstance(sessions, list) or len(sessions) != 1:
        raise CampaignStageError("preprint requires exactly one agent session per cell")
    session = sessions[0]
    session_fields = {
        "session_id", "started_at", "ended_at", "termination_reason", "usage",
    }
    if not isinstance(session, dict) or set(session) != session_fields:
        raise CampaignStageError("agent session record field set is not exact")
    for key in ("session_id", "termination_reason"):
        if not isinstance(session.get(key), str) or not session[key].strip():
            raise CampaignStageError(f"agent session {key} is invalid")
    try:
        started = datetime.fromisoformat(str(session["started_at"]))
        ended = datetime.fromisoformat(str(session["ended_at"]))
    except ValueError as exc:
        raise CampaignStageError("agent session timestamps are invalid") from exc
    if started.tzinfo is None or ended.tzinfo is None or ended < started:
        raise CampaignStageError("agent session time interval is invalid")
    usage = session.get("usage")
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
    return json.loads(json.dumps(payload))


def register_agent_session(
    cell_root: Path, attestation: Mapping[str, Any],
) -> dict[str, Any]:
    """Import the one runtime/resource attestation allowed for this cell."""
    with _stage_lock(cell_root):
        state = load_stage_state(cell_root)
        if state.get("phase") != "winner-frozen":
            raise CampaignStageError("agent session is registered after winner freeze")
        _, protocol_sha256 = _locked_agent_protocol(cell_root.parent)
        config = yaml.safe_load((cell_root / "automil/config.yaml").read_text()) or {}
        if (config.get("campaign") or {}).get(
            "agent_protocol_sha256"
        ) != protocol_sha256:
            raise CampaignStageError("cell config agent protocol binding mismatch")
        prepared = json.loads(json.dumps(attestation))
        if "attestation_sha256" not in prepared:
            prepared["attestation_sha256"] = content_sha256(prepared)
        validated = _validate_agent_session(
            prepared, state=state, agent_protocol_sha256=protocol_sha256,
        )
        target = cell_root / AGENT_SESSION_FILE
        if target.exists():
            try:
                existing = json.loads(target.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise CampaignStageError("existing agent session is corrupt") from exc
            if existing != validated:
                raise CampaignStageError("agent session attestation is immutable")
            return validated
        _atomic_write_json(target, validated)
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
    )


def _validated_selection_freeze(runtime_root: Path) -> dict[str, Any]:
    path = runtime_root / SELECTION_FREEZE_FILE
    try:
        artifact = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignStageError(
            "held-out certification requires the global 130-cell selection freeze"
        ) from exc
    if not isinstance(artifact, dict):
        raise CampaignStageError("campaign selection freeze must be a JSON object")
    recorded = artifact.get("freeze_sha256")
    payload = {
        key: value for key, value in artifact.items() if key != "freeze_sha256"
    }
    cells = artifact.get("cells")
    if (
        artifact.get("schema_version") != 1
        or artifact.get("campaign_id") != CAMPAIGN_ID
        or artifact.get("protocol_sha256") != content_sha256(PROTOCOL)
        or not isinstance(artifact.get("agent_protocol_sha256"), str)
        or len(artifact["agent_protocol_sha256"]) != 64
        or artifact.get("cell_count") != CAMPAIGN_CELL_COUNT
        or not isinstance(cells, list)
        or len(cells) != CAMPAIGN_CELL_COUNT
        or len({row.get("cell_id") for row in cells if isinstance(row, dict)})
        != CAMPAIGN_CELL_COUNT
        or not isinstance(recorded, str)
        or recorded != content_sha256(payload)
    ):
        raise CampaignStageError("campaign selection freeze integrity mismatch")
    return artifact


def _verify_selection_freeze_for_cell(
    cell_root: Path, state: Mapping[str, Any],
) -> str:
    artifact = _validated_selection_freeze(cell_root.parent)
    _, agent_protocol_sha256 = _locked_agent_protocol(cell_root.parent)
    if (
        artifact.get("manifest_sha256") != state.get("manifest_sha256")
        or artifact.get("base_commit") != state.get("base_commit")
        or artifact.get("agent_protocol_sha256") != agent_protocol_sha256
    ):
        raise CampaignStageError("campaign selection freeze binding mismatch")
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
    if (
        entry.get("cell_sha256") != state.get("cell_sha256")
        or entry.get("selection_sha256") != winner.get("selection_sha256")
        or entry.get("winner_candidate_sha256") != winner.get("candidate_sha256")
        or entry.get("winner_candidate_id") != winner.get("candidate_id")
        or entry.get("winner_kind") != winner.get("kind")
        or entry.get("agent_session_sha256")
        != session.get("attestation_sha256")
        or entry.get("agent_usage") != session["sessions"][0]["usage"]
    ):
        raise CampaignStageError("cell winner differs from the global selection freeze")
    return str(artifact["freeze_sha256"])


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
                if (
                    state.get("cell_sha256") != cell["cell_sha256"]
                    or entry.get("selection_sha256")
                    != winner.get("selection_sha256")
                    or entry.get("winner_candidate_sha256")
                    != winner.get("candidate_sha256")
                    or entry.get("agent_session_sha256")
                    != session.get("attestation_sha256")
                    or entry.get("agent_usage")
                    != session["sessions"][0]["usage"]
                ):
                    raise CampaignStageError(
                        f"{cell_id}: winner drift after campaign selection freeze"
                    )
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
        base_commits: set[str] = set()
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
            base_commits.add(str(state.get("base_commit", "")))
            session = _agent_session_for_freeze(
                runtime_root / cell_id, state, agent_protocol_sha256,
            )
            entries.append({
                "cell_id": cell_id,
                "cell_sha256": state["cell_sha256"],
                "state_sha256": state["state_sha256"],
                "selection_sha256": winner["selection_sha256"],
                "winner_kind": winner["kind"],
                "winner_candidate_id": winner["candidate_id"],
                "winner_candidate_sha256": winner["candidate_sha256"],
                "agent_session_sha256": session["attestation_sha256"],
                "agent_usage": session["sessions"][0]["usage"],
            })
        if len(base_commits) != 1 or "" in base_commits:
            raise CampaignStageError("campaign cells do not share one base commit")
        artifact: dict[str, Any] = {
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "manifest_sha256": manifest_sha256,
            "protocol_sha256": content_sha256(PROTOCOL),
            "agent_protocol_sha256": agent_protocol_sha256,
            "base_commit": next(iter(base_commits)),
            "cell_count": len(entries),
            "cells": entries,
            "frozen_at": _utc_now(),
        }
        artifact["freeze_sha256"] = content_sha256(artifact)
        _atomic_write_json(path, artifact)
        return artifact


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
    freeze = _validated_selection_freeze(runtime_root)
    if (
        freeze.get("manifest_sha256") != file_sha256(manifest_path)
        or sorted(row["cell_id"] for row in freeze["cells"]) != expected_ids
    ):
        raise CampaignStageError("campaign certification freeze roster mismatch")

    with _campaign_lock(runtime_root):
        entries: list[dict[str, Any]] = []
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
            recorded = bundle.pop("bundle_sha256", None)
            if (
                recorded != content_sha256(bundle)
                or recorded != certification.get("bundle_sha256")
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
        index: dict[str, Any] = {
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "manifest_sha256": file_sha256(manifest_path),
            "selection_freeze_sha256": freeze["freeze_sha256"],
            "cell_count": len(entries),
            "cells": entries,
            "certified_at": _utc_now(),
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
    selection_freeze_sha256 = _verify_selection_freeze_for_cell(cell_root, state)
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
            or bundle.get("selection_freeze_sha256")
            != selection_freeze_sha256
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
            or recovered.get("selection_freeze_sha256")
            != selection_freeze_sha256
        ):
            raise CampaignStageError("certification bundle is not bound to the winner")
        winner_sources = _winner_sealed_sources(cell_root, state, winner)
        winner_hashes = {
            f"fold_{fold}_result.json": file_sha256(winner_sources[fold])
            for fold in CERTIFICATION_FOLDS
        }
        baseline_sources = _baseline_sealed_sources(cell_root, state)
        baseline_hashes = {
            f"fold_{fold}_result.json": file_sha256(baseline_sources[fold])
            for fold in CERTIFICATION_FOLDS
        }
        if recovered.get("source_fold_sha256") != winner_hashes:
            raise CampaignStageError(
                "certification bundle source hashes differ from the frozen winner"
            )
        if recovered.get("baseline_source_fold_sha256") != baseline_hashes:
            raise CampaignStageError(
                "certification bundle source hashes differ from the native baseline"
            )
        if (
            recovered.get("schema_version") != 2
            or (recovered.get("baseline") or {}).get("candidate_sha256")
            != state["baseline"]["candidate_sha256"]
        ):
            raise CampaignStageError("certification bundle baseline binding mismatch")
        return _finalize_certification_state(
            cell_root, state, bundle_sha256=recorded,
            certified_at=recovered["certified_at"],
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
