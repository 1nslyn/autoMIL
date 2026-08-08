"""Restart-safe discovery ledger and validation-firewall campaign tests."""
from __future__ import annotations

import json
import hashlib
import shutil
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest
import yaml

import autobench.campaign_stages as campaign_stages
from automil.admissibility import load_candidate_policy
from automil.cells.activity import (
    ACTIVITY_SAMPLES_FILENAME,
    ingest_prometheus_metrics,
    read_activity_report,
    record_hook_event,
)
from automil.cells.state import Cell, CellStatus, make_cell_id, read_cell, write_cell
from autobench.campaign import (
    AGENT_PROTOCOL_FILE,
    CAMPAIGN_ID,
    CERTIFICATION_FOLDS,
    DISCOVERY_ATTEMPTS,
    PROTOCOL,
    PROTOCOL_VERSION,
    STAGE_FOLDS,
    SUBMIT_CLOCK_SKEW_TOLERANCE_SECONDS,
    content_sha256,
    file_sha256,
)
from autobench.campaign_stages import (
    BASELINE_ATTESTATION_FILE,
    CAMPAIGN_CELL_COUNT,
    SELECTION_FREEZE_FILE,
    SELECTION_FREEZE_SCHEMA_VERSION,
    CampaignStageError,
    _baseline_sealed_sources,
    _process_evidence,
    _source_fold_anchors,
    _winner_sealed_sources,
    certify_winner,
    finalize_agent_session,
    freeze_discovery as _freeze_discovery,
    freeze_promotion,
    initialize_stage_state,
    load_stage_state,
    materialize_promotion,
    open_agent_session,
    register_baseline,
    attest_and_register_baseline,
    run_native_baseline,
    select_winner,
    validate_selection_freeze_artifact,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_DIR = REPO_ROOT / "benchmarks/campaigns/preprint_130"


def _folds(indices, base=0.6):
    return [
        {
            "fold_index": index,
            "metrics": {"val_auc": base + index / 100, "val_bacc": base},
            "composite": base + index / 200,
        }
        for index in indices
    ]


AGENT_PROTOCOL = {
    "schema_version": 2,
    "campaign_id": CAMPAIGN_ID,
    "purpose": "publication",
    "provider": "test-provider",
    "runtime": "test-runtime",
    "runtime_version": "test-runtime-1",
    "model": "test-model",
    "model_version": "test-model-1",
    "effort": "high",
    "network_access": "enabled",
    "fallback_model": None,
    "proposal_policy_content": "test proposal policy",
    "proposal_policy_sha256": hashlib.sha256(
        b"test proposal policy"
    ).hexdigest(),
    "toolset_content": "test toolset",
    "toolset_sha256": hashlib.sha256(b"test toolset").hexdigest(),
    "max_sessions_per_cell": 1,
}


@pytest.fixture
def staged_cell(tmp_path):
    cell_root = tmp_path / "dataset__arm__task"
    adir = cell_root / "automil"
    adir.mkdir(parents=True)
    budget_cell_id = make_cell_id("dataset", "encoder", "model", "task")
    cell_without_hash = {
        "cell_id": "dataset__arm__task",
        "dataset": "dataset",
        "task": "task",
        "encoder": "encoder",
        "framework": "arm",
        "seed": 42,
        "model": "model",
        "identity": {
            "dataset": "dataset",
            "task": "task",
            "encoder": "encoder",
            "arm": "arm",
            "seed": 42,
            "protocol_version": PROTOCOL_VERSION,
        },
        "commands": {
            "baseline": (
                "python benchmarks/scripts/run_experiment.py --folds 0,1,2,3,4"
            ),
            "discovery": (
                "python benchmarks/scripts/run_experiment.py --folds 0,1,2"
            ),
            "promotion": (
                "python benchmarks/scripts/run_experiment.py --folds 3,4"
            ),
        },
        "budget_identity": {
            "cell_id": budget_cell_id,
            "dataset": "dataset",
            "encoder": "encoder",
            "mil_model": "model",
            "task": "task",
        },
    }
    cell = {**cell_without_hash, "cell_sha256": content_sha256(cell_without_hash)}
    manifest_path = tmp_path / "manifest.json"
    manifest_cells = [cell] + [
        {
            "cell_id": f"fixture-cell-{index:03d}",
            "cell_sha256": "a" * 64,
        }
        for index in range(CAMPAIGN_CELL_COUNT - 1)
    ]
    manifest_path.write_text(json.dumps({
        "schema_version": 4,
        "campaign_id": CAMPAIGN_ID,
        "cells": manifest_cells,
    }))
    manifest_hash = file_sha256(manifest_path)
    (tmp_path / AGENT_PROTOCOL_FILE).write_text(json.dumps(AGENT_PROTOCOL))
    (adir / "campaign_cell.json").write_text(json.dumps(cell))
    (adir / "config.yaml").write_text(yaml.safe_dump({
        "project": {"name": "dataset"},
        "task": {"name": "task"},
        "encoders": {"primary": "encoder"},
        "registry": {
            "mode": "architecture-preserving",
            "protected": ["models/**"],
            "allowed_override_options": ["--hparams", "--policy-variant"],
            "allowed_variant_kinds": ["policy"],
        },
        "files": {
            "editable": [
                "dataset__arm__task/automil/variants/_policies/*.py",
            ],
        },
        "run": {"command": cell["commands"]["discovery"], "mil_model": "model"},
        "cap": {
            "budget": "12h",
            "safety_buffer": "30m",
            "mode": "agent_active",
            "eval_budget": DISCOVERY_ATTEMPTS,
        },
        "campaign": {
            "campaign_id": CAMPAIGN_ID,
            "manifest": "manifest.json",
            "manifest_sha256": manifest_hash,
            "cell_id": cell["cell_id"],
            "cell_sha256": cell["cell_sha256"],
            "budget_cell_id": cell["budget_identity"]["cell_id"],
            "stage": "discovery",
            "protocol_version": PROTOCOL_VERSION,
            "agent_protocol_sha256": content_sha256(AGENT_PROTOCOL),
        },
    }))
    state = initialize_stage_state(
        cell_root, cell=cell, manifest_sha256=manifest_hash,
    )
    return cell_root, adir, cell, state, tmp_path


def _baseline(
    cell_root: Path, *, leak=False, invalid_sealed=False,
    attest_for: Path | None = None,
) -> Path:
    archive = cell_root / "baseline" / "archive"
    sealed = archive / "certify"
    sealed.mkdir(parents=True)
    result = {
        "status": "completed",
        "composite": 0.61,
        "metrics": {"val_auc": 0.62, "val_bacc": 0.60},
        "validation_folds": _folds(CERTIFICATION_FOLDS, 0.60),
    }
    if leak:
        result["held_out"] = {"test_auc": 0.99}
    (archive / "result.json").write_text(json.dumps(result))
    for fold in CERTIFICATION_FOLDS:
        payload = "not-json-and-must-not-be-parsed" if invalid_sealed else json.dumps({
            "fold_index": fold,
            "held_out": {
                "test_auc": 0.5 + fold / 100,
                "test_bacc": 0.5 + fold / 100,
            },
        })
        (sealed / f"fold_{fold}_result.json").write_text(payload)
    target = attest_for or cell_root
    state = load_stage_state(target)
    cell = json.loads((target / "automil/campaign_cell.json").read_text())
    attestation = {
        "schema_version": 2,
        "cell_id": state["cell_id"],
        "identity": cell["identity"],
        "result_sha256": file_sha256(archive / "result.json"),
        "sealed_fold_sha256": {
            f"fold_{fold}_result.json": file_sha256(
                sealed / f"fold_{fold}_result.json"
            )
            for fold in CERTIFICATION_FOLDS
        },
    }
    attestation["attestation_sha256"] = content_sha256(attestation)
    (archive / BASELINE_ATTESTATION_FILE).write_text(json.dumps(attestation))
    return archive


def _open_budget_cell(adir: Path, budget_id: str, consumed: int) -> None:
    write_cell(
        Cell(
            cell_id=budget_id,
            dataset="dataset",
            encoder="encoder",
            mil_model="model",
            started_at=1.0,
            budget_seconds=10_000,
            safety_buffer_seconds=10,
            status=CellStatus.ACTIVE,
            mode="agent_active",
            eval_budget=DISCOVERY_ATTEMPTS,
            consumed_evals=consumed,
            completed_evals=min(consumed, 12),
        ),
        adir / "cells",
    )


def _attempts(
    adir: Path, cell_id: str, *, completed=12, source_at: int | None = None,
) -> None:
    cell_root = adir.parent
    session_path = cell_root / "agent_session.json"
    if not session_path.exists():
        _record_session_start(adir)
        open_agent_session(cell_root, {
            "session_id": "fixture-session",
            "started_at": (
                datetime.now(timezone.utc) - timedelta(minutes=1)
            ).isoformat(),
        })
    session = json.loads(session_path.read_text())
    bound_at = datetime.fromisoformat(session["session"]["bound_at"])
    policy = load_candidate_policy(adir)
    archive_root = adir / "orchestrator" / "archive"
    for index in range(DISCOVERY_ATTEMPTS):
        node_id = f"node_{index + 1:04d}"
        archive = archive_root / node_id
        archive.mkdir(parents=True)
        candidate_paths = []
        overlay_manifest = {}
        if index == source_at:
            rel = (
                "dataset__arm__task/automil/variants/_policies/"
                f"candidate_{index}.py"
            )
            source = archive / rel
            source.parent.mkdir(parents=True)
            source.write_text("# exact train-only policy candidate\n")
            overlay_manifest[rel] = f"sha256:{file_sha256(source)}"
            candidate_paths = [rel]
            override = f"--policy-variant candidate_{index}"
        else:
            override = f'--hparams \'{{"lr":{0.0001 + index / 1_000_000:.7f}}}\''
        verdict = policy.classify(candidate_paths, override=override)
        spec = {
            "id": node_id,
            "base_commit": "d" * 40,
            "overlay_manifest": overlay_manifest,
            "deletions": [],
            "framework_overlay_files": [],
            "run_command_override": override,
            "admissibility": verdict.to_dict(),
            "submitted_at": (
                bound_at + timedelta(seconds=index + 1)
            ).isoformat(),
            "metadata": {
                "cell_id": yaml.safe_load(
                    (adir / "config.yaml").read_text()
                )["campaign"]["budget_cell_id"],
                "agent_session": {
                    "session_id": session["session"]["session_id"],
                    "agent_protocol_sha256": session["agent_protocol_sha256"],
                    "binding_sha256": session["binding_sha256"],
                },
                "campaign": {
                    "campaign_id": CAMPAIGN_ID,
                    "cell_id": cell_id,
                    "stage": "discovery",
                },
            },
        }
        (archive / "spec.json").write_text(json.dumps(spec))
        if index < completed:
            result = {
                "status": "completed",
                "composite": 0.5 + index / 100,
                "metrics": {"val_auc": 0.5, "val_bacc": 0.5},
                "validation_folds": _folds(
                    STAGE_FOLDS["discovery"], 0.5 + index / 100,
                ),
            }
            sealed = archive / "certify"
            sealed.mkdir()
            for fold in STAGE_FOLDS["discovery"]:
                (sealed / f"fold_{fold}_result.json").write_text(json.dumps({
                    "fold_index": fold,
                    "held_out": {
                        "test_auc": 0.70 + fold / 100,
                        "test_bacc": 0.70 + fold / 100,
                    },
                }))
        else:
            result = {"status": "crash", "composite": 0.0, "metrics": {}}
        (archive / "result.json").write_text(json.dumps(result))


def test_initial_state_is_restart_idempotent_and_integrity_checked(staged_cell):
    cell_root, _, cell, original, _ = staged_cell
    restarted = initialize_stage_state(
        cell_root, cell=cell, manifest_sha256=original["manifest_sha256"],
    )
    assert restarted == original
    assert original["protocol_version"] == PROTOCOL_VERSION
    assert "base_commit" not in original
    raw = json.loads((cell_root / "campaign_state.json").read_text())
    raw["phase"] = "winner-frozen"
    (cell_root / "campaign_state.json").write_text(json.dumps(raw))
    with pytest.raises(CampaignStageError, match="integrity hash"):
        load_stage_state(cell_root)


def test_candidate_identity_ignores_operational_git_commit():
    verdict = {
        "candidate_class": "config-only",
        "policy_hash": "p" * 64,
        "variant_selection_hash": None,
        "override_hash": "o" * 64,
    }
    common = {
        "overlay_manifest": {},
        "deletions": [],
    }
    first, first_payload = campaign_stages._candidate_identity(
        {**common, "base_commit": "a" * 40}, verdict,
    )
    second, second_payload = campaign_stages._candidate_identity(
        {**common, "base_commit": "b" * 40}, verdict,
    )
    assert first == second
    assert first_payload == second_payload
    assert "base_commit" not in first_payload


def test_baseline_registration_hashes_but_does_not_parse_sealed_test(staged_cell):
    cell_root, adir, cell, _, _ = staged_cell
    state = register_baseline(
        cell_root, _baseline(cell_root, invalid_sealed=True),
    )
    assert state["baseline"]["candidate_id"] == "baseline"
    assert len(state["baseline"]["sealed_fold_sha256"]) == 5
    assert state["baseline"]["validation_mean"] == pytest.approx(0.61)
    root_id = state["baseline"]["discovery_root_node_id"]
    graph = json.loads((adir / "graph.json").read_text())
    assert list(graph["nodes"]) == [root_id]
    root = graph["nodes"][root_id]
    assert root["parent_id"] is None
    assert root["status"] == "keep"
    assert root["composite"] == pytest.approx(0.605)
    assert root["metadata"]["cell_id"] == cell["budget_identity"]["cell_id"]
    assert [
        fold["fold_index"] for fold in root["metadata"]["validation_folds"]
    ] == list(STAGE_FOLDS["discovery"])

    restarted = register_baseline(
        cell_root, cell_root / state["baseline"]["archive"],
    )
    assert restarted == state
    assert list(json.loads((adir / "graph.json").read_text())["nodes"]) == [root_id]


def test_baseline_registration_rejects_test_bearing_public_result(staged_cell):
    cell_root, _, _, _, _ = staged_cell
    with pytest.raises(CampaignStageError, match="test-bearing"):
        register_baseline(cell_root, _baseline(cell_root, leak=True))


@pytest.mark.parametrize("invalid", [True, -1, float("nan"), "12"])
def test_baseline_registration_rejects_invalid_resource_values(
    staged_cell, invalid,
):
    cell_root, _, _, _, _ = staged_cell
    source_root = cell_root.parent / "baseline-source"
    archive = _baseline(source_root, attest_for=cell_root)
    result_path = archive / "result.json"
    result = json.loads(result_path.read_text())
    result["elapsed_seconds"] = invalid
    result_path.write_text(json.dumps(result))

    with pytest.raises(CampaignStageError, match="elapsed_seconds is invalid"):
        register_baseline(cell_root, archive)

    assert not (cell_root / "baseline").exists()


def test_native_baseline_runs_at_frozen_commit_and_registers(
    staged_cell, monkeypatch,
):
    cell_root, adir, _, _, repo_root = staged_cell
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        if command[:3] == ["git", "worktree", "add"]:
            worktree = Path(command[4])
            worktree.mkdir(parents=True)
            observed["base_commit"] = command[5]
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if command[:3] == ["git", "worktree", "remove"]:
            shutil.rmtree(Path(command[4]), ignore_errors=True)
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

        worktree = Path(kwargs["cwd"])
        env = kwargs["env"]
        observed["command"] = command
        observed["gpu"] = (
            env["CUDA_VISIBLE_DEVICES"], env["AUTOMIL_GPU"],
        )
        public = {
            "status": "completed",
            "composite": 0.61,
            "metrics": {"val_auc": 0.62, "val_bacc": 0.60},
            "validation_folds": _folds(CERTIFICATION_FOLDS, 0.60),
        }
        (worktree / "result.json").write_text(json.dumps(public))
        sealed = Path(env["AUTOMIL_RESULTS_DIR"])
        sealed.mkdir(parents=True, exist_ok=True)
        for fold in CERTIFICATION_FOLDS:
            (sealed / f"fold_{fold}_result.json").write_text(json.dumps({
                "fold_index": fold,
                "held_out": {
                    "test_auc": 0.5 + fold / 100,
                    "test_bacc": 0.5 + fold / 100,
                },
            }))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("autobench.campaign_stages.subprocess.run", fake_run)
    state = run_native_baseline(cell_root, repo_root=repo_root, gpu_id=3)

    assert observed["base_commit"] == "HEAD"
    assert observed["gpu"] == ("3", "0")
    assert "--folds" in observed["command"]
    assert "0,1,2,3,4" in observed["command"]
    assert state["baseline"]["candidate_id"] == "baseline"
    assert (cell_root / "baseline/archive/result.json").is_file()
    assert (adir / "graph.json").is_file()


def test_native_baseline_cached_registration_revalidates_local_artifacts(
    staged_cell,
):
    cell_root, _, _, _, repo_root = staged_cell
    state = register_baseline(cell_root, _baseline(cell_root))
    archive = cell_root / state["baseline"]["archive"]
    (archive / "certify/fold_5_result.json").write_text(json.dumps({
        "fold_index": 5,
        "held_out": {},
    }))

    with pytest.raises(CampaignStageError, match="sealed fold set changed"):
        run_native_baseline(cell_root, repo_root=repo_root)


def test_native_baseline_cached_registration_is_idempotent_after_freeze(
    staged_cell,
):
    cell_root, adir, cell, _, repo_root = staged_cell
    registered = register_baseline(cell_root, _baseline(cell_root))
    _attempts(adir, cell["cell_id"], completed=0)
    _open_budget_cell(
        adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS,
    )
    frozen = freeze_discovery(cell_root)

    assert frozen["phase"] == "selection-ready"
    assert run_native_baseline(cell_root, repo_root=repo_root) == frozen
    assert frozen["baseline"]["candidate_sha256"] == (
        registered["baseline"]["candidate_sha256"]
    )


def test_external_baseline_is_atomically_imported_and_portable(staged_cell, tmp_path):
    cell_root, adir, cell, _, _ = staged_cell
    external_root = tmp_path / "external-cell"
    external_adir = external_root / "automil"
    external_adir.mkdir(parents=True)
    external_cell = json.loads(json.dumps(cell))
    external_cell.pop("cell_sha256")
    external_cell["commands"]["baseline"] += " --external-provenance"
    external_cell["cell_sha256"] = content_sha256(external_cell)
    (external_adir / "campaign_cell.json").write_text(json.dumps(external_cell))
    initialize_stage_state(
        external_root, cell=external_cell, manifest_sha256="f" * 64,
    )
    external = _baseline(external_root)
    state = register_baseline(cell_root, external)
    imported = cell_root / state["baseline"]["archive"]
    assert imported == cell_root / "baseline/archive"
    assert (imported / "result.json").is_file()
    assert len(list((imported / "certify").glob("fold_*_result.json"))) == 5
    attestation = json.loads((imported / BASELINE_ATTESTATION_FILE).read_text())
    assert "campaign_id" not in attestation

    # The registered incumbent no longer depends on the external result tree.
    shutil.rmtree(external_root)
    _attempts(adir, cell["cell_id"], completed=0)
    _open_budget_cell(
        adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS,
    )
    freeze_discovery(cell_root)
    selected = select_winner(cell_root)
    assert selected["winner"]["kind"] == "baseline"


def test_locally_converted_baseline_is_attested_and_registered(staged_cell):
    cell_root, _, _, _, _ = staged_cell
    archive = _baseline(cell_root)
    (archive / BASELINE_ATTESTATION_FILE).unlink()

    state = attest_and_register_baseline(cell_root, archive)

    imported = cell_root / state["baseline"]["archive"]
    assert imported == cell_root / "baseline/archive"
    assert (imported / BASELINE_ATTESTATION_FILE).is_file()


@pytest.mark.parametrize("axis", [
    "dataset", "task", "encoder", "arm", "seed", "protocol_version",
])
def test_external_baseline_rejects_a_foreign_identity(staged_cell, tmp_path, axis):
    cell_root, _, _, _, _ = staged_cell
    external = _baseline(tmp_path / f"external-{axis}", attest_for=cell_root)
    path = external / BASELINE_ATTESTATION_FILE
    attestation = json.loads(path.read_text())
    current = attestation["identity"][axis]
    attestation["identity"][axis] = (
        current + 1 if isinstance(current, int) else f"foreign-{current}"
    )
    attestation.pop("attestation_sha256")
    attestation["attestation_sha256"] = content_sha256(attestation)
    path.write_text(json.dumps(attestation))

    with pytest.raises(CampaignStageError, match="not bound to this cell/protocol"):
        register_baseline(cell_root, external)
    assert not (cell_root / "baseline").exists()

    # A failed registration must not poison the cell for a corrected archive.
    external = _baseline(tmp_path / f"corrected-{axis}", attest_for=cell_root)
    state = register_baseline(cell_root, external)
    assert state["baseline"]["candidate_id"] == "baseline"


def test_baseline_rejects_an_unexpected_sixth_sealed_fold(staged_cell):
    cell_root, _, _, _, _ = staged_cell
    archive = _baseline(cell_root)
    extra = archive / "certify/fold_5_result.json"
    extra.write_text(json.dumps({"fold_index": 5, "held_out": {}}))
    with pytest.raises(CampaignStageError, match="sealed folds must be exactly"):
        register_baseline(cell_root, archive)


def test_baseline_registration_rejects_mutated_campaign_cell(staged_cell):
    cell_root, adir, _, _, _ = staged_cell
    archive = _baseline(cell_root)
    cell_path = adir / "campaign_cell.json"
    cell = json.loads(cell_path.read_text())
    cell["identity"]["encoder"] = "mutated-encoder"
    cell["encoder"] = "mutated-encoder"
    cell.pop("cell_sha256")
    cell["cell_sha256"] = content_sha256(cell)
    cell_path.write_text(json.dumps(cell))

    with pytest.raises(CampaignStageError, match="differs from stage state"):
        register_baseline(cell_root, archive)


def test_freeze_requires_baseline_and_exact_attempt_budget(staged_cell):
    cell_root, adir, cell, _, _ = staged_cell
    # B7: the session itself now refuses to open without a registered baseline
    # (the freeze's own baseline check stays as defense-in-depth behind it).
    with pytest.raises(CampaignStageError, match="baseline"):
        open_agent_session(cell_root, {
            "session_id": "fixture-session",
            "started_at": (
                datetime.now(timezone.utc) - timedelta(minutes=1)
            ).isoformat(),
        })
    register_baseline(cell_root, _baseline(cell_root))
    _attempts(adir, cell["cell_id"])
    _open_budget_cell(
        adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS - 1,
    )
    with pytest.raises(
        CampaignStageError, match=f"exactly {DISCOVERY_ATTEMPTS}",
    ):
        freeze_discovery(cell_root)


def test_freeze_charges_failures_and_promotes_top_ten_complete(staged_cell):
    cell_root, adir, cell, _, _ = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _attempts(adir, cell["cell_id"], completed=12)
    _open_budget_cell(
        adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS,
    )

    state = freeze_discovery(cell_root)
    promoted = state["discovery"]["promoted_candidates"]
    assert state["phase"] == "promotion-ready"
    assert state["discovery"]["attempts_charged"] == DISCOVERY_ATTEMPTS
    assert state["discovery"]["complete_candidates"] == 12
    assert state["discovery"]["unique_complete_candidates"] == 12
    assert len(state["discovery"]["attempt_audit"]) == DISCOVERY_ATTEMPTS
    assert len(promoted) == PROTOCOL["promotion_candidates"] == 10
    assert [candidate["candidate_id"] for candidate in promoted] == [
        f"node_{index:04d}" for index in range(12, 2, -1)
    ]
    assert all(set(candidate["validation_folds"][0]) == {
        "fold_index", "metrics", "composite",
    } for candidate in promoted)
    assert all(len(candidate["sealed_fold_sha256"]) == 3 for candidate in promoted)

    # Restarting cannot spend, reorder, or refreeze anything.
    assert freeze_discovery(cell_root) == state


def test_discovery_audit_joins_failure_completion_without_reclassifying_spec(
    staged_cell,
):
    cell_root, adir, cell, _, _ = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _attempts(adir, cell["cell_id"], completed=0)
    archive = adir / "orchestrator/archive/node_0001"
    (archive / "result.json").unlink()
    completed = adir / "orchestrator/completed"
    completed.mkdir(parents=True)
    (completed / "node_0001.json").write_text(json.dumps({
        "id": "node_0001",
        "status": "partial",
        "termination_reason": "sigterm",
        "budget_killed": True,
        "elapsed_seconds": 12.5,
        "peak_vram_mb": 0,
    }))
    _open_budget_cell(adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS)

    state = freeze_discovery(cell_root)
    audit = state["discovery"]["attempt_audit"][0]

    assert audit["candidate_class"] == "config-only"
    assert audit["result_status"] == "partial"
    assert audit["budget_killed"] is True
    assert audit["outcome_class"] == "budget-killed"
    assert audit["termination_reason"] == "sigterm"
    assert audit["elapsed_seconds"] == pytest.approx(12.5)
    assert audit["peak_vram_mb"] is None
    assert audit["eligible"] is False


def test_discovery_excludes_out_of_range_validation_metrics(staged_cell):
    cell_root, adir, cell, _, _ = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _attempts(adir, cell["cell_id"], completed=1)
    result_path = adir / "orchestrator/archive/node_0001/result.json"
    result = json.loads(result_path.read_text())
    result["validation_folds"][0]["composite"] = 1.01
    result_path.write_text(json.dumps(result))
    _open_budget_cell(adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS)

    state = freeze_discovery(cell_root)

    assert state["discovery"]["complete_candidates"] == 0
    assert "outside [0, 1]" in state["discovery"]["attempt_audit"][0]["reason"]


def test_discovery_deduplicates_semantically_identical_hparams(staged_cell):
    cell_root, adir, cell, _, _ = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _attempts(adir, cell["cell_id"], completed=12)
    policy = load_candidate_policy(adir)
    archive_root = adir / "orchestrator/archive"
    overrides = (
        '--hparams \'{"lr":0.001,"wd":0.01}\' --policy-variant cosine',
        '--policy-variant=cosine --hparams=\'{ "wd": 0.01, "lr": 0.001 }\'',
    )
    for node_id, override in zip(("node_0011", "node_0012"), overrides):
        path = archive_root / node_id / "spec.json"
        spec = json.loads(path.read_text())
        spec["run_command_override"] = override
        spec["admissibility"] = policy.classify(
            [], override=override,
        ).to_dict()
        path.write_text(json.dumps(spec))
    _open_budget_cell(
        adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS,
    )

    state = freeze_discovery(cell_root)

    assert state["discovery"]["complete_candidates"] == 12
    assert state["discovery"]["unique_complete_candidates"] == 11
    assert "node_0012" in {
        candidate["candidate_id"]
        for candidate in state["discovery"]["promoted_candidates"]
    }
    assert "node_0011" not in {
        candidate["candidate_id"]
        for candidate in state["discovery"]["promoted_candidates"]
    }


def test_zero_complete_candidates_falls_through_to_selection_ready(staged_cell):
    cell_root, adir, cell, _, _ = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _attempts(adir, cell["cell_id"], completed=0)
    _open_budget_cell(
        adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS,
    )
    state = freeze_discovery(cell_root)
    assert state["phase"] == "selection-ready"
    assert state["discovery"]["complete_candidates"] == 0
    assert state["discovery"]["promoted_candidates"] == []


def test_promotion_materializes_exact_jobs_and_an_independent_budget(staged_cell):
    cell_root, adir, cell, _, repo_root = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _attempts(adir, cell["cell_id"], completed=12, source_at=11)
    _open_budget_cell(
        adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS,
    )
    frozen = freeze_discovery(cell_root)

    state = materialize_promotion(cell_root, repo_root=repo_root)
    promotion = cell_root / "promotion" / "automil"
    jobs = state["promotion"]["jobs"]
    assert state["phase"] == "promotion"
    assert state["promotion"]["materialized"] is True
    assert len(jobs) == 10
    assert len(list((promotion / "orchestrator" / "queue").glob("*.json"))) == 10

    budget = read_cell(
        promotion / "cells" / f"{cell['budget_identity']['cell_id']}.json"
    )
    assert budget.eval_budget == 10
    assert budget.consumed_evals == 0
    config = yaml.safe_load((promotion / "config.yaml").read_text())
    assert config["run"]["command"] == cell["commands"]["promotion"]
    assert config["training"]["fold_count"] == 2
    assert config["campaign"]["stage"] == "promotion"

    first = jobs[0]
    assert first["source_node_id"] == "node_0012"
    spec = json.loads((
        promotion / "orchestrator" / "queue"
        / f"{first['promotion_node_id']}.json"
    ).read_text())
    assert spec["metadata"]["promotion"]["source_candidate_sha256"] == (
        first["source_candidate_sha256"]
    )
    assert spec["metadata"]["campaign"]["stage"] == "promotion"
    mapped = (
            promotion / "orchestrator" / "archive"
            / first["promotion_node_id"]
            / (
                "dataset__arm__task/promotion/automil/variants/"
                "_policies/candidate_11.py"
            )
        )
    assert mapped.read_text() == "# exact train-only policy candidate\n"

    # A restart returns the frozen jobs without duplicating graph/queue entries.
    assert materialize_promotion(cell_root, repo_root=repo_root) == state

    # Simulate power loss after atomic directory publication but before the
    # state transition commit. The immutable plan is adopted, not duplicated.
    (cell_root / "campaign_state.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n"
    )
    recovered = materialize_promotion(cell_root, repo_root=repo_root)
    assert recovered["phase"] == "promotion"
    assert recovered["promotion"]["jobs"] == jobs
    assert len(list((promotion / "orchestrator" / "queue").glob("*.json"))) == 10


def test_promotion_requires_closed_discovery_activity_before_mutation(staged_cell):
    cell_root, adir, cell, _, repo_root = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _attempts(adir, cell["cell_id"], completed=12)
    _open_budget_cell(
        adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS,
    )
    frozen = freeze_discovery(cell_root, end_session=False)

    with pytest.raises(
        CampaignStageError,
        match="SessionEnd and a durable final Claude active-time sample",
    ):
        materialize_promotion(cell_root, repo_root=repo_root)

    assert load_stage_state(cell_root) == frozen
    assert not (cell_root / "promotion").exists()


def _finish_promotion(
    cell_root: Path, *, completed: int, promotion_base: float = 0.7,
    promotion_bases: list[float] | None = None,
) -> None:
    adir = cell_root / "promotion" / "automil"
    state = load_stage_state(cell_root)
    for index, job in enumerate(state["promotion"]["jobs"]):
        job_base = (
            promotion_bases[index]
            if promotion_bases is not None else promotion_base + index / 100
        )
        node_id = job["promotion_node_id"]
        queue = adir / "orchestrator" / "queue" / f"{node_id}.json"
        archive = adir / "orchestrator" / "archive" / node_id
        spec = json.loads(queue.read_text())
        queue.unlink()
        (archive / "spec.json").write_text(json.dumps(spec))
        result = (
            {
                "status": "completed",
                "composite": job_base,
                "metrics": {"val_auc": 0.7, "val_bacc": 0.7},
                "validation_folds": _folds(
                    STAGE_FOLDS["promotion"], job_base,
                ),
            }
            if index < completed else
            {"status": "crash", "composite": 0.0, "metrics": {}}
        )
        (archive / "result.json").write_text(json.dumps(result))
        if index < completed:
            sealed = archive / "certify"
            sealed.mkdir(exist_ok=True)
            for fold in STAGE_FOLDS["promotion"]:
                (sealed / f"fold_{fold}_result.json").write_text(json.dumps({
                    "fold_index": fold,
                    "held_out": {
                        "test_auc": 0.70 + fold / 100,
                        "test_bacc": 0.70 + fold / 100,
                    },
                }))
    cell_path = (
        adir / "cells"
        / f"{json.loads((adir / 'campaign_cell.json').read_text())['budget_identity']['cell_id']}.json"
    )
    budget = read_cell(cell_path)
    write_cell(
        replace(
            budget,
            consumed_evals=len(state["promotion"]["jobs"]),
            completed_evals=completed,
        ),
        adir / "cells",
    )


def test_promotion_freeze_excludes_crashes_but_keeps_their_cost(staged_cell):
    cell_root, adir, cell, _, repo_root = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _attempts(adir, cell["cell_id"], completed=12)
    _open_budget_cell(
        adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS,
    )
    freeze_discovery(cell_root)
    materialize_promotion(cell_root, repo_root=repo_root)
    _finish_promotion(cell_root, completed=8)

    state = freeze_promotion(cell_root)
    assert state["phase"] == "selection-ready"
    assert state["promotion"]["attempts_charged"] == 10
    assert len(state["promotion"]["eligible_candidates"]) == 8
    assert [job["status"] for job in state["promotion"]["jobs"]] == (
        ["eligible"] * 8 + ["ineligible"] * 2
    )
    for candidate in state["promotion"]["eligible_candidates"]:
        assert [fold["fold_index"] for fold in candidate["validation_folds"]] == [
            0, 1, 2, 3, 4,
        ]
        assert len(candidate["sealed_fold_sha256"]) == 5
    assert freeze_promotion(cell_root) == state


def test_promotion_freeze_marks_missing_terminal_artifacts_ineligible(staged_cell):
    cell_root, adir, cell, _, repo_root = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _attempts(adir, cell["cell_id"], completed=12)
    _open_budget_cell(
        adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS,
    )
    freeze_discovery(cell_root)
    materialize_promotion(cell_root, repo_root=repo_root)
    _finish_promotion(cell_root, completed=10)
    state = load_stage_state(cell_root)
    archives = cell_root / "promotion/automil/orchestrator/archive"
    first = archives / state["promotion"]["jobs"][0]["promotion_node_id"]
    second = archives / state["promotion"]["jobs"][1]["promotion_node_id"]
    (first / "result.json").unlink()
    (second / "certify/fold_3_result.json").unlink()

    frozen = freeze_promotion(cell_root)

    assert frozen["phase"] == "selection-ready"
    assert len(frozen["promotion"]["eligible_candidates"]) == 8
    assert frozen["promotion"]["jobs"][0]["status"] == "ineligible"
    assert "missing result.json" in frozen["promotion"]["jobs"][0]["reason"]
    assert len(frozen["promotion"]["jobs"][0]["promotion_spec_sha256"]) == 64
    assert frozen["promotion"]["jobs"][0]["submitted_at"]
    assert frozen["promotion"]["jobs"][1]["status"] == "ineligible"
    assert "sealed folds must be exactly" in frozen["promotion"]["jobs"][1]["reason"]
    select_winner(cell_root)
    _write_global_selection_freeze(cell_root)
    artifact = json.loads((cell_root.parent / SELECTION_FREEZE_FILE).read_text())
    assert validate_selection_freeze_artifact(artifact) == artifact


def test_promotion_freeze_fails_closed_when_durable_spec_is_missing(staged_cell):
    cell_root, adir, cell, _, repo_root = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _attempts(adir, cell["cell_id"], completed=12)
    _open_budget_cell(
        adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS,
    )
    freeze_discovery(cell_root)
    materialize_promotion(cell_root, repo_root=repo_root)
    _finish_promotion(cell_root, completed=10)
    state = load_stage_state(cell_root)
    first_id = state["promotion"]["jobs"][0]["promotion_node_id"]
    (
        cell_root / "promotion/automil/orchestrator/archive"
        / first_id / "spec.json"
    ).unlink()

    with pytest.raises(CampaignStageError, match="lost its durable spec"):
        freeze_promotion(cell_root)
    assert not (cell_root.parent / SELECTION_FREEZE_FILE).exists()


def test_promotion_freeze_rejects_cross_stage_identity_drift(staged_cell):
    cell_root, adir, cell, _, repo_root = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _attempts(adir, cell["cell_id"], completed=12)
    _open_budget_cell(
        adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS,
    )
    freeze_discovery(cell_root)
    materialize_promotion(cell_root, repo_root=repo_root)
    _finish_promotion(cell_root, completed=10)
    state = load_stage_state(cell_root)
    node_id = state["promotion"]["jobs"][0]["promotion_node_id"]
    spec_path = cell_root / "promotion/automil/orchestrator/archive" / node_id / "spec.json"
    spec = json.loads(spec_path.read_text())
    spec["metadata"]["promotion"]["source_candidate_sha256"] = "0" * 64
    spec_path.write_text(json.dumps(spec))

    with pytest.raises(CampaignStageError, match="source link drifted"):
        freeze_promotion(cell_root)


def test_zero_complete_discovery_freezes_baseline_with_zero_lift(staged_cell):
    cell_root, adir, cell, _, _ = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _attempts(adir, cell["cell_id"], completed=0)
    _open_budget_cell(
        adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS,
    )
    freeze_discovery(cell_root)

    state = select_winner(cell_root)
    assert state["phase"] == "winner-frozen"
    assert state["winner"]["kind"] == "baseline"
    assert state["winner"]["candidate_id"] == "baseline"
    assert state["winner"]["lift_over_baseline"] == pytest.approx(0.0)
    assert select_winner(cell_root) == state


def test_zero_candidate_selection_requires_closed_discovery_activity(staged_cell):
    cell_root, adir, cell, _, _ = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _attempts(adir, cell["cell_id"], completed=0)
    _open_budget_cell(
        adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS,
    )
    frozen = freeze_discovery(cell_root, end_session=False)

    with pytest.raises(
        CampaignStageError,
        match="SessionEnd and a durable final Claude active-time sample",
    ):
        select_winner(cell_root)

    assert load_stage_state(cell_root) == frozen
    assert frozen["winner"] is None


def test_selection_rechecks_exclusive_discovery_activity(staged_cell):
    cell_root, adir, cell, _, _ = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _attempts(adir, cell["cell_id"], completed=0)
    _open_budget_cell(
        adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS,
    )
    freeze_discovery(cell_root)
    record_hook_event(
        adir,
        cell["budget_identity"]["cell_id"],
        {
            "hook_event_name": "SessionStart",
            "session_id": "unexpected-session",
            "source": "startup",
        },
    )
    final_observed_at = datetime.now(timezone.utc).timestamp()
    ingest_prometheus_metrics(
        adir,
        "claude_code_active_time_total"
        '{session_id="unexpected-session",type="cli"} 1.0\n',
        observed_at=final_observed_at,
    )
    record_hook_event(
        adir,
        cell["budget_identity"]["cell_id"],
        {"hook_event_name": "SessionEnd", "session_id": "unexpected-session"},
        observed_at=final_observed_at,
        final_sample_observed_at=final_observed_at,
    )

    with pytest.raises(CampaignStageError, match="not exclusive"):
        select_winner(cell_root)


def test_selection_requires_the_durable_final_activity_sample(staged_cell):
    cell_root, adir, cell, _, _ = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _attempts(adir, cell["cell_id"], completed=0)
    _open_budget_cell(
        adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS,
    )
    freeze_discovery(cell_root)
    (adir / ACTIVITY_SAMPLES_FILENAME).unlink()

    with pytest.raises(
        CampaignStageError,
        match="SessionEnd and a durable final Claude active-time sample",
    ):
        select_winner(cell_root)


def test_fivefold_validation_mean_can_select_searched_candidate(staged_cell):
    cell_root, adir, cell, _, repo_root = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _attempts(adir, cell["cell_id"], completed=12)
    _open_budget_cell(
        adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS,
    )
    freeze_discovery(cell_root)
    materialize_promotion(cell_root, repo_root=repo_root)
    _finish_promotion(cell_root, completed=10, promotion_base=0.75)
    freeze_promotion(cell_root)

    state = select_winner(cell_root)
    winner = state["winner"]
    assert winner["kind"] == "searched"
    assert winner["candidate_id"] == "node_0012"
    assert [fold["fold_index"] for fold in winner["validation_folds"]] == [
        0, 1, 2, 3, 4,
    ]
    assert winner["validation_mean"] == pytest.approx(
        sum(fold["composite"] for fold in winner["validation_folds"]) / 5
    )
    assert winner["lift_over_baseline"] > 0


def test_stage_process_evidence_rejects_inconsistent_eligible_promotion(
    staged_cell,
):
    cell_root, adir, cell, _, repo_root = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _attempts(adir, cell["cell_id"], completed=12)
    _open_budget_cell(
        adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS,
    )
    freeze_discovery(cell_root)
    materialize_promotion(cell_root, repo_root=repo_root)
    _finish_promotion(cell_root, completed=10, promotion_base=0.75)
    state = freeze_promotion(cell_root)
    assert _process_evidence(state)["promotion"]["status_counts"]["eligible"] == 10

    drifted = json.loads(json.dumps(state))
    drifted["promotion"]["jobs"][0].update({
        "result_status": "crash",
        "outcome_class": "crash",
        "validation_mean": None,
    })
    with pytest.raises(CampaignStageError, match="eligible promotion"):
        _process_evidence(drifted)


def test_exact_validation_tie_prefers_native_baseline(staged_cell):
    cell_root, adir, cell, _, repo_root = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _attempts(adir, cell["cell_id"], completed=12)
    _open_budget_cell(
        adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS,
    )
    freeze_discovery(cell_root)
    materialize_promotion(cell_root, repo_root=repo_root)
    # Rank-1 source node_0012 has discovery composites .610/.615/.620.
    # Promotion .600/.605 makes the exact five-fold mean .610, baseline's mean.
    _finish_promotion(cell_root, completed=1, promotion_base=0.585)
    freeze_promotion(cell_root)

    state = select_winner(cell_root)
    assert state["winner"]["validation_mean"] == pytest.approx(0.61)
    assert state["winner"]["kind"] == "baseline"


def test_searched_tie_uses_stable_discovery_node_id(staged_cell):
    cell_root, adir, cell, _, repo_root = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _attempts(adir, cell["cell_id"], completed=12)
    _open_budget_cell(
        adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS,
    )
    freeze_discovery(cell_root)
    materialize_promotion(cell_root, repo_root=repo_root)
    # node_0012 has .03 more discovery-composite mass than node_0011;
    # adding .015 to each of node_0011's two promotion folds makes them tie.
    bases = [0.75, 0.765] + [0.5] * 8
    _finish_promotion(
        cell_root, completed=2, promotion_bases=bases,
    )
    freeze_promotion(cell_root)

    state = select_winner(cell_root)
    audit = state["winner"]["selection_audit"]
    searched = [row for row in audit if row["kind"] == "searched"]
    assert searched[0]["validation_mean"] == pytest.approx(
        searched[1]["validation_mean"]
    )
    assert state["winner"]["candidate_id"] == "node_0011"


def test_winner_selection_detects_baseline_artifact_drift(staged_cell):
    cell_root, adir, cell, _, _ = staged_cell
    baseline = _baseline(cell_root)
    register_baseline(cell_root, baseline)
    _attempts(adir, cell["cell_id"], completed=0)
    _open_budget_cell(
        adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS,
    )
    freeze_discovery(cell_root)
    (baseline / "certify" / "fold_4_result.json").write_text("tampered")

    with pytest.raises(CampaignStageError, match="sealed artifact changed"):
        select_winner(cell_root)


def _write_searched_sealed_folds(
    cell_root: Path, state: dict, *, selected_valid: bool = True,
) -> None:
    winner = state["winner"]
    discovery = cell_root / "automil/orchestrator/archive"
    promotion = cell_root / "promotion/automil/orchestrator/archive"
    # Every loser is deliberately malformed after selection. Certification
    # must not open or verify an unselected candidate.
    for candidate in state["discovery"]["promoted_candidates"]:
        sealed = discovery / candidate["candidate_id"] / "certify"
        sealed.mkdir(parents=True, exist_ok=True)
        for fold in STAGE_FOLDS["discovery"]:
            (sealed / f"fold_{fold}_result.json").write_text("loser-not-json")
    for job in state["promotion"]["jobs"]:
        sealed = promotion / job["promotion_node_id"] / "certify"
        sealed.mkdir(parents=True, exist_ok=True)
        for fold in STAGE_FOLDS["promotion"]:
            (sealed / f"fold_{fold}_result.json").write_text("loser-not-json")

    selected_discovery = discovery / winner["candidate_id"] / "certify"
    selected_promotion = promotion / winner["promotion_node_id"] / "certify"
    if selected_valid:
        for fold in CERTIFICATION_FOLDS:
            path = (
                selected_discovery if fold in STAGE_FOLDS["discovery"]
                else selected_promotion
            ) / f"fold_{fold}_result.json"
            path.write_text(json.dumps({
                "fold_index": fold,
                "held_out": {
                    "test_auc": 0.70 + fold / 100,
                    "test_bacc": 0.70 + fold / 100,
                },
            }))
    else:
        (selected_discovery / "fold_0_result.json").write_text("selected-not-json")


def _write_global_selection_freeze(cell_root: Path) -> None:
    """Build a full-roster freeze fixture around this isolated unit-test cell."""
    state = load_stage_state(cell_root)
    winner = state["winner"]
    usage = {
        "status": "exact",
        "input_tokens": 100,
        "output_tokens": 50,
        "cached_input_tokens": 0,
        "cost_usd": 0.1,
        "basis": "test fixture",
    }
    session = finalize_agent_session(cell_root, _agent_session_end(cell_root))
    process = _process_evidence(state)
    winner_source_folds = _source_fold_anchors(
        cell_root.parent,
        _winner_sealed_sources(cell_root, state, winner),
    )
    baseline_source_folds = _source_fold_anchors(
        cell_root.parent,
        _baseline_sealed_sources(cell_root, state),
    )
    entry = {
        "cell_id": state["cell_id"],
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
        "winner_source_folds": winner_source_folds,
        "baseline_source_folds": baseline_source_folds,
        "agent_session_sha256": session["attestation_sha256"],
        "agent_session_id": session["session"]["session_id"],
        "agent_session_binding_sha256": session["binding_sha256"],
        "agent_usage": usage,
        "process_sha256": content_sha256(process),
        "process_evidence": process,
    }
    def sibling_entry(index: int) -> dict:
        cell_id = f"fixture-cell-{index:03d}"
        session_id = f"fixture-session-{index:03d}"
        session_binding = f"{index + 1:064x}"
        sibling_process = json.loads(json.dumps(process))
        for attempt in sibling_process["discovery"]["attempts"]:
            attempt["agent_session_id"] = session_id
            attempt["agent_session_binding_sha256"] = session_binding
        sibling_sources = {
            filename: {
                **record,
                "path": PurePosixPath(
                    cell_id, "baseline", "archive", "certify", filename,
                ).as_posix(),
            }
            for filename, record in baseline_source_folds.items()
        }
        return {
            "cell_id": cell_id,
            "cell_sha256": "a" * 64,
            "state_sha256": "b" * 64,
            "selection_sha256": "c" * 64,
            "winner_kind": "baseline",
            "winner_candidate_id": "baseline",
            "winner_candidate_sha256": "d" * 64,
            "winner_promotion_node_id": None,
            "winner_validation_mean": 0.5,
            "baseline_validation_mean": 0.5,
            "baseline_candidate_sha256": "d" * 64,
            "winner_source_folds": sibling_sources,
            "baseline_source_folds": sibling_sources,
            "agent_session_sha256": "e" * 64,
            "agent_session_id": session_id,
            "agent_session_binding_sha256": session_binding,
            "agent_usage": usage,
            "process_sha256": content_sha256(sibling_process),
            "process_evidence": sibling_process,
        }
    cells = [entry] + [
        sibling_entry(index) for index in range(CAMPAIGN_CELL_COUNT - 1)
    ]
    roster = dict(sorted(
        (row["cell_id"], row["cell_sha256"]) for row in cells
    ))
    artifact = {
        "schema_version": SELECTION_FREEZE_SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "manifest_sha256": state["manifest_sha256"],
        "protocol_version": PROTOCOL_VERSION,
        "agent_protocol_sha256": content_sha256(AGENT_PROTOCOL),
        "roster_sha256": content_sha256(roster),
        "cell_count": CAMPAIGN_CELL_COUNT,
        "cells": cells,
        "frozen_at": "2026-08-04T00:00:00+00:00",
    }
    artifact["freeze_sha256"] = content_sha256(artifact)
    (cell_root.parent / SELECTION_FREEZE_FILE).write_text(json.dumps(artifact))


def _agent_session_end(cell_root: Path) -> dict:
    session = json.loads((cell_root / "agent_session.json").read_text())
    _record_session_end(cell_root / "automil")
    ended_at = (
        datetime.fromisoformat(session["session"]["bound_at"])
        + timedelta(hours=1)
    ).isoformat()
    return {
        "session_id": "fixture-session",
        "ended_at": ended_at,
        "termination_reason": "budget-complete",
        "usage": {
            "status": "exact",
            "input_tokens": 100,
            "output_tokens": 50,
            "cached_input_tokens": 0,
            "cost_usd": 0.1,
            "basis": "test fixture",
        },
    }


def _record_session_start(
    adir: Path, session_id: str = "fixture-session",
) -> None:
    record_hook_event(
        adir,
        None,
        {
            "hook_event_name": "SessionStart",
            "session_id": session_id,
            "source": "startup",
        },
    )
    ingest_prometheus_metrics(
        adir,
        "claude_code_active_time_total"
        f'{{session_id="{session_id}",type="cli"}} 1.0\n',
    )


def _record_session_end(
    adir: Path, session_id: str = "fixture-session",
) -> None:
    config = yaml.safe_load((adir / "config.yaml").read_text())
    cell_id = config["campaign"]["budget_cell_id"]
    report = read_activity_report(adir, cell_id)
    if report.complete:
        return
    observed_at = datetime.now(timezone.utc).timestamp()
    ingest_prometheus_metrics(
        adir,
        "claude_code_active_time_total"
        f'{{session_id="{session_id}",type="cli"}} 1.0\n',
        observed_at=observed_at,
    )
    record_hook_event(
        adir,
        cell_id,
        {"hook_event_name": "SessionEnd", "session_id": session_id},
        observed_at=observed_at,
        final_sample_observed_at=observed_at,
    )


def freeze_discovery(
    cell_root: Path, *, end_session: bool = True,
) -> dict:
    """Freeze discovery, then model the real SessionEnd hook for success paths."""
    state = _freeze_discovery(cell_root)
    if end_session:
        _record_session_end(cell_root / "automil")
    return state


def test_agent_session_is_prebound_to_every_proposal_then_finalized(
    staged_cell,
):
    cell_root, adir, cell, _, _ = staged_cell
    register_baseline(cell_root, _baseline(cell_root))  # B7: baseline precedes the session
    _record_session_start(adir)
    opened = open_agent_session(cell_root, {
        "session_id": "fixture-session",
        "started_at": (
            datetime.now(timezone.utc) - timedelta(minutes=1)
        ).isoformat(),
    })
    assert opened["status"] == "open"
    with pytest.raises(CampaignStageError, match="immutable"):
        open_agent_session(cell_root, {
            "session_id": "second-session",
            "started_at": (
                datetime.now(timezone.utc) - timedelta(minutes=1)
            ).isoformat(),
        })
    _attempts(adir, cell["cell_id"], completed=0)
    _open_budget_cell(
        adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS,
    )
    freeze_discovery(cell_root)
    activity = read_activity_report(adir, cell["budget_identity"]["cell_id"])
    assert activity.complete is True
    session_path = cell_root / "agent_session.json"
    assert json.loads(session_path.read_text())["status"] == "open"
    select_winner(cell_root)
    assert json.loads(session_path.read_text())["status"] == "open"

    registered = finalize_agent_session(cell_root, _agent_session_end(cell_root))
    assert registered["status"] == "finalized"
    assert registered["attestation_sha256"] == content_sha256({
        key: value for key, value in registered.items()
        if key != "attestation_sha256"
    })
    assert finalize_agent_session(cell_root, _agent_session_end(cell_root)) == registered


def test_checked_in_session_templates_execute_the_controller_contract(staged_cell):
    cell_root, adir, cell, _, _ = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    start = json.loads((CAMPAIGN_DIR / "agent_session.template.json").read_text())
    start.update({
        "session_id": "fixture-session",
        "started_at": (
            datetime.now(timezone.utc) - timedelta(minutes=1)
        ).isoformat(),
    })
    _record_session_start(adir)
    opened = open_agent_session(cell_root, start)
    _attempts(adir, cell["cell_id"], completed=0)
    _open_budget_cell(
        adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS,
    )
    freeze_discovery(cell_root)
    select_winner(cell_root)

    end = json.loads(
        (CAMPAIGN_DIR / "agent_session_end.template.json").read_text()
    )
    end.update({
        "session_id": "fixture-session",
        "ended_at": (
            datetime.fromisoformat(opened["session"]["bound_at"])
            + timedelta(hours=1)
        ).isoformat(),
    })
    end["usage"]["basis"] = "fixture runtime does not expose usage"
    _record_session_end(adir)
    finalized = finalize_agent_session(cell_root, end)

    assert finalized["status"] == "finalized"
    readme = (CAMPAIGN_DIR / "README.md").read_text()
    assert "open-agent-session" in readme
    assert "finalize-agent-session" in readme
    assert "register-agent-session" not in readme


def test_agent_session_open_rejects_preexisting_graph_proposal(staged_cell):
    cell_root, adir, _, _, _ = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    graph_path = adir / "graph.json"
    graph = json.loads(graph_path.read_text())
    graph["nodes"]["node_9999"] = {"type": "proposed", "status": "pending"}
    graph_path.write_text(json.dumps(graph))

    with pytest.raises(CampaignStageError, match="non-baseline proposal"):
        open_agent_session(cell_root, {
            "session_id": "fixture-session",
            "started_at": (
                datetime.now(timezone.utc) - timedelta(minutes=1)
            ).isoformat(),
        })


def test_agent_session_open_rejects_preexisting_candidate_file(staged_cell):
    cell_root, adir, _, _, _ = staged_cell
    register_baseline(cell_root, _baseline(cell_root))  # B7
    policy = adir / "variants/_policies/prebuilt.py"
    policy.parent.mkdir(parents=True)
    policy.write_text("# created before the bound session\n")

    with pytest.raises(CampaignStageError, match="candidate policy file"):
        open_agent_session(cell_root, {
            "session_id": "fixture-session",
            "started_at": (
                datetime.now(timezone.utc) - timedelta(minutes=1)
            ).isoformat(),
        })


def test_agent_session_id_is_reserved_across_cells_at_open(staged_cell):
    cell_root, _, _, _, _ = staged_cell
    register_baseline(cell_root, _baseline(cell_root))  # B7
    sibling = cell_root.parent / "other-cell"
    sibling.mkdir()
    (sibling / "agent_session.json").write_text(json.dumps({
        "session": {"session_id": "fixture-session"},
    }))

    with pytest.raises(CampaignStageError, match="already reserved"):
        open_agent_session(cell_root, {
            "session_id": "fixture-session",
            "started_at": (
                datetime.now(timezone.utc) - timedelta(minutes=1)
            ).isoformat(),
        })


def test_agent_session_open_rejects_nonobject_sibling_reservation(staged_cell):
    cell_root, _, _, _, _ = staged_cell
    register_baseline(cell_root, _baseline(cell_root))  # B7
    sibling = cell_root.parent / "other-cell"
    sibling.mkdir()
    (sibling / "agent_session.json").write_text("[]")

    with pytest.raises(CampaignStageError, match="cannot verify session reservation"):
        open_agent_session(cell_root, {
            "session_id": "fixture-session",
            "started_at": (
                datetime.now(timezone.utc) - timedelta(minutes=1)
            ).isoformat(),
        })


def test_agent_session_open_rejects_nonobject_nested_sibling_record(staged_cell):
    cell_root, _, _, _, _ = staged_cell
    register_baseline(cell_root, _baseline(cell_root))  # B7
    sibling = cell_root.parent / "other-cell"
    sibling.mkdir()
    (sibling / "agent_session.json").write_text(json.dumps({"session": [1]}))

    with pytest.raises(CampaignStageError, match="cannot verify session reservation"):
        open_agent_session(cell_root, {
            "session_id": "fixture-session",
            "started_at": (
                datetime.now(timezone.utc) - timedelta(minutes=1)
            ).isoformat(),
        })


def test_discovery_rejects_nonobject_agent_session_json(staged_cell):
    cell_root, adir, cell, _, _ = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _attempts(adir, cell["cell_id"], completed=0)
    _open_budget_cell(
        adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS,
    )
    (cell_root / "agent_session.json").write_text("[]")

    with pytest.raises(CampaignStageError, match="agent session field set"):
        freeze_discovery(cell_root)


def test_discovery_rejects_a_proposal_before_the_bound_session(staged_cell):
    cell_root, adir, cell, _, _ = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _attempts(adir, cell["cell_id"], completed=0)
    spec_path = adir / "orchestrator/archive/node_0001/spec.json"
    spec = json.loads(spec_path.read_text())
    session = json.loads((cell_root / "agent_session.json").read_text())
    spec["submitted_at"] = (
        datetime.fromisoformat(session["session"]["bound_at"])
        - timedelta(seconds=SUBMIT_CLOCK_SKEW_TOLERANCE_SECONDS + 1)
    ).isoformat()
    spec_path.write_text(json.dumps(spec))
    _open_budget_cell(adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS)

    # C-j: beyond the declared tolerance -> still fail closed.
    with pytest.raises(CampaignStageError, match="clock-skew tolerance"):
        freeze_discovery(cell_root)


def test_discovery_tolerates_ntp_level_skew_on_the_first_proposal(staged_cell):
    """C-j (claims-alignment): a submitted_at seconds before bound_at (cross-host
    NTP skew) used to brick the cell permanently — the timestamps live in hashed
    archived specs and cannot be legitimately corrected."""
    cell_root, adir, cell, _, _ = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _attempts(adir, cell["cell_id"], completed=0)
    spec_path = adir / "orchestrator/archive/node_0001/spec.json"
    spec = json.loads(spec_path.read_text())
    session = json.loads((cell_root / "agent_session.json").read_text())
    spec["submitted_at"] = (
        datetime.fromisoformat(session["session"]["bound_at"])
        - timedelta(seconds=SUBMIT_CLOCK_SKEW_TOLERANCE_SECONDS - 30)
    ).isoformat()
    spec_path.write_text(json.dumps(spec))
    _open_budget_cell(adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS)

    state = freeze_discovery(cell_root)
    # completed=0 -> no promotable candidates -> straight to selection-ready;
    # the point is that freeze SUCCEEDED despite the within-tolerance skew.
    assert state["phase"] == "selection-ready"


def test_finalization_rejects_a_proposal_after_the_session_end(staged_cell):
    cell_root, adir, cell, _, _ = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _attempts(adir, cell["cell_id"], completed=0)
    _open_budget_cell(adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS)
    freeze_discovery(cell_root)
    select_winner(cell_root)
    session_end = _agent_session_end(cell_root)
    bound_at = datetime.fromisoformat(
        json.loads((cell_root / "agent_session.json").read_text())["session"]["bound_at"]
    )
    session_end["ended_at"] = (
        bound_at + timedelta(seconds=DISCOVERY_ATTEMPTS // 2)
    ).isoformat()

    with pytest.raises(CampaignStageError, match="outside the agent session interval"):
        finalize_agent_session(cell_root, session_end)


def test_cell_certification_requires_global_campaign_freeze(staged_cell):
    cell_root, adir, cell, _, _ = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _attempts(adir, cell["cell_id"], completed=0)
    _open_budget_cell(
        adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS,
    )
    freeze_discovery(cell_root)
    select_winner(cell_root)

    with pytest.raises(CampaignStageError, match="global 130-cell selection freeze"):
        certify_winner(cell_root)


def test_baseline_winner_certification_unseals_existing_folds_once(staged_cell):
    cell_root, adir, cell, _, _ = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _attempts(adir, cell["cell_id"], completed=0)
    _open_budget_cell(
        adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS,
    )
    freeze_discovery(cell_root)
    winner_frozen = select_winner(cell_root)
    _write_global_selection_freeze(cell_root)

    state = certify_winner(cell_root)
    bundle_path = cell_root / state["certification"]["bundle"]
    bundle = json.loads(bundle_path.read_text())
    assert state["phase"] == "certified"
    assert state["certification"]["selection_state_sha256"] == (
        winner_frozen["state_sha256"]
    )
    assert bundle["winner"]["kind"] == "baseline"
    assert bundle["held_out"]["test_auc"] == pytest.approx(0.52)
    assert bundle["baseline_held_out"]["test_auc"] == pytest.approx(0.52)
    assert bundle["held_out_lift"]["test_auc"] == pytest.approx(0.0)
    assert all(
        row["held_out_delta"]["test_auc"] == pytest.approx(0.0)
        for row in bundle["paired_fold_deltas"]
    )
    assert bundle["retrained"] is False
    assert certify_winner(cell_root) == state

    # Recover the narrow crash window after atomic bundle publication but
    # before the stage-state commit, without reading/revealing another winner.
    (cell_root / "campaign_state.json").write_text(
        json.dumps(winner_frozen, indent=2, sort_keys=True) + "\n"
    )
    recovered = certify_winner(cell_root)
    assert recovered["phase"] == "certified"
    assert recovered["certification"]["bundle_sha256"] == bundle["bundle_sha256"]


def test_cell_certification_rejects_nonwinner_state_drift_after_global_freeze(
    staged_cell,
):
    cell_root, adir, cell, _, _ = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _attempts(adir, cell["cell_id"], completed=0)
    _open_budget_cell(
        adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS,
    )
    freeze_discovery(cell_root)
    select_winner(cell_root)
    _write_global_selection_freeze(cell_root)

    state_path = cell_root / "campaign_state.json"
    state = json.loads(state_path.read_text())
    state["baseline"]["validation_mean"] += 0.01
    state["state_sha256"] = content_sha256({
        key: value for key, value in state.items() if key != "state_sha256"
    })
    state_path.write_text(json.dumps(state))

    with pytest.raises(CampaignStageError, match="global selection freeze"):
        certify_winner(cell_root)


def test_selection_freeze_rejects_old_extra_and_incomplete_schemas(staged_cell):
    cell_root, adir, cell, _, _ = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _attempts(adir, cell["cell_id"], completed=0)
    _open_budget_cell(
        adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS,
    )
    freeze_discovery(cell_root)
    select_winner(cell_root)
    _write_global_selection_freeze(cell_root)
    valid = json.loads(
        (cell_root.parent / SELECTION_FREEZE_FILE).read_text()
    )

    variants = []
    old = json.loads(json.dumps(valid))
    old["schema_version"] = SELECTION_FREEZE_SCHEMA_VERSION - 1
    variants.append(old)
    extra = json.loads(json.dumps(valid))
    extra["unexpected"] = True
    variants.append(extra)
    missing = json.loads(json.dumps(valid))
    missing["cells"][0].pop("process_evidence")
    variants.append(missing)
    for artifact in variants:
        artifact["freeze_sha256"] = content_sha256({
            key: value for key, value in artifact.items()
            if key != "freeze_sha256"
        })
        with pytest.raises(CampaignStageError, match="integrity mismatch"):
            validate_selection_freeze_artifact(artifact)


def test_selection_freeze_rejects_rehashed_nonstring_cell_id(staged_cell):
    cell_root, adir, cell, _, _ = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _attempts(adir, cell["cell_id"], completed=0)
    _open_budget_cell(
        adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS,
    )
    freeze_discovery(cell_root)
    select_winner(cell_root)
    _write_global_selection_freeze(cell_root)
    artifact = json.loads((cell_root.parent / SELECTION_FREEZE_FILE).read_text())
    artifact["cells"][0]["cell_id"] = 7
    artifact["roster_sha256"] = content_sha256({})
    artifact["freeze_sha256"] = content_sha256({
        key: value for key, value in artifact.items()
        if key != "freeze_sha256"
    })

    with pytest.raises(CampaignStageError, match="roster row is invalid"):
        validate_selection_freeze_artifact(artifact)


def test_selection_freeze_rejects_rehashed_duplicate_session_id(staged_cell):
    cell_root, adir, cell, _, _ = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _attempts(adir, cell["cell_id"], completed=0)
    _open_budget_cell(
        adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS,
    )
    freeze_discovery(cell_root)
    select_winner(cell_root)
    _write_global_selection_freeze(cell_root)
    artifact = json.loads((cell_root.parent / SELECTION_FREEZE_FILE).read_text())
    first_session = artifact["cells"][0]["agent_session_id"]
    sibling = artifact["cells"][1]
    sibling["agent_session_id"] = first_session
    for attempt in sibling["process_evidence"]["discovery"]["attempts"]:
        attempt["agent_session_id"] = first_session
    sibling["process_sha256"] = content_sha256(sibling["process_evidence"])
    artifact["freeze_sha256"] = content_sha256({
        key: value for key, value in artifact.items()
        if key != "freeze_sha256"
    })

    with pytest.raises(CampaignStageError, match="integrity mismatch"):
        validate_selection_freeze_artifact(artifact)


@pytest.mark.parametrize("field", ["node_id", "source_spec_sha256"])
def test_selection_freeze_rejects_rehashed_duplicate_attempt_identity(
    staged_cell, field,
):
    cell_root, adir, cell, _, _ = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _attempts(adir, cell["cell_id"], completed=0)
    _open_budget_cell(
        adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS,
    )
    freeze_discovery(cell_root)
    select_winner(cell_root)
    _write_global_selection_freeze(cell_root)
    artifact = json.loads((cell_root.parent / SELECTION_FREEZE_FILE).read_text())
    entry = artifact["cells"][0]
    attempts = entry["process_evidence"]["discovery"]["attempts"]
    attempts[1][field] = attempts[0][field]
    if field == "node_id":
        entry["process_evidence"]["discovery"]["validation_anytime"][1][
            "node_id"
        ] = attempts[0]["node_id"]
    entry["process_sha256"] = content_sha256(entry["process_evidence"])
    artifact["freeze_sha256"] = content_sha256({
        key: value for key, value in artifact.items()
        if key != "freeze_sha256"
    })

    with pytest.raises(CampaignStageError, match="identities are not unique"):
        validate_selection_freeze_artifact(artifact)


def test_certification_rejects_rehashed_incomplete_sibling_process(staged_cell):
    cell_root, adir, cell, _, _ = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _attempts(adir, cell["cell_id"], completed=0)
    _open_budget_cell(
        adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS,
    )
    freeze_discovery(cell_root)
    select_winner(cell_root)
    _write_global_selection_freeze(cell_root)
    path = cell_root.parent / SELECTION_FREEZE_FILE
    artifact = json.loads(path.read_text())
    sibling = artifact["cells"][1]
    sibling["process_evidence"]["discovery"]["attempts"].pop()
    sibling["process_sha256"] = content_sha256(sibling["process_evidence"])
    artifact["freeze_sha256"] = content_sha256({
        key: value for key, value in artifact.items()
        if key != "freeze_sha256"
    })
    path.write_text(json.dumps(artifact))

    with pytest.raises(CampaignStageError, match="discovery census"):
        certify_winner(cell_root)
    assert not (cell_root / "certification/certify.json").exists()


def test_cell_certification_rejects_a_rehashed_nonmanifest_freeze_roster(
    staged_cell,
):
    cell_root, adir, cell, _, _ = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _attempts(adir, cell["cell_id"], completed=0)
    _open_budget_cell(
        adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS,
    )
    freeze_discovery(cell_root)
    select_winner(cell_root)
    _write_global_selection_freeze(cell_root)

    path = cell_root.parent / SELECTION_FREEZE_FILE
    artifact = json.loads(path.read_text())
    forged = artifact["cells"][1]
    original_cell_id = forged["cell_id"]
    forged["cell_id"] = "forged-cell"
    for field in ("winner_source_folds", "baseline_source_folds"):
        for record in forged[field].values():
            record["path"] = record["path"].replace(
                original_cell_id, "forged-cell", 1,
            )
    artifact["roster_sha256"] = content_sha256(dict(sorted(
        (row["cell_id"], row["cell_sha256"]) for row in artifact["cells"]
    )))
    artifact["freeze_sha256"] = content_sha256({
        key: value for key, value in artifact.items()
        if key != "freeze_sha256"
    })
    path.write_text(json.dumps(artifact))

    with pytest.raises(CampaignStageError, match="locked manifest"):
        certify_winner(cell_root)


def test_searched_certification_reads_winner_and_never_losers(staged_cell):
    cell_root, adir, cell, _, repo_root = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _attempts(adir, cell["cell_id"], completed=12)
    _open_budget_cell(
        adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS,
    )
    freeze_discovery(cell_root)
    materialize_promotion(cell_root, repo_root=repo_root)
    _finish_promotion(cell_root, completed=10, promotion_base=0.75)
    freeze_promotion(cell_root)
    selected = select_winner(cell_root)
    _write_searched_sealed_folds(cell_root, selected)
    _write_global_selection_freeze(cell_root)
    freeze_artifact = json.loads(
        (cell_root.parent / SELECTION_FREEZE_FILE).read_text()
    )
    process = freeze_artifact["cells"][0]["process_evidence"]["discovery"]
    assert process["baseline_validation_mean"] == pytest.approx(0.605)
    assert process["validation_anytime"][-1][
        "running_best_validation_mean"
    ] > process["baseline_validation_mean"]

    state = certify_winner(cell_root)
    bundle = json.loads((cell_root / "certification/certify.json").read_text())
    assert bundle["winner"]["candidate_id"] == selected["winner"]["candidate_id"]
    assert bundle["held_out"]["test_auc"] == pytest.approx(0.72)
    assert bundle["baseline_held_out"]["test_auc"] == pytest.approx(0.52)
    assert bundle["held_out_lift"]["test_auc"] == pytest.approx(0.20)
    assert all(
        row["held_out_delta"]["test_auc"] == pytest.approx(0.20)
        for row in bundle["paired_fold_deltas"]
    )
    assert len(bundle["source_fold_sha256"]) == 5
    assert len(bundle["baseline_source_fold_sha256"]) == 5


def test_searched_certification_rejects_baseline_comparator_drift(staged_cell):
    cell_root, adir, cell, _, repo_root = staged_cell
    baseline = _baseline(cell_root)
    register_baseline(cell_root, baseline)
    _attempts(adir, cell["cell_id"], completed=12)
    _open_budget_cell(
        adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS,
    )
    freeze_discovery(cell_root)
    materialize_promotion(cell_root, repo_root=repo_root)
    _finish_promotion(cell_root, completed=10, promotion_base=0.75)
    freeze_promotion(cell_root)
    selected = select_winner(cell_root)
    _write_searched_sealed_folds(cell_root, selected)
    _write_global_selection_freeze(cell_root)
    (baseline / "certify/fold_4_result.json").write_text("tampered")

    with pytest.raises(CampaignStageError, match="baseline sealed artifact changed"):
        certify_winner(cell_root)
    assert not (cell_root / "certification").exists()


def test_selected_sealed_corruption_blocks_certification(staged_cell):
    cell_root, adir, cell, _, repo_root = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _attempts(adir, cell["cell_id"], completed=12)
    _open_budget_cell(
        adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS,
    )
    freeze_discovery(cell_root)
    materialize_promotion(cell_root, repo_root=repo_root)
    _finish_promotion(cell_root, completed=10, promotion_base=0.75)
    freeze_promotion(cell_root)
    selected = select_winner(cell_root)
    _write_searched_sealed_folds(cell_root, selected, selected_valid=False)

    with pytest.raises(CampaignStageError, match="sealed artifact changed"):
        _write_global_selection_freeze(cell_root)
    assert load_stage_state(cell_root)["phase"] == "winner-frozen"
    assert not (cell_root.parent / SELECTION_FREEZE_FILE).exists()
    assert not (cell_root / "certification").exists()


def test_existing_certification_bundle_is_immutable(staged_cell):
    cell_root, adir, cell, _, _ = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _attempts(adir, cell["cell_id"], completed=0)
    _open_budget_cell(
        adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS,
    )
    freeze_discovery(cell_root)
    select_winner(cell_root)
    _write_global_selection_freeze(cell_root)
    certify_winner(cell_root)
    bundle = cell_root / "certification/certify.json"
    bundle.write_text(bundle.read_text().replace("0.52", "0.99"))

    with pytest.raises(CampaignStageError, match="certification bundle"):
        certify_winner(cell_root)


def test_existing_certification_rejects_rehashed_state_timestamp_drift(staged_cell):
    cell_root, adir, cell, _, _ = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _attempts(adir, cell["cell_id"], completed=0)
    _open_budget_cell(
        adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS,
    )
    freeze_discovery(cell_root)
    select_winner(cell_root)
    _write_global_selection_freeze(cell_root)
    certify_winner(cell_root)
    state_path = cell_root / "campaign_state.json"
    state = json.loads(state_path.read_text())
    state["certification"]["certified_at"] = "2030-01-01T00:00:00+00:00"
    state["state_sha256"] = content_sha256({
        key: value for key, value in state.items() if key != "state_sha256"
    })
    state_path.write_text(json.dumps(state))

    with pytest.raises(CampaignStageError, match="certified cell state"):
        certify_winner(cell_root)


def test_existing_certification_cannot_predate_selection_freeze(staged_cell):
    cell_root, adir, cell, _, _ = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _attempts(adir, cell["cell_id"], completed=0)
    _open_budget_cell(
        adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS,
    )
    freeze_discovery(cell_root)
    select_winner(cell_root)
    _write_global_selection_freeze(cell_root)
    certify_winner(cell_root)
    bundle_path = cell_root / "certification/certify.json"
    bundle = json.loads(bundle_path.read_text())
    bundle["certified_at"] = "2000-01-01T00:00:00+00:00"
    bundle["bundle_sha256"] = content_sha256({
        key: value for key, value in bundle.items() if key != "bundle_sha256"
    })
    bundle_path.write_text(json.dumps(bundle))
    state_path = cell_root / "campaign_state.json"
    state = json.loads(state_path.read_text())
    state["certification"]["certified_at"] = bundle["certified_at"]
    state["certification"]["bundle_sha256"] = bundle["bundle_sha256"]
    state["state_sha256"] = content_sha256({
        key: value for key, value in state.items() if key != "state_sha256"
    })
    state_path.write_text(json.dumps(state))

    with pytest.raises(CampaignStageError, match="freeze/bundle/index order"):
        certify_winner(cell_root)


def test_first_certification_rejects_clock_before_selection_freeze(
    staged_cell, monkeypatch,
):
    cell_root, adir, cell, _, _ = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _attempts(adir, cell["cell_id"], completed=0)
    _open_budget_cell(
        adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS,
    )
    freeze_discovery(cell_root)
    select_winner(cell_root)
    _write_global_selection_freeze(cell_root)
    monkeypatch.setattr(
        campaign_stages, "_utc_now",
        lambda: "2000-01-01T00:00:00+00:00",
    )

    with pytest.raises(CampaignStageError, match="freeze/bundle/index order"):
        certify_winner(cell_root)
    assert not (cell_root / "certification").exists()
    assert load_stage_state(cell_root)["phase"] == "winner-frozen"


def test_existing_certification_rejects_rehashed_winner_identity(staged_cell):
    cell_root, adir, cell, _, _ = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _attempts(adir, cell["cell_id"], completed=0)
    _open_budget_cell(
        adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS,
    )
    freeze_discovery(cell_root)
    select_winner(cell_root)
    _write_global_selection_freeze(cell_root)
    certify_winner(cell_root)
    bundle_path = cell_root / "certification/certify.json"
    bundle = json.loads(bundle_path.read_text())
    bundle["selection_sha256"] = "1" * 64
    bundle["winner"]["candidate_sha256"] = "2" * 64
    bundle["baseline"]["candidate_sha256"] = "2" * 64
    bundle["bundle_sha256"] = content_sha256({
        key: value for key, value in bundle.items()
        if key != "bundle_sha256"
    })
    bundle_path.write_text(json.dumps(bundle))
    state_path = cell_root / "campaign_state.json"
    state = json.loads(state_path.read_text())
    state["certification"]["bundle_sha256"] = bundle["bundle_sha256"]
    state["state_sha256"] = content_sha256({
        key: value for key, value in state.items()
        if key != "state_sha256"
    })
    state_path.write_text(json.dumps(state))

    with pytest.raises(CampaignStageError, match="frozen validation winner"):
        certify_winner(cell_root)
