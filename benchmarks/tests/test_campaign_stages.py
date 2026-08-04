"""Restart-safe discovery ledger and validation-firewall campaign tests."""
from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from automil.admissibility import load_candidate_policy
from automil.cells.state import Cell, CellStatus, read_cell, write_cell
from autobench.campaign import (
    CAMPAIGN_ID,
    CERTIFICATION_FOLDS,
    DISCOVERY_ATTEMPTS,
    PROTOCOL,
    STAGE_FOLDS,
    content_sha256,
    file_sha256,
)
from autobench.campaign_stages import (
    CampaignStageError,
    certify_winner,
    freeze_discovery,
    freeze_promotion,
    initialize_stage_state,
    load_stage_state,
    materialize_promotion,
    register_baseline,
    run_native_baseline,
    select_winner,
)


def _folds(indices, base=0.6):
    return [
        {
            "fold_index": index,
            "metrics": {"val_auc": base + index / 100, "val_bacc": base},
            "composite": base + index / 200,
        }
        for index in indices
    ]


@pytest.fixture
def staged_cell(tmp_path):
    cell_root = tmp_path / "cell"
    adir = cell_root / "automil"
    adir.mkdir(parents=True)
    cell_without_hash = {
        "cell_id": "dataset__arm__task",
        "dataset": "dataset",
        "task": "task",
        "encoder": "encoder",
        "model": "model",
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
            "cell_id": "b" * 16,
            "dataset": "dataset",
            "encoder": "encoder",
            "mil_model": "model",
            "task": "task",
        },
    }
    cell = {**cell_without_hash, "cell_sha256": content_sha256(cell_without_hash)}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "campaign_id": CAMPAIGN_ID,
        "cells": [cell],
    }))
    manifest_hash = file_sha256(manifest_path)
    (adir / "campaign_cell.json").write_text(json.dumps(cell))
    (adir / "config.yaml").write_text(yaml.safe_dump({
        "registry": {
            "mode": "architecture-preserving",
            "protected": ["models/**"],
            "allowed_override_options": ["--hparams", "--policy-variant"],
            "allowed_variant_kinds": ["policy"],
        },
        "files": {"editable": ["cell/automil/variants/_policies/*.py"]},
        "run": {"command": cell["commands"]["discovery"], "mil_model": "model"},
        "campaign": {
            "campaign_id": CAMPAIGN_ID,
            "manifest": "manifest.json",
            "manifest_sha256": manifest_hash,
            "cell_id": cell["cell_id"],
            "cell_sha256": cell["cell_sha256"],
            "budget_cell_id": cell["budget_identity"]["cell_id"],
            "stage": "discovery",
        },
    }))
    state = initialize_stage_state(
        cell_root, cell=cell, manifest_sha256=manifest_hash,
        base_commit="d" * 40,
    )
    return cell_root, adir, cell, state, tmp_path


def _baseline(cell_root: Path, *, leak=False, invalid_sealed=False) -> Path:
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
            "held_out": {"test_auc": 0.5 + fold / 100},
        })
        (sealed / f"fold_{fold}_result.json").write_text(payload)
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
    policy = load_candidate_policy(adir)
    archive_root = adir / "orchestrator" / "archive"
    for index in range(DISCOVERY_ATTEMPTS):
        node_id = f"node_{index + 1:04d}"
        archive = archive_root / node_id
        archive.mkdir(parents=True)
        candidate_paths = []
        overlay_manifest = {}
        if index == source_at:
            rel = f"cell/automil/variants/_policies/candidate_{index}.py"
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
            "metadata": {
                "cell_id": "b" * 16,
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
        else:
            result = {"status": "crash", "composite": 0.0, "metrics": {}}
        (archive / "result.json").write_text(json.dumps(result))


def test_initial_state_is_restart_idempotent_and_integrity_checked(staged_cell):
    cell_root, _, cell, original, _ = staged_cell
    restarted = initialize_stage_state(
        cell_root, cell=cell, manifest_sha256=original["manifest_sha256"],
        base_commit="d" * 40,
    )
    assert restarted == original
    with pytest.raises(CampaignStageError, match="different campaign inputs"):
        initialize_stage_state(
            cell_root, cell=cell,
            manifest_sha256=original["manifest_sha256"],
            base_commit="e" * 40,
        )
    raw = json.loads((cell_root / "campaign_state.json").read_text())
    raw["phase"] = "winner-frozen"
    (cell_root / "campaign_state.json").write_text(json.dumps(raw))
    with pytest.raises(CampaignStageError, match="integrity hash"):
        load_stage_state(cell_root)


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
                "held_out": {"test_auc": 0.5 + fold / 100},
            }))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("autobench.campaign_stages.subprocess.run", fake_run)
    state = run_native_baseline(cell_root, repo_root=repo_root, gpu_id=3)

    assert observed["base_commit"] == "d" * 40
    assert observed["gpu"] == ("3", "0")
    assert "--folds" in observed["command"]
    assert "0,1,2,3,4" in observed["command"]
    assert state["baseline"]["candidate_id"] == "baseline"
    assert (cell_root / "baseline/archive/result.json").is_file()
    assert (adir / "graph.json").is_file()


def test_external_baseline_is_atomically_imported_and_portable(staged_cell, tmp_path):
    cell_root, adir, cell, _, _ = staged_cell
    external_root = tmp_path / "external-cell"
    external = _baseline(external_root)
    state = register_baseline(cell_root, external)
    imported = cell_root / state["baseline"]["archive"]
    assert imported == cell_root / "baseline/archive"
    assert (imported / "result.json").is_file()
    assert len(list((imported / "certify").glob("fold_*_result.json"))) == 5

    # The registered incumbent no longer depends on the external result tree.
    shutil.rmtree(external_root)
    _attempts(adir, cell["cell_id"], completed=0)
    _open_budget_cell(
        adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS,
    )
    freeze_discovery(cell_root)
    selected = select_winner(cell_root)
    assert selected["winner"]["kind"] == "baseline"


def test_freeze_requires_baseline_and_exact_attempt_budget(staged_cell):
    cell_root, adir, cell, _, _ = staged_cell
    _attempts(adir, cell["cell_id"])
    _open_budget_cell(adir, cell["budget_identity"]["cell_id"], 59)
    with pytest.raises(CampaignStageError, match="baseline"):
        freeze_discovery(cell_root)
    register_baseline(cell_root, _baseline(cell_root))
    with pytest.raises(CampaignStageError, match="exactly 60"):
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
    assert state["discovery"]["attempts_charged"] == 60
    assert state["discovery"]["complete_candidates"] == 12
    assert state["discovery"]["unique_complete_candidates"] == 12
    assert len(state["discovery"]["attempt_audit"]) == 60
    assert len(promoted) == PROTOCOL["promotion_candidates"] == 10
    assert [candidate["candidate_id"] for candidate in promoted] == [
        f"node_{index:04d}" for index in range(12, 2, -1)
    ]
    assert all(set(candidate["validation_folds"][0]) == {
        "fold_index", "metrics", "composite",
    } for candidate in promoted)

    # Restarting cannot spend, reorder, or refreeze anything.
    assert freeze_discovery(cell_root) == state


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
        / "cell/promotion/automil/variants/_policies/candidate_11.py"
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
    assert freeze_promotion(cell_root) == state


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
    # Every loser is deliberately malformed. Certification must not open one.
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
    for fold in CERTIFICATION_FOLDS:
        path = (
            selected_discovery if fold in STAGE_FOLDS["discovery"]
            else selected_promotion
        ) / f"fold_{fold}_result.json"
        path.write_text(
            json.dumps({
                "fold_index": fold,
                "held_out": {"test_auc": 0.70 + fold / 100},
            })
            if selected_valid else "selected-not-json"
        )


def test_baseline_winner_certification_unseals_existing_folds_once(staged_cell):
    cell_root, adir, cell, _, _ = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _attempts(adir, cell["cell_id"], completed=0)
    _open_budget_cell(
        adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS,
    )
    freeze_discovery(cell_root)
    winner_frozen = select_winner(cell_root)

    state = certify_winner(cell_root)
    bundle_path = cell_root / state["certification"]["bundle"]
    bundle = json.loads(bundle_path.read_text())
    assert state["phase"] == "certified"
    assert bundle["winner"]["kind"] == "baseline"
    assert bundle["held_out"]["test_auc"] == pytest.approx(0.52)
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

    state = certify_winner(cell_root)
    bundle = json.loads((cell_root / "certification/certify.json").read_text())
    assert bundle["winner"]["candidate_id"] == selected["winner"]["candidate_id"]
    assert bundle["held_out"]["test_auc"] == pytest.approx(0.72)
    assert len(bundle["source_fold_sha256"]) == 5


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

    with pytest.raises(CampaignStageError, match="selected winner sealed fold"):
        certify_winner(cell_root)
    assert load_stage_state(cell_root)["phase"] == "winner-frozen"
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
    certify_winner(cell_root)
    bundle = cell_root / "certification/certify.json"
    bundle.write_text(bundle.read_text().replace("0.52", "0.99"))

    with pytest.raises(CampaignStageError, match="bundle hash mismatch"):
        certify_winner(cell_root)
