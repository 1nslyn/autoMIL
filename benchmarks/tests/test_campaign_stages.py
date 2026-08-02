"""Restart-safe discovery ledger and validation-firewall campaign tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from automil.admissibility import load_candidate_policy
from automil.cells.state import Cell, CellStatus, write_cell
from autobench.campaign import (
    CAMPAIGN_ID,
    CERTIFICATION_FOLDS,
    DISCOVERY_ATTEMPTS,
    PROTOCOL,
    STAGE_FOLDS,
)
from autobench.campaign_stages import (
    CampaignStageError,
    freeze_discovery,
    initialize_stage_state,
    load_stage_state,
    register_baseline,
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
    cell = {
        "cell_id": "dataset__arm__task",
        "cell_sha256": "a" * 64,
        "budget_identity": {"cell_id": "b" * 16},
    }
    (adir / "campaign_cell.json").write_text(json.dumps(cell))
    (adir / "config.yaml").write_text(yaml.safe_dump({
        "registry": {
            "mode": "architecture-preserving",
            "protected": ["models/**"],
            "allowed_override_options": ["--hparams"],
            "allowed_variant_kinds": ["policy"],
        },
        "files": {"editable": ["cell/automil/variants/_policies/*.py"]},
        "run": {"command": "python train.py --folds 0,1,2"},
    }))
    state = initialize_stage_state(
        cell_root, cell=cell, manifest_sha256="c" * 64,
    )
    return cell_root, adir, cell, state


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


def _attempts(adir: Path, cell_id: str, *, completed=12) -> None:
    policy = load_candidate_policy(adir)
    archive_root = adir / "orchestrator" / "archive"
    for index in range(DISCOVERY_ATTEMPTS):
        node_id = f"node_{index + 1:04d}"
        archive = archive_root / node_id
        archive.mkdir(parents=True)
        override = f'--hparams \'{{"lr":{0.0001 + index / 1_000_000:.7f}}}\''
        verdict = policy.classify([], override=override)
        spec = {
            "id": node_id,
            "base_commit": "d" * 40,
            "overlay_manifest": {},
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
    cell_root, _, cell, original = staged_cell
    restarted = initialize_stage_state(
        cell_root, cell=cell, manifest_sha256="c" * 64,
    )
    assert restarted == original
    raw = json.loads((cell_root / "campaign_state.json").read_text())
    raw["phase"] = "winner-frozen"
    (cell_root / "campaign_state.json").write_text(json.dumps(raw))
    with pytest.raises(CampaignStageError, match="integrity hash"):
        load_stage_state(cell_root)


def test_baseline_registration_hashes_but_does_not_parse_sealed_test(staged_cell):
    cell_root, _, _, _ = staged_cell
    state = register_baseline(
        cell_root, _baseline(cell_root, invalid_sealed=True),
    )
    assert state["baseline"]["candidate_id"] == "baseline"
    assert len(state["baseline"]["sealed_fold_sha256"]) == 5
    assert state["baseline"]["validation_mean"] == pytest.approx(0.61)


def test_baseline_registration_rejects_test_bearing_public_result(staged_cell):
    cell_root, _, _, _ = staged_cell
    with pytest.raises(CampaignStageError, match="test-bearing"):
        register_baseline(cell_root, _baseline(cell_root, leak=True))


def test_freeze_requires_baseline_and_exact_attempt_budget(staged_cell):
    cell_root, adir, cell, _ = staged_cell
    _attempts(adir, cell["cell_id"])
    _open_budget_cell(adir, cell["budget_identity"]["cell_id"], 59)
    with pytest.raises(CampaignStageError, match="baseline"):
        freeze_discovery(cell_root)
    register_baseline(cell_root, _baseline(cell_root))
    with pytest.raises(CampaignStageError, match="exactly 60"):
        freeze_discovery(cell_root)


def test_freeze_charges_failures_and_promotes_top_ten_complete(staged_cell):
    cell_root, adir, cell, _ = staged_cell
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
    cell_root, adir, cell, _ = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _attempts(adir, cell["cell_id"], completed=0)
    _open_budget_cell(
        adir, cell["budget_identity"]["cell_id"], DISCOVERY_ATTEMPTS,
    )
    state = freeze_discovery(cell_root)
    assert state["phase"] == "selection-ready"
    assert state["discovery"]["complete_candidates"] == 0
    assert state["discovery"]["promoted_candidates"] == []
