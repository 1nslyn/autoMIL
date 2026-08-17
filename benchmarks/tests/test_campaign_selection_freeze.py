"""Campaign-wide validation freeze gates every held-out certification."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

import autobench.campaign_stages as campaign_stages
from autobench.campaign import (
    AGENT_PROTOCOL_FILE,
    CAMPAIGN_ID,
    DISCOVERY_ATTEMPTS,
    content_sha256,
    file_sha256,
    load_manifest,
)
from autobench.campaign_stages import (
    BASELINE_ATTESTATION_FILE,
    CAMPAIGN_CELL_COUNT,
    CampaignStageError,
    certify_campaign,
    certify_winner,
    freeze_campaign_selections,
    initialize_stage_state,
    validate_selection_freeze_artifact,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "benchmarks/campaigns/preprint_130/manifest.json"
AGENT_PROTOCOL = {
    "schema_version": 2,
    "campaign_id": CAMPAIGN_ID,
    "purpose": "publication",
    "provider": "test-provider",
    "runtime": "test-runtime",
    "runtime_version": "test-runtime-1",
    "model": "test-model",
    "model_version": "test-model-1",
    "effort": "max",
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


def _freeze_ready_state(runtime_root: Path, cell: dict, manifest_hash: str) -> Path:
    cell_root = runtime_root / cell["cell_id"]
    adir = cell_root / "automil"
    adir.mkdir(parents=True)
    (adir / "config.yaml").write_text(json.dumps({
        "campaign": {
            "agent_protocol_sha256": content_sha256(AGENT_PROTOCOL),
            "manifest": "manifest.json",
        },
    }))
    (adir / "campaign_cell.json").write_text(json.dumps(cell))
    state = initialize_stage_state(
        cell_root,
        cell=cell,
        manifest_sha256=manifest_hash,
    )
    candidate_sha256 = content_sha256({"cell_id": cell["cell_id"]})
    baseline_archive = cell_root / "baseline/archive"
    (baseline_archive / "certify").mkdir(parents=True)
    baseline_result = baseline_archive / "result.json"
    baseline_result.write_text(json.dumps({"status": "completed"}))
    sealed_fold_sha256 = {}
    # Evidence in the cell's own family schema — the certify build gate
    # rejects a fold roster that mismatches the frozen task_family.
    if cell["task_family"] == "survival":
        held_out = {"test_c_index": 0.5}
    elif cell["task_family"] == "ordinal":
        held_out = {"test_auc": 0.5, "test_bacc": 0.5, "test_qwk": 0.5}
    else:
        held_out = {"test_auc": 0.5, "test_bacc": 0.5}
    for fold in range(5):
        fold_path = baseline_archive / "certify" / f"fold_{fold}_result.json"
        fold_path.write_text(json.dumps({
            "fold_index": fold,
            "held_out": held_out,
        }))
        sealed_fold_sha256[fold_path.name] = file_sha256(fold_path)
    baseline_attestation = {
        "schema_version": 2,
        "cell_id": cell["cell_id"],
        "identity": cell["identity"],
        "result_sha256": file_sha256(baseline_result),
        "sealed_fold_sha256": sealed_fold_sha256,
    }
    baseline_attestation["attestation_sha256"] = content_sha256(
        baseline_attestation
    )
    (baseline_archive / BASELINE_ATTESTATION_FILE).write_text(
        json.dumps(baseline_attestation)
    )
    session_binding = content_sha256({
        "campaign_id": CAMPAIGN_ID,
        "cell_id": cell["cell_id"],
        "agent_protocol_sha256": content_sha256(AGENT_PROTOCOL),
        "session_id": f"session-{cell['cell_id']}",
        "started_at": "2026-08-04T00:00:00+00:00",
        "bound_at": "2026-08-04T00:00:00+00:00",
    })
    state["phase"] = "winner-frozen"
    state["baseline"] = {
        "candidate_sha256": candidate_sha256,
        "archive": "baseline/archive",
        "result_sha256": file_sha256(baseline_result),
        "sealed_fold_sha256": sealed_fold_sha256,
        "attestation_sha256": baseline_attestation["attestation_sha256"],
        "validation_mean": 0.5,
        "validation_folds": [
            {"fold_index": fold, "composite": 0.5, "metrics": {}}
            for fold in range(5)
        ],
        "result_status": "completed",
        "resources": {
            "elapsed_seconds": {
                "reported": 0, "missing": 1, "maximum": None,
                "total": None, "gpu_attached_job_hours": None,
            },
            "peak_vram_mb": {"reported": 0, "missing": 1, "maximum": None},
        },
    }
    state["discovery"].update({
        "attempts_charged": DISCOVERY_ATTEMPTS,
        "complete_candidates": 0,
        "unique_complete_candidates": 0,
        "frozen": True,
        "attempt_audit": [
            {
                "node_id": f"node_{index + 1:04d}",
                "source_spec_sha256": f"{index + 1:064x}",
                "submitted_at": f"2026-08-04T00:{index:02d}:00+00:00",
                "agent_session_id": f"session-{cell['cell_id']}",
                "agent_session_binding_sha256": session_binding,
                "candidate_class": "config-only",
                "policy_hash": "b" * 64,
                "result_status": "crash",
                "termination_reason": "unspecified",
                "budget_killed": False,
                "outcome_class": "crash",
                "elapsed_seconds": None,
                "peak_vram_mb": None,
                "eligible": False,
                "reason": "fixture crash",
                "candidate_sha256": None,
                "validation_mean": None,
            }
            for index in range(DISCOVERY_ATTEMPTS)
        ],
        "promoted_candidates": [],
    })
    state["promotion"]["attempts_charged"] = 0
    state["winner"] = {
        "kind": "baseline",
        "candidate_id": "baseline",
        "candidate_sha256": candidate_sha256,
        "promotion_node_id": None,
        "validation_mean": 0.5,
        "lift_over_baseline": 0.0,
        "selection_sha256": content_sha256({
            "cell_id": cell["cell_id"],
            "candidate_sha256": candidate_sha256,
        }),
    }
    state["state_sha256"] = content_sha256({
        key: value for key, value in state.items() if key != "state_sha256"
    })
    (cell_root / "campaign_state.json").write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n"
    )
    session = {
        "schema_version": 3,
        "campaign_id": CAMPAIGN_ID,
        "cell_id": cell["cell_id"],
        "agent_protocol_sha256": content_sha256(AGENT_PROTOCOL),
        "status": "finalized",
        "session": {
            "session_id": f"session-{cell['cell_id']}",
            "started_at": "2026-08-04T00:00:00+00:00",
            "bound_at": "2026-08-04T00:00:00+00:00",
            "ended_at": "2026-08-04T01:00:00+00:00",
            "termination_reason": "budget-complete",
            "usage": {
                "status": "unavailable",
                "input_tokens": None,
                "output_tokens": None,
                "cached_input_tokens": None,
                "cost_usd": None,
                "basis": "fixture runtime does not expose usage",
            },
            "activity": {
                "source": "claude-native-active-time-v1",
                "active_seconds": 3600.0,
                "event_count": 3,
                "sha256": "d" * 64,
            },
        },
        "binding_sha256": session_binding,
        "attestation_sha256": None,
    }
    session["attestation_sha256"] = content_sha256({
        key: value for key, value in session.items()
        if key != "attestation_sha256"
    })
    (cell_root / "agent_session.json").write_text(json.dumps(session))
    return cell_root


def _materialize_frozen_roster(runtime_root: Path) -> list[Path]:
    runtime_root.mkdir(parents=True)
    (runtime_root / AGENT_PROTOCOL_FILE).write_text(json.dumps(AGENT_PROTOCOL))
    (runtime_root / "manifest.json").write_bytes(MANIFEST.read_bytes())
    manifest = load_manifest(MANIFEST)
    manifest_hash = file_sha256(MANIFEST)
    return [
        _freeze_ready_state(runtime_root, cell, manifest_hash)
        for cell in manifest["cells"]
    ]


def test_campaign_freeze_fails_closed_until_all_130_winners_exist(tmp_path):
    runtime_root = tmp_path / "runtime"
    roots = _materialize_frozen_roster(runtime_root)
    missing = roots[-1]
    backup = tmp_path / "missing-cell"
    missing.rename(backup)

    with pytest.raises(CampaignStageError, match="runtime roster differs"):
        freeze_campaign_selections(runtime_root, MANIFEST)

    backup.rename(missing)
    artifact = freeze_campaign_selections(runtime_root, MANIFEST)
    assert artifact["cell_count"] == CAMPAIGN_CELL_COUNT == 130
    assert len(artifact["cells"]) == 130


def test_campaign_freeze_validates_before_atomic_publication(tmp_path, monkeypatch):
    runtime_root = tmp_path / "runtime"
    _materialize_frozen_roster(runtime_root)
    original = campaign_stages._process_evidence

    def duplicate_attempt_identity(state):
        process = original(state)
        attempts = process["discovery"]["attempts"]
        attempts[1]["node_id"] = attempts[0]["node_id"]
        process["discovery"]["validation_anytime"][1]["node_id"] = (
            attempts[0]["node_id"]
        )
        return process

    monkeypatch.setattr(
        campaign_stages, "_process_evidence", duplicate_attempt_identity,
    )

    with pytest.raises(CampaignStageError, match="identities are not unique"):
        freeze_campaign_selections(runtime_root, MANIFEST)
    assert not (runtime_root / "selection_freeze.json").exists()


@pytest.mark.parametrize("mutation", ["dot", "duplicate-slash", "sibling-cell"])
def test_selection_freeze_rejects_noncanonical_or_cross_cell_source_paths(
    tmp_path, mutation,
):
    runtime_root = tmp_path / "runtime"
    _materialize_frozen_roster(runtime_root)
    artifact = freeze_campaign_selections(runtime_root, MANIFEST)
    entry = artifact["cells"][0]
    winner_record = entry["winner_source_folds"]["fold_0_result.json"]
    baseline_record = entry["baseline_source_folds"]["fold_0_result.json"]
    original = winner_record["path"]
    if mutation == "dot":
        mutated = original.replace("/certify/", "/certify/./", 1)
    elif mutation == "duplicate-slash":
        mutated = original.replace("/certify/", "//certify/", 1)
    else:
        sibling = artifact["cells"][1]["cell_id"]
        mutated = original.replace(entry["cell_id"], sibling, 1)
    winner_record["path"] = mutated
    baseline_record["path"] = mutated
    artifact["freeze_sha256"] = content_sha256({
        key: value for key, value in artifact.items() if key != "freeze_sha256"
    })

    with pytest.raises(CampaignStageError, match="source-fold anchor"):
        validate_selection_freeze_artifact(artifact)


def test_global_freeze_binds_every_cell_winner_and_blocks_later_drift(tmp_path):
    runtime_root = tmp_path / "runtime"
    roots = _materialize_frozen_roster(runtime_root)
    artifact = freeze_campaign_selections(runtime_root, MANIFEST)
    first = roots[0]
    baseline_result = first / "baseline/archive/result.json"
    original_result = baseline_result.read_bytes()
    baseline_result.write_text("tampered")

    with pytest.raises(CampaignStageError, match="baseline validation artifact changed"):
        certify_winner(first)
    baseline_result.write_bytes(original_result)

    state_path = first / "campaign_state.json"
    state = json.loads(state_path.read_text())
    state["winner"]["candidate_sha256"] = "f" * 64
    state["state_sha256"] = content_sha256({
        key: value for key, value in state.items() if key != "state_sha256"
    })
    state_path.write_text(json.dumps(state))

    with pytest.raises(CampaignStageError, match="global selection freeze"):
        certify_winner(first)
    assert artifact["freeze_sha256"]


def test_certify_campaign_indexes_exactly_the_frozen_130_bundles(
    tmp_path, monkeypatch,
):
    runtime_root = tmp_path / "runtime"
    roots = _materialize_frozen_roster(runtime_root)
    freeze = freeze_campaign_selections(runtime_root, MANIFEST)
    frozen_states = {
        row["cell_id"]: row["state_sha256"] for row in freeze["cells"]
    }
    frozen_entries = {row["cell_id"]: row for row in freeze["cells"]}
    visited: list[str] = []

    def fake_certify(cell_root: Path) -> dict:
        state = json.loads((cell_root / "campaign_state.json").read_text())
        visited.append(state["cell_id"])
        cell = json.loads(
            (cell_root / "automil" / "campaign_cell.json").read_text()
        )
        # Mirror the family-shaped sealed evidence _freeze_ready_state wrote.
        if cell["task_family"] == "survival":
            held_out = {"test_c_index": 0.5}
        elif cell["task_family"] == "ordinal":
            held_out = {"test_auc": 0.5, "test_bacc": 0.5, "test_qwk": 0.5}
        else:
            held_out = {"test_auc": 0.5, "test_bacc": 0.5}
        zero_delta = {key: 0.0 for key in held_out}
        held_out_folds = [
            {
                "fold_index": fold,
                "held_out": held_out,
            }
            for fold in range(5)
        ]
        bundle = {
            "schema_version": 2,
            "campaign_id": state["campaign_id"],
            "cell_id": state["cell_id"],
            "selection_freeze_sha256": freeze["freeze_sha256"],
            "selection_state_sha256": frozen_states[state["cell_id"]],
            "selection_sha256": state["winner"]["selection_sha256"],
            "winner": {
                "kind": "baseline",
                "candidate_id": "baseline",
                "candidate_sha256": state["winner"]["candidate_sha256"],
                "promotion_node_id": None,
            },
            "validation_mean": 0.5,
            "baseline": {
                "candidate_id": "baseline",
                "candidate_sha256": state["baseline"]["candidate_sha256"],
                "validation_mean": 0.5,
            },
            "held_out_folds": held_out_folds,
            "held_out": held_out,
            "source_fold_sha256": {
                filename: record["sha256"]
                for filename, record in frozen_entries[state["cell_id"]][
                    "winner_source_folds"
                ].items()
            },
            "baseline_held_out_folds": held_out_folds,
            "baseline_held_out": held_out,
            "baseline_source_fold_sha256": {
                filename: record["sha256"]
                for filename, record in frozen_entries[state["cell_id"]][
                    "baseline_source_folds"
                ].items()
            },
            "paired_fold_deltas": [
                {
                    "fold_index": fold,
                    "held_out_delta": zero_delta,
                }
                for fold in range(5)
            ],
            "held_out_lift": zero_delta,
            "retrained": False,
            "certified_at": freeze["frozen_at"],
        }
        bundle["bundle_sha256"] = content_sha256(bundle)
        bundle_path = cell_root / "certification/certify.json"
        bundle_path.parent.mkdir()
        bundle_path.write_text(json.dumps(bundle))
        state["phase"] = "certified"
        state["certification"] = {
            "bundle": "certification/certify.json",
            "bundle_sha256": bundle["bundle_sha256"],
            "selection_state_sha256": frozen_states[state["cell_id"]],
            "certified_at": bundle["certified_at"],
        }
        return campaign_stages._commit_state(cell_root, state)

    monkeypatch.setattr(campaign_stages, "certify_winner", fake_certify)
    index = certify_campaign(runtime_root, MANIFEST)
    first_bytes = (runtime_root / "campaign_certification.json").read_bytes()
    restarted = certify_campaign(runtime_root, MANIFEST)

    assert len(visited) == len(set(visited)) == CAMPAIGN_CELL_COUNT
    assert set(visited) == {root.name for root in roots}
    assert index["cell_count"] == 130
    assert len(index["cells"]) == 130
    assert restarted == index
    assert (runtime_root / "campaign_certification.json").read_bytes() == first_bytes
    assert (runtime_root / "campaign_certification.json").is_file()

    index_path = runtime_root / "campaign_certification.json"
    first_entry = index["cells"][0]
    bundle_path = runtime_root / first_entry["bundle"]
    original_bundle = bundle_path.read_bytes()
    forged_bundle = json.loads(bundle_path.read_text())
    forged_bundle["selection_sha256"] = "1" * 64
    forged_bundle["winner"]["candidate_sha256"] = "2" * 64
    forged_bundle["baseline"]["candidate_sha256"] = "2" * 64
    forged_bundle["bundle_sha256"] = content_sha256({
        key: value for key, value in forged_bundle.items()
        if key != "bundle_sha256"
    })
    bundle_path.write_text(json.dumps(forged_bundle))
    forged_index = json.loads(index_path.read_text())
    forged_entry = next(
        row for row in forged_index["cells"]
        if row["cell_id"] == first_entry["cell_id"]
    )
    forged_entry["bundle_sha256"] = forged_bundle["bundle_sha256"]
    forged_entry["file_sha256"] = file_sha256(bundle_path)
    forged_index["certification_sha256"] = content_sha256({
        key: value for key, value in forged_index.items()
        if key != "certification_sha256"
    })
    index_path.write_text(json.dumps(forged_index))

    with pytest.raises(CampaignStageError, match="frozen validation winner"):
        certify_campaign(runtime_root, MANIFEST)
    assert len(visited) == CAMPAIGN_CELL_COUNT

    bundle_path.write_bytes(original_bundle)
    index_path.write_bytes(first_bytes)
    tampered = json.loads(index_path.read_text())
    tampered["cells"][0]["file_sha256"] = "0" * 64
    tampered["certification_sha256"] = content_sha256({
        key: value for key, value in tampered.items()
        if key != "certification_sha256"
    })
    index_path.write_text(json.dumps(tampered))

    with pytest.raises(CampaignStageError, match="bundle integrity mismatch"):
        certify_campaign(runtime_root, MANIFEST)
    assert len(visited) == CAMPAIGN_CELL_COUNT
    assert json.loads(index_path.read_text()) == tampered


def test_certify_campaign_restart_rejects_rehashed_invalid_timestamp(
    tmp_path, monkeypatch,
):
    runtime_root = tmp_path / "runtime"
    _materialize_frozen_roster(runtime_root)
    freeze_campaign_selections(runtime_root, MANIFEST)

    original = campaign_stages.certify_winner

    def certify_with_real_state(cell_root: Path) -> dict:
        return original(cell_root)

    monkeypatch.setattr(campaign_stages, "certify_winner", certify_with_real_state)
    certify_campaign(runtime_root, MANIFEST)
    index_path = runtime_root / "campaign_certification.json"
    index = json.loads(index_path.read_text())
    index["certified_at"] = "not-a-time"
    index["certification_sha256"] = content_sha256({
        key: value for key, value in index.items()
        if key != "certification_sha256"
    })
    index_path.write_text(json.dumps(index))

    with pytest.raises(CampaignStageError, match="index integrity"):
        certify_campaign(runtime_root, MANIFEST)


def test_first_campaign_index_rejects_clock_before_certified_bundles(
    tmp_path, monkeypatch,
):
    runtime_root = tmp_path / "runtime"
    roots = _materialize_frozen_roster(runtime_root)
    freeze_campaign_selections(runtime_root, MANIFEST)
    for cell_root in roots:
        certify_winner(cell_root)
    monkeypatch.setattr(
        campaign_stages, "_utc_now",
        lambda: "2000-01-01T00:00:00+00:00",
    )

    with pytest.raises(CampaignStageError, match="freeze/bundle/index order"):
        certify_campaign(runtime_root, MANIFEST)
    assert not (runtime_root / "campaign_certification.json").exists()
