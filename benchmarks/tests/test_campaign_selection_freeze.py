"""Campaign-wide validation freeze gates every held-out certification."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import uuid
from pathlib import Path

import pytest

import autobench.campaign_stages as campaign_stages
from autobench.campaign import (
    ACTIVE_ROSTER,
    AGENT_PROTOCOL_FILE,
    ATTEMPT_OUTCOME_CLASSES,
    CAMPAIGN_ID,
    DISCOVERY_ATTEMPTS,
    PROTOCOL_VERSION,
    content_sha256,
    max_lift_winner,
    file_sha256,
    load_manifest,
    unique_complete_sources,
)
from autobench.campaign_stages import (
    BASELINE_ATTESTATION_FILE,
    CAMPAIGN_CELL_COUNT,
    SELECTION_FREEZE_SCHEMA_VERSION,
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
    baseline_folds = [
        {"fold_index": fold, "primary_value": 0.5, "metrics": {}}
        for fold in range(5)
    ]
    state["phase"] = "winner-frozen"
    state["baseline"] = {
        "candidate_sha256": candidate_sha256,
        "archive": "baseline/archive",
        "result_sha256": file_sha256(baseline_result),
        "sealed_fold_sha256": sealed_fold_sha256,
        "attestation_sha256": baseline_attestation["attestation_sha256"],
        "validation_mean": 0.5,
        "validation_folds": baseline_folds,
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
                "attempt_seq": index + 1,
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
    })
    # No attempt completed, so the pool is empty and the rule keeps the
    # baseline; the record is the rule's own, never written by hand.
    selection = max_lift_winner(
        [fold["primary_value"] for fold in baseline_folds], [],
    )
    state["winner"] = {
        "kind": "baseline",
        "candidate_id": "baseline",
        "candidate_sha256": candidate_sha256,
        "sealed_fold_sha256": None,
        "validation_folds": baseline_folds,
        "validation_mean": 0.5,
        "baseline_validation_mean": 0.5,
        "lift_over_baseline": 0.0,
        "selection": selection,
        "selection_sha256": content_sha256(selection),
        "selected_at": "2026-08-04T01:00:00+00:00",
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
    # The manifest stays the frozen 130-cell superset; only the active
    # roster's cells are ever materialized into a real runtime.
    return [
        _freeze_ready_state(runtime_root, cell, manifest_hash)
        for cell in manifest["cells"]
        if cell["dataset"] in ACTIVE_ROSTER["cohorts"]
    ]


def test_campaign_freeze_fails_closed_until_all_roster_winners_exist(tmp_path):
    runtime_root = tmp_path / "runtime"
    roots = _materialize_frozen_roster(runtime_root)
    missing = roots[-1]
    backup = tmp_path / "missing-cell"
    missing.rename(backup)

    with pytest.raises(CampaignStageError, match="runtime roster differs"):
        freeze_campaign_selections(runtime_root, MANIFEST)

    backup.rename(missing)
    artifact = freeze_campaign_selections(runtime_root, MANIFEST)
    assert artifact["cell_count"] == CAMPAIGN_CELL_COUNT
    assert len(artifact["cells"]) == CAMPAIGN_CELL_COUNT


def test_campaign_freeze_validates_before_atomic_publication(tmp_path, monkeypatch):
    runtime_root = tmp_path / "runtime"
    _materialize_frozen_roster(runtime_root)
    original = campaign_stages._process_evidence

    def duplicate_attempt_identity(cell_root, state):
        process = original(cell_root, state)
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


def test_certify_campaign_indexes_exactly_the_frozen_roster_bundles(
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
            "schema_version": 3,
            "campaign_id": state["campaign_id"],
            "cell_id": state["cell_id"],
            "selection_freeze_sha256": freeze["freeze_sha256"],
            "selection_state_sha256": frozen_states[state["cell_id"]],
            "selection_sha256": state["winner"]["selection_sha256"],
            "winner": {
                "kind": "baseline",
                "candidate_id": "baseline",
                "candidate_sha256": state["winner"]["candidate_sha256"],
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
    assert index["cell_count"] == CAMPAIGN_CELL_COUNT
    assert len(index["cells"]) == CAMPAIGN_CELL_COUNT
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


def test_campaign_freeze_refuses_an_extra_off_roster_cell_even_with_valid_state(
    tmp_path,
):
    """The roster, not "does this directory look legitimate", is the census
    authority: an off-roster cell directory is refused even when its
    campaign_state.json is exactly as well-formed as every roster cell's."""
    runtime_root = tmp_path / "runtime"
    _materialize_frozen_roster(runtime_root)
    manifest = load_manifest(MANIFEST)
    manifest_hash = file_sha256(MANIFEST)
    off_roster_cell = next(
        cell for cell in manifest["cells"]
        if cell["dataset"] not in ACTIVE_ROSTER["cohorts"]
    )
    _freeze_ready_state(runtime_root, off_roster_cell, manifest_hash)

    with pytest.raises(CampaignStageError, match="runtime roster differs") as exc_info:
        freeze_campaign_selections(runtime_root, MANIFEST)
    assert "unexpected=" in str(exc_info.value)
    assert off_roster_cell["cell_id"] in str(exc_info.value)


# ---------------------------------------------------------------------------
# A freeze entry must agree with the winner rule's own record. A searched
# winner needs a real candidate spec on disk to pass through
# freeze_campaign_selections, so these tests assemble a one-cell freeze whose
# evidence is genuine where the validator reads it (the selection is the rule's
# output on the census pool, the process evidence reconciles with it, every
# hash is recomputed) and forge one field of the entry, re-hashing the freeze
# so that field is the only thing wrong.

#: The first attempts to complete: (candidate sha256, validation mean), every
#: fold at that mean, over a 0.5 baseline. The rule elects the first.
SEARCHED_RUNS = (("d" * 64, 0.7), ("e" * 64, 0.6))
BASELINE_VALUE = 0.5


def _missing_resources(count: int) -> dict:
    return {
        "elapsed_seconds": {
            "reported": 0, "missing": count, "maximum": None,
            "total": None, "gpu_attached_job_hours": None,
        },
        "peak_vram_mb": {"reported": 0, "missing": count, "maximum": None},
    }


def _searched_attempt(index: int, session_id: str, binding: str) -> dict:
    run = SEARCHED_RUNS[index] if index < len(SEARCHED_RUNS) else None
    return {
        "node_id": f"node_{index + 1:04d}",
        "source_spec_sha256": f"{index + 1:064x}",
        "submitted_at": f"2026-08-04T00:{index:02d}:00+00:00",
        "attempt_seq": index + 1,
        "agent_session_id": session_id,
        "agent_session_binding_sha256": binding,
        "candidate_class": "config-only",
        "policy_hash": "b" * 64,
        "result_status": "crash" if run is None else "completed",
        "termination_reason": "unspecified",
        "budget_killed": False,
        "outcome_class": "crash" if run is None else "completed",
        "elapsed_seconds": None,
        "peak_vram_mb": None,
        "eligible": run is not None,
        "reason": "fixture crash" if run is None else "complete",
        "candidate_sha256": None if run is None else run[0],
        "validation_mean": None if run is None else run[1],
        "outcome_sha256": None if run is None else content_sha256({"run": index}),
    }


def _searched_process(session_id: str, binding: str) -> dict:
    """Process evidence (schema 3) of a cell whose first attempts completed and
    whose winner is the rule's choice among them."""
    attempts = [
        _searched_attempt(index, session_id, binding)
        for index in range(DISCOVERY_ATTEMPTS)
    ]
    means = {row["node_id"]: row["validation_mean"] for row in attempts}
    selection = max_lift_winner(
        [BASELINE_VALUE] * 5,
        [
            {
                "candidate_id": source["source_node_id"],
                "candidate_sha256": source["source_candidate_sha256"],
                "fold_values": [means[source["source_node_id"]]] * 5,
            }
            for source in unique_complete_sources(attempts)
        ],
    )
    assert selection["accepted"], "the fixture pool must elect its leader"
    best, best_id, anytime = BASELINE_VALUE, "baseline", []
    for index, row in enumerate(attempts, 1):
        if row["eligible"] and row["validation_mean"] > best:
            best, best_id = row["validation_mean"], row["node_id"]
        anytime.append({
            "attempt_index": index,
            "node_id": row["node_id"],
            "result_status": row["result_status"],
            "outcome_class": row["outcome_class"],
            "eligible": row["eligible"],
            "validation_mean": row["validation_mean"],
            "running_best_candidate_id": best_id,
            "running_best_validation_mean": best,
        })
    completed = len(SEARCHED_RUNS)
    outcomes = {"completed": completed, "crash": DISCOVERY_ATTEMPTS - completed}
    return {
        "schema_version": 3,
        "baseline": {
            "folds": list(range(5)),
            "result_status": "completed",
            "resources": _missing_resources(1),
        },
        "discovery": {
            "attempt_budget": DISCOVERY_ATTEMPTS,
            "attempts_charged": DISCOVERY_ATTEMPTS,
            "baseline_validation_mean": BASELINE_VALUE,
            "complete_candidates": completed,
            "unique_complete_candidates": completed,
            "candidate_class_counts": {
                "config-only": DISCOVERY_ATTEMPTS,
                "train-only-source": 0,
                "inadmissible": 0,
            },
            "result_status_counts": outcomes,
            "outcome_class_counts": {
                outcome: outcomes.get(outcome, 0)
                for outcome in ATTEMPT_OUTCOME_CLASSES
            },
            "attempts": attempts,
            "validation_anytime": anytime,
            "resources": _missing_resources(DISCOVERY_ATTEMPTS),
        },
        "selection": selection,
    }


def _rehashed(artifact: dict) -> dict:
    return {
        **artifact,
        "freeze_sha256": content_sha256({
            key: value for key, value in artifact.items()
            if key != "freeze_sha256"
        }),
    }


def _searched_freeze_artifact() -> dict:
    """A one-cell selection freeze whose winner is a searched candidate."""
    cell = next(
        row for row in load_manifest(MANIFEST)["cells"]
        if row["dataset"] in ACTIVE_ROSTER["cohorts"]
    )
    cell_id = cell["cell_id"]
    session_id = f"session-{cell_id}"
    binding = content_sha256({"binding": cell_id})
    process = _searched_process(session_id, binding)
    selection = process["selection"]
    winner = next(
        row for row in process["discovery"]["attempts"]
        if row["node_id"] == selection["winner"]
    )

    def anchors(label: str) -> dict:
        return {
            f"fold_{fold}_result.json": {
                "path": f"{cell_id}/{label}/fold_{fold}_result.json",
                "sha256": content_sha256({"label": label, "fold": fold}),
            }
            for fold in range(5)
        }

    entry = {
        "cell_id": cell_id,
        "cell_sha256": cell["cell_sha256"],
        "state_sha256": content_sha256({"state": cell_id}),
        "selection_sha256": content_sha256(selection),
        "winner_kind": "searched",
        "winner_candidate_id": winner["node_id"],
        "winner_candidate_sha256": winner["candidate_sha256"],
        "winner_validation_mean": winner["validation_mean"],
        "baseline_validation_mean": BASELINE_VALUE,
        "baseline_candidate_sha256": content_sha256({"baseline": cell_id}),
        "winner_source_folds": anchors("winner"),
        "baseline_source_folds": anchors("baseline"),
        "agent_session_sha256": content_sha256({"session": cell_id}),
        "agent_session_id": session_id,
        "agent_session_binding_sha256": binding,
        "agent_usage": {
            "status": "unavailable",
            "input_tokens": None,
            "output_tokens": None,
            "cached_input_tokens": None,
            "cost_usd": None,
            "basis": "fixture runtime does not expose usage",
        },
        "process_sha256": content_sha256(process),
        "process_evidence": process,
    }
    return _rehashed({
        "schema_version": SELECTION_FREEZE_SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "manifest_sha256": file_sha256(MANIFEST),
        "protocol_version": PROTOCOL_VERSION,
        "agent_protocol_sha256": content_sha256(AGENT_PROTOCOL),
        "roster_sha256": content_sha256({cell_id: cell["cell_sha256"]}),
        "cell_count": 1,
        "cells": [entry],
        "frozen_at": "2026-08-04T02:00:00+00:00",
    })


def _forged(artifact: dict, **changes: object) -> dict:
    """The artifact with its one entry's fields replaced and the freeze
    re-hashed: the forgery passes the integrity hash."""
    return _rehashed({**artifact, "cells": [{**artifact["cells"][0], **changes}]})


def test_selection_freeze_accepts_the_winner_the_rule_elected():
    artifact = _searched_freeze_artifact()
    entry = artifact["cells"][0]
    selection = entry["process_evidence"]["selection"]

    assert validate_selection_freeze_artifact(artifact) == artifact
    assert entry["winner_candidate_id"] == selection["winner"] != "baseline"


def test_selection_freeze_rejects_a_winner_the_rule_did_not_elect():
    artifact = _searched_freeze_artifact()
    process = artifact["cells"][0]["process_evidence"]
    selection = process["selection"]
    runner_up = next(
        row for row in selection["pool"]
        if row["candidate_id"] != selection["winner"]
    )
    attempt = next(
        row for row in process["discovery"]["attempts"]
        if row["node_id"] == runner_up["candidate_id"]
    )
    # An entry that is coherent for the runner-up: only the rule's record
    # disagrees with it.
    forged = _forged(
        artifact,
        winner_candidate_id=runner_up["candidate_id"],
        winner_candidate_sha256=runner_up["candidate_sha256"],
        winner_validation_mean=attempt["validation_mean"],
    )

    with pytest.raises(
        CampaignStageError, match="winner differs from the winner rule's record",
    ):
        validate_selection_freeze_artifact(forged)


def test_selection_freeze_rejects_a_selection_hash_that_is_not_the_records():
    forged = _forged(
        _searched_freeze_artifact(),
        selection_sha256=content_sha256({"selection": "another record"}),
    )

    with pytest.raises(
        CampaignStageError, match="winner differs from the winner rule's record",
    ):
        validate_selection_freeze_artifact(forged)


@pytest.mark.parametrize("field, value", [
    ("winner_candidate_sha256", "1" * 64),
    ("winner_validation_mean", 0.65),
])
def test_selection_freeze_rejects_a_searched_winner_that_differs_from_its_pool_entry(
    field, value,
):
    forged = _forged(_searched_freeze_artifact(), **{field: value})

    with pytest.raises(
        CampaignStageError, match="searched winner differs from process evidence",
    ):
        validate_selection_freeze_artifact(forged)


def test_selection_freeze_rejects_a_baseline_mean_that_differs_from_process_evidence():
    forged = _forged(
        _searched_freeze_artifact(), baseline_validation_mean=BASELINE_VALUE + 0.01,
    )

    with pytest.raises(
        CampaignStageError, match="baseline mean differs from process evidence",
    ):
        validate_selection_freeze_artifact(forged)


# ---------------------------------------------------------------------------
# Rehearsal sets: a set with <set>.roster.json beside it freezes and certifies
# exactly the cells its roster names, never the publication grid.

CAMPAIGN_DIR = REPO_ROOT / "benchmarks/campaigns/preprint_130"
#: The committed aihub rehearsal set the tests below copy: two TCGA-HNSC grade
#: cells (a tile arm and the slide arm).
REHEARSAL_SET = "runtime-aihub-hnsc-a"
AIHUB_ROSTER = json.loads((CAMPAIGN_DIR / f"{REHEARSAL_SET}.roster.json").read_text())
AIHUB_IDS = sorted(AIHUB_ROSTER["cell_ids"])


def _manifest_rows() -> dict[str, dict]:
    return {cell["cell_id"]: cell for cell in load_manifest(MANIFEST)["cells"]}


def _materialize_rehearsal_set(
    tmp_path: Path, roster: object = None, name: str = REHEARSAL_SET,
    cell_ids: list[str] | None = None,
) -> Path:
    """The agent protocol, the manifest, ``<name>.roster.json`` beside the set
    (a copy of the committed aihub roster unless given; a string is written
    verbatim), and one freeze-ready root per cell (default: the aihub cells)."""
    runtime_root = tmp_path / name
    runtime_root.mkdir(parents=True)
    (runtime_root / AGENT_PROTOCOL_FILE).write_text(json.dumps(AGENT_PROTOCOL))
    (runtime_root / "manifest.json").write_bytes(MANIFEST.read_bytes())
    roster = AIHUB_ROSTER if roster is None else roster
    (tmp_path / f"{name}.roster.json").write_text(
        roster if isinstance(roster, str) else json.dumps(roster)
    )
    rows = _manifest_rows()
    manifest_hash = file_sha256(MANIFEST)
    for cell_id in AIHUB_IDS if cell_ids is None else cell_ids:
        _freeze_ready_state(runtime_root, rows[cell_id], manifest_hash)
    return runtime_root


def _certification_untouched(cell_root: Path) -> bool:
    state = json.loads((cell_root / "campaign_state.json").read_text())
    return state["phase"] == "winner-frozen" and not (
        cell_root / "certification"
    ).exists()


@pytest.mark.parametrize(
    "roster_path", sorted(CAMPAIGN_DIR.glob("*.roster.json")),
    ids=lambda path: path.name,
)
def test_every_committed_rehearsal_roster_resolves_to_its_own_cells(roster_path):
    roster = json.loads(roster_path.read_text())
    set_root = CAMPAIGN_DIR / roster_path.name.removesuffix(".roster.json")

    rows = campaign_stages._set_census(set_root, load_manifest(MANIFEST)["cells"])

    assert sorted(row["cell_id"] for row in rows) == sorted(roster["cell_ids"])


def test_a_rehearsal_set_freezes_and_certifies_exactly_its_roster(tmp_path):
    runtime_root = _materialize_rehearsal_set(tmp_path)

    freeze = freeze_campaign_selections(runtime_root, MANIFEST)
    rows = _manifest_rows()
    assert freeze["cell_count"] == len(AIHUB_IDS) == 2
    assert sorted(row["cell_id"] for row in freeze["cells"]) == AIHUB_IDS
    assert freeze["roster_sha256"] == content_sha256(
        campaign_stages._roster_payload([rows[cell_id] for cell_id in AIHUB_IDS])
    )

    index = certify_campaign(runtime_root, MANIFEST)
    index_path = runtime_root / "campaign_certification.json"
    first_bytes = index_path.read_bytes()
    assert index["cell_count"] == len(AIHUB_IDS)
    assert sorted(row["cell_id"] for row in index["cells"]) == AIHUB_IDS
    for cell_id in AIHUB_IDS:
        state = json.loads((runtime_root / cell_id / "campaign_state.json").read_text())
        bundle = json.loads(
            (runtime_root / cell_id / "certification/certify.json").read_text()
        )
        assert state["phase"] == "certified"
        assert bundle["retrained"] is False
    assert certify_campaign(runtime_root, MANIFEST) == index
    assert index_path.read_bytes() == first_bytes


_DROP = object()


def _roster_with(**changes: object) -> dict:
    roster = json.loads(json.dumps(AIHUB_ROSTER))
    for key, value in changes.items():
        if value is _DROP:
            roster.pop(key)
        else:
            roster[key] = value
    return roster


def _off_roster_id() -> str:
    return next(
        cell["cell_id"] for cell in load_manifest(MANIFEST)["cells"]
        if cell["dataset"] not in ACTIVE_ROSTER["cohorts"]
    )


def _active_ids() -> list[str]:
    return [
        cell["cell_id"] for cell in load_manifest(MANIFEST)["cells"]
        if cell["dataset"] in ACTIVE_ROSTER["cohorts"]
    ]


BAD_ROSTERS = {
    "unknown-id": lambda: _roster_with(
        cell_ids=AIHUB_IDS[:-1] + ["tcga_hnsc__grade__nope__abmil__s42__preprint-v5"],
    ),
    "off-roster-id": lambda: _roster_with(cell_ids=AIHUB_IDS[:-1] + [_off_roster_id()]),
    "duplicate-id": lambda: _roster_with(cell_ids=AIHUB_IDS[:-1] + [AIHUB_IDS[0]]),
    "empty": lambda: _roster_with(cell_ids=[], cells=0, cohorts=[]),
    "wrong-purpose": lambda: _roster_with(purpose="publication"),
    "missing-purpose": lambda: _roster_with(purpose=_DROP),
    "extra-key": lambda: _roster_with(extra=1),
    "cells-mismatch": lambda: _roster_with(cells=len(AIHUB_IDS) + 1),
    "cells-bool": lambda: _roster_with(cell_ids=AIHUB_IDS[:1], cells=True),
    "cohorts-missing": lambda: _roster_with(cohorts=[]),
    "cohorts-extra": lambda: _roster_with(cohorts=["tcga_luad", "tcga_hnsc"]),
    "whole-grid": lambda: _roster_with(
        cell_ids=_active_ids(), cells=len(_active_ids()),
        cohorts=sorted(ACTIVE_ROSTER["cohorts"]),
    ),
    "unreadable": lambda: "{not json",
}


@pytest.mark.parametrize("case", sorted(BAD_ROSTERS))
def test_a_malformed_rehearsal_roster_is_refused_before_any_freeze(tmp_path, case):
    runtime_root = _materialize_rehearsal_set(tmp_path, roster=BAD_ROSTERS[case]())

    with pytest.raises(CampaignStageError, match="rehearsal roster"):
        freeze_campaign_selections(runtime_root, MANIFEST)
    assert not (runtime_root / "selection_freeze.json").exists()


def test_a_rehearsal_set_refuses_a_cell_directory_its_roster_does_not_name(tmp_path):
    extra = next(cell_id for cell_id in _active_ids() if cell_id not in AIHUB_IDS)
    runtime_root = _materialize_rehearsal_set(tmp_path, cell_ids=AIHUB_IDS + [extra])

    with pytest.raises(CampaignStageError, match="runtime roster differs") as exc_info:
        freeze_campaign_selections(runtime_root, MANIFEST)
    assert "unexpected=" in str(exc_info.value) and extra in str(exc_info.value)


def test_a_rehearsal_set_refuses_a_missing_roster_cell(tmp_path):
    runtime_root = _materialize_rehearsal_set(tmp_path, cell_ids=AIHUB_IDS[:-1])

    with pytest.raises(CampaignStageError, match="runtime roster differs") as exc_info:
        freeze_campaign_selections(runtime_root, MANIFEST)
    assert "missing=" in str(exc_info.value) and AIHUB_IDS[-1] in str(exc_info.value)


@pytest.mark.parametrize("name", ["runtime", REHEARSAL_SET])
def test_a_set_without_a_roster_file_is_the_whole_publication_grid(tmp_path, name):
    runtime_root = _materialize_rehearsal_set(tmp_path, name=name)
    (tmp_path / f"{name}.roster.json").unlink()

    with pytest.raises(CampaignStageError, match="runtime roster differs") as exc_info:
        freeze_campaign_selections(runtime_root, MANIFEST)
    assert "missing=" in str(exc_info.value)


def test_the_publication_set_cannot_carry_a_rehearsal_roster(tmp_path):
    runtime_root = _materialize_rehearsal_set(tmp_path, name="runtime", cell_ids=[])

    with pytest.raises(CampaignStageError, match="rehearsal roster"):
        freeze_campaign_selections(runtime_root, MANIFEST)


def test_a_rehearsal_freeze_never_certifies_a_publication_cell(tmp_path):
    rehearsal_root = _materialize_rehearsal_set(tmp_path)
    freeze_campaign_selections(rehearsal_root, MANIFEST)
    publication_root = _materialize_rehearsal_set(
        tmp_path / "grid", name="runtime", cell_ids=AIHUB_IDS[:1],
    )
    (tmp_path / "grid" / "runtime.roster.json").unlink()
    (publication_root / "selection_freeze.json").write_bytes(
        (rehearsal_root / "selection_freeze.json").read_bytes()
    )
    shared = publication_root / AIHUB_IDS[0]

    with pytest.raises(CampaignStageError, match="roster mismatch"):
        freeze_campaign_selections(publication_root, MANIFEST)
    with pytest.raises(CampaignStageError, match="freeze roster mismatch"):
        certify_campaign(publication_root, MANIFEST)
    with pytest.raises(CampaignStageError, match="locked manifest"):
        certify_winner(shared)
    assert _certification_untouched(shared)
    assert not (publication_root / "campaign_certification.json").exists()


def test_a_roster_edited_after_the_freeze_certifies_nothing(tmp_path):
    runtime_root = _materialize_rehearsal_set(tmp_path)
    freeze_campaign_selections(runtime_root, MANIFEST)
    dropped = AIHUB_IDS[-1]
    (runtime_root / dropped).rename(tmp_path / "parked-cell")
    (tmp_path / f"{REHEARSAL_SET}.roster.json").write_text(json.dumps(_roster_with(
        cell_ids=AIHUB_IDS[:-1], cells=len(AIHUB_IDS) - 1,
    )))

    with pytest.raises(CampaignStageError, match="freeze roster mismatch"):
        certify_campaign(runtime_root, MANIFEST)
    assert all(_certification_untouched(runtime_root / cell_id) for cell_id in AIHUB_IDS[:-1])
    assert not (runtime_root / "campaign_certification.json").exists()


def test_a_freeze_whose_count_disagrees_with_its_cells_is_invalid(tmp_path):
    runtime_root = _materialize_rehearsal_set(tmp_path)
    artifact = freeze_campaign_selections(runtime_root, MANIFEST)
    artifact["cell_count"] = artifact["cell_count"] + 1
    artifact["freeze_sha256"] = content_sha256({
        key: value for key, value in artifact.items() if key != "freeze_sha256"
    })

    with pytest.raises(CampaignStageError, match="integrity mismatch"):
        validate_selection_freeze_artifact(artifact)


def _load_manifest_script():
    path = REPO_ROOT / "benchmarks/scripts/campaign_manifest.py"
    spec = importlib.util.spec_from_file_location("campaign_manifest_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("action", ["freeze-selections", "certify-all", "report"])
def test_the_set_commands_refuse_a_set_that_was_never_materialized(action, capsys):
    """Without the guard, the set lock would create the missing directory,
    and the launchers read an existing set directory as materialized."""
    missing = CAMPAIGN_DIR / f"runtime-unmaterialized-{uuid.uuid4().hex}"
    try:
        with pytest.raises(SystemExit) as exc:
            _load_manifest_script().main(
                [action, "--output-root", str(missing.relative_to(REPO_ROOT))]
            )
        assert exc.value.code == 2
        assert "is not a materialized set" in capsys.readouterr().err
        assert not missing.exists()
    finally:
        if missing.exists():
            shutil.rmtree(missing)
