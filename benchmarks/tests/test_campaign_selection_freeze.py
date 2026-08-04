"""Campaign-wide validation freeze gates every held-out certification."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import autobench.campaign_stages as campaign_stages
from autobench.campaign import (
    AGENT_PROTOCOL_FILE,
    CAMPAIGN_ID,
    content_sha256,
    file_sha256,
    load_manifest,
)
from autobench.campaign_stages import (
    CAMPAIGN_CELL_COUNT,
    CampaignStageError,
    certify_campaign,
    certify_winner,
    freeze_campaign_selections,
    initialize_stage_state,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "benchmarks/campaigns/preprint_130/manifest.json"
BASE_COMMIT = "d" * 40
AGENT_PROTOCOL = {
    "schema_version": 1,
    "campaign_id": CAMPAIGN_ID,
    "purpose": "publication",
    "provider": "test-provider",
    "runtime": "test-runtime",
    "runtime_version": "test-runtime-1",
    "model": "test-model",
    "model_version": "test-model-1",
    "proposal_policy_sha256": "a" * 64,
    "toolset_sha256": "b" * 64,
    "max_sessions_per_cell": 1,
}


def _freeze_ready_state(runtime_root: Path, cell: dict, manifest_hash: str) -> Path:
    cell_root = runtime_root / cell["cell_id"]
    adir = cell_root / "automil"
    adir.mkdir(parents=True)
    (adir / "config.yaml").write_text(json.dumps({
        "campaign": {
            "agent_protocol_sha256": content_sha256(AGENT_PROTOCOL),
        },
    }))
    state = initialize_stage_state(
        cell_root,
        cell=cell,
        manifest_sha256=manifest_hash,
        base_commit=BASE_COMMIT,
    )
    candidate_sha256 = content_sha256({"cell_id": cell["cell_id"]})
    state["phase"] = "winner-frozen"
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
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "cell_id": cell["cell_id"],
        "agent_protocol_sha256": content_sha256(AGENT_PROTOCOL),
        "sessions": [{
            "session_id": f"session-{cell['cell_id']}",
            "started_at": "2026-08-04T00:00:00+00:00",
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
        }],
    }
    session["attestation_sha256"] = content_sha256(session)
    (cell_root / "agent_session.json").write_text(json.dumps(session))
    return cell_root


def _materialize_frozen_roster(runtime_root: Path) -> list[Path]:
    runtime_root.mkdir(parents=True)
    (runtime_root / AGENT_PROTOCOL_FILE).write_text(json.dumps(AGENT_PROTOCOL))
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


def test_global_freeze_binds_every_cell_winner_and_blocks_later_drift(tmp_path):
    runtime_root = tmp_path / "runtime"
    roots = _materialize_frozen_roster(runtime_root)
    artifact = freeze_campaign_selections(runtime_root, MANIFEST)
    first = roots[0]

    with pytest.raises(CampaignStageError, match="native baseline is not registered"):
        certify_winner(first)

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
    visited: list[str] = []

    def fake_certify(cell_root: Path) -> dict:
        state = json.loads((cell_root / "campaign_state.json").read_text())
        visited.append(state["cell_id"])
        bundle = {
            "schema_version": 2,
            "campaign_id": state["campaign_id"],
            "cell_id": state["cell_id"],
            "selection_freeze_sha256": freeze["freeze_sha256"],
        }
        bundle["bundle_sha256"] = content_sha256(bundle)
        bundle_path = cell_root / "certification/certify.json"
        bundle_path.parent.mkdir()
        bundle_path.write_text(json.dumps(bundle))
        state["phase"] = "certified"
        state["certification"] = {
            "bundle": "certification/certify.json",
            "bundle_sha256": bundle["bundle_sha256"],
        }
        return state

    monkeypatch.setattr(campaign_stages, "certify_winner", fake_certify)
    index = certify_campaign(runtime_root, MANIFEST)

    assert len(visited) == len(set(visited)) == CAMPAIGN_CELL_COUNT
    assert set(visited) == {root.name for root in roots}
    assert index["cell_count"] == 130
    assert len(index["cells"]) == 130
    assert (runtime_root / "campaign_certification.json").is_file()
