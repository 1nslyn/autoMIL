"""Certification reads the winner and the baseline on three partitions: the
discovery partition and the two re-score partitions, 15 fold models each."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from autobench.campaign import (
    CERTIFICATION_PARTITIONS,
    CERTIFICATION_SOURCES,
    RESCORE_STRATEGIES,
    SPLIT_STRATEGY,
    content_sha256,
)
from autobench.campaign_stages import (
    CampaignStageError,
    _commit_state,
    _validate_source_fold_anchors_artifact,
    certify_winner,
    load_stage_state,
    validate_certification_bundle_artifact,
)

from test_campaign_stages import (  # noqa: F401  (staged_cell is a fixture)
    CLEAR_WINNER_LIFTS,
    TRAILING_POOL_LIFTS,
    _held_out_for,
    _rescored,
    _selected,
    _write_global_selection_freeze,
    staged_cell,
)

#: Added to a candidate's discovery-fold held-out value on each re-score
#: partition, so the three partitions disagree and a mixed-up pairing shows.
PARTITION_SHIFT = {
    SPLIT_STRATEGY: {"baseline": 0.0, "winner": 0.0},
    "devcv_r1": {"baseline": 0.02, "winner": 0.10},
    "devcv_r2": {"baseline": 0.06, "winner": -0.04},
}
#: winner - baseline, per partition: 0.70 - 0.50, then the shifts' difference.
EXPECTED_PARTITION_LIFT = {
    SPLIT_STRATEGY: 0.20,
    "devcv_r1": 0.28,
    "devcv_r2": 0.10,
}


def _shifted_held_out(strategy, candidate_id, fold):
    role = "baseline" if candidate_id == "baseline" else "winner"
    base = _held_out_for(candidate_id, fold)
    return {
        key: value + PARTITION_SHIFT[strategy][role] for key, value in base.items()
    }


def _certified_searched(staged_cell):
    cell_root = staged_cell[0]
    _selected(staged_cell, CLEAR_WINNER_LIFTS)
    _write_global_selection_freeze(cell_root, held_out=_shifted_held_out)
    state = certify_winner(cell_root)
    return cell_root, json.loads((cell_root / state["certification"]["bundle"]).read_text())


def test_the_bundle_pairs_fifteen_winner_folds_with_fifteen_baseline_folds(
    staged_cell,
):
    cell_root, bundle = _certified_searched(staged_cell)

    expected = [list(source) for source in CERTIFICATION_SOURCES]
    for field in ("held_out_folds", "baseline_held_out_folds", "paired_fold_deltas"):
        assert [
            [row["partition"], row["fold_index"]] for row in bundle[field]
        ] == expected
    assert validate_certification_bundle_artifact(bundle) == bundle
    assert len(bundle["source_fold_sha256"]) == 15
    assert len(bundle["baseline_source_fold_sha256"]) == 15
    assert set(bundle["source_fold_sha256"]) == {
        f"{partition}/fold_{fold}_result.json"
        for partition, fold in CERTIFICATION_SOURCES
    }

    winner_auc = [row["held_out"]["test_auc"] for row in bundle["held_out_folds"]]
    baseline_auc = [
        row["held_out"]["test_auc"] for row in bundle["baseline_held_out_folds"]
    ]
    assert bundle["held_out"]["test_auc"] == pytest.approx(math.fsum(winner_auc) / 15)
    assert bundle["baseline_held_out"]["test_auc"] == pytest.approx(
        math.fsum(baseline_auc) / 15
    )
    deltas = [row["held_out_delta"]["test_auc"] for row in bundle["paired_fold_deltas"]]
    assert deltas == pytest.approx([w - b for w, b in zip(winner_auc, baseline_auc)])
    assert bundle["held_out_lift"]["test_auc"] == pytest.approx(
        math.fsum(deltas) / 15
    )
    assert bundle["held_out_lift"]["test_auc"] == pytest.approx(
        sum(EXPECTED_PARTITION_LIFT.values()) / 3
    )


def test_the_bundle_reports_the_lift_of_each_partition(staged_cell):
    cell_root, bundle = _certified_searched(staged_cell)

    by_partition = bundle["held_out_lift_by_partition"]
    assert list(by_partition) == list(CERTIFICATION_PARTITIONS)
    for partition, lift in EXPECTED_PARTITION_LIFT.items():
        assert by_partition[partition]["test_auc"] == pytest.approx(lift)
        assert by_partition[partition]["test_bacc"] == pytest.approx(lift)


def test_the_bundle_carries_the_validation_lift_the_rescore_recorded(staged_cell):
    cell_root = staged_cell[0]
    _selected(staged_cell, CLEAR_WINNER_LIFTS)

    def validation(strategy, candidate_id, fold):
        return 0.60 + fold / 100 + (0.0 if candidate_id == "baseline" else 0.03)

    _write_global_selection_freeze(cell_root, validation=validation)
    recorded = load_stage_state(cell_root)["rescore"]["validation_lift"]
    assert recorded == pytest.approx(0.03)

    state = certify_winner(cell_root)

    bundle = json.loads((cell_root / state["certification"]["bundle"]).read_text())
    assert bundle["rescore_validation_lift"] == recorded


def test_a_baseline_winner_certifies_on_the_baselines_own_rescore_folds(
    staged_cell,
):
    cell_root = staged_cell[0]
    _selected(staged_cell, TRAILING_POOL_LIFTS)
    _write_global_selection_freeze(cell_root, held_out=_shifted_held_out)

    state = certify_winner(cell_root)

    bundle = json.loads((cell_root / state["certification"]["bundle"]).read_text())
    assert bundle["winner"]["kind"] == "baseline"
    assert bundle["held_out_folds"] == bundle["baseline_held_out_folds"]
    assert bundle["source_fold_sha256"] == bundle["baseline_source_fold_sha256"]
    assert bundle["rescore_validation_lift"] == 0.0
    assert bundle["held_out_lift"]["test_auc"] == pytest.approx(0.0)
    assert all(
        lift["test_auc"] == pytest.approx(0.0)
        for lift in bundle["held_out_lift_by_partition"].values()
    )
    shifted = [row["held_out"]["test_auc"] for row in bundle["held_out_folds"]]
    assert shifted[5] == pytest.approx(0.50 + 0.02)   # devcv_r1, fold 0
    assert shifted[10] == pytest.approx(0.50 + 0.06)  # devcv_r2, fold 0


@pytest.mark.parametrize("strategy", RESCORE_STRATEGIES)
def test_a_missing_rescore_sealed_fold_refuses_the_selection_freeze(
    staged_cell, strategy,
):
    cell_root = staged_cell[0]
    _selected(staged_cell, CLEAR_WINNER_LIFTS)
    state = _rescored(cell_root)
    record = state["rescore"]["strategies"][strategy]["node_0007"]
    (cell_root / record["archive"] / "certify/fold_2_result.json").unlink()

    with pytest.raises(CampaignStageError, match="re-score sealed artifact changed"):
        _write_global_selection_freeze(cell_root)


@pytest.mark.parametrize("candidate", ["baseline", "node_0007"])
def test_a_tampered_rescore_sealed_fold_refuses_certification(
    staged_cell, candidate,
):
    cell_root = staged_cell[0]
    _selected(staged_cell, CLEAR_WINNER_LIFTS)
    _write_global_selection_freeze(cell_root)
    record = load_stage_state(cell_root)["rescore"]["strategies"]["devcv_r1"][candidate]
    (cell_root / record["archive"] / "certify/fold_4_result.json").write_text(
        json.dumps({"fold_index": 4, "held_out": {"test_auc": 0.99, "test_bacc": 0.99}})
    )

    with pytest.raises(CampaignStageError, match="re-score sealed artifact changed"):
        certify_winner(cell_root)
    assert not (cell_root / "certification").exists()


def test_certification_refuses_a_winner_that_is_only_frozen(staged_cell):
    cell_root = staged_cell[0]
    _selected(staged_cell, CLEAR_WINNER_LIFTS)
    rescored = _rescored(cell_root)
    # A state that carries the re-score evidence but never reached the phase.
    _commit_state(cell_root, {**rescored, "phase": "winner-frozen"})
    _write_global_selection_freeze(cell_root)

    with pytest.raises(CampaignStageError, match="winner-rescored"):
        certify_winner(cell_root)
    assert not (cell_root / "certification").exists()


def _rehash(bundle):
    bundle["bundle_sha256"] = content_sha256({
        key: value for key, value in bundle.items() if key != "bundle_sha256"
    })
    return bundle


def _forged(bundle, mutate):
    forged = json.loads(json.dumps(bundle))
    mutate(forged)
    return _rehash(forged)


def _drop_a_fold(bundle):
    bundle["held_out_folds"].pop()


def _relabel_a_fold(bundle):
    bundle["held_out_folds"][7]["partition"] = "devcv_r2"


def _mislabel_a_delta(bundle):
    bundle["paired_fold_deltas"][6]["partition"] = "devcv_r2"


def _skew_a_partition_lift(bundle):
    bundle["held_out_lift_by_partition"]["devcv_r1"]["test_auc"] += 0.01


def _drop_a_partition_lift(bundle):
    del bundle["held_out_lift_by_partition"]["devcv_r2"]


def _forge_a_delta(bundle):
    bundle["paired_fold_deltas"][12]["held_out_delta"]["test_auc"] += 0.01


def _legacy_schema(bundle):
    bundle["schema_version"] = 3


def _non_finite_validation_lift(bundle):
    bundle["rescore_validation_lift"] = float("nan")


def _drop_the_validation_lift(bundle):
    del bundle["rescore_validation_lift"]


def _drop_an_anchor_hash(bundle):
    bundle["source_fold_sha256"].pop("devcv_r1/fold_0_result.json")


@pytest.mark.parametrize("mutation", [
    _drop_a_fold, _relabel_a_fold, _mislabel_a_delta, _skew_a_partition_lift,
    _drop_a_partition_lift, _forge_a_delta, _legacy_schema,
    _non_finite_validation_lift, _drop_the_validation_lift, _drop_an_anchor_hash,
], ids=lambda mutation: mutation.__name__.strip("_"))
def test_the_bundle_validator_refuses_a_forged_rehashed_bundle(
    staged_cell, mutation,
):
    _, bundle = _certified_searched(staged_cell)

    with pytest.raises(CampaignStageError):
        validate_certification_bundle_artifact(_forged(bundle, mutation))


def test_a_baseline_bundle_cannot_claim_a_validation_lift(staged_cell):
    cell_root = staged_cell[0]
    _selected(staged_cell, TRAILING_POOL_LIFTS)
    _write_global_selection_freeze(cell_root)
    state = certify_winner(cell_root)
    bundle = json.loads((cell_root / state["certification"]["bundle"]).read_text())

    def claim(forged):
        forged["rescore_validation_lift"] = 0.01

    with pytest.raises(CampaignStageError, match="rescore_validation_lift"):
        validate_certification_bundle_artifact(_forged(bundle, claim))


def _anchors():
    return {
        f"{partition}/fold_{fold}_result.json": {
            "path": f"cell/{partition}/fold_{fold}_result.json",
            "sha256": "a" * 64,
        }
        for partition, fold in CERTIFICATION_SOURCES
    }


def test_source_anchors_are_keyed_by_partition_and_fold():
    anchors = _anchors()
    assert _validate_source_fold_anchors_artifact(anchors, label="winner") == anchors


@pytest.mark.parametrize("mutate", [
    lambda anchors: anchors.pop("devcv_r2/fold_4_result.json"),
    lambda anchors: {
        key.split("/", 1)[1]: record for key, record in anchors.items()
        if key.startswith(f"{SPLIT_STRATEGY}/")
    },
    lambda anchors: anchors["devcv_r1/fold_1_result.json"].update(
        path="cell/devcv_r1/fold_2_result.json"
    ),
    lambda anchors: anchors["devcv_r1/fold_1_result.json"].update(
        path="cell/devcv_r1/fold_1_result.json/../fold_1_result.json"
    ),
], ids=["missing-source", "legacy-five-fold-keys", "path-of-another-fold", "dotdot"])
def test_malformed_source_anchors_are_refused(mutate):
    anchors = _anchors()
    replacement = mutate(anchors)

    with pytest.raises(CampaignStageError, match="source-fold"):
        _validate_source_fold_anchors_artifact(
            replacement if isinstance(replacement, dict) else anchors,
            label="winner",
        )
