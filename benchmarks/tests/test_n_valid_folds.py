"""H-8: a run missing any declared fold must not masquerade as completed.

``compute_confidence_intervals`` silently drops NaN folds, so
``summary_to_result_json`` records support and quarantines incomplete stage
subsets as ``partial`` before they can enter autoMIL keep/discard.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest

NAN = float("nan")


def _load_run_experiment() -> ModuleType:
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    script_path = scripts_dir / "run_experiment.py"
    if not script_path.exists():
        pytest.skip(f"run_experiment.py not found at {script_path}")
    mod_name = "run_experiment_h8"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, script_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except SystemExit:
        pass
    return mod


def _cls_summary(fold_aucs, fold_smooths=None):
    """A classification summary. Each fold's smoothed AUC, the selection metric,
    sits 0.04 below the restored model's own auc_roc unless ``fold_smooths``
    sets it, so a test can tell which of the two a number came from."""
    if fold_smooths is None:
        fold_smooths = [a - 0.04 for a in fold_aucs]
    finite = [value for value in fold_smooths if math.isfinite(value)]
    return {
        "test": {"auc_roc": {"mean": 0.70}, "balanced_accuracy": {"mean": 0.60}},
        "val": {
            "auc_roc_smooth": {"mean": sum(finite) / len(finite) if finite else NAN},
            "auc_roc": {"mean": 0.70},
            "balanced_accuracy": {"mean": 0.60},
        },
        "per_fold_val": [
            {"auc_roc_smooth": s, "auc_roc": a, "balanced_accuracy": 0.6}
            for a, s in zip(fold_aucs, fold_smooths)
        ],
        "per_fold_test": [],
        "n_folds": len(fold_aucs),
        "fold_indices": list(range(len(fold_aucs))),
    }


def test_full_folds_reported_completed():
    m = _load_run_experiment()
    r = m.summary_to_result_json(_cls_summary([0.70, 0.72, 0.68, 0.71, 0.69]), 10.0)
    assert r["status"] == "completed"
    assert r["n_valid_folds"] == 5
    assert r["n_folds"] == 5


def test_single_valid_fold_quarantined_partial():
    m = _load_run_experiment()
    r = m.summary_to_result_json(_cls_summary([0.70, NAN, NAN, NAN, NAN]), 10.0)
    assert r["status"] == "partial"
    assert r["n_valid_folds"] == 1
    assert r["n_folds"] == 5


def test_two_valid_folds_out_of_five_are_partial():
    m = _load_run_experiment()
    r = m.summary_to_result_json(_cls_summary([0.70, 0.72, NAN, NAN, NAN]), 10.0)
    assert r["status"] == "partial"
    assert r["n_valid_folds"] == 2


def test_promotion_is_completed_only_with_both_declared_folds():
    m = _load_run_experiment()
    summary = _cls_summary([0.70, 0.72])
    summary["n_folds"] = 5
    summary["fold_indices"] = [3, 4]
    assert m.summary_to_result_json(summary, 10.0)["status"] == "completed"


def test_discovery_with_two_of_three_valid_folds_is_partial():
    m = _load_run_experiment()
    summary = _cls_summary([0.70, 0.72, NAN])
    summary["n_folds"] = 5
    summary["fold_indices"] = [0, 1, 2]
    assert m.summary_to_result_json(summary, 10.0)["status"] == "partial"


def test_ordinal_held_out_carries_clamp_then_mean_test_qwk():
    """Ordinal cells report on test_qwk (primary_by_task_family), so the
    sealed aggregate must exist and equal the mean of PER-FOLD clamped
    values — mean(max(0, qwk)), never max(0, mean(qwk))."""
    m = _load_run_experiment()
    summary = _cls_summary([0.70, 0.72, 0.68])
    for fm, qwk in zip(summary["per_fold_val"], (0.30, 0.10, 0.20)):
        fm["qwk"] = qwk
    summary["per_fold_test"] = [
        {"auc_roc": 0.70, "balanced_accuracy": 0.60, "qwk": qwk}
        for qwk in (0.40, -0.20, 0.20)
    ]
    r = m.summary_to_result_json(summary, 10.0, ordinal=True)
    # mean(max(0, .)) = (0.40 + 0.0 + 0.20) / 3 = 0.20; max(0, mean) would
    # give 0.1333 — the wrong function.
    assert r["held_out"]["test_qwk"] == pytest.approx(0.20)
    assert r["metrics"]["val_qwk"] == pytest.approx(0.20)
    # qwk is recorded evidence, never a vote: primary_value is still the
    # smoothed AUC's fold mean (0.66), not the restored model's val_auc (0.70).
    assert r["primary_value"] == pytest.approx((0.66 + 0.68 + 0.64) / 3)


def test_non_ordinal_summary_never_carries_test_qwk():
    m = _load_run_experiment()
    summary = _cls_summary([0.70, 0.72, 0.68])
    summary["per_fold_test"] = [
        {"auc_roc": 0.70, "balanced_accuracy": 0.60, "qwk": 0.5}
    ] * 3
    r = m.summary_to_result_json(summary, 10.0)
    assert "test_qwk" not in r["held_out"]
    assert "val_qwk" not in r["metrics"]


def test_classification_fold_requires_full_recorded_evidence():
    """Only val_auc_smooth votes, but fold VALIDITY spans the recorded set: a
    fold that lost its companion is the fold the campaign validator rejects at
    ingest, so it must quarantine as partial on this side too."""
    m = _load_run_experiment()
    summary = _cls_summary([0.70, 0.72])
    summary["per_fold_val"][1]["balanced_accuracy"] = NAN
    result = m.summary_to_result_json(summary, 10.0)
    assert result["status"] == "partial"
    assert result["n_valid_folds"] == 1
    # Losing the selection metric invalidates the fold just the same.
    summary["per_fold_val"][1]["balanced_accuracy"] = 0.60
    summary["per_fold_val"][1]["auc_roc_smooth"] = NAN
    assert m.summary_to_result_json(summary, 10.0)["status"] == "partial"
    # So does losing the restored model's own AUC, which is recorded beside it.
    summary["per_fold_val"][1]["auc_roc_smooth"] = 0.68
    summary["per_fold_val"][1]["auc_roc"] = NAN
    assert m.summary_to_result_json(summary, 10.0)["status"] == "partial"


def test_classification_primary_value_is_the_smoothed_fold_mean_not_val_auc():
    """Selection scores each fold on its validation AUC smoothed around the
    restored epoch, and the run on the mean of those. The restored model's own
    val_auc is recorded beside it and does not vote."""
    m = _load_run_experiment()
    summary = _cls_summary([0.70, 0.72, 0.68], fold_smooths=[0.60, 0.66, 0.63])
    r = m.summary_to_result_json(summary, 10.0)
    assert r["status"] == "completed"
    assert r["metrics"]["val_auc_smooth"] == pytest.approx(0.63)
    assert r["metrics"]["val_auc"] == pytest.approx(0.70)
    assert r["primary_value"] == pytest.approx(0.63)
    assert r["primary_value"] != pytest.approx(r["metrics"]["val_auc"], abs=1e-3)
    assert [fold["primary_value"] for fold in r["validation_folds"]] == [
        pytest.approx(0.60), pytest.approx(0.66), pytest.approx(0.63),
    ]


def test_survival_degenerate_partial():
    m = _load_run_experiment()
    summary = {
        "test": {"c_index": {"mean": 0.60}},
        "val": {"c_index_smooth": {"mean": 0.55}, "c_index": {"mean": 0.60}},
        "per_fold_val": (
            [{"c_index_smooth": 0.55, "c_index": 0.60}]
            + [{"c_index_smooth": NAN, "c_index": NAN}] * 4
        ),
        "per_fold_test": [],
        "n_folds": 5,
    }
    r = m.summary_to_result_json(summary, 5.0)
    assert r["status"] == "partial"
    assert r["n_valid_folds"] == 1
    assert set(r["metrics"]) == {"val_c_index_smooth", "val_c_index"}


def test_survival_metrics_carry_both_c_index_keys_and_select_on_the_smoothed_mean():
    m = _load_run_experiment()
    summary = {
        "test": {"c_index": {"mean": 0.62}},
        "val": {"c_index_smooth": {"mean": 0.57}, "c_index": {"mean": 0.64}},
        "per_fold_val": [
            {"c_index_smooth": 0.52, "c_index": 0.60},
            {"c_index_smooth": 0.62, "c_index": 0.68},
        ],
        "per_fold_test": [],
        "n_folds": 5,
        "fold_indices": [3, 4],
    }
    r = m.summary_to_result_json(summary, 5.0)
    assert r["status"] == "completed"
    assert r["metrics"] == {
        "val_c_index_smooth": pytest.approx(0.57), "val_c_index": pytest.approx(0.64),
    }
    # The mean of the fold smoothed values; the restored model's C-index (0.64)
    # is recorded beside it and does not vote.
    assert r["primary_value"] == pytest.approx(0.57)


@pytest.mark.parametrize("lost_in", ["aggregate", "every-fold"])
def test_survival_run_without_an_estimable_raw_c_index_is_partial_with_no_metrics(
    lost_in,
):
    """The recorded set is all-or-nothing on the survival side too. The smoothed
    score votes, but a run that lost the restored model's own C-index broke the
    campaign's evidence contract, so it reports no metrics and the 0.0 sentinel
    rather than half a block."""
    m = _load_run_experiment()
    fold_raw = [NAN, NAN] if lost_in == "every-fold" else [0.60, 0.68]
    summary = {
        "test": {"c_index": {"mean": 0.62}},
        "val": {"c_index_smooth": {"mean": 0.57}, "c_index": {"mean": NAN}},
        "per_fold_val": [
            {"c_index_smooth": smooth, "c_index": raw}
            for smooth, raw in zip((0.52, 0.62), fold_raw)
        ],
        "per_fold_test": [],
        "n_folds": 5,
        "fold_indices": [3, 4],
    }
    r = m.summary_to_result_json(summary, 5.0)
    assert r["status"] == "partial"
    assert r["metrics"] == {}
    assert r["primary_value"] == 0.0
    assert "val_c_index" in r["error"]


def test_validation_fold_evidence_is_public_and_fold_indexed():
    m = _load_run_experiment()
    summary = _cls_summary([0.70, 0.72, 0.68], fold_smooths=[0.66, 0.69, 0.63])
    summary["fold_indices"] = [0, 1, 2]
    result = m.summary_to_result_json(summary, 10.0)

    # val_predictions_sha256 (A4') is part of the entry schema, at ENTRY level
    # (never inside the exact-key-locked `metrics`); None when the summary
    # carries no per-fold hash, as this hand-built one does not.
    assert result["validation_folds"] == [
        {
            "fold_index": 0,
            "metrics": {"val_auc_smooth": 0.66, "val_auc": 0.70, "val_bacc": 0.6},
            "primary_value": pytest.approx(0.66),
            "val_predictions_sha256": None,
        },
        {
            "fold_index": 1,
            "metrics": {"val_auc_smooth": 0.69, "val_auc": 0.72, "val_bacc": 0.6},
            "primary_value": pytest.approx(0.69),
            "val_predictions_sha256": None,
        },
        {
            "fold_index": 2,
            "metrics": {"val_auc_smooth": 0.63, "val_auc": 0.68, "val_bacc": 0.6},
            "primary_value": pytest.approx(0.63),
            "val_predictions_sha256": None,
        },
    ]
    assert all("test" not in str(fold).lower()
               for fold in result["validation_folds"])


def test_invalid_fold_is_visible_but_never_given_a_numeric_primary_value():
    m = _load_run_experiment()
    result = m.summary_to_result_json(_cls_summary([0.70, NAN]), 10.0)
    assert result["validation_folds"][1]["fold_index"] == 1
    assert result["validation_folds"][1]["primary_value"] is None


def test_survival_selection_uses_fold_mean_not_pooled_stage_value():
    m = _load_run_experiment()
    summary = {
        "test": {"c_index": {"mean": 0.60}},
        "val": {"c_index_smooth": {"mean": 0.60}, "c_index": {"mean": 0.63}},
        "val_pooled": {"c_index": 0.91},
        "per_fold_val": [
            {"c_index_smooth": 0.55, "c_index": 0.58},
            {"c_index_smooth": 0.65, "c_index": 0.68},
        ],
        "per_fold_test": [],
        "n_folds": 5,
        "fold_indices": [3, 4],
    }
    result = m.summary_to_result_json(summary, 5.0)
    assert result["primary_value"] == pytest.approx(0.60)
    assert [fold["primary_value"] for fold in result["validation_folds"]] == [
        pytest.approx(0.55), pytest.approx(0.65),
    ]
