"""H-8: a run missing any declared fold must not masquerade as completed.

``compute_confidence_intervals`` silently drops NaN folds, so
``summary_to_result_json`` records support and quarantines incomplete stage
subsets as ``partial`` before they can enter autoMIL keep/discard.
"""
from __future__ import annotations

import importlib.util
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


def _cls_summary(fold_aucs):
    return {
        "test": {"auc_roc": {"mean": 0.70}, "balanced_accuracy": {"mean": 0.60}},
        "val": {"auc_roc": {"mean": 0.70}, "balanced_accuracy": {"mean": 0.60}},
        "per_fold_val": [{"auc_roc": a, "balanced_accuracy": 0.6} for a in fold_aucs],
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
    # qwk is recorded evidence, never a vote: composite is still val_auc.
    assert r["composite"] == pytest.approx((0.70 + 0.72 + 0.68) / 3)


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
    """Only val_auc votes, but fold VALIDITY spans the recorded set: a fold
    that lost its companion is the fold the campaign validator rejects at
    ingest, so it must quarantine as partial on this side too."""
    m = _load_run_experiment()
    summary = _cls_summary([0.70, 0.72])
    summary["per_fold_val"][1]["balanced_accuracy"] = NAN
    result = m.summary_to_result_json(summary, 10.0)
    assert result["status"] == "partial"
    assert result["n_valid_folds"] == 1
    # Losing the selection metric invalidates the fold just the same.
    summary["per_fold_val"][1]["balanced_accuracy"] = 0.60
    summary["per_fold_val"][1]["auc_roc"] = NAN
    assert m.summary_to_result_json(summary, 10.0)["status"] == "partial"


def test_survival_degenerate_partial():
    m = _load_run_experiment()
    summary = {
        "test": {"c_index": {"mean": 0.60}},
        "val": {"c_index": {"mean": 0.60}},
        "per_fold_val": [{"c_index": 0.60}] + [{"c_index": NAN}] * 4,
        "per_fold_test": [],
        "n_folds": 5,
    }
    r = m.summary_to_result_json(summary, 5.0)
    assert r["status"] == "partial"
    assert r["n_valid_folds"] == 1
    assert "val_c_index" in r["metrics"]


def test_validation_fold_evidence_is_public_and_fold_indexed():
    m = _load_run_experiment()
    summary = _cls_summary([0.70, 0.72, 0.68])
    summary["fold_indices"] = [0, 1, 2]
    result = m.summary_to_result_json(summary, 10.0)

    # val_predictions_sha256 (A4') is part of the entry schema, at ENTRY level
    # (never inside the exact-key-locked `metrics`); None when the summary
    # carries no per-fold hash, as this hand-built one does not.
    assert result["validation_folds"] == [
        {
            "fold_index": 0,
            "metrics": {"val_auc": 0.70, "val_bacc": 0.6},
            "composite": pytest.approx(0.70),
            "val_predictions_sha256": None,
        },
        {
            "fold_index": 1,
            "metrics": {"val_auc": 0.72, "val_bacc": 0.6},
            "composite": pytest.approx(0.72),
            "val_predictions_sha256": None,
        },
        {
            "fold_index": 2,
            "metrics": {"val_auc": 0.68, "val_bacc": 0.6},
            "composite": pytest.approx(0.68),
            "val_predictions_sha256": None,
        },
    ]
    assert all("test" not in str(fold).lower()
               for fold in result["validation_folds"])


def test_invalid_fold_is_visible_but_never_given_a_numeric_composite():
    m = _load_run_experiment()
    result = m.summary_to_result_json(_cls_summary([0.70, NAN]), 10.0)
    assert result["validation_folds"][1]["fold_index"] == 1
    assert result["validation_folds"][1]["composite"] is None


def test_survival_selection_uses_fold_mean_not_pooled_stage_value():
    m = _load_run_experiment()
    summary = {
        "test": {"c_index": {"mean": 0.60}},
        "val": {"c_index": {"mean": 0.60}},
        "val_pooled": {"c_index": 0.91},
        "per_fold_val": [{"c_index": 0.55}, {"c_index": 0.65}],
        "per_fold_test": [],
        "n_folds": 5,
        "fold_indices": [3, 4],
    }
    result = m.summary_to_result_json(summary, 5.0)
    assert result["composite"] == pytest.approx(0.60)
    assert [fold["composite"] for fold in result["validation_folds"]] == [
        pytest.approx(0.55), pytest.approx(0.65),
    ]
