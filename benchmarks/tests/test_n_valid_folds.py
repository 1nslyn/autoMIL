"""H-8 (audit 2026-07-23): a run with <2 valid folds must NOT masquerade as a
complete K-fold "completed" result. compute_confidence_intervals silently drops
NaN folds and reports a zero-variance point estimate when <2 survive, so
summary_to_result_json now records n_valid_folds / n_folds and quarantines the
result (status=partial → kept out of autoMIL keep/discard) when it is degenerate.
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


def test_two_valid_folds_still_completed():
    m = _load_run_experiment()
    r = m.summary_to_result_json(_cls_summary([0.70, 0.72, NAN, NAN, NAN]), 10.0)
    assert r["status"] == "completed"
    assert r["n_valid_folds"] == 2


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
