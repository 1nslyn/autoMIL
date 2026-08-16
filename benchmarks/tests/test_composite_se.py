"""CR-4 (audit 2026-07-23): ``summary_to_result_json`` must MEASURE the noise the
Ladder keep-margin is supposed to exceed.

Before CR-4 the composite was reported as a bare point estimate and the keep
margin δ was a guessed constant (0.0 by default). The composite is a mean over K
folds, so its cross-fold standard error is available for free — emit it, and the
margin can be derived rather than guessed.

Two constraints these tests pin:
  * the SE goes at the result TOP LEVEL, never inside ``metrics`` — CR-1b
    recomputes the selection composite from ``metrics`` (the mean of its values
    under the ``mean`` reducer, the named metric under a ``val_*`` selector), so
    an extra key there would corrupt the val-firewall's selection signal;
  * fewer than two finite folds → ``None``, never 0.0 (consistent with H-8's
    ``n_valid_folds`` / ``status=partial`` treatment of the same degeneracy).
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
    mod_name = "run_experiment_cr4"
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


def _cls_summary(fold_aucs, fold_baccs=None):
    fold_baccs = fold_baccs if fold_baccs is not None else [0.60] * len(fold_aucs)
    finite = [a for a in fold_aucs if isinstance(a, float) and math.isfinite(a)]
    mean_auc = sum(finite) / len(finite) if finite else 0.0
    return {
        "test": {"auc_roc": {"mean": 0.70}, "balanced_accuracy": {"mean": 0.60}},
        "val": {"auc_roc": {"mean": mean_auc},
                "balanced_accuracy": {"mean": 0.60}},
        "per_fold_val": [
            {"auc_roc": a, "balanced_accuracy": b}
            for a, b in zip(fold_aucs, fold_baccs)
        ],
        "per_fold_test": [],
        "n_folds": len(fold_aucs),
    }


# --- the measurement --------------------------------------------------------

def test_classification_composite_se_is_the_cross_fold_sem():
    m = _load_run_experiment()
    # per-fold composites = val_auc alone → {0.70,0.72,0.68,0.71,0.69}
    # ddof=1 SD = 0.01581138830..., SE = SD/sqrt(5) = 0.00707106781...
    r = m.summary_to_result_json(_cls_summary([0.70, 0.72, 0.68, 0.71, 0.69]), 10.0)
    assert r["composite_se"] == pytest.approx(0.007071, abs=1e-6)


def test_composite_se_is_top_level_not_inside_metrics():
    """CR-1b: metrics is the selection signal's input; nothing else may live there."""
    m = _load_run_experiment()
    r = m.summary_to_result_json(_cls_summary([0.70, 0.72, 0.68, 0.71, 0.69]), 10.0)
    assert "composite_se" in r
    assert "composite_se" not in r["metrics"]
    assert set(r["metrics"]) == {"val_auc", "val_bacc"}


def test_composite_se_does_not_shift_the_recomputed_composite():
    """The CR-1b recompute over metrics must still reproduce the composite
    under the campaign's declared selector (scoring.formula: val_auc)."""
    from automil.scoring import recompute_composite

    m = _load_run_experiment()
    r = m.summary_to_result_json(_cls_summary([0.70, 0.72, 0.68, 0.71, 0.69]), 10.0)
    assert recompute_composite(r["metrics"], "val_auc") == pytest.approx(
        r["composite"], abs=1e-3)


def test_composite_se_none_below_two_valid_folds():
    """H-8 consistency: not estimable is None, never 0.0."""
    m = _load_run_experiment()
    r = m.summary_to_result_json(_cls_summary([0.70, NAN, NAN, NAN, NAN]), 10.0)
    assert r["status"] == "partial"
    assert r["n_valid_folds"] == 1
    assert r["composite_se"] is None


def test_composite_se_none_when_per_fold_val_absent():
    m = _load_run_experiment()
    summary = {
        "test": {"auc_roc": {"mean": 0.70}, "balanced_accuracy": {"mean": 0.60}},
        "val": {"auc_roc": {"mean": 0.70}, "balanced_accuracy": {"mean": 0.60}},
        "per_fold_test": [],
    }
    r = m.summary_to_result_json(summary, 10.0)
    assert r["composite_se"] is None


def test_a_fold_missing_the_selection_metric_is_dropped_whole():
    """No val_auc on a fold → no composite to measure on that fold."""
    m = _load_run_experiment()
    r = m.summary_to_result_json(_cls_summary([0.70, NAN, 0.68]), 10.0)
    # Folds 0 and 2 carry the selection metric → composites {0.70, 0.68}
    # ddof=1 SD = 0.01414213562, SE = /sqrt(2) = 0.01
    assert r["composite_se"] == pytest.approx(0.01, abs=1e-6)


def test_a_fold_missing_a_companion_is_dropped_whole_too():
    """Fold validity spans the full RECORDED evidence set even though only
    val_auc votes: the campaign validator rejects a companion-lossy fold at
    ingest, so this side must quarantine the same fold — not sail it through
    selection to die silently at discovery freeze."""
    m = _load_run_experiment()
    r = m.summary_to_result_json(
        _cls_summary([0.70, 0.72, 0.68], fold_baccs=[0.60, NAN, 0.60]), 10.0)
    # Folds 0 and 2 carry full evidence → composites {0.70, 0.68}
    # ddof=1 SD = 0.01414213562, SE = /sqrt(2) = 0.01
    assert r["composite_se"] == pytest.approx(0.01, abs=1e-6)
    assert r["validation_folds"][1]["metrics"]["val_bacc"] is None
    assert r["validation_folds"][1]["composite"] is None
    assert r["status"] == "partial"


def test_degenerate_identical_folds_report_zero_se_not_none():
    m = _load_run_experiment()
    r = m.summary_to_result_json(_cls_summary([0.70, 0.70, 0.70]), 10.0)
    assert r["composite_se"] == 0.0


def test_survival_composite_se_is_the_cross_fold_c_index_sem():
    """Survival selection and its SE use the same per-fold evidence."""
    m = _load_run_experiment()
    summary = {
        "test": {"c_index": {"mean": 0.62}},
        "val": {"c_index": {"mean": 0.60}},
        "val_pooled": {"c_index": 0.61},
        "per_fold_val": [{"c_index": c} for c in (0.58, 0.60, 0.62, 0.59, 0.61)],
        "per_fold_test": [],
        "n_folds": 5,
    }
    r = m.summary_to_result_json(summary, 5.0)
    assert r["composite"] == pytest.approx(0.60)      # equal-weight fold mean
    assert r["metrics"]["val_c_index"] == pytest.approx(r["composite"])
    assert r["composite"] == pytest.approx(
        sum(fold["composite"] for fold in r["validation_folds"]) / 5
    )
    # ddof=1 SD over the five per-fold c-indices = 0.01581138830, /sqrt(5)
    assert r["composite_se"] == pytest.approx(0.007071, abs=1e-6)


def test_result_with_composite_se_still_validates_against_the_schema():
    from automil.schemas import validate_result

    m = _load_run_experiment()
    r = m.summary_to_result_json(_cls_summary([0.70, 0.72, 0.68, 0.71, 0.69]), 10.0)
    validate_result(r)   # must not raise
    r_none = m.summary_to_result_json(_cls_summary([0.70, NAN, NAN, NAN, NAN]), 10.0)
    validate_result(r_none)
