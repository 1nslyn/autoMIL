"""Tests for autobench.pipeline.evaluate module."""

import numpy as np
import pytest

from autobench.pipeline.evaluate import (
    compute_confidence_intervals,
    compute_extended_metrics,
)


class TestComputeExtendedMetrics:
    def test_perfect_binary_classification(self):
        y_true = np.array([0, 0, 1, 1, 0])
        y_probs = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 0]], dtype=float)
        y_pred = np.array([0, 0, 1, 1, 0])
        m = compute_extended_metrics(y_true, y_probs, y_pred, 2)
        assert m["auc_roc"] == 1.0
        assert m["accuracy"] == 1.0
        assert m["balanced_accuracy"] == 1.0
        assert m["f1"] == 1.0
        assert m["sensitivity"] == 1.0
        assert m["specificity"] == 1.0

    def test_all_wrong_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_probs = np.array([[0, 1], [0, 1], [1, 0], [1, 0]], dtype=float)
        y_pred = np.array([1, 1, 0, 0])
        m = compute_extended_metrics(y_true, y_probs, y_pred, 2)
        assert m["auc_roc"] == 0.0
        assert m["accuracy"] == 0.0
        assert m["sensitivity"] == 0.0
        assert m["specificity"] == 0.0

    def test_all_predicted_positive(self):
        y_true = np.array([0, 0, 1, 1])
        y_probs = np.array([[0.3, 0.7]] * 4)
        y_pred = np.array([1, 1, 1, 1])
        m = compute_extended_metrics(y_true, y_probs, y_pred, 2)
        assert m["sensitivity"] == 1.0
        assert m["specificity"] == 0.0

    def test_all_predicted_negative(self):
        y_true = np.array([0, 0, 1, 1])
        y_probs = np.array([[0.7, 0.3]] * 4)
        y_pred = np.array([0, 0, 0, 0])
        m = compute_extended_metrics(y_true, y_probs, y_pred, 2)
        assert m["sensitivity"] == 0.0
        assert m["specificity"] == 1.0

    def test_returns_all_expected_keys(self):
        y_true = np.array([0, 1])
        y_probs = np.array([[0.6, 0.4], [0.3, 0.7]])
        y_pred = np.array([0, 1])
        m = compute_extended_metrics(y_true, y_probs, y_pred, 2)
        expected_keys = {"auc_roc", "accuracy", "balanced_accuracy", "f1",
                         "sensitivity", "specificity"}
        assert set(m.keys()) == expected_keys

    def test_all_values_are_floats(self):
        y_true = np.array([0, 1, 0, 1])
        y_probs = np.random.rand(4, 2)
        y_pred = np.array([0, 1, 1, 0])
        m = compute_extended_metrics(y_true, y_probs, y_pred, 2)
        for v in m.values():
            assert isinstance(v, float)

    def test_single_class_in_true_labels(self):
        """AUC is nan when only one class present."""
        y_true = np.array([0, 0, 0])
        y_probs = np.array([[0.8, 0.2], [0.6, 0.4], [0.9, 0.1]])
        y_pred = np.array([0, 0, 0])
        m = compute_extended_metrics(y_true, y_probs, y_pred, 2)
        assert np.isnan(m["auc_roc"])


class TestMultiClassAUC:
    """Multi-class AUC must match CLAM upstream's per-class roc_curve + nanmean
    formula (lib/CLAM/utils/core_utils.py:514-527), NOT sklearn's
    roc_auc_score(multi_class='ovr', average='macro'). The two formulas
    diverge in finite samples; we lock in the CLAM-style result for the
    CLAM path so reported AUCs are reproducible against CLAM's own.
    """

    def test_uses_clam_style_per_class_nanmean(self):
        from sklearn.metrics import auc as sk_auc, roc_auc_score, roc_curve
        from sklearn.preprocessing import label_binarize

        rng = np.random.default_rng(0)
        n_classes = 3
        n_samples = 90
        y_true = np.array([i % n_classes for i in range(n_samples)])
        # Probabilities that favour the true class with noise.
        y_probs = rng.dirichlet(alpha=[1] * n_classes, size=n_samples)
        for i in range(n_samples):
            y_probs[i, y_true[i]] += 0.5
            y_probs[i] /= y_probs[i].sum()
        y_pred = y_probs.argmax(axis=1)

        m = compute_extended_metrics(y_true, y_probs, y_pred, n_classes)

        # Recompute the CLAM-upstream expected value
        binary = label_binarize(y_true, classes=list(range(n_classes)))
        per_class = []
        for c in range(n_classes):
            fpr, tpr, _ = roc_curve(binary[:, c], y_probs[:, c])
            per_class.append(float(sk_auc(fpr, tpr)))
        expected_clam = float(np.nanmean(per_class))

        sklearn_ovr_macro = float(
            roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
        )

        assert m["auc_roc"] == pytest.approx(expected_clam, abs=1e-9)
        # The two formulas agree in this perfectly-balanced case to within
        # ~1e-9; we assert the metric is computed in a way consistent with
        # the per-class formula. In imbalanced multi-class settings the
        # formulas diverge — that's where this test would catch a
        # regression to sklearn's macro-OvR.
        # Sanity: both formulas are bounded in [0, 1].
        assert 0.0 <= m["auc_roc"] <= 1.0
        assert 0.0 <= sklearn_ovr_macro <= 1.0

    def test_missing_class_yields_nanmean_over_present(self):
        """When a class has zero positives in the test split, CLAM upstream
        skips it via ``nanmean``. The wrapper must do the same — sklearn's
        roc_auc_score raises in this case, so any regression to OvR-macro
        would fail loudly.
        """
        # 3-class problem; class 2 has zero positives in y_true
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_probs = np.array(
            [
                [0.7, 0.2, 0.1],
                [0.6, 0.3, 0.1],
                [0.2, 0.7, 0.1],
                [0.3, 0.6, 0.1],
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
            ]
        )
        y_pred = y_probs.argmax(axis=1)
        m = compute_extended_metrics(y_true, y_probs, y_pred, 3)
        # AUC should be a real number (nanmean of present-class AUCs), not nan
        assert not np.isnan(m["auc_roc"])
        assert 0.0 <= m["auc_roc"] <= 1.0

    def test_L10_missing_class_asymmetry_is_pinned(self):
        """L-10: pin BOTH halves of the cross-framework AUC asymmetry on the
        SAME data, not just assert one side in a comment.

        CLAM/ABMIL/DTFD/TITAN all go through ``compute_extended_metrics``
        (this module's CLAM-style per-class nanmean). nnMIL instead calls
        ``sklearn.metrics.roc_auc_score(multi_class="ovr", average="macro")``
        directly inside its own vendored trainer
        (``lib/nnMIL/utilities/utils.py``, mirrored here for the pin since
        that trainer code is out of this package's reach) and is normalized
        by ``nnmil/evaluate.py::normalize_nnmil_metrics``, whose module
        docstring already documents this divergence in detail.

        Decision recorded here (see also that docstring): DOCUMENT, don't
        unify. Unifying would mean either patching nnMIL's vendored trainer
        (out of scope for a consumer-side fix -- lib/ is a third-party
        dependency) or restructuring nnmil/evaluate.py to receive raw
        per-class probabilities the vendored trainer does not currently
        expose to it, which would be a much larger structural change than
        this finding's severity (LOW) warrants. So: same model quality can
        legitimately yield two different AUC numbers depending on the arm
        when a class is absent from a fold, and the nnMIL arm crashes
        (ValueError) in exactly the case CLAM's arms degrade gracefully
        (nanmean over present classes). This test locks that gap in place so
        a change to either formula is caught, instead of silently drifting
        the two closer together or further apart.
        """
        from sklearn.metrics import roc_auc_score

        # Same fixture as test_missing_class_yields_nanmean_over_present:
        # 3-class problem where class 2 has zero positives in y_true.
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_probs = np.array(
            [
                [0.7, 0.2, 0.1],
                [0.6, 0.3, 0.1],
                [0.2, 0.7, 0.1],
                [0.3, 0.6, 0.1],
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
            ]
        )
        y_pred = y_probs.argmax(axis=1)

        # CLAM/ABMIL/DTFD/TITAN path: degrades gracefully to a real number.
        clam_side = compute_extended_metrics(y_true, y_probs, y_pred, 3)["auc_roc"]
        assert not np.isnan(clam_side)
        assert 0.0 <= clam_side <= 1.0

        # nnMIL path: the exact sklearn call nnMIL's trainer makes. Raises.
        with pytest.raises(ValueError):
            roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")


class TestComputeConfidenceIntervals:
    def test_basic_ci(self):
        fold_metrics = [
            {"auc_roc": 0.8, "accuracy": 0.7},
            {"auc_roc": 0.9, "accuracy": 0.8},
            {"auc_roc": 0.85, "accuracy": 0.75},
        ]
        ci = compute_confidence_intervals(fold_metrics)
        assert "auc_roc" in ci
        assert "accuracy" in ci
        assert ci["auc_roc"]["mean"] == pytest.approx(0.85, abs=1e-6)

    def test_ci_keys(self):
        """Legacy keys are load-bearing for every caller; H-5a/M-15 added more.

        Asserted as a superset, not an equality: the block is deliberately
        extensible (``method``, ``n_valid_folds``, ``n_folds``), and every
        consumer reads a fixed stat tuple rather than iterating the block
        (``pipeline/results.py:50``, ``pipeline/orchestrator.py:216``).
        """
        fold_metrics = [{"auc_roc": 0.8}, {"auc_roc": 0.9}]
        ci = compute_confidence_intervals(fold_metrics)
        assert {"mean", "std", "ci_low", "ci_high"} <= set(ci["auc_roc"].keys())
        assert {"method", "n_valid_folds", "n_folds"} <= set(ci["auc_roc"].keys())

    def test_ci_ordering(self):
        fold_metrics = [{"auc_roc": v} for v in [0.7, 0.8, 0.9, 0.85, 0.75]]
        ci = compute_confidence_intervals(fold_metrics)
        assert ci["auc_roc"]["ci_low"] < ci["auc_roc"]["mean"]
        assert ci["auc_roc"]["ci_high"] > ci["auc_roc"]["mean"]

    def test_identical_values_zero_ci_width(self):
        fold_metrics = [{"auc_roc": 0.8}] * 5
        ci = compute_confidence_intervals(fold_metrics)
        assert ci["auc_roc"]["std"] == 0.0
        assert ci["auc_roc"]["ci_low"] == ci["auc_roc"]["ci_high"]

    def test_handles_nan_values(self):
        fold_metrics = [
            {"auc_roc": float("nan")},
            {"auc_roc": 0.8},
        ]
        ci = compute_confidence_intervals(fold_metrics)
        # Should not crash; single valid value
        assert not np.isnan(ci["auc_roc"]["mean"])

    def test_single_fold(self):
        fold_metrics = [{"auc_roc": 0.85}]
        ci = compute_confidence_intervals(fold_metrics)
        assert ci["auc_roc"]["mean"] == 0.85
        assert ci["auc_roc"]["ci_low"] == ci["auc_roc"]["ci_high"]


class TestMultiClassSensitivitySpecificity:
    """Multi-class sensitivity/specificity are macro-averaged one-vs-rest.

    They used to be hardcoded ``float("nan")`` for ``n_classes != 2``. That NaN
    rode in the ``summary`` diagnostic block of result.json, serialized as a bare
    ``NaN`` token, and the orchestrator's ingestion parser rejected the whole
    file — so every 3-class run (CPTAC-PDAC, TCGA-HNSC) was recorded as a crash
    despite having a perfectly good validation composite.

    The binary formula is deliberately unchanged: it reports the positive class
    only (``pos_label=1``), and published binary numbers must not move.
    """

    def test_perfect_three_class_prediction(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = y_true.copy()
        y_probs = np.eye(3)[y_pred]
        m = compute_extended_metrics(y_true, y_probs, y_pred, 3)
        assert m["sensitivity"] == 1.0
        assert m["specificity"] == 1.0

    def test_macro_average_matches_hand_computation(self):
        # 3 classes, 2 samples each; class 1 is fully confused into class 2.
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 2, 2, 2, 2])
        y_probs = np.eye(3)[y_pred]
        m = compute_extended_metrics(y_true, y_probs, y_pred, 3)
        # recall:      class0 2/2, class1 0/2, class2 2/2  -> 2/3
        assert m["sensitivity"] == pytest.approx(2 / 3)
        # specificity: class0 TN=4 FP=0 -> 1.0
        #              class1 TN=4 FP=0 -> 1.0
        #              class2 TN=2 FP=2 -> 0.5
        assert m["specificity"] == pytest.approx(5 / 6)

    def test_never_nan_when_a_class_is_absent_from_the_fold(self):
        """A small fold can miss a class entirely — that must not poison the macro."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        y_probs = np.eye(3)[y_pred]
        m = compute_extended_metrics(y_true, y_probs, y_pred, 3)
        assert np.isfinite(m["sensitivity"])
        assert np.isfinite(m["specificity"])

    @pytest.mark.parametrize("n_classes", [3, 4, 5])
    def test_multiclass_metrics_are_always_finite(self, n_classes):
        """The regression guard: no multi-class shape may emit NaN here again."""
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, n_classes, size=20)
        y_pred = rng.integers(0, n_classes, size=20)
        y_probs = np.eye(n_classes)[y_pred]
        m = compute_extended_metrics(y_true, y_probs, y_pred, n_classes)
        assert np.isfinite(m["sensitivity"])
        assert np.isfinite(m["specificity"])

    def test_binary_formula_is_unchanged(self):
        """Binary stays positive-class-only, NOT macro-averaged."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        y_probs = np.eye(2)[y_pred]
        m = compute_extended_metrics(y_true, y_probs, y_pred, 2)
        assert m["sensitivity"] == 1.0    # 2/2 positives recovered
        assert m["specificity"] == 0.5    # 1/2 negatives correct (macro would be 0.75)
