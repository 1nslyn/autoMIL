"""H-5a (audit 2026-07-23): the K=5 percentile bootstrap CI is indefensible.

``compute_confidence_intervals`` resampled the five fold-level scalars. A
bootstrap over five numbers draws from at most C(9,5) = 126 distinct resample
multisets, and — the defect these tests pin — the percentile interval of a mean
can never leave the convex hull of the observed folds. It is therefore
structurally incapable of expressing the uncertainty of a 5-fold mean, and it
under-covers exactly when C3's within-lineage lift analysis needs it most
(selection on ~10 validation patients).

The fix makes a Student-t interval on K-1 = 4 df the default, keeps the
bootstrap behind an explicit ``method="bootstrap"``, and stamps every emitted
block with the ``method`` that produced it so a figure can never silently mix
the two.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from autobench.pipeline.evaluate import compute_confidence_intervals

# Student-t 97.5th percentile on 4 df. Hardcoded so a scipy behaviour change
# is caught here rather than silently rescaling every error bar in the paper.
T_CRIT_4DF_975 = 2.7764451051977934

# A realistic, non-degenerate 5-fold val AUC vector.
FIVE_FOLDS = [0.70, 0.72, 0.68, 0.71, 0.69]


def _folds(values, name="auc_roc"):
    return [{name: v} for v in values]


class TestStudentTIsTheDefault:
    def test_default_method_is_reported_as_t(self):
        ci = compute_confidence_intervals(_folds(FIVE_FOLDS))
        assert ci["auc_roc"]["method"] == "t"

    def test_t_interval_matches_hand_computed_t4(self):
        """mean +/- t(0.975, 4) * s / sqrt(5), with s on ddof=1."""
        ci = compute_confidence_intervals(_folds(FIVE_FOLDS))["auc_roc"]
        values = np.array(FIVE_FOLDS)
        half_width = T_CRIT_4DF_975 * values.std(ddof=1) / math.sqrt(len(values))
        assert ci["ci_low"] == pytest.approx(values.mean() - half_width, abs=1e-12)
        assert ci["ci_high"] == pytest.approx(values.mean() + half_width, abs=1e-12)

    def test_df_is_k_minus_one_not_k(self):
        """A 3-fold run must widen relative to a 5-fold run at the same s.

        Guards against an off-by-one in the df (t(2) = 4.303 vs t(4) = 2.776).
        """
        three = compute_confidence_intervals(_folds([0.68, 0.70, 0.72]))["auc_roc"]
        five = compute_confidence_intervals(_folds([0.68, 0.69, 0.70, 0.71, 0.72]))["auc_roc"]
        three_width = three["ci_high"] - three["ci_low"]
        five_width = five["ci_high"] - five["ci_low"]
        assert three_width > five_width

    def test_confidence_level_is_honoured(self):
        narrow = compute_confidence_intervals(_folds(FIVE_FOLDS), confidence=0.80)["auc_roc"]
        wide = compute_confidence_intervals(_folds(FIVE_FOLDS), confidence=0.99)["auc_roc"]
        assert (wide["ci_high"] - wide["ci_low"]) > (narrow["ci_high"] - narrow["ci_low"])


class TestWhyTheBootstrapWasIndefensible:
    """The two numerical statements of H-5a, asserted rather than argued."""

    def test_percentile_bootstrap_cannot_leave_the_fold_range(self):
        """A percentile bootstrap of a mean is trapped inside [min, max].

        Every bootstrap replicate is an average of observed folds, so no
        percentile of that distribution can fall outside the observed range —
        no matter how few folds there are. The interval therefore reports
        ~zero uncertainty about values just outside the data, which at K=5 is
        precisely where the truth plausibly lies.
        """
        ci = compute_confidence_intervals(_folds(FIVE_FOLDS), method="bootstrap")["auc_roc"]
        assert ci["ci_low"] >= min(FIVE_FOLDS)
        assert ci["ci_high"] <= max(FIVE_FOLDS)

    def test_t_interval_is_strictly_wider_than_the_bootstrap(self):
        """The bootstrap's plug-in scale (ddof=0) and normal-ish quantile both
        understate a K=5 interval; t(4) with ddof=1 is ~1.6x wider here."""
        t_ci = compute_confidence_intervals(_folds(FIVE_FOLDS))["auc_roc"]
        boot_ci = compute_confidence_intervals(_folds(FIVE_FOLDS), method="bootstrap")["auc_roc"]
        t_width = t_ci["ci_high"] - t_ci["ci_low"]
        boot_width = boot_ci["ci_high"] - boot_ci["ci_low"]
        assert t_width > boot_width
        assert t_width / boot_width > 1.3


class TestMethodIsAlwaysLabelled:
    def test_bootstrap_block_is_labelled_bootstrap(self):
        ci = compute_confidence_intervals(_folds(FIVE_FOLDS), method="bootstrap")
        assert ci["auc_roc"]["method"] == "bootstrap"

    def test_degenerate_block_is_labelled_degenerate(self):
        """<2 valid folds yields a zero-width point estimate; it must not be
        labelled 't' — a figure filtering on method must be able to drop it."""
        ci = compute_confidence_intervals(_folds([0.85, float("nan")]))["auc_roc"]
        assert ci["method"] == "degenerate"
        assert ci["ci_low"] == ci["ci_high"] == pytest.approx(0.85)

    def test_method_is_per_metric_not_per_block(self):
        """One metric can be degenerate while another is fine in the same run."""
        folds = [
            {"auc_roc": 0.70, "c_index": 0.55},
            {"auc_roc": 0.72, "c_index": float("nan")},
            {"auc_roc": 0.68, "c_index": float("nan")},
        ]
        ci = compute_confidence_intervals(folds)
        assert ci["auc_roc"]["method"] == "t"
        assert ci["c_index"]["method"] == "degenerate"

    def test_unknown_method_fails_fast(self):
        with pytest.raises(ValueError, match="method"):
            compute_confidence_intervals(_folds(FIVE_FOLDS), method="bca")


class TestBackwardCompatibleContract:
    def test_legacy_keys_are_all_preserved(self):
        ci = compute_confidence_intervals(_folds(FIVE_FOLDS))["auc_roc"]
        assert {"mean", "std", "ci_low", "ci_high"} <= set(ci)

    def test_selection_signal_is_unchanged_by_the_method_switch(self):
        """LOAD-BEARING: primary_value/keep-discard read only ``mean``.

        run_experiment.py::summary_to_result_json consumes ``.get(metric).get("mean")``
        and nothing else, so switching the default interval must not move any
        number that has already driven a dispatched run's selection.
        """
        t_ci = compute_confidence_intervals(_folds(FIVE_FOLDS))["auc_roc"]
        boot_ci = compute_confidence_intervals(_folds(FIVE_FOLDS), method="bootstrap")["auc_roc"]
        assert t_ci["mean"] == boot_ci["mean"]
        assert t_ci["std"] == boot_ci["std"]

    def test_positional_call_signature_still_works(self):
        """Runners call with one positional arg; the historical positional
        params must keep their slots so no call site silently rebinds."""
        ci = compute_confidence_intervals(_folds(FIVE_FOLDS), 0.95, 200, 7)
        assert ci["auc_roc"]["method"] == "t"

    def test_identical_folds_still_give_zero_width(self):
        ci = compute_confidence_intervals(_folds([0.8] * 5))["auc_roc"]
        assert ci["std"] == 0.0
        assert ci["ci_low"] == ci["ci_high"] == pytest.approx(0.8)
        assert ci["method"] == "t"
