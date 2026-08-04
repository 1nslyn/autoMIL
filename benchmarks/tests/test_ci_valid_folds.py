"""M-15 (audit 2026-07-23): NaN folds were dropped from the CI block silently.

``compute_confidence_intervals`` filters non-finite fold values and averages
what is left, so a survival run whose val c-index was NaN in 3 of 5 folds
reported a mean over 2 folds that is indistinguishable, in every emitted field,
from a clean 5-fold mean. H-8 fixed this one level up on the autoMIL path
(``run_experiment.py`` records ``n_valid_folds``/``n_folds``); these tests pin
the same guarantee in the shared helper, per metric.

The interaction to keep in mind: pooled cross-fold concordance
(``pooled_val_block``) remains a diagnostic because the per-fold val c-index
sits on ~2 events. The campaign selection signal is nevertheless the locked
equal-weight fold mean, and ``n_valid_folds`` records its actual support.
"""
from __future__ import annotations

import math

import pytest

from autobench.pipeline.evaluate import compute_confidence_intervals

NAN = float("nan")


def _folds(values, name="c_index"):
    return [{name: v} for v in values]


class TestValidFoldCountsAreSurfaced:
    def test_counts_are_reported(self):
        ci = compute_confidence_intervals(_folds([0.60, 0.70, NAN, NAN, NAN]))["c_index"]
        assert ci["n_valid_folds"] == 2
        assert ci["n_folds"] == 5

    def test_clean_run_reports_full_denominator(self):
        ci = compute_confidence_intervals(_folds([0.60, 0.62, 0.64, 0.66, 0.68]))["c_index"]
        assert ci["n_valid_folds"] == 5
        assert ci["n_folds"] == 5

    def test_degraded_run_is_distinguishable_from_a_clean_one(self):
        """THE defect: identical means, different denominators.

        Before this fix the two blocks agreed on ``mean`` and there was no
        other field that could tell a 2-fold average from a 5-fold one.
        """
        degraded = compute_confidence_intervals(_folds([0.60, 0.70, NAN, NAN, NAN]))["c_index"]
        clean = compute_confidence_intervals(_folds([0.60, 0.70, 0.65, 0.65, 0.65]))["c_index"]

        # The old surface cannot separate them...
        assert degraded["mean"] == pytest.approx(clean["mean"])
        # ...the new one can.
        assert degraded["n_valid_folds"] != clean["n_valid_folds"]
        assert (degraded["n_valid_folds"], clean["n_valid_folds"]) == (2, 5)

    def test_counts_are_per_metric(self):
        """A run can lose c_index folds while keeping every auc_roc fold."""
        folds = [
            {"auc_roc": 0.70, "c_index": 0.55},
            {"auc_roc": 0.72, "c_index": 0.58},
            {"auc_roc": 0.68, "c_index": NAN},
            {"auc_roc": 0.71, "c_index": NAN},
            {"auc_roc": 0.69, "c_index": NAN},
        ]
        ci = compute_confidence_intervals(folds)
        assert ci["auc_roc"]["n_valid_folds"] == 5
        assert ci["c_index"]["n_valid_folds"] == 2
        assert ci["auc_roc"]["n_folds"] == ci["c_index"]["n_folds"] == 5

    def test_metric_missing_from_a_fold_counts_as_invalid(self):
        """A key absent in one fold is imputed NaN; the denominator must show it."""
        folds = [{"auc_roc": 0.70}, {"auc_roc": 0.72}, {}]
        ci = compute_confidence_intervals(folds)["auc_roc"]
        assert ci["n_valid_folds"] == 2
        assert ci["n_folds"] == 3

    def test_degenerate_block_also_carries_counts(self):
        ci = compute_confidence_intervals(_folds([0.60, NAN, NAN, NAN, NAN]))["c_index"]
        assert ci["method"] == "degenerate"
        assert ci["n_valid_folds"] == 1
        assert ci["n_folds"] == 5

    def test_all_nan_metric_reports_zero_valid(self):
        ci = compute_confidence_intervals(_folds([NAN, NAN]))["c_index"]
        assert ci["n_valid_folds"] == 0
        assert math.isnan(ci["mean"])


class TestNonFiniteHandling:
    def test_infinities_are_not_counted_valid(self):
        """+/-inf must be excluded like NaN: an inf fold would otherwise poison
        the mean while inflating the denominator."""
        ci = compute_confidence_intervals(_folds([0.60, 0.70, float("inf")]))["c_index"]
        assert ci["n_valid_folds"] == 2
        assert ci["mean"] == pytest.approx(0.65)
        assert math.isfinite(ci["mean"])

    def test_matches_the_h8_valid_fold_predicate(self):
        """The helper's denominator must agree with the H-8 counter in
        run_experiment.py, which uses ``math.isfinite`` on the primary metric.
        Two different predicates would report two different ``n_valid_folds``
        for the same run.
        """
        values = [0.60, NAN, float("inf"), 0.70, float("-inf")]
        h8_count = sum(
            1 for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))
        )
        ci = compute_confidence_intervals(_folds(values))["c_index"]
        assert ci["n_valid_folds"] == h8_count == 2
