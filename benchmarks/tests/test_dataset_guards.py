"""M-9: retained-fraction / per-class-floor guard, unit-level.

See ``autobench.pipeline.dataset_guards`` module docstring for the defect
this closes: dataset loaders silently dropped slides missing a feature file,
so a partially-extracted cohort returned a confident metric computed on
whatever fraction of val/test happened to survive.
"""
from __future__ import annotations

import pytest

from autobench.pipeline.dataset_guards import (
    MIN_CLASS_COUNT,
    MIN_RETAINED_FRACTION,
    SplitRetentionError,
    check_split_retention,
)


class TestRetainedFraction:
    def test_full_retention_passes(self):
        check_split_retention(
            context="cohort=x task=y", split="val",
            expected_total=10, retained_total=10,
        )

    def test_below_floor_raises_on_val(self):
        with pytest.raises(SplitRetentionError, match=r"7/10"):
            check_split_retention(
                context="cohort=x task=y", split="val",
                expected_total=10, retained_total=7,
            )

    def test_below_floor_raises_on_test(self):
        with pytest.raises(SplitRetentionError):
            check_split_retention(
                context="cohort=x task=y", split="test",
                expected_total=10, retained_total=7,
            )

    def test_at_exactly_the_floor_passes(self):
        # min_retained_fraction default 0.9 -> 9/10 == 0.9, not < 0.9.
        check_split_retention(
            context="cohort=x task=y", split="val",
            expected_total=10, retained_total=9,
        )

    def test_just_below_the_floor_raises(self):
        check_split_retention(
            context="c", split="val", expected_total=100, retained_total=90,
        )
        with pytest.raises(SplitRetentionError):
            check_split_retention(
                context="c", split="val", expected_total=100, retained_total=89,
            )

    def test_empty_split_is_not_this_guards_problem(self):
        """Nothing was ever assigned to this split -- not a retention defect."""
        check_split_retention(
            context="cohort=x task=y", split="val",
            expected_total=0, retained_total=0,
        )

    def test_custom_threshold_is_honoured(self):
        check_split_retention(
            context="c", split="val", expected_total=10, retained_total=6,
            min_retained_fraction=0.5,
        )
        with pytest.raises(SplitRetentionError):
            check_split_retention(
                context="c", split="val", expected_total=10, retained_total=4,
                min_retained_fraction=0.5,
            )

    def test_error_names_the_context_and_split(self):
        with pytest.raises(SplitRetentionError, match=r"cohort=luad task=kras"):
            check_split_retention(
                context="cohort=luad task=kras", split="test",
                expected_total=10, retained_total=1,
            )


class TestTrainWarnsInsteadOfRaising(object):
    def test_train_below_floor_warns_and_proceeds(self, capsys):
        check_split_retention(
            context="cohort=x task=y", split="train",
            expected_total=10, retained_total=1, warn_only=True,
        )
        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "1/10" in out

    def test_train_full_retention_prints_nothing(self, capsys):
        check_split_retention(
            context="cohort=x task=y", split="train",
            expected_total=10, retained_total=10, warn_only=True,
        )
        assert capsys.readouterr().out == ""


class TestPerClassFloor:
    def test_starved_class_raises_even_with_high_overall_fraction(self):
        """95% overall retention, but the minority class lost its only slide."""
        with pytest.raises(SplitRetentionError, match=r"per-class floor"):
            check_split_retention(
                context="c", split="val",
                expected_total=20, retained_total=19,
                expected_by_class={"neg": 15, "pos": 5},
                retained_by_class={"neg": 15, "pos": 4},
                min_class_count=5,
            )

    def test_class_reduced_to_zero_is_flagged(self):
        with pytest.raises(SplitRetentionError, match=r"'pos': 0/5"):
            check_split_retention(
                context="c", split="test",
                expected_total=20, retained_total=15,
                expected_by_class={"neg": 15, "pos": 5},
                retained_by_class={"neg": 15},
                min_class_count=1,
            )

    def test_all_classes_above_floor_passes(self):
        check_split_retention(
            context="c", split="val",
            expected_total=20, retained_total=19,
            expected_by_class={"neg": 15, "pos": 5},
            retained_by_class={"neg": 14, "pos": 5},
            min_class_count=1,
        )

    def test_class_absent_from_expected_is_never_a_problem(self):
        """A class that never appeared in the split's assignment (n_expected=0)
        cannot be "starved" -- guards against a spurious KeyError-style
        report for a class the fold simply doesn't contain."""
        check_split_retention(
            context="c", split="val",
            expected_total=15, retained_total=15,
            expected_by_class={"neg": 15, "pos": 0},
            retained_by_class={"neg": 15},
            min_class_count=1,
        )

    def test_per_class_floor_omitted_entirely_when_no_class_info_given(self):
        """Survival splits have no classes -- omitting both dicts must skip
        the per-class half cleanly, not raise a TypeError."""
        check_split_retention(
            context="c", split="val", expected_total=10, retained_total=9,
        )


class TestDefaults:
    def test_default_fraction_is_point_nine(self):
        assert MIN_RETAINED_FRACTION == 0.9

    def test_default_class_floor_is_one(self):
        assert MIN_CLASS_COUNT == 1
