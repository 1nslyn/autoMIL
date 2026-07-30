"""Tests for ``paper/preprint/figures/roster.py``.

The roster filter is load-bearing for every baseline figure: unfiltered,
TCGA-LUAD contributes 105 experiments where the roster is 26, and 35 partial
``cox`` runs join the survival arms. Both were verified on ``fir`` 2026-07-30
(``paper/preprint/GRID-CENSUS-2026-07-30.md``), and both are the kind of drift
that makes a figure quietly wrong rather than obviously broken -- so the exact
counts are pinned here.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "..", "paper", "preprint", "figures"),
)

from roster import (  # noqa: E402
    CELLS_PER_COHORT,
    ROSTER_MODELS,
    ROSTER_TASKS,
    RosterError,
    filter_roster,
    format_report,
)

ENCODERS = ("uni_v2", "virchow2", "hoptimus1")


def _row(dataset, task, task_type, model_type, encoder, survival_loss=None):
    return {
        "dataset": dataset,
        "task": task,
        "task_type": task_type,
        "model_type": model_type,
        "encoder": encoder,
        "survival_loss": survival_loss,
        "test_auc_roc_mean": 0.7,
        "test_c_index_mean": 0.6,
    }


def _complete_roster_rows():
    """The 130 real roster cells: 5 cohorts x (13 classification + 13 survival)."""
    rows = []
    for cohort, task in ROSTER_TASKS.items():
        for model in ("clam_mb", "simple_mil", "abmil", "dtfd_mil"):
            for enc in ENCODERS:
                rows.append(_row(cohort, task, "classification", model, enc))
                rows.append(_row(cohort, "os", "survival", model, enc, "nllsurv"))
        rows.append(_row(cohort, task, "classification", "titan", "titan"))
        rows.append(_row(cohort, "os", "survival", "titan", "titan", "nllsurv"))
    return rows


@pytest.fixture
def roster_df():
    return pd.DataFrame(_complete_roster_rows())


class TestCompleteRoster:
    def test_keeps_exactly_130_cells(self, roster_df):
        kept, _, report = filter_roster(roster_df)
        assert len(kept) == 130
        assert report["n_dropped"] == 0

    def test_every_cohort_has_26_cells(self, roster_df):
        _, _, report = filter_roster(roster_df)
        assert set(report["per_cohort"]) == set(ROSTER_TASKS)
        assert all(n == CELLS_PER_COHORT for n in report["per_cohort"].values())

    def test_split_is_13_classification_13_survival(self, roster_df):
        kept, _, _ = filter_roster(roster_df)
        counts = kept.groupby(["dataset", "task_type"]).size()
        for cohort in ROSTER_TASKS:
            assert counts[(cohort, "classification")] == 13
            assert counts[(cohort, "survival")] == 13


class TestDropsOffRoster:
    def test_drops_cox_survival(self, roster_df):
        extra = [
            _row(c, "os", "survival", m, e, "cox")
            for c in ROSTER_TASKS for m in ("simple_mil", "abmil") for e in ENCODERS
        ] + [_row(c, "os", "survival", "titan", "titan", "cox") for c in ROSTER_TASKS]
        assert len(extra) == 35  # the real cox count on disk
        kept, _, report = filter_roster(pd.concat([roster_df, pd.DataFrame(extra)]))
        assert len(kept) == 130
        assert report["dropped_cox"] == 35
        assert set(kept[kept["task_type"] == "survival"]["survival_loss"]) == {"nllsurv"}

    def test_drops_luad_second_task(self, roster_df):
        """LUAD's off-roster ``egfr`` task must not join its pinned ``kras``."""
        extra = [
            _row("tcga_luad", "egfr", "classification", m, e)
            for m in ("clam_mb", "simple_mil", "abmil", "dtfd_mil") for e in ENCODERS
        ]
        kept, _, report = filter_roster(pd.concat([roster_df, pd.DataFrame(extra)]))
        assert len(kept) == 130
        assert report["dropped_off_roster_task"] == len(extra)
        assert set(kept[kept["task_type"] == "classification"]["task"]) == set(
            ROSTER_TASKS.values()
        )

    def test_drops_off_roster_aggregators(self, roster_df):
        extra = [
            _row("tcga_luad", "kras", "classification", m, e)
            for m in ("clam_sb", "mil", "trans_mil") for e in ENCODERS
        ]
        kept, _, report = filter_roster(pd.concat([roster_df, pd.DataFrame(extra)]))
        assert len(kept) == 130
        assert report["dropped_off_roster_model"] == 9
        assert set(kept["model_type"]) == set(ROSTER_MODELS)

    def test_drops_unknown_cohort(self, roster_df):
        extra = pd.DataFrame([_row("cptac_ccrcc", "kras", "classification", "clam_mb", "uni_v2")])
        kept, _, _ = filter_roster(pd.concat([roster_df, extra]))
        assert len(kept) == 130
        assert "cptac_ccrcc" not in set(kept["dataset"])

    def test_all_three_drop_classes_at_once(self, roster_df):
        """The real 195-experiment tree: 130 roster + 35 cox + 30 off-roster."""
        off = [
            _row("tcga_luad", "egfr", "classification", m, e)
            for m in ("clam_mb", "simple_mil", "abmil", "dtfd_mil", "clam_sb", "mil", "trans_mil")
            for e in ENCODERS
        ][:21] + [
            _row("tcga_luad", "kras", "classification", m, e)
            for m in ("clam_sb", "mil", "trans_mil") for e in ENCODERS
        ]
        cox = [
            _row(c, "os", "survival", m, e, "cox")
            for c in ROSTER_TASKS for m in ("simple_mil", "abmil") for e in ENCODERS
        ] + [_row(c, "os", "survival", "titan", "titan", "cox") for c in ROSTER_TASKS]
        full = pd.concat([roster_df, pd.DataFrame(off), pd.DataFrame(cox)])
        assert len(full) == 195  # matches the real 5-fold trees
        kept, _, report = filter_roster(full)
        assert len(kept) == 130
        assert report["n_in"] == 195
        assert report["n_dropped"] == 65


class TestStrictness:
    def test_incomplete_cohort_raises(self, roster_df):
        short = roster_df.drop(roster_df.index[0])
        with pytest.raises(RosterError, match="roster is incomplete"):
            filter_roster(short)

    def test_missing_cohort_is_named(self, roster_df):
        without_lgg = roster_df[roster_df["dataset"] != "tcga_lgg"]
        with pytest.raises(RosterError, match="tcga_lgg"):
            filter_roster(without_lgg)

    def test_strict_false_allows_incomplete(self, roster_df):
        short = roster_df.drop(roster_df.index[0])
        kept, _, report = filter_roster(short, strict=False)
        assert len(kept) == 129
        assert report["per_cohort"]["tcga_luad"] == 25

    def test_missing_column_raises_naming_it(self):
        with pytest.raises(RosterError, match="survival_loss"):
            filter_roster(pd.DataFrame([{"dataset": "tcga_luad", "task": "kras",
                                         "task_type": "classification",
                                         "model_type": "clam_mb"}]))


class TestPerFoldFrame:
    def test_per_fold_filtered_by_same_rule(self, roster_df):
        """per_fold_frame has no task_type; it must be derived, not required."""
        pf = pd.DataFrame([
            {"dataset": "tcga_luad", "task": "kras", "model_type": "clam_mb",
             "encoder": "uni_v2", "survival_loss": None, "split": "test",
             "fold": 0, "metric": "auc_roc", "value": 0.7},
            {"dataset": "tcga_luad", "task": "os", "model_type": "clam_mb",
             "encoder": "uni_v2", "survival_loss": "nllsurv", "split": "test",
             "fold": 0, "metric": "c_index", "value": 0.6},
            {"dataset": "tcga_luad", "task": "os", "model_type": "abmil",
             "encoder": "uni_v2", "survival_loss": "cox", "split": "test",
             "fold": 0, "metric": "c_index", "value": 0.55},
            {"dataset": "tcga_luad", "task": "egfr", "model_type": "clam_mb",
             "encoder": "uni_v2", "survival_loss": None, "split": "test",
             "fold": 0, "metric": "auc_roc", "value": 0.65},
        ])
        _, kept_pf, _ = filter_roster(roster_df, pf)
        assert len(kept_pf) == 2
        assert set(kept_pf["task"]) == {"kras", "os"}
        assert "cox" not in set(kept_pf["survival_loss"].dropna())

    def test_none_per_fold_returns_none(self, roster_df):
        _, kept_pf, _ = filter_roster(roster_df, None)
        assert kept_pf is None


class TestReport:
    def test_format_report_flags_overlap(self, roster_df):
        _, _, report = filter_roster(roster_df)
        text = format_report(report)
        assert "OVERLAP" in text
        assert "kept 130 of 130" in text

    def test_report_lists_every_cohort(self, roster_df):
        _, _, report = filter_roster(roster_df)
        text = format_report(report)
        for cohort in ROSTER_TASKS:
            assert cohort in text
