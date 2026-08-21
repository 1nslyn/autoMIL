"""Companion-guard margin derivation: the arithmetic, and its fail-loud edges.

The margin decides when a balanced-accuracy drop counts as evidence of harm, so
its provenance has to be checkable by hand from the counts it publishes. These
tests pin both halves: the formula on synthetic splits, and the real TCGA-LUAD
derivation against the slides a completed run actually scored.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from autobench.guard_margin import (GuardMarginError, balanced_accuracy_margin,
                                    derive_guard, validation_class_counts,
                                    verify_against_run)

REPO_ROOT = Path(__file__).resolve().parents[2]
LUAD = REPO_ROOT / "datasets/TCGA-LUAD/benchmark"
CANARY_RESULTS = (
    REPO_ROOT / "benchmarks/campaigns/preprint_130/runtime-canary"
    / "tcga_luad__kras__virchow2__nnmil__s42__preprint-v2"
    / "baseline-execution/archive/certify/results"
)


def _cohort(tmp_path: Path, folds: dict[int, list[str]], labels: dict[str, str],
            task: str = "t", strategy: str = "standard") -> Path:
    """A minimal benchmark dir: ragged split CSVs plus a task CSV."""
    benchmark = tmp_path / "benchmark"
    (benchmark / "dataset_csv").mkdir(parents=True)
    with (benchmark / "dataset_csv" / f"{task}.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["case_id", "slide_id", "label"])
        for slide, label in labels.items():
            writer.writerow([slide, slide, label])
    splits = benchmark / "splits" / strategy / task
    splits.mkdir(parents=True)
    for fold, val_ids in folds.items():
        with (splits / f"splits_{fold}.csv").open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["train", "val", "test"])
            # Real split CSVs are ragged and blank-padded to the longest
            # column; the val column is shorter than the file is tall.
            for i in range(len(val_ids) + 3):
                writer.writerow([
                    f"train_{i}", val_ids[i] if i < len(val_ids) else "", "",
                ])
    return benchmark


def _balanced(n_per_class: dict[str, int]) -> tuple[dict, dict]:
    labels, val = {}, []
    for label, count in n_per_class.items():
        for i in range(count):
            slide = f"{label}_{i}"
            labels[slide] = label
            val.append(slide)
    return labels, val


class TestMarginArithmetic:
    def test_binary_margin_is_one_slide_of_the_minority_class(self, tmp_path):
        """1 / (K x C x min n) — the largest step one slide can produce."""
        labels, val = _balanced({"pos": 17, "neg": 30})
        benchmark = _cohort(tmp_path, {0: val, 1: val, 2: val}, labels)
        guard = derive_guard(benchmark, "standard", "t", (0, 1, 2))
        assert guard["margin"] == pytest.approx(1 / (3 * 2 * 17), abs=1e-6)
        assert guard["metric"] == "val_bacc"
        assert "17" in guard["basis"] and "'pos'" in guard["basis"]

    def test_three_class_task_divides_by_three_not_two(self, tmp_path):
        """C comes from the data: balanced accuracy averages C recalls."""
        labels, val = _balanced({"a": 6, "b": 20, "c": 20})
        benchmark = _cohort(tmp_path, {0: val, 1: val, 2: val}, labels)
        guard = derive_guard(benchmark, "standard", "t", (0, 1, 2))
        assert guard["margin"] == pytest.approx(1 / (3 * 3 * 6), abs=1e-6)

    def test_margin_tracks_the_smallest_class_of_any_fold(self, tmp_path):
        """A single ragged fold sets the lattice for the averaged number."""
        labels, val = _balanced({"pos": 20, "neg": 30})
        # Fold 2 holds only 11 of the 20 positives; the averaged metric's
        # coarsest single-slide step is set by that fold, not by the others.
        thin = [s for s in val if not s.startswith("pos_")][:30] + \
            [f"pos_{i}" for i in range(11)]
        benchmark = _cohort(tmp_path, {0: val, 1: val, 2: thin}, labels)
        guard = derive_guard(benchmark, "standard", "t", (0, 1, 2))
        assert guard["margin"] == pytest.approx(1 / (3 * 2 * 11), abs=1e-6)

    def test_fewer_folds_give_a_coarser_lattice(self, tmp_path):
        """Averaging over K folds refines the step by exactly K."""
        labels, val = _balanced({"pos": 17, "neg": 30})
        benchmark = _cohort(tmp_path, {0: val, 1: val, 2: val}, labels)
        three = derive_guard(benchmark, "standard", "t", (0, 1, 2))["margin"]
        one = derive_guard(benchmark, "standard", "t", (0,))["margin"]
        assert one == pytest.approx(3 * three, rel=1e-3)

    def test_counts_travel_with_the_margin(self, tmp_path):
        """The number must be re-derivable by hand from what it publishes."""
        labels, val = _balanced({"pos": 17, "neg": 30})
        benchmark = _cohort(tmp_path, {0: val, 1: val, 2: val}, labels)
        guard = derive_guard(benchmark, "standard", "t", (0, 1, 2))
        counts = guard["validation_class_counts"]
        assert set(counts) == {"0", "1", "2"}
        assert counts["0"] == {"neg": 30, "pos": 17}
        k, c = len(counts), len(counts["0"])
        assert guard["margin"] == pytest.approx(
            1 / (k * c * min(min(f.values()) for f in counts.values())), abs=1e-6
        )


class TestFailsLoud:
    def test_missing_split_file(self, tmp_path):
        labels, val = _balanced({"pos": 5, "neg": 5})
        benchmark = _cohort(tmp_path, {0: val}, labels)
        with pytest.raises(GuardMarginError, match="missing split file"):
            validation_class_counts(benchmark, "standard", "t", (0, 1))

    def test_validation_slide_without_a_label(self, tmp_path):
        labels, val = _balanced({"pos": 5, "neg": 5})
        benchmark = _cohort(tmp_path, {0: [*val, "ghost_slide"]}, labels)
        with pytest.raises(GuardMarginError, match="no label"):
            validation_class_counts(benchmark, "standard", "t", (0,))

    def test_empty_validation_column(self, tmp_path):
        labels, _ = _balanced({"pos": 5, "neg": 5})
        benchmark = _cohort(tmp_path, {0: []}, labels)
        with pytest.raises(GuardMarginError, match="no slides to validation"):
            validation_class_counts(benchmark, "standard", "t", (0,))

    def test_folds_with_different_class_sets(self):
        """A fold missing a class averages over a different C entirely."""
        with pytest.raises(GuardMarginError, match="different class sets"):
            balanced_accuracy_margin({0: {"a": 5, "b": 5}, 1: {"a": 10}})

    def test_single_class_fold(self):
        with pytest.raises(GuardMarginError, match="undefined below two"):
            balanced_accuracy_margin({0: {"a": 5}, 1: {"a": 5}})

    def test_no_folds(self):
        with pytest.raises(GuardMarginError, match="no folds"):
            balanced_accuracy_margin({})

    def test_survival_task_csv_has_no_labels(self, tmp_path):
        """A survival cohort takes no guard, and says so rather than guessing."""
        benchmark = tmp_path / "benchmark"
        (benchmark / "dataset_csv").mkdir(parents=True)
        (benchmark / "dataset_csv" / "os.csv").write_text(
            "case_id,slide_id,status,time\nc,s,1,10\n"
        )
        with pytest.raises(GuardMarginError, match="'slide_id' and 'label'"):
            validation_class_counts(benchmark, "standard", "os", (0,))


class TestRunCrossCheck:
    """The margin is derived from the split ASSIGNMENT; a run can retain less."""

    def test_detects_a_run_that_scored_fewer_slides(self, tmp_path):
        results = tmp_path / "results" / "fold_0"
        results.mkdir(parents=True)
        (results / "predictions_val.csv").write_text(
            "slide_id,y_true\n" + "".join(f"s{i},0\n" for i in range(9))
        )
        with pytest.raises(GuardMarginError, match="the run scored 9"):
            verify_against_run({0: {"a": 5, "b": 5}}, tmp_path / "results")

    def test_missing_predictions_are_not_silently_accepted(self, tmp_path):
        with pytest.raises(GuardMarginError, match="no scored validation"):
            verify_against_run({0: {"a": 5, "b": 5}}, tmp_path / "nowhere")


@pytest.mark.skipif(not LUAD.is_dir(), reason="TCGA-LUAD cohort is not mounted")
class TestRealCohort:
    def test_tcga_luad_kras_margin(self):
        """The number frozen in guard_margins.json, re-derived from the splits."""
        guard = derive_guard(LUAD, "standard", "kras", (0, 1, 2))
        assert guard["margin"] == pytest.approx(0.009804, abs=5e-7)
        for fold in ("0", "1", "2"):
            assert guard["validation_class_counts"][fold] == {
                "mutant": 17, "wildtype": 30,
            }

    def test_frozen_artifact_matches_a_fresh_derivation(self):
        import json

        frozen = json.loads(
            (REPO_ROOT / "benchmarks/campaigns/preprint_130/guard_margins.json")
            .read_text()
        )
        assert frozen["tcga_luad__kras"] == derive_guard(
            LUAD, "standard", "kras", (0, 1, 2)
        )

    @pytest.mark.skipif(not CANARY_RESULTS.is_dir(),
                        reason="no completed baseline to cross-check against")
    def test_derived_counts_match_what_a_real_run_scored(self):
        """Closes the assigned-vs-retained gap on real artifacts.

        A loader that dropped validation slides would leave the cohort with a
        COARSER lattice than the split assignment implies, making the derived
        margin tighter than one true slide flip. The canary run scored every
        assigned slide, so the frozen margin describes the lattice the runs
        actually have.
        """
        counts = validation_class_counts(LUAD, "standard", "kras", (0, 1, 2))
        verify_against_run(counts, CANARY_RESULTS)   # raises on any mismatch
