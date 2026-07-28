"""Tests for ``autobench.pipeline.collect`` (FIG-0).

Builds a synthetic ``results/`` tree on ``tmp_path`` that mirrors the REAL path
shape (``ExperimentConfig.results_subdir``): several cohorts across TCGA *and*
CPTAC (never a shared prefix), classification AND survival, a ``survival_loss``
level with two colliding-by-name losses under the same (dataset, framework,
model, encoder, seed) combo, two seeds under the same combo, and all five
frameworks. This is the exact grid of breakages the gitignored
``tasks/baseline_summary/scripts/00_aggregate.py`` had against the current
roster (see ``collect.py``'s module docstring) -- every one of those five
breakages gets a dedicated assertion below.
"""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from autobench.pipeline.collect import (
    _PER_FOLD_COLUMNS,
    collect_summaries,
    per_fold_frame,
    summaries_to_frame,
)
from autobench.pipeline.config import (
    ExperimentConfig,
    Framework,
    ModelConfig,
    TaskConfig,
    TrainConfig,
)
from autobench.pipeline.evaluate import compute_confidence_intervals

N_FOLDS = 3


def _write_summary(
    benchmark_dir: str,
    *,
    dataset: str,
    framework: Framework,
    task_name: str,
    encoder: str,
    model_type: str,
    seed: int,
    survival_loss: str | None = None,
    task_type: str = "classification",
) -> dict:
    """Write one summary.json at the REAL path shape and return the dict.

    Uses the real ``ExperimentConfig.results_subdir`` / ``.experiment_id`` and
    the real ``compute_confidence_intervals`` so the fixture cannot silently
    drift from what the five framework runners actually produce (the
    "Authoritative schema" instruction: exp_summary at the end of
    ``clam/runner.py``, identical across arms).
    """
    task = TaskConfig(
        name=task_name, label_col="label", label_dict={"a": 0, "b": 1},
        task_type=task_type,
    )
    exp_cfg = ExperimentConfig(
        task=task,
        encoder_key=encoder,
        embed_dim=768,
        model=ModelConfig(model_type=model_type),
        train=TrainConfig(seed=seed),
        n_folds=N_FOLDS,
        framework=framework,
        strategy="standard",
        dataset=dataset,
        survival_loss=survival_loss,
    )

    if task_type == "survival":
        per_fold_test = [{"c_index": 0.58 + 0.01 * i} for i in range(N_FOLDS)]
        per_fold_val = [{"c_index": 0.55 + 0.01 * i} for i in range(N_FOLDS)]
    else:
        per_fold_test = [
            {"auc_roc": 0.70 + 0.01 * i, "balanced_accuracy": 0.65 + 0.01 * i}
            for i in range(N_FOLDS)
        ]
        per_fold_val = [
            {"auc_roc": 0.68 + 0.01 * i, "balanced_accuracy": 0.60 + 0.01 * i}
            for i in range(N_FOLDS)
        ]

    summary = {
        "dataset": exp_cfg.dataset,
        "experiment_id": exp_cfg.experiment_id,
        "task": exp_cfg.task.name,
        "encoder": exp_cfg.encoder_key,
        "embed_dim": exp_cfg.embed_dim,
        "model_type": exp_cfg.model.model_type,
        "survival_loss": exp_cfg.survival_loss,
        "framework": exp_cfg.framework.value,
        "strategy": exp_cfg.strategy,
        "n_folds": exp_cfg.n_folds,
        "elapsed_seconds_total": 100,
        "seed": exp_cfg.train.seed,
        "test": compute_confidence_intervals(per_fold_test),
        "val": compute_confidence_intervals(per_fold_val),
        "val_pooled": {},
        "per_fold_test": per_fold_test,
        "per_fold_val": per_fold_val,
    }

    results_dir = os.path.join(benchmark_dir, "results", exp_cfg.results_subdir)
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f)
    return summary


@pytest.fixture
def results_tree(tmp_path) -> str:
    """A synthetic benchmark_dir covering the FIG-0 breakage grid.

    4 cohorts (2 TCGA + 2 CPTAC -- never a shared prefix), all 5 frameworks,
    classification AND survival, a survival_loss level with cox+nllsurv under
    the SAME combo (collision check), and two seeds under the SAME combo
    (collision check).
    """
    root = str(tmp_path / "benchmark")

    # classification, one per framework except TITAN (TITAN is survival-only here)
    _write_summary(root, dataset="TCGA-LUAD", framework=Framework.CLAM,
                   task_name="kras", encoder="uni_v2", model_type="clam_mb", seed=42)
    _write_summary(root, dataset="TCGA-LUAD", framework=Framework.CLAM,
                   task_name="kras", encoder="uni_v2", model_type="clam_mb", seed=43)
    _write_summary(root, dataset="CPTAC-GBM", framework=Framework.NNMIL,
                   task_name="tp53", encoder="virchow2", model_type="simple_mil", seed=42)
    _write_summary(root, dataset="CPTAC-PDAC", framework=Framework.ABMIL,
                   task_name="immune_class", encoder="hoptimus1", model_type="abmil", seed=42)
    _write_summary(root, dataset="TCGA-HNSC", framework=Framework.DTFD,
                   task_name="grade", encoder="uni_v2", model_type="dtfd_mil", seed=42)

    # survival: TITAN cox + nllsurv under the SAME (dataset, encoder, seed) combo
    _write_summary(root, dataset="TCGA-LUAD", framework=Framework.TITAN,
                   task_name="os", encoder="titan", model_type="titan", seed=42,
                   survival_loss="cox", task_type="survival")
    _write_summary(root, dataset="TCGA-LUAD", framework=Framework.TITAN,
                   task_name="os", encoder="titan", model_type="titan", seed=42,
                   survival_loss="nllsurv", task_type="survival")
    # survival on CLAM too, so CLAM appears in both task types
    _write_summary(root, dataset="CPTAC-GBM", framework=Framework.CLAM,
                   task_name="os", encoder="uni_v2", model_type="clam_mb", seed=42,
                   survival_loss="nllsurv", task_type="survival")

    return root


# ---------------------------------------------------------------------------
# collect_summaries
# ---------------------------------------------------------------------------


class TestCollectSummaries:
    def test_finds_all_five_frameworks(self, results_tree):
        summaries = collect_summaries([results_tree])
        frameworks = {s["framework"] for s in summaries}
        assert frameworks == {"clam", "nnmil", "dtfd", "abmil", "titan"}

    def test_finds_expected_total_count(self, results_tree):
        summaries = collect_summaries([results_tree])
        assert len(summaries) == 8

    def test_dataset_field_present_and_not_tcga_only(self, results_tree):
        summaries = collect_summaries([results_tree])
        datasets = {s["dataset"] for s in summaries}
        # The old aggregator globbed "TCGA-*" so both CPTAC cohorts were
        # structurally invisible. Assert they are found here.
        assert "CPTAC-GBM" in datasets
        assert "CPTAC-PDAC" in datasets
        assert "TCGA-LUAD" in datasets
        assert "TCGA-HNSC" in datasets
        assert all(s.get("dataset") for s in summaries), (
            "every summary must carry a non-empty dataset field"
        )

    def test_survival_loss_level_parses_and_does_not_collide(self, results_tree):
        summaries = collect_summaries([results_tree])
        titan_survival = [s for s in summaries if s["framework"] == "titan"]
        assert len(titan_survival) == 2, (
            "cox and nllsurv under the same (dataset, encoder, seed) combo "
            "must both survive the walk, not collide into one file"
        )
        losses = {s["survival_loss"] for s in titan_survival}
        assert losses == {"cox", "nllsurv"}

    def test_c_index_survives_for_every_survival_summary(self, results_tree):
        summaries = collect_summaries([results_tree])
        survival_summaries = [s for s in summaries if s.get("survival_loss") is not None]
        assert len(survival_summaries) == 3  # 2x TITAN + 1x CLAM
        for s in survival_summaries:
            assert "c_index" in s["test"], s["experiment_id"]
            assert "c_index" in s["val"], s["experiment_id"]

    def test_two_seeds_do_not_collide(self, results_tree):
        summaries = collect_summaries([results_tree])
        luad_clam_kras = [
            s for s in summaries
            if s["framework"] == "clam" and s["task"] == "kras"
            and s["dataset"] == "TCGA-LUAD"
        ]
        assert len(luad_clam_kras) == 2
        assert {s["seed"] for s in luad_clam_kras} == {42, 43}

    def test_multiple_roots_are_pooled(self, tmp_path, results_tree):
        second_root = str(tmp_path / "second_benchmark")
        _write_summary(second_root, dataset="TCGA-LGG", framework=Framework.CLAM,
                       task_name="idh1", encoder="uni_v2", model_type="clam_mb", seed=42)
        summaries = collect_summaries([results_tree, second_root])
        assert len(summaries) == 9
        assert "TCGA-LGG" in {s["dataset"] for s in summaries}

    def test_root_with_no_results_dir_is_skipped(self, tmp_path):
        empty_root = tmp_path / "no_results_here"
        empty_root.mkdir()
        assert collect_summaries([str(empty_root)]) == []

    def test_nonexistent_root_does_not_crash(self, tmp_path):
        assert collect_summaries([str(tmp_path / "does_not_exist_at_all")]) == []

    def test_corrupt_and_missing_summary_json_are_skipped(self, tmp_path):
        root = str(tmp_path / "mixed_tree")
        good = _write_summary(root, dataset="TCGA-LUAD", framework=Framework.CLAM,
                              task_name="kras", encoder="uni_v2",
                              model_type="clam_mb", seed=42)

        corrupt_dir = os.path.join(
            root, "results", "clam", "standard", "kras", "uni_v2", "clam_mb", "s99",
        )
        os.makedirs(corrupt_dir, exist_ok=True)
        with open(os.path.join(corrupt_dir, "summary.json"), "w") as f:
            f.write("{not valid json,,,")

        # A directory that looks like a leaf but the file itself is absent
        # (e.g. a race with an in-progress write) -- glob won't even match
        # this since there's no summary.json, but assert it doesn't crash
        # the walk of a sibling directory either.
        empty_leaf = os.path.join(
            root, "results", "clam", "standard", "kras", "uni_v2", "clam_mb", "s100",
        )
        os.makedirs(empty_leaf, exist_ok=True)

        summaries = collect_summaries([root])
        assert len(summaries) == 1
        assert summaries[0]["experiment_id"] == good["experiment_id"]


# ---------------------------------------------------------------------------
# summaries_to_frame
# ---------------------------------------------------------------------------


class TestSummariesToFrame:
    def test_row_per_experiment(self, results_tree):
        summaries = collect_summaries([results_tree])
        frame = summaries_to_frame(summaries)
        assert len(frame) == len(summaries) == 8

    def test_dataset_and_task_type_columns_present(self, results_tree):
        summaries = collect_summaries([results_tree])
        frame = summaries_to_frame(summaries)
        assert "dataset" in frame.columns
        assert "task_type" in frame.columns
        assert set(frame["task_type"].unique()) == {"classification", "survival"}

    def test_survival_rows_carry_c_index_mean(self, results_tree):
        summaries = collect_summaries([results_tree])
        frame = summaries_to_frame(summaries)
        survival_rows = frame[frame["task_type"] == "survival"]
        assert len(survival_rows) == 3
        assert "test_c_index_mean" in frame.columns
        assert survival_rows["test_c_index_mean"].notna().all()

    def test_classification_rows_carry_auc(self, results_tree):
        summaries = collect_summaries([results_tree])
        frame = summaries_to_frame(summaries)
        cls_rows = frame[frame["task_type"] == "classification"]
        assert len(cls_rows) == 5
        assert "test_auc_roc_mean" in frame.columns
        assert cls_rows["test_auc_roc_mean"].notna().all()

    def test_experiment_id_present_and_unique_within_dataset(self, results_tree):
        summaries = collect_summaries([results_tree])
        frame = summaries_to_frame(summaries)
        assert "experiment_id" in frame.columns
        # experiment_id has no dataset segment (config.py ExperimentConfig),
        # so uniqueness must be checked as (dataset, experiment_id).
        assert frame[["dataset", "experiment_id"]].drop_duplicates().shape[0] == len(frame)

    def test_empty_input_returns_empty_frame(self):
        frame = summaries_to_frame([])
        assert frame.empty


# ---------------------------------------------------------------------------
# per_fold_frame
# ---------------------------------------------------------------------------


class TestPerFoldFrame:
    def test_cardinality_for_one_classification_experiment(self, results_tree):
        summaries = collect_summaries([results_tree])
        fold_df = per_fold_frame(summaries)
        subset = fold_df[
            (fold_df["framework"] == "clam")
            & (fold_df["task"] == "kras")
            & (fold_df["dataset"] == "TCGA-LUAD")
            & (fold_df["seed"] == 42)
        ]
        # N_FOLDS folds x 2 metrics (auc_roc, balanced_accuracy) x 2 splits (test, val)
        assert len(subset) == N_FOLDS * 2 * 2

    def test_cardinality_for_one_survival_experiment(self, results_tree):
        summaries = collect_summaries([results_tree])
        fold_df = per_fold_frame(summaries)
        subset = fold_df[
            (fold_df["framework"] == "titan") & (fold_df["survival_loss"] == "cox")
        ]
        # N_FOLDS folds x 1 metric (c_index) x 2 splits (test, val)
        assert len(subset) == N_FOLDS * 1 * 2
        assert (subset["metric"] == "c_index").all()

    def test_survival_loss_column_disambiguates_cox_and_nllsurv(self, results_tree):
        summaries = collect_summaries([results_tree])
        fold_df = per_fold_frame(summaries)
        titan_rows = fold_df[fold_df["framework"] == "titan"]
        assert set(titan_rows["survival_loss"].unique()) == {"cox", "nllsurv"}
        # Each loss variant keeps its own independent fold rows (no collision).
        cox_rows = titan_rows[titan_rows["survival_loss"] == "cox"]
        nllsurv_rows = titan_rows[titan_rows["survival_loss"] == "nllsurv"]
        assert len(cox_rows) == len(nllsurv_rows) == N_FOLDS * 1 * 2

    def test_dataset_column_present(self, results_tree):
        summaries = collect_summaries([results_tree])
        fold_df = per_fold_frame(summaries)
        assert "dataset" in fold_df.columns
        assert set(fold_df["dataset"].unique()) == {
            "TCGA-LUAD", "CPTAC-GBM", "CPTAC-PDAC", "TCGA-HNSC",
        }

    def test_empty_input_keeps_stable_columns(self):
        fold_df = per_fold_frame([])
        assert fold_df.empty
        assert list(fold_df.columns) == _PER_FOLD_COLUMNS


# ---------------------------------------------------------------------------
# n_valid_folds / task_type presence in the static-grid summary.json (FIG-0 ask)
# ---------------------------------------------------------------------------


class TestStaticGridSchemaGaps:
    """Documents, rather than papers over, two absent fields.

    ``n_valid_folds`` is computed only in ``summary_to_result_json``
    (``benchmarks/scripts/run_experiment.py``) -- the autoMIL orchestrator
    path -- from ``summary["per_fold_val"]`` *after* summary.json has already
    been written by the runner. It is never written back into summary.json,
    so the static grid (this collector's input) never has it.
    ``task_type`` is never part of any of the five runners' ``exp_summary``
    dict (see ``_task_type``'s docstring) -- it must be derived, not read.
    """

    def test_n_valid_folds_absent_from_static_grid_summary(self, results_tree):
        summaries = collect_summaries([results_tree])
        assert summaries, "fixture must produce at least one summary"
        for s in summaries:
            assert "n_valid_folds" not in s

    def test_task_type_absent_from_static_grid_summary(self, results_tree):
        summaries = collect_summaries([results_tree])
        assert summaries, "fixture must produce at least one summary"
        for s in summaries:
            assert "task_type" not in s

    def test_summaries_to_frame_does_not_fabricate_n_valid_folds(self, results_tree):
        summaries = collect_summaries([results_tree])
        frame = summaries_to_frame(summaries)
        assert "n_valid_folds" not in frame.columns
