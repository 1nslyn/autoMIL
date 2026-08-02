"""Tests for nnMIL data preparation (dataset.json, dataset.csv, dataset_plan.json)."""

import json
import os

import h5py
import numpy as np
import pandas as pd
import pytest

from autobench.pipeline.config import (
    build_registries,
)
from autobench.pipeline.nnmil.evaluate import normalize_nnmil_metrics
from autobench.pipeline.nnmil.prepare import (
    _analyze_features,
    _generate_training_config,
    _load_splits_as_nnmil_format,
    prepare_nnmil_experiment,
)
from autobench.pipeline.splits import create_strategy_splits
from _helpers import make_test_ds


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ds():
    return make_test_ds()


@pytest.fixture
def registries(ds):
    return build_registries(ds)


@pytest.fixture
def benchmark_dir(tmp_path):
    """Create a minimal benchmark directory structure."""
    bd = str(tmp_path / "benchmark")
    os.makedirs(os.path.join(bd, "dataset_csv"), exist_ok=True)
    os.makedirs(os.path.join(bd, "splits"), exist_ok=True)
    return bd


@pytest.fixture
def h5_features_dir(tmp_path):
    """Create synthetic H5 feature files."""
    h5_dir = tmp_path / "features_conch_v15"
    h5_dir.mkdir()
    # M-9 (nnMIL follow-up, 2026-07-29): the task CSV below has 45 slides. This
    # used to write only 30 H5 files — a 33% silent drop, which is exactly the
    # defect the retention guard now refuses. The fixture's purpose is to build a
    # plan, not to exercise a partially-extracted cohort, so it is now complete;
    # the drop itself is covered by test_nnmil_prepare_retention.py.
    for i in range(45):
        n_patches = np.random.randint(50, 200)
        with h5py.File(h5_dir / f"slide_{i:05d}.h5", "w") as f:
            f.create_dataset("features", data=np.random.randn(n_patches, 768).astype(np.float32))
            f.create_dataset("coords", data=np.random.randint(0, 1000, (n_patches, 2)))
    return str(h5_dir)


@pytest.fixture
def task_csv_with_splits(benchmark_dir, registries):
    """Create a task CSV and splits for testing."""
    rows = []
    i = 0
    for _ in range(45):
        rows.append({
            "case_id": f"P{i:03d}",
            "slide_id": f"slide_{i:05d}",
            "label": "neg" if i % 2 == 0 else "pos",
        })
        i += 1

    # Create brca.csv (default task csv)
    default_csv_path = os.path.join(benchmark_dir, "dataset_csv", "brca.csv")
    pd.DataFrame(rows).to_csv(default_csv_path, index=False)

    # Create splits using standard strategy
    strategy_cfg = registries.strategy_registry["standard"]
    splits_dir = os.path.join(benchmark_dir, "splits", "standard", "brca")
    create_strategy_splits(default_csv_path, splits_dir, strategy_cfg, n_splits=3, seed=42)

    return default_csv_path


# ---------------------------------------------------------------------------
# Feature analysis
# ---------------------------------------------------------------------------


class TestAnalyzeFeatures:
    def test_correct_dimension(self, h5_features_dir):
        stats = _analyze_features(
            h5_features_dir, [f"slide_{i:05d}" for i in range(10)], 768
        )
        assert stats["feature_dimension"] == 768

    def test_patch_statistics(self, h5_features_dir):
        stats = _analyze_features(
            h5_features_dir, [f"slide_{i:05d}" for i in range(10)], 768
        )
        ps = stats["num_patches_per_slide"]
        assert ps["min"] >= 50
        assert ps["max"] <= 200
        assert ps["mean"] > 0
        assert ps["median"] > 0

    def test_max_seq_length(self, h5_features_dir):
        stats = _analyze_features(
            h5_features_dir, [f"slide_{i:05d}" for i in range(10)], 768
        )
        expected = int(stats["num_patches_per_slide"]["median"] * 0.5)
        assert stats["recommended_max_seq_length"] == expected

    def test_max_seq_length_uncapped(self, tmp_path):
        """Matches nnMIL experiment_planner.py:129 — int(median * 0.5), no cap."""
        h5_dir = tmp_path / "features_big"
        h5_dir.mkdir()
        for i in range(5):
            with h5py.File(h5_dir / f"slide_{i:05d}.h5", "w") as f:
                f.create_dataset("features", data=np.zeros((15000, 768), dtype=np.float32))
                f.create_dataset("coords", data=np.zeros((15000, 2), dtype=np.int32))

        stats = _analyze_features(str(h5_dir), [f"slide_{i:05d}" for i in range(5)], 768)
        # median=15000, 0.5*15000=7500 -> no cap
        assert stats["recommended_max_seq_length"] == 7500


# ---------------------------------------------------------------------------
# Training config generation
# ---------------------------------------------------------------------------


class TestGenerateTrainingConfig:
    def test_feature_dimension_preserved(self):
        stats = {
            "feature_dimension": 768,
            "num_patches_per_slide": {"median": 400},
        }
        config = _generate_training_config(stats, n_samples=200)
        assert config["feature_dimension"] == 768

    def test_small_dataset_smaller_batch(self):
        stats = {
            "feature_dimension": 768,
            "num_patches_per_slide": {"median": 400},
        }
        config = _generate_training_config(stats, n_samples=50)
        assert config["batch_size"] == 16

    def test_large_dataset_batch_matches_planner(self):
        """experiment_planner.py:639-657 — n_train>800 → batch_size=32."""
        stats = {
            "feature_dimension": 768,
            "num_patches_per_slide": {"median": 400},
        }
        config = _generate_training_config(stats, n_samples=2000)
        assert config["batch_size"] == 32

    def test_mid_dataset_batch_matches_planner(self):
        """experiment_planner.py:643 — 200<=n_train<=800 → 24 if <400 else 32."""
        stats = {
            "feature_dimension": 768,
            "num_patches_per_slide": {"median": 400},
        }
        cfg_400 = _generate_training_config(stats, n_samples=400)
        assert cfg_400["batch_size"] == 24
        cfg_600 = _generate_training_config(stats, n_samples=600)
        assert cfg_600["batch_size"] == 32

    def test_max_seq_length_uncapped(self):
        """experiment_planner.py:129 — int(median * 0.5) with no upper cap."""
        stats = {
            "feature_dimension": 768,
            "num_patches_per_slide": {"median": 13236},
        }
        config = _generate_training_config(stats, n_samples=200)
        assert config["max_seq_length"] == 6618

    def test_batch_size_independent_of_feat_dim(self):
        """Planner clamps to [16,48] regardless of encoder dim."""
        stats_virchow2 = {
            "feature_dimension": 2560,
            "num_patches_per_slide": {"median": 13236},
        }
        stats_conch = {
            "feature_dimension": 768,
            "num_patches_per_slide": {"median": 13236},
        }
        cfg_virchow2 = _generate_training_config(stats_virchow2, n_samples=200)
        cfg_conch = _generate_training_config(stats_conch, n_samples=200)
        assert cfg_virchow2["batch_size"] == cfg_conch["batch_size"]
        assert 16 <= cfg_virchow2["batch_size"] <= 48


# ---------------------------------------------------------------------------
# Splits -> nnMIL format conversion
# ---------------------------------------------------------------------------


class TestLoadSplitsAsNnmilFormat:
    def test_correct_fold_structure(self, task_csv_with_splits, benchmark_dir):
        task_df = pd.read_csv(task_csv_with_splits)
        label_dict = {"neg": 0, "pos": 1}
        splits_dir = os.path.join(benchmark_dir, "splits", "standard", "brca")

        data_splits = _load_splits_as_nnmil_format(
            splits_dir, task_df, label_dict, n_splits=3
        )
        assert "fold_0" in data_splits
        assert "fold_1" in data_splits
        assert "fold_2" in data_splits

    def test_each_fold_has_train_val_test(self, task_csv_with_splits, benchmark_dir):
        task_df = pd.read_csv(task_csv_with_splits)
        label_dict = {"neg": 0, "pos": 1}
        splits_dir = os.path.join(benchmark_dir, "splits", "standard", "brca")

        data_splits = _load_splits_as_nnmil_format(
            splits_dir, task_df, label_dict, n_splits=3
        )
        for fold_key, fold_data in data_splits.items():
            assert "train" in fold_data
            assert "val" in fold_data
            assert "test" in fold_data
            for split_name in ("train", "val", "test"):
                assert "slide_ids" in fold_data[split_name]
                assert "slide_info" in fold_data[split_name]

    def test_slide_info_has_correct_keys(self, task_csv_with_splits, benchmark_dir):
        task_df = pd.read_csv(task_csv_with_splits)
        label_dict = {"neg": 0, "pos": 1}
        splits_dir = os.path.join(benchmark_dir, "splits", "standard", "brca")

        data_splits = _load_splits_as_nnmil_format(
            splits_dir, task_df, label_dict, n_splits=3
        )
        info = data_splits["fold_0"]["train"]["slide_info"][0]
        assert "slide_id" in info
        assert "patient_id" in info
        assert "label" in info
        assert isinstance(info["label"], int)

    def test_no_patient_leakage_across_splits(self, task_csv_with_splits, benchmark_dir):
        task_df = pd.read_csv(task_csv_with_splits)
        label_dict = {"neg": 0, "pos": 1}
        splits_dir = os.path.join(benchmark_dir, "splits", "standard", "brca")

        data_splits = _load_splits_as_nnmil_format(
            splits_dir, task_df, label_dict, n_splits=3
        )
        for fold_key, fold_data in data_splits.items():
            train_ids = set(fold_data["train"]["slide_ids"])
            val_ids = set(fold_data["val"]["slide_ids"])
            test_ids = set(fold_data["test"]["slide_ids"])
            assert len(train_ids & val_ids) == 0, f"train/val leak in {fold_key}"
            assert len(train_ids & test_ids) == 0, f"train/test leak in {fold_key}"
            assert len(val_ids & test_ids) == 0, f"val/test leak in {fold_key}"


# ---------------------------------------------------------------------------
# Full nnMIL preparation
# ---------------------------------------------------------------------------


class TestPrepareNnmilExperiment:
    def test_creates_all_artifacts(
        self, benchmark_dir, h5_features_dir, task_csv_with_splits, registries,
    ):
        features_base_dir = os.path.dirname(h5_features_dir)
        plan_path = prepare_nnmil_experiment(
            benchmark_dir=benchmark_dir,
            task_name="brca",
            encoder_key="conch_v15",
            strategy="standard",
            label_col="BRCA_predict_label",
            label_dict={"neg": 0, "pos": 1},
            embed_dim=768,
            features_base_dir=features_base_dir,
            seed=42,
            n_splits=3,
        )
        dataset_dir = os.path.dirname(plan_path)
        assert os.path.exists(os.path.join(dataset_dir, "dataset.json"))
        assert os.path.exists(os.path.join(dataset_dir, "dataset.csv"))
        assert os.path.exists(plan_path)

    def test_plan_has_correct_structure(
        self, benchmark_dir, h5_features_dir, task_csv_with_splits, registries,
    ):
        features_base_dir = os.path.dirname(h5_features_dir)
        plan_path = prepare_nnmil_experiment(
            benchmark_dir=benchmark_dir,
            task_name="brca",
            encoder_key="conch_v15",
            strategy="standard",
            label_col="BRCA_predict_label",
            label_dict={"neg": 0, "pos": 1},
            embed_dim=768,
            features_base_dir=features_base_dir,
            seed=42,
            n_splits=3,
        )
        with open(plan_path) as f:
            plan = json.load(f)

        assert plan["task_type"] == "classification"
        # n_splits=3 was passed; evaluation_setting reflects actual fold count
        assert plan["evaluation_setting"] == "3fold"
        assert "feature_statistics" in plan
        assert "data_splits" in plan
        assert "training_configuration" in plan
        assert plan["random_seed"] == 42

    def test_idempotent(
        self, benchmark_dir, h5_features_dir, task_csv_with_splits, registries,
    ):
        features_base_dir = os.path.dirname(h5_features_dir)
        kwargs = dict(
            benchmark_dir=benchmark_dir,
            task_name="brca",
            encoder_key="conch_v15",
            strategy="standard",
            label_col="BRCA_predict_label",
            label_dict={"neg": 0, "pos": 1},
            embed_dim=768,
            features_base_dir=features_base_dir,
            seed=42,
            n_splits=3,
        )
        path1 = prepare_nnmil_experiment(**kwargs)
        path2 = prepare_nnmil_experiment(**kwargs)
        assert path1 == path2


# ---------------------------------------------------------------------------
# nnMIL metric normalization
# ---------------------------------------------------------------------------


class TestNormalizeNnmilMetrics:
    def test_maps_known_metrics(self):
        raw = {
            "test_test/acc": 0.85,
            "test_test/bacc": 0.82,
            "test_test/auroc": 0.90,
            "test_test/weighted_f1": 0.83,
            "test_test/kappa": 0.65,
        }
        result = normalize_nnmil_metrics(raw, split="test")
        assert result["accuracy"] == 0.85
        assert result["balanced_accuracy"] == 0.82
        assert result["auc_roc"] == 0.90
        assert result["f1"] == 0.83
        assert result["kappa"] == 0.65

    def test_adds_missing_sensitivity_specificity(self):
        raw = {"test_test/bacc": 0.82}
        result = normalize_nnmil_metrics(raw, split="test")
        assert "sensitivity" in result
        assert "specificity" in result
        assert np.isnan(result["sensitivity"])
        assert np.isnan(result["specificity"])
