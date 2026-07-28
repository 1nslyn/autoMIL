"""Integration test: end-to-end mini benchmark with synthetic data.

Creates fake H5 features, mapping CSV, runs data prep + training for a small
grid (1 task x 1 encoder x 1 model x 2 folds), and verifies the full output.
"""

import json
import os

import h5py
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from autobench.pipeline.config import (
    BenchmarkConfig,
    ExperimentConfig,
    ModelConfig,
    Registries,
    TaskConfig,
    TrainConfig,
    build_registries,
    generate_all_experiments,
)
from autobench.pipeline.evaluate import compute_extended_metrics
from autobench.pipeline.orchestrator import aggregate_results, run_benchmark
from autobench.pipeline.clam.prepare import convert_h5_to_pt
from autobench.pipeline.prepare import create_task_csv
from autobench.pipeline.splits import create_strategy_splits
from autobench.pipeline.clam.runner import run_experiment
from autobench.config import DatasetConfig, StrategyDef, TaskDef
from _helpers import make_test_ds


# ---------------------------------------------------------------------------
# Fixtures: synthetic data
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_benchmark(tmp_path, monkeypatch):
    """Build a complete synthetic benchmark environment.

    Creates:
      - 30 fake slides with H5 features (dim=64, small for speed)
      - A mapping CSV with BRCA labels (15 neg, 15 pos)
      - A DatasetConfig with a 'fake_enc' encoder (dim=64)
    """
    n_slides = 30
    feat_dim = 64

    # 1. Mapping CSV
    rows = []
    for i in range(n_slides):
        rows.append({
            "new_name": f"slide_{i:05d}.svs",
            "status": "mapped_unique_case_id",
            "primary_hospital": "UHN",
            "primary_case_id": f"K{i:03d}",
            "BRCA_predict_label": i % 2,
            "HRD_label": pd.NA,  # skip HRD to keep test fast
        })
    mapping_csv = str(tmp_path / "mapping.csv")
    pd.DataFrame(rows).to_csv(mapping_csv, index=False)

    # 2. H5 feature files
    feat_dir = tmp_path / "features" / "features_fake_enc"
    feat_dir.mkdir(parents=True)
    for i in range(n_slides):
        n_patches = np.random.RandomState(i).randint(20, 60)
        with h5py.File(feat_dir / f"slide_{i:05d}.h5", "w") as f:
            f.create_dataset(
                "features",
                data=np.random.RandomState(i).randn(n_patches, feat_dim).astype(np.float32),
            )
            f.create_dataset("coords", data=np.zeros((n_patches, 2), dtype=np.int64))

    # 3. Build a DatasetConfig for this synthetic benchmark
    benchmark_dir = str(tmp_path / "benchmark")
    ds = make_test_ds(
        data_root=str(tmp_path),
        wsi_dir=str(tmp_path / "wsi"),
        mapping_csv=mapping_csv,
        output_dir=str(tmp_path / "output"),
        benchmark_dir=benchmark_dir,
        features_base_dir=str(tmp_path / "features"),
        encoder_dims={"fake_enc": feat_dim},
        encoder_models={"test/fake": "fake_enc"},
        tasks={
            "brca": TaskDef(
                name="brca",
                label_col="BRCA_predict_label",
                label_map={0: "neg", 1: "pos"},
                n_classes=2,
            ),
        },
    )
    registries = build_registries(ds)

    yield {
        "tmp_path": tmp_path,
        "benchmark_dir": benchmark_dir,
        "mapping_csv": mapping_csv,
        "features_base_dir": str(tmp_path / "features"),
        "feat_dim": feat_dim,
        "n_slides": n_slides,
        "ds": ds,
        "registries": registries,
    }


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestDataPrepIntegration:
    """Test data preparation pipeline on synthetic data."""

    def test_full_prep_flow(self, synthetic_benchmark):
        sb = synthetic_benchmark
        ds = sb["ds"]
        # Create task CSV
        csv_path = os.path.join(sb["benchmark_dir"], "dataset_csv", "brca.csv")
        df = create_task_csv(
            sb["mapping_csv"], csv_path, "BRCA_predict_label",
            {0: "neg", 1: "pos"}, ds,
        )
        assert len(df) == sb["n_slides"]
        assert set(df["label"]) == {"neg", "pos"}

        # Create splits (under strategy subdir)
        splits_dir = os.path.join(sb["benchmark_dir"], "splits", "standard", "brca")
        paths = create_strategy_splits(csv_path, splits_dir, n_splits=2, seed=42)
        assert len(paths) == 2

        # Convert features
        h5_dir = os.path.join(sb["features_base_dir"], "features_fake_enc")
        pt_dir = os.path.join(sb["benchmark_dir"], "features", "fake_enc")
        slide_ids = df["slide_id"].tolist()
        n = convert_h5_to_pt(h5_dir, pt_dir, "fake_enc", slide_ids)
        assert n == sb["n_slides"]

        # Verify PT files
        for sid in slide_ids:
            pt = torch.load(os.path.join(pt_dir, "pt_files", f"{sid}.pt"), weights_only=True)
            assert pt.ndim == 2
            assert pt.shape[1] == sb["feat_dim"]


class TestSingleExperimentIntegration:
    """Test a single experiment (all folds) on synthetic data."""

    def test_run_experiment(self, synthetic_benchmark):
        sb = synthetic_benchmark
        ds = sb["ds"]

        # Prep data
        csv_path = os.path.join(sb["benchmark_dir"], "dataset_csv", "brca.csv")
        df = create_task_csv(
            sb["mapping_csv"], csv_path, "BRCA_predict_label",
            {0: "neg", 1: "pos"}, ds,
        )
        splits_dir = os.path.join(sb["benchmark_dir"], "splits", "standard", "brca")
        create_strategy_splits(csv_path, splits_dir, n_splits=2, seed=42)
        slide_ids = df["slide_id"].tolist()
        h5_dir = os.path.join(sb["features_base_dir"], "features_fake_enc")
        pt_dir = os.path.join(sb["benchmark_dir"], "features", "fake_enc")
        convert_h5_to_pt(h5_dir, pt_dir, "fake_enc", slide_ids)

        # Configure experiment
        task = TaskConfig(name="brca", label_col="BRCA_predict_label",
                          label_dict={"neg": 0, "pos": 1}, n_classes=2)
        model = ModelConfig(model_type="clam_sb", B=4)
        train = TrainConfig(max_epochs=3, lr=1e-3, seed=42, patience=5, stop_epoch=0)
        exp_cfg = ExperimentConfig(
            task=task, encoder_key="fake_enc",
            embed_dim=sb["feat_dim"], model=model, train=train, n_folds=2,
            strategy="standard",
        )

        device = torch.device("cpu")
        summary = run_experiment(exp_cfg, sb["benchmark_dir"], device)

        # Verify summary structure
        assert summary["experiment_id"] == exp_cfg.experiment_id
        assert summary["task"] == "brca"
        assert summary["encoder"] == "fake_enc"
        assert summary["n_folds"] == 2
        assert summary["framework"] == "clam"
        assert summary["strategy"] == "standard"
        assert "test" in summary
        assert "val" in summary
        assert "per_fold_test" in summary
        assert len(summary["per_fold_test"]) == 2

        # Verify metrics are present
        for fold_m in summary["per_fold_test"]:
            assert "auc_roc" in fold_m
            assert "accuracy" in fold_m
            assert "sensitivity" in fold_m

        # Verify CI computation
        assert "mean" in summary["test"]["auc_roc"]
        assert "ci_low" in summary["test"]["auc_roc"]
        assert "ci_high" in summary["test"]["auc_roc"]

        # Verify output files --
        # results/{framework}/{strategy}/{task}/{encoder}/{model}/s{seed}/ (CR-5b)
        results_dir = os.path.join(
            sb["benchmark_dir"], "results", exp_cfg.results_subdir,
        )
        assert results_dir.endswith(os.path.join("clam_sb", f"s{exp_cfg.train.seed}"))
        assert os.path.exists(os.path.join(results_dir, "summary.json"))
        assert os.path.exists(os.path.join(results_dir, "config.json"))
        for fold in range(2):
            fold_dir = os.path.join(results_dir, f"fold_{fold}")
            assert os.path.exists(os.path.join(fold_dir, "predictions.csv"))
            assert os.path.exists(os.path.join(fold_dir, "metrics.json"))
            assert os.path.exists(os.path.join(fold_dir, f"s_{fold}_checkpoint.pt"))

            # Verify predictions CSV
            pred_df = pd.read_csv(os.path.join(fold_dir, "predictions.csv"))
            assert set(pred_df.columns) == {"slide_id", "y_true", "y_prob_0", "y_prob_1", "y_hat"}
            assert pred_df["y_prob_0"].between(0, 1).all()
            assert pred_df["y_prob_1"].between(0, 1).all()
            assert set(pred_df["y_hat"].unique()).issubset({0, 1})


class TestResumability:
    """Test that completed experiments/folds are skipped on re-run."""

    def test_fold_resume(self, synthetic_benchmark):
        sb = synthetic_benchmark
        ds = sb["ds"]

        # Prep
        csv_path = os.path.join(sb["benchmark_dir"], "dataset_csv", "brca.csv")
        df = create_task_csv(
            sb["mapping_csv"], csv_path, "BRCA_predict_label",
            {0: "neg", 1: "pos"}, ds,
        )
        create_strategy_splits(csv_path,
                                 os.path.join(sb["benchmark_dir"], "splits", "standard", "brca"),
                                 n_splits=2, seed=42)
        slide_ids = df["slide_id"].tolist()
        convert_h5_to_pt(
            os.path.join(sb["features_base_dir"], "features_fake_enc"),
            os.path.join(sb["benchmark_dir"], "features", "fake_enc"),
            "fake_enc", slide_ids,
        )

        task = TaskConfig(name="brca", label_col="BRCA_predict_label",
                          label_dict={"neg": 0, "pos": 1}, n_classes=2)
        model = ModelConfig(model_type="mil", B=4)
        train = TrainConfig(max_epochs=2, lr=1e-3, seed=42, patience=5, stop_epoch=0)
        exp_cfg = ExperimentConfig(
            task=task, encoder_key="fake_enc",
            embed_dim=sb["feat_dim"], model=model, train=train, n_folds=2,
            strategy="standard",
        )

        device = torch.device("cpu")

        # First run
        summary1 = run_experiment(exp_cfg, sb["benchmark_dir"], device)
        # Second run -- should load from cache
        summary2 = run_experiment(exp_cfg, sb["benchmark_dir"], device)

        # Results should be identical (loaded from disk)
        assert summary1["per_fold_test"] == summary2["per_fold_test"]


class TestAggregateResults:
    def test_dataframe_format(self):
        summaries = [
            {
                "experiment_id": "brca__fake__clam_sb__s42",
                "task": "brca", "encoder": "fake", "model_type": "clam_sb",
                "embed_dim": 64, "n_folds": 2, "seed": 42,
                "framework": "clam", "strategy": "standard",
                "test": {
                    "auc_roc": {"mean": 0.8, "std": 0.1, "ci_low": 0.5, "ci_high": 1.0},
                    "accuracy": {"mean": 0.7, "std": 0.1, "ci_low": 0.4, "ci_high": 1.0},
                },
                "val": {
                    "auc_roc": {"mean": 0.75, "std": 0.1, "ci_low": 0.5, "ci_high": 1.0},
                    "accuracy": {"mean": 0.65, "std": 0.1, "ci_low": 0.4, "ci_high": 0.9},
                },
            }
        ]
        df = aggregate_results(summaries)
        assert len(df) == 1
        assert "test_auc_roc_mean" in df.columns
        assert "val_accuracy_ci_high" in df.columns
        assert df.iloc[0]["task"] == "brca"
        assert df.iloc[0]["test_auc_roc_mean"] == 0.8
