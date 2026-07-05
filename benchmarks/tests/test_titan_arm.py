"""Tests for the TITAN slide-encoder arm (frozen-embedding linear probe)."""

import json
import os

import h5py
import numpy as np
import pandas as pd
import pytest

from autobench.pipeline.config import (
    Framework,
    ModelConfig,
    TaskConfig,
    TrainConfig,
    ExperimentConfig,
    build_registries,
)
from autobench.pipeline.orchestrator import (
    _prepare_titan_plans,
    _run_single_experiment_dispatch,
)
from autobench.pipeline.splits import create_strategy_splits
from autobench.pipeline.titan.dataset import TitanSlideDataset, build_split_dataset
from autobench.pipeline.titan.model import TitanLinearProbe
from autobench.pipeline.titan.prepare import (
    _read_slide_embedding_dim,
    prepare_titan_experiment,
    validate_titan_features,
)
from autobench.pipeline.titan.runner import run_titan_experiment
from autobench.pipeline.titan.train import train_titan_fold
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
    """Minimal benchmark directory structure."""
    bd = str(tmp_path / "benchmark")
    os.makedirs(os.path.join(bd, "dataset_csv"), exist_ok=True)
    os.makedirs(os.path.join(bd, "splits"), exist_ok=True)
    return bd


def _write_titan_features(features_dir: str, slide_ids: list[str], dim: int = 768) -> None:
    os.makedirs(features_dir, exist_ok=True)
    rng = np.random.default_rng(42)
    for sid in slide_ids:
        with h5py.File(os.path.join(features_dir, f"{sid}.h5"), "w") as f:
            # shape [1, D] -- a leading singleton batch dim, as commonly
            # emitted by slide encoders (design spec §7).
            f.create_dataset(
                "slide_feature",
                data=rng.standard_normal((1, dim)).astype(np.float32),
            )


@pytest.fixture
def task_csv(benchmark_dir):
    """40 slides / 40 cases, binary label, enough for 2-fold patient CV."""
    rows = []
    for i in range(40):
        rows.append({
            "case_id": f"P{i:03d}",
            "slide_id": f"slide_{i:05d}",
            "label": "neg" if i % 2 == 0 else "pos",
        })
    csv_path = os.path.join(benchmark_dir, "dataset_csv", "brca.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def titan_features_dir(benchmark_dir, task_csv):
    """features_titan/<slide_id>.h5, one 768-d vector per slide in task_csv."""
    features_base_dir = benchmark_dir  # features_titan/ lives directly under it
    features_dir = os.path.join(features_base_dir, "features_titan")
    task_df = pd.read_csv(task_csv)
    _write_titan_features(features_dir, task_df["slide_id"].tolist(), dim=768)
    return features_dir


@pytest.fixture
def splits_2fold(benchmark_dir, task_csv, registries):
    """2-fold patient-stratified splits under splits/standard/brca/."""
    strategy_cfg = registries.strategy_registry["standard"]
    splits_dir = os.path.join(benchmark_dir, "splits", "standard", "brca")
    create_strategy_splits(task_csv, splits_dir, strategy_cfg, n_splits=2, seed=42)
    return splits_dir


@pytest.fixture
def titan_exp_cfg():
    """A minimal 2-fold TITAN ExperimentConfig, embed_dim matching the fixture."""
    return ExperimentConfig(
        task=TaskConfig(
            name="brca",
            label_col="BRCA_predict_label",
            label_dict={"neg": 0, "pos": 1},
            n_classes=2,
        ),
        encoder_key="titan",
        embed_dim=768,
        model=ModelConfig(model_type="titan"),
        train=TrainConfig(max_epochs=3, patience=2, seed=42),
        n_folds=2,
        framework=Framework.TITAN,
        strategy="standard",
    )


# ---------------------------------------------------------------------------
# prepare.py -- dimension detection + missing-feature fail-fast
# ---------------------------------------------------------------------------


class TestReadSlideEmbeddingDim:
    def test_reads_dim_from_leading_singleton_shape(self, tmp_path):
        h5_path = tmp_path / "slide_00000.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("slide_feature", data=np.zeros((1, 768), dtype=np.float32))
        assert _read_slide_embedding_dim(str(h5_path)) == 768

    def test_reads_dim_from_flat_shape(self, tmp_path):
        h5_path = tmp_path / "slide_00000.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("slide_feature", data=np.zeros(768, dtype=np.float32))
        assert _read_slide_embedding_dim(str(h5_path)) == 768

    def test_reads_different_dim_512(self, tmp_path):
        """Verifies the dim is truly READ, not hard-coded to 768."""
        h5_path = tmp_path / "slide_00000.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("slide_feature", data=np.zeros((1, 512), dtype=np.float32))
        assert _read_slide_embedding_dim(str(h5_path)) == 512

    def test_accepts_features_key_fallback(self, tmp_path):
        h5_path = tmp_path / "slide_00000.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("features", data=np.zeros((1, 768), dtype=np.float32))
        assert _read_slide_embedding_dim(str(h5_path)) == 768

    def test_missing_key_raises(self, tmp_path):
        h5_path = tmp_path / "slide_00000.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("unrelated", data=np.zeros((1, 768), dtype=np.float32))
        with pytest.raises(ValueError, match="no recognized dataset key"):
            _read_slide_embedding_dim(str(h5_path))


class TestValidateTitanFeatures:
    def test_detects_dim_768(self, task_csv, titan_features_dir):
        embed_dim = validate_titan_features(task_csv, titan_features_dir)
        assert embed_dim == 768

    def test_detects_dim_512(self, benchmark_dir, task_csv):
        """A different fixture (512-d) must yield 512, not the 768 default."""
        features_dir = os.path.join(benchmark_dir, "features_titan_512")
        task_df = pd.read_csv(task_csv)
        _write_titan_features(features_dir, task_df["slide_id"].tolist(), dim=512)

        embed_dim = validate_titan_features(task_csv, features_dir)
        assert embed_dim == 512

    def test_missing_features_raises_file_not_found(self, benchmark_dir, task_csv):
        empty_features_dir = os.path.join(benchmark_dir, "features_titan_empty")
        os.makedirs(empty_features_dir, exist_ok=True)
        with pytest.raises(FileNotFoundError, match="Missing TITAN slide features"):
            validate_titan_features(task_csv, empty_features_dir)

    def test_partial_missing_features_raises(self, benchmark_dir, task_csv):
        task_df = pd.read_csv(task_csv)
        # Only write features for half the slides.
        partial_ids = task_df["slide_id"].tolist()[:20]
        features_dir = os.path.join(benchmark_dir, "features_titan_partial")
        _write_titan_features(features_dir, partial_ids, dim=768)

        with pytest.raises(FileNotFoundError, match="Missing TITAN slide features"):
            validate_titan_features(task_csv, features_dir)


class TestPrepareTitanExperiment:
    def test_writes_manifest_with_detected_dim(self, benchmark_dir, task_csv, titan_features_dir):
        manifest = prepare_titan_experiment(
            benchmark_dir=benchmark_dir,
            task_name="brca",
            features_base_dir=benchmark_dir,
        )
        assert manifest["embed_dim"] == 768
        assert manifest["n_slides"] == 40
        assert manifest["features_dir"] == titan_features_dir

        manifest_path = os.path.join(benchmark_dir, "titan", "brca", "manifest.json")
        assert os.path.exists(manifest_path)
        with open(manifest_path) as f:
            on_disk = json.load(f)
        assert on_disk == manifest


# ---------------------------------------------------------------------------
# dataset.py
# ---------------------------------------------------------------------------


class TestBuildSplitDataset:
    def test_returns_embedding_and_int_label(
        self, benchmark_dir, task_csv, titan_features_dir, splits_2fold,
    ):
        task_df = pd.read_csv(task_csv)
        split_csv = os.path.join(splits_2fold, "splits_0.csv")

        train_ds = build_split_dataset(
            split_csv, "train", task_df, {"neg": 0, "pos": 1}, titan_features_dir,
        )
        assert len(train_ds) > 0
        embedding, label = train_ds[0]
        assert embedding.shape == (768,)
        assert isinstance(label, int)
        assert label in (0, 1)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            TitanSlideDataset(["a", "b"], [0], features_dir="/tmp/nonexistent")

    def test_missing_slide_column_raises(self, benchmark_dir, task_csv, titan_features_dir, splits_2fold):
        task_df = pd.read_csv(task_csv)
        split_csv = os.path.join(splits_2fold, "splits_0.csv")
        with pytest.raises(ValueError, match="no column"):
            build_split_dataset(
                split_csv, "nonexistent_split", task_df, {"neg": 0, "pos": 1},
                titan_features_dir,
            )


# ---------------------------------------------------------------------------
# model.py
# ---------------------------------------------------------------------------


class TestTitanLinearProbe:
    def test_forward_shape(self):
        import torch

        model = TitanLinearProbe(embed_dim=768, num_classes=2)
        x = torch.randn(4, 768)
        logits = model(x)
        assert logits.shape == (4, 2)

    def test_gradient_flows(self):
        import torch

        model = TitanLinearProbe(embed_dim=768, num_classes=2)
        x = torch.randn(4, 768)
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        assert model.linear.weight.grad is not None
        assert torch.any(model.linear.weight.grad != 0)


# ---------------------------------------------------------------------------
# train.py -- one fold trains and writes a valid metrics.json
# ---------------------------------------------------------------------------


class TestTrainTitanFold:
    def test_writes_valid_metrics_json(
        self, tmp_path, benchmark_dir, task_csv, titan_features_dir, splits_2fold,
        titan_exp_cfg,
    ):
        task_df = pd.read_csv(task_csv)
        split_csv = os.path.join(splits_2fold, "splits_0.csv")
        label_dict = {"neg": 0, "pos": 1}

        train_ds = build_split_dataset(split_csv, "train", task_df, label_dict, titan_features_dir)
        val_ds = build_split_dataset(split_csv, "val", task_df, label_dict, titan_features_dir)
        test_ds = build_split_dataset(split_csv, "test", task_df, label_dict, titan_features_dir)

        results_dir = str(tmp_path / "results")
        result = train_titan_fold(
            titan_exp_cfg, train_ds, val_ds, test_ds,
            fold=0, results_dir=results_dir, device="cpu",
        )

        assert result["fold"] == 0
        for split_key in ("test_metrics", "val_metrics"):
            metrics = result[split_key]
            assert "auc_roc" in metrics
            assert "balanced_accuracy" in metrics
            assert "accuracy" in metrics
            assert "f1" in metrics

        metrics_path = os.path.join(results_dir, "fold_0", "metrics.json")
        assert os.path.exists(metrics_path)
        with open(metrics_path) as f:
            on_disk = json.load(f)
        assert on_disk["fold"] == 0

    def test_resumes_from_disk(
        self, tmp_path, benchmark_dir, task_csv, titan_features_dir, splits_2fold,
        titan_exp_cfg,
    ):
        task_df = pd.read_csv(task_csv)
        split_csv = os.path.join(splits_2fold, "splits_0.csv")
        label_dict = {"neg": 0, "pos": 1}
        train_ds = build_split_dataset(split_csv, "train", task_df, label_dict, titan_features_dir)
        val_ds = build_split_dataset(split_csv, "val", task_df, label_dict, titan_features_dir)
        test_ds = build_split_dataset(split_csv, "test", task_df, label_dict, titan_features_dir)

        results_dir = str(tmp_path / "results")
        first = train_titan_fold(
            titan_exp_cfg, train_ds, val_ds, test_ds,
            fold=0, results_dir=results_dir, device="cpu",
        )
        second = train_titan_fold(
            titan_exp_cfg, train_ds, val_ds, test_ds,
            fold=0, results_dir=results_dir, device="cpu",
        )
        assert first == second


# ---------------------------------------------------------------------------
# runner.py -- full experiment (all folds) -> summary.json
# ---------------------------------------------------------------------------


class TestRunTitanExperiment:
    def test_produces_valid_summary(
        self, benchmark_dir, task_csv, titan_features_dir, splits_2fold, titan_exp_cfg,
    ):
        prepare_titan_experiment(
            benchmark_dir=benchmark_dir,
            task_name="brca",
            features_base_dir=benchmark_dir,
        )

        summary = run_titan_experiment(titan_exp_cfg, benchmark_dir, device="cpu")

        assert summary["framework"] == "titan"
        assert summary["task"] == "brca"
        assert summary["n_folds"] == 2
        assert len(summary["per_fold_test"]) == 2
        assert len(summary["per_fold_val"]) == 2
        for split_key in ("test", "val"):
            for metric_name in ("auc_roc", "balanced_accuracy"):
                stat = summary[split_key][metric_name]
                assert "mean" in stat
                assert "std" in stat
                assert "ci_low" in stat
                assert "ci_high" in stat

        summary_path = os.path.join(
            benchmark_dir, "results", titan_exp_cfg.results_subdir, "summary.json",
        )
        assert os.path.exists(summary_path)
        with open(summary_path) as f:
            on_disk = json.load(f)
        assert on_disk["experiment_id"] == titan_exp_cfg.experiment_id

    def test_missing_manifest_raises(self, benchmark_dir, task_csv, splits_2fold, titan_exp_cfg):
        """No prepare step run -> fail fast rather than crash deep in training."""
        with pytest.raises(FileNotFoundError, match="TITAN manifest not found"):
            run_titan_experiment(titan_exp_cfg, benchmark_dir, device="cpu")


# ---------------------------------------------------------------------------
# orchestrator.py -- _prepare_titan_plans + dispatch
# ---------------------------------------------------------------------------


class TestPrepareTitanPlansOrchestrator:
    def test_prepares_manifest_for_titan_experiments_only(
        self, benchmark_dir, task_csv, titan_features_dir, registries,
    ):
        from autobench.pipeline.config import BenchmarkConfig

        titan_exp = ExperimentConfig(
            task=TaskConfig(
                name="brca", label_col="BRCA_predict_label",
                label_dict={"neg": 0, "pos": 1}, n_classes=2,
            ),
            encoder_key="titan",
            embed_dim=768,
            model=ModelConfig(model_type="titan"),
            train=TrainConfig(),
            n_folds=2,
            framework=Framework.TITAN,
            strategy="standard",
        )
        non_titan_exp = ExperimentConfig(
            task=TaskConfig(
                name="brca", label_col="BRCA_predict_label",
                label_dict={"neg": 0, "pos": 1}, n_classes=2,
            ),
            encoder_key="conch_v15",
            embed_dim=768,
            model=ModelConfig(model_type="clam_mb"),
            train=TrainConfig(),
            n_folds=2,
            framework=Framework.CLAM,
            strategy="standard",
        )

        cfg = BenchmarkConfig.from_dataset_config(
            make_test_ds(), benchmark_dir=benchmark_dir, features_base_dir=benchmark_dir,
        )

        _prepare_titan_plans(cfg, [titan_exp, non_titan_exp], registries=registries)

        manifest_path = os.path.join(benchmark_dir, "titan", "brca", "manifest.json")
        assert os.path.exists(manifest_path)
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert manifest["embed_dim"] == 768


class TestDispatchRoutesToTitan:
    def test_dispatch_runs_titan_instead_of_raising(
        self, benchmark_dir, task_csv, titan_features_dir, splits_2fold, titan_exp_cfg,
    ):
        import torch

        prepare_titan_experiment(
            benchmark_dir=benchmark_dir,
            task_name="brca",
            features_base_dir=benchmark_dir,
        )

        summary = _run_single_experiment_dispatch(
            titan_exp_cfg, benchmark_dir, torch.device("cpu"),
        )
        assert summary["framework"] == "titan"
        assert summary["experiment_id"] == titan_exp_cfg.experiment_id
