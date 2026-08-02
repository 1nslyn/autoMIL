"""DATA-ID (audit 2026-07-23): dataset identity threaded through the results
pipeline.

Defect: dataset identity was recorded in NO results artifact.
``ExperimentConfig`` had no ``dataset`` field, so ``summary.json`` and
``aggregated/*.csv`` carried task/encoder/model/framework/seed but never
which cohort produced them -- dataset existed only as a filesystem path.
Every planned figure is cross-cohort (5 cohorts), so this blocked all of
them.

Covers:
  (a) ``ExperimentConfig.dataset`` exists and defaults to ``""``;
  (b) ``BenchmarkConfig.from_dataset_config`` populates it from
      ``DatasetConfig.name``;
  (c) ``generate_all_experiments`` stamps it on every generated experiment
      (reading it from ``cfg.dataset``, not a hardcoded source);
  (d) every one of the five runners (clam/nnmil/abmil/dtfd/titan) carries it
      into the ``summary.json`` dict it writes, on tiny CPU fixtures reusing
      the patterns established in test_abmil_arm.py / test_dtfd_arm.py /
      test_titan_arm.py / test_benchmark_integration.py. nnMIL's real
      trainer self-configures ~100 epochs (impractical for CI, per
      test_instrumentation.py's precedent), so its heavy vendored trainer is
      mocked the same way that file does -- the runner (and its
      ``dataset`` stamping) still executes for real;
  (e) ``aggregate_results`` produces a ``dataset`` column.
"""

from __future__ import annotations

import json
import os

import h5py
import numpy as np
import pandas as pd
import pytest

from autobench.config import DatasetConfig, StrategyDef, TaskDef
from autobench.pipeline.config import (
    BenchmarkConfig,
    ExperimentConfig,
    Framework,
    ModelConfig,
    TaskConfig,
    TrainConfig,
    build_registries,
    generate_all_experiments,
)
from autobench.pipeline.orchestrator import aggregate_results
from _helpers import make_test_ds


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ds():
    return make_test_ds()


@pytest.fixture
def registries(ds):
    return build_registries(ds)


# ---------------------------------------------------------------------------
# (a) ExperimentConfig.dataset -- exists, defaults to "", round-trips
# ---------------------------------------------------------------------------


class TestExperimentConfigDatasetField:
    def _exp(self, registries, **kw):
        params = dict(
            task=registries.task_registry["brca"],
            encoder_key="conch_v15",
            embed_dim=768,
            model=registries.model_registry["clam_sb"],
            train=TrainConfig(seed=42),
        )
        params.update(kw)
        return ExperimentConfig(**params)

    def test_defaults_to_empty_string(self, registries):
        assert self._exp(registries).dataset == ""

    def test_accepts_explicit_dataset(self, registries):
        assert self._exp(registries, dataset="luad").dataset == "luad"

    def test_to_dict_includes_dataset(self, registries):
        d = self._exp(registries, dataset="luad").to_dict()
        assert d["dataset"] == "luad"

    def test_existing_constructions_without_dataset_still_work(self, registries):
        """Regression guard: omitting dataset= must not break callers (D-01 style)."""
        exp = self._exp(registries)
        assert exp.experiment_id == "clam__standard__brca__conch_v15__clam_sb__s42"


# ---------------------------------------------------------------------------
# (b) BenchmarkConfig.dataset -- defaults to "", populated from ds.name
# ---------------------------------------------------------------------------


class TestBenchmarkConfigDatasetField:
    def test_default_is_empty_string(self):
        assert BenchmarkConfig().dataset == ""

    def test_from_dataset_config_populates_dataset(self, ds):
        cfg = BenchmarkConfig.from_dataset_config(ds)
        assert cfg.dataset == ds.name == "test"

    def test_from_dataset_config_override_wins(self, ds):
        """Explicit override still wins, same as every other from_dataset_config field."""
        cfg = BenchmarkConfig.from_dataset_config(ds, dataset="overridden")
        assert cfg.dataset == "overridden"


# ---------------------------------------------------------------------------
# (c) generate_all_experiments stamps dataset on every experiment
# ---------------------------------------------------------------------------


class TestGenerateAllExperimentsStampsDataset:
    def _cfg(self, ds, **kw):
        params = dict(
            strategies=["standard"], tasks=["brca"], encoder_keys=["conch_v15"],
        )
        params.update(kw)
        return BenchmarkConfig.from_dataset_config(ds, **params)

    def test_every_experiment_carries_dataset_name(self, ds, registries):
        cfg = self._cfg(ds)
        exps = generate_all_experiments(cfg, registries)
        assert len(exps) > 0
        assert all(e.dataset == "test" for e in exps)

    def test_reads_from_cfg_not_hardcoded(self, ds, registries):
        """Overriding cfg.dataset must change what lands on every experiment --
        proves generate_all_experiments reads cfg.dataset rather than some
        other (e.g. hardcoded ds.name) source."""
        cfg = self._cfg(ds, dataset="")
        exps = generate_all_experiments(cfg, registries)
        assert len(exps) > 0
        assert all(e.dataset == "" for e in exps)

    def test_multi_framework_grid_all_stamped(self, ds, registries):
        cfg = self._cfg(ds, frameworks=[Framework.CLAM, Framework.NNMIL])
        exps = generate_all_experiments(cfg, registries)
        assert len(exps) > 0
        assert all(e.dataset == "test" for e in exps)


# ---------------------------------------------------------------------------
# (e) aggregate_results produces a `dataset` column
# ---------------------------------------------------------------------------


class TestAggregateResultsDatasetColumn:
    def _summary(self, **overrides):
        base = {
            "dataset": "luad",
            "experiment_id": "clam__standard__brca__fake__clam_sb__s42",
            "task": "brca", "encoder": "fake", "model_type": "clam_sb",
            "embed_dim": 64, "n_folds": 2, "seed": 42,
            "framework": "clam", "strategy": "standard",
            "test": {
                "auc_roc": {"mean": 0.8, "std": 0.1, "ci_low": 0.5, "ci_high": 1.0},
            },
            "val": {
                "auc_roc": {"mean": 0.75, "std": 0.1, "ci_low": 0.5, "ci_high": 1.0},
            },
        }
        base.update(overrides)
        return base

    def test_dataset_column_present_and_populated(self):
        df = aggregate_results([self._summary()])
        assert "dataset" in df.columns
        assert df.iloc[0]["dataset"] == "luad"

    def test_multi_dataset_rows_distinguished(self):
        df = aggregate_results([
            self._summary(dataset="luad", experiment_id="a"),
            self._summary(dataset="lgg", experiment_id="b"),
        ])
        assert set(df["dataset"]) == {"luad", "lgg"}

    def test_missing_dataset_key_defaults_to_empty_string(self):
        """Old summaries written before DATA-ID lack the key entirely --
        must not crash aggregate_results, must default to ""."""
        summary = self._summary()
        del summary["dataset"]
        df = aggregate_results([summary])
        assert df.iloc[0]["dataset"] == ""


# ---------------------------------------------------------------------------
# (d) Every runner's summary carries dataset -- tiny CPU fixtures
# ---------------------------------------------------------------------------


class TestClamRunnerCarriesDataset:
    def test_summary_has_dataset(self, tmp_path):
        torch = pytest.importorskip("torch")
        from autobench.pipeline.clam.prepare import convert_h5_to_pt
        from autobench.pipeline.clam.runner import run_experiment
        from autobench.pipeline.prepare import create_task_csv
        from autobench.pipeline.splits import create_strategy_splits

        n_slides = 30
        feat_dim = 64
        rows = []
        for i in range(n_slides):
            rows.append({
                "new_name": f"slide_{i:05d}.svs",
                "status": "mapped_unique_case_id",
                "primary_hospital": "UHN",
                "primary_case_id": f"K{i:03d}",
                "BRCA_predict_label": i % 2,
                "HRD_label": pd.NA,
            })
        mapping_csv = str(tmp_path / "mapping.csv")
        pd.DataFrame(rows).to_csv(mapping_csv, index=False)

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

        benchmark_dir = str(tmp_path / "benchmark")
        fake_ds = make_test_ds(
            name="luad",
            data_root=str(tmp_path), wsi_dir=str(tmp_path / "wsi"),
            mapping_csv=mapping_csv, output_dir=str(tmp_path / "output"),
            benchmark_dir=benchmark_dir, features_base_dir=str(tmp_path / "features"),
            encoder_dims={"fake_enc": feat_dim}, encoder_models={"test/fake": "fake_enc"},
            tasks={
                "brca": TaskDef(
                    name="brca", label_col="BRCA_predict_label",
                    label_map={0: "neg", 1: "pos"}, n_classes=2,
                ),
            },
        )

        csv_path = os.path.join(benchmark_dir, "dataset_csv", "brca.csv")
        df = create_task_csv(
            mapping_csv, csv_path, "BRCA_predict_label", {0: "neg", 1: "pos"}, fake_ds,
        )
        splits_dir = os.path.join(benchmark_dir, "splits", "standard", "brca")
        create_strategy_splits(csv_path, splits_dir, n_splits=2, seed=42)
        slide_ids = df["slide_id"].tolist()
        pt_dir = os.path.join(benchmark_dir, "features", "fake_enc")
        convert_h5_to_pt(str(feat_dir), pt_dir, "fake_enc", slide_ids)

        task = TaskConfig(
            name="brca", label_col="BRCA_predict_label",
            label_dict={"neg": 0, "pos": 1}, n_classes=2,
        )
        model = ModelConfig(model_type="clam_sb", B=4)
        train = TrainConfig(max_epochs=2, lr=1e-3, seed=42, patience=5, stop_epoch=0)
        exp_cfg = ExperimentConfig(
            task=task, encoder_key="fake_enc", embed_dim=feat_dim,
            model=model, train=train, n_folds=2, strategy="standard",
            dataset="luad",
        )

        summary = run_experiment(exp_cfg, benchmark_dir, torch.device("cpu"))
        assert summary["dataset"] == "luad"

        summary_path = os.path.join(
            benchmark_dir, "results", exp_cfg.results_subdir, "summary.json",
        )
        with open(summary_path) as f:
            on_disk = json.load(f)
        assert on_disk["dataset"] == "luad"


# ---------------------------------------------------------------------------
# Shared H5-bag fixture -- DTFD-MIL and ABMIL consume the same H5 patch-bag
# format as nnMIL (design spec Â§6), mirroring the identical
# `_build_benchmark_fixture` helper duplicated in test_abmil_arm.py /
# test_dtfd_arm.py.
# ---------------------------------------------------------------------------


def _build_bag_fixture(root, task="brca", strategy="standard", encoder="conch_v15",
                        n_folds=2, n_slides=12, emb=32, n_patches=20):
    rng = np.random.default_rng(7)
    h5_dir = os.path.join(root, "features", f"features_{encoder}")
    os.makedirs(h5_dir, exist_ok=True)

    slide_ids = [f"s{i}" for i in range(n_slides)]
    labels = {sid: (i % 2) for i, sid in enumerate(slide_ids)}
    for sid in slide_ids:
        feats = rng.standard_normal((n_patches, emb)).astype("float32") + labels[sid] * 3.0
        with h5py.File(os.path.join(h5_dir, f"{sid}.h5"), "w") as f:
            f.create_dataset("features", data=feats)

    csv_dir = os.path.join(root, "dataset_csv")
    os.makedirs(csv_dir, exist_ok=True)
    label_names = {0: "neg", 1: "pos"}
    pd.DataFrame({
        "slide_id": slide_ids,
        "case_id": slide_ids,
        "label": [label_names[labels[sid]] for sid in slide_ids],
    }).to_csv(os.path.join(csv_dir, f"{task}.csv"), index=False)

    splits_dir = os.path.join(root, "splits", strategy, task)
    os.makedirs(splits_dir, exist_ok=True)
    for fold in range(n_folds):
        test_ids = [slide_ids[fold * 2], slide_ids[fold * 2 + 1]]
        val_ids = [slide_ids[(fold * 2 + 2) % n_slides], slide_ids[(fold * 2 + 3) % n_slides]]
        train_ids = [s for s in slide_ids if s not in test_ids and s not in val_ids]
        col = {
            "train": train_ids,
            "val": val_ids + [None] * (len(train_ids) - len(val_ids)),
            "test": test_ids + [None] * (len(train_ids) - len(test_ids)),
        }
        pd.DataFrame(col).to_csv(os.path.join(splits_dir, f"splits_{fold}.csv"), index=False)

    # nnMIL dataset_plan.json -- ABMIL/DTFD reuse nnMIL prep; only feature_dir is read.
    plan_dir = os.path.join(root, "nnmil", strategy, f"{task}_{encoder}")
    os.makedirs(plan_dir, exist_ok=True)
    with open(os.path.join(plan_dir, "dataset_plan.json"), "w") as f:
        json.dump({"feature_dir": h5_dir}, f)

    return h5_dir


class TestAbmilRunnerCarriesDataset:
    def test_summary_has_dataset(self, tmp_path):
        from autobench.pipeline.abmil.config import ABMILConfig
        from autobench.pipeline.abmil.runner import run_abmil_experiment

        _build_bag_fixture(str(tmp_path))
        exp_cfg = ExperimentConfig(
            task=TaskConfig(
                name="brca", label_col="label",
                label_dict={"neg": 0, "pos": 1}, n_classes=2,
            ),
            encoder_key="conch_v15", embed_dim=32,
            model=ModelConfig(model_type="abmil"),
            train=TrainConfig(seed=42), n_folds=2,
            framework=Framework.ABMIL, strategy="standard", dataset="luad",
        )
        cfg = ABMILConfig(M=16, L=8, max_epochs=2, lr=1e-2, early_stopping=False)

        summary = run_abmil_experiment(exp_cfg, str(tmp_path), device="cpu", cfg=cfg)
        assert summary["dataset"] == "luad"

        summary_path = os.path.join(
            str(tmp_path), "results", exp_cfg.results_subdir, "summary.json",
        )
        with open(summary_path) as f:
            on_disk = json.load(f)
        assert on_disk["dataset"] == "luad"


class TestDtfdRunnerCarriesDataset:
    def test_summary_has_dataset(self, tmp_path):
        from autobench.pipeline.dtfd.config import DTFDConfig
        from autobench.pipeline.dtfd.runner import run_dtfd_experiment

        _build_bag_fixture(str(tmp_path))
        exp_cfg = ExperimentConfig(
            task=TaskConfig(
                name="brca", label_col="label",
                label_dict={"neg": 0, "pos": 1}, n_classes=2,
            ),
            encoder_key="conch_v15", embed_dim=32,
            model=ModelConfig(model_type="dtfd_mil"),
            train=TrainConfig(seed=42), n_folds=2,
            framework=Framework.DTFD, strategy="standard", dataset="luad",
        )
        cfg = DTFDConfig(numGroup=2, mDim=16, max_epochs=2, lr=1e-3, early_stopping=False)

        summary = run_dtfd_experiment(exp_cfg, str(tmp_path), device="cpu", cfg=cfg)
        assert summary["dataset"] == "luad"

        summary_path = os.path.join(
            str(tmp_path), "results", exp_cfg.results_subdir, "summary.json",
        )
        with open(summary_path) as f:
            on_disk = json.load(f)
        assert on_disk["dataset"] == "luad"


class TestTitanRunnerCarriesDataset:
    def test_summary_has_dataset(self, tmp_path):
        from autobench.pipeline.titan.prepare import prepare_titan_experiment
        from autobench.pipeline.titan.runner import run_titan_experiment

        benchmark_dir = str(tmp_path / "benchmark")
        os.makedirs(os.path.join(benchmark_dir, "dataset_csv"), exist_ok=True)

        rows = []
        for i in range(20):
            rows.append({
                "case_id": f"P{i:03d}",
                "slide_id": f"slide_{i:05d}",
                "label": "neg" if i % 2 == 0 else "pos",
            })
        task_csv = os.path.join(benchmark_dir, "dataset_csv", "brca.csv")
        pd.DataFrame(rows).to_csv(task_csv, index=False)
        task_df = pd.read_csv(task_csv)

        features_dir = os.path.join(benchmark_dir, "features_titan")
        os.makedirs(features_dir, exist_ok=True)
        rng = np.random.default_rng(0)
        for sid in task_df["slide_id"]:
            with h5py.File(os.path.join(features_dir, f"{sid}.h5"), "w") as f:
                f.create_dataset(
                    "slide_feature", data=rng.standard_normal((1, 64)).astype(np.float32),
                )

        splits_dir = os.path.join(benchmark_dir, "splits", "standard", "brca")
        os.makedirs(splits_dir, exist_ok=True)
        slide_ids = task_df["slide_id"].tolist()
        split_df = pd.DataFrame({
            "train": slide_ids[:14] + [None] * 3,
            "val": slide_ids[14:17] + [None] * 14,
            "test": slide_ids[17:20] + [None] * 14,
        })
        # 2-fold run: reuse the same split for both folds (fold assignment is
        # irrelevant to this test -- only the runner's `dataset` stamping is).
        split_df.to_csv(os.path.join(splits_dir, "splits_0.csv"), index=False)
        split_df.to_csv(os.path.join(splits_dir, "splits_1.csv"), index=False)

        prepare_titan_experiment(
            benchmark_dir=benchmark_dir, task_name="brca", features_base_dir=benchmark_dir,
        )

        exp_cfg = ExperimentConfig(
            task=TaskConfig(
                name="brca", label_col="label",
                label_dict={"neg": 0, "pos": 1}, n_classes=2,
            ),
            encoder_key="titan", embed_dim=64,
            model=ModelConfig(model_type="titan"),
            train=TrainConfig(max_epochs=2, patience=1, seed=42),
            n_folds=2, framework=Framework.TITAN, strategy="standard",
            dataset="luad",
        )

        summary = run_titan_experiment(exp_cfg, benchmark_dir, device="cpu")
        assert summary["dataset"] == "luad"

        summary_path = os.path.join(
            benchmark_dir, "results", exp_cfg.results_subdir, "summary.json",
        )
        with open(summary_path) as f:
            on_disk = json.load(f)
        assert on_disk["dataset"] == "luad"


# ---------------------------------------------------------------------------
# nnMIL -- self-configures ~100 epochs for real (experiment_planner.py), impractical for
# CI (test_instrumentation.py precedent). Mock the heavy vendored trainer;
# the runner itself (and its `dataset` stamping) still runs for real.
# ---------------------------------------------------------------------------


class _FakeClassificationTrainer:
    """Stand-in for nnMIL's ClassificationTrainer (vendored, heavy)."""

    def __init__(self, plan_path, model_type, fold, save_dir, seed, **kwargs):
        self.plan_path = plan_path
        self.model_type = model_type
        self.fold = fold
        self.save_dir = save_dir
        self.seed = seed

    def create_model(self):
        pass

    def create_data_loaders(self):
        pass

    def train(self):
        pass

    def evaluate(self, split):
        return {
            f"{split}_{split}/auroc": 0.8,
            f"{split}_{split}/bacc": 0.75,
            f"{split}_{split}/acc": 0.78,
            f"{split}_{split}/weighted_f1": 0.76,
        }


class TestNnmilRunnerCarriesDataset:
    def test_summary_has_dataset(self, tmp_path, monkeypatch):
        import autobench.pipeline.nnmil._imports as nnmil_imports
        from autobench.pipeline.nnmil.prepare import nnmil_plan_dir
        from autobench.pipeline.nnmil.runner import run_nnmil_experiment

        monkeypatch.setattr(
            nnmil_imports, "ClassificationTrainer", _FakeClassificationTrainer,
        )

        benchmark_dir = str(tmp_path / "benchmark")
        plan_dir = nnmil_plan_dir(benchmark_dir, "standard", "brca", "conch_v15")
        os.makedirs(plan_dir, exist_ok=True)
        with open(os.path.join(plan_dir, "dataset_plan.json"), "w") as f:
            json.dump({"task_type": "classification"}, f)

        exp_cfg = ExperimentConfig(
            task=TaskConfig(
                name="brca", label_col="label",
                label_dict={"neg": 0, "pos": 1}, n_classes=2,
            ),
            encoder_key="conch_v15", embed_dim=768,
            model=ModelConfig(model_type="simple_mil"),
            train=TrainConfig(seed=42), n_folds=2,
            framework=Framework.NNMIL, strategy="standard", dataset="luad",
        )

        summary = run_nnmil_experiment(exp_cfg, benchmark_dir, device="cpu")
        assert summary["dataset"] == "luad"

        summary_path = os.path.join(
            benchmark_dir, "results", exp_cfg.results_subdir, "summary.json",
        )
        with open(summary_path) as f:
            on_disk = json.load(f)
        assert on_disk["dataset"] == "luad"
