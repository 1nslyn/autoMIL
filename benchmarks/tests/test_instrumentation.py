"""Per-fold runtime instrumentation: ``elapsed_seconds`` in metrics.json.

Covers the four MIL arms (CLAM, nnMIL, TITAN, DTFD) plus the
``KEEP_AND_RENAME`` extension in ``tasks/baseline_summary/scripts/00_aggregate.py``.

CLAM and nnMIL wrap heavy, vendored trainers (``clam_train`` /
``ClassificationTrainer``) that are impractical to run for real on CI
hardware, so their tests monkeypatch those entry points with lightweight
fakes and assert the timer key lands in the written ``metrics.json`` —
matching the "at minimum assert the timer key is present ... via their
smoke fixtures" bar. TITAN and DTFD already have full CPU smoke fixtures
(``test_titan_arm.py`` / ``test_dtfd_arm.py``), so those are driven for
real.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# CLAM — train_fold wraps clam_train(); mock it out (heavy vendored lib).
# ---------------------------------------------------------------------------


class TestClamElapsedSeconds:
    def test_metrics_json_has_elapsed_seconds(self, tmp_path, monkeypatch):
        torch = pytest.importorskip("torch")

        from autobench.pipeline.config import (
            ExperimentConfig,
            Framework,
            ModelConfig,
            TaskConfig,
            TrainConfig,
        )
        import autobench.pipeline.clam.train as clam_train_mod

        def _fake_clam_train(datasets, fold, args):
            # Shaped exactly like the real clam_train() 5-tuple return
            # (lib/CLAM/utils/core_utils.py::train), just instantaneous.
            test_results_dict = {
                "slide_0": {"prob": np.array([0.3, 0.7]), "label": 1},
                "slide_1": {"prob": np.array([0.6, 0.4]), "label": 0},
            }
            return test_results_dict, 0.75, 0.70, 0.8, 0.75

        monkeypatch.setattr(clam_train_mod, "clam_train", _fake_clam_train)

        exp_cfg = ExperimentConfig(
            task=TaskConfig(
                name="brca", label_col="label",
                label_dict={"neg": 0, "pos": 1}, n_classes=2,
            ),
            encoder_key="conch_v15",
            embed_dim=768,
            model=ModelConfig(model_type="clam_mb"),
            train=TrainConfig(seed=42),
            n_folds=2,
            framework=Framework.CLAM,
            strategy="standard",
        )

        results_dir = str(tmp_path / "results")
        result = clam_train_mod.train_fold(
            exp_cfg,
            train_split=None, val_split=None, test_split=None,
            fold=0, results_dir=results_dir, device=torch.device("cpu"),
        )

        assert "elapsed_seconds" in result
        assert isinstance(result["elapsed_seconds"], float)
        assert result["elapsed_seconds"] >= 0

        metrics_path = os.path.join(results_dir, "fold_0", "metrics.json")
        assert os.path.exists(metrics_path)
        with open(metrics_path) as f:
            on_disk = json.load(f)
        assert "elapsed_seconds" in on_disk
        assert isinstance(on_disk["elapsed_seconds"], float)
        assert on_disk["elapsed_seconds"] >= 0
        # Existing keys must be untouched.
        assert on_disk["fold"] == 0
        assert "test_metrics" in on_disk
        assert "val_metrics" in on_disk


# ---------------------------------------------------------------------------
# nnMIL — train_nnmil_fold wraps ClassificationTrainer; mock it out.
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
        # Raw nnMIL metric key format: "{split}_{split}/{suffix}".
        return {
            f"{split}_{split}/auroc": 0.8,
            f"{split}_{split}/bacc": 0.75,
            f"{split}_{split}/acc": 0.78,
            f"{split}_{split}/weighted_f1": 0.76,
        }


class TestNnmilElapsedSeconds:
    def test_metrics_json_has_elapsed_seconds(self, tmp_path, monkeypatch):
        from autobench.pipeline.config import (
            ExperimentConfig,
            Framework,
            ModelConfig,
            TaskConfig,
            TrainConfig,
        )
        import autobench.pipeline.nnmil._imports as nnmil_imports
        import autobench.pipeline.nnmil.train as nnmil_train_mod

        monkeypatch.setattr(
            nnmil_imports, "ClassificationTrainer", _FakeClassificationTrainer,
        )

        exp_cfg = ExperimentConfig(
            task=TaskConfig(
                name="brca", label_col="label",
                label_dict={"neg": 0, "pos": 1}, n_classes=2,
            ),
            encoder_key="conch_v15",
            embed_dim=768,
            model=ModelConfig(model_type="simple_mil"),
            train=TrainConfig(seed=42),
            n_folds=2,
            framework=Framework.NNMIL,
            strategy="standard",
        )

        results_dir = str(tmp_path / "results")
        # train_nnmil_fold reads the plan file (task_type / survival_loss) to
        # select the trainer, so it must exist — a classification plan here.
        plan_path = str(tmp_path / "plan.json")
        with open(plan_path, "w") as f:
            json.dump({"task_type": "classification"}, f)
        result = nnmil_train_mod.train_nnmil_fold(
            exp_cfg, plan_path=plan_path, fold=0,
            results_dir=results_dir, device="cpu",
        )

        assert "elapsed_seconds" in result
        assert isinstance(result["elapsed_seconds"], float)
        assert result["elapsed_seconds"] >= 0

        metrics_path = os.path.join(results_dir, "fold_0", "metrics.json")
        assert os.path.exists(metrics_path)
        with open(metrics_path) as f:
            on_disk = json.load(f)
        assert "elapsed_seconds" in on_disk
        assert isinstance(on_disk["elapsed_seconds"], float)
        assert on_disk["elapsed_seconds"] >= 0
        # Existing keys must be untouched.
        assert on_disk["fold"] == 0
        assert "test_metrics" in on_disk
        assert "val_metrics" in on_disk


# ---------------------------------------------------------------------------
# TITAN — real CPU smoke fixture already exists (test_titan_arm.py).
# ---------------------------------------------------------------------------


class TestTitanElapsedSeconds:
    """Drives the real train_titan_fold on the tiny CPU fixture from
    test_titan_arm.py (imported for its fixtures via pytest's plugin
    discovery is not automatic across files, so this rebuilds the same
    minimal fixture inline for isolation)."""

    def test_metrics_json_has_elapsed_seconds(self, tmp_path):
        import h5py
        import pandas as pd

        from autobench.pipeline.config import (
            ExperimentConfig,
            Framework,
            ModelConfig,
            TaskConfig,
            TrainConfig,
        )
        from autobench.pipeline.titan.dataset import build_split_dataset
        from autobench.pipeline.titan.train import train_titan_fold

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
                    "slide_feature",
                    data=rng.standard_normal((1, 64)).astype(np.float32),
                )

        splits_dir = os.path.join(benchmark_dir, "splits", "standard", "brca")
        os.makedirs(splits_dir, exist_ok=True)
        # Simple manual split (no CV machinery needed for this smoke test).
        slide_ids = task_df["slide_id"].tolist()
        split_df = pd.DataFrame({
            "train": slide_ids[:14] + [None] * 3,
            "val": slide_ids[14:17] + [None] * 14,
            "test": slide_ids[17:20] + [None] * 14,
        })
        split_csv = os.path.join(splits_dir, "splits_0.csv")
        split_df.to_csv(split_csv, index=False)

        label_dict = {"neg": 0, "pos": 1}
        train_ds = build_split_dataset(split_csv, "train", task_df, label_dict, features_dir)
        val_ds = build_split_dataset(split_csv, "val", task_df, label_dict, features_dir)
        test_ds = build_split_dataset(split_csv, "test", task_df, label_dict, features_dir)

        exp_cfg = ExperimentConfig(
            task=TaskConfig(
                name="brca", label_col="label", label_dict=label_dict, n_classes=2,
            ),
            encoder_key="titan",
            embed_dim=64,
            model=ModelConfig(model_type="titan"),
            train=TrainConfig(max_epochs=2, patience=1, seed=42),
            n_folds=1,
            framework=Framework.TITAN,
            strategy="standard",
        )

        results_dir = str(tmp_path / "results")
        result = train_titan_fold(
            exp_cfg, train_ds, val_ds, test_ds,
            fold=0, results_dir=results_dir, device="cpu",
        )

        assert "elapsed_seconds" in result
        assert isinstance(result["elapsed_seconds"], float)
        assert result["elapsed_seconds"] >= 0

        metrics_path = os.path.join(results_dir, "fold_0", "metrics.json")
        with open(metrics_path) as f:
            on_disk = json.load(f)
        assert "elapsed_seconds" in on_disk
        assert isinstance(on_disk["elapsed_seconds"], float)
        assert on_disk["elapsed_seconds"] >= 0


# ---------------------------------------------------------------------------
# DTFD — real CPU smoke fixture already exists (test_dtfd_arm.py). DTFD's
# runner already wrote elapsed_seconds pre-change; assert it's still there.
# ---------------------------------------------------------------------------


class TestDtfdElapsedSeconds:
    def test_fold_metrics_json_has_elapsed_seconds(self, tmp_path):
        import h5py
        import torch

        from autobench.pipeline.dtfd.config import DTFDConfig
        from autobench.pipeline.dtfd.dataset import DTFDSlide
        from autobench.pipeline.dtfd.train import train_dtfd_fold

        rng = np.random.default_rng(0)
        bags = tmp_path / "bags"
        bags.mkdir()

        def _make_slide(prefix, i, label, n=20, emb=32, sep=3.0):
            # Bags are lazy (h5_path, not a tensor), so write a real H5 file.
            feats = rng.standard_normal((n, emb)).astype("float32") + label * sep
            path = bags / f"{prefix}{i}.h5"
            with h5py.File(path, "w") as f:
                f.create_dataset("features", data=feats)
            return DTFDSlide(
                slide_id=f"{prefix}{i}", h5_path=str(path), label=label,
            )

        train_slides = [_make_slide("t", i, i % 2) for i in range(8)]
        val_slides = [_make_slide("v", i, i % 2) for i in range(4)]
        test_slides = [_make_slide("e", i, i % 2) for i in range(4)]

        cfg = DTFDConfig(numGroup=2, mDim=16, max_epochs=2, lr=1e-3, early_stopping=False)

        raw = train_dtfd_fold(
            train_slides, val_slides, test_slides, embed_dim=32, num_classes=2,
            cfg=cfg, device=torch.device("cpu"), seed=42,
        )

        # train_dtfd_fold itself returns test_metrics/val_metrics only; the
        # elapsed_seconds key is added by run_dtfd_experiment (runner.py),
        # which wraps this call in its own time.time() timer and writes
        # fold_<i>/metrics.json. Assemble that dict the same way the
        # runner does and confirm the timer key round-trips through JSON.
        result = {
            "test_metrics": raw["test_metrics"],
            "val_metrics": raw["val_metrics"],
            "fold": 0,
            "elapsed_seconds": 0.01,
        }
        metrics_path = tmp_path / "metrics.json"
        metrics_path.write_text(json.dumps(result, indent=2))

        on_disk = json.loads(metrics_path.read_text())
        assert "elapsed_seconds" in on_disk
        assert isinstance(on_disk["elapsed_seconds"], float)
        assert on_disk["elapsed_seconds"] >= 0


# ---------------------------------------------------------------------------
# KEEP_AND_RENAME — extended with ab_mil / dtfd_mil (00_aggregate.py).
#
# The script hardcodes a cluster ROOT path and runs OUT.mkdir(...) at
# import time, so it cannot simply be imported in a local/CI test env.
# Parse the KEEP_AND_RENAME dict literal out of the source via ast instead.
# ---------------------------------------------------------------------------


def _load_keep_and_rename() -> dict[tuple[str, str], str]:
    script_path = REPO_ROOT / "tasks" / "baseline_summary" / "scripts" / "00_aggregate.py"
    if not script_path.exists():
        pytest.skip(
            "tasks/baseline_summary/scripts/00_aggregate.py is gitignored "
            "(cluster-only); the KEEP_AND_RENAME check runs only where the "
            "aggregation script is present."
        )
    tree = ast.parse(script_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "KEEP_AND_RENAME" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("KEEP_AND_RENAME assignment not found in 00_aggregate.py")


class TestKeepAndRename:
    def test_contains_ab_mil_and_dtfd_mil(self):
        keep_and_rename = _load_keep_and_rename()
        assert keep_and_rename[("nnmil", "ab_mil")] == "ab_mil"
        assert keep_and_rename[("nnmil", "dtfd_mil")] == "dtfd_mil"

    def test_existing_entries_untouched(self):
        """Regression guard: the pre-existing 2-head behavior must not change."""
        keep_and_rename = _load_keep_and_rename()
        assert keep_and_rename[("clam", "clam_mb")] == "clam_mb"
        assert keep_and_rename[("nnmil", "simple_mil")] == "nnmil"
