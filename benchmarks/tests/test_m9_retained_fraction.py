"""M-9: missing feature files must not silently drop val/test slides.

Reproduces the original defect (evidence: abmil/dataset.py:90-101, guarded
only by an all-train-slides-empty check at abmil/train.py:118) for the two
loaders whose contract is identical -- ABMIL and DTFD both drop a slide
missing its H5 feature file with nothing but a ``print`` -- and for CLAM's
structurally different two-stage (global-filter-then-per-fold-select)
loader. Before the fix, all three would happily return a small val/test
list computed on an arbitrarily shrunk cohort; a metric computed on it looks
like a normal, confident result.
"""
from __future__ import annotations

import os

import h5py
import numpy as np
import pandas as pd
import pytest

from autobench.pipeline.dataset_guards import SplitRetentionError


def _write_h5(path: str, n_patches: int = 4, dim: int = 8) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("features", data=np.random.rand(n_patches, dim).astype("float32"))


def _write_task_csv(path: str, slide_ids: list[str], labels: list[str], case_ids=None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({
        "case_id": case_ids or [f"case_{i}" for i in range(len(slide_ids))],
        "slide_id": slide_ids,
        "label": labels,
    }).to_csv(path, index=False)


def _write_split_csv(path: str, **columns: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    max_len = max(len(v) for v in columns.values())
    padded = {k: v + [pd.NA] * (max_len - len(v)) for k, v in columns.items()}
    pd.DataFrame(padded).to_csv(path, index=False)


LABEL_DICT = {"neg": 0, "pos": 1}


class TestAbmilRetainedFraction:
    def _setup(self, tmp_path, n_slides=10, n_present=10, split="val"):
        slide_ids = [f"s{i:03d}" for i in range(n_slides)]
        labels = ["pos" if i % 2 == 0 else "neg" for i in range(n_slides)]
        task_csv = str(tmp_path / "task.csv")
        _write_task_csv(task_csv, slide_ids, labels)

        split_csv = str(tmp_path / "splits_0.csv")
        _write_split_csv(split_csv, **{split: slide_ids})

        h5_dir = str(tmp_path / "h5")
        for sid in slide_ids[:n_present]:
            _write_h5(os.path.join(h5_dir, f"{sid}.h5"))
        return task_csv, split_csv, h5_dir

    def test_full_retention_on_val_returns_all_slides(self, tmp_path):
        from autobench.pipeline.abmil.dataset import load_abmil_split

        task_csv, split_csv, h5_dir = self._setup(tmp_path, n_slides=10, n_present=10)
        slides = load_abmil_split(task_csv, split_csv, h5_dir, LABEL_DICT, "val")
        assert len(slides) == 10

    def test_severe_drop_on_val_raises(self, tmp_path):
        """The original defect: only 5 of 10 H5 files extracted. Before the
        fix this returned a silent 5-slide val set; now it must raise."""
        from autobench.pipeline.abmil.dataset import load_abmil_split

        task_csv, split_csv, h5_dir = self._setup(tmp_path, n_slides=10, n_present=5)
        with pytest.raises(SplitRetentionError, match=r"5/10"):
            load_abmil_split(task_csv, split_csv, h5_dir, LABEL_DICT, "val")

    def test_severe_drop_on_test_raises(self, tmp_path):
        from autobench.pipeline.abmil.dataset import load_abmil_split

        task_csv, split_csv, h5_dir = self._setup(
            tmp_path, n_slides=10, n_present=5, split="test",
        )
        with pytest.raises(SplitRetentionError):
            load_abmil_split(task_csv, split_csv, h5_dir, LABEL_DICT, "test")

    def test_severe_drop_on_train_only_warns(self, tmp_path, capsys):
        """Train tolerates shrink deliberately (lossy, not corrupting) --
        must still return whatever survived, not raise."""
        from autobench.pipeline.abmil.dataset import load_abmil_split

        task_csv, split_csv, h5_dir = self._setup(
            tmp_path, n_slides=10, n_present=1, split="train",
        )
        slides = load_abmil_split(task_csv, split_csv, h5_dir, LABEL_DICT, "train")
        assert len(slides) == 1
        assert "WARNING" in capsys.readouterr().out

    def test_minority_class_wiped_out_raises_even_at_high_overall_fraction(self, tmp_path):
        """9 of 10 slides present overall (90%, at the fraction floor) but the
        one dropped slide is the ONLY slide of its class -- a
        class-correlated shrink, exactly what the audit named."""
        from autobench.pipeline.abmil.dataset import load_abmil_split

        slide_ids = [f"s{i:03d}" for i in range(10)]
        # 9 "neg", 1 "pos" -- the single "pos" slide is the one we'll drop.
        labels = ["neg"] * 9 + ["pos"]
        task_csv = str(tmp_path / "task.csv")
        _write_task_csv(task_csv, slide_ids, labels)
        split_csv = str(tmp_path / "splits_0.csv")
        _write_split_csv(split_csv, val=slide_ids)
        h5_dir = str(tmp_path / "h5")
        for sid in slide_ids[:9]:  # drop the last slide (the "pos" one)
            _write_h5(os.path.join(h5_dir, f"{sid}.h5"))

        with pytest.raises(SplitRetentionError, match=r"per-class floor"):
            load_abmil_split(task_csv, split_csv, h5_dir, LABEL_DICT, "val")

    def test_survival_split_full_retention_passes(self, tmp_path):
        from autobench.pipeline.abmil.dataset import load_abmil_survival_split

        slide_ids = [f"s{i:03d}" for i in range(8)]
        task_csv = str(tmp_path / "surv_task.csv")
        os.makedirs(tmp_path, exist_ok=True)
        pd.DataFrame({
            "case_id": [f"c{i}" for i in range(8)],
            "slide_id": slide_ids,
            "status": [i % 2 for i in range(8)],
            "time": [float(i + 1) for i in range(8)],
        }).to_csv(task_csv, index=False)
        split_csv = str(tmp_path / "splits_0.csv")
        _write_split_csv(split_csv, val=slide_ids)
        h5_dir = str(tmp_path / "h5")
        for sid in slide_ids:
            _write_h5(os.path.join(h5_dir, f"{sid}.h5"))

        slides = load_abmil_survival_split(task_csv, split_csv, h5_dir, "val")
        assert len(slides) == 8

    def test_survival_split_severe_drop_on_val_raises(self, tmp_path):
        from autobench.pipeline.abmil.dataset import load_abmil_survival_split

        slide_ids = [f"s{i:03d}" for i in range(8)]
        task_csv = str(tmp_path / "surv_task.csv")
        pd.DataFrame({
            "case_id": [f"c{i}" for i in range(8)],
            "slide_id": slide_ids,
            "status": [i % 2 for i in range(8)],
            "time": [float(i + 1) for i in range(8)],
        }).to_csv(task_csv, index=False)
        split_csv = str(tmp_path / "splits_0.csv")
        _write_split_csv(split_csv, val=slide_ids)
        h5_dir = str(tmp_path / "h5")
        for sid in slide_ids[:3]:
            _write_h5(os.path.join(h5_dir, f"{sid}.h5"))

        with pytest.raises(SplitRetentionError):
            load_abmil_survival_split(task_csv, split_csv, h5_dir, "val")


class TestDtfdRetainedFraction:
    """DTFD's loader is a structural clone of ABMIL's -- lighter coverage,
    just enough to prove the wiring (not re-deriving every ABMIL case)."""

    def test_full_retention_on_val_returns_all_slides(self, tmp_path):
        from autobench.pipeline.dtfd.dataset import load_dtfd_split

        slide_ids = [f"s{i:03d}" for i in range(10)]
        labels = ["pos" if i % 2 == 0 else "neg" for i in range(10)]
        task_csv = str(tmp_path / "task.csv")
        _write_task_csv(task_csv, slide_ids, labels)
        split_csv = str(tmp_path / "splits_0.csv")
        _write_split_csv(split_csv, val=slide_ids)
        h5_dir = str(tmp_path / "h5")
        for sid in slide_ids:
            _write_h5(os.path.join(h5_dir, f"{sid}.h5"))

        slides = load_dtfd_split(task_csv, split_csv, h5_dir, LABEL_DICT, "val")
        assert len(slides) == 10

    def test_severe_drop_on_val_raises(self, tmp_path):
        from autobench.pipeline.dtfd.dataset import load_dtfd_split

        slide_ids = [f"s{i:03d}" for i in range(10)]
        labels = ["pos" if i % 2 == 0 else "neg" for i in range(10)]
        task_csv = str(tmp_path / "task.csv")
        _write_task_csv(task_csv, slide_ids, labels)
        split_csv = str(tmp_path / "splits_0.csv")
        _write_split_csv(split_csv, val=slide_ids)
        h5_dir = str(tmp_path / "h5")
        for sid in slide_ids[:4]:
            _write_h5(os.path.join(h5_dir, f"{sid}.h5"))

        with pytest.raises(SplitRetentionError, match=r"4/10"):
            load_dtfd_split(task_csv, split_csv, h5_dir, LABEL_DICT, "val")

    def test_severe_drop_on_train_only_warns(self, tmp_path, capsys):
        from autobench.pipeline.dtfd.dataset import load_dtfd_split

        slide_ids = [f"s{i:03d}" for i in range(10)]
        labels = ["pos" if i % 2 == 0 else "neg" for i in range(10)]
        task_csv = str(tmp_path / "task.csv")
        _write_task_csv(task_csv, slide_ids, labels)
        split_csv = str(tmp_path / "splits_0.csv")
        _write_split_csv(split_csv, train=slide_ids)
        h5_dir = str(tmp_path / "h5")
        for sid in slide_ids[:2]:
            _write_h5(os.path.join(h5_dir, f"{sid}.h5"))

        slides = load_dtfd_split(task_csv, split_csv, h5_dir, LABEL_DICT, "train")
        assert len(slides) == 2
        assert "WARNING" in capsys.readouterr().out

    def test_survival_split_severe_drop_on_test_raises(self, tmp_path):
        from autobench.pipeline.dtfd.dataset import load_dtfd_survival_split

        slide_ids = [f"s{i:03d}" for i in range(8)]
        task_csv = str(tmp_path / "surv_task.csv")
        pd.DataFrame({
            "case_id": [f"c{i}" for i in range(8)],
            "slide_id": slide_ids,
            "status": [i % 2 for i in range(8)],
            "time": [float(i + 1) for i in range(8)],
        }).to_csv(task_csv, index=False)
        split_csv = str(tmp_path / "splits_0.csv")
        _write_split_csv(split_csv, test=slide_ids)
        h5_dir = str(tmp_path / "h5")
        for sid in slide_ids[:3]:
            _write_h5(os.path.join(h5_dir, f"{sid}.h5"))

        with pytest.raises(SplitRetentionError):
            load_dtfd_survival_split(task_csv, split_csv, h5_dir, "test")


def _write_pt(path: str, n_patches: int = 4, dim: int = 8) -> None:
    import torch

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(torch.randn(n_patches, dim), path)


def _write_clam_split_csv(path: str, train=None, val=None, test=None) -> None:
    """CLAM's vendored ``get_split_from_df`` indexes ``all_splits['train']``
    /``['val']``/``['test']`` unconditionally (KeyError otherwise), unlike
    this package's own ``_load_split_ids`` (which tolerates a missing
    column). Always write all three; an unused split is an all-NaN column,
    which CLAM's own code already treats as "nothing assigned" (-> None)."""
    _write_split_csv(path, train=train or [], val=val or [], test=test or [])


def _make_clam_exp_cfg(task_name: str, encoder_key: str = "enc"):
    from autobench.pipeline.config import ExperimentConfig, ModelConfig, TaskConfig, TrainConfig

    return ExperimentConfig(
        task=TaskConfig(
            name=task_name, label_col="label",
            label_dict={"neg": 0, "pos": 1}, n_classes=2,
        ),
        encoder_key=encoder_key,
        embed_dim=8,
        model=ModelConfig(model_type="clam_sb"),
        train=TrainConfig(),
    )


class TestClamRetainedFraction:
    """CLAM's loader is structurally different: create_dataset() filters
    missing .pt files ONCE, globally, before any fold is chosen, so
    load_fold_splits must recover the TRUE per-split assignment from the
    raw split CSV rather than trusting dataset.slide_data (already
    filtered)."""

    def _setup(self, tmp_path, n_slides=10, n_present=10, task_name="task"):
        from autobench.pipeline.clam.dataset import create_dataset

        slide_ids = [f"s{i:03d}" for i in range(n_slides)]
        labels = ["pos" if i % 2 == 0 else "neg" for i in range(n_slides)]
        benchmark_dir = str(tmp_path)
        task_csv = os.path.join(benchmark_dir, "dataset_csv", f"{task_name}.csv")
        _write_task_csv(task_csv, slide_ids, labels)

        pt_dir = os.path.join(benchmark_dir, "features", "enc", "pt_files")
        for sid in slide_ids[:n_present]:
            _write_pt(os.path.join(pt_dir, f"{sid}.pt"))

        exp_cfg = _make_clam_exp_cfg(task_name)
        dataset = create_dataset(exp_cfg, benchmark_dir, task_csv_name=task_name)
        return exp_cfg, dataset, benchmark_dir, slide_ids

    def test_full_retention_on_val_returns_all_slides(self, tmp_path):
        from autobench.pipeline.clam.dataset import load_fold_splits

        exp_cfg, dataset, benchmark_dir, slide_ids = self._setup(
            tmp_path, n_slides=10, n_present=10,
        )
        split_csv = os.path.join(benchmark_dir, "splits", "standard", "task", "splits_0.csv")
        _write_clam_split_csv(split_csv, val=slide_ids)

        _, val_split, _ = load_fold_splits(
            dataset, benchmark_dir, "standard/task", 0, task_csv_name="task",
        )
        assert len(val_split) == 10

    def test_severe_drop_on_val_raises(self, tmp_path):
        """The original defect, CLAM flavour: only 4 of 10 .pt files
        extracted. create_dataset() filters them out globally and silently
        (a print only); before the fix, load_fold_splits then handed back a
        4-slide val Generic_Split with no error."""
        from autobench.pipeline.clam.dataset import load_fold_splits

        exp_cfg, dataset, benchmark_dir, slide_ids = self._setup(
            tmp_path, n_slides=10, n_present=4,
        )
        split_csv = os.path.join(benchmark_dir, "splits", "standard", "task", "splits_0.csv")
        _write_clam_split_csv(split_csv, val=slide_ids)

        with pytest.raises(SplitRetentionError, match=r"4/10"):
            load_fold_splits(
                dataset, benchmark_dir, "standard/task", 0, task_csv_name="task",
            )

    def test_severe_drop_on_test_raises(self, tmp_path):
        from autobench.pipeline.clam.dataset import load_fold_splits

        exp_cfg, dataset, benchmark_dir, slide_ids = self._setup(
            tmp_path, n_slides=10, n_present=4,
        )
        split_csv = os.path.join(benchmark_dir, "splits", "standard", "task", "splits_0.csv")
        _write_clam_split_csv(split_csv, test=slide_ids)

        with pytest.raises(SplitRetentionError):
            load_fold_splits(
                dataset, benchmark_dir, "standard/task", 0, task_csv_name="task",
            )

    def test_severe_drop_on_train_only_warns(self, tmp_path, capsys):
        from autobench.pipeline.clam.dataset import load_fold_splits

        exp_cfg, dataset, benchmark_dir, slide_ids = self._setup(
            tmp_path, n_slides=10, n_present=1,
        )
        split_csv = os.path.join(benchmark_dir, "splits", "standard", "task", "splits_0.csv")
        _write_clam_split_csv(split_csv, train=slide_ids)

        train_split, _, _ = load_fold_splits(
            dataset, benchmark_dir, "standard/task", 0, task_csv_name="task",
        )
        assert len(train_split) == 1
        assert "WARNING" in capsys.readouterr().out

    def test_minority_class_wiped_out_raises(self, tmp_path):
        """9 of 10 slides present (90%, exactly at the fraction floor) but
        the dropped slide is the cohort's only "pos" -- per-class floor
        must catch what the fraction floor cannot."""
        from autobench.pipeline.clam.dataset import create_dataset, load_fold_splits

        slide_ids = [f"s{i:03d}" for i in range(10)]
        labels = ["neg"] * 9 + ["pos"]
        benchmark_dir = str(tmp_path)
        task_csv = os.path.join(benchmark_dir, "dataset_csv", "task.csv")
        _write_task_csv(task_csv, slide_ids, labels)

        pt_dir = os.path.join(benchmark_dir, "features", "enc", "pt_files")
        for sid in slide_ids[:9]:  # drop the last slide (the "pos" one)
            _write_pt(os.path.join(pt_dir, f"{sid}.pt"))

        exp_cfg = _make_clam_exp_cfg("task")
        dataset = create_dataset(exp_cfg, benchmark_dir, task_csv_name="task")

        split_csv = os.path.join(benchmark_dir, "splits", "standard", "task", "splits_0.csv")
        _write_clam_split_csv(split_csv, val=slide_ids)

        with pytest.raises(SplitRetentionError, match=r"per-class floor"):
            load_fold_splits(
                dataset, benchmark_dir, "standard/task", 0, task_csv_name="task",
            )

    def test_task_csv_name_omitted_skips_per_class_but_keeps_fraction_check(self, tmp_path):
        """Backward-compat: task_csv_name is optional. Without it the
        per-class floor is skipped, but the retained-fraction check (which
        needs no labels) still runs."""
        from autobench.pipeline.clam.dataset import load_fold_splits

        exp_cfg, dataset, benchmark_dir, slide_ids = self._setup(
            tmp_path, n_slides=10, n_present=4,
        )
        split_csv = os.path.join(benchmark_dir, "splits", "standard", "task", "splits_0.csv")
        _write_clam_split_csv(split_csv, val=slide_ids)

        with pytest.raises(SplitRetentionError, match=r"4/10"):
            load_fold_splits(dataset, benchmark_dir, "standard/task", 0)


class TestClamSurvivalRetainedFraction:
    def _setup(self, tmp_path, n_slides=8, n_present=8):
        slide_ids = [f"s{i:03d}" for i in range(n_slides)]
        benchmark_dir = str(tmp_path)
        task_csv = os.path.join(benchmark_dir, "dataset_csv", "surv_task.csv")
        os.makedirs(os.path.dirname(task_csv), exist_ok=True)
        pd.DataFrame({
            "case_id": [f"c{i}" for i in range(n_slides)],
            "slide_id": slide_ids,
            "status": [i % 2 for i in range(n_slides)],
            "time": [float(i + 1) for i in range(n_slides)],
        }).to_csv(task_csv, index=False)

        pt_dir = os.path.join(benchmark_dir, "features", "enc", "pt_files")
        for sid in slide_ids[:n_present]:
            _write_pt(os.path.join(pt_dir, f"{sid}.pt"))

        return benchmark_dir, slide_ids

    def test_full_retention_passes(self, tmp_path):
        from autobench.pipeline.clam.dataset import load_survival_fold_splits

        benchmark_dir, slide_ids = self._setup(tmp_path, n_slides=8, n_present=8)
        split_csv = os.path.join(
            benchmark_dir, "splits", "standard", "surv_task", "splits_0.csv",
        )
        _write_split_csv(split_csv, val=slide_ids)
        exp_cfg = _make_clam_exp_cfg("surv_task")

        _, val_samples, _ = load_survival_fold_splits(exp_cfg, benchmark_dir, 0)
        assert len(val_samples) == 8

    def test_severe_drop_on_val_raises(self, tmp_path):
        from autobench.pipeline.clam.dataset import load_survival_fold_splits

        benchmark_dir, slide_ids = self._setup(tmp_path, n_slides=8, n_present=2)
        split_csv = os.path.join(
            benchmark_dir, "splits", "standard", "surv_task", "splits_0.csv",
        )
        _write_split_csv(split_csv, val=slide_ids)
        exp_cfg = _make_clam_exp_cfg("surv_task")

        with pytest.raises(SplitRetentionError):
            load_survival_fold_splits(exp_cfg, benchmark_dir, 0)

    def test_severe_drop_on_train_only_warns(self, tmp_path, capsys):
        from autobench.pipeline.clam.dataset import load_survival_fold_splits

        benchmark_dir, slide_ids = self._setup(tmp_path, n_slides=8, n_present=1)
        split_csv = os.path.join(
            benchmark_dir, "splits", "standard", "surv_task", "splits_0.csv",
        )
        _write_split_csv(split_csv, train=slide_ids)
        exp_cfg = _make_clam_exp_cfg("surv_task")

        train_samples, _, _ = load_survival_fold_splits(exp_cfg, benchmark_dir, 0)
        assert len(train_samples) == 1
        assert "WARNING" in capsys.readouterr().out
