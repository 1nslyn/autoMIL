"""Tests for autobench.pipeline.prepare module."""

import os

import h5py
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from autobench.pipeline.clam.prepare import convert_h5_to_pt
from autobench.pipeline.prepare import create_task_csv
from autobench.pipeline.splits import create_strategy_splits
from autobench.pipeline.config import StrategyConfig

STANDARD = StrategyConfig(strategy="standard", train_cohorts=[], test_cohorts=[])
from _helpers import make_test_ds


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ds():
    return make_test_ds()


@pytest.fixture
def mapping_csv(tmp_path):
    """Create a minimal mapping CSV that load_all_slides will accept."""
    rows = []
    for i in range(20):
        rows.append({
            "new_name": f"slide_{i:05d}.svs",
            "status": "mapped_unique_case_id",
            "primary_hospital": "UHN",
            "primary_case_id": f"K{i:03d}",
            "BRCA_predict_label": i % 2,       # 10 neg, 10 pos
            "HRD_label": i % 3 if i < 15 else pd.NA,  # some missing
        })
    csv_path = tmp_path / "mapping.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def h5_features_dir(tmp_path):
    """Create synthetic H5 feature files (2D)."""
    h5_dir = tmp_path / "features_conch_v15"
    h5_dir.mkdir()
    for i in range(20):
        n_patches = np.random.randint(10, 50)
        with h5py.File(h5_dir / f"slide_{i:05d}.h5", "w") as f:
            f.create_dataset("features", data=np.random.randn(n_patches, 768).astype(np.float32))
            f.create_dataset("coords", data=np.random.randint(0, 1000, (n_patches, 2)))
    return str(h5_dir)


# ---------------------------------------------------------------------------
# convert_h5_to_pt
# ---------------------------------------------------------------------------


class TestConvertH5ToPt:
    def test_creates_pt_files(self, h5_features_dir, tmp_path):
        pt_dir = str(tmp_path / "conch_v15")
        slide_ids = [f"slide_{i:05d}" for i in range(5)]
        n = convert_h5_to_pt(h5_features_dir, pt_dir, "conch_v15", slide_ids)
        assert n == 5
        for sid in slide_ids:
            assert os.path.exists(os.path.join(pt_dir, "pt_files", f"{sid}.pt"))

    def test_pt_tensor_shape_2d(self, h5_features_dir, tmp_path):
        pt_dir = str(tmp_path / "conch_v15")
        convert_h5_to_pt(h5_features_dir, pt_dir, "conch_v15", ["slide_00000"])
        t = torch.load(os.path.join(pt_dir, "pt_files", "slide_00000.pt"), weights_only=True)
        assert t.ndim == 2
        assert t.shape[1] == 768

    def test_skips_existing_files(self, h5_features_dir, tmp_path):
        pt_dir = str(tmp_path / "conch_v15")
        slide_ids = ["slide_00000"]
        convert_h5_to_pt(h5_features_dir, pt_dir, "conch_v15", slide_ids)
        n = convert_h5_to_pt(h5_features_dir, pt_dir, "conch_v15", slide_ids)
        assert n == 0  # all skipped

    def test_skips_missing_h5(self, h5_features_dir, tmp_path):
        pt_dir = str(tmp_path / "conch_v15")
        n = convert_h5_to_pt(h5_features_dir, pt_dir, "conch_v15", ["nonexistent_slide"])
        assert n == 0

    def test_returns_conversion_count(self, h5_features_dir, tmp_path):
        pt_dir = str(tmp_path / "conch_v15")
        slide_ids = [f"slide_{i:05d}" for i in range(3)]
        n = convert_h5_to_pt(h5_features_dir, pt_dir, "conch_v15", slide_ids)
        assert n == 3

    def test_float32_output(self, h5_features_dir, tmp_path):
        pt_dir = str(tmp_path / "conch_v15")
        convert_h5_to_pt(h5_features_dir, pt_dir, "conch_v15", ["slide_00000"])
        t = torch.load(os.path.join(pt_dir, "pt_files", "slide_00000.pt"), weights_only=True)
        assert t.dtype == torch.float32


# ---------------------------------------------------------------------------
# create_task_csv
# ---------------------------------------------------------------------------


class TestCreateTaskCsv:
    def test_creates_csv(self, mapping_csv, tmp_path, ds):
        out = str(tmp_path / "task.csv")
        df = create_task_csv(mapping_csv, out, "BRCA_predict_label", {0: "neg", 1: "pos"}, ds)
        assert os.path.isfile(out)
        assert isinstance(df, pd.DataFrame)

    def test_correct_columns(self, mapping_csv, tmp_path, ds):
        out = str(tmp_path / "task.csv")
        df = create_task_csv(mapping_csv, out, "BRCA_predict_label", {0: "neg", 1: "pos"}, ds)
        assert list(df.columns) == ["case_id", "slide_id", "label"]

    def test_drops_missing_labels(self, mapping_csv, tmp_path, ds):
        out = str(tmp_path / "task.csv")
        df = create_task_csv(mapping_csv, out, "HRD_label", {0: "neg", 1: "pos", 2: "other"}, ds)
        # HRD has pd.NA for i >= 15, so 5 rows dropped
        assert len(df) == 15

    def test_label_values_are_strings(self, mapping_csv, tmp_path, ds):
        out = str(tmp_path / "task.csv")
        df = create_task_csv(mapping_csv, out, "BRCA_predict_label", {0: "neg", 1: "pos"}, ds)
        assert set(df["label"].unique()) == {"neg", "pos"}

    def test_slide_id_no_extension(self, mapping_csv, tmp_path, ds):
        out = str(tmp_path / "task.csv")
        df = create_task_csv(mapping_csv, out, "BRCA_predict_label", {0: "neg", 1: "pos"}, ds)
        assert not any(df["slide_id"].str.endswith(".svs"))

    def test_brca_all_20_slides(self, mapping_csv, tmp_path, ds):
        out = str(tmp_path / "task.csv")
        df = create_task_csv(mapping_csv, out, "BRCA_predict_label", {0: "neg", 1: "pos"}, ds)
        assert len(df) == 20  # all 20 have BRCA labels


# ---------------------------------------------------------------------------
# create_strategy_splits
# ---------------------------------------------------------------------------


class TestCreateStratifiedSplits:
    @pytest.fixture
    def task_csv(self, mapping_csv, tmp_path, ds):
        out = str(tmp_path / "brca.csv")
        create_task_csv(mapping_csv, out, "BRCA_predict_label", {0: "neg", 1: "pos"}, ds)
        return out

    def test_creates_split_files(self, task_csv, tmp_path):
        splits_dir = str(tmp_path / "splits")
        paths = create_strategy_splits(task_csv, splits_dir, STANDARD, n_splits=3, seed=42)
        assert len(paths) == 3
        for p in paths:
            assert os.path.isfile(p)

    def test_split_columns(self, task_csv, tmp_path):
        splits_dir = str(tmp_path / "splits")
        create_strategy_splits(task_csv, splits_dir, STANDARD, n_splits=3, seed=42)
        df = pd.read_csv(os.path.join(splits_dir, "splits_0.csv"))
        assert set(df.columns) == {"train", "val", "test"}

    def test_no_overlap_between_splits(self, task_csv, tmp_path):
        splits_dir = str(tmp_path / "splits")
        create_strategy_splits(task_csv, splits_dir, STANDARD, n_splits=3, seed=42)
        for fold in range(3):
            df = pd.read_csv(os.path.join(splits_dir, f"splits_{fold}.csv"))
            train = set(df["train"].dropna())
            val = set(df["val"].dropna())
            test = set(df["test"].dropna())
            assert len(train & val) == 0, f"train/val overlap in fold {fold}"
            assert len(train & test) == 0, f"train/test overlap in fold {fold}"
            assert len(val & test) == 0, f"val/test overlap in fold {fold}"

    def test_all_slides_appear_in_exactly_one_test_fold(self, task_csv, tmp_path):
        splits_dir = str(tmp_path / "splits")
        n_splits = 3
        create_strategy_splits(task_csv, splits_dir, STANDARD, n_splits=n_splits, seed=42)
        all_test = []
        for fold in range(n_splits):
            df = pd.read_csv(os.path.join(splits_dir, f"splits_{fold}.csv"))
            all_test.extend(df["test"].dropna().tolist())
        task_df = pd.read_csv(task_csv)
        assert set(all_test) == set(task_df["slide_id"])

    def test_reproducible_with_same_seed(self, task_csv, tmp_path):
        dir1 = str(tmp_path / "splits1")
        dir2 = str(tmp_path / "splits2")
        create_strategy_splits(task_csv, dir1, STANDARD, n_splits=3, seed=42)
        create_strategy_splits(task_csv, dir2, STANDARD, n_splits=3, seed=42)
        for fold in range(3):
            df1 = pd.read_csv(os.path.join(dir1, f"splits_{fold}.csv"))
            df2 = pd.read_csv(os.path.join(dir2, f"splits_{fold}.csv"))
            assert df1.equals(df2)

    def test_different_seed_gives_different_splits(self, task_csv, tmp_path):
        dir1 = str(tmp_path / "splits1")
        dir2 = str(tmp_path / "splits2")
        create_strategy_splits(task_csv, dir1, STANDARD, n_splits=3, seed=42)
        create_strategy_splits(task_csv, dir2, STANDARD, n_splits=3, seed=99)
        df1 = pd.read_csv(os.path.join(dir1, "splits_0.csv"))
        df2 = pd.read_csv(os.path.join(dir2, "splits_0.csv"))
        assert not df1.equals(df2)
