"""M-9, nnMIL follow-up: a partially-extracted cohort must not be baked into the plan.

The other four arms drop missing-feature slides in their per-split loaders, which
is where the M-9 guard was installed. nnMIL is architecturally different: it drops
ONCE, at plan time, for the whole cohort — before any fold exists — so a per-split
guard could never see it.

The consequence is the same and arguably worse, because the plan is CACHED
(`prepare_nnmil_experiment` early-returns on an existing `dataset_plan.json`). An
under-extracted cohort would be baked in and reused by every later experiment on
that (strategy, task, encoder) combination, silently reporting confident numbers
computed on whatever happened to be extracted.
"""
from __future__ import annotations

import os

import h5py
import numpy as np
import pandas as pd
import pytest

from autobench.pipeline.dataset_guards import SplitRetentionError
from autobench.pipeline.nnmil.prepare import prepare_nnmil_experiment
from autobench.pipeline.splits import create_strategy_splits
from _helpers import make_test_ds
from autobench.pipeline.config import build_registries


N_SLIDES = 40


@pytest.fixture
def cohort(tmp_path):
    """A 40-slide balanced cohort with an empty feature dir the test fills in."""
    bd = tmp_path / "benchmark"
    (bd / "dataset_csv").mkdir(parents=True)
    (bd / "splits").mkdir(parents=True)
    h5 = tmp_path / "features_conch_v15"
    h5.mkdir()

    rows = [{"case_id": f"P{i:03d}", "slide_id": f"slide_{i:05d}",
             "label": "neg" if i % 2 == 0 else "pos"} for i in range(N_SLIDES)]
    csv = bd / "dataset_csv" / "brca.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    registries = build_registries(make_test_ds())
    create_strategy_splits(str(csv), str(bd / "splits" / "standard" / "brca"),
                           registries.strategy_registry["standard"],
                           n_splits=3, seed=42)
    return str(bd), h5, rows


def _write_h5(d, sid):
    with h5py.File(d / f"{sid}.h5", "w") as f:
        f.create_dataset("features", data=np.random.randn(20, 768).astype(np.float32))
        f.create_dataset("coords", data=np.random.randint(0, 1000, (20, 2)))


def _prepare(bd, tmp_path):
    return prepare_nnmil_experiment(
        benchmark_dir=bd, task_name="brca", encoder_key="conch_v15",
        strategy="standard", label_col="label",
        label_dict={"neg": 0, "pos": 1}, embed_dim=768,
        features_base_dir=str(tmp_path), seed=42, n_splits=3,
    )


class TestSevereDropIsRefused:
    def test_a_third_of_the_cohort_missing_raises(self, cohort, tmp_path):
        bd, h5, rows = cohort
        for r in rows[:26]:                       # 26/40 = 65% retained
            _write_h5(h5, r["slide_id"])
        with pytest.raises(SplitRetentionError):
            _prepare(bd, tmp_path)

    def test_the_error_names_the_task_and_encoder(self, cohort, tmp_path):
        bd, h5, rows = cohort
        for r in rows[:26]:
            _write_h5(h5, r["slide_id"])
        with pytest.raises(SplitRetentionError) as exc:
            _prepare(bd, tmp_path)
        msg = str(exc.value)
        assert "brca" in msg and "conch_v15" in msg

    def test_a_wiped_out_class_raises_despite_high_overall_retention(self, cohort, tmp_path):
        """The fraction floor alone cannot catch this: dropping one class almost
        entirely can still leave most of the cohort in place."""
        bd, h5, rows = cohort
        for r in rows:
            if r["label"] == "neg" or int(r["slide_id"][-5:]) > 33:
                _write_h5(h5, r["slide_id"])
        with pytest.raises(SplitRetentionError):
            _prepare(bd, tmp_path)

    def test_nothing_is_cached_when_the_guard_fires(self, cohort, tmp_path):
        """A cached bad plan would be reused by every later experiment."""
        bd, h5, rows = cohort
        for r in rows[:26]:
            _write_h5(h5, r["slide_id"])
        with pytest.raises(SplitRetentionError):
            _prepare(bd, tmp_path)
        assert not [p for p in os.listdir(bd) if "dataset_plan" in p]


class TestCompleteCohortIsUnaffected:
    def test_a_fully_extracted_cohort_prepares_normally(self, cohort, tmp_path):
        bd, h5, rows = cohort
        for r in rows:
            _write_h5(h5, r["slide_id"])
        assert os.path.exists(_prepare(bd, tmp_path))
