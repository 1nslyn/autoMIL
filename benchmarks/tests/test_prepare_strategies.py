"""prepare_all builds split folds under every requested strategy, and only those."""
from __future__ import annotations

import os

import pandas as pd
import pytest

from _helpers import make_test_ds

from autobench.config import TaskDef
from autobench.pipeline.prepare import prepare_all


def _setup(tmp_path):
    mapping_csv = str(tmp_path / "mapping.csv")
    pd.DataFrame([
        {"new_name": f"slide_{i:03d}.svs", "status": "mapped_unique_case_id",
         "primary_case_id": f"K{i:03d}", "BRCA_predict_label": i % 2}
        for i in range(60)
    ]).to_csv(mapping_csv, index=False)
    benchmark_dir = str(tmp_path / "benchmark")
    ds = make_test_ds(
        mapping_csv=mapping_csv, benchmark_dir=benchmark_dir,
        features_base_dir=str(tmp_path / "features"),
        tasks={"brca": TaskDef(name="brca", label_col="BRCA_predict_label",
                               label_map={0: "neg", 1: "pos"}, n_classes=2)},
        task_strategy_feasibility={"brca": ["standard", "devcv"]},
    )
    return ds, mapping_csv, benchmark_dir


def _prepare(ds, mapping_csv, benchmark_dir, strategies):
    prepare_all(
        benchmark_dir=benchmark_dir, mapping_csv=mapping_csv,
        features_base_dir=ds.features_base_dir, encoder_keys=[], ds=ds,
        seed=42, n_splits=5, strategies=strategies,
    )


def test_a_devcv_run_prepares_devcv_folds_only(tmp_path):
    ds, mapping_csv, benchmark_dir = _setup(tmp_path)
    _prepare(ds, mapping_csv, benchmark_dir, ("devcv",))
    devcv = os.path.join(benchmark_dir, "splits", "devcv", "brca")
    assert sorted(os.listdir(devcv)) == [f"splits_{k}.csv" for k in range(5)]
    assert not os.path.exists(os.path.join(benchmark_dir, "splits", "standard"))


def test_devcv_folds_share_one_test_set_that_is_never_validated(tmp_path):
    ds, mapping_csv, benchmark_dir = _setup(tmp_path)
    _prepare(ds, mapping_csv, benchmark_dir, ("standard", "devcv"))
    folds = [pd.read_csv(os.path.join(benchmark_dir, "splits", "devcv", "brca", f"splits_{k}.csv"))
             for k in range(5)]
    tests = [set(f["test"].dropna()) for f in folds]
    validated = set().union(*(set(f["val"].dropna()) for f in folds))
    standard0 = pd.read_csv(os.path.join(benchmark_dir, "splits", "standard", "brca", "splits_0.csv"))
    assert all(t == set(standard0["test"].dropna()) for t in tests)
    assert not validated & tests[0]


def test_an_undeclared_strategy_is_refused(tmp_path):
    ds, mapping_csv, benchmark_dir = _setup(tmp_path)
    with pytest.raises(ValueError, match="holdout"):
        _prepare(ds, mapping_csv, benchmark_dir, ("holdout",))
