"""End-to-end test for ``make_figures.py`` (FIG-0): exercises the plotting path.

Deliberately decoupled from ``autobench`` -- it builds the two CSVs
``make_figures.py`` actually consumes (the shape ``summaries_to_frame`` /
``per_fold_frame`` produce) directly with pandas and invokes the script as a
subprocess, exactly as a user or CI would. ``benchmarks/tests/test_collect.py``
is where the collector itself (the producer of these CSVs) is tested.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

# Moved out of paper/preprint/figures/ (2026-07-28): pyproject's testpaths is
# ["tests"] and benchmarks/tests is the autobench gate, so a test sitting beside
# the figure scripts was auto-discovered by neither and silently never ran.
SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "paper", "preprint", "figures", "make_figures.py",
)


def _results_rows():
    return [
        # classification -- two datasets, two aggregators, one shared encoder
        {"dataset": "TCGA-LUAD", "framework": "clam", "strategy": "standard",
         "task": "kras", "encoder": "uni_v2", "model_type": "clam_mb",
         "survival_loss": "", "task_type": "classification",
         "test_auc_roc_mean": 0.70, "test_c_index_mean": None,
         "experiment_id": "clam__standard__kras__uni_v2__clam_mb__s42"},
        {"dataset": "TCGA-LUAD", "framework": "nnmil", "strategy": "standard",
         "task": "kras", "encoder": "uni_v2", "model_type": "simple_mil",
         "survival_loss": "", "task_type": "classification",
         "test_auc_roc_mean": 0.68, "test_c_index_mean": None,
         "experiment_id": "nnmil__standard__kras__uni_v2__simple_mil__s42"},
        {"dataset": "CPTAC-GBM", "framework": "clam", "strategy": "standard",
         "task": "tp53", "encoder": "uni_v2", "model_type": "clam_mb",
         "survival_loss": "", "task_type": "classification",
         "test_auc_roc_mean": 0.60, "test_c_index_mean": None,
         "experiment_id": "clam__standard__tp53__uni_v2__clam_mb__s42"},
        {"dataset": "CPTAC-GBM", "framework": "nnmil", "strategy": "standard",
         "task": "tp53", "encoder": "uni_v2", "model_type": "simple_mil",
         "survival_loss": "", "task_type": "classification",
         "test_auc_roc_mean": 0.62, "test_c_index_mean": None,
         "experiment_id": "nnmil__standard__tp53__uni_v2__simple_mil__s42"},
        # survival -- TITAN (its own encoder) + CLAM, two datasets
        {"dataset": "TCGA-LUAD", "framework": "titan", "strategy": "standard",
         "task": "os", "encoder": "titan", "model_type": "titan",
         "survival_loss": "cox", "task_type": "survival",
         "test_auc_roc_mean": None, "test_c_index_mean": 0.58,
         "experiment_id": "titan__standard__os__titan__titan__s42__cox"},
        {"dataset": "TCGA-LUAD", "framework": "clam", "strategy": "standard",
         "task": "os", "encoder": "uni_v2", "model_type": "clam_mb",
         "survival_loss": "nllsurv", "task_type": "survival",
         "test_auc_roc_mean": None, "test_c_index_mean": 0.61,
         "experiment_id": "clam__standard__os__uni_v2__clam_mb__s42__nllsurv"},
        {"dataset": "CPTAC-GBM", "framework": "titan", "strategy": "standard",
         "task": "os", "encoder": "titan", "model_type": "titan",
         "survival_loss": "cox", "task_type": "survival",
         "test_auc_roc_mean": None, "test_c_index_mean": 0.55,
         "experiment_id": "titan__standard__os__titan__titan__s42__cox__gbm"},
    ]


def _per_fold_rows():
    rows = []
    survival_experiments = [
        ("TCGA-LUAD", "titan", "titan", "cox", [0.57, 0.58, 0.59]),
        ("TCGA-LUAD", "clam", "uni_v2", "nllsurv", [0.60, 0.61, 0.62]),
        ("CPTAC-GBM", "titan", "titan", "cox", [0.54, 0.55, 0.56]),
    ]
    for dataset, framework, encoder, loss, values in survival_experiments:
        for fold, value in enumerate(values):
            rows.append({
                "dataset": dataset, "framework": framework, "strategy": "standard",
                "task": "os", "encoder": encoder, "model_type": framework if framework == "titan" else "clam_mb",
                "survival_loss": loss, "seed": 42, "split": "test",
                "fold": fold, "metric": "c_index", "value": value,
            })
    return rows


@pytest.fixture
def csv_paths(tmp_path):
    results_csv = tmp_path / "results.csv"
    per_fold_csv = tmp_path / "per_fold.csv"
    pd.DataFrame(_results_rows()).to_csv(results_csv, index=False)
    pd.DataFrame(_per_fold_rows()).to_csv(per_fold_csv, index=False)
    return str(results_csv), str(per_fold_csv)


def _run(results_csv: str, per_fold_csv: str, out_dir: str):
    return subprocess.run(
        [sys.executable, SCRIPT,
         "--results", results_csv, "--per-fold", per_fold_csv, "--out-dir", out_dir],
        capture_output=True, text=True,
    )


class TestMakeFiguresEndToEnd:
    def test_both_figures_produced_from_synthetic_data(self, tmp_path, csv_paths):
        results_csv, per_fold_csv = csv_paths
        out_dir = tmp_path / "out"
        proc = _run(results_csv, per_fold_csv, str(out_dir))

        assert proc.returncode == 0, proc.stderr

        fig1 = out_dir / "fig1_leaderboard_heatmap.png"
        fig4 = out_dir / "fig4_survival_cindex.png"
        assert fig1.is_file() and fig1.stat().st_size > 0
        assert fig4.is_file() and fig4.stat().st_size > 0

    def test_missing_task_type_column_fails_loudly_with_no_partial_plot(self, tmp_path):
        rows = [{k: v for k, v in row.items() if k != "task_type"} for row in _results_rows()]
        results_csv = tmp_path / "results_missing_column.csv"
        per_fold_csv = tmp_path / "per_fold.csv"
        pd.DataFrame(rows).to_csv(results_csv, index=False)
        pd.DataFrame(_per_fold_rows()).to_csv(per_fold_csv, index=False)
        out_dir = tmp_path / "out_missing_column"

        proc = _run(str(results_csv), str(per_fold_csv), str(out_dir))

        assert proc.returncode != 0
        assert "task_type" in proc.stderr
        # No plot from partial data: neither figure should have been written.
        assert not out_dir.exists() or list(out_dir.iterdir()) == []
