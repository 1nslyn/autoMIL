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
FIGURES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "paper", "preprint", "figures",
)
SCRIPT = os.path.join(FIGURES_DIR, "make_figures.py")

# roster.py lives beside make_figures.py and defines the baseline roster the
# script filters to by default; importing it here keeps the fixture's cohort
# names from drifting out of sync with the filter they are meant to satisfy.
sys.path.insert(0, FIGURES_DIR)


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


def _run(results_csv: str, per_fold_csv: str, out_dir: str, *extra: str):
    """Invoke make_figures.py on the synthetic CSVs.

    Passes ``--no-roster-filter`` by default: the fixture rows above are a
    minimal plot-mechanics grid (uppercase ``TCGA-LUAD``-style names, a couple
    of cells per cohort), NOT the 130-cell baseline roster, so the default
    roster filter would correctly reject them. Roster-filtered behaviour is
    covered by ``TestRosterFilterPath`` below and by
    ``benchmarks/tests/test_roster_filter.py``.
    """
    return subprocess.run(
        [sys.executable, SCRIPT,
         "--results", results_csv, "--per-fold", per_fold_csv, "--out-dir", out_dir,
         "--no-roster-filter", *extra],
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


def _roster_csvs(tmp_path):
    """The full 130-cell baseline roster, in the collector's output shape."""
    from roster import ROSTER_TASKS  # noqa: PLC0415 -- path set up at import time

    encoders = ("uni_v2", "virchow2", "hoptimus1")
    results, folds = [], []

    def add(dataset, task, task_type, model, encoder, loss=None):
        metric = "c_index" if task_type == "survival" else "auc_roc"
        results.append({
            "dataset": dataset, "framework": "x", "strategy": "standard",
            "task": task, "encoder": encoder, "model_type": model,
            "survival_loss": loss or "", "task_type": task_type,
            "test_auc_roc_mean": None if task_type == "survival" else 0.70,
            "test_c_index_mean": 0.60 if task_type == "survival" else None,
            "experiment_id": f"{dataset}__{task}__{encoder}__{model}__{loss}",
        })
        for fold in range(3):
            folds.append({
                "dataset": dataset, "framework": "x", "strategy": "standard",
                "task": task, "encoder": encoder, "model_type": model,
                "survival_loss": loss or "", "seed": 42,
                "experiment_id": f"{dataset}__{task}__{encoder}__{model}__{loss}",
                "split": "test", "fold": fold, "metric": metric,
                "value": 0.60 + 0.01 * fold,
            })

    for cohort, task in ROSTER_TASKS.items():
        for model in ("clam_mb", "simple_mil", "abmil", "dtfd_mil"):
            for enc in encoders:
                add(cohort, task, "classification", model, enc)
                add(cohort, "os", "survival", model, enc, "nllsurv")
        add(cohort, task, "classification", "titan", "titan")
        add(cohort, "os", "survival", "titan", "titan", "nllsurv")

    # off-roster noise the filter must strip: a cox arm and a second LUAD task
    for enc in encoders:
        add("tcga_luad", "os", "survival", "abmil", enc, "cox")
        add("tcga_luad", "egfr", "classification", "clam_mb", enc)

    results_csv = tmp_path / "roster_results.csv"
    per_fold_csv = tmp_path / "roster_per_fold.csv"
    pd.DataFrame(results).to_csv(results_csv, index=False)
    pd.DataFrame(folds).to_csv(per_fold_csv, index=False)
    return str(results_csv), str(per_fold_csv)


class TestRosterFilterPath:
    """The DEFAULT path: roster filtering on."""

    def test_full_roster_plots_and_reports_what_it_dropped(self, tmp_path):
        results_csv, per_fold_csv = _roster_csvs(tmp_path)
        out_dir = tmp_path / "out_roster"
        proc = subprocess.run(
            [sys.executable, SCRIPT, "--results", results_csv,
             "--per-fold", per_fold_csv, "--out-dir", str(out_dir)],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert "kept 130 of 136" in proc.stdout, proc.stdout
        assert (out_dir / "fig1_leaderboard_heatmap.png").stat().st_size > 0
        assert (out_dir / "fig4_survival_cindex.png").stat().st_size > 0

    def test_incomplete_roster_fails_loudly_with_no_partial_plot(self, tmp_path):
        results_csv, per_fold_csv = _roster_csvs(tmp_path)
        trimmed = pd.read_csv(results_csv)
        trimmed = trimmed[~(
            (trimmed["dataset"] == "tcga_lgg") & (trimmed["model_type"] == "titan")
        )]
        trimmed.to_csv(results_csv, index=False)
        out_dir = tmp_path / "out_incomplete"

        proc = subprocess.run(
            [sys.executable, SCRIPT, "--results", results_csv,
             "--per-fold", per_fold_csv, "--out-dir", str(out_dir)],
            capture_output=True, text=True,
        )
        assert proc.returncode != 0
        assert "roster is incomplete" in proc.stderr
        assert "tcga_lgg" in proc.stderr
        assert not out_dir.exists() or list(out_dir.iterdir()) == []

    def test_allow_incomplete_roster_plots_anyway(self, tmp_path):
        results_csv, per_fold_csv = _roster_csvs(tmp_path)
        trimmed = pd.read_csv(results_csv)
        trimmed = trimmed[~(
            (trimmed["dataset"] == "tcga_lgg") & (trimmed["model_type"] == "titan")
        )]
        trimmed.to_csv(results_csv, index=False)
        out_dir = tmp_path / "out_allow"

        proc = subprocess.run(
            [sys.executable, SCRIPT, "--results", results_csv,
             "--per-fold", per_fold_csv, "--out-dir", str(out_dir),
             "--allow-incomplete-roster"],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert (out_dir / "fig1_leaderboard_heatmap.png").stat().st_size > 0
