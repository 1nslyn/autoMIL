"""DATA-ID: the CLI path must stamp the dataset name into BenchmarkConfig.

Found during the preprint-130 baseline-repair campaign (2026-08-06).
``BenchmarkConfig.from_dataset_config`` stamps ``dataset=ds.name``, but
``run_benchmark.py`` — the entry point every static-grid sbatch dispatches
through — constructs ``BenchmarkConfig(...)`` directly and never passed
``dataset``. The field's ``""`` default was then threaded into every
``ExperimentConfig`` and written to every ``config.json`` / ``summary.json``.

Consequence: ``repair_baselines.py verify`` rejected 12 freshly-trained,
otherwise-valid CPTAC-GBM CLAM/ABMIL cells with

    config dataset mismatch: expected 'cptac_gbm', got ''

and its post-cohort assertion aborted the sbatch before the remaining four
cohorts ran (job 53319143).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _helpers import make_test_ds  # noqa: E402


def _cli_config(ds, argv):
    import run_benchmark

    parser = run_benchmark._build_parser()
    args = parser.parse_args(argv)
    return run_benchmark._benchmark_config_from_args(args, ds)


class TestCLIStampsDatasetIdentity:
    def test_dataset_name_reaches_benchmark_config(self):
        ds = make_test_ds(name="cptac_gbm")
        cfg = _cli_config(ds, ["--dataset", "cptac_gbm"])
        assert cfg.dataset == "cptac_gbm"

    def test_dataset_is_never_the_empty_default(self):
        """The exact defect: '' silently satisfied every writer downstream."""
        ds = make_test_ds(name="tcga_luad")
        cfg = _cli_config(ds, ["--dataset", "tcga_luad", "--no_wandb"])
        assert cfg.dataset != ""

    def test_cli_matches_from_dataset_config_stamp(self):
        """The CLI path and the library path must agree on identity."""
        from autobench.pipeline.config import BenchmarkConfig

        ds = make_test_ds(name="tcga_hnsc")
        cli_cfg = _cli_config(ds, ["--dataset", "tcga_hnsc"])
        lib_cfg = BenchmarkConfig.from_dataset_config(ds)
        assert cli_cfg.dataset == lib_cfg.dataset == ds.name
