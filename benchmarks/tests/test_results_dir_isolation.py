"""CR-5 (audit 2026-07-23): every runner must accept an isolated ``results_dir``
so the per-fold ``metrics.json`` resume-cache is never shared across
experiments, seeds, or search variants. Previously only CLAM received the
isolated ``AUTOMIL_RESULTS_DIR``; nnMIL/ABMIL/DTFD/TITAN hardcoded the shared
``benchmark_dir/results/<subdir>`` path, so a re-run silently resumed stale
folds (defeating the OS censoring fix and colliding across variants).
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from autobench.pipeline.clam.runner import run_experiment as run_clam
from autobench.pipeline.abmil.runner import run_abmil_experiment
from autobench.pipeline.dtfd.runner import run_dtfd_experiment
from autobench.pipeline.nnmil.runner import run_nnmil_experiment
from autobench.pipeline.titan.runner import run_titan_experiment


@pytest.mark.parametrize(
    "fn",
    [run_clam, run_abmil_experiment, run_dtfd_experiment,
     run_nnmil_experiment, run_titan_experiment],
    ids=["clam", "abmil", "dtfd", "nnmil", "titan"],
)
def test_runner_accepts_isolated_results_dir(fn):
    params = inspect.signature(fn).parameters
    assert "results_dir" in params, f"{fn.__name__} must accept results_dir (CR-5)"
    # Default None → standalone runs still fall back to the shared benchmark dir.
    assert params["results_dir"].default is None


def test_dispatch_forwards_isolated_results_dir():
    """run_experiment.py must forward the isolated dir to all five frameworks."""
    run_experiment_py = Path(__file__).resolve().parents[1] / "scripts" / "run_experiment.py"
    text = run_experiment_py.read_text()
    # One forward per framework dispatch (clam + titan + abmil + dtfd + nnmil).
    assert text.count("results_dir=automil_results_dir") >= 5
