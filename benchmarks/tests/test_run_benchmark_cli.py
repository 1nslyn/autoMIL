"""CLI default-value tests for ``benchmarks/scripts/run_benchmark.py``.

The CLI argparse defaults must match the config.py dataclass defaults so a
CLI-side default never silently overrides the config. ``--lr`` stays at the
CLAM README's 2e-4; ``--n_folds`` is the lab-standard 5-fold (2026-07) — a
deliberate deviation from CLAM's ``--k 10``, with CLI and config.py both
defaulting to 5. This guards against a default drifting apart from the config.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types

import pytest


def _load_run_benchmark_module():
    """Import benchmarks/scripts/run_benchmark.py without executing main()."""
    script_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "scripts"
    )
    path = os.path.join(script_dir, "run_benchmark.py")
    spec = importlib.util.spec_from_file_location("run_benchmark_for_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def parser():
    mod = _load_run_benchmark_module()
    # parse_args() reads sys.argv; pass --dataset to satisfy required arg
    # without triggering further setup.
    monkey_argv = ["run_benchmark.py", "--dataset", "placeholder"]
    real_argv = sys.argv
    sys.argv = monkey_argv
    try:
        args = mod.parse_args()
    finally:
        sys.argv = real_argv
    return args


def test_default_lr_matches_clam_readme(parser):
    """CLAM README: ``--lr 2e-4``. CLI default must match."""
    assert parser.lr == 2e-4, (
        f"CLI --lr default {parser.lr} drifted from CLAM README's 2e-4. "
        "If lowering, update config.py TrainConfig.lr and the methods "
        "section together."
    )


def test_default_n_folds_is_lab_standard_five(parser):
    """Lab standard (2026-07): 5-fold patient-stratified CV — a deliberate
    deviation from CLAM README's ``--k 10``. Larger per-fold test sets are
    more stable for the imbalanced mutation tasks (paper/shared/BACKGROUND.md:
    with few events, 10-fold starves per-fold counts). Guards against a silent
    drift back to 10 and keeps the CLI aligned with config.py's default."""
    assert parser.n_folds == 5, (
        f"CLI --n_folds default {parser.n_folds} != 5. The benchmark's lab "
        "standard is 5-fold (config.py ExperimentConfig/BenchmarkConfig also "
        "default to 5); keep the CLI default aligned."
    )


def test_early_stopping_on_by_default(parser):
    """CLAM README: ``--early_stopping`` flag present. CLI inverts via
    ``--no_early_stopping``; default-off-of-no means on."""
    assert parser.no_early_stopping is False


def test_weighted_sample_on_by_default(parser):
    """CLAM README: ``--weighted_sample`` flag present."""
    assert parser.no_weighted_sample is False


def test_seed_and_max_epochs_match_config(parser):
    """Less critical but locked: any drift requires a deliberate update."""
    assert parser.seed == 42
    assert parser.max_epochs == 200
    assert parser.patience == 20
    assert parser.stop_epoch == 50
