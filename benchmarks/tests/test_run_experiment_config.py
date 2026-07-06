"""RED stubs for CFG-01 — Phase 11 Wave 0.

Tests guard against the argparse-defaults-mask-config-defaults bug
(ISSUE-015). All RED tests fail until Plan 11-02 changes the hard-coded
argparse defaults to `None` and conditionally passes values into TrainConfig.

CFG-01: run_experiment.py parse_args() defaults for training overrides
  (--lr, --n_folds, etc.) must NOT mask TrainConfig / ExperimentConfig
  dataclass defaults. Fix: change hard-coded defaults to None; only pass
  the value into TrainConfig when it was explicitly supplied.

  Current (broken) state:
    - parse_args() defaults --lr to 1e-4 → args.lr == 1e-4 (not None)
    - parse_args() defaults --n_folds to 5 → args.n_folds == 5 (not None)

  Expected (fixed) state:
    - parse_args() defaults --lr to None → args.lr is None when not supplied
    - parse_args() defaults --n_folds to None → args.n_folds is None when not supplied
    - Explicit flags still override: --lr 5e-4 → args.lr == 5e-4

No subprocess, no real data, no GPU — pure argparse-layer unit tests.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


# ---------------------------------------------------------------------------
# Import run_experiment.py as a module (it lives in benchmarks/scripts/, not
# in any package, so we use importlib.util to load it by file path).
# ---------------------------------------------------------------------------

def _load_run_experiment() -> ModuleType:
    """Load benchmarks/scripts/run_experiment.py as a module."""
    # benchmarks/tests/ → parents[0]; benchmarks/ → parents[1]
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    script_path = scripts_dir / "run_experiment.py"
    if not script_path.exists():
        pytest.skip(f"run_experiment.py not found at {script_path}")

    mod_name = "run_experiment_cfg01"
    if mod_name in sys.modules:
        return sys.modules[mod_name]

    spec = importlib.util.spec_from_file_location(mod_name, script_path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[mod_name] = mod
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except SystemExit:
        pass  # guard against __name__ == "__main__" blocks that call sys.exit
    return mod


@pytest.fixture(scope="module")
def run_experiment_mod() -> ModuleType:
    return _load_run_experiment()


# ---------------------------------------------------------------------------
# Helper: call parse_args() with a controlled argv list.
# parse_args() in run_experiment.py calls p.parse_args() with no argument,
# so it reads sys.argv[1:]. We patch sys.argv for the call.
# ---------------------------------------------------------------------------

def _parse(run_experiment_mod: ModuleType, cli_args: list[str]):
    """Invoke run_experiment_mod.parse_args() with the given CLI args."""
    saved = sys.argv[:]
    try:
        sys.argv = ["run_experiment.py"] + cli_args
        return run_experiment_mod.parse_args()
    finally:
        sys.argv = saved


# ---------------------------------------------------------------------------
# Minimal required args (the five positional-required flags in parse_args).
# --lr, --n_folds, etc. are intentionally OMITTED to test None-default behavior.
# ---------------------------------------------------------------------------

_REQUIRED_ARGS = [
    "--dataset", "x",
    "--task", "x",
    "--encoder", "x",
    "--model", "x",
    "--framework", "clam",
]


# ---------------------------------------------------------------------------
# CFG-01 RED tests: None defaults honor dataclass defaults
# ---------------------------------------------------------------------------

class TestCFG01NoneDefaults:
    def test_no_lr_flag_uses_trainconfig_default(self, run_experiment_mod):
        """RED: args.lr must be None when --lr is not supplied.

        Fails today because parse_args() defaults --lr to 1e-4.
        Fix (D-01): change default to None so TrainConfig.lr (2e-4) is honored.
        """
        args = _parse(run_experiment_mod, _REQUIRED_ARGS)
        assert args.lr is None, (
            f"When --lr is not supplied, args.lr must be None so the TrainConfig "
            f"dataclass default (2e-4) is honored. Got args.lr={args.lr!r}. "
            f"Fix: change parse_args --lr default to None (D-01, CFG-01)."
        )

    def test_no_n_folds_flag_uses_experimentconfig_default(self, run_experiment_mod):
        """RED: args.n_folds must be None when --n_folds is not supplied.

        Fails today because parse_args() defaults --n_folds to 5.
        Fix (D-01): change default to None so ExperimentConfig.n_folds (5) is honored.
        """
        args = _parse(run_experiment_mod, _REQUIRED_ARGS)
        assert args.n_folds is None, (
            f"When --n_folds is not supplied, args.n_folds must be None so the "
            f"ExperimentConfig dataclass default (5) is honored. "
            f"Got args.n_folds={args.n_folds!r}. "
            f"Fix: change parse_args --n_folds default to None (D-01, CFG-01)."
        )


# ---------------------------------------------------------------------------
# CFG-01 GREEN tests: explicit flags still override (regression coverage)
# ---------------------------------------------------------------------------

class TestCFG01ExplicitFlagsHonored:
    def test_explicit_lr_flag_is_honored(self, run_experiment_mod):
        """GREEN: --lr 5e-4 must set args.lr == 5e-4.

        Passes today and must continue to pass after the CFG-01 fix.
        Verifies that switching the default to None does not break explicit overrides.
        """
        args = _parse(run_experiment_mod, _REQUIRED_ARGS + ["--lr", "5e-4"])
        assert args.lr == pytest.approx(5e-4), (
            f"Explicit --lr 5e-4 must be honored; got args.lr={args.lr!r}."
        )

    def test_explicit_n_folds_flag_is_honored(self, run_experiment_mod):
        """GREEN: --n_folds 3 must set args.n_folds == 3.

        Passes today and must continue to pass after the CFG-01 fix.
        Verifies that switching the default to None does not break explicit overrides.
        """
        args = _parse(run_experiment_mod, _REQUIRED_ARGS + ["--n_folds", "3"])
        assert args.n_folds == 3, (
            f"Explicit --n_folds 3 must be honored; got args.n_folds={args.n_folds!r}."
        )
