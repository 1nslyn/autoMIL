"""CFG-3: `run_benchmark.py` must not shadow the dataclass defaults.

Found while returning CLAM and ABMIL to their upstream hyperparameters
(2026-07-28). `run_benchmark.py` — the entry point the whole static grid is
dispatched through, via `submit_benchmark.sh` — declared its own CLI literals for
every training knob:

    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--stop_epoch", type=int, default=50)

and then passed them into ``TrainConfig(...)`` unconditionally. So the dataclass
defaults were **dead on the static-grid path**: the grid's hyperparameters came
from this file, and correcting `TrainConfig.lr` to CLAM's upstream `1e-4` would
have been silently overridden back to `2e-4` on every launch.

`run_experiment.py` had already been fixed for exactly this (CFG-01 / D-01):
parse unset flags as ``None`` and only pass what was explicitly supplied, leaving
the dataclass as the single source of truth. This applies the same pattern to
`run_benchmark.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from autobench.pipeline.config import TrainConfig  # noqa: E402


def _parser():
    import run_benchmark
    return run_benchmark.parse_args, run_benchmark


class TestUnsetFlagsDoNotShadowTheDataclass:
    """The defect: every knob had a CLI literal that won over the dataclass."""

    @pytest.mark.parametrize(
        "flag", ["max_epochs", "lr", "seed", "patience", "stop_epoch"],
    )
    def test_training_flags_default_to_none(self, flag):
        import run_benchmark
        parser = run_benchmark._build_parser()
        args = parser.parse_args(["--dataset", "x"])
        assert getattr(args, flag) is None, (
            f"--{flag} carries a CLI literal, which silently overrides "
            f"TrainConfig.{flag} on the static-grid path"
        )

    def test_no_flags_resolves_to_the_dataclass_defaults(self):
        import run_benchmark
        parser = run_benchmark._build_parser()
        args = parser.parse_args(["--dataset", "x"])
        cfg = run_benchmark._train_config_from_args(args)
        assert cfg == TrainConfig()

    def test_the_upstream_lr_survives(self):
        """The regression this test exists for: CLAM's upstream 1e-4 must reach
        training, not be reset to the old shared 2e-4 by the CLI."""
        import run_benchmark
        parser = run_benchmark._build_parser()
        cfg = run_benchmark._train_config_from_args(parser.parse_args(["--dataset", "x"]))
        assert cfg.lr == TrainConfig().lr == 1e-4


class TestExplicitFlagsStillWin:
    def test_explicit_lr_is_honoured(self):
        import run_benchmark
        parser = run_benchmark._build_parser()
        cfg = run_benchmark._train_config_from_args(
            parser.parse_args(["--dataset", "x", "--lr", "7e-4"]),
        )
        assert cfg.lr == 7e-4

    def test_explicit_seed_is_honoured(self):
        import run_benchmark
        parser = run_benchmark._build_parser()
        cfg = run_benchmark._train_config_from_args(
            parser.parse_args(["--dataset", "x", "--seed", "43"]),
        )
        assert cfg.seed == 43

    def test_early_stopping_switch_still_works(self):
        import run_benchmark
        parser = run_benchmark._build_parser()
        on = run_benchmark._train_config_from_args(parser.parse_args(["--dataset", "x"]))
        off = run_benchmark._train_config_from_args(
            parser.parse_args(["--dataset", "x", "--no_early_stopping"]),
        )
        assert on.early_stopping is True and off.early_stopping is False

    def test_weighted_sample_switch_still_works(self):
        import run_benchmark
        parser = run_benchmark._build_parser()
        on = run_benchmark._train_config_from_args(parser.parse_args(["--dataset", "x"]))
        off = run_benchmark._train_config_from_args(
            parser.parse_args(["--dataset", "x", "--no_weighted_sample"]),
        )
        assert on.weighted_sample is True and off.weighted_sample is False
