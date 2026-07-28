"""CLI default-value tests for ``benchmarks/scripts/run_benchmark.py``.

**Contract changed 2026-07-28 (CFG-3).** These tests used to assert that the CLI
argparse defaults *matched* the ``config.py`` dataclass defaults, so that neither
could silently override the other. That is the weaker of the two available
guarantees: it made drift detectable, but only if someone remembered to update
the test. The CLI now carries **no training defaults at all** — every training
flag parses as ``None`` and is dropped unless explicitly supplied, so the
dataclass is the single source of truth and drift is structurally impossible.

The change was forced by a real shadowing: the grid is dispatched through this
script, so correcting ``TrainConfig.lr`` to CLAM's upstream ``1e-4`` was being
silently reset to ``2e-4`` on every launch by the CLI literal.

One claim in the original docstring is worth recording as false. It justified
``--lr 2e-4`` as "the CLAM README's 2e-4". It is not: ``lib/CLAM/main.py:74``
declares ``--lr default=1e-4``, the README documents no learning rate, and the
strings ``2e-4`` / ``0.0002`` appear NOWHERE in the vendored CLAM repository. The
benchmark's 2x learning rate had not merely lost its rationale — it had a stated
rationale that does not check out. See ``pipeline/provenance.py``.
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


def test_lr_has_no_cli_default(parser):
    """CFG-3: unset means unset. The dataclass owns the value.

    Previously this asserted a CLI literal of 2e-4, attributed to the CLAM
    README — a rationale that does not check out (see module docstring).
    """
    assert parser.lr is None, (
        f"--lr carries a CLI literal ({parser.lr}), which shadows "
        "TrainConfig.lr on the static-grid path — the exact defect CFG-3 fixed."
    )


def test_unset_lr_resolves_to_clams_upstream_value():
    """The value that actually reaches training must be CLAM's own upstream
    default (lib/CLAM/main.py:74), not a benchmark invention."""
    from autobench.pipeline.config import TrainConfig
    assert TrainConfig().lr == 1e-4


def test_default_n_folds_is_lab_standard_five(parser):
    """Lab standard (2026-07): 5-fold patient-stratified CV — a deliberate
    deviation from CLAM README's ``--k 10``. Larger per-fold test sets are
    more stable for the imbalanced mutation tasks (paper/shared/BACKGROUND.md:
    with few events, 10-fold starves per-fold counts). Guards against a silent
    drift back to 10 and keeps the CLI aligned with config.py's default."""
    from autobench.pipeline.config import BenchmarkConfig
    assert parser.n_folds is None, (
        f"--n_folds carries a CLI literal ({parser.n_folds}); CFG-3 requires "
        "the dataclass to own it."
    )
    assert BenchmarkConfig().n_folds == 5


def test_early_stopping_on_by_default(parser):
    """CLAM README: ``--early_stopping`` flag present. CLI inverts via
    ``--no_early_stopping``; default-off-of-no means on."""
    assert parser.no_early_stopping is False


def test_weighted_sample_on_by_default(parser):
    """CLAM README: ``--weighted_sample`` flag present."""
    assert parser.no_weighted_sample is False


def test_no_training_flag_carries_a_cli_default(parser):
    """CFG-3, generalised: not one training knob may shadow the dataclass.

    Asserted over the whole set rather than value-by-value, so adding a flag
    with a literal is caught without anyone remembering to extend this test.
    """
    from autobench.pipeline.config import TrainConfig
    shadowing = {
        name: getattr(parser, name)
        for name in ("seed", "max_epochs", "patience", "stop_epoch", "lr")
        if getattr(parser, name) is not None
    }
    assert not shadowing, f"CLI literals shadowing TrainConfig: {shadowing}"

    # ...and the values that actually reach training are the dataclass's.
    t = TrainConfig()
    assert (t.seed, t.max_epochs, t.patience, t.stop_epoch) == (42, 200, 20, 50)
