"""Upstream-fidelity tests for the arms returned to their published defaults.

2026-07-28 decision (see ``provenance.py``): CLAM and ABMIL now train at their
own upstream argparse defaults instead of the shared/copied schedule they
previously inherited by accident. These tests read the *actual* upstream
default straight out of the vendored source via a static AST parse rather than
importing the vendored ``main.py`` entrypoints: both call ``parser.parse_args()``
at module level (not guarded by ``if __name__``), so importing them would parse
pytest's own argv; CLAM's also drags in its internal ``utils.*``/torch/pandas
import chain, which needs CLAM's own package layout on ``sys.path``. A static
parse touches none of that, and it means this test cannot silently drift if the
vendored copy is ever updated -- whichever side (ours or upstream) moves, this
fails and says which.
"""
from __future__ import annotations

import ast
import os

from autobench.pipeline.abmil.config import ABMILConfig
from autobench.pipeline.config import TrainConfig
from autobench.pipeline.provenance import ARM_PROVENANCE

# benchmarks/tests/test_provenance.py -> benchmarks/
_BENCHMARKS_ROOT = os.path.dirname(os.path.dirname(__file__))
_CLAM_MAIN = os.path.join(_BENCHMARKS_ROOT, "lib", "CLAM", "main.py")
_ABMIL_MAIN = os.path.join(_BENCHMARKS_ROOT, "lib", "AttentionDeepMIL", "main.py")


def _upstream_argparse_default(path: str, flag: str):
    """Statically read ``default=`` for ``add_argument(flag, ...)`` out of ``path``.

    Walks the AST looking for a call whose attribute is ``add_argument`` and
    whose first positional argument is the literal ``flag`` string, then
    returns the literal value bound to its ``default=`` keyword. Raises if the
    flag is missing or its default isn't a plain literal -- both mean the
    vendored file changed shape and this needs a human, not a silent pass.
    """
    with open(path) as f:
        tree = ast.parse(f.read(), filename=path)

    for node in ast.walk(tree):
        is_add_argument_call = (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        )
        if not is_add_argument_call:
            continue
        if not (
            node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == flag
        ):
            continue
        for kw in node.keywords:
            if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                return kw.value.value
        raise ValueError(f"{flag!r} in {path} has a non-literal default=")

    raise ValueError(f"no add_argument({flag!r}, ...) found in {path}")


class TestClamMatchesItsOwnUpstreamDefault:
    """CLAM (lib/CLAM/main.py:74): ``--lr`` default is 1e-4."""

    def test_the_vendored_source_still_says_1e_minus_4(self):
        """Pins the upstream value itself, so a vendored-copy update that moves
        it is caught here rather than only showing up as a silent mismatch."""
        assert _upstream_argparse_default(_CLAM_MAIN, "--lr") == 1e-4

    def test_trainconfig_lr_matches_the_vendored_default(self):
        assert TrainConfig().lr == _upstream_argparse_default(_CLAM_MAIN, "--lr")

    def test_provenance_declares_clam_upstream_faithful(self):
        clam = next(p for p in ARM_PROVENANCE if p.arm == "clam")
        assert clam.matches_upstream is True


class TestAbmilMatchesItsOwnUpstreamDefaults:
    """ABMIL (lib/AttentionDeepMIL/main.py:16-20): ``--epochs``/``--lr``/``--reg``."""

    def test_the_vendored_source_still_says_lr_5e_minus_4(self):
        assert _upstream_argparse_default(_ABMIL_MAIN, "--lr") == 5e-4

    def test_the_vendored_source_still_says_reg_1e_minus_4(self):
        assert _upstream_argparse_default(_ABMIL_MAIN, "--reg") == 1e-4

    def test_the_vendored_source_still_says_epochs_20(self):
        assert _upstream_argparse_default(_ABMIL_MAIN, "--epochs") == 20

    def test_abmilconfig_lr_matches_the_vendored_default(self):
        assert ABMILConfig().lr == _upstream_argparse_default(_ABMIL_MAIN, "--lr")

    def test_abmilconfig_weight_decay_matches_the_vendored_default(self):
        assert ABMILConfig().weight_decay == _upstream_argparse_default(_ABMIL_MAIN, "--reg")

    def test_abmilconfig_max_epochs_matches_the_vendored_default(self):
        assert ABMILConfig().max_epochs == _upstream_argparse_default(_ABMIL_MAIN, "--epochs")

    def test_architecture_stays_paper_exact_regardless(self):
        """The optimizer/schedule change must not have touched the architecture."""
        cfg = ABMILConfig()
        assert (cfg.M, cfg.L) == (500, 128)

    def test_provenance_declares_abmil_upstream_faithful(self):
        abmil = next(p for p in ARM_PROVENANCE if p.arm == "abmil")
        assert abmil.matches_upstream is True


class TestUnchangedArmsAreUntouchedByThisMigration:
    """DTFD/nnMIL were already upstream-faithful; this change must not move them."""

    def test_dtfd_still_reports_upstream_faithful(self):
        dtfd = next(p for p in ARM_PROVENANCE if p.arm == "dtfd")
        assert dtfd.matches_upstream is True

    def test_nnmil_still_reports_upstream_faithful(self):
        nnmil = next(p for p in ARM_PROVENANCE if p.arm == "nnmil")
        assert nnmil.matches_upstream is True

    def test_titan_still_has_no_upstream_recipe_to_match(self):
        titan = next(p for p in ARM_PROVENANCE if p.arm == "titan")
        assert titan.matches_upstream is False


class TestTrainerProvenanceIsNotOverclaimed:
    """Matching scalar defaults must not be reported as a faithful train loop."""

    def test_every_arm_declares_its_training_loop_relation(self):
        for arm in ARM_PROVENANCE:
            assert arm.trainer_provenance.strip(), arm.arm

    def test_benchmark_added_stopping_is_disclosed(self):
        for name in ("clam", "abmil", "dtfd"):
            arm = next(p for p in ARM_PROVENANCE if p.arm == name)
            assert "early stopping" in arm.trainer_provenance.lower()

    def test_table_separates_defaults_from_the_training_loop(self):
        from autobench.pipeline.provenance import provenance_table

        table = provenance_table()
        assert "Native scalar defaults?" in table
        assert "Trainer provenance" in table
        assert "fully upstream-faithful" not in table
