"""H-3: one declared hyperparameter source per arm, one override path.

The FIRST test class is a behaviour-freeze guard: it pins each arm's currently
effective values against the live config objects the runners actually use. If
adopting the shared table ever changed a running experiment's hyperparameters,
these tests fail — that is the property that makes this refactor safe to land
while a benchmark campaign is already dispatched.
"""
from __future__ import annotations

import pytest

from autobench.pipeline.abmil.config import ABMILConfig
from autobench.pipeline.config import TrainConfig
from autobench.pipeline.dtfd.config import DTFDConfig
from autobench.pipeline.hparams import (
    ARM_HPARAMS,
    resolve_arm_hparams,
    provenance_table,
)


class TestValuesMatchWhatTheBenchmarkAlreadyRan:
    """Freeze-guard: the table must reproduce today's effective values exactly."""

    def test_clam_matches_shared_trainconfig(self):
        # CLAM reads exp_cfg.train (clam/train.py:83-84).
        t = TrainConfig()
        h = resolve_arm_hparams("clam", "classification")
        assert (h.lr, h.weight_decay, h.max_epochs, h.patience) == (
            t.lr, t.weight_decay, t.max_epochs, t.patience
        )

    def test_abmil_matches_its_own_config(self):
        c = ABMILConfig()
        h = resolve_arm_hparams("abmil", "classification")
        assert (h.lr, h.weight_decay, h.max_epochs, h.patience) == (
            c.lr, c.weight_decay, c.max_epochs, c.patience
        )

    def test_dtfd_matches_its_own_paper_exact_config(self):
        c = DTFDConfig()
        h = resolve_arm_hparams("dtfd", "classification")
        assert (h.lr, h.weight_decay, h.max_epochs, h.patience) == (
            c.lr, c.wd, c.max_epochs, c.patience
        )

    def test_titan_matches_its_head_config(self):
        from autobench.pipeline.titan.config import TitanHeadConfig
        c = TitanHeadConfig()
        h = resolve_arm_hparams("titan", "classification")
        assert (h.lr, h.weight_decay, h.patience) == (c.lr, c.weight_decay, c.patience)
        # max_epochs previously came from the shared TrainConfig — the mixed
        # provenance this module records.
        assert h.max_epochs == TrainConfig().max_epochs

    def test_nnmil_task_dependent_lr_is_preserved(self):
        # nnMIL genuinely uses a different lr per task type (prepare.py literals).
        assert resolve_arm_hparams("nnmil", "classification").lr == 3e-4
        assert resolve_arm_hparams("nnmil", "survival").lr == 1e-4
        # ...and 100 epochs, unlike every other arm's 200.
        assert resolve_arm_hparams("nnmil", "classification").max_epochs == 100


class TestUniformOverridePath:
    """Every arm honours an explicit override — the H-3 defect."""

    @pytest.mark.parametrize("fw", ["clam", "abmil", "dtfd", "titan", "nnmil"])
    def test_explicit_override_reaches_every_arm(self, fw):
        h = resolve_arm_hparams(fw, "classification", {"lr": 5e-4})
        assert h.lr == 5e-4, f"{fw} silently discarded an explicit lr override"

    @pytest.mark.parametrize("fw", ["clam", "abmil", "dtfd", "titan", "nnmil"])
    def test_none_overrides_do_not_disturb_the_arm_default(self, fw):
        """The CLI parses unset flags as None — those must not flatten arms
        onto one schedule (that would silently change DTFD's paper-exact lr)."""
        base = resolve_arm_hparams(fw, "classification")
        h = resolve_arm_hparams(
            fw, "classification",
            {"lr": None, "max_epochs": None, "patience": None, "weight_decay": None},
        )
        assert h == base

    def test_unknown_arm_fails_loud(self):
        with pytest.raises(KeyError):
            resolve_arm_hparams("bogus_net", "classification")

    def test_unknown_override_key_ignored(self):
        base = resolve_arm_hparams("dtfd", "classification")
        assert resolve_arm_hparams("dtfd", "classification", {"nonsense": 1}) == base


class TestProvenanceIsDeclared:
    def test_every_arm_declares_provenance(self):
        for key, h in ARM_HPARAMS.items():
            assert h.provenance.strip(), f"{key} has no declared provenance"

    def test_upstream_deviations_are_recorded_not_hidden(self):
        # DTFD is the one arm that is paper-exact today; CLAM is knowingly 2x
        # its upstream lr. Both facts must be visible in the table.
        assert ARM_HPARAMS[("dtfd", "classification")].matches_upstream is True
        clam = ARM_HPARAMS[("clam", "classification")]
        assert clam.matches_upstream is False
        assert "1e-4" in clam.provenance  # names the upstream value it deviates from

    def test_table_renders_for_the_methods_section(self):
        md = provenance_table()
        for fw in ("clam", "abmil", "dtfd", "titan", "nnmil"):
            assert fw in md
