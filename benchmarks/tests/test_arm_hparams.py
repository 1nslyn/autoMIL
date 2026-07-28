"""H-3: one uniform override mechanism across arms, defaults untouched.

Design under test (see hparams.py): each arm's own config dataclass stays the
single source of truth for its defaults — no shared value table, so arm-specific
knobs (DTFD's numGroup/grad_clip, ABMIL's M/L) keep their full tunable surface and
nnMIL keeps its data-adaptive self-configuration. Only the *application of an
explicit override* is unified, and an inapplicable override now fails loudly
instead of vanishing.

The freeze-guard class is the property that made this safe to land mid-campaign:
with no explicit override, every arm resolves to exactly its own defaults.
"""
from __future__ import annotations

import pytest

from autobench.pipeline.abmil.config import ABMILConfig
from autobench.pipeline.config import ModelConfig, TaskConfig, TrainConfig
from autobench.pipeline.dtfd.config import DTFDConfig
from autobench.pipeline.hparams import (
    apply_overrides,
    apply_overrides_to_plan,
    explicit_overrides,
    overrides_from_exp_cfg,
)


class _ExpCfgStub:
    def __init__(self, train):
        self.train = train


class TestNoOverrideLeavesDefaultsExactlyAlone:
    """Freeze-guard: a plain run must not move any arm off its own schedule."""

    def test_unset_flags_produce_no_overrides(self):
        # TrainConfig() == "nothing was passed on the CLI".
        assert overrides_from_exp_cfg(_ExpCfgStub(TrainConfig())) == {}

    @pytest.mark.parametrize("cfg", [ABMILConfig(), DTFDConfig()],
                             ids=["abmil", "dtfd"])
    def test_empty_overrides_return_the_config_unchanged(self, cfg):
        assert apply_overrides(cfg, {}) is cfg
        assert apply_overrides(cfg, None) is cfg

    def test_dtfd_keeps_its_paper_exact_values_without_an_override(self):
        """The regression that would have been catastrophic: threading the shared
        defaults through would silently retune DTFD (lr 1e-4 -> 2e-4, wd 1e-4 ->
        1e-5) and invalidate every dispatched DTFD run."""
        out = apply_overrides(DTFDConfig(), overrides_from_exp_cfg(_ExpCfgStub(TrainConfig())))
        assert out.lr == 1e-4
        assert out.wd == 1e-4

    def test_abmil_keeps_its_own_values_without_an_override(self):
        out = apply_overrides(ABMILConfig(), overrides_from_exp_cfg(_ExpCfgStub(TrainConfig())))
        assert (out.lr, out.weight_decay, out.dropout) == (2e-4, 1e-5, 0.0)


class TestExplicitOverrideReachesEveryArm:
    """The H-3 defect: ABMIL/DTFD silently discarded tuning knobs."""

    def test_detects_an_explicitly_set_lr(self):
        t = TrainConfig(lr=5e-4)
        assert overrides_from_exp_cfg(_ExpCfgStub(t)) == {"lr": 5e-4}

    def test_abmil_honours_lr_override(self):
        out = apply_overrides(ABMILConfig(), {"lr": 5e-4}, arm="abmil")
        assert out.lr == 5e-4
        assert out.M == 500  # arm-specific knobs untouched

    def test_dtfd_honours_lr_override(self):
        out = apply_overrides(DTFDConfig(), {"lr": 5e-4}, arm="dtfd")
        assert out.lr == 5e-4
        assert out.numGroup == 4  # arm-specific knob survives

    def test_weight_decay_alias_maps_onto_dtfds_wd_field(self):
        """DTFD follows its upstream repo and calls the field `wd`."""
        out = apply_overrides(DTFDConfig(), {"weight_decay": 3e-4}, arm="dtfd")
        assert out.wd == 3e-4

    def test_full_arm_specific_surface_is_tunable(self):
        """Not flattened to a lowest-common-denominator field set: an agent can
        tune DTFD's own knobs, which a shared 5-field table could not express."""
        out = apply_overrides(DTFDConfig(), {"numGroup": 8, "grad_clip": 1.0}, arm="dtfd")
        assert (out.numGroup, out.grad_clip) == (8, 1.0)


class TestInapplicableOverrideFailsLoudly:
    def test_unknown_knob_raises_instead_of_vanishing(self):
        with pytest.raises(ValueError, match="cannot accept"):
            apply_overrides(ABMILConfig(), {"bag_weight": 0.9}, arm="abmil")

    def test_error_names_the_arm_and_the_knob(self):
        with pytest.raises(ValueError) as exc:
            apply_overrides(DTFDConfig(), {"nonexistent_knob": 1}, arm="dtfd")
        assert "dtfd" in str(exc.value)
        assert "nonexistent_knob" in str(exc.value)


class TestNnmilPlanPath:
    """nnMIL's config is computed from data stats — overrides layer on top."""

    def _plan(self):
        return {"learning_rate": 3e-4, "num_epochs": 100, "warmup_epochs": 10,
                "weight_decay": 1e-4, "dropout": 0.25}

    def test_adaptive_values_survive_when_nothing_is_overridden(self):
        plan = self._plan()
        assert apply_overrides_to_plan(plan, {}) == plan

    def test_lr_alias_maps_onto_learning_rate(self):
        out = apply_overrides_to_plan(self._plan(), {"lr": 7e-4})
        assert out["learning_rate"] == 7e-4
        assert out["warmup_epochs"] == 10  # self-configured value preserved

    def test_max_epochs_alias_maps_onto_num_epochs(self):
        out = apply_overrides_to_plan(self._plan(), {"max_epochs": 50})
        assert out["num_epochs"] == 50

    def test_does_not_mutate_the_caller(self):
        plan = self._plan()
        apply_overrides_to_plan(plan, {"lr": 9e-4})
        assert plan["learning_rate"] == 3e-4

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError):
            apply_overrides_to_plan(self._plan(), {"bag_weight": 0.5})


class TestExplicitOverridesHelper:
    def test_drops_none_values(self):
        assert explicit_overrides(lr=None, max_epochs=50) == {"max_epochs": 50}

    def test_empty_when_all_none(self):
        assert explicit_overrides(lr=None, patience=None) == {}
