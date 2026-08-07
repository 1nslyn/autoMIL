"""H-3b: the opaque per-arm override channel, and the declared-space guard.

Before this, the search space was whatever the shared transport
(``ModelConfig`` + ``TrainConfig``) happened to carry — and that transport was
designed around CLAM. Measured coverage of each arm's own knobs:

    CLAM 12/15 · ABMIL 5/8 · TITAN 3/4 · DTFD 5/15 · **nnMIL 0/11**

nnMIL was literally zero: ``prepare_nnmil_experiment`` declared a
``hparam_overrides`` parameter and forwarded it internally, but no production
caller ever passed one. DTFD could not receive ``numGroup``, ``mDim``,
``grad_clip`` — its own paper's contributions.

An equal-effort search under that asymmetry reports channel width as a model
result, on exactly the axis the paper compares. These tests pin the channel open
and pin the guard shut.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from autobench.pipeline.abmil.config import ABMILConfig
from autobench.pipeline.config import (
    ExperimentConfig,
    Framework,
    ModelConfig,
    TaskConfig,
    TrainConfig,
)
from autobench.pipeline.dtfd.config import DTFDConfig
from autobench.pipeline.hparams import (
    all_overrides,
    apply_overrides,
    apply_overrides_to_exp_cfg,
    apply_overrides_to_plan,
    overrides_from_exp_cfg,
)


def _exp(*, hparams: dict | None = None, **train_kw) -> ExperimentConfig:
    return ExperimentConfig(
        task=TaskConfig(name="kras", label_col="KRAS", label_dict={"WT": 0, "MUT": 1}),
        encoder_key="uni_v2",
        embed_dim=1536,
        model=ModelConfig(model_type="clam_mb"),
        train=TrainConfig(**train_kw),
        framework=Framework.CLAM,
        hparam_overrides=hparams or {},
    )


def _nnmil_plan() -> dict:
    return {
        "feature_dimension": 1536, "hidden_dim": 384, "max_seq_length": 2048,
        "use_original_length": False, "batch_size": 8, "batch_sampler": "auc",
        "learning_rate": 3e-4, "weight_decay": 1e-4, "num_epochs": 100,
        "warmup_epochs": 10, "dropout": 0.25, "patience": 10, "num_classes": 2,
    }


class TestFreezeGuard:
    """The property that made this safe to land mid-campaign: a plain run moves
    no arm off its own defaults."""

    def test_a_plain_experiment_produces_no_overrides(self):
        assert all_overrides(_exp()) == {}

    @pytest.mark.parametrize("cfg", [ABMILConfig(), DTFDConfig()], ids=["abmil", "dtfd"])
    def test_arms_keep_their_own_defaults(self, cfg):
        assert apply_overrides(cfg, all_overrides(_exp())) is cfg

    def test_dtfd_keeps_its_paper_exact_values(self):
        out = apply_overrides(DTFDConfig(), all_overrides(_exp()), arm="dtfd")
        assert (out.lr, out.wd, out.numGroup) == (1e-4, 1e-4, 4)

    def test_nnmil_plan_is_untouched(self):
        plan = _nnmil_plan()
        assert apply_overrides_to_plan(plan, all_overrides(_exp())) == plan

    def test_default_model_config_contributes_nothing(self):
        """H-3b widened the diff to exp_cfg.model. Every ModelConfig in the grid
        is built with model_type only, so this must stay empty — otherwise a CLAM
        knob would be forwarded to DTFD and raise."""
        assert overrides_from_exp_cfg(_exp()) == {}


class TestOpaqueChannelReachesArmSpecificKnobs:
    """The H-3b headline: knobs with no home on the shared transport."""

    def test_dtfd_receives_its_own_paper_knobs(self):
        exp = _exp(hparams={"numGroup": 8, "grad_clip": 1.0, "mDim": 256})
        out = apply_overrides(DTFDConfig(), all_overrides(exp), arm="dtfd")
        assert (out.numGroup, out.grad_clip, out.mDim) == (8, 1.0, 256)

    def test_dtfd_receives_both_dropout_tiers_by_their_real_names(self):
        exp = _exp(hparams={"droprate": 0.5, "droprate_2": 0.3})
        out = apply_overrides(DTFDConfig(), all_overrides(exp), arm="dtfd")
        assert (out.droprate, out.droprate_2) == (0.5, 0.3)

    def test_abmil_receives_its_architecture_knobs(self):
        exp = _exp(hparams={"M": 256, "L": 64})
        out = apply_overrides(ABMILConfig(), all_overrides(exp), arm="abmil")
        assert (out.M, out.L) == (256, 64)

    def test_nnmil_receives_its_adaptive_knobs(self):
        """nnMIL's coverage was 0/11 — no caller ever fed its override seam."""
        exp = _exp(hparams={"warmup_epochs": 3, "batch_size": 32, "hidden_dim": 512})
        out = apply_overrides_to_plan(_nnmil_plan(), all_overrides(exp), arm="nnmil")
        assert (out["warmup_epochs"], out["batch_size"], out["hidden_dim"]) == (3, 32, 512)

    def test_nnmil_self_configured_values_survive_a_partial_override(self):
        out = apply_overrides_to_plan(
            _nnmil_plan(), all_overrides(_exp(hparams={"dropout": 0.1})), arm="nnmil",
        )
        assert out["dropout"] == 0.1
        assert out["batch_size"] == 8  # adaptive value untouched

    def test_opaque_beats_canonical_for_the_same_knob(self):
        exp = _exp(lr=5e-4, hparams={"lr": 9e-4})
        assert all_overrides(exp)["lr"] == 9e-4

    def test_canonical_and_opaque_merge(self):
        exp = _exp(lr=5e-4, hparams={"numGroup": 8})
        assert all_overrides(exp) == {"lr": 5e-4, "numGroup": 8}


class TestDeclaredSpaceIsEnforced:
    """A-2: an undeclared knob must not be silently applied OR silently dropped."""

    def test_locked_knob_raises_with_its_reason(self):
        exp = _exp(hparams={"distill": "MaxS"})
        with pytest.raises(ValueError, match="LOCKED"):
            apply_overrides(DTFDConfig(), all_overrides(exp), arm="dtfd")

    def test_the_lock_reason_is_actually_reported(self):
        with pytest.raises(ValueError) as exc:
            apply_overrides(DTFDConfig(), {"distill": "MaxS"}, arm="dtfd")
        assert "AFS" in str(exc.value)

    def test_dead_knob_is_refused(self):
        """total_instance is never read under AFS — searching it would waste
        budget on a no-op and pollute the trajectory."""
        with pytest.raises(ValueError, match="LOCKED|dead"):
            apply_overrides(DTFDConfig(), {"total_instance": 16}, arm="dtfd")

    def test_undeclared_knob_names_the_declared_set(self):
        """A knob that exists as a field but was never declared searchable."""
        with pytest.raises(ValueError) as exc:
            apply_overrides_to_plan(
                _nnmil_plan(), {"use_original_length": True}, arm="nnmil",
            )
        assert "declared" in str(exc.value).lower()

    def test_nonexistent_knob_still_raises(self):
        with pytest.raises(ValueError, match="cannot accept"):
            apply_overrides(ABMILConfig(), {"nonexistent_knob": 1}, arm="abmil")

    def test_seed_is_refused_on_every_arm(self):
        """Frozen substrate: an agent that could set the seed could select a
        favourable partition."""
        with pytest.raises(ValueError):
            apply_overrides(DTFDConfig(), {"seed": 7}, arm="dtfd")

    def test_unknown_arm_is_not_silently_narrowed(self):
        """smmile is vendored but unreachable from --framework; enforcing an
        empty declared space on it would refuse everything."""
        out = apply_overrides(DTFDConfig(), {"numGroup": 8}, arm="")
        assert out.numGroup == 8


class TestClamsInstanceClusteringBranchIsReachable:
    """Three live upstream knobs were hardcoded in _make_clam_args, so they sat
    outside the space while the rest of CLAM's surface sat inside it."""

    def test_defaults_reproduce_the_previous_literals(self):
        m = ModelConfig(model_type="clam_mb")
        assert (m.bag_loss, m.inst_loss, m.no_inst_cluster) == ("ce", None, False)

    def test_they_reach_the_clam_args_namespace(self, tmp_path):
        from autobench.pipeline.clam.train import _make_clam_args

        exp = _exp()
        exp.model.bag_loss = "svm"
        exp.model.inst_loss = "svm"
        exp.model.no_inst_cluster = True
        args = _make_clam_args(exp, str(tmp_path))
        assert (args.bag_loss, args.inst_loss, args.no_inst_cluster) == ("svm", "svm", True)

    def test_they_are_detected_as_overrides(self):
        exp = _exp()
        exp.model.bag_loss = "svm"
        assert all_overrides(exp) == {"bag_loss": "svm"}


class TestClamOpaqueChannel:
    """A1 (claims-alignment): CLAM's --hparams channel must actually land.

    CLAM trains off the shared ModelConfig + TrainConfig, so the opaque keys are
    partitioned across both transport dataclasses in place — with the same
    declared-space enforcement every other arm gets.
    """

    def test_opaque_keys_partition_across_model_and_train(self):
        exp = _exp(hparams={"lr": 9e-4, "dropout": 0.4, "bag_loss": "svm", "patience": 25})
        apply_overrides_to_exp_cfg(exp, arm="clam")
        assert exp.train.lr == 9e-4
        assert exp.train.patience == 25
        assert exp.model.dropout == 0.4
        assert exp.model.bag_loss == "svm"

    def test_the_wired_values_reach_the_clam_args_namespace(self, tmp_path):
        from autobench.pipeline.clam.train import _make_clam_args

        exp = _exp(hparams={"lr": 9e-4, "dropout": 0.4})
        apply_overrides_to_exp_cfg(exp, arm="clam")
        args = _make_clam_args(exp, str(tmp_path))
        assert (args.lr, args.drop_out) == (9e-4, 0.4)

    def test_identity_locked_keys_raise_at_train_time_too(self):
        exp = _exp(hparams={"no_inst_cluster": True})
        with pytest.raises(ValueError, match="LOCKED"):
            apply_overrides_to_exp_cfg(exp, arm="clam")

    def test_unknown_keys_fail_loud(self):
        exp = _exp(hparams={"numGroup": 8})  # DTFD's knob, not CLAM's
        with pytest.raises(ValueError, match="cannot accept"):
            apply_overrides_to_exp_cfg(exp, arm="clam")

    def test_empty_channel_is_a_noop(self):
        from dataclasses import replace

        exp = _exp()
        model_before, train_before = replace(exp.model), replace(exp.train)
        apply_overrides_to_exp_cfg(exp, arm="clam")
        assert exp.model == model_before
        assert exp.train == train_before


class TestCliHparamsFlag:
    """`--override "--numGroup 8"` used to be SystemExit(2): run_experiment.py
    parses with parse_args(), so an arm-specific flag killed the run. The channel
    is a JSON *value* for exactly that reason."""

    def _parse(self, raw):
        sys.path.insert(0, "benchmarks/scripts")
        from run_experiment import _parse_hparams
        return _parse_hparams(raw)

    def test_none_and_empty_are_empty(self):
        assert self._parse(None) == {} and self._parse("") == {}

    def test_flat_object_parses(self):
        assert self._parse('{"numGroup": 8, "grad_clip": 1.0}') == {
            "numGroup": 8, "grad_clip": 1.0,
        }

    def test_invalid_json_exits_with_a_message(self):
        with pytest.raises(SystemExit, match="not valid JSON"):
            self._parse("{numGroup: 8}")

    def test_non_object_exits(self):
        with pytest.raises(SystemExit, match="JSON object"):
            self._parse('[1, 2, 3]')

    def test_nested_value_exits(self):
        with pytest.raises(SystemExit, match="scalars"):
            self._parse('{"a": {"b": 1}}')

    def test_the_flag_is_actually_registered(self):
        """Guards against the channel existing in the library but not the CLI."""
        out = subprocess.run(
            [sys.executable, "benchmarks/scripts/run_experiment.py", "--help"],
            capture_output=True, text=True,
        )
        assert "--hparams" in out.stdout
        assert "--weight_decay" in out.stdout
        assert "--no_early_stopping" in out.stdout
