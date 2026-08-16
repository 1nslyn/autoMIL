"""Every DECLARED tunable knob reaches its arm's constructed config (A4').

``search_space.py`` declares, per arm, what an agent may tune. This suite
closes the loop: for EVERY declared knob, an override sent through the real
hparams path (``ExperimentConfig.hparam_overrides`` — the opaque ``--hparams``
channel every campaign cell uses) must land on the object the trainer actually
consumes. A declared-but-unreachable knob is the H-3 defect wearing a
declaration as camouflage.

Config-level on purpose (R12 trim): the config is produced through each arm's
own application seam — the exact expression its runner/trainer executes — but
nothing trains. The knob lists are ITERATED from ``SEARCH_SPACE`` itself, so a
knob declared later is covered automatically (or fails loudly for a missing
sentinel, which is the correct prompt to extend this table).

nnMIL note: overrides layer ABOVE the prep-time planner, its ``batch_size``
clamp included — an explicit override replaces the clamped value and is not
re-clamped. Intended semantics, documented in the search_space.py docstring
and pinned by ``test_nnmil_batch_size_override_bypasses_the_planner_clamp``.
"""

from __future__ import annotations

import pytest

from autobench.pipeline.config import (
    ExperimentConfig,
    Framework,
    ModelConfig,
    TaskConfig,
    TrainConfig,
)
from autobench.pipeline.hparams import (
    all_overrides,
    apply_overrides,
    apply_overrides_to_exp_cfg,
    apply_overrides_to_plan,
)
from autobench.pipeline.search_space import SEARCH_SPACE

#: One type-plausible, non-default value per declared knob name (union across
#: arms). A knob declared in search_space.py without an entry here fails the
#: suite loudly — extend the table, never skip the knob.
_SENTINELS = {
    # shared optimisation / schedule
    "lr": 0.0123,
    "weight_decay": 0.0456,
    "wd": 0.0456,
    "max_epochs": 91,
    "early_stopping": False,
    "patience": 13,
    "dropout": 0.4321,
    # clam
    "model_size": "big",
    "B": 12,
    "bag_loss": "svm",
    "inst_loss": "ce",
    "optimizer": "sgd",
    "stop_epoch": 44,
    "weighted_sample": False,
    # abmil
    "M": 77,
    "L": 33,
    # dtfd
    "numGroup": 6,
    "mDim": 96,
    "numLayer_Res": 2,
    "droprate": 0.11,
    "droprate_2": 0.22,
    "grad_clip": 3.3,
    "lr_decay_ratio": 0.31,
    "lr_decay_step": 33,
    # nnmil (plan keys)
    "hidden_dim": 192,
    "max_seq_length": 512,
    "batch_size": 3,
    "batch_sampler": "balanced",
    "learning_rate": 0.0123,
    "num_epochs": 91,
    "warmup_epochs": 2,
}

#: What _make_clam_args names each transported field on the args namespace
#: CLAM's vendored train() consumes.
_CLAM_ARG_NAME = {
    "model_size": "model_size",
    "dropout": "drop_out",
    "B": "B",
    "bag_loss": "bag_loss",
    "inst_loss": "inst_loss",
    "lr": "lr",
    "weight_decay": "reg",
    "optimizer": "opt",
    "max_epochs": "max_epochs",
    "early_stopping": "early_stopping",
    "patience": "patience",
    "stop_epoch": "stop_epoch",
    "weighted_sample": "weighted_sample",
}

_FRAMEWORK = {
    "clam": Framework.CLAM,
    "abmil": Framework.ABMIL,
    "dtfd": Framework.DTFD,
    "titan": Framework.TITAN,
    "nnmil": Framework.NNMIL,
}

_MODEL_TYPE = {
    "clam": "clam_sb",
    "abmil": "abmil",
    "dtfd": "dtfd_mil",
    "titan": "titan",
    "nnmil": "ab_mil",
}

#: A realistic nnMIL plan `training_configuration` block, shaped like
#: prepare.py's self-configured output (batch_size already planner-clamped).
_NNMIL_PLAN = {
    "batch_size": 16,
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "num_epochs": 100,
    "warmup_epochs": 5,
    "dropout": 0.25,
    "patience": 10,
    "hidden_dim": 256,
    "max_seq_length": 4096,
    "batch_sampler": "random",
}


def _exp_with_override(arm: str, knob: str, value) -> ExperimentConfig:
    """An ExperimentConfig carrying one override on the REAL opaque channel."""
    return ExperimentConfig(
        task=TaskConfig(
            name="brca", label_col="label",
            label_dict={"neg": 0, "pos": 1}, n_classes=2,
        ),
        encoder_key="conch_v15",
        embed_dim=64,
        model=ModelConfig(model_type=_MODEL_TYPE[arm]),
        train=TrainConfig(),
        framework=_FRAMEWORK[arm],
        strategy="standard",
        hparam_overrides={knob: value},
    )


def _assert_reaches(arm: str, knob: str, sentinel) -> None:
    """Run the arm's own application seam and assert the knob landed."""
    exp = _exp_with_override(arm, knob, sentinel)

    if arm == "abmil":
        from autobench.pipeline.abmil.config import ABMILConfig

        cfg = apply_overrides(ABMILConfig(), all_overrides(exp), arm="abmil")
        assert getattr(cfg, knob) == sentinel

    elif arm == "dtfd":
        from autobench.pipeline.dtfd.config import DTFDConfig

        cfg = apply_overrides(DTFDConfig(), all_overrides(exp), arm="dtfd")
        assert getattr(cfg, knob) == sentinel

    elif arm == "titan":
        from autobench.pipeline.titan.config import (
            apply_train_overrides,
            resolve_head_config,
        )

        # Runner order: the train-side slice lands first (before the runner
        # saves config.json), then the trainer filters the head side.
        apply_train_overrides(exp)
        head = resolve_head_config(exp)
        if knob in ("max_epochs", "early_stopping"):
            # Mixed provenance: the trainers read these off exp_cfg.train.
            assert getattr(exp.train, knob) == sentinel
        else:
            assert getattr(head, knob) == sentinel

    elif arm == "clam":
        from autobench.pipeline.clam.train import _make_clam_args

        apply_overrides_to_exp_cfg(exp, arm="clam")
        args = _make_clam_args(exp, fold_dir="unused")
        assert getattr(args, _CLAM_ARG_NAME[knob]) == sentinel

    elif arm == "nnmil":
        out = apply_overrides_to_plan(
            dict(_NNMIL_PLAN), all_overrides(exp), arm="nnmil",
        )
        assert out[knob] == sentinel

    else:  # pragma: no cover - SEARCH_SPACE gained an arm this suite ignores
        pytest.fail(f"no wiring path defined for arm {arm!r}")


def test_every_declared_knob_has_a_sentinel():
    missing = sorted({
        knob
        for space in SEARCH_SPACE.values()
        for knob in space.tunable
        if knob not in _SENTINELS
    })
    assert not missing, (
        f"declared knobs without a sentinel value: {missing}. Extend "
        "_SENTINELS so the wiring test covers them."
    )


@pytest.mark.parametrize(
    "arm,knob",
    [
        (arm, knob)
        for arm in sorted(SEARCH_SPACE)
        for knob in sorted(SEARCH_SPACE[arm].tunable)
    ],
)
def test_declared_knob_reaches_the_constructed_config(arm, knob):
    _assert_reaches(arm, knob, _SENTINELS[knob])


def test_nnmil_batch_size_override_bypasses_the_planner_clamp():
    """Override layers ABOVE the clamp: the planner's value is replaced, not
    re-clamped — the intended semantics stated in search_space.py."""
    exp = _exp_with_override("nnmil", "batch_size", 3)
    out = apply_overrides_to_plan(dict(_NNMIL_PLAN), all_overrides(exp), arm="nnmil")
    assert out["batch_size"] == 3          # exactly the override
    assert _NNMIL_PLAN["batch_size"] == 16  # planner value untouched elsewhere


def test_titan_opaque_max_epochs_lands_on_the_train_transport():
    """Regression: an --hparams max_epochs on TITAN was silently dropped
    (head filtering excluded it and nothing else consumed the opaque
    channel). apply_train_overrides — the RUNNER-level seam, run before
    config.json is saved — routes it onto exp_cfg.train, which is what both
    TITAN trainers read."""
    from autobench.pipeline.titan.config import (
        apply_train_overrides,
        resolve_head_config,
    )

    exp = _exp_with_override("titan", "max_epochs", 7)
    apply_train_overrides(exp)
    head = resolve_head_config(exp)
    assert exp.train.max_epochs == 7
    assert not hasattr(head, "max_epochs")  # never double-applied to the head


def test_titan_head_resolution_does_not_mutate_the_transport():
    """resolve_head_config is head filtering ONLY. Its old exp_cfg.train
    mutation ran inside the trainers — AFTER the runner had saved
    config.json — so the archived provenance lied about max_epochs."""
    from autobench.pipeline.titan.config import resolve_head_config

    exp = _exp_with_override("titan", "max_epochs", 7)
    before = exp.train
    resolve_head_config(exp)
    assert exp.train is before
    assert exp.train.max_epochs == before.max_epochs
