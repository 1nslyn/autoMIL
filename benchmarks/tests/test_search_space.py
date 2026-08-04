"""A-2 / H-3b: the search space must be DECLARED, and the declaration must be true.

The finding this encodes: before ``search_space.py`` the space was never declared,
so the de-facto space was "whatever the transport happened to carry" — and the
transport (``ModelConfig`` + ``TrainConfig``) is CLAM-shaped. An equal-effort
search in which CLAM can be tuned on twelve knobs and nnMIL on none reports
channel width as a model result.

The load-bearing class here is :class:`TestDeclarationMatchesReality`. A
declaration that drifts from the configs is worse than none — it would let the
paper print a search-space table that is not what ran. Every field on every arm
must be **either** declared tunable **or** explicitly locked with a reason.
"""
from __future__ import annotations

from dataclasses import fields as dc_fields

import pytest

from autobench.pipeline.abmil.config import ABMILConfig
from autobench.pipeline.config import ModelConfig, TrainConfig
from autobench.pipeline.dtfd.config import DTFDConfig
from autobench.pipeline.search_space import (
    SEARCH_SPACE,
    coverage_table,
    declared_knobs,
    lock_reason,
)
from autobench.pipeline.titan.config import TitanHeadConfig

#: Fields that are plumbing rather than hyperparameters — not part of any recipe.
_PLUMBING = {"seed", "model_type", "head"}


def _clam_fields() -> set[str]:
    """CLAM has no config dataclass of its own: the transport IS its config."""
    return (
        {f.name for f in dc_fields(ModelConfig)}
        | {f.name for f in dc_fields(TrainConfig)}
    )


def _nnmil_plan_keys() -> set[str]:
    """nnMIL's config is computed into a plan dict, not a dataclass."""
    from autobench.pipeline.nnmil.prepare import _generate_training_config

    # Only the key SET matters here, so the statistics fed in are irrelevant.
    stats = {"feature_dimension": 1536, "recommended_max_seq_length": 4096}
    keys = set(_generate_training_config(stats, n_samples=400, n_classes=2, metric="auc"))
    # Survival plans add their own keys; both shapes must be accounted for.
    for loss in ("cox", "nllsurv"):
        keys |= set(_generate_training_config(
            stats, n_samples=400, task_type="survival", survival_loss=loss,
        ))
    return keys


ARM_FIELDS = {
    "clam": _clam_fields,
    "abmil": lambda: {f.name for f in dc_fields(ABMILConfig)},
    "dtfd": lambda: {f.name for f in dc_fields(DTFDConfig)},
    "titan": lambda: (
        {f.name for f in dc_fields(TitanHeadConfig)}
        # TITAN reads these two straight off the shared TrainConfig
        | {"max_epochs", "early_stopping"}
    ),
    "nnmil": _nnmil_plan_keys,
}


class TestDeclarationMatchesReality:
    """The declaration must not drift from the code it describes."""

    @pytest.mark.parametrize("arm", sorted(SEARCH_SPACE))
    def test_every_declared_knob_actually_exists(self, arm):
        real = ARM_FIELDS[arm]()
        phantom = sorted(declared_knobs(arm) - real)
        assert not phantom, f"{arm} declares knobs that do not exist: {phantom}"

    @pytest.mark.parametrize("arm", sorted(SEARCH_SPACE))
    def test_every_real_field_is_declared_or_locked(self, arm):
        space = SEARCH_SPACE[arm]
        accounted = space.tunable | set(space.locked) | _PLUMBING
        missing = sorted(ARM_FIELDS[arm]() - accounted)
        assert not missing, (
            f"{arm} has field(s) {missing} that are neither declared searchable "
            f"nor explicitly locked. Undeclared is indistinguishable from an "
            f"oversight — add them to `tunable` or to `locked` with a reason."
        )

    @pytest.mark.parametrize("arm", sorted(SEARCH_SPACE))
    def test_every_lock_carries_a_reason(self, arm):
        blank = [k for k, v in SEARCH_SPACE[arm].locked.items() if not v.strip()]
        assert not blank, f"{arm} locks {blank} without saying why"

    @pytest.mark.parametrize("arm", sorted(SEARCH_SPACE))
    def test_tunable_and_locked_are_disjoint(self, arm):
        space = SEARCH_SPACE[arm]
        assert not (space.tunable & set(space.locked))


class TestTheAsymmetryIsClosed:
    """H-3b measured 12/15 CLAM vs 0/11 nnMIL. The declaration fixes the target."""

    def test_dtfds_own_paper_contributions_are_searchable(self):
        """numGroup/mDim/droprate/grad_clip were entirely unreachable before."""
        assert {"numGroup", "mDim", "numLayer_Res", "droprate", "droprate_2",
                "grad_clip", "lr_decay_ratio", "lr_decay_step"} <= declared_knobs("dtfd")

    def test_nnmil_is_no_longer_empty(self):
        assert len(declared_knobs("nnmil")) >= 10

    def test_abmils_architecture_knobs_are_searchable(self):
        assert {"M", "L", "dropout"} <= declared_knobs("abmil")

    def test_clams_instance_loss_choices_are_searchable_but_branch_is_fixed(self):
        """The loss implementation can vary without erasing the CLAM mechanism.

        ``no_inst_cluster`` and ``bag_weight`` are different: they can remove a
        defining loss branch, so the architecture-preserving campaign locks them
        while leaving the loss family and instance-sampling recipe searchable.
        """
        assert {"bag_loss", "inst_loss", "B"} <= declared_knobs("clam")
        assert "no_inst_cluster" not in declared_knobs("clam")
        assert "bag_weight" not in declared_knobs("clam")
        assert lock_reason("clam", "no_inst_cluster")
        assert lock_reason("clam", "bag_weight")

    def test_no_arm_is_starved_relative_to_clam(self):
        """Not equality — arms genuinely differ in surface. But an order-of-
        magnitude gap is the confound H-3b describes."""
        n_clam = len(declared_knobs("clam"))
        for arm in SEARCH_SPACE:
            if arm == "titan":
                continue  # a linear probe genuinely has ~5 knobs; see A-3
            assert len(declared_knobs(arm)) >= n_clam / 3


class TestLocksAreDeliberate:
    def test_dtfd_distill_is_locked_with_a_correctness_reason(self):
        reason = lock_reason("dtfd", "distill")
        assert reason and "AFS" in reason

    def test_dtfd_total_instance_is_locked_as_dead(self):
        assert "dead" in (lock_reason("dtfd", "total_instance") or "")

    def test_titan_head_is_locked(self):
        assert lock_reason("titan", "head")

    def test_seed_is_never_searchable(self):
        """The evaluation protocol is frozen substrate, not a recipe knob."""
        for arm in SEARCH_SPACE:
            assert "seed" not in declared_knobs(arm)

    def test_unknown_arm_declares_nothing(self):
        assert declared_knobs("smmile") == frozenset()
        assert lock_reason("smmile", "lr") is None


class TestMethodsTable:
    def test_renders_every_arm(self):
        table = coverage_table()
        for arm in SEARCH_SPACE:
            assert f"| {arm} |" in table

    def test_states_the_lock_reasons(self):
        assert "AFS" in coverage_table()
