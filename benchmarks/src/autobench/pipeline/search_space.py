"""The declared per-arm search space (A-2, H-3b).

**Why this exists.** The search space must be declared independently of
*whatever the transport happens to carry*. The
transport is ``ModelConfig`` + ``TrainConfig``, which were designed around CLAM
(``bag_weight``, ``B``, ``model_size`` are CLAM concepts), so CLAM's whole surface
is natively in the channel and nobody else's is. The measured transport coverage is:

    CLAM   12/15   ABMIL 5/8   TITAN 3/4   DTFD 5/15   nnMIL 0/11

nnMIL was zero: ``prepare_nnmil_experiment`` declared a ``hparam_overrides``
parameter and forwarded it internally, but no production caller ever passed one.

That asymmetry is not a nuisance, it is a confound on the axis the paper compares:
an equal-effort search in which one arm can be tuned on twelve knobs and another on
none would report *channel width* as a model result.

**What this module is.** A machine-readable declaration, per arm, of

* which knobs an agent may tune (``tunable``), and
* which are deliberately **locked**, each with the reason (``locked``).

The target is a *declared* space, not literally every field. DTFD's ``distill`` is
locked to ``AFS`` for a correctness reason recorded in its own config; TITAN's
``head`` has only one implementation. Declaring the lock and its reason is the
scientific act — an undeclared lock is indistinguishable from an oversight.

Enforcement lives in ``hparams.apply_overrides``: a request for an undeclared knob
raises rather than being silently dropped, which is the H-3 failure mode this
whole line of work exists to remove.

**Relation to campaign identity locks (A4).** This module is mode-independent
plumbing: free mode legitimately tunes capacity knobs declared here (``model_size``,
``M``/``L``, ``mDim``/``numLayer_Res``, ``hidden_dim``). The architecture-preserving
campaign locks those same names one layer up, in the hash-locked
``registry.identity_locked_hparams`` block of each cell config (audited against
``EXPECTED_IDENTITY_LOCKED_HPARAMS`` in ``autobench.campaign``); the campaign's
tunable set is this module's ``tunable`` minus that lock list.

**nnMIL overrides layer ABOVE the planner — the batch_size clamp included,
and that is intended.** nnMIL's training config is *computed* at prep time
(nnU-Net-style self-configuration: ``prepare.py`` derives ``batch_size`` from
rare-class prevalence and cohort size, clamping it into the planner's band).
``hparams.apply_overrides_to_plan`` applies declared overrides AFTER that
computation, so an explicit ``batch_size`` override REPLACES the planner's
clamped value — it is not re-clamped. The clamp is a heuristic default for the
unattended path, not a validity bound; an agent that sets the knob owns the
value, exactly as on every other arm (and gets the standard fail-loud error
for undeclared/locked names, never a silent re-clamp).
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ArmSearchSpace",
    "SEARCH_SPACE",
    "TRANSPORT_COVERAGE_BEFORE",
    "coverage_table",
    "declared_knobs",
    "lock_reason",
]

#: Knobs each arm could actually reach through ``ModelConfig``/``TrainConfig``
#: before this module declared a space, as measured in the module docstring.
#: ``(reachable, arm config fields)``. Held here rather than in prose so the
#: methods figure reads the same numbers the docstring quotes.
TRANSPORT_COVERAGE_BEFORE: dict[str, tuple[int, int]] = {
    "clam": (12, 15),
    "abmil": (5, 8),
    "dtfd": (5, 15),
    "titan": (3, 4),
    "nnmil": (0, 11),
}


@dataclass(frozen=True)
class ArmSearchSpace:
    """What an agent may and may not tune on one arm."""

    arm: str
    #: Knobs an agent may set. Names are the arm's OWN field names; the canonical
    #: CLI aliases (``weight_decay`` -> DTFD's ``wd``) are resolved separately in
    #: ``hparams.FIELD_ALIASES``.
    tunable: frozenset[str]
    #: Knob -> why it is not searchable. Present so a refusal can explain itself.
    locked: dict[str, str]
    #: Where the arm's defaults come from, for the methods section.
    source: str = ""

    @property
    def n_declared(self) -> int:
        return len(self.tunable)


SEARCH_SPACE: dict[str, ArmSearchSpace] = {
    "clam": ArmSearchSpace(
        arm="clam",
        tunable=frozenset({
            # architecture / bag handling (ModelConfig)
            "model_size", "dropout", "B",
            # CLAM's instance-clustering branch (upstream core_utils.py:117,141,185)
            "bag_loss", "inst_loss",
            # optimisation / schedule (TrainConfig)
            "lr", "weight_decay", "optimizer", "max_epochs",
            "early_stopping", "patience", "stop_epoch", "weighted_sample",
        }),
        locked={
            "model_type": "grid axis, not a search knob — it selects the arm",
            "seed": "evaluation protocol, not a recipe knob (frozen substrate)",
            "no_inst_cluster": (
                "identity lock: True removes CLAM's defining instance-clustering branch"
            ),
            "bag_weight": (
                "identity lock: changing the fixed bag/instance-loss mixture can "
                "zero a defining loss branch (bag_weight=1 makes CLAM degenerate to "
                "bag-loss-only attention MIL)"
            ),
        },
        source="shared TrainConfig + ModelConfig (pipeline/config.py)",
    ),
    "abmil": ArmSearchSpace(
        arm="abmil",
        tunable=frozenset({
            "M", "L", "dropout",
            "lr", "weight_decay", "max_epochs", "early_stopping", "patience",
        }),
        locked={},
        source="ABMILConfig (abmil/config.py)",
    ),
    "dtfd": ArmSearchSpace(
        arm="dtfd",
        tunable=frozenset({
            # DTFD's own paper contributions — unreachable before H-3b
            "numGroup", "mDim", "numLayer_Res", "droprate", "droprate_2",
            "grad_clip", "lr_decay_ratio", "lr_decay_step",
            # optimisation / schedule
            "lr", "wd", "max_epochs", "early_stopping", "patience",
        }),
        locked={
            "distill": (
                "locked to 'AFS' by DTFDConfig.validate() — the other distill "
                "modes select instances by tier-1 attention, which leaks the "
                "tier-1 objective into tier-2 training"
            ),
            "total_instance": (
                "dead under AFS: attention-feature-sum consumes the whole "
                "pseudo-bag, so the field is never read"
            ),
        },
        source="DTFDConfig (dtfd/config.py), upstream Main_DTFD_MIL.py",
    ),
    "titan": ArmSearchSpace(
        arm="titan",
        tunable=frozenset({
            "lr", "weight_decay", "patience", "max_epochs", "early_stopping",
        }),
        locked={
            "head": (
                "only 'linear' is implemented — the design spec's locked "
                "frozen-embedding linear probe"
            ),
        },
        source="TitanHeadConfig (titan/config.py) + shared TrainConfig",
    ),
    "nnmil": ArmSearchSpace(
        arm="nnmil",
        tunable=frozenset({
            "hidden_dim", "max_seq_length", "batch_size", "batch_sampler",
            "learning_rate", "weight_decay", "num_epochs", "warmup_epochs",
            "dropout", "patience",
        }),
        locked={
            "feature_dimension": "derived from the encoder — part of the frozen substrate",
            "num_classes": "derived from the task definition",
            "survival_loss": "grid axis (cox / nllsurv), not a search knob",
            "nll_bins": "task definition (TaskConfig.nll_bins), not a recipe knob",
            "use_original_length": (
                "sequence-length policy is coupled to max_seq_length; exposing "
                "both invites contradictory settings"
            ),
        },
        source="computed at prep time in nnmil/prepare.py (nnU-Net-style self-configuration)",
    ),
}


def declared_knobs(arm: str) -> frozenset[str]:
    """Knobs an agent may tune on ``arm``. Empty for an unknown arm."""
    space = SEARCH_SPACE.get(arm)
    return space.tunable if space else frozenset()


def lock_reason(arm: str, knob: str) -> str | None:
    """Why ``knob`` is not searchable on ``arm``, if it is deliberately locked."""
    space = SEARCH_SPACE.get(arm)
    return space.locked.get(knob) if space else None


def coverage_table() -> str:
    """Render the declared space as markdown for the paper's methods section."""
    lines = [
        "| Arm | Searchable knobs | n | Locked (with reason) |",
        "|---|---|--:|---|",
    ]
    for arm in sorted(SEARCH_SPACE):
        space = SEARCH_SPACE[arm]
        locked = "; ".join(f"`{k}` — {v}" for k, v in sorted(space.locked.items()))
        knobs = ", ".join(f"`{k}`" for k in sorted(space.tunable))
        lines.append(f"| {arm} | {knobs} | {space.n_declared} | {locked or '—'} |")
    return "\n".join(lines)
