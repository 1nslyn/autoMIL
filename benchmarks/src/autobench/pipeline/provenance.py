"""Declared provenance of each arm's default hyperparameters (documentation).

This module is **reporting only** — nothing here is read at training time. Each
arm's own config dataclass remains the single runtime source of truth (see
``hparams.py`` for why: a shared value table would duplicate state, flatten the
arms onto a lowest-common-denominator field set, and could not represent nnMIL's
data-adaptive self-configuration).

What this records is the answer to a question a reviewer will ask and the code
previously could not answer: *where does each arm's schedule come from, and does
it match that method's published defaults?*

Audit finding (2026-07-23), after checking every vendored package under
``benchmarks/lib/`` field by field:

  DTFD   faithful — every value reproduced from Main_DTFD_MIL.py
  nnMIL  faithful — 3e-4/1e-4 + 100 epochs ARE nnMIL's own trainer defaults
  CLAM   deviates on ONE knob: lr=2e-4 vs upstream 1e-4 (2x), no rationale
  ABMIL  deviates on ALL THREE optimizer settings (lr, reg, epochs 20 -> 200)
  TITAN  n/a — a linear probe with no upstream training recipe

Two of these correct an earlier, less careful reading: nnMIL was initially
recorded as deviating (it does not — 100 epochs is its own design, not a
benchmark choice), and ABMIL's deviation was understated (it is the largest of
any arm, not a minor schedule difference).

The surviving concern is unchanged in kind but narrower in scope: CLAM's lr and
ABMIL's optimizer are inherited-by-accident rather than chosen, and they sit on
the aggregator axis — the axis the preprint's headline compares. Whether to
return them to upstream or keep them and disclose is a decision to make
deliberately; either is defensible, silence is not.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ArmProvenance", "ARM_PROVENANCE", "provenance_table"]


@dataclass(frozen=True)
class ArmProvenance:
    """Where one arm's default hyperparameters come from."""

    arm: str
    source: str
    #: True when the defaults reproduce that method's published/upstream values.
    matches_upstream: bool
    note: str = ""


ARM_PROVENANCE: tuple[ArmProvenance, ...] = (
    ArmProvenance(
        arm="clam",
        source="shared TrainConfig (autobench.pipeline.config)",
        matches_upstream=False,
        note=(
            "VERIFIED against lib/CLAM/main.py: lr=2e-4 DEVIATES from upstream "
            "default 1e-4 (2x) with no recorded rationale. Everything else matches "
            "upstream: reg/weight_decay 1e-5, max_epochs 200, drop_out 0.25, "
            "bag_weight 0.7, B=8. So CLAM is upstream-faithful EXCEPT the lr."
        ),
    ),
    ArmProvenance(
        arm="abmil",
        source="ABMILConfig (abmil/config.py)",
        matches_upstream=False,
        note=(
            "VERIFIED against lib/AttentionDeepMIL: architecture is paper-exact "
            "(M=500, L=128), but ALL THREE optimizer settings deviate — upstream "
            "lr=5e-4 (here 2e-4), reg=1e-4 (here weight_decay 1e-5), epochs=20 "
            "(here 200, a 10x change). Note upstream ABMIL is a toy MNIST-bags "
            "experiment, so its 20 epochs is arguably not transferable to WSI "
            "training — but that is a rationale to STATE, not to leave implicit. "
            "This is the largest deviation of any arm."
        ),
    ),
    ArmProvenance(
        arm="dtfd",
        source="DTFDConfig (dtfd/config.py)",
        matches_upstream=True,
        note=(
            "VERIFIED field-by-field against lib/DTFD-MIL/Main_DTFD_MIL.py: EPOCH=200, "
            "lr=1e-4, weight_decay=1e-4, numGroup=4, total_instance=4, mDim=512, "
            "grad_clipping=5 — all reproduced exactly. Fully upstream-faithful."
        ),
    ),
    ArmProvenance(
        arm="titan",
        source="TitanHeadConfig (titan/config.py) + shared TrainConfig",
        matches_upstream=False,
        note=(
            "MIXED provenance: lr/weight_decay/patience come from TitanHeadConfig, "
            "while max_epochs and the early_stopping switch are read off the shared "
            "TrainConfig. Linear-probe head on a frozen slide embedding."
        ),
    ),
    ArmProvenance(
        arm="nnmil",
        source="computed at prep time in nnmil/prepare.py, written to the plan JSON",
        matches_upstream=True,
        note=(
            "CORRECTED after verifying lib/nnMIL: these literals REPRODUCE nnMIL's "
            "own upstream defaults — classification_trainer.py uses learning_rate "
            "3e-4 + num_epochs 100; survival_porpoise_trainer.py uses 1e-4 + 100. "
            "So the 100-epoch schedule and the per-task lr are nnMIL's design, NOT "
            "a benchmark deviation. Self-configuring (nnU-Net style): batch size, "
            "warmup_epochs and a conditional weight_decay are derived from data "
            "statistics, extending upstream's fixed warmup default of 5."
        ),
    ),
)


def provenance_table() -> str:
    """Render as markdown for the paper's methods section."""
    lines = [
        "| Arm | Defaults come from | Matches upstream? | Notes |",
        "|---|---|:-:|---|",
    ]
    for p in ARM_PROVENANCE:
        mark = "yes" if p.matches_upstream else "**no**"
        lines.append(f"| {p.arm} | {p.source} | {mark} | {p.note} |")
    return "\n".join(lines)
