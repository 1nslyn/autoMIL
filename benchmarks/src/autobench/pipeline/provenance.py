"""Declared provenance of each arm's default hyperparameters (documentation).

This module is **reporting only** — nothing here is read at training time. Each
arm's own config dataclass remains the single runtime source of truth (see
``hparams.py`` for why: a shared value table would duplicate state, flatten the
arms onto a lowest-common-denominator field set, and could not represent nnMIL's
data-adaptive self-configuration).

What this records is the answer to a question a reviewer will ask and the code
previously could not answer: *where does each arm's schedule come from, and does
it match that method's published defaults?*

Audit finding (2026-07-23): only DTFD is paper-exact. In particular **CLAM runs at
2x its own upstream default learning rate** (2e-4 vs ``CLAM/main.py:74`` = 1e-4)
with no recorded rationale, and nnMIL trains for 100 epochs where every other arm
uses 200. That is a real confound on the aggregator axis — the axis the preprint's
headline compares — and it is a decision to be made deliberately, not a default to
be inherited by accident.

Keep the strings below in sync with the arm configs when a value is changed on
purpose; ``tests/test_arm_provenance.py`` pins them against the live configs so a
silent drift fails the suite.
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
            "lr=2e-4 DEVIATES from upstream CLAM main.py:74 default 1e-4 (2x), with "
            "no recorded rationale. weight_decay=1e-5 matches upstream --reg "
            "(main.py:78); max_epochs=200 matches (main.py:72)."
        ),
    ),
    ArmProvenance(
        arm="abmil",
        source="ABMILConfig (abmil/config.py)",
        matches_upstream=False,
        note=(
            "Architecture is paper-exact (Ilse et al. 2018: M=500, L=128), but the "
            "optimizer/schedule deliberately mirrors the shared benchmark schedule "
            "rather than the paper's own optimizer settings."
        ),
    ),
    ArmProvenance(
        arm="dtfd",
        source="DTFDConfig (dtfd/config.py)",
        matches_upstream=True,
        note=(
            "Paper-exact from the DTFD-MIL reference implementation: lr Main:29, "
            "weight_decay Main:30, EPOCH Main:21. The only fully upstream arm."
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
        matches_upstream=False,
        note=(
            "Self-configuring (nnU-Net style): batch size, warmup_epochs "
            "(10 if n_train<500 else 5) and weight_decay (0.01 if hidden_dim>=512 "
            "else 1e-4) are derived from data statistics. Uses 100 epochs where "
            "every other arm uses 200, and a different lr per task type "
            "(3e-4 classification / 1e-4 survival)."
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
