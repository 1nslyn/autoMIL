"""Declared provenance of each arm's default hyperparameters (documentation).

This module is **reporting only** — nothing here is read at training time. Each
arm's own config dataclass remains the single runtime source of truth (see
``hparams.py`` for why: a shared value table would duplicate state, flatten the
arms onto a lowest-common-denominator field set, and could not represent nnMIL's
data-adaptive self-configuration).

This records two distinct provenance questions: *where do the arm's scalar
defaults come from, and what is the relation between the live trainer and
upstream?* Matching
learning-rate/regularization/epoch literals does **not** make a benchmark-owned
training loop upstream-faithful.

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

Resolution (2026-07-28): decided and executed — return to upstream for both.
CLAM's lr moves 2e-4 -> 1e-4 (lib/CLAM/main.py:74); ABMIL's lr/weight_decay/
max_epochs move 2e-4/1e-5/200 -> 5e-4/1e-4/20 (lib/AttentionDeepMIL/main.py:
16-21). Both now report ``matches_upstream=True`` below. This does not retire
the disclosure obligation: ABMIL's 20-epoch schedule is still upstream's toy
MNIST-bags value, arguably not transferable to WSI training, and it now makes
``patience=20`` equal to ``max_epochs``, which effectively disables early
stopping — see ``abmil/config.py``'s docstring, which states both explicitly.
Any dispatched CLAM/ABMIL run trained under the old schedule is stale: it must
be purged and re-run, and CR-5b's results-cache fingerprint guard will refuse
to silently resume those folds (it raises ``StaleResultsCacheError`` naming
the changed field and prints the ``rm -rf`` purge command) rather than
reporting the old numbers back as if nothing changed.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ArmProvenance", "ARM_PROVENANCE", "provenance_table"]


@dataclass(frozen=True)
class ArmProvenance:
    """Where one arm's default hyperparameters come from."""

    arm: str
    source: str
    #: True when the declared native scalar defaults reproduce the corresponding
    #: published/upstream values. This is deliberately narrower than training-
    #: loop fidelity; see ``trainer_provenance``.
    matches_upstream: bool
    #: Relation between the live training loop and upstream, including benchmark
    #: additions such as checkpoint selection or early stopping.
    trainer_provenance: str
    note: str = ""


ARM_PROVENANCE: tuple[ArmProvenance, ...] = (
    ArmProvenance(
        arm="clam",
        source="shared TrainConfig (autobench.pipeline.config)",
        matches_upstream=True,
        trainer_provenance=(
            "Classification calls the vendored CLAM trainer, but the benchmark "
            "enables CLAM's optional early stopping by default whereas upstream's "
            "CLI default is off. Survival uses a benchmark adapter loop with "
            "validation-loss checkpointing and optional early stopping."
        ),
        note=(
            "RESOLVED 2026-07-28 (was lr=2e-4 DEVIATING from upstream default "
            "1e-4, lib/CLAM/main.py:74, 2x with no recorded rationale): lr "
            "changed to 1e-4. Everything else already matched upstream: "
            "reg/weight_decay 1e-5, max_epochs 200, drop_out 0.25, bag_weight "
            "0.7, B=8. The scalar defaults now match; trainer differences are "
            "reported separately."
        ),
    ),
    ArmProvenance(
        arm="abmil",
        source="ABMILConfig (abmil/config.py)",
        matches_upstream=True,
        trainer_provenance=(
            "Benchmark reimplementation of the published model and recipe. It "
            "adds validation checkpoint selection and an early stopping switch "
            "that upstream ABMIL does not provide; patience=20 is inert under the "
            "native 20-epoch default."
        ),
        note=(
            "RESOLVED 2026-07-28 (was deviating on ALL THREE optimizer settings "
            "— lr=2e-4 vs upstream 5e-4, weight_decay=1e-5 vs upstream 1e-4, "
            "max_epochs=200 vs upstream 20, a 10x change, lib/AttentionDeepMIL/"
            "main.py:16-21): all three changed to upstream's own values. "
            "Architecture stays paper-exact (M=500, L=128) — unaffected either "
            "way. Caveat carried forward, not resolved: upstream's 20 epochs is "
            "a toy MNIST-bags value, arguably not transferable to WSI training "
            "— reproduced faithfully and disclosed (abmil/config.py docstring) "
            "rather than silently overridden. That also makes patience=20 equal "
            "max_epochs, effectively disabling early stopping; patience itself "
            "is unchanged, not a newly chosen value."
        ),
    ),
    ArmProvenance(
        arm="dtfd",
        source="DTFDConfig (dtfd/config.py)",
        matches_upstream=True,
        trainer_provenance=(
            "Benchmark reimplementation of DTFD's two-tier trainer. It adds "
            "validation checkpoint selection and early stopping, neither of "
            "which appears in the vendored upstream entry point."
        ),
        note=(
            "VERIFIED field-by-field against lib/DTFD-MIL/Main_DTFD_MIL.py: EPOCH=200, "
            "lr=1e-4, weight_decay=1e-4, numGroup=4, total_instance=4, mDim=512, "
            "grad_clipping=5 — all scalar defaults reproduced exactly. Trainer "
            "differences are reported separately."
        ),
    ),
    ArmProvenance(
        arm="titan",
        source="TitanHeadConfig (titan/config.py) + shared TrainConfig",
        matches_upstream=False,
        trainer_provenance=(
            "Benchmark-owned frozen-embedding linear-probe trainer with validation "
            "checkpointing and early stopping; there is no upstream TITAN head "
            "training loop to reproduce."
        ),
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
        trainer_provenance=(
            "Uses nnMIL's vendored planner and task-specific trainers; the "
            "benchmark adapter prepares inputs and normalizes outputs without "
            "adding a separate early stopping mechanism."
        ),
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
        "| Arm | Defaults come from | Native scalar defaults? | Trainer provenance | Notes |",
        "|---|---|:-:|---|---|",
    ]
    for p in ARM_PROVENANCE:
        mark = "yes" if p.matches_upstream else "**no**"
        lines.append(
            f"| {p.arm} | {p.source} | {mark} | {p.trainer_provenance} | {p.note} |"
        )
    return "\n".join(lines)
