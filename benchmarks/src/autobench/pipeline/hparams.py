"""Single source of truth for per-arm training hyperparameters (H-3).

Before this module the benchmark had **five different hyperparameter
provenances**, none of them declared:

    CLAM   -> the shared TrainConfig (lr 2e-4)
    ABMIL  -> ABMILConfig  (own dataclass, mirrors the shared schedule)
    DTFD   -> DTFDConfig   (own dataclass, paper-exact, cites Main:29/30)
    TITAN  -> TitanHeadConfig for lr/wd/patience, but max_epochs and
              early_stopping from the shared TrainConfig  (mixed!)
    nnMIL  -> literals hardcoded inside prepare.py, and *different* per task
              type (classification lr 3e-4, survival lr 1e-4, epochs 100)

Two consequences, both load-bearing for the paper:

1. **Unanswerable provenance.** A reviewer asking "why does CLAM run at 2x its
   published default learning rate?" had nowhere to look. Some arms are
   paper-exact, some are not, and nothing recorded which was which.
2. **The agentic search could not tune two of the four aggregators.**
   ``run_experiment.py`` advertises ``--lr/--max_epochs/--patience``, but ABMIL
   and DTFD build their own config and read only ``seed`` from ``exp_cfg`` — so
   those flags were silently discarded. An equal-effort recipe search would have
   optimized CLAM/nnMIL while ABMIL/DTFD sat at baseline under the variant's
   label, an interface artifact on the very axis the paper compares.

This module fixes the *mechanism*: every arm declares its hyperparameters in one
table with an explicit ``provenance`` string, and every arm resolves overrides
through one function.

**The values below are the EFFECTIVE values the benchmark already ran with**, so
adopting this module changes no result. Where an arm deviates from its upstream
repository, the deviation is now recorded in ``provenance`` rather than being
invisible — making it a decision that can be reviewed and flipped deliberately,
instead of an accident.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

__all__ = ["ArmHParams", "ARM_HPARAMS", "resolve_arm_hparams", "provenance_table"]


@dataclass(frozen=True)
class ArmHParams:
    """Training hyperparameters for one benchmark arm, with declared provenance."""

    lr: float
    weight_decay: float
    max_epochs: int
    patience: int
    early_stopping: bool = True
    #: Where these values come from, and whether they match the upstream repo.
    provenance: str = ""
    #: True when the values reproduce the arm's own published/upstream defaults.
    matches_upstream: bool = False


# Keyed by (framework, task_type). task_type is "classification" or "survival"
# because nnMIL genuinely differs across the two.
ARM_HPARAMS: dict[tuple[str, str], ArmHParams] = {
    ("clam", "classification"): ArmHParams(
        lr=2e-4, weight_decay=1e-5, max_epochs=200, patience=20,
        provenance=(
            "benchmark schedule (shared TrainConfig). DEVIATES from upstream "
            "CLAM main.py:74 default lr=1e-4 (this is 2x). weight_decay matches "
            "upstream --reg 1e-5 (main.py:78); max_epochs matches (main.py:72)."
        ),
        matches_upstream=False,
    ),
    ("clam", "survival"): ArmHParams(
        lr=2e-4, weight_decay=1e-5, max_epochs=200, patience=20,
        provenance="same as CLAM classification; upstream CLAM has no survival arm.",
        matches_upstream=False,
    ),
    ("abmil", "classification"): ArmHParams(
        lr=2e-4, weight_decay=1e-5, max_epochs=200, patience=20,
        provenance=(
            "ABMILConfig — architecture is paper-exact (Ilse et al. 2018: M=500, "
            "L=128), but the optimizer/schedule deliberately mirrors the shared "
            "benchmark schedule rather than the paper's own optimizer settings."
        ),
        matches_upstream=False,
    ),
    ("abmil", "survival"): ArmHParams(
        lr=2e-4, weight_decay=1e-5, max_epochs=200, patience=20,
        provenance="same as ABMIL classification.",
        matches_upstream=False,
    ),
    ("dtfd", "classification"): ArmHParams(
        lr=1e-4, weight_decay=1e-4, max_epochs=200, patience=20,
        provenance=(
            "DTFDConfig — paper-exact from the DTFD-MIL reference implementation "
            "(lr Main:29, weight_decay Main:30, EPOCH Main:21)."
        ),
        matches_upstream=True,
    ),
    ("dtfd", "survival"): ArmHParams(
        lr=1e-4, weight_decay=1e-4, max_epochs=200, patience=20,
        provenance="same as DTFD classification (upstream DTFD has no survival arm).",
        matches_upstream=True,
    ),
    ("titan", "classification"): ArmHParams(
        lr=1e-3, weight_decay=1e-4, max_epochs=200, patience=10,
        provenance=(
            "TitanHeadConfig supplies lr/weight_decay/patience; max_epochs and the "
            "early_stopping switch previously came from the shared TrainConfig — a "
            "MIXED provenance, now recorded in one place. Linear-probe head only."
        ),
        matches_upstream=False,
    ),
    ("titan", "survival"): ArmHParams(
        lr=1e-3, weight_decay=1e-4, max_epochs=200, patience=10,
        provenance="same as TITAN classification.",
        matches_upstream=False,
    ),
    ("nnmil", "classification"): ArmHParams(
        lr=3e-4, weight_decay=1e-4, max_epochs=100, patience=10,
        provenance=(
            "hardcoded in nnmil/prepare.py (plan literals). NOTE: differs from every "
            "other arm on epochs (100 vs 200) and carries a conditional weight_decay "
            "(0.01 when hidden_dim>=512, else 1e-4) that this table cannot express — "
            "the plan file remains authoritative for that conditional."
        ),
        matches_upstream=False,
    ),
    ("nnmil", "survival"): ArmHParams(
        lr=1e-4, weight_decay=1e-4, max_epochs=100, patience=10,
        provenance=(
            "hardcoded in nnmil/prepare.py — DIFFERENT lr from nnMIL classification "
            "(1e-4 vs 3e-4). Same conditional weight_decay caveat as above."
        ),
        matches_upstream=False,
    ),
}


def resolve_arm_hparams(
    framework: str,
    task_type: str = "classification",
    overrides: dict | None = None,
) -> ArmHParams:
    """Return an arm's hyperparameters with explicit overrides applied.

    This is the ONE path through which a CLI flag or an agentic-search variant
    reaches any arm, so a tuning knob can no longer be honored by CLAM/TITAN and
    silently discarded by ABMIL/DTFD.

    Args:
        framework: "clam" | "abmil" | "dtfd" | "titan" | "nnmil".
        task_type: "classification" or "survival".
        overrides: e.g. ``{"lr": 5e-4}``. **None values are ignored** — the CLI
            parses unset flags as None, so only explicitly-set values override
            the arm's declared default. This is what keeps each arm on its own
            schedule unless the operator (or the agent) deliberately changes it.

    Raises:
        KeyError: unknown (framework, task_type) — fail loud rather than
            silently substituting some other arm's schedule.
    """
    key = (framework.lower(), task_type.lower())
    if key not in ARM_HPARAMS:
        raise KeyError(
            f"no declared hyperparameters for {key}; add an entry to "
            f"autobench.pipeline.hparams.ARM_HPARAMS (known: {sorted(ARM_HPARAMS)})"
        )
    base = ARM_HPARAMS[key]
    if not overrides:
        return base
    explicit = {
        k: v for k, v in overrides.items()
        if v is not None and k in {"lr", "weight_decay", "max_epochs",
                                   "patience", "early_stopping"}
    }
    return replace(base, **explicit) if explicit else base


def provenance_table() -> str:
    """Render the declared provenance as markdown (for the paper's methods table)."""
    lines = [
        "| Arm | Task | lr | weight decay | epochs | patience | upstream? | provenance |",
        "|---|---|--:|--:|--:|--:|:-:|---|",
    ]
    for (fw, tt), h in sorted(ARM_HPARAMS.items()):
        mark = "yes" if h.matches_upstream else "**no**"
        lines.append(
            f"| {fw} | {tt} | {h.lr:g} | {h.weight_decay:g} | {h.max_epochs} | "
            f"{h.patience} | {mark} | {h.provenance} |"
        )
    return "\n".join(lines)
