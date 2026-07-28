"""Uniform hyperparameter-override application across benchmark arms (H-3).

**The problem.** ``run_experiment.py`` advertises ``--lr/--max_epochs/--patience``
and the agentic search injects ``CLAM_ARGS`` through ``variant_dispatch``, but the
five arms consumed hyperparameters through five different mechanisms:

    CLAM   -> reads exp_cfg.train directly                    (override works)
    ABMIL  -> builds ABMILConfig() in its runner              (override discarded)
    DTFD   -> builds DTFDConfig() in its runner               (override discarded)
    TITAN  -> TitanHeadConfig() + exp_cfg.train.max_epochs    (partial)
    nnMIL  -> literals computed into a plan JSON at prep time (override discarded)

Worse, the safety net checked the wrong object: ``variant_dispatch`` warns only
when a field is missing from ``ModelConfig``/``TrainConfig`` — but ``lr`` *is*
present there, so it was set successfully, logged nothing, and then never reached
ABMIL/DTFD/nnMIL. Of the four aggregators in the roster, only CLAM could actually
be tuned, which makes an "equal-effort" recipe search structurally impossible.

**The design.** Each arm's own config dataclass stays the single source of truth —
it is the right home, because it carries that arm's specific knobs (DTFD's
``numGroup``/``grad_clip``/``lr_decay_ratio``, ABMIL's ``M``/``L``, CLAM's
``bag_weight``/``B``). This module does not hold values and does not flatten the
arms onto a shared field set; it only unifies *how an explicit override is
applied*, and makes an inapplicable override fail loudly instead of vanishing.

nnMIL is deliberately handled by the same function on its plan dict: its config is
*computed* from data statistics (nnU-Net-style self-configuration, e.g.
``warmup_epochs = 10 if n_train < 500 else 5``), so overrides are layered on top of
that computation rather than replacing it.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass, replace

__all__ = [
    "FIELD_ALIASES",
    "apply_overrides",
    "apply_overrides_to_plan",
    "explicit_overrides",
    "overrides_from_exp_cfg",
]

#: Canonical knobs an operator/agent may override on any arm.
OVERRIDABLE = ("lr", "weight_decay", "max_epochs", "patience", "early_stopping")


def overrides_from_exp_cfg(exp_cfg) -> dict:
    """Detect which training knobs were *explicitly* set on this experiment.

    The CLI parses unset flags as ``None`` and then fills ``TrainConfig``
    defaults, which erases "was this set?". Rather than re-plumbing every call
    site, recover it by diffing the live ``exp_cfg.train`` against a pristine
    ``TrainConfig()``: a value that differs was set — either by a CLI flag or by
    ``variant_dispatch`` (which mutates the same object for agentic variants).

    A value explicitly set to exactly the default is reported as not-overridden;
    that is a harmless false negative (the arm keeps its own default, and the
    requested value equalled the shared default anyway).
    """
    from autobench.pipeline.config import TrainConfig

    defaults = TrainConfig()
    train = getattr(exp_cfg, "train", None)
    if train is None:
        return {}
    out: dict = {}
    for name in OVERRIDABLE:
        current = getattr(train, name, None)
        if current is not None and current != getattr(defaults, name, None):
            out[name] = current
    return out

#: Some arms name the same knob differently (DTFD follows its upstream repo's
#: ``wd``). Map the canonical CLI/variant name -> the arm's own field name.
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "weight_decay": ("weight_decay", "wd"),
    "max_epochs": ("max_epochs", "num_epochs", "EPOCH"),
    "lr": ("lr", "learning_rate"),
    "patience": ("patience",),
    "early_stopping": ("early_stopping",),
    "dropout": ("dropout", "droprate"),
}


def explicit_overrides(**kwargs) -> dict:
    """Drop ``None`` values — the CLI parses unset flags as None.

    This is what keeps every arm on its own schedule unless the operator (or the
    agent) deliberately changed something. Without it, threading the shared
    defaults through would silently retune DTFD off its paper-exact values.
    """
    return {k: v for k, v in kwargs.items() if v is not None}


def _resolve_field(cfg_or_plan, name: str) -> str | None:
    """Return the actual attribute/key on this arm matching the canonical name."""
    candidates = FIELD_ALIASES.get(name, (name,))
    if is_dataclass(cfg_or_plan):
        available = {f.name for f in fields(cfg_or_plan)}
    else:
        available = set(cfg_or_plan)
    for cand in candidates:
        if cand in available:
            return cand
    # Allow the canonical name itself even if not aliased.
    return name if name in available else None


def apply_overrides(cfg, overrides: dict | None, *, arm: str = ""):
    """Return a copy of an arm's config dataclass with explicit overrides applied.

    Args:
        cfg: the arm's own config instance (ABMILConfig, DTFDConfig, ...). It stays
            the single source of truth for defaults; this only layers on top.
        overrides: canonical names -> values. ``None`` values must already be
            stripped (see :func:`explicit_overrides`); any that slip through are
            ignored.
        arm: arm name, for the error message only.

    Raises:
        ValueError: an override cannot be applied to this arm. Failing loudly is
            the point — silently discarding a tuning knob is the H-3 defect, and
            it would make an agentic search report an untuned arm under a
            variant's label.
    """
    if not overrides:
        return cfg
    concrete: dict = {}
    unknown: list[str] = []
    for name, value in overrides.items():
        if value is None:
            continue
        field = _resolve_field(cfg, name)
        if field is None:
            unknown.append(name)
        else:
            concrete[field] = value
    if unknown:
        available = sorted(f.name for f in fields(cfg)) if is_dataclass(cfg) else []
        raise ValueError(
            f"{arm or type(cfg).__name__} cannot accept hyperparameter override(s) "
            f"{sorted(unknown)} — it has no such knob. Available: {available}. "
            f"(Refusing to discard the override silently; that is the H-3 defect.)"
        )
    return replace(cfg, **concrete) if concrete else cfg


def apply_overrides_to_plan(plan_cfg: dict, overrides: dict | None,
                            *, arm: str = "nnmil") -> dict:
    """Same contract for nnMIL, whose training config is a computed dict.

    Applied AFTER the adaptive computation so nnMIL keeps its self-configuring
    behaviour (batch size, warmup, conditional weight decay) and the override only
    layers on top. Returns a new dict — never mutates the caller's.
    """
    if not overrides:
        return plan_cfg
    out = dict(plan_cfg)
    unknown: list[str] = []
    for name, value in overrides.items():
        if value is None:
            continue
        key = _resolve_field(plan_cfg, name)
        if key is None:
            unknown.append(name)
        else:
            out[key] = value
    if unknown:
        raise ValueError(
            f"{arm} training plan cannot accept override(s) {sorted(unknown)}; "
            f"available keys: {sorted(plan_cfg)}"
        )
    return out
