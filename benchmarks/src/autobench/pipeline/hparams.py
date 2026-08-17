"""Uniform hyperparameter-override application across benchmark arms (H-3, H-3b).

**The original problem (H-3).** ``run_experiment.py`` advertised tuning flags and
the agentic search injected ``CLAM_ARGS`` through ``variant_dispatch``, but the
five arms consumed hyperparameters through five different mechanisms:

    CLAM   -> reads exp_cfg.train directly                    (override works)
    ABMIL  -> builds ABMILConfig() in its runner              (override discarded)
    DTFD   -> builds DTFDConfig() in its runner               (override discarded)
    TITAN  -> TitanHeadConfig() + exp_cfg.train.max_epochs    (partial)
    nnMIL  -> literals computed into a plan JSON at prep time (override discarded)

Worse, the safety net checked the wrong object: ``variant_dispatch`` warned only
when a field was missing from ``ModelConfig``/``TrainConfig`` — but ``lr`` *is*
present there, so it was set successfully, logged nothing, and then never reached
ABMIL/DTFD/nnMIL.

**The second problem (H-3b).** Fixing the plumbing was not enough, because the
transport itself is CLAM-shaped: ``ModelConfig`` + ``TrainConfig`` carry CLAM's
whole surface and nobody else's. Measured coverage of each arm's own knobs was
CLAM 12/15, ABMIL 5/8, TITAN 3/4, DTFD 5/15, **nnMIL 0/11**. An equal-effort
search under that asymmetry reports channel width as a model result — on exactly
the axis the paper compares.

**The design.** Each arm's own config dataclass stays the single source of truth
for defaults; this module does not hold values and does not flatten the arms onto
a shared field set. Three pieces:

1. *Canonical channel* — knobs that exist on the shared transport, detected by
   diffing the live ``exp_cfg`` against a pristine one (the CLI parses unset flags
   as ``None`` and then fills defaults, which erases "was this set?").
2. *Opaque channel* — ``ExperimentConfig.hparam_overrides``, an arbitrary
   ``{name: value}`` dict fed by ``--hparams`` or a variant's ``HPARAMS``. This is
   what lets DTFD receive ``numGroup`` and nnMIL receive ``warmup_epochs``.
3. *Declared space* — ``search_space.py`` says, per arm, which knobs are
   searchable and which are locked and why. An undeclared or locked knob raises
   here instead of vanishing.

nnMIL is handled by the same function on its plan dict: its config is *computed*
from data statistics (nnU-Net-style self-configuration, e.g.
``warmup_epochs = 10 if n_train < 500 else 5``), so overrides layer on top of that
computation rather than replacing it.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from typing import Any, Mapping, TypeVar

from autobench.pipeline.search_space import SEARCH_SPACE, declared_knobs, lock_reason

__all__ = [
    "FIELD_ALIASES",
    "all_overrides",
    "apply_overrides",
    "apply_overrides_to_exp_cfg",
    "apply_overrides_to_plan",
    "explicit_overrides",
    "overrides_from_exp_cfg",
]

#: Never reported as an override, whatever their value: these are the evaluation
#: protocol and the grid axis, not recipe knobs. ``seed`` in particular is frozen
#: substrate — an agent that could set it could select a favourable partition.
_NEVER_OVERRIDABLE = frozenset({"seed", "model_type"})
_ConfigT = TypeVar("_ConfigT")


def _pristine(cfg: Any) -> Any | None:
    """A default-constructed twin of ``cfg``, for the was-this-set? diff."""
    from autobench.pipeline.config import ModelConfig, TrainConfig

    if isinstance(cfg, ModelConfig):
        # model_type has no default and is the grid axis, so carry it across:
        # it must never show up as a difference.
        return ModelConfig(model_type=cfg.model_type)
    if isinstance(cfg, TrainConfig):
        return TrainConfig()
    return None


def overrides_from_exp_cfg(exp_cfg: Any) -> dict[str, Any]:
    """Detect which transport knobs were *explicitly* set on this experiment.

    Recovered by diffing the live ``exp_cfg.model`` / ``exp_cfg.train`` against
    pristine ones: a value that differs was set, either by a CLI flag or by
    ``variant_dispatch`` (which mutates the same objects for agentic variants).

    H-3b widened this from ``exp_cfg.train`` alone to both dataclasses, so CLAM's
    ``dropout`` / ``bag_weight`` / ``B`` / ``bag_loss`` are detected too rather
    than being silently confined to the arm that happens to read them directly.

    A value explicitly set to exactly the default is reported as not-overridden;
    that is a harmless false negative *where the arm's own default equals the
    shared default*. Where it does not — DTFD's ``lr``, TITAN's ``patience`` — use
    the opaque channel, which has no such blind spot.
    """
    out: dict[str, Any] = {}
    for attr in ("model", "train"):
        live = getattr(exp_cfg, attr, None)
        if live is None:
            continue
        base = _pristine(live)
        if base is None:
            continue
        for f in fields(live):
            if f.name in _NEVER_OVERRIDABLE:
                continue
            current = getattr(live, f.name)
            if current is not None and current != getattr(base, f.name, None):
                out[f.name] = current
    return out


def all_overrides(exp_cfg: Any) -> dict[str, Any]:
    """Canonical transport diff, then the opaque per-arm channel on top.

    The opaque channel wins: it is the explicit, arm-aware request, whereas the
    canonical diff is an inference about what the CLI meant.
    """
    merged = overrides_from_exp_cfg(exp_cfg)
    opaque = getattr(exp_cfg, "hparam_overrides", None) or {}
    merged.update(opaque)
    return merged


#: Some arms name the same knob differently because each config follows its own
#: upstream repository (DTFD uses ``wd``; nnMIL's plan uses ``learning_rate`` /
#: ``num_epochs``). Map the canonical CLI/variant name -> that arm's field name.
#:
#: **Every entry must be an unambiguous 1:1 rename of the SAME knob.** Anything
#: that is a guess is deliberately absent:
#:   - no ``dropout -> droprate``: DTFD has ``droprate`` (tier-1 classifier) AND
#:     ``droprate_2`` (tier-2). They are different knobs, so a generic "dropout"
#:     would silently tune only half the model. Address them by their real names
#:     (``apply_overrides`` accepts an arm's own field names directly).
#:   - no ``EPOCH``: no config actually declares it (DTFD's field is
#:     ``max_epochs``; ``Main:21`` in its docstring is the upstream argparse flag,
#:     not the field), so the alias was dead and misleading.
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "weight_decay": ("weight_decay", "wd"),
    "max_epochs": ("max_epochs", "num_epochs"),
    "lr": ("lr", "learning_rate"),
    "patience": ("patience",),
    "early_stopping": ("early_stopping",),
}


def explicit_overrides(**kwargs: Any) -> dict[str, Any]:
    """Drop ``None`` values — the CLI parses unset flags as None.

    This is what keeps every arm on its own schedule unless the operator (or the
    agent) deliberately changed something. Without it, threading the shared
    defaults through would silently retune DTFD off its paper-exact values.
    """
    return {k: v for k, v in kwargs.items() if v is not None}


def _resolve_field(cfg_or_plan: Any, name: str) -> str | None:
    """Return the actual attribute/key on this arm matching the canonical name.

    Raises:
        ValueError: the canonical name matches MORE THAN ONE field on this arm
            (e.g. a config declaring both ``weight_decay`` and ``wd``). Picking
            the first would silently tune one and leave the other stale — the
            exact class of failure this module exists to remove.
    """
    candidates = FIELD_ALIASES.get(name, (name,))
    if is_dataclass(cfg_or_plan):
        available = {f.name for f in fields(cfg_or_plan)}
    else:
        available = set(cfg_or_plan)
    matches = [c for c in candidates if c in available]
    if len(matches) > 1:
        raise ValueError(
            f"ambiguous override {name!r}: it matches multiple fields {matches} on "
            f"this arm. Refusing to guess — address the field by its own name."
        )
    if matches:
        return matches[0]
    # Allow an arm's own field name to be used directly (e.g. DTFD's droprate_2).
    return name if name in available else None


def _check_declared(arm: str, requested: str, resolved: str) -> None:
    """Refuse a knob the arm's declared search space does not contain (A-2).

    Enforcement is skipped for arms with no declared space (``arm=""`` in tests,
    ), so
    this cannot silently narrow anything that was never declared in the first
    place.
    """
    if arm not in SEARCH_SPACE:
        return
    if resolved in declared_knobs(arm):
        return
    why = lock_reason(arm, resolved)
    if why:
        raise ValueError(
            f"{arm}: {resolved!r} is deliberately LOCKED and not searchable — {why}. "
            f"(Declared in search_space.py; change the declaration, with a "
            f"rationale, if this lock is wrong.)"
        )
    raise ValueError(
        f"{arm}: {requested!r} (resolves to {resolved!r}) is not in this arm's "
        f"declared search space. Declared: {sorted(declared_knobs(arm))}. "
        f"(Add it to search_space.py if it should be searchable — an undeclared "
        f"knob is indistinguishable from an oversight.)"
    )


def apply_overrides(
    cfg: _ConfigT,
    overrides: Mapping[str, Any] | None,
    *,
    arm: str = "",
) -> _ConfigT:
    """Return a copy of an arm's config dataclass with explicit overrides applied.

    Args:
        cfg: the arm's own config instance (ABMILConfig, DTFDConfig, ...). It stays
            the single source of truth for defaults; this only layers on top.
        overrides: names -> values, either canonical (``weight_decay``) or the
            arm's own (``numGroup``). ``None`` values are ignored.
        arm: arm name — selects the declared search space, and names the arm in
            error messages.

    Raises:
        ValueError: an override cannot be applied to this arm, or is not in its
            declared search space. Failing loudly is the point — silently
            discarding a tuning knob is the H-3 defect, and it would make an
            agentic search report an untuned arm under a variant's label.
    """
    if not overrides:
        return cfg
    concrete: dict[str, Any] = {}
    unknown: list[str] = []
    for name, value in overrides.items():
        if value is None:
            continue
        field = _resolve_field(cfg, name)
        if field is None:
            unknown.append(name)
            continue
        _check_declared(arm, name, field)
        concrete[field] = value
    if unknown:
        available = sorted(f.name for f in fields(cfg)) if is_dataclass(cfg) else []
        raise ValueError(
            f"{arm or type(cfg).__name__} cannot accept hyperparameter override(s) "
            f"{sorted(unknown)} — it has no such knob. Available: {available}. "
            f"(Refusing to discard the override silently; that is the H-3 defect.)"
        )
    return replace(cfg, **concrete) if concrete else cfg


def apply_overrides_to_exp_cfg(exp_cfg: Any, *, arm: str = "clam") -> None:
    """Same contract for the arm that trains off the shared transport (CLAM).

    CLAM's knobs live directly on ``exp_cfg.model`` / ``exp_cfg.train`` — its
    canonical channel is natively live, so unlike the sibling arms there is no
    separate config dataclass to hand to ``apply_overrides``, and the opaque
    ``--hparams`` channel was parsed but never consumed (the H-3 defect on the
    reference arm; claims-alignment A1). Only the opaque channel is applied:
    re-routing the canonical diff would be a second path for values that are
    already in effect.

    Partitions the opaque keys across the two transport dataclasses and applies
    each slice through ``apply_overrides`` with full declared-space enforcement;
    a key unknown to both raises through the standard unknown-knob error rather
    than vanishing. Mutates ``exp_cfg`` in place. Must run before results-dir
    resolution and ``exp_cfg.save`` so CR-5b cache identity and the archived
    provenance record the effective values.
    """
    overrides = getattr(exp_cfg, "hparam_overrides", None)
    if not overrides:
        return
    remaining = {k: v for k, v in overrides.items() if v is not None}
    for attr in ("model", "train"):
        section = getattr(exp_cfg, attr, None)
        if section is None:
            continue
        slice_ = {
            name: value for name, value in remaining.items()
            if _resolve_field(section, name) is not None
        }
        if slice_:
            setattr(exp_cfg, attr, apply_overrides(section, slice_, arm=arm))
            for name in slice_:
                del remaining[name]
    if remaining:
        # Neither dataclass knows these keys — surface the standard fail-loud
        # unknown-knob error (with TrainConfig's field list) instead of a
        # bespoke message.
        apply_overrides(exp_cfg.train, remaining, arm=arm)


def apply_overrides_to_plan(
    plan_cfg: Mapping[str, Any],
    overrides: Mapping[str, Any] | None,
    *,
    arm: str = "nnmil",
) -> dict[str, Any]:
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
            continue
        _check_declared(arm, name, key)
        out[key] = value
    if unknown:
        raise ValueError(
            f"{arm} training plan cannot accept override(s) {sorted(unknown)}; "
            f"available keys: {sorted(plan_cfg)}"
        )
    return out
