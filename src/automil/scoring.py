"""Composite recomputation from the agent-facing validation metrics (CR-1b).

The val-firewall's central claim is that **test never drives selection**. But the
orchestrator historically trusted the ``composite`` scalar in ``result.json``
verbatim, and that file is written by agent-editable training code. A script that
computed its composite from the sealed test block (by bug or by design) produced a
perfectly schema-valid result, the graph selected on it, and no code path could
detect the leak.

This module closes that hole by deriving the selection signal from the declared
**validation** ``metrics`` block instead of trusting the reported scalar. The
reducer is declared per-project in ``automil/config.yaml``:

    scoring:
      formula: mean          # default

``mean`` reproduces the established composites exactly — classification
``{val_auc, val_bacc}`` → their mean; survival ``{val_c_index}`` → itself — so the
default is behaviour-preserving while making test-derived composites detectable.

Set ``formula: trust_reported`` to opt out (documented as weakening the firewall).
"""
from __future__ import annotations

import math

DEFAULT_FORMULA = "mean"

#: Absolute tolerance when comparing the reported vs recomputed composite.
#: result.json rounds both ``composite`` and each metric to 4 decimals, so a
#: faithful writer can differ by ~1e-4 purely from rounding.
COMPOSITE_TOLERANCE = 1e-3

#: Formula values that disable recomputation (the pre-CR-1b trust-verbatim path).
_OPT_OUT = frozenset({"trust_reported", "none", "reported"})

_REDUCERS = {
    "mean": lambda vs: sum(vs) / len(vs),
    "max": max,
    "min": min,
}


def recompute_composite(metrics: dict, formula: str = DEFAULT_FORMULA) -> float | None:
    """Derive the composite from validation ``metrics``.

    Returns ``None`` when recomputation does not apply — the project opted out,
    the metrics block is absent/empty (crash and partial results), or it holds no
    finite numeric value. Callers keep the reported composite in that case.

    Raises:
        ValueError: unknown formula name (fail loud on a typo rather than
            silently falling back to trusting the agent-reported scalar).
    """
    if not formula or formula in _OPT_OUT:
        return None
    reducer = _REDUCERS.get(formula)
    if reducer is None:
        raise ValueError(
            f"unknown scoring.formula {formula!r}; expected one of "
            f"{sorted(_REDUCERS)} or 'trust_reported'"
        )
    if not isinstance(metrics, dict):
        return None
    values = [
        float(v)
        for v in metrics.values()
        # bool is an int subclass — exclude it explicitly.
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    ]
    if not values:
        return None
    return float(reducer(values))


def composite_disagrees(reported: float, recomputed: float,
                        tolerance: float = COMPOSITE_TOLERANCE) -> bool:
    """True when the reported composite cannot be explained by the val metrics."""
    try:
        return abs(float(reported) - float(recomputed)) > tolerance
    except (TypeError, ValueError):
        return True
