"""Composite recomputation from the agent-facing validation metrics (CR-1b).

The val-firewall's central claim is that **test never drives selection**. But the
orchestrator cannot trust the ``composite`` scalar in ``result.json`` verbatim,
because that file is written by agent-editable training code. A script that
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
from collections.abc import Iterable, Mapping

DEFAULT_FORMULA = "mean"

#: Absolute tolerance when comparing the reported vs recomputed composite.
#: result.json rounds both ``composite`` and each metric to 4 decimals, so a
#: faithful writer can differ by ~1e-4 purely from rounding.
COMPOSITE_TOLERANCE = 1e-3

#: Formula values that explicitly disable validation-metric recomputation.
_OPT_OUT = frozenset({"trust_reported", "none", "reported"})

_REDUCERS = {
    "mean": lambda vs: sum(vs) / len(vs),
    "max": max,
    "min": min,
}


def recompute_composite(
    metrics: Mapping[str, object] | None,
    formula: str = DEFAULT_FORMULA,
) -> float | None:
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
    if not isinstance(metrics, Mapping):
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


def cross_fold_se(fold_values: Iterable[object] | None) -> float | None:
    """Standard error of a cross-fold mean: sample SD (ddof=1) / sqrt(n) (CR-4).

    This is the number the Ladder keep-margin is supposed to exceed and that
    quantifies whether the Ladder keep-margin exceeds observed fold variation.
    It matters most where the selection is weakest: the validation split is
    12.5% of train_val, so on
    CPTAC-GBM (n=99) and CPTAC-PDAC (n=105) the composite is a mean over folds
    scored on ~10 patients, and a discovery sweep screening ~60 candidates keeps
    the maximum of 60 draws from that distribution.

    Returns ``None`` — never 0.0 — when fewer than two folds carry a finite
    numeric value. 0.0 would read downstream as "measured, and noise-free",
    which is the opposite of "not estimable"; the caller must fall back to the
    predeclared margin instead. Zero spread across two or more folds IS 0.0,
    because that is a real (if degenerate) measurement.

    Non-numeric and non-finite folds are dropped rather than raising: a partial
    run legitimately carries NaN folds (see H-8 / M-15), and booleans are
    excluded because ``bool`` is an ``int`` subclass and would otherwise be
    silently averaged in as 0/1.
    """
    values: list[float] = []
    for v in fold_values or []:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        f = float(v)
        if math.isfinite(f):
            values.append(f)
    n = len(values)
    if n < 2:
        return None
    if len(set(values)) == 1:
        # Exact, rather than the ~1e-17 that (x - mean) leaves behind: identical
        # folds are a real measurement of zero spread and should read as 0.0.
        return 0.0
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    return math.sqrt(variance) / math.sqrt(n)


def recompute_composite_se(result: Mapping[str, object] | None) -> float | None:
    """Derive the cross-fold SE from the result's own validation-fold evidence.

    B1 (claims-alignment): ``composite_se`` gates the Ladder keep-margin
    (``max(δ, k·SE)``), yet it was read verbatim off ``result.json`` — the same
    agent-editable payload the composite machinery refuses to trust. When the
    result carries ``validation_folds`` (the val-only per-fold projection every
    benchmark runner emits), recompute the SE from those composites; the caller
    prefers this value and keeps the reported one only as the legacy fallback.

    Returns ``None`` when fewer than two folds carry a finite composite —
    same contract as :func:`cross_fold_se`.
    """
    if not isinstance(result, Mapping):
        return None
    folds = result.get("validation_folds")
    if not isinstance(folds, list):
        return None
    return cross_fold_se(
        fold.get("composite") for fold in folds if isinstance(fold, Mapping)
    )


def ingest_signal(
    result: Mapping[str, object] | None,
    formula: str | None,
) -> tuple[tuple[str, ...], float | None, float | None]:
    """One sanitation contract for every mouth that turns a result payload into
    graph state (the terminal writer and the reconcile scans — B6).

    Returns ``(leaking_keys, composite_recomputed, se_recomputed)``:

    - ``leaking_keys``: held-out-named keys found inside ``metrics``. Non-empty
      means the payload violates the val-firewall and the caller must ingest it
      as a crash (composite 0.0, metrics dropped) — recomputing over it would
      *average test into selection*, worse than trusting the reported scalar.
    - ``composite_recomputed``: the val-derived composite, or ``None`` to keep
      the reported value (opt-out formula, no usable metrics, or an unknown
      reducer name — the caller logs that case).
    - ``se_recomputed``: the val-fold-derived SE, or ``None`` to keep the
      reported value.
    """
    from automil.firewall import held_out_metric_keys

    if not isinstance(result, Mapping):
        return (), None, None
    leaking = held_out_metric_keys(result.get("metrics"))
    if leaking:
        return leaking, None, None
    try:
        recomputed = recompute_composite(result.get("metrics") or {}, formula)
    except ValueError:
        recomputed = None
    return (), recomputed, recompute_composite_se(result)
