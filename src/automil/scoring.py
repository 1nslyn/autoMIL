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


def known_formula(name: object) -> bool:
    """Is this a valid ``scoring.formula`` value? (B2, claims-alignment.)

    Valid: empty/None (framework default), a reducer name, or an explicit
    opt-out. Anything else — e.g. an arithmetic expression like
    ``"(val_auc + val_bacc) / 2"`` — is a config error that used to be caught
    only per-result at ERROR level while silently trusting the reported
    composite (CR-1b disabled by a comment's own example). Config seeding and
    ``automil check`` validate with this predicate so the state is
    unrepresentable before any run.
    """
    return (not name) or name in _REDUCERS or name in _OPT_OUT


def recompute_composite(
    metrics: Mapping[str, object] | None,
    formula: str = DEFAULT_FORMULA,
) -> float | None:
    """Derive the composite from validation ``metrics``.

    Returns ``None`` when recomputation does not apply — the project opted out,
    the metrics block is absent/empty (crash and partial results), or it holds no
    finite numeric value. Callers keep the reported composite in that case.

    Raises:
        ValueError: unknown formula name. Note the terminal writer catches this
            per-result and keeps the reported composite as a last resort — the
            fail-closed guard against a typo'd formula lives at config-load
            time (graph seeding validates with :func:`known_formula`, and
            ``automil check`` reports it), so this raise is defense-in-depth
            for values injected outside the config path.
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
    folds = fold_composite_map(result.get("validation_folds"))
    if folds is None:
        return None
    # One parser for every consumer of validation_folds: keyed and deduplicated
    # by fold_index exactly like the paired-margin projection, so the marginal
    # SE and the paired SE can never be computed over different fold multisets
    # of the same payload.
    return cross_fold_se(folds.values())


def fold_composite_map(entries: object) -> dict[int, float] | None:
    """``fold_index -> composite`` from a ``validation_folds``-shaped list.

    Accepts any list of mappings carrying ``fold_index`` (int) and
    ``composite`` (finite number) — the shape shared by ``result.json``
    ``validation_folds``, the baseline root's ``metadata.validation_folds``,
    and the graph-node ``fold_composites`` projection. Entries missing either
    field, or non-finite, are skipped; a duplicate ``fold_index`` keeps the
    last occurrence. Returns ``None`` when nothing usable remains, never ``{}``.
    """
    if not isinstance(entries, list):
        return None
    out: dict[int, float] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        idx = entry.get("fold_index")
        val = entry.get("composite")
        if isinstance(idx, bool) or not isinstance(idx, int):
            continue
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            continue
        f = float(val)
        if math.isfinite(f):
            out[idx] = f
    return out or None


def paired_delta_se(
    child_folds: Mapping[int, float] | None,
    parent_folds: Mapping[int, float] | None,
) -> float | None:
    """SE of the per-fold ``child − parent`` deltas over a shared fold set.

    Parent and child are trained and validated on identical folds under a
    locked seed, so the fold effect — the dominant noise term at n=3 folds on
    ~47-patient validation splits — cancels in the paired difference. This is
    the statistic the Ladder keep-margin should be scaled by; the marginal
    ``cross_fold_se`` of either node measures between-fold heterogeneity
    instead and overstates the comparison noise by a large factor (0.07 vs
    0.01 on the virchow2 canary baseline).

    Returns ``None`` — never 0.0 — unless the two fold-index sets are
    IDENTICAL with at least two folds: on differing sets the paired mean no
    longer equals the composite difference the accept predicate compares, so
    the caller must fall back to the marginal basis. Identical deltas across
    all shared folds ARE 0.0 (a real, degenerate measurement — same contract
    as :func:`cross_fold_se`).
    """
    if not child_folds or not parent_folds:
        return None
    if set(child_folds) != set(parent_folds):
        return None
    deltas = [float(child_folds[i]) - float(parent_folds[i]) for i in sorted(child_folds)]
    if len(deltas) < 2 or not all(math.isfinite(d) for d in deltas):
        return None
    if len(set(deltas)) == 1:
        return 0.0
    mean = sum(deltas) / len(deltas)
    variance = sum((d - mean) ** 2 for d in deltas) / (len(deltas) - 1)
    return math.sqrt(variance) / math.sqrt(len(deltas))


def fold_composite_entries(result: Mapping[str, object] | None) -> list[dict] | None:
    """The minimal ``[{fold_index, composite}]`` projection of a result's
    ``validation_folds`` — what the graph stores per node so the paired
    keep-margin can pair a child with its parent without re-reading archives.
    Validation-only by construction; ``None`` when no usable folds remain.

    Each entry's composite is RECOMPUTED as the mean of its own val ``metrics``
    whenever that block is present (CR-1b at fold granularity): result.json is
    agent-editable, and a reported fold composite that disagrees with its own
    metrics could otherwise shape the paired SE (uniform deltas → bar drops to
    the δ floor) while the honest aggregate metrics pass every node-level
    check. The reported value survives only for entries with no metrics block.
    The mean reducer is the right recompute here regardless of the configured
    formula, because the paired margin is enabled only under ``mean``.
    """
    if not isinstance(result, Mapping):
        return None
    raw = result.get("validation_folds")
    if isinstance(raw, list):
        recomputed = []
        for entry in raw:
            if not isinstance(entry, Mapping):
                continue
            fold_mean = recompute_composite(entry.get("metrics"), "mean")
            if fold_mean is not None:
                entry = {**entry, "composite": fold_mean}
            recomputed.append(entry)
        raw = recomputed
    folds = fold_composite_map(raw)
    if folds is None:
        return None
    return [{"fold_index": i, "composite": folds[i]} for i in sorted(folds)]


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
    except ValueError as exc:
        # Reachable only via a legacy graph.json whose STORED formula is
        # invalid (B2 blocks the config path at seeding). Loud, because the
        # fallback is trusting the reported scalar — CR-1b off for this node.
        import logging

        logging.getLogger(__name__).error(
            "ingest: %s — trusting the reported composite for this payload", exc
        )
        recomputed = None
    return (), recomputed, recompute_composite_se(result)
