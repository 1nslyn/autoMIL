"""Primary-value recomputation from the agent-facing validation metrics (CR-1b).

The val-firewall's central claim is that **test never drives selection**. But the
orchestrator cannot trust the ``primary_value`` scalar in ``result.json`` verbatim,
because that file is written by agent-editable training code. A script that
computed its primary_value from the sealed test block (by bug or by design) produced a
perfectly schema-valid result, the graph selected on it, and no code path could
detect the leak.

This module closes that hole by deriving the selection signal from the declared
**validation** ``metrics`` block instead of trusting the reported scalar. The
formula is declared per-project in ``automil/config.yaml`` and is either a
reducer over every value in ``metrics`` or a ``val_``-prefixed metric selector:

    scoring:
      formula: mean          # default reducer
      # formula: val_auc     # selector: the primary_value IS this one val metric

``mean`` reproduces the established primary values exactly — classification
``{val_auc, val_bacc}`` → their mean; survival ``{val_c_index}`` → itself — so the
default is behaviour-preserving while making test-derived primary values detectable.
A selector (``val_auc``, ``val_c_index``, …) makes the named validation metric
the entire selection signal; companion metrics stay recorded but do not vote.

Set ``formula: trust_reported`` to opt out (documented as weakening the firewall).
An empty/unset formula means the DEFAULT, never the opt-out — opting out of
CR-1b requires the explicit token.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

DEFAULT_FORMULA = "mean"

#: Absolute tolerance when comparing the reported vs recomputed primary_value.
#: result.json rounds both ``primary_value`` and each metric to 4 decimals, so a
#: faithful writer can differ by ~1e-4 purely from rounding.
PRIMARY_VALUE_TOLERANCE = 1e-3

#: Formula values that explicitly disable validation-metric recomputation.
_OPT_OUT = frozenset({"trust_reported", "none", "reported"})

_REDUCERS = {
    "mean": lambda vs: sum(vs) / len(vs),
    "max": max,
    "min": min,
}


def known_formula(name: object) -> bool:
    """Is this a valid ``scoring.formula`` value? (B2, claims-alignment.)

    Valid: empty/None (framework default), a reducer name, an explicit
    opt-out, or a ``val_``-prefixed METRIC SELECTOR (``val_auc``,
    ``val_c_index``, …) that names the single validation metric the primary_value
    equals. The prefix restriction keeps B2's reducer guarantee — a typo'd
    reducer (``"meen"``) can never be mistaken for a selector — but selector
    NAMES cannot be validated statically (the metric vocabulary is the
    trainer's). A typo'd selector (``"val_aucc"``) therefore passes here and
    is caught at INGEST instead: a recomputing formula over a present metrics
    block that yields no value is a refusal (:func:`recompute_refused`), and
    every ingest mouth fails that payload closed rather than trusting the
    reported scalar. Anything else — e.g. an arithmetic expression like
    ``"(val_auc + val_bacc) / 2"`` — is a config error caught at config
    seeding and ``automil check``.
    """
    if not name:
        return True
    if not isinstance(name, str):
        return False
    return name in _REDUCERS or name in _OPT_OUT or name.startswith("val_")


def formula_recomputes(formula: object) -> bool:
    """Does this formula derive the primary_value from val metrics at ingest?

    True for reducers and selectors (including empty/None, which resolve to
    the default reducer); False only for the explicit opt-out tokens. The
    inverse question of ``formula in _OPT_OUT``, kept as the public name so
    callers never touch the private token set.
    """
    return not (isinstance(formula, str) and formula in _OPT_OUT)


def recompute_primary_value(
    metrics: Mapping[str, object] | None,
    formula: str = DEFAULT_FORMULA,
) -> float | None:
    """Derive the primary_value from validation ``metrics``.

    Returns ``None`` when recomputation does not apply — the project opted out,
    the metrics block is absent/empty (crash and partial results), or it holds no
    finite numeric value. Callers keep the reported primary_value ONLY when
    :func:`recompute_refused` is also False; a present-but-unusable metrics
    block under a recomputing formula must fail closed instead.

    An empty/unset formula resolves to the DEFAULT reducer — never the
    opt-out; opting out requires the explicit token.

    Raises:
        ValueError: unknown formula name. Note the terminal writer catches this
            per-result and keeps the reported primary_value as a last resort — the
            fail-closed guard against a typo'd formula lives at config-load
            time (graph seeding validates with :func:`known_formula`, and
            ``automil check`` reports it), so this raise is defense-in-depth
            for values injected outside the config path.
    """
    formula = formula or DEFAULT_FORMULA
    if formula in _OPT_OUT:
        return None
    if isinstance(formula, str) and formula.startswith("val_"):
        # Metric selector: the primary_value IS this one validation metric. On a
        # few-dozen-slide validation split a rank statistic (val_auc) carries
        # the selection signal; averaging in a threshold-quantized companion
        # (val_bacc jumps ~1/17 per flipped minority slide — the size of the
        # accept-margin floor) injects lattice noise at exactly the decision
        # scale. The companion metrics stay recorded in ``metrics``; they
        # just no longer vote.
        if not isinstance(metrics, Mapping):
            return None
        value = metrics.get(formula)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value) if math.isfinite(float(value)) else None
    reducer = _REDUCERS.get(formula)
    if reducer is None:
        raise ValueError(
            f"unknown scoring.formula {formula!r}; expected one of "
            f"{sorted(_REDUCERS)}, 'trust_reported', or a 'val_'-prefixed "
            f"metric selector"
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


def recompute_refused(
    metrics: Mapping[str, object] | None,
    formula: str | None,
) -> bool:
    """True when CR-1b SHOULD have recomputed but the metrics cannot support it.

    The fail-closed complement of :func:`recompute_primary_value` returning
    ``None`` (B2/B3): under a recomputing formula (reducer or selector), a
    PRESENT, non-empty validation metrics block that yields no recomputed
    value — a typo'd selector, a stripped or non-finite selector key — means
    the payload cannot be scored on the declared estimand at all. Trusting
    the reported scalar there would silently hand selection back to the
    agent-editable number, so every ingest mouth treats this as a
    disagreement-class failure (primary_value 0.0 + audit stamp) instead.

    False when recompute simply does not apply: explicit opt-out, or an
    absent/empty metrics block (crashes and partials legitimately carry
    none — those stay on their reported 0.0-class primary_values).
    """
    if not formula_recomputes(formula):
        return False
    if not isinstance(metrics, Mapping) or not metrics:
        return False
    try:
        return recompute_primary_value(metrics, formula) is None
    except ValueError:
        # Unknown reducer name: the legacy-graph escape hatch documented on
        # recompute_primary_value — the caller already logs it loudly.
        return False


def primary_value_disagrees(reported: float, recomputed: float,
                        tolerance: float = PRIMARY_VALUE_TOLERANCE) -> bool:
    """True when the reported primary_value cannot be explained by the val metrics."""
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
    CPTAC-GBM (n=99) and CPTAC-PDAC (n=105) the primary_value is a mean over folds
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


def recompute_primary_se(
    result: Mapping[str, object] | None,
    formula: str | None,
) -> float | None:
    """Derive the cross-fold SE from the result's own validation-fold evidence.

    B1 (claims-alignment): ``primary_se`` gates the Ladder keep-margin
    (``max(δ, k·SE)``), yet it was read verbatim off ``result.json`` — the same
    agent-editable payload the primary_value machinery refuses to trust. When the
    result carries ``validation_folds`` (the val-only per-fold projection every
    benchmark runner emits), recompute the SE from those primary values; the caller
    prefers this value and keeps the reported one only as the legacy fallback.

    The SE is measured over the SAME per-fold projection the graph stores
    (:func:`fold_primary_value_entries` — fold primary values recomputed from their
    own metrics under the node's formula, unverifiable entries dropped), so
    the marginal SE and the paired SE can never be computed over different
    fold multisets or different per-fold values of the same payload.

    Returns ``None`` when fewer than two folds carry a finite primary_value —
    same contract as :func:`cross_fold_se`.
    """
    entries = fold_primary_value_entries(result, formula)
    if entries is None:
        return None
    return cross_fold_se(entry["primary_value"] for entry in entries)


def fold_primary_value_map(entries: object) -> dict[int, float] | None:
    """``fold_index -> primary_value`` from a ``validation_folds``-shaped list.

    Accepts any list of mappings carrying ``fold_index`` (int) and
    ``primary_value`` (finite number) — the shape shared by ``result.json``
    ``validation_folds``, the baseline root's ``metadata.validation_folds``,
    and the graph-node ``fold_primary_values`` projection. Entries missing either
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
        val = entry.get("primary_value")
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
    longer equals the primary_value difference the accept predicate compares, so
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


def fold_primary_value_entries(
    result: Mapping[str, object] | None,
    formula: str | None,
) -> list[dict] | None:
    """The minimal ``[{fold_index, primary_value}]`` projection of a result's
    ``validation_folds`` — what the graph stores per node so the paired
    keep-margin can pair a child with its parent without re-reading archives.
    Validation-only by construction; ``None`` when no usable folds remain.

    Each entry's primary_value is RECOMPUTED from its own val ``metrics`` with the
    node's OWN formula whenever that block is present (CR-1b at fold
    granularity): result.json is agent-editable, and a reported fold primary_value
    that disagrees with its own metrics could otherwise shape the paired SE
    (uniform deltas → bar drops to the δ floor) while the honest aggregate
    metrics pass every node-level check. Using the node's formula — not a
    hardcoded reducer — keeps the paired-margin identity
    ``primary_value == mean(fold primary_values)`` intact under metric selectors:
    the node primary_value is the per-key mean over folds, so per-fold selector
    values average back to it exactly as per-fold means do.

    Fail-closed at fold granularity, under the SAME full-recorded-evidence
    rule as the trainer and the campaign validator: an entry whose metrics
    block carries any non-finite value — a lost companion included — or that
    cannot support the formula is DROPPED, never trusted and never
    resurrected on its surviving keys (a selector reading only val_auc would
    otherwise re-validate a fold the trainer explicitly nulled, and the
    projection would disagree with the payload's own n_valid_folds). The
    reported value survives only for entries carrying no metrics block at
    all (legacy state artifacts) or under an unknown legacy formula, both of
    which the node-level ingest already logs.
    """
    if not isinstance(result, Mapping):
        return None
    # An unset formula means the framework default, not the recompute opt-out:
    # the projection must keep its CR-1b protection for default-config projects.
    formula = formula or DEFAULT_FORMULA
    raw = result.get("validation_folds")
    if isinstance(raw, list):
        recomputed = []
        for entry in raw:
            if not isinstance(entry, Mapping):
                continue
            metrics_block = entry.get("metrics")
            if isinstance(metrics_block, Mapping) and metrics_block and any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in metrics_block.values()
            ):
                continue
            try:
                fold_value = recompute_primary_value(metrics_block, formula)
            except ValueError:
                fold_value = None   # unknown formula: keep the reported value
            if fold_value is not None:
                entry = {**entry, "primary_value": fold_value}
            elif recompute_refused(metrics_block, formula):
                continue
            recomputed.append(entry)
        raw = recomputed
    folds = fold_primary_value_map(raw)
    if folds is None:
        return None
    return [{"fold_index": i, "primary_value": folds[i]} for i in sorted(folds)]


def ingest_signal(
    result: Mapping[str, object] | None,
    formula: str | None,
) -> tuple[tuple[str, ...], float | None, float | None, bool]:
    """One sanitation contract for every mouth that turns a result payload into
    graph state (the terminal writer and the reconcile scans — B6).

    Returns ``(leaking_keys, primary_value_recomputed, se_recomputed, refused)``:

    - ``leaking_keys``: held-out-named keys found inside ``metrics``. Non-empty
      means the payload violates the val-firewall and the caller must ingest it
      as a crash (primary_value 0.0, metrics dropped) — recomputing over it would
      *average test into selection*, worse than trusting the reported scalar.
    - ``primary_value_recomputed``: the val-derived primary_value, or ``None`` to keep
      the reported value (opt-out formula, no usable metrics, or an unknown
      reducer name — the caller logs that case).
    - ``se_recomputed``: the val-fold-derived SE, or ``None`` to keep the
      reported value.
    - ``refused``: :func:`recompute_refused` — the metrics block is present but
      cannot support the declared recomputing formula (typo'd selector,
      stripped selector key). The caller must fail the payload closed
      (primary_value 0.0 + audit stamp), NOT keep the reported scalar.
    """
    from automil.firewall import held_out_metric_keys

    if not isinstance(result, Mapping):
        return (), None, None, False
    leaking = held_out_metric_keys(result.get("metrics"))
    if leaking:
        return leaking, None, None, False
    metrics = result.get("metrics") or {}
    try:
        recomputed = recompute_primary_value(metrics, formula)
    except ValueError as exc:
        # Reachable only via a legacy graph.json whose STORED formula is
        # invalid (B2 blocks the config path at seeding). Loud, because the
        # fallback is trusting the reported scalar — CR-1b off for this node.
        import logging

        logging.getLogger(__name__).error(
            "ingest: %s — trusting the reported primary_value for this payload", exc
        )
        recomputed = None
    refused = recomputed is None and recompute_refused(metrics, formula)
    return (), recomputed, recompute_primary_se(result, formula), refused
