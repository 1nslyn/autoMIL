"""Experiment graph: directed tree tracking for multi-branch exploration.

Provides atomic read/write to graph.json. Concurrent writers (daemon +
CLI) coordinate through an advisory ``flock`` on a sidecar ``.lock``
file; see ``locked_update``.

Immutability (L-8a, audit 2026-07-23): this module's own methods
(``add_executed``, ``promote``, ``mark_failed``, ``reconcile``, ...)
mutate node dicts stored in ``self._data`` in place, field by field —
they do NOT rebuild and reassign a fresh dict per update. This is a
deliberate, pragmatic choice, not an oversight: ``self._data`` is
single-owner for the lifetime of one ``locked_update`` transaction (the
flock above serializes every writer), and it is serialized wholesale on
``save()``, so there is no aliasing surface between transactions — every
``locked_update`` call constructs a brand-new ``ExperimentGraph`` from a
fresh ``json.loads`` of the file, so no Python object outlives its lock.
Converting every one of those in-place field assignments to copy-on-write
would be a sweeping rewrite of this module for no correctness gain, so it
is deliberately NOT done.

The one nested structure that genuinely IS reachable from two writers is
a node's ``metadata`` sub-dict: ``gate/evaluate.py`` creates a gate-eval
child node via a SHALLOW ``dict(node)`` copy, which leaves the child's
``metadata`` key aliased to the same dict object as its source. A caller
that then mutates ``gnode["metadata"]`` in place (``.setdefault(...)
.update(...)``) could silently corrupt whichever node it is aliased
with. ``merged_metadata`` below is the copy-on-write fix for exactly that
structure, used by ``terminal_writer``, ``cli/cancel``, ``cli/propose``,
``cli/reconcile``, and the daemon's cap-refusal path — the sites where a
node read via ``get_node()`` needs to add or change ``metadata`` keys.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import io
import json
import logging
import math
import os
import tempfile
import tokenize
from datetime import datetime, timedelta, timezone
from pathlib import Path

from automil.scoring import DEFAULT_FORMULA as _DEFAULT_SCORING_FORMULA

logger = logging.getLogger(__name__)


def node_cell_id(node: dict | None) -> str | None:
    """Return the budget-cell id a graph node belongs to, or ``None`` (CELL-1).

    Two shapes are accepted because two writers exist: ``automil submit`` stamps
    a top-level ``cell_id`` (parallel to ``config_hash``), while gate-eval
    children are created with the id under ``metadata`` (``gate/evaluate.py``).

    ``None`` is the legacy answer — nodes created before cells existed carry no
    identity and must simply never match a lookup.
    """
    if not isinstance(node, dict):
        return None
    cell_id = node.get("cell_id")
    if not cell_id:
        meta = node.get("metadata")
        cell_id = meta.get("cell_id") if isinstance(meta, dict) else None
    return cell_id if isinstance(cell_id, str) and cell_id else None


def merged_metadata(node: dict | None, updates: dict) -> dict:
    """Copy-on-write merge into a node's ``metadata`` sub-dict (L-8a).

    Several call sites (``terminal_writer``, ``cli/cancel``, ``cli/propose``,
    ``cli/reconcile``, the daemon's cap-refusal path) read a node via
    ``get_node()`` and then need to add or change a few ``metadata`` keys.
    The naive way — ``gnode.setdefault("metadata", {}).update(updates)`` or
    ``gnode.setdefault("metadata", {})[k] = v`` — mutates whatever dict
    object is already stored at ``node["metadata"]``, in place.

    That is reachable from two writers: ``gate/evaluate.py`` creates a
    gate-eval child node via a SHALLOW copy of a node dict (``dict(node)``),
    which leaves the child's ``metadata`` key pointing at the exact same
    dict object as its source node's. An in-place mutation through either
    alias would silently corrupt the other — a plain dict has no
    copy-on-write semantics of its own.

    Callers use this as ``gnode["metadata"] = merged_metadata(gnode,
    {...})``: the OUTER node dict is still updated by direct key assignment
    (matching every other field mutation in this codebase — self._data is
    single-owner per flock-guarded ``locked_update`` transaction and
    serialized wholesale on save, so that part has no aliasing surface and
    converting it too would be a much larger, unrelated rewrite). Only this
    specific NESTED, cross-writer-reachable structure is made copy-on-write.

    Tolerates ``node=None`` and a non-dict ``metadata`` value (legacy/corrupt
    data) by treating the base as empty, matching ``node_cell_id``'s
    defensiveness above.
    """
    base = (node or {}).get("metadata")
    if not isinstance(base, dict):
        base = {}
    return {**base, **updates}


def _accept_margin(meta: dict | None) -> float:
    """Predeclared Ladder keep-margin δ from ``meta.scoring.accept_margin``.

    δ=0.0 (the default) reproduces plain primary_value dominance. A δ>0 requires a
    child to beat its parent's validation primary_value by more than the margin
    before it is kept — a Ladder-style gate against promoting within-noise
    improvements over a long agentic search.
    """
    try:
        raw = ((meta or {}).get("scoring") or {}).get("accept_margin", 0.0)
        return max(0.0, float(raw or 0.0))   # clamp: a negative δ would invert the gate
    except (TypeError, ValueError, AttributeError):
        return 0.0


def _accept(child_primary_value: float, parent_primary_value: float, margin: float = 0.0) -> bool:
    """Keep a child iff its primary_value beats the parent's by more than ``margin``.

    The single keep/discard predicate, shared by every decision site (the live
    terminal writer, descendant re-evaluation, and both reconcile paths) so the
    Ladder margin is applied uniformly. margin=0.0 → strict dominance.
    """
    return child_primary_value > parent_primary_value + margin


#: One SE. The point of CR-4 is that the bar is the measured noise; a default of
#: 0 would ship the feature switched off, which is how it got missed the first time.
DEFAULT_SE_MULTIPLIER = 1.0

#: B5 (claims-alignment): the statuses that count as "kept" for best-node and
#: certify selection. `candidate` (nominated for the gate) and `registered`
#: (gate-passed) are *better*-validated keep-states — walking `keep` alone
#: silently evicted a node from `best_node` and from `automil certify`'s
#: default target the moment it was nominated.
KEEP_CLASS = frozenset({"keep", "candidate", "registered"})


def node_primary_se(node: dict | None) -> float | None:
    """Cross-fold SE of a node's primary_value, or ``None`` if it was never measured.

    ``None`` covers three real cases and they must not be conflated with zero:
    a legacy node written before CR-4, a partial run with fewer than two finite
    folds (H-8 / M-15), and a corrupt or negative value. A caller seeing ``None``
    falls back to the predeclared δ; a caller seeing 0.0 is being told the folds
    genuinely agreed.
    """
    if not isinstance(node, dict):
        return None
    raw = node.get("primary_se")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    val = float(raw)
    if not math.isfinite(val) or val < 0:
        return None
    return val


def _se_multiplier(meta: dict | None) -> float:
    """How many SEs a child must clear, from ``meta.scoring.se_multiplier``.

    Clamped at 0: a negative multiplier would turn the noise floor into a
    discount, letting a noisy parent be beaten by *less* than nothing.
    """
    try:
        raw = ((meta or {}).get("scoring") or {}).get("se_multiplier", DEFAULT_SE_MULTIPLIER)
        if raw is None:
            return DEFAULT_SE_MULTIPLIER
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return DEFAULT_SE_MULTIPLIER


def node_fold_primary_values(node: dict | None) -> dict[int, float] | None:
    """``fold_index -> primary_value`` for a node, from whichever source it has.

    Two sources, one shape (entries of ``{fold_index, primary_value, ...}``):

    - ``node["fold_primary_values"]`` — written at ingest for every executed node
      (terminal writer, both reconcile scans, ``reconcile --refresh``).
    - ``node["metadata"]["validation_folds"]`` — the baseline root's folds,
      written by the campaign controller's ``_ensure_discovery_baseline_root``.
      The baseline is the dominant parent of a discovery cell, and it never
      passes through the terminal writer, so the paired margin must read the
      metadata form or it would silently fall back to the marginal SE in
      exactly the topology it exists for.

    Returns ``None`` when neither source yields a usable fold map.
    """
    if not isinstance(node, dict):
        return None
    from automil.scoring import fold_primary_value_map

    folds = fold_primary_value_map(node.get("fold_primary_values"))
    if folds is not None:
        return folds
    meta = node.get("metadata")
    if isinstance(meta, dict):
        return fold_primary_value_map(meta.get("validation_folds"))
    return None


def _formula_pairs_folds(meta: dict | None) -> bool:
    """True when the primary_value is the mean of the per-fold primary_values.

    The paired margin substitutes ``SE(per-fold child−parent deltas)`` for the
    marginal SE while the accept predicate still compares node primary_values.
    That substitution is coherent under the ``mean`` reducer (the default) AND
    under ``val_*`` metric selectors: the node-level metric is the per-key
    mean over folds, so a per-fold selector value averages back to the node
    primary_value exactly as per-fold means do. Under ``max``/``min`` (or an
    opt-out formula trusting reported scalars) the identity breaks, so the
    margin falls back to the marginal basis.
    """
    formula = ((meta or {}).get("scoring") or {}).get("formula")
    if formula in (None, "", "mean"):
        return True
    return isinstance(formula, str) and formula.startswith("val_")


def effective_accept_margin(
    meta: dict | None, parent_node: dict | None, child_node: dict | None = None,
) -> float:
    """The margin actually applied: ``max(predeclared δ, k × SE)`` (CR-4).

    The SE basis is chosen by evidence available, best first:

    **Paired** — when parent and child both carry primary values for the SAME fold
    set (and the reducer is ``mean``; see :func:`_formula_pairs_folds`), the
    basis is ``SE(per-fold child−parent deltas)``. Runs share folds under a
    locked seed, so the fold effect — the dominant noise term — cancels in the
    difference; on the canary cells this basis is 3–6× tighter than the
    marginal SE and moves the detectable-effect floor into the range train-only
    recipe changes actually produce. The paired basis is child-derived by
    construction: the screen becomes "paired t > k, with floor δ", which is the
    standard form for a noisy-CV accept rule. The old incumbent-only argument
    (child-derived bars let the argmax select on the gate) traded a real
    multiplicity concern for a bar so wide it discarded everything — the
    virchow2 canary discarded 30/30 attempts against a bar its per-fold oracle
    could not reach. The multiplicity concern is real and handled downstream:
    at k=1 a null child passes with p ≈ 0.21 (one-sided t, 2 df), so over ~30
    screened candidates several false keeps are EXPECTED — promotion re-runs
    the top-10 on held-back folds 3/4 and the winner is selected on the 5-fold
    mean, which is the arbitration this screen defers to. A zero paired SE
    (fold-uniform delta) keeps at the δ floor; that is legitimately strong
    paired evidence at this n, but note the sign-test bound: n=3 uniform
    deltas reach one-sided p = 1/8 at best. NEVER report a keep as
    significance — it is a search-steering screen.

    **Marginal** — otherwise, ``k × parent primary_se`` (the pre-existing
    CR-4 basis). Still monotone: measured noise can only RAISE the bar above
    the predeclared δ; a campaign that predeclared δ=0.05 must not silently
    drop to 0.01 because one parent happened to have a tight CV.
    """
    delta = _accept_margin(meta)
    _basis, se = margin_se_basis(meta, parent_node, child_node)
    if se is None:
        return delta
    return max(delta, _se_multiplier(meta) * se)


def margin_se_basis(
    meta: dict | None, parent_node: dict | None, child_node: dict | None = None,
) -> tuple[str, float | None]:
    """The SE basis :func:`effective_accept_margin` actually applies.

    Returns ``("paired", se)``, ``("marginal", se)`` or ``("none", None)``.
    Public so display surfaces (``automil rank``) label the evidence with the
    SAME choice the gate makes — printing the raw paired SE beside a bar that
    fell back to the marginal basis misrepresents the decision's evidence.
    """
    if child_node is not None and _formula_pairs_folds(meta):
        from automil.scoring import paired_delta_se

        child_folds = node_fold_primary_values(child_node)
        parent_folds = node_fold_primary_values(parent_node)
        # Identity guard: the paired basis substitutes SE(per-fold deltas)
        # while _accept still compares node primary values, so it is coherent only
        # when each node's primary_value IS the mean of its fold primary values. The
        # reducer name alone cannot guarantee that — a recovery aggregate with
        # sparse per-fold metrics reports a primary_value whose denominator
        # differs from the fold vector's. Verify the property on the data in
        # hand and fall back to the marginal basis when it fails.
        if _primary_value_matches_folds(child_node, child_folds) and \
                _primary_value_matches_folds(parent_node, parent_folds):
            paired_se = paired_delta_se(child_folds, parent_folds)
            if paired_se is not None:
                return "paired", paired_se
    se = node_primary_se(parent_node)
    if se is None:
        return "none", None
    return "marginal", se


def _guard_declaration(meta: dict | None) -> tuple[str, float] | None:
    """``(metric, margin)`` from ``meta.scoring.guard``, or ``None`` when undeclared.

    The companion non-inferiority guard names ONE validation metric a kept
    child may not regress on by more than ``margin``. It exists because the
    primary signal is deliberately single-metric: ``val_auc`` navigates
    because its resolution (one swapped pair) is finer than the effect sizes
    being searched for, while a threshold-quantized companion like
    ``val_bacc`` cannot vote without injecting lattice noise at the decision
    scale. The guard restores the companion's veto without giving it a vote —
    it can only reject, never promote, so it adds no noise to the argmax.

    ``margin`` is PREDECLARED per dataset+task, never estimated from the
    comparison it gates: a non-inferiority margin derived from the data under
    test would let a noisy child widen its own acceptance region. The consumer
    derives it from its own frozen validation splits — the framework only
    consumes the declaration and never learns what the metric means.

    Raises:
        ValueError: the block is PRESENT but unreadable — a partial pair, a
            non-numeric or negative margin. Refused rather than half-applied:
            a margin without a metric names nothing to guard, a metric without
            a margin would guard at zero tolerance, and either read as "no
            guard" would let a campaign claim a protection it is not applying.
            :func:`guard_basis` turns this into a fail-CLOSED verdict, the same
            rule an unknown frozen ``formula`` gets.
    """
    raw = ((meta or {}).get("scoring") or {}).get("guard")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"scoring.guard must be a mapping, got {type(raw).__name__}")
    metric = raw.get("metric")
    margin = raw.get("margin")
    if not isinstance(metric, str) or not metric:
        raise ValueError(f"scoring.guard.metric must be a metric name, got {metric!r}")
    if isinstance(margin, bool) or not isinstance(margin, (int, float)):
        raise ValueError(f"scoring.guard.margin must be a number, got {margin!r}")
    value = float(margin)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"scoring.guard.margin must be finite and >= 0, got {margin!r}")
    return metric, value


#: Float-representation slack for the guard comparison. Both the margin and
#: the metrics it is compared against are DECIMALS, and binary floats put a
#: drop of exactly the margin a few ulps on the wrong side of it
#: (``0.5408 - 0.5507 == -0.00990000000000002``, which is "worse than" 0.0099).
#: Without this, "a drop of exactly `margin` passes" held for only a third of
#: the cases it names. Five orders of magnitude below any real recording grid,
#: so it can never admit a genuinely larger drop.
_GUARD_EPS = 1e-9


def _finite(raw: object) -> float | None:
    """``raw`` as a finite float, or ``None``. ``bool`` is an ``int`` subclass."""
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


def _node_metric(node: dict | None, metric: str) -> float | None:
    """A node's companion metric out of its opaque metrics dict.

    Reading one named key out of ``metrics`` is the same contract a ``val_*``
    primary selector already uses (:func:`scoring.recompute_primary_value`) —
    the framework stays vocabulary-agnostic and the consumer's declared name
    is the only coupling.

    ONE source, deliberately. An earlier version also fell back to the mean of
    ``metadata.validation_folds`` so that a bootstrapped baseline root — which
    records no aggregate — could still be guarded. That fallback was worse than
    the hole it filled, twice over: the fold values are unrounded while every
    run-recorded aggregate is not, so the two sides of the comparison landed on
    different grids and the margin's grid alignment stopped covering the
    guard's most common comparison; and ``metadata`` is merged from the
    agent-authored result payload, so a node could carry a healthy fold value
    from one ingest and have it resurrected after a later ingest dropped the
    metric — defeating the child-side fail-closed rule. The consumer records
    the aggregate on its own grid instead (see the campaign's baseline root),
    which is both correct and one mechanism rather than two.
    """
    if not isinstance(node, dict):
        return None
    metrics = node.get("metrics")
    if not isinstance(metrics, dict):
        return None
    return _finite(metrics.get(metric))


def guard_basis(
    meta: dict | None, parent_node: dict | None, child_node: dict | None,
) -> tuple[str, float | None, str | None]:
    """The companion guard's verdict, observed child−parent delta, and metric.

    Returns ``(verdict, delta, metric)`` where verdict is ``"none"`` (no
    guard applies), ``"pass"``, or ``"fail"``; ``delta`` is ``None`` whenever
    it could not be measured, and ``metric`` is ``None`` only when nothing was
    validly declared. Public so display surfaces label a discard with the SAME
    evidence the gate used — a node rejected by the guard while winning on
    the primary signal is otherwise indistinguishable from an ordinary loss —
    and carrying the metric name here keeps those surfaces from re-parsing a
    declaration that may be exactly what is broken.

    Three asymmetries, each load-bearing:

    - **A child without the metric fails closed, whatever the parent has.**
      ``metrics`` is written by agent-editable training code; if dropping a
      key disabled the guard, dropping the key would be the dominant strategy.
      This is checked FIRST so the exemption below can never be inherited.
    - **A parent without the metric opens the guard.** There is nothing to be
      non-inferior TO — a legacy incumbent, a guard added mid-campaign, or a
      crashed parent whose metrics were cleared at ingest. The child has
      already been required to carry the metric, so this exempts one
      comparison, never a lineage. It is not weaker than the primary gate
      either: a parent with no evidence scores 0.0, so ``_accept`` is equally
      vacuous against it. Comparing against the nearest healthy ancestor
      instead of the literal parent would be a change to the whole
      parent-relative gate, not something for the guard to special-case.
    - **A drop of exactly ``margin`` passes.** ``margin`` is one quantization
      step of the companion metric on this cell's validation splits, so a
      drop that size is arithmetically explainable by a single validation
      slide changing side — the finest distinction the metric can make, and
      therefore not evidence. Anything larger is rejected.
    """
    try:
        declared = _guard_declaration(meta)
    except ValueError:
        # Declared but unreadable — a hand-edited or corrupt frozen graph
        # (the config path raises at seeding). Fail CLOSED for the same
        # reason an unknown frozen formula does: one typo must not silently
        # switch a declared protection off for the whole graph.
        return "fail", None, None
    if declared is None or parent_node is None:
        return "none", None, None
    metric, margin = declared
    # CHILD FIRST. The child-side rule is the anti-gaming rule, so it must not
    # be conditional on the parent's evidence: checking the parent first made
    # the exemption HEREDITARY — a metric-less child under a metric-less parent
    # was kept, and became a metric-less parent itself, so a trainer that never
    # wrote the key disabled the guard for its whole lineage.
    child_value = _node_metric(child_node, metric)
    if child_value is None:
        return "fail", None, metric
    parent_value = _node_metric(parent_node, metric)
    if parent_value is None:
        return "none", None, metric
    delta = child_value - parent_value
    return ("fail" if delta + margin < -_GUARD_EPS else "pass"), delta, metric


def _primary_value_matches_folds(node: dict | None, folds: dict[int, float] | None) -> bool:
    """True when the node's primary_value equals the mean of its fold primary_values
    (within the ingest rounding tolerance) — the identity the paired margin
    rests on."""
    if not isinstance(node, dict) or not folds:
        return False
    primary_value = node.get("primary_value")
    if isinstance(primary_value, bool) or not isinstance(primary_value, (int, float)):
        return False
    from automil.scoring import PRIMARY_VALUE_TOLERANCE

    mean = sum(folds.values()) / len(folds)
    return abs(float(primary_value) - mean) <= PRIMARY_VALUE_TOLERANCE


def keep_or_discard(meta: dict | None, parent_node: dict | None, child_node: dict) -> str:
    """THE keep/discard decision — every accept site routes through here.

    ``child_node`` is REQUIRED (no default): a site that cannot supply child
    evidence must say so by constructing the evidence dict explicitly, never
    by omission — an omitted child would silently select the wider marginal
    bar and stamp a genuinely improved node ``discard``, indistinguishable
    from a real rejection (the exact failure the paired margin exists to fix).
    Root semantics (no parent): keep iff primary_value > 0, margin N/A.

    Two conditions, and the second can only ever REJECT: the child must beat
    its parent on the primary signal by more than the Ladder margin, and it
    must not have regressed past the declared companion guard
    (:func:`guard_basis`). Giving the companion metric a veto but no vote is
    what lets selection stay single-metric — the argmax is taken over the
    primary signal alone, so the companion's quantization noise never enters
    it — while still blocking the failure a single-metric search is accused
    of: an AUC gain bought with a real balanced-accuracy collapse.
    """
    primary_value = child_node.get("primary_value")
    primary_value = float(primary_value) if isinstance(primary_value, (int, float)) \
        and not isinstance(primary_value, bool) else 0.0
    if parent_node is None:
        return "keep" if primary_value > 0 else "discard"
    p_comp = parent_node.get("primary_value")
    p_comp = float(p_comp) if isinstance(p_comp, (int, float)) \
        and not isinstance(p_comp, bool) else 0.0
    margin = effective_accept_margin(meta, parent_node, child_node)
    if not _accept(primary_value, p_comp, margin):
        return "discard"
    return "discard" if guard_basis(meta, parent_node, child_node)[0] == "fail" \
        else "keep"


def _config_accept_margin(graph_path) -> float | None:
    """Best-effort read of ``scoring.accept_margin`` from the sibling config.yaml.

    Lets an operator predeclare the Ladder keep-margin δ per-dataset in
    ``automil/config.yaml`` (``scoring.accept_margin``); a fresh graph seeds its
    ``meta.scoring.accept_margin`` from it. Returns None when there is no config,
    the key is absent, or it cannot be parsed as a number (callers fall back to
    0.0). The graph stays config-agnostic everywhere else.
    """
    config_path = Path(graph_path).parent / "config.yaml"
    if not config_path.exists():
        return None
    try:
        import yaml
        cfg = yaml.safe_load(config_path.read_text()) or {}
        raw = (cfg.get("scoring") or {}).get("accept_margin")
        return max(0.0, float(raw)) if raw is not None else None   # clamp negative δ
    except Exception as exc:  # noqa: BLE001 — best-effort seed; bad config → default
        logger.warning("Could not read scoring.accept_margin from %s: %s", config_path, exc)
        return None


def _config_se_multiplier(graph_path) -> float | None:
    """Best-effort read of ``scoring.se_multiplier`` from the sibling config.yaml (CR-4).

    Predeclared per-dataset alongside δ, and clamped at 0 for the same reason
    ``_se_multiplier`` clamps: a negative multiplier would turn the measured
    noise floor into a discount. Returns None when there is no config or the key
    is absent, so the caller falls back to ``DEFAULT_SE_MULTIPLIER`` (one SE)
    rather than to 0, which would ship the gate switched off.
    """
    config_path = Path(graph_path).parent / "config.yaml"
    if not config_path.exists():
        return None
    try:
        import yaml
        cfg = yaml.safe_load(config_path.read_text()) or {}
        raw = (cfg.get("scoring") or {}).get("se_multiplier")
        return max(0.0, float(raw)) if raw is not None else None
    except Exception as exc:  # noqa: BLE001 — best-effort seed; bad config → default
        logger.warning("Could not read scoring.se_multiplier from %s: %s", config_path, exc)
        return None


def _config_scoring_formula(graph_path) -> str | None:
    """Best-effort read of ``scoring.formula`` from the sibling config.yaml (CR-1b).

    Lets an operator predeclare the primary_value reducer per-dataset. Returns None
    when there is no config or the key is absent (callers fall back to the
    framework default). The graph stays config-agnostic everywhere else.
    """
    config_path = Path(graph_path).parent / "config.yaml"
    if not config_path.exists():
        return None
    try:
        import yaml
        cfg = yaml.safe_load(config_path.read_text()) or {}
        raw = (cfg.get("scoring") or {}).get("formula")
        value = str(raw) if raw else None
    except Exception as exc:  # noqa: BLE001 — best-effort seed; bad config → default
        logger.warning("Could not read scoring.formula from %s: %s", config_path, exc)
        return None
    # B2 (claims-alignment): validate OUTSIDE the blanket except — an unknown
    # reducer name used to fall through per-result to "trusting the reported
    # primary_value", i.e. a typo (or following the template's old arithmetic
    # examples) silently disabled CR-1b. Fail at config load instead.
    from automil.scoring import known_formula
    if value is not None and not known_formula(value):
        raise ValueError(
            f"scoring.formula {value!r} in {config_path} is not a known reducer. "
            "Valid values: mean | max | min (reducers over the validation "
            "metrics), a 'val_'-prefixed metric selector (val_auc | "
            "val_c_index | ... — the primary_value IS that one metric), or "
            "trust_reported (explicit opt-out, weakens the val-firewall). "
            "Arithmetic expressions are not evaluated."
        )
    return value


def _config_scoring_guard(graph_path) -> dict | None:
    """Best-effort read of ``scoring.guard`` from the sibling config.yaml.

    The companion non-inferiority guard, predeclared per-dataset alongside δ
    and frozen the same way. Returns the ``{metric, margin}`` block (plus any
    ``basis`` provenance string, carried through verbatim so the frozen graph
    records WHY its margin is that number) or ``None`` when unset.

    Validated OUTSIDE the blanket except for the same reason
    :func:`_config_scoring_formula` is: a declared-but-malformed guard must
    fail at config load, not degrade into "no guard" at gate time and let a
    campaign claim a protection it never applied.
    """
    config_path = Path(graph_path).parent / "config.yaml"
    if not config_path.exists():
        return None
    try:
        import yaml
        cfg = yaml.safe_load(config_path.read_text()) or {}
        raw = (cfg.get("scoring") or {}).get("guard")
    except Exception as exc:  # noqa: BLE001 — best-effort seed; bad config → default
        logger.warning("Could not read scoring.guard from %s: %s", config_path, exc)
        return None
    if raw is None:
        return None
    try:
        _guard_declaration({"scoring": {"guard": raw}})
    except ValueError as exc:
        raise ValueError(f"scoring.guard in {config_path} is invalid: {exc}") from exc
    return dict(raw)


@contextlib.contextmanager
def locked_update(graph_path: str | Path, *, technique_map: dict[str, str] | None = None):
    """Read-modify-write context manager for graph.json under a fcntl lock.

    Use this whenever a process needs to mutate ``graph.json`` to prevent
    lost updates between the daemon and CLI:

        with locked_update(path) as graph:
            graph.add_proposed(...)
            # graph.save() runs on context exit

    Acquires an exclusive POSIX advisory lock on ``<graph_path>.lock``
    BEFORE constructing the in-memory ExperimentGraph, so the snapshot
    read by the constructor cannot be invalidated by another writer
    until the block exits.

    Atomic-rename in ``save()`` alone prevented torn writes but not
    lost updates; this context manager is the fix for the race the
    audit flagged.

    ``technique_map`` is forwarded to the constructed ExperimentGraph so
    consumer-supplied vocabularies (declared in ``automil/config.yaml``:
    ``scoring.technique_map``) drive auto-extraction inside the locked
    block. None preserves the framework's empty default.
    """
    path = Path(graph_path)
    lock_path = path.with_suffix(path.suffix + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_f = open(lock_path, "w")
    try:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        graph = ExperimentGraph(path=path, technique_map=technique_map)
        yield graph
        graph.save()
    finally:
        try:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
        finally:
            lock_f.close()


class ExperimentGraph:
    # Generic by default. Consumers that want technique-name normalisation
    # supply their own dict via ``technique_map=`` on the constructor (or
    # to ``import_from_tsv``). The framework ships no domain-specific
    # vocabulary here — the empty default is the contract.
    DEFAULT_TECHNIQUE_MAP: dict[str, str] = {}

    def __init__(self, path: str | Path, technique_map: dict[str, list[str]] | None = None, data: dict | None = None):
        self.path = Path(path)
        self._technique_map = technique_map if technique_map is not None else self.DEFAULT_TECHNIQUE_MAP
        loaded_from_disk = False
        if data is not None:
            self._data = data
        elif self.path.exists():
            self._data = json.loads(self.path.read_text())
            loaded_from_disk = True
        else:
            self._data = {}
        # Capture the on-disk schema_version BEFORE setdefault fills in the new
        # default of 2.  Absent key → 1 (legacy); present → whatever was stored.
        # Used by the DBT-01 migration gate below.
        _on_disk_schema_version = self._data.get("schema_version", 1)
        # Normalize: fill in missing top-level / meta keys with defaults
        # so legacy schemas and fresh-init paths both work. When loading
        # an existing file that's missing keys, log a warning — partial-
        # write corruption silently filled in with defaults would mask
        # real data loss, and operators need a paper trail.
        # Ladder keep-margin δ: a fresh (or legacy) graph seeds accept_margin from
        # the sibling config.yaml so an operator can predeclare it per-dataset.
        # Once persisted in graph.json, the stored value wins over config.
        _meta = self._data.get("meta")
        _has_margin = (
            isinstance(_meta, dict)
            and isinstance(_meta.get("scoring"), dict)
            and "accept_margin" in _meta["scoring"]
        )
        _cfg_margin = None if _has_margin else _config_accept_margin(self.path)
        _default_margin = _cfg_margin if _cfg_margin is not None else 0.0
        # CR-4: the SE multiplier is predeclared alongside δ and frozen the same
        # way — once in graph.json the stored value wins, so a campaign cannot
        # loosen its own gate halfway through.
        _has_mult = (
            isinstance(_meta, dict)
            and isinstance(_meta.get("scoring"), dict)
            and "se_multiplier" in _meta["scoring"]
        )
        _cfg_mult = None if _has_mult else _config_se_multiplier(self.path)
        _default_mult = _cfg_mult if _cfg_mult is not None else DEFAULT_SE_MULTIPLIER
        # The companion non-inferiority guard, frozen on the same terms as δ:
        # its margin is a predeclared per-cell constant derived from that
        # cell's validation splits, so a mid-campaign config edit must not be
        # able to widen (or quietly remove) it.
        _has_guard = (
            isinstance(_meta, dict)
            and isinstance(_meta.get("scoring"), dict)
            and "guard" in _meta["scoring"]
        )
        _cfg_guard = None if _has_guard else _config_scoring_guard(self.path)
        # CR-1b: the primary_value reducer, predeclarable per-dataset in config.yaml.
        _cfg_formula = _config_scoring_formula(self.path)
        _default_formula = _cfg_formula if _cfg_formula else _DEFAULT_SCORING_FORMULA
        defaults = {
            "schema_version": 3,
            "meta": {
                "best_primary_value": 0.0,
                "best_node_id": None,
                "total_executed": 0,
                "total_proposed": 0,
                "next_id": 1,
                "baseline_primary_value": 0.0,
                "scoring": {
                    "exploration_weight": 0.005,
                    "novelty_weight": 0.003,
                    "accept_margin": _default_margin,
                    "se_multiplier": _default_mult,
                    "formula": _default_formula,
                },
            },
            "nodes": {},
            "technique_stats": {},
        }
        missing_top = [k for k in defaults if k not in self._data]
        for k, v in defaults.items():
            self._data.setdefault(k, v if not isinstance(v, dict) else dict(v))
        missing_meta = [k for k in defaults["meta"] if k not in self._data["meta"]]
        for mk, mv in defaults["meta"].items():
            self._data["meta"].setdefault(mk, mv if not isinstance(mv, dict) else dict(mv))
        # Backfill accept_margin into a pre-existing scoring block (legacy graphs
        # that predate the Ladder gate) so a predeclared config δ still applies.
        if not isinstance(self._data["meta"].get("scoring"), dict):
            self._data["meta"]["scoring"] = dict(defaults["meta"]["scoring"])
        # M-1 (audit 2026-07-23): backfill EVERY scoring key (not only
        # accept_margin) so a legacy / hand-edited scoring block missing
        # exploration_weight or novelty_weight cannot KeyError in
        # recalculate_scores() and silently turn every reconcile() into a no-op.
        for _sk, _sv in defaults["meta"]["scoring"].items():
            self._data["meta"]["scoring"].setdefault(_sk, _sv)
        self._data["meta"]["scoring"].setdefault("accept_margin", _default_margin)
        # Seeded only when declared: an undeclared guard leaves no key at all,
        # so a project without one keeps a graph.json free of a null it would
        # have to explain (and `guard_basis` reads "absent" as "no guard").
        if _cfg_guard is not None:
            self._data["meta"]["scoring"].setdefault("guard", _cfg_guard)
        if loaded_from_disk and (missing_top or missing_meta):
            # Top-level missing keys are the more alarming signal (file
            # exists but is structurally incomplete). Meta-only gaps are
            # usually schema-version drift from an old graph and are
            # safe to fill silently — but we still report the meta keys
            # so a schema migration audit can pick them up.
            logger.warning(
                "graph.json at %s loaded with missing keys "
                "(top-level=%r, meta=%r); filled with defaults. If this "
                "is not a known schema migration, check for partial-"
                "write corruption.",
                self.path, missing_top, missing_meta,
            )
        # DBT-01: migrate pre-D-200 nodes (flat metric keys) to metrics-dict layout on read.
        # Gate: on-disk schema_version < 2 AND node lacks "metrics" — idempotent on post-D-200.
        # Uses _on_disk_schema_version (captured before setdefault filled in the new default of 2)
        # so that graphs written without a schema_version key are correctly treated as legacy.
        # Migration is in-memory only; caller decides when to save.
        if _on_disk_schema_version < 2:
            _LEGACY_METRIC_KEYS = ("val_auc", "val_bacc", "test_auc", "test_bacc")
            _migrated = 0
            for _node in self._data.get("nodes", {}).values():
                if "metrics" not in _node:
                    # Only keys the node ACTUALLY carried. Defaulting an absent
                    # key to 0.0 invents evidence: a legacy node that never
                    # recorded the companion metric would look as though it
                    # had, which defeats the guard's child-side fail-closed
                    # rule and silences the `automil check` warning about
                    # adding a guard to a graph with history (it tests for the
                    # key's presence). It also mattered before the guard: an
                    # all-zero synthetic block makes CR-1b's mean reducer
                    # recompute the node's selection signal as 0.0.
                    _node["metrics"] = {
                        k: _node[k] for k in _LEGACY_METRIC_KEYS if k in _node
                    }
                    _migrated += 1
            self._data["schema_version"] = 2
            if loaded_from_disk and _migrated > 0:
                logger.warning(
                    "graph.json at %s: legacy schema (pre-D-200) detected; "
                    "migrated %d node(s) to metrics-dict layout on read. "
                    "Re-save to persist the migration.",
                    self.path,
                    _migrated,
                )
        # Schema 3: the "composite" concept was retired (single-metric
        # optimization); the selection field is `primary_value` and its SE is
        # `primary_se`. A pre-rename graph read by the every-`.get(...)`-
        # defaults-to-0.0 code silently zeroes every node's selection signal
        # and strands the real best under the orphaned keys — so migrate on
        # read, same in-memory-only contract as DBT-01 above.
        if _on_disk_schema_version < 3:
            _RENAMES = (
                ("composite", "primary_value"),
                ("composite_se", "primary_se"),
                ("fold_composites", "fold_primary_values"),
            )
            _META_RENAMES = (
                ("best_composite", "best_primary_value"),
                ("baseline_composite", "baseline_primary_value"),
            )
            from automil.firewall import is_held_out_metric_key as _is_held_out

            _migrated = 0
            for _node in self._data.get("nodes", {}).values():
                _hit = False
                for _old, _new in _RENAMES:
                    if _old in _node:
                        _node.setdefault(_new, _node.pop(_old))
                        _hit = True
                # Legacy metrics hygiene: pre-firewall writers copied the
                # flat test_* keys into `metrics` (DBT-01 above still does,
                # for schema-1 graphs) and some stored the derived scalar
                # there too. Held-out-named keys in the validation block are
                # an A6 violation on every agent-facing surface, and a
                # derived `composite` scalar is not a metric — the mean
                # reducer would average it into selection.
                _metrics_block = _node.get("metrics")
                if isinstance(_metrics_block, dict):
                    for _stale in ("composite", "composite_se"):
                        if _stale in _metrics_block:
                            _metrics_block.pop(_stale)
                            _hit = True
                    for _leak in [k for k in _metrics_block
                                  if _is_held_out(str(k))]:
                        _metrics_block.pop(_leak)
                        _hit = True
                for _entry in _node.get("fold_primary_values") or []:
                    if isinstance(_entry, dict) and "composite" in _entry:
                        _entry.setdefault("primary_value", _entry.pop("composite"))
                        _hit = True
                _meta_block = _node.get("metadata")
                # The campaign's discovery baseline root stores its fold
                # vector under metadata.validation_folds (it never passes
                # through the terminal writer) — node_fold_primary_values
                # reads that form for exactly this topology, so leaving the
                # legacy key there would silently widen every child of the
                # baseline root onto the marginal-SE bar.
                if isinstance(_meta_block, dict):
                    for _entry in _meta_block.get("validation_folds") or []:
                        if isinstance(_entry, dict) and "composite" in _entry:
                            _entry.setdefault(
                                "primary_value", _entry.pop("composite")
                            )
                            _hit = True
                if isinstance(_meta_block, dict) and "composite_disagreement" in _meta_block:
                    _meta_block.setdefault(
                        "primary_value_disagreement",
                        _meta_block.pop("composite_disagreement"),
                    )
                    _hit = True
                _migrated += _hit
            for _old, _new in _META_RENAMES:
                if _old in self._data["meta"]:
                    _value = self._data["meta"].pop(_old)
                    if not self._data["meta"].get(_new):
                        self._data["meta"][_new] = _value
                    _migrated += 1
            self._data["schema_version"] = 3
            if loaded_from_disk and _migrated > 0:
                logger.warning(
                    "graph.json at %s: pre-rename schema detected; migrated "
                    "%d node/meta record(s) from 'composite' to "
                    "'primary_value' on read. Re-save to persist.",
                    self.path,
                    _migrated,
                )

    @staticmethod
    def load(path: str | Path, technique_map: dict[str, str] | None = None) -> ExperimentGraph:
        return ExperimentGraph(path=path, technique_map=technique_map)

    @property
    def meta(self) -> dict:
        return self._data["meta"]

    @property
    def nodes(self) -> dict:
        return self._data["nodes"]

    @property
    def technique_stats_data(self) -> dict:
        return self._data["technique_stats"]

    # --- ID generation ---
    def next_id(self) -> str:
        nid = self.meta["next_id"]
        self.meta["next_id"] = nid + 1
        return f"node_{nid:04d}"

    # --- Reading ---
    def get_node(self, node_id: str) -> dict | None:
        return self.nodes.get(node_id)

    def best_node(self) -> dict | None:
        best_id = self.meta.get("best_node_id")
        node = self.nodes.get(best_id) if best_id else None
        # D-01: partial results are quarantined — excluded from best_node
        if node and node.get("status") == "partial":
            return None
        return node

    def children(self, node_id: str) -> list[dict]:
        return [n for n in self.nodes.values() if n.get("parent_id") == node_id]

    def lineage(self, node_id: str) -> list[dict]:
        path = []
        current = node_id
        visited: set[str] = set()  # M-4 (audit 2026-07-23): guard a parent_id cycle
        while current:
            if current in visited:
                logger.warning("lineage: parent_id cycle detected at %s; truncating", current)
                break
            visited.add(current)
            node = self.get_node(current)
            if node is None:
                break
            path.append(node)
            current = node.get("parent_id")
        path.reverse()
        return path

    def technique_stats(self, technique: str) -> dict:
        return self.technique_stats_data.get(technique, {
            "times_tried": 0, "best_parent_delta": 0.0, "avg_parent_delta": 0.0,
        })

    # --- Budget-cell membership (CELL-1) ---
    def nodes_in_cell(self, cell_id: str) -> list[dict]:
        """Return the nodes belonging to budget cell ``cell_id``, ordered by id.

        The join between the experiment tree (``graph.json``) and the budget
        cells (``automil/cells/<cell_id>.json``). Legacy nodes carry no cell
        identity and never match — including for a falsy ``cell_id`` query,
        which must not sweep every untagged node in.
        """
        if not cell_id:
            return []
        return [
            node for _, node in sorted(self.nodes.items())
            if node_cell_id(node) == cell_id
        ]

    def count_in_cell(self, cell_id: str, *, executed_only: bool = False) -> int:
        """Count the nodes in a budget cell.

        ``executed_only=True`` counts evaluations (proposals are not evaluations),
        which is the graph-side cross-check for ``Cell.consumed_evals``. The cell
        counter remains authoritative — it also bills nodes the graph never got
        (e.g. a spec launched by a non-CLI submission path).
        """
        nodes = self.nodes_in_cell(cell_id)
        if executed_only:
            nodes = [n for n in nodes if n.get("type") == "executed"]
        return len(nodes)

    # --- Writing ---
    def _auto_extract_if_empty(self, description: str, techniques: list[str]) -> list[str]:
        """If techniques is empty and a consumer technique_map is configured,
        auto-extract tags from the description.

        Backward-compatible: when the technique_map is empty (framework default)
        OR techniques is non-empty (explicit caller input), this is a no-op.
        Consumers opt in by populating ``scoring.technique_map`` in
        ``automil/config.yaml`` and threading it through via
        ``cli/_helpers._load_technique_map``.
        """
        if techniques:
            return techniques
        if not self._technique_map:
            return techniques
        return self._extract_techniques(description)

    def add_executed(self, parent_id: str | None, description: str,
                     techniques: list[str], metrics: dict,
                     status: str = "discard", commit: str | None = None,
                     config_hash: str | None = None,
                     bootstrapped: bool = False) -> str:
        nid = self.next_id()
        parent = self.get_node(parent_id) if parent_id else None
        parent_primary_value = parent.get("primary_value", 0.0) if parent else 0.0
        primary_value = metrics.get("primary_value", 0.0)
        # CR-4: the cross-fold SE is a framework-owned scalar like `primary_value`, so
        # it is lifted to the top level rather than left inside the opaque consumer
        # metrics dict — where CR-1b's mean-of-metrics reducer would average it in.
        primary_se = node_primary_se({"primary_se": metrics.get("primary_se")})
        techniques = self._auto_extract_if_empty(description, techniques)

        node = {
            "id": nid,
            "parent_id": parent_id,
            "type": "executed",
            "status": status,
            "description": description,
            "techniques": techniques,
            # Framework-owned scalars (D-200): preserved at top level.
            "primary_value": primary_value,
            "primary_se": primary_se,
            "global_delta": metrics.get("global_delta", metrics.get("delta", 0.0)),
            "parent_delta": primary_value - parent_primary_value,
            # Consumer metrics stored as opaque dict (D-200 / DEC-04).
            "metrics": {k: v for k, v in metrics.items() if k != "primary_se"},
            # Orchestrator-measured scalars (kept top-level for ergonomics; read
            # by init.py for empirical default_vram_estimate_gb).
            "vram_gb": metrics.get("vram_gb", 0.0),
            "elapsed_min": metrics.get("elapsed_min", 0.0),
            "gpu": metrics.get("gpu", -1),
            "commit": commit,
            "archive_id": nid,
            "config_hash": config_hash,
            "potential": 0.0,
            "child_count": 0,
            "created_at": datetime.now().isoformat(),
        }
        if bootstrapped:
            node["bootstrapped"] = True

        self.nodes[nid] = node
        self.meta["total_executed"] += 1

        # H-6 (audit 2026-07-23): only a keep node may become best (this path has
        # no descendant re-evaluation, so a keep-gated inline update suffices and
        # avoids an O(N) recompute per insert).
        if status == "keep" and primary_value > self.meta["best_primary_value"]:
            self.meta["best_primary_value"] = primary_value
            self.meta["best_node_id"] = nid

        if parent_id and parent_id in self.nodes:
            self.nodes[parent_id]["child_count"] = len(self.children(parent_id))

        self._update_technique_stats(techniques, primary_value - parent_primary_value)
        return nid

    def add_proposed(self, parent_id: str, description: str,
                     techniques: list[str], rationale: str = "",
                     reference: str | None = None,
                     expected_gain: str = "low", effort: str = "low",
                     tier: int = 2, kind: str = "unspecified") -> str:
        nid = self.next_id()
        techniques = self._auto_extract_if_empty(description, techniques)
        node = {
            "id": nid,
            "parent_id": parent_id,
            "type": "proposed",
            "status": "pending",
            "description": description,
            "techniques": techniques,
            "tier": tier,
            # kind classifies the experiment for the architecture-vs-HP portfolio
            # (P1.2): architecture | regularization | hp | data | ensemble |
            # unspecified. Drives `automil portfolio` so the loop stays
            # structurally exploratory, not a pure hyperparameter sweep.
            "kind": kind,
            "rationale": rationale,
            "reference": reference,
            "expected_gain": expected_gain,
            "effort": effort,
            "potential": 0.0,
            "created_at": datetime.now().isoformat(),
        }
        self.nodes[nid] = node
        self.meta["total_proposed"] += 1
        return nid

    def mark_running(self, node_id: str) -> bool:
        node = self.nodes[node_id]
        if node["type"] != "proposed" or node["status"] != "pending":
            logger.warning(
                "mark_running skipped for %s: type=%s status=%s",
                node_id, node["type"], node["status"],
            )
            return False
        node["status"] = "running"
        return True

    def promote(self, node_id: str, metrics: dict):
        node = self.nodes[node_id]
        parent = self.get_node(node.get("parent_id")) if node.get("parent_id") else None
        parent_primary_value = parent.get("primary_value", 0.0) if parent else 0.0
        primary_value = metrics.get("primary_value", 0.0)
        status = metrics.get("status", "discard")

        node["type"] = "executed"
        node["status"] = status
        node["primary_value"] = primary_value
        # CR-4: keep the measured noise attached when a node is promoted from a
        # reconcile artifact, or the recovered incumbent would set its children's
        # bar from the bare predeclared margin instead of its own CV spread.
        _se = node_primary_se({"primary_se": metrics.get("primary_se")})
        if _se is not None or "primary_se" not in node:
            node["primary_se"] = _se
        # Paired margin: lift the fold projection alongside the SE — same
        # framework-owned contract, same "recovered incumbent must not lose its
        # evidence" rationale. Assign-or-CLEAR: a re-ingest without usable
        # folds must not leave a previous run's vector beside a new primary_value
        # (the paired deltas would difference across runs).
        from automil.scoring import fold_primary_value_map as _fold_map
        _folds = metrics.get("fold_primary_values")
        if _fold_map(_folds) is not None:
            node["fold_primary_values"] = _folds
        else:
            node.pop("fold_primary_values", None)
        node["global_delta"] = metrics.get("global_delta", metrics.get("delta", 0.0))
        node["parent_delta"] = primary_value - parent_primary_value
        # D-200: store consumer metrics as opaque dict. `primary_se` and
        # `fold_primary_values` are framework-owned (lifted above), so they are
        # excluded here for the same reason as in add_executed: CR-1b recomputes
        # the primary_value as the mean of `metrics`, and foreign values averaged or
        # carried in would corrupt it.
        node["metrics"] = {k: v for k, v in metrics.items()
                           if k not in ("primary_se", "fold_primary_values")}
        # Orchestrator-measured scalars stay top-level.
        node["vram_gb"] = metrics.get("vram_gb", 0.0)
        node["elapsed_min"] = metrics.get("elapsed_min", 0.0)
        node["gpu"] = metrics.get("gpu", -1)
        node["commit"] = metrics.get("commit")
        node["archive_id"] = node_id
        node["config_hash"] = metrics.get("config_hash")
        node["child_count"] = 0

        self.meta["total_executed"] += 1
        self.meta["total_proposed"] = max(0, self.meta["total_proposed"] - 1)

        pid = node.get("parent_id")
        if pid and pid in self.nodes:
            self.nodes[pid]["child_count"] = len([
                n for n in self.nodes.values()
                if n.get("parent_id") == pid and n["type"] == "executed"
            ])

        self._update_technique_stats(node.get("techniques", []),
                                     primary_value - parent_primary_value)

        self._reevaluate_descendants(node_id)
        # H-6 (audit 2026-07-23): recompute best from keep nodes only, AFTER
        # _reevaluate_descendants may have flipped nodes to discard. Replaces the
        # status-agnostic inline update that could leave best on a discarded node.
        self.recompute_best()

    def _reevaluate_descendants(self, root_id: str) -> None:
        """Recompute keep/discard for executed descendants of root_id.

        Children can be promoted before their parent completes, in which case
        parent metrics default to 0 and the Pareto check spuriously yields
        'keep'. Re-run the check now that root_id has real metrics.
        """
        stack = [root_id]
        visited: set[str] = set()  # M-4 (audit 2026-07-23): guard a parent/child cycle
        while stack:
            pid = stack.pop()
            if pid in visited:
                continue
            visited.add(pid)
            parent = self.nodes.get(pid)
            if not parent or parent.get("type") != "executed":
                continue
            p_comp = parent.get("primary_value", 0)
            for child in self.nodes.values():
                if child.get("parent_id") != pid:
                    continue
                if child.get("type") != "executed":
                    continue
                if child.get("status") == "partial":
                    continue   # D-01: partial nodes are not keep/discard candidates
                if child.get("status") not in ("keep", "discard"):
                    continue
                c_comp = child.get("primary_value", 0)
                # D-200 Option B: primary_value-only dominance, gated by the Ladder
                # keep-margin (δ=0.0 → strict dominance). The primary_value is the
                # consumer-computed validation selection signal (val-firewall).
                child["status"] = keep_or_discard(self.meta, parent, child)
                child["parent_delta"] = c_comp - p_comp
                stack.append(child["id"])

    def mark_failed(self, node_id: str, status: str, error: str = "",
                    config_hash: str | None = None):
        node = self.nodes[node_id]
        node["type"] = "executed"
        node["status"] = status
        node["primary_value"] = 0.0
        node["parent_delta"] = 0.0
        node["global_delta"] = 0.0
        node["error"] = error
        node["child_count"] = 0
        node["archive_id"] = node_id
        if config_hash:
            node["config_hash"] = config_hash
        self.meta["total_executed"] += 1
        self.meta["total_proposed"] = max(0, self.meta["total_proposed"] - 1)

    def cancel(self, node_id: str):
        node = self.nodes[node_id]
        node["status"] = "cancelled"
        self.meta["total_proposed"] = max(0, self.meta["total_proposed"] - 1)

    # --- Technique stats ---
    def _update_technique_stats(self, techniques: list[str], parent_delta: float):
        for tech in techniques:
            if tech not in self.technique_stats_data:
                self.technique_stats_data[tech] = {
                    "times_tried": 0,
                    "best_parent_delta": float("-inf"),
                    "avg_parent_delta": 0.0,
                    "_total_delta": 0.0,
                }
            stats = self.technique_stats_data[tech]
            stats["times_tried"] += 1
            stats["_total_delta"] = stats.get("_total_delta", 0.0) + parent_delta
            stats["avg_parent_delta"] = stats["_total_delta"] / stats["times_tried"]
            if parent_delta > stats["best_parent_delta"]:
                stats["best_parent_delta"] = parent_delta

    # --- Scoring ---
    def recalculate_scores(self):
        total = max(1, self.meta["total_executed"])
        w_e = self.meta["scoring"]["exploration_weight"]
        w_n = self.meta["scoring"]["novelty_weight"]

        for node in self.nodes.values():
            if node["type"] == "executed":
                child_count = len([
                    n for n in self.nodes.values()
                    if n.get("parent_id") == node["id"] and n["type"] == "executed"
                ])
                node["child_count"] = child_count
                node["potential"] = round(
                    node.get("primary_value", 0) +
                    w_e * math.sqrt(math.log(total) / (1 + child_count)),
                    6,
                )
            elif node["type"] == "proposed" and node["status"] != "cancelled":
                parent = self.get_node(node.get("parent_id"))
                parent_primary_value = parent.get("primary_value", 0.0) if parent else 0.0
                siblings_tried = len([
                    n for n in self.nodes.values()
                    if n.get("parent_id") == node.get("parent_id")
                    and n["type"] == "executed"
                ])
                tech_novelty = 0.0
                for tech in node.get("techniques", []):
                    stats = self.technique_stats_data.get(tech, {})
                    tech_novelty += 1.0 / (1 + stats.get("times_tried", 0))
                if node.get("techniques"):
                    tech_novelty /= len(node["techniques"])

                node["potential"] = round(
                    parent_primary_value +
                    w_e * math.sqrt(math.log(total) / (1 + siblings_tried)) +
                    w_n * tech_novelty,
                    6,
                )

    def recompute_best(self) -> tuple[str | None, float, str | None, float]:
        """Walk executed/keep nodes; pick max-primary_value node as best (CLI-07 / D-10..D-12).

        Returns ``(old_node_id, old_primary_value, new_node_id, new_primary_value)``.
        Mutates ``self._data["meta"]`` in place. The caller decides whether to
        call ``self.save()`` — recompute_best does NOT persist (so the CLI
        ``--dry-run`` flag can skip save).

        Walk semantics (D-10, B5): only nodes where ``type == "executed"`` AND
        ``status`` is in the keep-class (``keep`` / ``candidate`` /
        ``registered`` — nomination and gate passage must not evict a node from
        best). Discarded / crashed / cancelled / budget-killed / proposed nodes
        are excluded.

        Primary-value rule (D-11): uses the existing per-node ``primary_value`` field
        as already populated by train.py → result.json → orchestrator pipeline.
        Phase 0 does NOT redefine the formula — that's Phase 8 / DEC-04.

        Tie-break (D-12): equal primary values resolve to lexicographic min on
        ``node_id``. Stable and deterministic.
        """
        old_id = self.meta.get("best_node_id")
        old_c = float(self.meta.get("best_primary_value", 0.0))

        keep_nodes: list[tuple[str, float]] = []
        for node_id, node in self.nodes.items():
            if node.get("type") == "executed" and node.get("status") in KEEP_CLASS:
                keep_nodes.append((node_id, float(node.get("primary_value", 0.0))))

        if not keep_nodes:
            new_id: str | None = None
            new_c = 0.0
        else:
            # Sort: primary_value DESC, node_id ASC (lex tie-break — D-12).
            keep_nodes.sort(key=lambda x: (-x[1], x[0]))
            new_id, new_c = keep_nodes[0]

        self.meta["best_node_id"] = new_id
        self.meta["best_primary_value"] = new_c
        return old_id, old_c, new_id, new_c

    def rank_proposals(self, n: int = 6, max_per_branch: int = 2) -> list[dict]:
        proposals = [
            nd for nd in self.nodes.values()
            if nd["type"] == "proposed" and nd["status"] == "pending"
        ]
        proposals.sort(key=lambda x: x.get("potential", 0), reverse=True)

        result = []
        branch_counts: dict[str, int] = {}
        for p in proposals:
            pid = p.get("parent_id", "")
            if branch_counts.get(pid, 0) >= max_per_branch:
                continue
            result.append(p)
            branch_counts[pid] = branch_counts.get(pid, 0) + 1
            if len(result) >= n:
                break
        return result

    # --- Gate helpers (D-144, GTE-06) ---

    def nominations_in_window(self, days: int = 30) -> list[dict]:
        """Return nodes whose history contains a 'nominated' event in the last ``days`` days.

        A node may have multiple 'nominated' events (e.g. retire+re-nominate).
        The first matching event within the window is sufficient to include the
        node in the result — each node appears at most once.

        Legacy nodes without a ``history`` key are silently skipped (D-147).
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        result = []
        for node in self.nodes.values():
            for event in node.get("history", []):
                if event.get("event") != "nominated":
                    continue
                try:
                    ts = datetime.fromisoformat(event["timestamp"])
                except (ValueError, KeyError, TypeError):
                    continue
                # Normalise naive timestamps to UTC for comparison
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts > cutoff:
                    result.append(node)
                    break
        return result

    def promotion_rate(self, days: int = 30) -> float:
        """Return promoted / nominated over a rolling window (D-144).

        Returns 0.0 when no nominations exist in the window (zero-division guard).
        Promoted nodes are those whose current status is ``'registered'``.
        """
        nominated = self.nominations_in_window(days)
        if not nominated:
            return 0.0
        promoted = [n for n in nominated if n.get("status") == "registered"]
        return len(promoted) / len(nominated)

    # --- Deduplication ---
    @staticmethod
    def compute_config_hash(content: str | dict[str, str], base_commit: str = "") -> str:
        """Hash experiment config. Single script or {path: content} dict."""
        if isinstance(content, dict):
            parts = []
            for path in sorted(content.keys()):
                file_hash = hashlib.sha256(content[path].encode()).hexdigest()
                parts.append(f"{path}:{file_hash}")
            combined = base_commit + "\n" + "\n".join(parts)
            return hashlib.sha256(combined.encode()).hexdigest()[:16]
        else:
            # Keep existing tokenizer-based hash logic for single file
            try:
                tokens = tokenize.generate_tokens(io.StringIO(content).readline)
                code_tokens = [
                    tok.string for tok in tokens
                    if tok.type not in (tokenize.COMMENT, tokenize.NL,
                                        tokenize.NEWLINE, tokenize.INDENT,
                                        tokenize.DEDENT, tokenize.ENCODING)
                ]
                normalized = " ".join(code_tokens)
            except tokenize.TokenError:
                normalized = content
            return hashlib.sha256(normalized.encode()).hexdigest()

    def has_config(self, config_hash: str) -> bool:
        return any(
            n.get("config_hash") == config_hash
            for n in self.nodes.values()
            if n.get("config_hash")
        )

    # --- Technique extraction ---
    def _extract_techniques(self, description: str) -> list[str]:
        """Extract technique tags from a description string."""
        techniques = []
        desc_lower = description.lower()
        for pattern, tag in self._technique_map.items():
            if pattern in desc_lower and tag not in techniques:
                techniques.append(tag)
        return techniques

    # --- Reconciliation ---
    def reconcile(self, queue_dir: str, running_dir: str,
                  completed_dir: str, archive_dir: str,
                  proposal_stale_hours: float = 6.0):
        queue_path = Path(queue_dir)
        running_path = Path(running_dir)
        completed_path = Path(completed_dir)
        archive_path = Path(archive_dir)

        orch_ids = set()
        # queue/ is flat (no subdirs); running/ is namespaced per D-169 (Phase 6):
        # running/local/*.json, running/slurm/*.json, running/ray/*.json.
        # Use rglob for running_path to find entries across all backend subdirs.
        for d, glob_fn in ((queue_path, "glob"), (running_path, "rglob")):
            if d.exists():
                for f in getattr(d, glob_fn)("*.json"):
                    try:
                        spec = json.loads(f.read_text())
                        orch_ids.add(spec.get("id", f.stem))
                    except (json.JSONDecodeError, Exception):
                        orch_ids.add(f.stem)

        if completed_path.exists():
            for f in completed_path.glob("*.json"):
                try:
                    completion = json.loads(f.read_text())
                except (json.JSONDecodeError, Exception):
                    continue
                node_id = completion.get("id", f.stem)
                orch_ids.add(node_id)

                node = self.get_node(node_id)
                if node and node["type"] == "executed":
                    continue

                # B6 (claims-alignment): reconcile ingest runs the same
                # sanitation as the terminal writer — key-guard first (a
                # held-out-named metrics key means crash-not-ingest; averaging
                # it would put test into selection), then the val-recomputed
                # primary_value and fold-derived SE, preferring both over the
                # reported values.
                from automil.scoring import ingest_signal as _ingest_signal
                _leaking, _comp_rec, _se_rec, _refused = _ingest_signal(
                    completion, (self.meta.get("scoring") or {}).get("formula")
                )
                if _leaking:
                    logger.error(
                        "reconcile: val-firewall violation for %s — held-out-named "
                        "metrics key(s) %s; ingesting as crash.",
                        node_id, ", ".join(_leaking),
                    )
                    completion = {
                        **completion, "status": "crash", "primary_value": 0.0,
                        "metrics": {},
                        "error": (
                            "val-firewall violation: held-out-named key(s) in "
                            f"`metrics`: {', '.join(_leaking)}"
                        ),
                    }
                elif _refused:
                    # Fail-closed (B2/B3, same contract as terminal_writer):
                    # metrics present but unable to support the declared
                    # formula — the reported primary_value must not survive.
                    logger.error(
                        "reconcile: metrics for %s cannot support the declared "
                        "scoring.formula; refusing the reported primary_value and "
                        "scoring the node 0.0.",
                        node_id,
                    )
                    completion = {**completion, "primary_value": 0.0}

                # Initialized for every status (a crash-only completion used to
                # leave primary_se unbound); the completed branch upgrades
                # both to the recomputed values.
                primary_value = completion.get("primary_value", 0.0)
                primary_se = node_primary_se(completion)   # CR-4 legacy fallback

                orch_status = completion.get("status", "")
                if orch_status in ("oom", "crash", "timeout"):
                    graph_status = orch_status
                elif orch_status == "completed":
                    if _comp_rec is not None:
                        primary_value = _comp_rec
                    if _se_rec is not None:
                        primary_se = _se_rec
                    comp_metrics = completion.get("metrics", {})
                    gm = completion.get("graph_metadata", {})
                    if not gm:
                        spec_file = archive_path / node_id / "spec.json"
                        if spec_file.exists():
                            try:
                                gm = json.loads(spec_file.read_text()).get("graph_metadata", {})
                            except Exception:
                                pass
                    parent_id_check = gm.get("parent_id")
                    # Fall back to existing node's parent if metadata is missing
                    if not parent_id_check and node:
                        parent_id_check = node.get("parent_id")
                    parent_node = self.get_node(parent_id_check) if parent_id_check else None
                    # D-200 Option B: primary_value-only dominance + Ladder margin.
                    # The completion artifact carries the full child evidence
                    # (primary_value + SE + fold projection), so the paired basis
                    # survives this recovery path too.
                    child_evidence = {
                        "primary_value": primary_value,
                        "primary_se": primary_se,
                        "fold_primary_values": completion.get("fold_primary_values"),
                        # The companion guard reads its metric here; omitting
                        # the block would make every recovered child look like
                        # one that lost the metric, i.e. fail the guard closed.
                        "metrics": completion.get("metrics", {}),
                    }
                    graph_status = keep_or_discard(self.meta, parent_node, child_evidence)
                else:
                    graph_status = "discard"

                comp_metrics = completion.get("metrics", {})
                metrics = dict(comp_metrics)  # D-200: spread consumer metrics
                # B6: store the same primary_value the keep/discard decision used
                # (val-recomputed when available), never a diverging reported one.
                metrics["primary_value"] = primary_value
                metrics["vram_gb"] = completion.get("peak_vram_mb", 0) / 1024
                metrics["elapsed_min"] = completion.get("elapsed_seconds", 0) / 60
                metrics["gpu"] = completion.get("gpu", -1)
                metrics["status"] = graph_status
                metrics["global_delta"] = primary_value - self.meta.get("best_primary_value", 0)
                metrics["primary_se"] = primary_se   # CR-4: lifted by add_executed
                # Paired margin: the fold projection travels with the recovery so
                # promote() can lift it onto the node (framework-owned, like the SE).
                metrics["fold_primary_values"] = completion.get("fold_primary_values")

                config_hash = completion.get("config_hash")
                if not config_hash:
                    spec_file = archive_path / node_id / "spec.json"
                    if spec_file.exists():
                        try:
                            spec_data = json.loads(spec_file.read_text())
                            config_hash = spec_data.get("graph_metadata", {}).get("config_hash")
                        except (json.JSONDecodeError, Exception):
                            pass
                metrics["config_hash"] = config_hash

                if node:
                    if graph_status in ("keep", "discard"):
                        self.promote(node_id, metrics)
                    else:
                        self.mark_failed(node_id, graph_status,
                                         completion.get("error", ""),
                                         config_hash=config_hash)
                else:
                    parent_id = None
                    techniques = []
                    spec_file = archive_path / node_id / "spec.json"
                    if spec_file.exists():
                        try:
                            spec = json.loads(spec_file.read_text())
                            gm = spec.get("graph_metadata", {})
                            parent_id = gm.get("parent_id")
                            techniques = gm.get("techniques", [])
                            if not config_hash:
                                config_hash = gm.get("config_hash")
                                metrics["config_hash"] = config_hash
                        except (json.JSONDecodeError, Exception):
                            pass

                    self.nodes[node_id] = {
                        "id": node_id,
                        "parent_id": parent_id,
                        "type": "executed",
                        "status": graph_status,
                        "description": completion.get("description", "recovered"),
                        "techniques": techniques,
                        "primary_value": metrics["primary_value"],
                        # CR-4 + paired margin: a node REBUILT from completed/
                        # must carry the same evidence promote() lifts, or the
                        # recovered incumbent screens its children against the
                        # wider marginal bar (or none) with no error anywhere.
                        "primary_se": primary_se,
                        "fold_primary_values": completion.get("fold_primary_values"),
                        "global_delta": metrics["global_delta"],
                        "parent_delta": 0.0,
                        # D-200: consumer metrics opaque dict.
                        "metrics": dict(comp_metrics),
                        "vram_gb": metrics["vram_gb"],
                        "elapsed_min": metrics["elapsed_min"],
                        "gpu": metrics["gpu"],
                        "commit": None,
                        "archive_id": node_id,
                        "config_hash": metrics.get("config_hash"),
                        "potential": 0.0,
                        "child_count": 0,
                        "created_at": datetime.now().isoformat(),
                        "recovered": True,
                    }
                    if parent_id and parent_id in self.nodes:
                        parent_comp = self.nodes[parent_id].get("primary_value", 0)
                        self.nodes[node_id]["parent_delta"] = metrics["primary_value"] - parent_comp
                    self.meta["total_executed"] += 1
                    # H-6 (audit 2026-07-23): only a keep node may become best
                    # (keep-gated inline update preserves the D-14 no-full-recompute
                    # contract of default reconcile while never selecting a discard).
                    if graph_status == "keep" and metrics["primary_value"] > self.meta["best_primary_value"]:
                        self.meta["best_primary_value"] = metrics["primary_value"]
                        self.meta["best_node_id"] = node_id

                    self._update_technique_stats(
                        techniques, self.nodes[node_id]["parent_delta"])

                    if node_id.startswith("node_"):
                        try:
                            recovered_num = int(node_id.split("_")[1])
                            if recovered_num >= self.meta["next_id"]:
                                self.meta["next_id"] = recovered_num + 1
                        except (ValueError, IndexError):
                            pass

        # Archive-based recovery: scan for result.json in archive dirs
        if archive_path.exists():
            for node_dir in archive_path.iterdir():
                if not node_dir.is_dir():
                    continue
                node_id_r = node_dir.name
                result_file = node_dir / "result.json"
                if node_id_r not in self.nodes and result_file.exists():
                    try:
                        result = json.loads(result_file.read_text())
                        spec_file = node_dir / "spec.json"
                        spec = json.loads(spec_file.read_text()) if spec_file.exists() else {}
                        gm = spec.get("graph_metadata", {})
                        # B6: same ingest sanitation as the terminal writer.
                        from automil.scoring import ingest_signal as _ingest_signal
                        _leaking, _comp_rec, _se_rec, _refused = _ingest_signal(
                            result, (self.meta.get("scoring") or {}).get("formula")
                        )
                        if _leaking:
                            logger.error(
                                "reconcile(archive): val-firewall violation for %s — "
                                "held-out-named metrics key(s) %s; ingesting as crash.",
                                node_id_r, ", ".join(_leaking),
                            )
                            result = {
                                **result, "status": "crash", "primary_value": 0.0,
                                "metrics": {},
                                "error": (
                                    "val-firewall violation: held-out-named key(s) "
                                    f"in `metrics`: {', '.join(_leaking)}"
                                ),
                            }
                        elif _refused:
                            # Fail-closed (B2/B3): same contract as the other
                            # ingest mouths — never keep the reported scalar.
                            logger.error(
                                "reconcile(archive): metrics for %s cannot support "
                                "the declared scoring.formula; refusing the "
                                "reported primary_value and scoring the node 0.0.",
                                node_id_r,
                            )
                            result = {**result, "primary_value": 0.0}
                        r_metrics = result.get("metrics", {})
                        primary_value = (
                            _comp_rec
                            if _comp_rec is not None and result.get("status") == "completed"
                            else result.get("primary_value", 0.0)
                        )
                        primary_se = (
                            _se_rec if _se_rec is not None
                            else node_primary_se(result)   # CR-4 legacy fallback
                        )
                        num = int(node_id_r.split("_")[1])
                        if num >= self.meta["next_id"]:
                            self.meta["next_id"] = num + 1

                        parent_id = gm.get("parent_id")
                        parent = self.get_node(parent_id) if parent_id else None
                        parent_primary_value = parent.get("primary_value", 0.0) if parent else 0.0
                        # Paired margin: the archive result carries the full
                        # validation_folds; project it once for both the margin
                        # and the recovered node below.
                        from automil.scoring import fold_primary_value_entries as _fold_entries
                        _folds_r = _fold_entries(
                            result, (self.meta.get("scoring") or {}).get("formula")
                        )
                        raw_status = result.get("status", "completed")
                        if raw_status == "completed":
                            # D-200 Option B: primary_value-only dominance + Ladder
                            # margin, decided on the full child evidence.
                            status = keep_or_discard(self.meta, parent, {
                                "primary_value": primary_value,
                                "primary_se": primary_se,
                                "fold_primary_values": _folds_r,
                                "metrics": r_metrics,   # companion-guard evidence
                            })
                        else:
                            status = raw_status

                        techniques = gm.get("techniques", [])
                        self.nodes[node_id_r] = {
                            "id": node_id_r, "parent_id": parent_id,
                            "type": "executed", "status": status,
                            "description": spec.get("description", f"recovered {node_id_r}"),
                            "techniques": techniques, "primary_value": primary_value,
                            "primary_se": primary_se,   # CR-4
                            "fold_primary_values": _folds_r,    # paired-margin evidence
                            "global_delta": primary_value - self.meta.get("best_primary_value", 0),
                            "parent_delta": primary_value - parent_primary_value,
                            # D-200: consumer metrics opaque dict.
                            "metrics": dict(r_metrics),
                            "vram_gb": result.get("peak_vram_mb", 0) / 1024,
                            "elapsed_min": result.get("elapsed_seconds", 0) / 60,
                            "gpu": -1,
                            "config_hash": gm.get("config_hash"),
                            "archive_id": node_id_r, "recovered": True,
                            "created_at": datetime.now().isoformat(),
                        }
                        self.meta["total_executed"] += 1
                        # H-6 (audit 2026-07-23): only a keep node may become best.
                        if status == "keep" and primary_value > self.meta.get("best_primary_value", 0):
                            self.meta["best_primary_value"] = primary_value
                            self.meta["best_node_id"] = node_id_r
                        parent_delta = primary_value - parent_primary_value
                        self._update_technique_stats(techniques, parent_delta)
                    except (json.JSONDecodeError, Exception):
                        continue

        for node in list(self.nodes.values()):
            if node["type"] == "proposed" and node["status"] == "running":
                if node["id"] not in orch_ids:
                    node["status"] = "pending"

        # Zombie sweep: proposed/pending nodes that have no presence in
        # orchestrator state (queue/running/completed) and no archive result,
        # and whose created_at is older than proposal_stale_hours, are
        # cancelled. This cleans up stale proposals left behind by agent
        # resubmissions and orchestrator restarts — the class of zombies
        # that accumulated as 0018/0047/0048/0049 in the ccrcc run.
        now = datetime.now()
        stale_sec = proposal_stale_hours * 3600
        archive_path_obj = Path(archive_dir)
        for node in list(self.nodes.values()):
            if node.get("type") != "proposed":
                continue
            if node.get("status") != "pending":
                continue
            if node["id"] in orch_ids:
                continue
            result_file = archive_path_obj / node["id"] / "result.json"
            if result_file.exists():
                continue
            created = node.get("created_at")
            if not created:
                continue
            try:
                age_s = (now - datetime.fromisoformat(created)).total_seconds()
            except (ValueError, TypeError):
                continue
            if age_s <= stale_sec:
                continue
            node["status"] = "cancelled"
            node["cancel_reason"] = (
                f"stale: no orchestrator state, no archive result, "
                f"age {age_s / 3600:.1f}h > {proposal_stale_hours}h"
            )
            self.meta["total_proposed"] = max(
                0, self.meta["total_proposed"] - 1
            )

        self.recalculate_scores()

    # --- Migration ---
    @staticmethod
    def import_from_tsv(tsv_path: str, strategies_path: str | None = None,
                        graph_path: str | Path = "graph.json",
                        technique_map: dict[str, str] | None = None) -> ExperimentGraph:
        """Bootstrap a graph from a TSV produced by ``_append_results_tsv``.

        Column order is read from the header row, not hardcoded — any
        columns beyond ``node_id``, ``primary_value``, ``vram_gb``,
        ``elapsed_min``, ``status``, ``description`` are mapped into the
        node's ``metrics`` dict by their header name. Any consumer's TSV
        round-trips without framework changes.

        ``technique_map`` is the optional consumer-specific shorthand
        dict for tagging techniques from the description; default empty
        (no tagging). Pass the consumer's own map to recover
        domain-shorthand behaviour.
        """
        g = ExperimentGraph(path=graph_path, technique_map=technique_map or {})

        with open(tsv_path) as f:
            lines = f.readlines()

        if not lines or len(lines) < 2:
            return g

        header_cols = lines[0].strip().split("\t")
        # First column accepted as identifier under either name: post-v1.0
        # is "node_id"; pre-v1.0 was "commit". Both round-trip.
        if header_cols and header_cols[0] in ("node_id", "commit"):
            i_node = 0
        else:
            try:
                i_node = header_cols.index("node_id")
            except ValueError:
                raise ValueError(
                    f"TSV {tsv_path} has no 'node_id' or 'commit' column "
                    "as the identifier."
                )
        try:
            i_primary_value = header_cols.index("primary_value")
            i_vram = header_cols.index("vram_gb")
            i_elapsed = header_cols.index("elapsed_min")
            i_status = header_cols.index("status")
            i_desc = header_cols.index("description")
        except ValueError as exc:
            raise ValueError(
                f"TSV {tsv_path} is missing one of the required columns "
                f"(primary_value, vram_gb, elapsed_min, status, description): "
                f"{exc}"
            )
        _RESERVED = {header_cols[i_node], "primary_value", "vram_gb",
                     "elapsed_min", "status", "description", "delta"}
        # All other columns are treated as metrics.
        metric_idx = [
            (col, idx) for idx, col in enumerate(header_cols)
            if col not in _RESERVED
        ]

        rows = lines[1:]
        current_best_id = None

        for row in rows:
            parts = row.strip().split("\t")
            if len(parts) < len(header_cols):
                continue

            commit = parts[i_node]
            try:
                primary_value = float(parts[i_primary_value])
            except ValueError:
                continue
            try:
                vram_gb = float(parts[i_vram])
                elapsed_min = float(parts[i_elapsed])
            except ValueError:
                vram_gb, elapsed_min = 0.0, 0.0
            status = parts[i_status]
            description = parts[i_desc]

            metrics: dict[str, float] = {
                "primary_value": primary_value,
                "vram_gb": vram_gb,
                "elapsed_min": elapsed_min,
                "gpu": -1,
            }
            # Carry the optional pre-v1.0 `delta` column into metrics for
            # round-trip fidelity, parsing "+0.013"-style strings.
            if "delta" in header_cols:
                try:
                    metrics["delta"] = float(parts[header_cols.index("delta")].replace("+", ""))
                except (ValueError, IndexError):
                    pass
            for col_name, idx in metric_idx:
                cell = parts[idx]
                if cell == "":
                    continue
                try:
                    metrics[col_name] = float(cell)
                except ValueError:
                    # non-numeric metric column — store raw
                    metrics[col_name] = cell  # type: ignore[assignment]

            techniques: list[str] = []
            desc_lower = description.lower()
            for pattern, tag in (g._technique_map or {}).items():
                if pattern in desc_lower and tag not in techniques:
                    techniques.append(tag)

            nid = g.add_executed(
                parent_id=current_best_id,
                description=description,
                techniques=techniques,
                metrics=metrics,
                status=status,
                commit=commit,
                bootstrapped=True,
            )

            if status == "keep":
                current_best_id = nid

        if strategies_path and os.path.exists(strategies_path):
            with open(strategies_path) as f:
                strat_data = json.loads(f.read())
            best_id = g.meta.get("best_node_id")
            for strat in strat_data.get("strategies", []):
                if strat.get("status") == "not_started" and best_id:
                    g.add_proposed(
                        parent_id=best_id,
                        description=strat.get("description", strat.get("name", "")),
                        techniques=[strat["id"]],
                        rationale=strat.get("description", ""),
                        reference=strat.get("reference"),
                        expected_gain=strat.get("expected_gain", "low"),
                        effort=strat.get("effort", "low"),
                        tier=strat.get("tier", 2),
                    )

        g.recalculate_scores()
        return g

    # --- Persistence ---
    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(self.path.parent), suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                # CR-1a / M-3 (audit 2026-07-23): allow_nan=False guarantees
                # graph.json is standards-valid JSON. A NaN/Infinity would
                # otherwise serialize as a bare token that breaks every non-Python
                # reader (viz SSE JSON.parse, jq, serde). Non-finite values are
                # rejected upstream at result ingestion (validate_result), so this
                # raises only on a genuine internal invariant violation — loudly,
                # instead of persisting silent corruption.
                json.dump(self._data, f, indent=2, allow_nan=False)
                f.write("\n")
            os.rename(tmp_path, str(self.path))
            os.utime(str(self.path))
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def to_dict(self) -> dict:
        return self._data
