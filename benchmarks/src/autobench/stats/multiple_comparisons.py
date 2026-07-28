"""Holm-Bonferroni (FWER) and Benjamini-Hochberg (FDR) p-value adjustment (H-5b).

Pure python. No scipy, no numpy — both procedures are a sort plus a running
extremum, and keeping them dependency-free means the correction can be applied
anywhere a p-value is produced.

Deliberately NOT placed in ``src/automil/``: the framework carries a purity gate
(``tests/test_framework_purity.py``) and confines its only scipy import to
``gate/stats.py``. Choosing a comparison family is a *consumer* decision about
one paper's claims, not framework behaviour.


What the comparison family is — a scientific choice, stated not smuggled
=======================================================================
An adjusted p-value is meaningless without saying which set of tests it was
adjusted over. That set is chosen, not derived, so the options are laid out
here and the recommendation is argued rather than silently applied.

The headline is a **per-cell lift**: for each cell (dataset, task, encoder,
aggregator — ``automil.cells.state.make_cell_id``), the difference between the
default recipe and the equal-effort searched recipe, reported on sealed test.
The candidate families:

**(A) Per-dataset, across arms.** One family per cohort (~3-12 cells each).
    Cheap to pass. But nothing about the *claim* is per-dataset: the paper
    asserts that the search works, not that it works on LUAD. Slicing the
    family by cohort shrinks every denominator with no pre-registered reason,
    which is family-shopping — the same result gets more stars purely by
    choosing a narrower frame. Use only if a per-cohort claim is actually
    being made *and* was declared before the numbers were seen.

**(B) The whole grid at once.** Every comparison anyone might draw from the
    ~165-experiment grid in a single family. Maximally conservative and
    superficially safest, but wrong in a different direction: it charges the
    headline for encoder-vs-aggregator comparisons the paper (post-2026-07-28
    scope) no longer claims. Correcting over tests you do not report is not
    rigour, it is self-sabotage.

**(C) One family per claim — RECOMMENDED.** The family is exactly the set of
    per-cell lift tests behind the headline figure, and nothing else. Purely
    descriptive benchmark comparisons (Figs 1/4) either carry no p-values at
    all or form their own separately-corrected family, labelled as such. Never
    pool the two: they answer different questions.

Recommended procedure: **Holm-Bonferroni** on family (C), at alpha = 0.05.

  1. Holm dominates Bonferroni uniformly — same FWER guarantee, never fewer
     rejections — so plain Bonferroni is never the right choice here.
  2. Holm's FWER control holds under *arbitrary* dependence. These cells are
     strongly and unquantifiably dependent: cells within a cohort share
     patients, splits, and folds; cells across aggregators share the encoder's
     features. BH controls FDR under independence or PRDS, and PRDS is not
     something we can demonstrate for this design. Holm needs no such argument.
  3. FWER is the error rate that matches the claim's shape. A per-cell lift
     table read as a set of confirmed findings makes one assertion per cell; a
     single false cell in a 12-18 cell table is a visible, citable error.
     Controlling the expected *proportion* of false ones is the wrong currency.

Report BH as a clearly-labelled secondary column when the table is offered as
a screen ("which cells look promising") rather than as confirmed findings. It
belongs in an appendix, never in the abstract's count of significant cells.

**The primary inference should need no correction at all.** The strongest form
of the headline is a single paired test across cells — the same shape as the
generalization gate's one-sided Wilcoxon on per-cell deltas
(``src/automil/gate/stats.py::paired_wilcoxon_with_bootstrap``). That is m = 1;
multiplicity does not arise. Per-cell p-values are then a descriptive
breakdown, and Holm's role is to stop that breakdown from being read as N
independent confirmations. Given CR-4 (delta = 0 winner's curse inflating the
val lift by an estimated +0.1 to +0.2), leaning on the pooled paired test
rather than on a count of individually-significant cells is also the more
defensible reading of the same data.

Related: ``src/automil/gate/stats.py::bonferroni_correct`` corrects *within one
gate decision* over ``K_effective`` held-out cells (default 2). It is a
different family, at a different level, and the two must not be combined.


Adjusted-p conventions
======================
Both functions return adjusted p-values (compare against alpha directly) rather
than a corrected threshold. Values match R's ``p.adjust(method="holm")`` and
``p.adjust(method="BH")``, including the tie and monotonicity rules:

- Holm is step-down: ``adj_(i) = max_{j<=i} min(1, (m-j+1) * p_(j))``. The
  running max forces a later, larger raw p upward when an earlier one was
  already heavily penalised.
- BH is step-up: ``adj_(i) = min_{j>=i} min(1, (m/j) * p_(j))``. The reverse
  running min pulls an earlier value *down* when a later one adjusts lower —
  e.g. p = (0.01, 0.04, 0.03) gives (0.03, 0.04, 0.04), not 0.045 for 0.03.

For BH, ``p_adjusted <= alpha`` is exactly equivalent to the textbook step-up
rejection rule ``k = max{i : p_(i) <= i*alpha/m}``; the equivalence is pinned
by a test rather than assumed.
"""
from __future__ import annotations

from collections.abc import Mapping

__all__ = [
    "CORRECTION_METHODS",
    "adjust",
    "benjamini_hochberg",
    "holm_bonferroni",
]

#: Supported family-wise / false-discovery procedures.
CORRECTION_METHODS = ("holm", "bh")

#: Result entry keys, for callers building a table
#: (``pd.DataFrame.from_dict(result, orient="index")``).
_ENTRY_KEYS = ("p_value", "p_adjusted", "reject")


def _validated(p_values: Mapping[str, float], alpha: float) -> list[tuple[str, float]]:
    """Validate at the boundary; never silently drop a comparison.

    Dropping an unusable p-value would shrink ``m`` and weaken the correction
    without saying so — the same hidden-denominator failure as M-15. Fail fast
    and name the comparison instead.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha!r}")

    pairs: list[tuple[str, float]] = []
    for comparison_id, raw in p_values.items():
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(
                f"p-value for {comparison_id!r} must be a real number; "
                f"got {raw!r} ({type(raw).__name__})"
            )
        value = float(raw)
        # NaN fails every comparison, so this also catches it.
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"p-value for {comparison_id!r} must be in [0, 1]; got {value!r}. "
                "A non-finite p-value is a bug upstream, not a comparison to drop."
            )
        pairs.append((comparison_id, value))
    return pairs


def _assemble(
    pairs: list[tuple[str, float]],
    adjusted_by_id: dict[str, float],
    alpha: float,
) -> dict[str, dict[str, float | bool]]:
    """Build the result in the caller's original key order."""
    return {
        comparison_id: {
            "p_value": value,
            "p_adjusted": adjusted_by_id[comparison_id],
            "reject": adjusted_by_id[comparison_id] <= alpha,
        }
        for comparison_id, value in pairs
    }


def holm_bonferroni(
    p_values: Mapping[str, float],
    alpha: float = 0.05,
) -> dict[str, dict[str, float | bool]]:
    """Holm-Bonferroni step-down adjusted p-values. Controls FWER at ``alpha``.

    Valid under arbitrary dependence between the tests, which is why it is the
    recommended procedure for the per-cell lift family (see module docstring).

    Args:
        p_values: ``{comparison_id: raw p-value}``. Order is preserved.
        alpha: Family-wise error rate. Must be in (0, 1).

    Returns:
        ``{comparison_id: {"p_value", "p_adjusted", "reject"}}``, in input
        order. Empty mapping in, empty mapping out.

    Raises:
        ValueError: a p-value outside [0, 1] or non-numeric (the offending
            ``comparison_id`` is named), or ``alpha`` outside (0, 1).
    """
    pairs = _validated(p_values, alpha)
    if not pairs:
        return {}

    m = len(pairs)
    # Stable sort: tied p-values keep input order, and the running max makes
    # their adjusted values identical regardless.
    ascending = sorted(pairs, key=lambda pair: pair[1])

    adjusted_by_id: dict[str, float] = {}
    running_max = 0.0
    for rank, (comparison_id, value) in enumerate(ascending, start=1):
        running_max = max(running_max, min(1.0, (m - rank + 1) * value))
        adjusted_by_id[comparison_id] = running_max

    return _assemble(pairs, adjusted_by_id, alpha)


def benjamini_hochberg(
    p_values: Mapping[str, float],
    alpha: float = 0.05,
) -> dict[str, dict[str, float | bool]]:
    """Benjamini-Hochberg step-up adjusted p-values. Controls FDR at ``alpha``.

    Assumes independence or positive regression dependence (PRDS). That
    assumption is NOT demonstrable for cells sharing cohorts, splits, and
    encoders — prefer :func:`holm_bonferroni` for confirmatory claims and use
    this as a labelled exploratory column.

    Args:
        p_values: ``{comparison_id: raw p-value}``. Order is preserved.
        alpha: False discovery rate. Must be in (0, 1).

    Returns:
        ``{comparison_id: {"p_value", "p_adjusted", "reject"}}``, in input
        order. ``reject`` is exactly the classic step-up rejection set.

    Raises:
        ValueError: a p-value outside [0, 1] or non-numeric (the offending
            ``comparison_id`` is named), or ``alpha`` outside (0, 1).
    """
    pairs = _validated(p_values, alpha)
    if not pairs:
        return {}

    m = len(pairs)
    ascending = sorted(pairs, key=lambda pair: pair[1])

    adjusted_by_id: dict[str, float] = {}
    running_min = 1.0
    # Walk from the largest p-value down; the running min enforces
    # monotonicity, pulling an earlier value down to a later, smaller one.
    for rank in range(m, 0, -1):
        comparison_id, value = ascending[rank - 1]
        running_min = min(running_min, min(1.0, (m / rank) * value))
        adjusted_by_id[comparison_id] = running_min

    return _assemble(pairs, adjusted_by_id, alpha)


def adjust(
    p_values: Mapping[str, float],
    alpha: float = 0.05,
    method: str = "holm",
) -> dict[str, dict[str, float | bool]]:
    """Dispatch to a correction procedure. Defaults to the recommended Holm.

    Args:
        p_values: ``{comparison_id: raw p-value}``. Order is preserved.
        alpha: Error rate for the chosen procedure.
        method: ``"holm"`` (FWER, default) or ``"bh"`` (FDR).

    Raises:
        ValueError: unknown ``method``, or any :func:`holm_bonferroni` error.
    """
    if method not in CORRECTION_METHODS:
        raise ValueError(
            f"unknown correction method {method!r}; expected one of "
            f"{CORRECTION_METHODS}. Plain Bonferroni is not offered: Holm "
            "controls the same FWER and never rejects less."
        )
    if method == "holm":
        return holm_bonferroni(p_values, alpha=alpha)
    return benjamini_hochberg(p_values, alpha=alpha)
