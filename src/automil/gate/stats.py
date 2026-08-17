"""Pure-function statistical core for the generalization gate (D-141 / GTE-04).

Pattern mirrors cells/cap.py — pure functions, no filesystem I/O, no clock reads,
caller injects all state. The ONLY scipy importer in src/automil/ (D-148).

Bonferroni direction: divide alpha by K (Wikipedia convention). Never multiply
p-values. See Pitfall 4 in research.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import bootstrap, wilcoxon


def paired_wilcoxon_with_bootstrap(
    deltas: np.ndarray,
    p_threshold: float,
    bootstrap_reps: int = 1000,
    rng_seed: int | None = None,
) -> tuple[bool, float, tuple[float, float], int]:
    """Paired one-sided Wilcoxon + BCa bootstrap CI on per-cell deltas.

    Args:
        deltas: 1-D ndarray of (candidate_primary_value_i - parent_primary_value_i)
            paired by held-out cell_id.
        p_threshold: Bonferroni-corrected alpha — caller pre-divides by K
            via bonferroni_correct(). Compare scipy's raw pvalue against this.
        bootstrap_reps: 1000 per GTE-04 default.
        rng_seed: deterministic if not None (test-friendly).

    Returns:
        (passes, p_value, (ci_low, ci_high), individual_wins)
        - passes: pvalue <= p_threshold AND ci_low > 0
        - individual_wins: int(np.sum(deltas > 0)) — caller compares to K_effective
    """
    if len(deltas) < 1:
        return (False, 1.0, (0.0, 0.0), 0)
    if np.all(deltas == 0):
        return (False, 1.0, (0.0, 0.0), 0)

    wres = wilcoxon(deltas, zero_method="wilcox", alternative="greater")

    rng = np.random.default_rng(rng_seed)
    bres = bootstrap(
        (deltas,),
        np.median,
        n_resamples=bootstrap_reps,
        confidence_level=0.95,
        method="BCa",
        rng=rng,
    )
    ci_low = float(bres.confidence_interval.low)
    ci_high = float(bres.confidence_interval.high)
    individual_wins = int(np.sum(deltas > 0))
    passes = bool((wres.pvalue <= p_threshold) and (ci_low > 0))
    return (passes, float(wres.pvalue), (ci_low, ci_high), individual_wins)


def bonferroni_correct(p_threshold: float, K: int) -> float:
    """Bonferroni: divide alpha by K. Pitfall 4 — never multiply p-values.

    We divide alpha by K (Wikipedia convention) rather than multiply p-values by K
    (multipletests convention) — equivalent for single decisions, less surprising in
    test logs.
    """
    if K < 1:
        raise ValueError(f"K must be >= 1; got {K}")
    return p_threshold / K


def diagnose_gate_health(
    promotion_rate_30d: float,
    threshold_low: float = 0.05,
    threshold_high: float = 0.5,
) -> str:
    """Human-readable diagnostic for `automil status` and viz dashboard (D-144 / Pitfall 6)."""
    if promotion_rate_30d < threshold_low:
        return (
            f"gate too strict OR search space too narrow "
            f"(promotion_rate_30d={promotion_rate_30d:.1%} < {threshold_low:.0%})"
        )
    if promotion_rate_30d > threshold_high:
        return (
            f"gate too loose OR pre-registration didn't capture genuinely-held-out cells "
            f"(promotion_rate_30d={promotion_rate_30d:.1%} > {threshold_high:.0%})"
        )
    return f"gate healthy (promotion_rate_30d={promotion_rate_30d:.1%})"
