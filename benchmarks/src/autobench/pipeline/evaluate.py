"""Extended evaluation metrics and cross-fold confidence intervals."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc as sk_auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def compute_extended_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> dict[str, float]:
    """Compute comprehensive classification metrics for one evaluation split."""
    metrics: dict[str, float] = {}

    # AUC-ROC
    if n_classes == 2:
        try:
            metrics["auc_roc"] = float(roc_auc_score(y_true, y_probs[:, 1]))
        except ValueError:
            metrics["auc_roc"] = float("nan")
    else:
        # Match CLAM upstream's per-class roc_curve + nanmean
        # (lib/CLAM/utils/core_utils.py:514-527)
        try:
            binary_labels = label_binarize(y_true, classes=list(range(n_classes)))
            aucs: list[float] = []
            present = set(np.unique(y_true).tolist())
            for class_idx in range(n_classes):
                if class_idx in present:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], y_probs[:, class_idx])
                    aucs.append(float(sk_auc(fpr, tpr)))
                else:
                    aucs.append(float("nan"))
            metrics["auc_roc"] = float(np.nanmean(aucs))
        except ValueError:
            metrics["auc_roc"] = float("nan")

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    if n_classes == 2:
        metrics["f1"] = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    else:
        metrics["f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    if n_classes == 2:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    else:
        metrics["sensitivity"] = float("nan")
        metrics["specificity"] = float("nan")

    return metrics


def pooled_c_index(fold_records: list[dict]) -> float:
    """Concordance over the POOLED cross-fold validation set (CR-3).

    The per-fold val c-index is computed on ~2 events for the small cohorts —
    the survival trainers themselves refuse to select checkpoints on it
    ("near-random", see clam/survival_train.py). Averaging five such near-random
    numbers does not fix that: the mean of noise is still noise, and autoMIL's
    agentic search was selecting recipes on it.

    Pooling instead concatenates every fold's validation risk scores and scores
    concordance ONCE over all comparable pairs (~5x the events). It stays a
    concordance index — comparable across recipes and across cox/nllsurv losses,
    unlike a raw validation loss.

    Each record is ``{"risks", "statuses", "times", "patient_ids"}``. Returns NaN
    when no usable record survives.
    """
    # Deferred imports: keeps this module importable without a torch env, and
    # the vendored nnMIL tree needs LIB_ROOT on sys.path (same setup the survival
    # trainers use).
    import sys

    import torch

    from autobench import LIB_ROOT
    if str(LIB_ROOT) not in sys.path:
        sys.path.insert(0, str(LIB_ROOT))
    from nnMIL.training.losses.survival_loss import survival_c_index

    risks: list[float] = []
    statuses: list[float] = []
    times: list[float] = []
    pids: list = []
    for rec in fold_records or []:
        if not isinstance(rec, dict) or not rec.get("risks"):
            continue
        risks.extend(rec["risks"])
        statuses.extend(rec["statuses"])
        times.extend(rec["times"])
        pids.extend(rec["patient_ids"])

    if not risks:
        return float("nan")

    ci = survival_c_index(
        torch.tensor(risks, dtype=torch.float32),
        torch.tensor(statuses, dtype=torch.float32),
        torch.tensor(times, dtype=torch.float32),
        pids,
    )
    return float(ci) if ci is not None else float("nan")


def pooled_val_block(fold_results: list[dict]) -> dict:
    """Summary ``val_pooled`` block: pooled val concordance, or {} (CR-3).

    Returns ``{}`` for classification experiments (no ``val_records``), so every
    runner can call this unconditionally.
    """
    records = [
        fr.get("val_records") for fr in (fold_results or []) if isinstance(fr, dict)
    ]
    if not any(records):
        return {}
    return {"c_index": pooled_c_index(records)}


#: Interval methods ``compute_confidence_intervals`` accepts. Reported verbatim
#: in each metric block's ``method`` key so a figure can never mix them silently.
CI_METHODS = ("t", "bootstrap")

#: Label stamped on blocks with <2 valid folds, where no interval is estimable
#: and a zero-width point estimate is emitted instead. Never plot these as if
#: they were intervals — filter on ``method`` first.
CI_METHOD_DEGENERATE = "degenerate"


def _t_interval(valid: np.ndarray, confidence: float) -> tuple[float, float]:
    """Student-t interval on n-1 df: ``mean +/- t_(1-a/2, n-1) * s / sqrt(n)``."""
    from scipy.stats import t as student_t

    n = len(valid)
    crit = float(student_t.ppf(1 - (1 - confidence) / 2, df=n - 1))
    half_width = crit * float(np.std(valid, ddof=1)) / np.sqrt(n)
    mean_val = float(np.mean(valid))
    return (mean_val - half_width, mean_val + half_width)


def _percentile_bootstrap_interval(
    valid: np.ndarray,
    confidence: float,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Percentile bootstrap of the fold mean. Retained for continuity only."""
    alpha = 1 - confidence
    boot_means = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        boot_means[i] = rng.choice(valid, size=len(valid), replace=True).mean()
    return (
        float(np.percentile(boot_means, 100 * (alpha / 2))),
        float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
    )


def compute_confidence_intervals(
    fold_metrics: list[dict[str, float]],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 42,
    *,
    method: str = "t",
) -> dict[str, dict[str, float]]:
    """Cross-fold interval on per-fold scalar metrics. Student-t by default (H-5a).

    Why t and not the bootstrap (H-5a, audit 2026-07-23)
    ----------------------------------------------------
    This function summarises K = 5 fold-level scalars. A percentile bootstrap at
    that K is not a conservative choice, it is a broken one:

    - It draws from at most ``C(2K-1, K) = 126`` distinct resample multisets, so
      the interval endpoints live on a coarse lattice.
    - Every replicate is an average of the observed folds, so **no percentile of
      the bootstrap distribution can fall outside ``[min, max]`` of the five
      folds**. The interval structurally asserts near-zero probability just
      outside the observed range — which at K = 5 is exactly where the truth
      plausibly sits. Percentile intervals are also the worst-behaved bootstrap
      variant at tiny n (no bias/skew correction).

    The Student-t interval makes a distributional assumption (approximately
    normal fold means) that is *explicit, stated, and checkable by a reader*.
    The percentile bootstrap makes an assumption too — that the empirical
    5-point distribution approximates the sampling distribution — which is
    simply false at this n, and it hides that assumption behind the word
    "nonparametric". Preferring the visible assumption is the defensible call.

    Caveats, deliberately not papered over:

    - The folds are **not independent** (overlapping training sets), so the t
      interval understates uncertainty. It is a floor on the error bar, not a
      calibrated one. Any claim resting on a single cell needs a paired
      across-cell test (see ``autobench.stats.multiple_comparisons``), not this.
    - The interval is **not clipped to [0, 1]**. An AUC interval that escapes
      the unit interval is honest evidence that K = 5 cannot resolve the
      quantity; clipping would disguise that. Clip at plot time if you must.

    Emitted per metric (existing keys unchanged; the rest are additive):

    ``mean``, ``std`` (ddof=1)
        Unchanged by the method switch. ``mean`` is what
        ``run_experiment.py::summary_to_result_json`` reads, so no selection
        signal, composite, or keep/discard decision moves because of H-5a.
    ``ci_low``, ``ci_high``
        The interval. Widens ~1.6x relative to the old percentile bootstrap.
    ``method``
        ``"t"``, ``"bootstrap"``, or ``"degenerate"`` — which procedure produced
        *this* metric's interval. Per metric, not per block: one metric can be
        degenerate while another in the same run is fine.
    ``n_valid_folds``, ``n_folds`` (M-15)
        Non-finite fold values are excluded from the mean. Without the
        denominator, a survival run whose val c-index was NaN in 3 of 5 folds
        reports a 2-fold average indistinguishable from a clean 5-fold one.
        ``n_valid_folds`` uses the same finiteness predicate as the H-8 counter
        in ``run_experiment.py`` so the two never disagree. Interacts with
        CR-3's :func:`pooled_val_block`, which exists precisely because the
        per-fold val c-index sits on ~2 events.

    Args:
        fold_metrics: One flat ``{metric: value}`` dict per fold.
        confidence: Two-sided coverage; 0.95 gives t(0.975, K-1).
        n_bootstrap: Replicates, ``method="bootstrap"`` only.
        seed: Bootstrap RNG seed, ``method="bootstrap"`` only.
        method: ``"t"`` (default) or ``"bootstrap"``. Keyword-only, so no
            existing positional call site can rebind it by accident.

    Raises:
        ValueError: unknown ``method``, or ``confidence`` outside (0, 1).

    Note: neither method is upstream nnMIL's bootstrap
    (``lib/nnMIL/utilities/utils.py:180``), which resamples per-sample
    predictions on a fixed test set. We summarise K-fold CV, not one test set.
    """
    if method not in CI_METHODS:
        raise ValueError(
            f"unknown CI method {method!r}; expected one of {CI_METHODS}. "
            "'t' is the K-fold default (H-5a); 'bootstrap' is retained only "
            "for continuity with pre-2026-07 summaries."
        )
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1); got {confidence!r}")

    # Union of keys across all folds — a key absent in one fold gets NaN there.
    metric_names = list(dict.fromkeys(k for fm in fold_metrics for k in fm))
    n_folds = len(fold_metrics)
    rng = np.random.default_rng(seed)

    results: dict[str, dict[str, float]] = {}
    for name in metric_names:
        values = np.array([fm.get(name, float("nan")) for fm in fold_metrics], dtype=float)
        # isfinite, not ~isnan: an inf fold would otherwise poison the mean
        # while still counting toward the denominator (M-15).
        valid = values[np.isfinite(values)]
        counts = {"n_valid_folds": len(valid), "n_folds": n_folds}

        if len(valid) < 2:
            # No interval is estimable. Emit the point estimate and say so,
            # rather than passing a zero-width interval off as a real one.
            mean_val = float(valid[0]) if len(valid) else float("nan")
            results[name] = {
                "mean": mean_val,
                "std": 0.0,
                "ci_low": mean_val,
                "ci_high": mean_val,
                "method": CI_METHOD_DEGENERATE,
                **counts,
            }
            continue

        if method == "t":
            ci_low, ci_high = _t_interval(valid, confidence)
        else:
            ci_low, ci_high = _percentile_bootstrap_interval(
                valid, confidence, n_bootstrap, rng,
            )

        results[name] = {
            "mean": float(np.mean(valid)),
            "std": float(np.std(valid, ddof=1)),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "method": method,
            **counts,
        }

    return results
