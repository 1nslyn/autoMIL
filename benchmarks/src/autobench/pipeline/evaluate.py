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


def compute_confidence_intervals(
    fold_metrics: list[dict[str, float]],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Cross-fold percentile bootstrap CI on per-fold scalar metrics.

    Resamples the K fold-level metric values with replacement n_bootstrap
    times, takes the mean of each resample, and reports 2.5/97.5
    percentiles of the bootstrap distribution. This is the standard
    procedure for reporting CIs on K-fold CV summary statistics and
    captures variability across the random partition.

    Note: this is NOT the same as upstream nnMIL's bootstrap at
    ``lib/nnMIL/utilities/utils.py:180``, which resamples per-sample
    predictions on a fixed test set and returns mean+std only. We
    deliberately use a different statistical object because we report
    K-fold CV summaries, not single-test-set numbers.
    """
    # Union of keys across all folds — a key absent in one fold gets NaN there.
    metric_names = list(dict.fromkeys(k for fm in fold_metrics for k in fm))
    alpha = 1 - confidence
    lo_pct = 100 * (alpha / 2)
    hi_pct = 100 * (1 - alpha / 2)
    rng = np.random.default_rng(seed)

    results: dict[str, dict[str, float]] = {}
    for name in metric_names:
        values = np.array([fm.get(name, float("nan")) for fm in fold_metrics], dtype=float)
        valid = values[~np.isnan(values)]

        if len(valid) < 2:
            mean_val = float(np.nanmean(values))
            results[name] = {
                "mean": mean_val,
                "std": 0.0,
                "ci_low": mean_val,
                "ci_high": mean_val,
            }
            continue

        boot_means = np.empty(n_bootstrap, dtype=float)
        for i in range(n_bootstrap):
            sample = rng.choice(valid, size=len(valid), replace=True)
            boot_means[i] = sample.mean()

        results[name] = {
            "mean": float(np.mean(valid)),
            "std": float(np.std(valid, ddof=1)),
            "ci_low": float(np.percentile(boot_means, lo_pct)),
            "ci_high": float(np.percentile(boot_means, hi_pct)),
        }

    return results
