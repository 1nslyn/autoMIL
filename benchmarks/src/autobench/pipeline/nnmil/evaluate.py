"""Normalize nnMIL metrics to shared benchmark format.

Methods-note on AUC formula provenance
--------------------------------------
The CLAM and nnMIL wrapper paths get their AUC from different code paths,
but the two formulas are mathematically equivalent in the common case.

- CLAM path:  ``pipeline/evaluate.py::compute_extended_metrics`` recomputes
  AUC from predictions using upstream CLAM's per-class ``roc_curve`` +
  ``nanmean`` formula (``lib/CLAM/utils/core_utils.py:514-527``).
- nnMIL path: the AUC value passed in via ``raw_metrics["{split}/auroc"]``
  is produced by nnMIL's trainer using
  ``sklearn.metrics.roc_auc_score(multi_class='ovr', average='macro')``
  (``lib/nnMIL/utilities/utils.py:130-141``). We map it through without
  recomputation.

These two formulas compute the SAME thing for binary tasks and for
multi-class tasks where every class appears in every test fold. The
inner binary AUC per class is identical; the only difference is how
missing classes are handled in a fold:

- CLAM's ``nanmean`` skips a class with zero positives and averages
  over present classes.
- ``roc_auc_score(multi_class='ovr')`` raises ``ValueError`` in the same
  case (sklearn refuses an undefined macro-mean).

Each path matches its own upstream's published behaviour. Numbers
should agree to floating-point noise unless a minority class is
literally absent from some test fold, in which case CLAM's path
degrades gracefully while sklearn's would have crashed upstream too.
"""

from __future__ import annotations

# nnMIL's evaluate(split='test') returns keys like "test_test/bacc", "test_test/auroc", etc.
# We map these to our unified metric names used by compute_confidence_intervals().

_NNMIL_TO_SHARED: dict[str, str] = {
    "acc": "accuracy",
    "bacc": "balanced_accuracy",
    "auroc": "auc_roc",
    "weighted_f1": "f1",
    "kappa": "kappa",
}

# Survival trainers return flat keys like ``test_c_index`` (no "/" separator),
# so they need their own prefix-stripping path.
_NNMIL_SURVIVAL_TO_SHARED: dict[str, str] = {
    "c_index": "c_index",
    "events": "events",
    "censored": "censored",
    "event_rate": "event_rate",
    "mean_time": "mean_time",
    "median_time": "median_time",
}


def _normalize_survival_metrics(raw_metrics: dict, split: str) -> dict[str, float]:
    """Map survival trainer keys (``{split}_c_index`` etc.) to shared names.

    The shared ``c_index`` then flows through the task-agnostic
    ``compute_confidence_intervals`` to produce ``{split}_c_index_mean`` etc.
    """
    prefix = f"{split}_"
    result: dict[str, float] = {}
    for raw_key, value in raw_metrics.items():
        suffix = raw_key[len(prefix):] if raw_key.startswith(prefix) else raw_key
        if suffix in _NNMIL_SURVIVAL_TO_SHARED:
            result[_NNMIL_SURVIVAL_TO_SHARED[suffix]] = float(value)
    # Keep a consistent key set across folds even when a fold's c-index is
    # undefined (e.g. zero comparable pairs).
    result.setdefault("c_index", float("nan"))
    return result


def normalize_nnmil_metrics(
    raw_metrics: dict, split: str = "test", task_type: str = "classification",
) -> dict[str, float]:
    """Map nnMIL metric keys to the shared benchmark schema.

    Classification: nnMIL returns keys like ``{split}_{split}/bacc`` (e.g.
    ``test_test/bacc``); we extract the suffix after the last "/" and map to
    our standard names (auc_roc, accuracy, balanced_accuracy, f1, ...).

    Survival: trainers return flat ``{split}_c_index`` etc.; routed through
    ``_normalize_survival_metrics``.

    The ``auc_roc`` value here is the OvR-macro AUC produced by nnMIL's
    trainer; see the module docstring for the provenance asymmetry vs. the
    CLAM path.
    """
    if task_type == "survival":
        return _normalize_survival_metrics(raw_metrics, split)

    result: dict[str, float] = {}

    for raw_key, value in raw_metrics.items():
        # Extract the metric suffix after the last "/"
        if "/" not in raw_key:
            continue
        suffix = raw_key.rsplit("/", 1)[1]
        if suffix in _NNMIL_TO_SHARED:
            result[_NNMIL_TO_SHARED[suffix]] = float(value)

    # nnMIL silently skips auroc when only one class is present in a fold;
    # default all expected metrics to NaN so downstream CI code sees a
    # consistent key set across folds.
    result.setdefault("auc_roc", float("nan"))
    result.setdefault("accuracy", float("nan"))
    result.setdefault("balanced_accuracy", float("nan"))
    result.setdefault("f1", float("nan"))
    result.setdefault("sensitivity", float("nan"))
    result.setdefault("specificity", float("nan"))

    return result
