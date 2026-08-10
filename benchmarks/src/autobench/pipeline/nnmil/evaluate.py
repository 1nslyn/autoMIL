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

L-10 decision: DOCUMENT this caveat (this docstring), don't unify the two
formulas. Unifying would mean either patching nnMIL's vendored trainer
(``lib/nnMIL/utilities/utils.py`` -- out of scope for a consumer-side fix)
or restructuring this module to receive raw per-class probabilities that
the vendored trainer does not currently expose to it -- a structural change
disproportionate to a LOW-severity finding, and one that would change
already-dispatched numbers on one arm but not the others. The asymmetry is
pinned by
``tests/test_benchmark_evaluate.py::TestMultiClassAUC::test_L10_missing_class_asymmetry_is_pinned``,
which exercises both formulas on the same missing-class data so a change to
either is caught rather than silently drifting.
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
    # Not produced by nnMIL itself — supplied by pipeline/nnmil/metrics_addon.py,
    # which wraps the trainer's get_eval_metrics binding without touching lib/.
    # Absent when the add-on is not installed, and the setdefault below then
    # restores the previous NaN behaviour.
    #
    # Binary and multi-class carry DIFFERENT names because they are not on the
    # same scale (see evaluate.sensitivity_specificity): a binary specificity is
    # positive-class-only, while macro_specificity_ovr is inflated by the large
    # one-vs-rest negative set, and macro_recall is numerically identical to
    # balanced_accuracy. Both name pairs are mapped so this arm matches whichever
    # shape the task has.
    "sensitivity": "sensitivity",
    "specificity": "specificity",
    "macro_recall": "macro_recall",
    "macro_specificity_ovr": "macro_specificity_ovr",
    # Ordinal tasks only, and supplied by the same add-on using the SAME shared
    # implementation every other arm calls -- not nnMIL's own `kappa`, which is
    # also quadratic but infers its label set from the data rather than pinning
    # labels=range(n_classes), so it drifts on a fold that misses a class.
    "qwk": "qwk",
}

#: The multi-class half of that pair. Their presence means the add-on ran on a
#: multi-class task, so the binary names must NOT also be defaulted in.
_MACRO_KEYS = frozenset({"macro_recall", "macro_specificity_ovr"})

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
    # default the mapped metrics to NaN so downstream CI code sees a consistent
    # key set across folds.
    result.setdefault("auc_roc", float("nan"))
    result.setdefault("accuracy", float("nan"))
    result.setdefault("balanced_accuracy", float("nan"))
    result.setdefault("f1", float("nan"))

    # sensitivity/specificity are a different case, and the "skipped this fold"
    # framing above does not apply: nnMIL's trainer never emits them at all.
    # ``get_eval_metrics`` (lib/nnMIL/utilities/utils.py:72-78) returns only
    # acc / bacc / kappa / nw_kappa / weighted_f1 / loss / auroc, so before the
    # add-on these were NaN on EVERY nnMIL run, binary included -- not a
    # multi-class artifact, and a hole in the results table on one arm only.
    #
    # ``pipeline/nnmil/metrics_addon.py`` now supplies them by wrapping the
    # trainer's ``get_eval_metrics`` binding -- an add-on, with no edit to
    # lib/nnMIL -- and computing them with the SAME
    # ``evaluate.sensitivity_specificity`` every other arm uses. These
    # setdefaults stay as the fallback for a path where the add-on was not
    # installed (the vendored tree missing, or the seam having moved): the arm
    # degrades to null rather than failing the run.
    #
    # Skipped when the macro pair is present, i.e. a multi-class task where the
    # add-on DID run. Defaulting unconditionally emitted a NaN `sensitivity`
    # alongside a real `macro_recall`, which (a) no sibling arm emits on
    # multi-class -- closing an asymmetry on binary while opening a new one on
    # 3-class -- and (b) made a NaN there indistinguishable from "the add-on
    # failed", so the diagnostic lied about its own health.
    if not _MACRO_KEYS & result.keys():
        result.setdefault("sensitivity", float("nan"))
        result.setdefault("specificity", float("nan"))

    return result
