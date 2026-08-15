"""Extended evaluation metrics and cross-fold confidence intervals.

L-10: this module's multi-class AUC (below) is the CLAM-style per-class
``roc_curve`` + ``nanmean`` formula, used by CLAM/ABMIL/DTFD/TITAN. nnMIL
instead calls ``sklearn.roc_auc_score(multi_class="ovr", average="macro")``
inside its own vendored trainer, which RAISES when a class is absent from a
fold rather than degrading gracefully. See
``autobench/pipeline/nnmil/evaluate.py`` module docstring for the full
provenance and rationale (documented rather than unified: unifying would
require patching nnMIL's vendored trainer or exposing raw probabilities it
does not currently pass out), and
``tests/test_benchmark_evaluate.py::TestMultiClassAUC::test_L10_missing_class_asymmetry_is_pinned``
for the pinned regression test.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import t as student_t
from sklearn.metrics import (
    accuracy_score,
    auc as sk_auc,
    balanced_accuracy_score,
    cohen_kappa_score,
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
    ordinal: bool = False,
) -> dict[str, float]:
    """Compute comprehensive classification metrics for one evaluation split.

    ``ordinal`` marks a task whose classes are ORDERED. TCGA-HNSC ``grade``
    (g1<g2<g3) is the only one here; CPTAC-PDAC ``immune_class`` deliberately is
    NOT, because its upstream task definition scores it as ordinary multi-class
    subtyping (see cptac_pdac.yaml). It adds ``qwk``,
    which is the field standard for graded pathology targets and the only
    metric here that uses the ordering at all -- every other metric treats a
    g1->g3 error exactly like a g1->g2 one. Off by default so nominal
    multi-class tasks are unaffected.
    """
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

    metrics.update(sensitivity_specificity(y_true, y_pred, n_classes))
    if ordinal:
        metrics["qwk"] = quadratic_weighted_kappa(y_true, y_pred, n_classes)

    return metrics


def quadratic_weighted_kappa(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int,
) -> float:
    """Cohen's kappa with quadratic weights, for ORDERED classes.

    The only metric in this module that uses the class ordering. ``accuracy``,
    ``balanced_accuracy``, ``f1`` and the macro pair all treat a g1->g3 error
    exactly like a g1->g2 one; QWK penalises by squared distance, so confusing
    adjacent grades costs far less than confusing the extremes. It is the
    standard primary metric for graded pathology targets (ISUP/Gleason grading
    challenges report it as such, and Patho-Bench's Histologic_Grade config
    specifies weighted_kappa), which is why it is worth adding for ``grade``.

    ``labels=range(n_classes)`` is passed explicitly so a fold that happens to
    miss a class still produces a K x K matrix and stays comparable across
    folds; without it sklearn infers labels from the data and the weighting
    changes shape fold to fold.

    Returns the RAW value, which lives in [-1, 1] and is negative when
    agreement is worse than chance. Clamping is a selection-policy decision and
    belongs to the caller that builds the composite (see run_experiment.py), not
    to the measurement -- the campaign's fold-composite validator requires
    [0, 1], but a diagnostic should still be able to say "worse than chance".

    NaN when kappa is undefined (a fold where every sample, true and predicted,
    is one single class -- zero expected disagreement). That serializes to JSON
    ``null`` and is honest: it is not a zero.
    """
    try:
        value = float(cohen_kappa_score(
            y_true, y_pred, weights="quadratic", labels=list(range(n_classes)),
        ))
    except ValueError:
        return float("nan")
    return value


def sensitivity_specificity(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int,
) -> dict[str, float]:
    """The benchmark's one definition of sensitivity/specificity, by task shape.

    Returns DIFFERENTLY-NAMED keys per shape, deliberately -- the binary and
    multi-class quantities are not on the same scale and must never share a
    table column:

    ``sensitivity`` / ``specificity`` (binary only)
        The positive class alone: the clinical reading, and what every published
        binary number in this benchmark already means.
    ``macro_recall`` / ``macro_specificity_ovr`` (multi-class only)
        Macro-averaged one-vs-rest. See :func:`_macro_sensitivity_specificity`
        for why these carry their own names.

    Public so the nnMIL arm can share it. nnMIL's trainer emits neither metric,
    so that arm reports nothing while every other arm reports both; the consumer-
    side add-on that closes that gap calls THIS function rather than restating
    the formula, which makes the arms identical by construction -- a stronger
    guarantee than the AUC situation (L-10), where two separate formulas merely
    agree in the common case. Until that add-on lands, nnMIL still reports null
    for the pair (see nnmil/evaluate.py), so a multi-class cross-arm table has
    the macro pair on four arms and neither name on nnMIL.
    """
    if n_classes == 2:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        return {
            "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        }
    recall, spec = _macro_sensitivity_specificity(
        confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    )
    return {"macro_recall": recall, "macro_specificity_ovr": spec}


def _macro_sensitivity_specificity(cm: np.ndarray) -> tuple[float, float]:
    """Macro-averaged one-vs-rest recall/specificity from a K x K matrix.

    NOT named sensitivity/specificity, and that is the point. Two properties,
    both measured rather than assumed, make the multi-class quantities different
    animals from their binary namesakes:

    ``macro_recall`` IS ``balanced_accuracy``, exactly -- macro one-vs-rest
    recall is the definition of balanced accuracy. Verified against
    ``sklearn.balanced_accuracy_score`` at max abs difference 0.0 over 5000
    random draws (K in 3..5), including 2880 with a class absent from y_true.
    So it is a duplicate of a column already in every summary, and printing it
    beside ``balanced_accuracy`` as if it were independent evidence would
    mislead. It is kept only so the key set is uniform across task shapes.

    ``macro_specificity_ovr`` is close to a rescaled accuracy
    (``1 - (1-acc)/(K-1)`` exactly under class balance; corr 0.989 with accuracy
    under Dirichlet(0.7) imbalance at K=3), structurally inflated toward 1 by the
    large one-vs-rest negative set. Reported on the same axis as a binary
    specificity it would flatter every multi-class cell.

    These were ``float("nan")`` for every multi-class task until 2026-08-10. The
    NaN itself was contained -- these are diagnostics, never in ``metrics`` and
    never in ``composite`` -- but it serialized into result.json as a bare ``NaN``
    token, which the orchestrator's ingestion parser rejects outright (CR-1a).
    Every 3-class run (CPTAC-PDAC, TCGA-HNSC) was therefore recorded as a crash
    despite carrying a perfectly good validation composite. Note the SERIALIZER
    fix is what repaired that; naming these honestly is a separate concern.

    A class absent from ``y_true`` has an undefined recall, and one that consumes
    every sample has an undefined specificity. Such classes are dropped from
    their own average rather than counted as 0.0, which would report a model as
    worse than it is purely because a small fold missed a label. That matches
    sklearn's ``balanced_accuracy_score`` exactly, and it is the defensible
    choice -- but it is not free: on a ~10-patient validation split the absent
    class is usually a RARE one, i.e. the one the model handles worst, so the
    reported macro is biased upward and moves discontinuously with whether a
    single patient landed in the split. A cross-fold mean of macro-averages over
    different class subsets is therefore not a population quantity; report the
    macro from a POOLED cross-fold confusion matrix (the pattern
    :func:`pooled_c_index` already uses for CR-3) if this is ever put in a table.

    An empty split has no defined class at all, and yields NaN -- not 0.0, which
    would fabricate "the model got everything wrong" out of "there was no data",
    exactly what ``scoring.cross_fold_se``'s docstring forbids. That NaN is safe
    now: the serializer writes it as JSON ``null``.
    """
    tp = np.diag(cm).astype(float)
    fn = cm.sum(axis=1) - tp        # true class c, predicted otherwise
    fp = cm.sum(axis=0) - tp        # predicted class c, truly otherwise
    tn = cm.sum() - tp - fn - fp

    def _macro(numerator: np.ndarray, denominator: np.ndarray) -> float:
        defined = denominator > 0
        if not defined.any():
            return float("nan")     # not estimable != scored zero
        return float(np.mean(numerator[defined] / denominator[defined]))

    return _macro(tp, tp + fn), _macro(tn, tn + fp)


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


def write_predictions_csv(
    path: str,
    slide_ids: list,
    y_true: np.ndarray,
    y_probs: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """Persist one split's per-slide predictions, in CLAM's existing column set.

    ``slide_id, y_true, y_prob_0..y_prob_<K-1>, y_hat`` -- the format
    ``clam/train.py`` already writes, so every arm becomes readable by the same
    tooling. Until now only CLAM (test split) and nnMIL (its own differently
    shaped CSV) saved anything; abmil / dtfd / titan wrote nothing but
    ``metrics.json``, which meant a metric that was not computed at training
    time could never be recovered without a full retrain -- no confusion matrix,
    and not even a checkpoint to re-infer from.

    Writing these makes any future confusion-matrix-derived metric (per-class
    recall, QWK, specificity) a recomputation instead of a re-run.
    """
    import csv as _csv
    import os as _os

    _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
    y_probs = np.atleast_2d(np.asarray(y_probs))
    with open(path, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["slide_id", "y_true"]
                   + [f"y_prob_{i}" for i in range(y_probs.shape[1])] + ["y_hat"])
        for i in range(len(y_true)):
            sid = slide_ids[i] if slide_ids is not None and i < len(slide_ids) else f"sample_{i}"
            w.writerow([sid, int(y_true[i])]
                       + [float(x) for x in y_probs[i]] + [int(y_pred[i])])


def write_survival_predictions_csv(path: str, records: dict) -> None:
    """Persist one split's per-sample survival risk scores (A4').

    ``records`` is the ``val_records`` dict every survival trainer already
    materializes at fold end (CR-3): ``risks`` / ``statuses`` / ``times`` /
    ``patient_ids``, parallel lists. Written as
    ``patient_id, status, time, risk_score`` — the same column set nnMIL's own
    test-side CSV uses — so a survival fold's validation split can be re-scored
    (or byte-compared across runs) without a retrain, exactly what
    ``write_predictions_csv`` provides on the classification side.
    """
    import csv as _csv
    import os as _os

    _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["patient_id", "status", "time", "risk_score"])
        for pid, status, time_, risk in zip(
            records["patient_ids"], records["statuses"],
            records["times"], records["risks"],
        ):
            w.writerow([pid, int(status), float(time_), float(risk)])


def file_sha256(path: str) -> str | None:
    """sha256 hex of a file's bytes, or None when the file does not exist.

    The no-op detector (A4'): two runs whose recipes differ but whose selected
    models score the validation split identically are indistinguishable by
    metrics on a small split — the hash of the persisted per-fold val
    predictions tells a changed model from an unchanged one.
    """
    import hashlib
    import os as _os

    if not _os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
