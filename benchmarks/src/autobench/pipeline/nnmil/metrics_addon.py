"""Give the nnMIL arm sensitivity/specificity, without editing nnMIL.

The gap
-------
nnMIL's ``get_eval_metrics`` (``lib/nnMIL/utilities/utils.py``) returns only
acc / bacc / kappa / nw_kappa / weighted_f1 / loss / auroc. Sensitivity and
specificity are not among them and never were, so
``normalize_nnmil_metrics`` fell through to ``setdefault(..., nan)`` on EVERY
nnMIL run -- binary included, not just multi-class. Every other arm (CLAM,
ABMIL, DTFD, TITAN) routes through ``compute_extended_metrics`` and reports
both, which left a hole in the results table on exactly one arm.

Why an add-on rather than a patch
---------------------------------
``benchmarks/lib/nnMIL/`` is vendored upstream code. Editing it makes every
future rebase a merge, and buries a benchmark-level decision inside a
dependency where no one reviewing the benchmark would look for it. So nothing
under ``lib/`` changes here; the whole mechanism lives on the autobench side and
is pinned by
``tests/test_nnmil_sensitivity_specificity.py::test_vendored_nnmil_is_not_modified``.

How it works
------------
``classification_trainer.py`` does ``from nnMIL.utilities.utils import
get_eval_metrics`` at import time, so the callable it invokes is an attribute of
the TRAINER module, not of ``utils``. Rebinding that attribute intercepts the
call. The interception is well-positioned: the trainer hands that call the very
``targets_all`` / ``preds_all`` / ``unique_classes`` a confusion matrix needs,
then discards them. We compute the two metrics from those arrays, add them under
nnMIL's own ``{prefix}/{name}`` convention, and let them ride out through the
trainer's normal metric dict into ``_NNMIL_TO_SHARED``.

The formula is not restated here -- it is imported from
``pipeline/evaluate.py::sensitivity_specificity``, the same function every other
arm uses. The arms are therefore identical by construction, a stronger guarantee
than the L-10 AUC asymmetry, where two separate formulas agree only when every
class is present in every fold.

Failure is contained: these are diagnostics, and a diagnostic must never take a
training run down. If anything raises, the original metrics are returned
unchanged and the caller degrades to the pre-existing NaN behaviour.
"""
from __future__ import annotations

import functools
import logging
import os
from typing import Callable

from autobench.pipeline.evaluate import (
    quadratic_weighted_kappa,
    sensitivity_specificity,
    write_predictions_csv,
)

logger = logging.getLogger(__name__)

#: Marker attribute so a second install is a no-op rather than a double wrap.
_WRAPPED_FLAG = "_autobench_sensitivity_specificity"

#: Per-fold context read by the wrapper AT CALL TIME, not captured at wrap time.
#: install() is idempotent, so a closure over fold_dir would freeze the first
#: fold's directory and every later fold would overwrite fold 0's predictions.
#: The binding is installed once; only this dict moves.
_CONTEXT: dict = {"ordinal": False, "fold_dir": None}


def with_sensitivity_specificity(get_eval_metrics: Callable) -> Callable:
    """Wrap nnMIL's ``get_eval_metrics`` so its result also carries the two metrics.

    Returns the same return type, and carries the wrapped function's identity via
    ``functools.wraps`` so ``inspect`` still reports nnMIL's own name and source
    rather than this shim. The wrapped function's own output is never mutated --
    a new dict is built -- so a caller holding a reference to the original
    mapping is unaffected.

    Positional fallbacks cover every argument this needs, not just the first two.
    nnMIL calls with all-keyword arguments today; if upstream ever switches to
    positional, a keyword-only lookup would silently disable the add-on and send
    the arm back to NaN with no diagnostic -- the exact hole this module exists
    to close. Indices follow ``get_eval_metrics(targets_all, preds_all,
    probs_all, unique_classes, get_report, prefix, roc_kwargs)``.
    """

    def _arg(kwargs: dict, args: tuple, name: str, index: int, default=None):
        if name in kwargs:
            return kwargs[name]
        return args[index] if len(args) > index else default

    @functools.wraps(get_eval_metrics)
    def _wrapper(*args, **kwargs):
        metrics = get_eval_metrics(*args, **kwargs)
        try:
            targets = _arg(kwargs, args, "targets_all", 0)
            preds = _arg(kwargs, args, "preds_all", 1)
            unique_classes = _arg(kwargs, args, "unique_classes", 3)
            probs = _arg(kwargs, args, "probs_all", 2)
            prefix = _arg(kwargs, args, "prefix", 5, "")
            if targets is None or preds is None or unique_classes is None \
                    or len(unique_classes) == 0:
                logger.warning(
                    "sensitivity/specificity add-on had no targets/preds/classes "
                    "to work from; nnMIL's own metrics are unaffected, but this "
                    "arm will report null for the pair",
                )
                return metrics
            # Returns DIFFERENT keys per task shape -- sensitivity/specificity
            # for binary, macro_recall/macro_specificity_ovr for multi-class --
            # because the two are not on the same scale. Pass whichever came
            # back straight through rather than naming them here, so this add-on
            # never has to know which shape it is in and cannot drift from the
            # shared definition.
            computed = sensitivity_specificity(targets, preds, len(unique_classes))
            if _CONTEXT["ordinal"]:
                computed["qwk"] = quadratic_weighted_kappa(
                    targets, preds, len(unique_classes),
                )
            # Same predictions.csv every other arm writes, so this benchmark has
            # ONE per-slide format rather than nnMIL's differently-named
            # results_<model>.csv. Written from the autobench side because
            # benchmarks/lib/nnMIL stays untouched; this call is the only place
            # outside the vendored trainer that sees the raw predictions.
            #
            # `prefix` is the split name, which is how one interception point
            # covers both val and test. Rows are positional sample_<i>: this
            # call does not receive slide ids (nnMIL's own CSV has them, but
            # only for test).
            fold_dir = _CONTEXT["fold_dir"]
            if fold_dir and probs is not None:
                name = "predictions.csv" if prefix == "test" else f"predictions_{prefix}.csv"
                write_predictions_csv(
                    os.path.join(fold_dir, name), None, targets, probs, preds,
                )
            # Inside the try, deliberately: a wrapped callable that returned a
            # non-mapping would otherwise raise HERE, past the guard, breaking
            # the containment this module promises.
            return {
                **metrics,
                **{f"{prefix}/{name}": value for name, value in computed.items()},
            }
        except Exception as exc:      # diagnostics must not break evaluation
            logger.warning(
                "sensitivity/specificity add-on skipped for this split (%s); "
                "nnMIL's own metrics are unaffected", exc,
            )
            return metrics

    setattr(_wrapper, _WRAPPED_FLAG, True)
    return _wrapper


def install_sensitivity_specificity(
    ordinal: bool = False, fold_dir: str | None = None,
) -> bool:
    """Rebind the trainer module's ``get_eval_metrics`` to the wrapped version.

    Idempotent — a second call is a no-op. Returns True when the binding is in
    place (whether this call or an earlier one put it there), False when the
    vendored trainer could not be imported, which leaves the arm on its previous
    NaN behaviour rather than failing the run.
    """
    _CONTEXT["ordinal"] = bool(ordinal)
    _CONTEXT["fold_dir"] = fold_dir
    try:
        from autobench.pipeline.nnmil import _imports  # noqa: F401  (sys.path setup)
        from nnMIL.training.trainers import classification_trainer
    except ImportError as exc:
        logger.warning(
            "sensitivity/specificity add-on not installed (%s); the nnMIL arm "
            "will report null for both", exc,
        )
        return False

    current = getattr(classification_trainer, "get_eval_metrics", None)
    if current is None:
        logger.warning(
            "nnMIL's classification_trainer has no get_eval_metrics binding; "
            "the add-on seam has moved and the nnMIL arm will report null",
        )
        return False
    if getattr(current, _WRAPPED_FLAG, False):
        return True   # context already refreshed above; binding stays

    classification_trainer.get_eval_metrics = with_sensitivity_specificity(current)
    logger.info("sensitivity/specificity add-on installed for the nnMIL arm")
    return True
