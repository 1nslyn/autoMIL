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

import logging
from typing import Callable

from autobench.pipeline.evaluate import sensitivity_specificity

logger = logging.getLogger(__name__)

#: Marker attribute so a second install is a no-op rather than a double wrap.
_WRAPPED_FLAG = "_autobench_sensitivity_specificity"


def with_sensitivity_specificity(get_eval_metrics: Callable) -> Callable:
    """Wrap nnMIL's ``get_eval_metrics`` so its result also carries the two metrics.

    Returns a callable with the same signature and the same return type. The
    wrapped function's own output is never mutated -- a new dict is built -- so a
    caller holding a reference to the original mapping is unaffected.
    """

    def _wrapper(*args, **kwargs):
        metrics = get_eval_metrics(*args, **kwargs)
        try:
            targets = kwargs.get("targets_all", args[0] if args else None)
            preds = kwargs.get("preds_all", args[1] if len(args) > 1 else None)
            unique_classes = kwargs.get("unique_classes")
            prefix = kwargs.get("prefix", "")
            if targets is None or preds is None or not unique_classes:
                return metrics
            # Returns DIFFERENT keys per task shape -- sensitivity/specificity
            # for binary, macro_recall/macro_specificity_ovr for multi-class --
            # because the two are not on the same scale. Pass whichever came
            # back straight through rather than naming them here, so this add-on
            # never has to know which shape it is in and cannot drift from the
            # shared definition.
            computed = sensitivity_specificity(targets, preds, len(unique_classes))
        except Exception as exc:      # diagnostics must not break evaluation
            logger.warning(
                "sensitivity/specificity add-on skipped for this split (%s); "
                "nnMIL's own metrics are unaffected", exc,
            )
            return metrics
        return {
            **metrics,
            **{f"{prefix}/{name}": value for name, value in computed.items()},
        }

    setattr(_wrapper, _WRAPPED_FLAG, True)
    return _wrapper


def install_sensitivity_specificity() -> bool:
    """Rebind the trainer module's ``get_eval_metrics`` to the wrapped version.

    Idempotent — a second call is a no-op. Returns True when the binding is in
    place (whether this call or an earlier one put it there), False when the
    vendored trainer could not be imported, which leaves the arm on its previous
    NaN behaviour rather than failing the run.
    """
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
        return True

    classification_trainer.get_eval_metrics = with_sensitivity_specificity(current)
    logger.info("sensitivity/specificity add-on installed for the nnMIL arm")
    return True
