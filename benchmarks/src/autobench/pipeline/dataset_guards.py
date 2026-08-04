"""Retained-fraction / per-class-floor guard for split loading (M-9).

Every per-arm dataset loader can drop slides whose feature file (or, on a stale
task CSV, whose label) is missing. This guard prevents training or evaluation
from proceeding on an unaccounted subset of the intended split.

On TRAIN that shrink is merely lossy -- fewer bags to fit on. On VAL it
corrupts the model-selection signal (keep/discard reads a metric computed on
an arbitrary subset of the intended cohort), and on TEST it corrupts the
number the paper reports -- and it was silent, so a partially-extracted
cohort still returned a confident AUC/c-index, computed on whichever slides
happened to survive.

This module is the single place that turns that shrink into a loud failure
for val/test (and an explicit, attributed warning for train, which tolerates
it deliberately -- see ``warn_only`` below).
"""
from __future__ import annotations

from typing import Mapping

__all__ = [
    "MIN_RETAINED_FRACTION",
    "MIN_CLASS_COUNT",
    "SplitRetentionError",
    "check_split_retention",
]

#: Default floor on retained/expected for a val or test split. Deliberately
#: strict: on the small cohorts this benchmark runs against (some val splits
#: are ~10 patients -- see CR-4's per-cohort accept_margin), losing even 1
#: in 10 slides is already enough to shift the selection signal by a full
#: patient's worth of evidence. A legitimately-lossy extraction should be
#: re-run, not silently absorbed by whichever slides happened to survive.
MIN_RETAINED_FRACTION = 0.9

#: A class assigned to the split (per ``splits_<fold>.csv``) that retains
#: fewer than this many slides fails independently of the overall fraction --
#: a 95%-retained split can still have silently lost its only minority-class
#: slide, which is exactly the "class-correlated shrink" the audit named.
MIN_CLASS_COUNT = 1


class SplitRetentionError(RuntimeError):
    """A val/test split lost too many (or a whole class of) slides."""


def check_split_retention(
    *,
    context: str,
    split: str,
    expected_total: int,
    retained_total: int,
    expected_by_class: Mapping[object, int] | None = None,
    retained_by_class: Mapping[object, int] | None = None,
    min_retained_fraction: float = MIN_RETAINED_FRACTION,
    min_class_count: int = MIN_CLASS_COUNT,
    warn_only: bool = False,
) -> None:
    """Assert a split retained enough of the slides assigned to it.

    Call this ONCE per split, after a loader's drop loop has finished (with
    final counts) -- not per slide. A per-slide check would either fire on
    the very first drop (before the fraction is even knowable) or spam one
    message per dropped slide instead of one loud failure per split.

    Args:
        context: human-readable identity for the error message -- cohort,
            task, fold, framework: whatever the caller has on hand. Callers
            are expected to make this specific enough to locate the data
            (e.g. a task-CSV or split-CSV path), not just a framework name.
        split: "train" | "val" | "test".
        expected_total: slides assigned to this split before any drop (i.e.
            the split CSV's column count for this split).
        retained_total: slides that actually loaded (feature file + label
            both present).
        expected_by_class / retained_by_class: optional per-class counts,
            for the per-class floor. Both keyed on the SAME label
            representation (e.g. both int-mapped, or both raw strings) --
            mismatched keys would silently skip the floor for every class.
            Omit either to skip the per-class half of the check (e.g. a
            survival split, which has no classes).
        warn_only: TRAIN passes this. A shrunk train split is lossy, not
            corrupting -- fewer bags to fit on -- so it warns instead of
            raising; VAL/TEST corrupt the selection signal or the reported
            result respectively and must never proceed silently.
    """
    if expected_total == 0:
        return  # nothing was assigned to this split; not this guard's problem

    fraction = retained_total / expected_total
    problems: list[str] = []
    if fraction < min_retained_fraction:
        problems.append(
            f"retained {retained_total}/{expected_total} slides "
            f"({fraction:.1%}, floor {min_retained_fraction:.0%})"
        )

    if expected_by_class is not None and retained_by_class is not None:
        starved = sorted(
            (str(cls), retained_by_class.get(cls, 0), n_expected)
            for cls, n_expected in expected_by_class.items()
            if n_expected > 0 and retained_by_class.get(cls, 0) < min_class_count
        )
        if starved:
            detail = ", ".join(
                f"{cls!r}: {retained}/{n_expected}"
                for cls, retained, n_expected in starved
            )
            problems.append(
                f"{len(starved)} class(es) below the per-class floor "
                f"(min {min_class_count}): {detail}"
            )

    if not problems:
        return

    message = (
        f"{context}: {split} split lost too many slides to missing "
        f"features/labels -- {'; '.join(problems)}."
    )
    if warn_only:
        print(
            f"  [WARNING] {message} Proceeding: a shrunk TRAIN split is "
            "lossy (fewer bags), not corrupting, so this is a warning, not "
            "a failure."
        )
        return
    raise SplitRetentionError(
        f"{message} A {split} metric computed on this subset would not "
        "represent the intended cohort -- on VAL this corrupts model "
        "selection, on TEST it corrupts the reported result. Verify "
        "feature extraction completed for this encoder/cohort, or "
        "explicitly lower min_retained_fraction/min_class_count if this "
        "drop is expected."
    )
