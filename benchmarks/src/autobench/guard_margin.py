"""Derive each cell's companion non-inferiority margin from its frozen splits.

Selection is single-metric (``scoring.formula: val_auc``). Balanced accuracy is
recorded but does not vote, because on a few-dozen-slide validation split it is
a LATTICE statistic: it can only take values on a grid whose spacing is set by
the class counts, and one slide changing side moves it by a whole grid step —
which on these cohorts is the size of the effects the search is looking for.

The companion guard restores balanced accuracy's ability to REJECT a candidate
without giving it a vote in the argmax. That needs a tolerated-drop margin, and
the margin has to be honest about the lattice:

    margin = 1 / (K x C x min_{f,c} n_{f,c})

- ``K``   folds averaged into the reported number (discovery: 3)
- ``C``   classes balanced accuracy averages recall over
- ``n_{f,c}``  slides of class ``c`` in fold ``f``'s VALIDATION column

That expression is exactly the largest change a single validation slide can
make to the reported number. A drop no bigger than it is arithmetically
explainable by one borderline slide changing side — the finest distinction the
metric can draw on this split — so it cannot be evidence of harm. Any larger
drop is rejected. "One slide" means one WORST-CASE slide: the margin is the
coarsest single-slide step, so on a very lopsided cohort a couple of
majority-class slides can fall inside it. That is the metric's resolution,
not slack in the guard.

Two properties make this defensible where a hand-picked constant would not be:

**It is predeclared.** A non-inferiority margin fitted to the comparison it
gates lets a noisy candidate widen its own acceptance region. This one is
computed from the frozen split definition before anything runs, and frozen into
the manifest alongside the counts it came from, so the arithmetic is checkable
by hand from published numbers.

**It is per dataset AND per task.** Both inputs vary by cell: a 3-class task
divides the step by 3 where a binary task divides by 2, and a task whose
minority class holds 11 slides has a step six times coarser than one holding
65. A single campaign-wide epsilon would be simultaneously too tight for the
small cohorts and too loose for the large ones.

Survival cells report no balanced accuracy and get no guard.
"""
from __future__ import annotations

import csv
import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

#: The companion metric guarded on every classification cell. Balanced accuracy
#: is the metric Jun's objection is about and the one whose quantization the
#: formula below models; guarding a different metric would need a different
#: derivation, so the name is fixed here rather than passed in.
GUARD_METRIC = "val_bacc"

#: Decimals the runner rounds every recorded metric to (``run_experiment.py``
#: writes ``round(value, 4)``). The guard compares RECORDED values, so the
#: margin has to live on the recording grid too: a true one-slide drop of
#: 0.0098039 is recorded as 0.0099 whenever the parent rounds up and the child
#: rounds down, and a margin left on the true lattice rejects it — the one
#: thing "a drop of exactly `margin` passes" promises cannot happen (21 of 527
#: reachable one-slide states on tcga_luad/kras). Rounding the margin UP to
#: this grid admits every one-slide drop while still rejecting anything
#: larger than one worst-case slide. ("One WORST-CASE slide", not "one slide
#: of every class": on a cohort whose majority class is more than twice its
#: minority, two majority-class slides legitimately fall inside the margin —
#: that is the metric's own resolution, which is what the guard is
#: calibrated to.)
RECORDED_DECIMALS = 4


class GuardMarginError(RuntimeError):
    """The margin could not be derived — never silently defaulted.

    A cell whose validation composition cannot be read must block the manifest
    freeze. Emitting a fallback margin would ship a guard whose tolerance has
    no relationship to the split it is guarding, which is worse than no guard
    at all: it would be quoted in the paper as if it had been derived.
    """


def _read_validation_slide_ids(split_csv: Path) -> list[str]:
    """The non-empty ``val`` column of a split CSV.

    The three columns are ragged and blank-padded to the longest (``train``),
    so the row count is not the split size — only non-empty cells count.
    """
    try:
        with split_csv.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
    except OSError as exc:
        raise GuardMarginError(f"cannot read split file {split_csv}: {exc}") from exc
    if not rows or "val" not in rows[0]:
        raise GuardMarginError(f"{split_csv} has no 'val' column")
    ids = [(r.get("val") or "").strip() for r in rows]
    return [sid for sid in ids if sid]


def _read_labels(task_csv: Path) -> dict[str, str]:
    """``slide_id -> label`` from a task CSV."""
    try:
        with task_csv.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
    except OSError as exc:
        raise GuardMarginError(f"cannot read task CSV {task_csv}: {exc}") from exc
    if not rows or not {"slide_id", "label"} <= set(rows[0]):
        raise GuardMarginError(
            f"{task_csv} must have 'slide_id' and 'label' columns "
            "(a survival task CSV has neither and takes no guard)"
        )
    return {(r["slide_id"] or "").strip(): (r["label"] or "").strip() for r in rows}


def validation_class_counts(
    benchmark_dir: Path | str, strategy: str, task: str, folds,
) -> dict[int, dict[str, int]]:
    """``fold_index -> {label: count}`` over each fold's validation split.

    Reads the split definition and the task labels off disk — the same two
    files the loaders read — so the counts are the split's own arithmetic and
    not a claim about a run.
    """
    benchmark_dir = Path(benchmark_dir)
    labels = _read_labels(benchmark_dir / "dataset_csv" / f"{task}.csv")
    counts: dict[int, dict[str, int]] = {}
    for fold in folds:
        split_csv = benchmark_dir / "splits" / strategy / task / f"splits_{fold}.csv"
        if not split_csv.exists():
            raise GuardMarginError(f"missing split file {split_csv}")
        val_ids = _read_validation_slide_ids(split_csv)
        if not val_ids:
            raise GuardMarginError(f"{split_csv} assigns no slides to validation")
        unlabelled = [sid for sid in val_ids if sid not in labels]
        if unlabelled:
            raise GuardMarginError(
                f"{split_csv}: {len(unlabelled)} validation slide(s) have no "
                f"label in the task CSV (first: {unlabelled[0]})"
            )
        counts[int(fold)] = dict(Counter(labels[sid] for sid in val_ids))
    return counts


def balanced_accuracy_margin(fold_counts: dict[int, dict[str, int]]) -> float:
    """``1 / (K x C x min n)`` — one validation slide, worst case.

    Requires every fold to carry the SAME class set. Balanced accuracy averages
    recall over the classes PRESENT in the fold, so a fold missing a class is
    averaging over a different C and the reported mean is not on the lattice
    this margin describes — that is a data defect worth surfacing, not
    something to average over.
    """
    if not fold_counts:
        raise GuardMarginError("no folds to derive a margin from")
    class_sets = {frozenset(c) for c in fold_counts.values()}
    if len(class_sets) != 1:
        raise GuardMarginError(
            f"validation folds carry different class sets ({sorted(class_sets, key=sorted)}); "
            "balanced accuracy would average over a different C per fold"
        )
    n_folds = len(fold_counts)
    n_classes = len(next(iter(class_sets)))
    if n_classes < 2:
        raise GuardMarginError(
            f"validation folds carry {n_classes} class(es); balanced accuracy "
            "is undefined below two"
        )
    smallest = min(n for counts in fold_counts.values() for n in counts.values())
    if smallest <= 0:
        raise GuardMarginError("a validation class holds no slides")
    return 1.0 / (n_folds * n_classes * smallest)


def derived_margin_for_counts(counts: Mapping[str, Mapping[str, int]]) -> float:
    """The margin a PUBLISHED ``validation_class_counts`` block implies.

    The auditability claim is that the margin is re-derivable by hand from the
    counts that travel with it. This is that derivation, in code, so the claim
    is enforced at every freeze rather than merely stated: it re-runs the same
    arithmetic ``derive_guard`` ran, from the counts alone.
    """
    if not isinstance(counts, Mapping) or not counts:
        raise GuardMarginError("no validation class counts")
    folds = {}
    for index, (fold, block) in enumerate(sorted(counts.items())):
        if not isinstance(block, Mapping) or not block:
            raise GuardMarginError(f"fold {fold!r} carries no class counts")
        parsed = {}
        for label, value in block.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise GuardMarginError(
                    f"fold {fold!r} class {label!r} count {value!r} is not an integer"
                )
            parsed[label] = value
        folds[index] = parsed
    quantum = balanced_accuracy_margin(folds)
    return math.ceil(quantum * 10 ** RECORDED_DECIMALS) / 10 ** RECORDED_DECIMALS


def verify_against_run(fold_counts: dict[int, dict[str, int]], results_dir: Path | str) -> None:
    """Assert the derived counts match what a completed run actually SCORED.

    The margin is derived from the split assignment, which is what exists at
    freeze time. A loader may retain fewer slides than were assigned (a
    missing feature file), and a smaller validation set has a COARSER lattice
    than the derived margin assumes — so the guard would be tighter than one
    true slide flip and could reject on jitter. That gap is checkable the
    moment a baseline exists: ``predictions_val.csv`` holds one row per slide
    actually scored.

    Compares the sorted count vector rather than the labels: the split CSV
    carries class names and the predictions carry encoded integers, and the
    margin depends only on ``C`` and ``min n``, both naming-invariant.
    """
    results_dir = Path(results_dir)
    for fold, counts in sorted(fold_counts.items()):
        preds = results_dir / f"fold_{fold}" / "predictions_val.csv"
        if not preds.exists():
            raise GuardMarginError(f"no scored validation predictions at {preds}")
        with preds.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        if not rows or "y_true" not in rows[0]:
            raise GuardMarginError(f"{preds} has no 'y_true' column")
        scored = sorted(Counter(r["y_true"] for r in rows).values())
        assigned = sorted(counts.values())
        if scored != assigned:
            raise GuardMarginError(
                f"fold {fold}: the run scored {sum(scored)} validation slides "
                f"{scored} but the split assigns {sum(assigned)} {assigned}; "
                "the derived margin describes a lattice the run does not have"
            )


def derive_guard(
    benchmark_dir: Path | str, strategy: str, task: str, folds,
) -> dict:
    """The frozen ``{metric, margin, basis}`` declaration for one cell.

    ``basis`` is provenance, not configuration: the framework ignores it, and
    it travels into the materialized config and then into ``graph.json`` so
    every frozen artifact records the arithmetic behind its own margin.
    """
    counts = validation_class_counts(benchmark_dir, strategy, task, folds)
    margin = balanced_accuracy_margin(counts)
    n_folds = len(counts)
    n_classes = len(next(iter(counts.values())))
    smallest_class, smallest = min(
        ((label, n) for c in counts.values() for label, n in c.items()),
        key=lambda item: item[1],
    )
    return {
        "metric": GUARD_METRIC,
        "margin": math.ceil(margin * 10 ** RECORDED_DECIMALS) / 10 ** RECORDED_DECIMALS,
        "basis": (
            f"one validation slide: 1/({n_folds} folds x {n_classes} classes x "
            f"{smallest} slides in the smallest validation class {smallest_class!r})"
            f" = {margin:.6f}, rounded up to the {RECORDED_DECIMALS}-decimal "
            "grid the metric is recorded on"
        ),
        "validation_class_counts": {
            str(fold): dict(sorted(c.items())) for fold, c in sorted(counts.items())
        },
    }
