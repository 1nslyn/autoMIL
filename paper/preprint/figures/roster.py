"""The preprint's baseline roster: which collected experiments are the grid.

``collect_results.py`` walks whole ``benchmark_dir`` trees and returns
everything it finds. That is deliberately broad -- the cohort trees on ``fir``
also hold exploratory work -- but it means a figure built straight off the
collector's output is **not** the baseline grid.

Verified on ``fir`` 2026-07-30 (``paper/preprint/GRID-CENSUS-2026-07-30.md``),
the five ``benchmark_5fold`` trees hold 195 experiments, of which **130 are the
roster**. The other 65 must be filtered out, not averaged in:

- **35 ``cox`` survival runs.** Survival is ``nllsurv``-only as of 2026-07-30.
  The ``cox`` arm is structurally incomplete anyway -- ``clam_mb`` and
  ``dtfd_mil`` are cox-ineligible in ``config.py``, so it is 7/13 per cohort by
  construction and cannot be reported as an arm.
- **30 off-roster TCGA-LUAD classification runs.** LUAD carries a second
  mutation task (``egfr``, 21) and three extra aggregators on ``kras``
  (``clam_sb``/``mil``/``trans_mil``, 9). Unfiltered, LUAD contributes 105
  experiments where the roster is 26 -- it would dominate every pooled figure.

Each cohort is pinned to exactly ONE classification task (``ROSTER_TASKS``), so
a cohort's classification rows are dropped unless they match its pinned task.
Nothing here silently repairs bad input: :func:`filter_roster` reports what it
dropped and raises if a roster cell is missing.
"""

from __future__ import annotations

import pandas as pd

#: Each cohort's single pinned classification task (EXPERIMENT_GRID.md Sec.1).
ROSTER_TASKS = {
    "tcga_luad": "kras",
    "tcga_lgg": "idh1",
    "cptac_gbm": "tp53",
    "cptac_pdac": "immune_class",
    "tcga_hnsc": "grade",
}

#: The four MIL aggregators plus the TITAN slide-encoder arm.
ROSTER_MODELS = ("clam_mb", "simple_mil", "abmil", "dtfd_mil", "titan")

#: The only survival loss the preprint reports (2026-07-30 decision).
ROSTER_SURVIVAL_LOSS = "nllsurv"

#: 13 classification + 13 survival per cohort.
CELLS_PER_COHORT = 26


class RosterError(RuntimeError):
    """The input does not contain the expected roster."""


def _roster_mask(df: pd.DataFrame) -> pd.Series:
    """Rows that belong to the baseline roster.

    A row is kept iff its cohort is in the roster AND its aggregator is a
    roster aggregator AND either:
      - it is a classification row whose task is that cohort's pinned task, or
      - it is a survival row whose ``survival_loss`` is ``nllsurv``.

    ``task_type`` is the collector's derived column (presence of ``c_index``),
    not a passthrough -- see ``collect._task_type``.
    """
    cohort_ok = df["dataset"].isin(ROSTER_TASKS)
    model_ok = df["model_type"].isin(ROSTER_MODELS)
    pinned = df["dataset"].map(ROSTER_TASKS)
    is_survival = df["task_type"] == "survival"
    cls_ok = ~is_survival & (df["task"] == pinned)
    srv_ok = is_survival & (df["survival_loss"] == ROSTER_SURVIVAL_LOSS)
    return cohort_ok & model_ok & (cls_ok | srv_ok)


def filter_roster(
    results_df: pd.DataFrame,
    per_fold_df: pd.DataFrame | None = None,
    *,
    strict: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    """Reduce collector output to the 130 baseline roster cells.

    Returns ``(results, per_fold, report)``. ``report`` records what was kept
    and dropped so a caller can print it -- silent filtering is how a "complete
    grid" claim gets made about partial data.

    With ``strict=True`` (default) a cohort that does not have exactly
    :data:`CELLS_PER_COHORT` roster cells raises :class:`RosterError`. Pass
    ``strict=False`` to plot an incomplete grid deliberately.
    """
    required = {"dataset", "task", "task_type", "model_type", "survival_loss"}
    missing = required - set(results_df.columns)
    if missing:
        raise RosterError(
            f"filter_roster: --results is missing column(s) {sorted(missing)}; "
            f"have {sorted(results_df.columns)}"
        )

    mask = _roster_mask(results_df)
    kept = results_df[mask].copy()

    dropped = results_df[~mask]
    report = {
        "n_in": int(len(results_df)),
        "n_kept": int(len(kept)),
        "n_dropped": int(len(dropped)),
        "dropped_cox": int(
            ((dropped["task_type"] == "survival")
             & (dropped["survival_loss"] != ROSTER_SURVIVAL_LOSS)).sum()
        ),
        "dropped_off_roster_task": int(
            ((dropped["task_type"] != "survival")
             & (dropped["task"] != dropped["dataset"].map(ROSTER_TASKS))).sum()
        ),
        "dropped_off_roster_model": int((~dropped["model_type"].isin(ROSTER_MODELS)).sum()),
        "per_cohort": {
            str(c): int(n) for c, n in kept.groupby("dataset").size().items()
        },
    }

    if strict:
        bad = {c: n for c, n in report["per_cohort"].items() if n != CELLS_PER_COHORT}
        absent = sorted(set(ROSTER_TASKS) - set(report["per_cohort"]))
        if bad or absent:
            raise RosterError(
                "filter_roster: roster is incomplete -- expected "
                f"{CELLS_PER_COHORT} cells per cohort. Wrong counts: {bad or '{}'}; "
                f"cohorts absent entirely: {absent or '[]'}. "
                "Pass strict=False to plot anyway."
            )

    kept_per_fold = None
    if per_fold_df is not None:
        pf_missing = required - set(per_fold_df.columns)
        if pf_missing - {"task_type"}:
            raise RosterError(
                f"filter_roster: --per-fold is missing column(s) "
                f"{sorted(pf_missing - {'task_type'})}"
            )
        # per_fold_frame has no task_type column; derive it the same way the
        # collector does (survival rows are exactly the c_index rows) so the
        # two frames are filtered by identical logic rather than two rules.
        pf = per_fold_df.copy()
        if "task_type" not in pf.columns:
            pf["task_type"] = pf["metric"].map(
                lambda m: "survival" if m == "c_index" else "classification"
            )
        kept_per_fold = pf[_roster_mask(pf)].copy()

    return kept, kept_per_fold, report


def format_report(report: dict) -> str:
    """One-line-per-fact summary of :func:`filter_roster`'s report."""
    lines = [
        f"roster filter: kept {report['n_kept']} of {report['n_in']} experiments "
        f"({report['n_dropped']} dropped)",
        "  reasons below are diagnostic and OVERLAP (one row can hit several); "
        "they do not sum to the dropped total:",
        f"    cox survival runs:      {report['dropped_cox']}",
        f"    off-roster tasks:       {report['dropped_off_roster_task']}",
        f"    off-roster aggregators: {report['dropped_off_roster_model']}",
    ]
    lines += [f"  {c}: {n} cells" for c, n in sorted(report["per_cohort"].items())]
    return "\n".join(lines)
