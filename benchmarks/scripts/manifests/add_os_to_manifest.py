"""Add OS_event / OS_time columns to a GDC-derived normalized_manifest.csv.

Joins overall-survival labels from a GDC clinical export (`clinical.tsv`) onto
the manifest by its case-id column (`--case-col`, default `case_id`; TCGA
manifests key on `sample_names`). Works for any GDC program (CPTAC, TCGA, ...) —
the GDC clinical schema is shared.

Mapping (per case, from `demographic.vital_status`):
  Dead          -> OS_event=1, OS_time=days_to_death
  Alive         -> OS_event=0, OS_time=days_to_last_follow_up
  Not Reported  -> OS_event=NaN, OS_time=NaN

Cases absent from the clinical export (or with vital_status outside
{Dead, Alive}, or missing the relevant day count) get NaN labels. The survival
task CSV later drops NaN rows, so those slides are simply excluded from survival
experiments — no error.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

# GDC clinical TSV columns (shared across programs).
_CASE = "cases.submitter_id"
_VITAL = "demographic.vital_status"
_DEATH = "demographic.days_to_death"
_FOLLOWUP = "diagnoses.days_to_last_follow_up"


def build_os_table(clinical_tsv: str) -> pd.DataFrame:
    """Return a per-case table with columns ``case_id, OS_event, OS_time``."""
    clin = pd.read_csv(clinical_tsv, sep="\t", low_memory=False)
    clin = clin.replace(["'--", "--"], np.nan)  # GDC missing-value sentinels

    missing = [c for c in (_CASE, _VITAL, _DEATH, _FOLLOWUP) if c not in clin.columns]
    if missing:
        raise ValueError(
            f"clinical.tsv is missing expected GDC columns: {missing}.\n"
            f"Found columns: {list(clin.columns)}"
        )

    # Reduce PER COLUMN over each case's rows, ignoring nulls.
    #
    # A GDC clinical export has several rows per case (one per diagnosis /
    # treatment), and `diagnoses.days_to_last_follow_up` is a *diagnoses*-level
    # field that is frequently blank on the first row. Taking `drop_duplicates`
    # (first row) therefore hands back OS_time=NaN for many living patients,
    # who are then dropped by the survival task's dropna. Because only *Alive*
    # cases lose their time this way, the dropout is entirely censored cases —
    # informative censoring that enriches the cohort for deaths and biases every
    # Cox / nllsurv fit. Reducing per column over non-null values avoids it.
    #
    # `max` is the correct reducer for the two time fields: the longest observed
    # follow-up, and the latest recorded death day. (On the current TCGA exports
    # no case has >1 distinct non-null value for either, so max == first-non-null
    # today; the choice only matters if a future export disagrees.)
    def _reduce(col: str, how: str = "first") -> pd.Series:
        s = clin[[_CASE, col]].dropna(subset=[col])
        g = s.groupby(_CASE)[col]
        return g.max() if how == "max" else g.first()

    sub = pd.concat(
        {
            "vital_status": _reduce(_VITAL),
            "days_to_death": pd.to_numeric(_reduce(_DEATH, "max"), errors="coerce"),
            "days_to_last_follow_up": pd.to_numeric(_reduce(_FOLLOWUP, "max"), errors="coerce"),
        },
        axis=1,
    ).rename_axis("case_id").reset_index()

    # Vectorized: Dead -> (1, days_to_death); Alive -> (0, days_to_last_follow_up);
    # anything else -> (NaN, NaN).
    vs = sub["vital_status"]
    sub["OS_event"] = np.where(vs == "Dead", 1.0, np.where(vs == "Alive", 0.0, np.nan))
    sub["OS_time"] = np.where(
        vs == "Dead", sub["days_to_death"],
        np.where(vs == "Alive", sub["days_to_last_follow_up"], np.nan),
    )
    return sub[["case_id", "OS_event", "OS_time"]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifest", required=True, help="Path to normalized_manifest.csv")
    parser.add_argument("--clinical", required=True, help="Path to GDC clinical.tsv")
    parser.add_argument("--output", default=None, help="Output path (default: overwrite manifest)")
    parser.add_argument(
        "--case-col",
        default="case_id",
        help="Manifest column holding the GDC case submitter_id to join on "
        "(default: case_id; TCGA manifests use sample_names)",
    )
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    case_col = args.case_col
    if case_col not in manifest.columns:
        raise ValueError(f"manifest has no {case_col!r} column; found {list(manifest.columns)}")

    os_table = build_os_table(args.clinical)

    # Make re-runs idempotent. This script overwrites the manifest in place, so
    # a second run would merge OS_event/OS_time onto columns that already exist
    # and pandas would silently emit OS_event_x / OS_event_y — which then fails
    # far away with a confusing KeyError. Drop the previous values first so a
    # re-run simply refreshes them.
    stale = [c for c in ("OS_event", "OS_time") if c in manifest.columns]
    if stale:
        print(f"  refreshing existing column(s): {stale}")
        manifest = manifest.drop(columns=stale)

    before = len(manifest)
    merged = manifest.merge(
        os_table.rename(columns={"case_id": case_col}), on=case_col, how="left"
    )
    if len(merged) != before:
        raise RuntimeError(
            f"Row count changed after merge ({before} -> {len(merged)}); "
            "duplicate case_id in the clinical export?"
        )

    n_event = int((merged["OS_event"] == 1).sum())
    n_censored = int((merged["OS_event"] == 0).sum())
    n_nan = int(merged["OS_event"].isna().sum())
    print(f"Slides: {len(merged)}")
    print(f"  OS_event=1 (Dead):         {n_event}")
    print(f"  OS_event=0 (Alive):        {n_censored}")
    print(f"  OS_event=NaN (Not Rep.):   {n_nan}  <- dropped by survival task dropna")
    # Report OS_time NaNs separately: an event can be known while its time is
    # missing, and those rows are dropped too. Counting only OS_event NaNs hides
    # exactly the censored-case dropout this script's per-column reduce fixes.
    n_time_nan = int(merged["OS_time"].isna().sum())
    n_time_nan_censored = int((merged["OS_time"].isna() & (merged["OS_event"] == 0)).sum())
    print(f"  OS_time=NaN:               {n_time_nan}  "
          f"({n_time_nan_censored} of them censored)  <- also dropped")

    if n_event + n_censored == 0:
        print(
            f"  WARNING: no cases matched — check that the manifest's {case_col!r} "
            "matches the GDC 'cases.submitter_id' format."
        )
    n_nonpos = int((merged["OS_time"] <= 0).sum())
    if n_nonpos:
        print(
            f"  WARNING: {n_nonpos} slides have OS_time <= 0 (a GDC date artifact); "
            "verify these before relying on them."
        )

    out_path = args.output or args.manifest
    merged.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
