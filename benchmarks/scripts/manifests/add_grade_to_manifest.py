"""Add a GRADE column to a GDC-derived normalized_manifest.csv.

Joins tumour grade from a GDC clinical export (`clinical.tsv`) onto the manifest
by its case-id column (`--case-col`, default `case_id`; TCGA manifests key on
`sample_names`), mapping the ordinal grade to integer classes:

    G1 -> 0,  G2 -> 1,  G3 -> 2

Cases whose grade is **GX**, **G4**, or absent get NaN, so `create_task_csv`
drops them from the grade task (it dropna's the label column). G4 is excluded
because the roster defines grade as a 3-class G1/G2/G3 task; on TCGA-HNSC this
leaves 54/260/100 = 414 gradeable of 431 cases (GX 11 + G4 4 + 2 unlabelled).

Note on `diagnoses.tumor_grade`: it is a *diagnoses*-level field, so a case can
span several clinical rows and the first row is often blank. This script takes
the first **non-null** value per case — taking merely the first row undercounts
(51/247/100 instead of 54/260/100 on HNSC).

Usage:
    uv run --package autobench python benchmarks/scripts/manifests/add_grade_to_manifest.py \\
        --manifest /path/to/normalized_manifest.csv \\
        --clinical /path/to/clinical.tsv \\
        --case-col sample_names
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

# GDC clinical TSV columns (shared across programs).
_CASE = "cases.submitter_id"
_GRADE = "diagnoses.tumor_grade"

# Ordinal 3-class mapping; anything not listed (GX, G4, blank) -> NaN -> dropped.
GRADE_MAP = {"G1": 0, "G2": 1, "G3": 2}


def build_grade_table(clinical_tsv: str) -> pd.DataFrame:
    """Return a per-case table with columns ``case_id, GRADE``."""
    clin = pd.read_csv(clinical_tsv, sep="\t", low_memory=False)
    clin = clin.replace(["'--", "--"], np.nan)  # GDC missing-value sentinels

    missing = [c for c in (_CASE, _GRADE) if c not in clin.columns]
    if missing:
        raise ValueError(
            f"clinical.tsv is missing expected GDC columns: {missing}.\n"
            f"Found columns: {list(clin.columns)}"
        )

    # First NON-NULL grade per case (see module docstring).
    sub = (
        clin[[_CASE, _GRADE]]
        .dropna(subset=[_GRADE])
        .drop_duplicates(_CASE)
        .rename(columns={_CASE: "case_id", _GRADE: "tumor_grade"})
    )
    sub["GRADE"] = sub["tumor_grade"].map(GRADE_MAP)
    return sub[["case_id", "GRADE", "tumor_grade"]]


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

    grade_table = build_grade_table(args.clinical)

    before = len(manifest)
    merged = manifest.merge(
        grade_table[["case_id", "GRADE"]].rename(columns={"case_id": case_col}),
        on=case_col,
        how="left",
    )
    if len(merged) != before:
        raise RuntimeError(
            f"Row count changed after merge ({before} -> {len(merged)}); "
            "duplicate case_id in the clinical export?"
        )

    counts = merged["GRADE"].value_counts(dropna=False).sort_index()
    inv = {v: k for k, v in GRADE_MAP.items()}
    print(f"Slides: {len(merged)}")
    for val, n in counts.items():
        label = inv.get(val, "unlabelled (GX / G4 / absent)") if pd.notna(val) else \
            "NaN  <- dropped by the grade task"
        print(f"  GRADE={val!s:<5} {label:<38} {n}")

    n_labelled = int(merged["GRADE"].notna().sum())
    print(f"  gradeable slides: {n_labelled} of {len(merged)}")
    if n_labelled == 0:
        print(
            f"  WARNING: no cases matched — check that the manifest's {case_col!r} "
            "matches the GDC 'cases.submitter_id' format."
        )

    # Report what was excluded, so the drop is visible rather than silent.
    # Scope this to the MANIFEST's cases, not the whole clinical export: the
    # export usually covers more cases than the manifest, so a clinical-wide
    # count overstates the loss. Cases with no grade row at all are folded in
    # here too — build_grade_table drops them, so they cannot appear otherwise.
    raw_by_case = grade_table.set_index("case_id")["tumor_grade"]
    scoped = pd.Series(
        {c: raw_by_case.get(c, None) for c in manifest[case_col].dropna().unique()},
        dtype="object",
    )
    excluded = (
        scoped[~scoped.isin(GRADE_MAP)].fillna("no grade row").value_counts().to_dict()
    )
    if excluded:
        print(f"  excluded non-{{G1,G2,G3}} cases (manifest-scoped): {excluded}")

    out_path = args.output or args.manifest
    merged.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
