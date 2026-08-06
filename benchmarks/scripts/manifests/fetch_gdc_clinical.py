"""Fetch overall-survival fields from the GDC API into a GDC-style clinical.tsv.

Some cohorts ship neither a `clinical.tsv` nor a Patho-Bench survival task — e.g.
CPTAC-GBM, whose Patho-Bench split set is {EGFR_mutation, Immune_class,
TP53_mutation} with no OS. This script pulls the OS fields straight from the
public GDC API for the cases in a manifest and writes them in the same column
layout a GDC clinical export uses, so `add_os_to_manifest.py` consumes the result
unchanged.

Usage:
    uv run --package autobench python benchmarks/scripts/manifests/fetch_gdc_clinical.py \\
        --manifest /path/to/normalized_manifest.csv \\
        --case-col case_id \\
        --output /path/to/clinical.tsv

    uv run --package autobench python benchmarks/scripts/manifests/add_os_to_manifest.py \\
        --manifest /path/to/normalized_manifest.csv \\
        --clinical /path/to/clinical.tsv --case-col case_id

Only the public/open GDC `cases` endpoint is used — no token required.
"""

from __future__ import annotations

import argparse
import json
import urllib.request

import pandas as pd

GDC_CASES = "https://api.gdc.cancer.gov/cases"

# Emit exactly the column names a GDC clinical export uses, so the downstream
# add_os_to_manifest.py needs no changes.
_CASE = "cases.submitter_id"
_VITAL = "demographic.vital_status"
_DEATH = "demographic.days_to_death"
_FOLLOWUP = "diagnoses.days_to_last_follow_up"

_FIELDS = ",".join([
    "submitter_id",
    "demographic.vital_status",
    "demographic.days_to_death",
    "diagnoses.days_to_last_follow_up",
])


def fetch_cases(case_ids: list[str], chunk_size: int = 400, timeout: int = 60) -> list[dict]:
    """Query the GDC cases endpoint for the given submitter_ids, in chunks."""
    hits: list[dict] = []
    for i in range(0, len(case_ids), chunk_size):
        chunk = case_ids[i:i + chunk_size]
        payload = {
            "filters": {
                "op": "in",
                "content": {"field": "submitter_id", "value": chunk},
            },
            "fields": _FIELDS,
            "format": "JSON",
            # GDC defaults to size=10; ask for the whole chunk.
            "size": str(len(chunk)),
        }
        req = urllib.request.Request(
            GDC_CASES,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            body = json.load(resp)
        hits.extend(body["data"]["hits"])
        print(f"  fetched {len(chunk)} ids -> {len(body['data']['hits'])} hits")
    return hits


def flatten(hits: list[dict]) -> pd.DataFrame:
    """Flatten GDC case hits into the four clinical columns we need."""
    rows = []
    for h in hits:
        demo = h.get("demographic") or {}
        diags = h.get("diagnoses") or []
        # diagnoses is a list; take the first non-null follow-up value.
        followup = next(
            (d.get("days_to_last_follow_up") for d in diags
             if d.get("days_to_last_follow_up") is not None),
            None,
        )
        rows.append({
            _CASE: h.get("submitter_id"),
            _VITAL: demo.get("vital_status"),
            _DEATH: demo.get("days_to_death"),
            _FOLLOWUP: followup,
        })
    return pd.DataFrame(rows, columns=[_CASE, _VITAL, _DEATH, _FOLLOWUP])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifest", required=True, help="Path to normalized_manifest.csv")
    parser.add_argument("--output", required=True, help="Path to write clinical.tsv")
    parser.add_argument(
        "--case-col",
        default="case_id",
        help="Manifest column holding the GDC case submitter_id "
        "(default: case_id; TCGA manifests use sample_names)",
    )
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    if args.case_col not in manifest.columns:
        raise ValueError(
            f"manifest has no {args.case_col!r} column; found {list(manifest.columns)}"
        )

    case_ids = sorted(manifest[args.case_col].dropna().astype(str).unique())
    print(f"Manifest cases: {len(case_ids)}")

    hits = fetch_cases(case_ids)
    df = flatten(hits)
    df = df.dropna(subset=[_CASE]).drop_duplicates(_CASE)

    matched = set(df[_CASE]) & set(case_ids)
    print(f"GDC returned {len(df)} cases; matched {len(matched)} of {len(case_ids)} manifest cases")
    if not matched:
        print(
            "  WARNING: nothing matched — check that the manifest's case ids use the "
            "GDC submitter_id format (CPTAC uses C3L-/C3N-…, TCGA uses TCGA-XX-XXXX)."
        )
    missing = sorted(set(case_ids) - matched)
    if missing:
        print(f"  {len(missing)} manifest cases absent from GDC, e.g. {missing[:5]}")

    vital = df[_VITAL].value_counts(dropna=False).to_dict()
    print(f"  vital_status: {vital}")

    df.to_csv(args.output, sep="\t", index=False)
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
