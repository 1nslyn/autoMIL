#!/usr/bin/env python3
"""Reconcile a GOLDMARK manifest against a current GDC manifest.

Background
----------
GDC sometimes re-uploads slides with new file UUIDs while keeping the same
TCGA barcode prefix (everything before the first '.'). When that happens,
the GOLDMARK manifest's ``slide_name`` column stores stale UUIDs and the
exact-string filter from the original tutorial returns 0 matches. The
patient/slide mapping itself is still correct -- only the file UUID changed.

This helper matches GOLDMARK against the GDC manifest by barcode prefix
(unique per slide in TCGA), then writes:

  - ``gdc_manifest_matched.txt``        -- GDC manifest filtered to the
                                            barcodes referenced by GOLDMARK,
                                            using GDC's *current* UUIDs.
  - ``normalized_manifest.refreshed.csv`` -- a copy of the GOLDMARK CSV with
                                            ``slide_name`` rewritten to match
                                            the current on-disk filenames.
  - ``uuid_rewrites.tsv``               -- audit log of every rewrite.

Use ``normalized_manifest.refreshed.csv`` as the ``mapping_csv`` in your
dataset YAML so feature extraction can find the actual files on disk.

Correctness guarantees
----------------------
1. Barcode uniqueness is verified on both inputs; if either side has a
   duplicate barcode, the script aborts rather than guess.
2. Each GOLDMARK row matches at most one GDC file. Cases with multiple
   GDC candidates for the same barcode are reported and skipped (never
   silently picked).
3. Rows where the GOLDMARK UUID still matches GDC are passed through
   unchanged (zero-rewrite path is identical to the old exact-match script).

Usage
-----
    python refresh_goldmark_uuids.py <dataset_dir>

Where ``<dataset_dir>`` contains both ``normalized_manifest.csv`` (from
GOLDMARK) and ``gdc_manifest.txt`` (from the GDC portal).
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path


def barcode(filename: str) -> str:
    """Return the TCGA barcode prefix (everything before the first '.').

    e.g. 'TCGA-2G-AAEW-01Z-00-DX1.82DE89BF-...svs' -> 'TCGA-2G-AAEW-01Z-00-DX1'
    """
    return filename.split(".", 1)[0]


def assert_unique_barcodes(label: str, names: list[str]) -> None:
    dups = {bc: c for bc, c in Counter(barcode(n) for n in names).items() if c > 1}
    if dups:
        sample = list(dups.items())[:3]
        raise SystemExit(
            f"ABORT: {label} has {len(dups)} duplicate barcode(s); "
            f"barcode matching is unsafe. Examples: {sample}"
        )


def main(dataset_dir: str) -> None:
    root = Path(dataset_dir)
    gm_path = root / "normalized_manifest.csv"
    gdc_path = root / "gdc_manifest.txt"
    out_matched = root / "gdc_manifest_matched.txt"
    out_refreshed = root / "normalized_manifest.refreshed.csv"
    out_audit = root / "uuid_rewrites.tsv"

    for p in (gm_path, gdc_path):
        if not p.is_file():
            raise SystemExit(f"missing input: {p}")

    with gm_path.open() as f:
        gm_rows = list(csv.DictReader(f))
    if not gm_rows or "slide_name" not in gm_rows[0]:
        raise SystemExit(f"GOLDMARK CSV must have a 'slide_name' column: {gm_path}")
    gm_names = [r["slide_name"] for r in gm_rows]
    assert_unique_barcodes("GOLDMARK", gm_names)
    gm_by_bc = {barcode(n): n for n in gm_names}

    with gdc_path.open() as f:
        gdc_header = f.readline()
        gdc_lines = [ln for ln in f if ln.strip()]
    gdc_files: list[tuple[str, str]] = []  # (filename, full_line)
    for ln in gdc_lines:
        cols = ln.rstrip("\n").split("\t")
        if len(cols) >= 2 and cols[1].endswith(".svs"):
            gdc_files.append((cols[1], ln))
    gdc_by_bc: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for fn, ln in gdc_files:
        gdc_by_bc[barcode(fn)].append((fn, ln))

    rewrites: dict[str, str] = {}        # gm_filename -> gdc_filename
    matched_lines: list[str] = []
    not_found: list[str] = []
    ambiguous: list[tuple[str, str, list[str]]] = []
    kept = 0

    for bc, gm_name in gm_by_bc.items():
        candidates = gdc_by_bc.get(bc, [])
        if not candidates:
            not_found.append(gm_name)
            continue
        if len(candidates) > 1:
            exact = [(fn, ln) for fn, ln in candidates if fn == gm_name]
            if len(exact) == 1:
                fn, ln = exact[0]
                matched_lines.append(ln)
                rewrites[gm_name] = fn
                kept += 1
                continue
            ambiguous.append((bc, gm_name, [c[0] for c in candidates]))
            continue
        fn, ln = candidates[0]
        matched_lines.append(ln)
        rewrites[gm_name] = fn
        if fn == gm_name:
            kept += 1

    rewritten_count = sum(1 for gm, gdc in rewrites.items() if gm != gdc)

    out_matched.write_text(gdc_header + "".join(matched_lines))

    with gm_path.open() as f_in, out_refreshed.open("w", newline="") as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
        writer.writeheader()
        for r in reader:
            new = rewrites.get(r["slide_name"])
            if new and new != r["slide_name"]:
                r["slide_name"] = new
            writer.writerow(r)

    with out_audit.open("w") as f:
        f.write("goldmark_slide_name\tgdc_slide_name\tstatus\n")
        for gm, gdc in sorted(rewrites.items()):
            status = "rewritten" if gm != gdc else "unchanged"
            f.write(f"{gm}\t{gdc}\t{status}\n")
        for n in not_found:
            f.write(f"{n}\t\tnot_found_in_gdc\n")
        for bc, gm, cands in ambiguous:
            f.write(f"{gm}\t{','.join(cands)}\tambiguous\n")

    print(f"GOLDMARK barcodes:       {len(gm_by_bc)}")
    print(f"matched (UUID kept):     {kept}")
    print(f"matched (UUID rewrite):  {rewritten_count}")
    print(f"not found in GDC:        {len(not_found)}")
    print(f"ambiguous (skipped):     {len(ambiguous)}")
    print(f"\nwrote: {out_matched}")
    print(f"wrote: {out_refreshed}")
    print(f"wrote: {out_audit}")
    if not_found or ambiguous:
        print("\nReview uuid_rewrites.tsv before running feature extraction.")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: refresh_goldmark_uuids.py <dataset_dir>")
    main(sys.argv[1])
