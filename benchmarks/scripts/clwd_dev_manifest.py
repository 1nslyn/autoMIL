#!/usr/bin/env python
"""Write CLWD_dev.csv: CLWD.csv without the cases whose slides disagree on the label.

The case-level stratification (autobench.pipeline.splits) requires one label per
case, and five CLWD cases carry two 6-class growth patterns across their slides.
Dropping those cases, rather than their conflicting slides, keeps every remaining
case whole, so no patient straddles train and validation.

    python benchmarks/scripts/clwd_dev_manifest.py $AUTOBENCH_CLWD_ROOT/CLWD.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

CASE_COLUMN = "SampleNumber"
LABEL_COLUMN = "Benchmark_Label_2021_6class"
OUTPUT_NAME = "CLWD_dev.csv"


def conflicting_cases(rows: list[dict[str, str]]) -> dict[str, set[str]]:
    """Cases whose slides carry more than one label."""
    labels: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        labels[row[CASE_COLUMN]].add(row[LABEL_COLUMN])
    return {case: found for case, found in labels.items() if len(found) > 1}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path, help="the original CLWD.csv")
    args = parser.parse_args(argv)
    with args.manifest.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    for column in (CASE_COLUMN, LABEL_COLUMN):
        if column not in fieldnames:
            print(f"{args.manifest} has no column {column!r}", file=sys.stderr)
            return 1
    dropped = conflicting_cases(rows)
    kept = [row for row in rows if row[CASE_COLUMN] not in dropped]
    output = args.manifest.with_name(OUTPUT_NAME)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
    for case, labels in sorted(dropped.items()):
        print(f"dropped case {case}: {sorted(labels)}")
    print(f"{output}: kept {len(kept)} of {len(rows)} slides ({len(dropped)} cases dropped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
