#!/usr/bin/env python
"""Collect ``summary.json`` results across one or more benchmark_dir trees into CSVs.

Walks ``results/**/summary.json`` under each ``--roots`` directory (typically
one per dataset cohort -- the roster spans TCGA *and* CPTAC, so pass every
cohort's ``benchmark_dir`` that should be pooled) and writes:

  --out            wide frame, one row per experiment
                    (``autobench.pipeline.collect.summaries_to_frame``)
  --per-fold-out    long frame, one row per (experiment, split, fold, metric)
                    (``autobench.pipeline.collect.per_fold_frame``)

Examples
--------
python benchmarks/scripts/collect_results.py \
    --roots /data/tcga-luad/benchmark /data/cptac-gbm/benchmark \
    --out paper/preprint/figures/results.csv \
    --per-fold-out paper/preprint/figures/per_fold.csv
"""

from __future__ import annotations

import argparse
import os
import sys

from autobench.pipeline.collect import collect_summaries, per_fold_frame, summaries_to_frame


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--roots", nargs="+", required=True,
                   help="One or more benchmark_dir roots to walk "
                        "(results/**/summary.json under each)")
    p.add_argument("--out", required=True,
                   help="Output CSV path for the wide per-experiment frame")
    p.add_argument("--per-fold-out", dest="per_fold_out", default=None,
                   help="Optional output CSV path for the long per-fold frame")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    summaries = collect_summaries(args.roots)
    if not summaries:
        print(f"[collect_results] WARNING: no summary.json found under {args.roots}",
              file=sys.stderr)

    results_df = summaries_to_frame(summaries)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    results_df.to_csv(args.out, index=False)
    print(f"[collect_results] {len(results_df)} experiments -> {args.out}")

    if args.per_fold_out:
        fold_df = per_fold_frame(summaries)
        per_fold_dir = os.path.dirname(os.path.abspath(args.per_fold_out))
        if per_fold_dir:
            os.makedirs(per_fold_dir, exist_ok=True)
        fold_df.to_csv(args.per_fold_out, index=False)
        print(f"[collect_results] {len(fold_df)} per-fold rows -> {args.per_fold_out}")


if __name__ == "__main__":
    main()
