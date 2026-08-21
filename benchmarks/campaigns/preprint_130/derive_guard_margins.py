#!/usr/bin/env python
"""Derive the companion-guard margins for every classification cell.

Writes ``guard_margins.json`` — the frozen record of how much balanced accuracy
a candidate may lose before the keep/discard gate rejects it, one entry per
``<dataset>__<task>``, each carrying the per-fold validation class counts the
number was computed from.

Run this where the dataset roots are MOUNTED; it reads each cohort's split CSVs
and task labels. It is deliberately a separate step from freezing the manifest:
the derivation needs the data, the freeze must be reproducible anywhere, and
the artifact in between is a reviewable record whose arithmetic can be checked
by hand from the counts it prints.

    uv run python benchmarks/campaigns/preprint_130/derive_guard_margins.py
    uv run python benchmarks/campaigns/preprint_130/derive_guard_margins.py --write

Datasets whose root is not mounted are reported and the run FAILS: a partial
margins file would freeze a manifest in which some cells silently carry no
guard. To cross-check a derived count against a cohort that has actually run,
use ``autobench.guard_margin.verify_against_run`` on that cell's baseline
``.../certify/results`` directory — it compares the split assignment against
the slides the run really scored.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "src"))

from dotenv import load_dotenv  # noqa: E402

# Same convention as run_experiment.py: the cohort roots live in the gitignored
# benchmarks/.env, so resolve them the way every other entry point does.
load_dotenv(REPO_ROOT / "benchmarks" / ".env")

from autobench.campaign import (DATASETS, GUARD_MARGINS_PATH,  # noqa: E402
                                STAGE_FOLDS, _dataset_config_path)
from autobench.guard_margin import GuardMarginError, derive_guard  # noqa: E402


def _benchmark_dir(dataset: str) -> Path:
    """The cohort's benchmark directory, or a GuardMarginError naming the mount."""
    import yaml

    raw = yaml.safe_load(_dataset_config_path(REPO_ROOT, dataset).read_text()) or {}
    paths = raw.get("paths") or {}
    data_root = os.path.expandvars(paths.get("data_root", ""))
    if not data_root or "${" in data_root:
        raise GuardMarginError(
            f"{dataset}: {paths.get('data_root')} is unset in benchmarks/.env"
        )
    benchmark_dir = Path(os.path.expandvars(
        (paths.get("benchmark_dir") or "${data_root}/benchmark")
        .replace("${data_root}", data_root)
    ))
    if not benchmark_dir.is_dir():
        raise GuardMarginError(f"{dataset}: {benchmark_dir} is not mounted")
    return benchmark_dir


def derive_all() -> dict[str, dict]:
    import yaml

    folds = STAGE_FOLDS["discovery"]
    margins: dict[str, dict] = {}
    failures: list[str] = []
    for dataset in DATASETS:
        raw = yaml.safe_load(_dataset_config_path(REPO_ROOT, dataset).read_text()) or {}
        tasks = raw.get("tasks") or {}
        for task, spec in tasks.items():
            if (spec or {}).get("task_type", "classification") == "survival":
                continue   # no balanced accuracy, no guard
            try:
                margins[f"{dataset}__{task}"] = derive_guard(
                    _benchmark_dir(dataset), "standard", task, folds
                )
            except GuardMarginError as exc:
                failures.append(str(exc))
    if failures:
        raise GuardMarginError(
            "could not derive every classification cell's margin:\n  "
            + "\n  ".join(failures)
        )
    return margins


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help=f"write {GUARD_MARGINS_PATH} (default: print only)")
    args = parser.parse_args()

    try:
        margins = derive_all()
    except GuardMarginError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    width = max(len(k) for k in margins)
    for key, guard in sorted(margins.items()):
        print(f"{key:{width}}  margin {guard['margin']:.6f}  {guard['basis']}")

    if args.write:
        path = REPO_ROOT / GUARD_MARGINS_PATH
        path.write_text(json.dumps(margins, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {len(margins)} margins to {path}")
        print("Now regenerate the manifest so the cells pick them up.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
