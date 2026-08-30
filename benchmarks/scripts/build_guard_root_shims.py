#!/usr/bin/env python3
"""Rebuild guard-root benchmark shims so the frozen source tree stays frozen.

The campaign reads its canonical inputs through
``<guard_roots>/<cohort>/benchmark``. Until now that was one symlink straight
into the frozen results tree (version2 ``benchmark_5fold``), which let
campaign-reachable prepare writes land in it: the TITAN per-task manifest is
rewritten on EVERY titan run, and dataset_csv/splits/nnmil artifacts are
written whenever missing. This builder replaces the symlink with a real
directory whose shape follows the audited write map:

  benchmark/                     REAL directory (``.shim_source`` records
  |                              the frozen source it was built from)
  |- dataset_csv/   COPY   fingerprint mkstemp+rename and task CSVs land here
  |- splits/        COPY   fold CSVs MUST be real files: the split cache
  |                        rejects symlinks and would regenerate
  |- titan/         COPY   per-task manifest.json, rewritten every titan run
  |- nnmil/         COPY   plan dirs, written only when dataset_plan.json
  |                        is missing
  |- features/      REAL dirs + per-FILE symlinks: the pt_files mkdir is
  |                        unconditional and a missing .pt would regenerate
  |                        THROUGH a plain directory symlink
  |- results/       EMPTY  write sinks: unreachable from the campaign
  |- aggregated/    EMPTY  entrypoint (orchestrator-only writers), and
  |- logs/          EMPTY  nothing in the campaign path reads them

An unknown top-level entry in the source fails the build closed — a new
entry means the write map must be re-audited, never silently symlinked.
Copied bytes are hash-verified against the source before the swap, and the
swap is refused while any campaign job could be running (see --i-know-idle).

Stdlib-only on purpose: it runs on the cluster checkout standalone.

Usage (on the cluster, from the repo root, with NO campaign job running):
    python3 benchmarks/scripts/build_guard_root_shims.py \
        --guard-roots ~/scratch/autoMIL/guard_roots --i-know-idle
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

CAMPAIGN_REL = Path("benchmarks/campaigns/preprint_130")
SHIM_SOURCE_MARKER = ".shim_source"
COPY_ENTRIES = ("dataset_csv", "splits", "titan", "nnmil")
LINKED_TREE_ENTRIES = ("features",)
SINK_ENTRIES = ("results", "aggregated", "logs")
EXPECTED_ENTRIES = frozenset(COPY_ENTRIES + LINKED_TREE_ENTRIES + SINK_ENTRIES)


class ShimError(RuntimeError):
    """The shim cannot be built safely; nothing was swapped."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def roster_cohorts(repo_root: Path) -> tuple[str, ...]:
    roster_path = repo_root / CAMPAIGN_REL / "active_roster.json"
    try:
        roster = json.loads(roster_path.read_text())
        cohorts = tuple(roster["cohorts"])
    except (OSError, ValueError, KeyError) as exc:
        raise ShimError(f"cannot read active roster {roster_path}: {exc!r}") from exc
    if not cohorts or any(not isinstance(cohort, str) for cohort in cohorts):
        raise ShimError(f"invalid roster cohorts: {cohorts!r}")
    return cohorts


def resolve_source(benchmark_path: Path) -> Path:
    """The frozen source tree this shim mirrors.

    First build: ``benchmark`` is the legacy symlink into the frozen tree.
    Rebuild: ``benchmark`` is a prior shim directory whose marker records
    the source. Anything else is unexpected and fails closed.
    """
    if benchmark_path.is_symlink():
        source = benchmark_path.resolve()
    elif benchmark_path.is_dir() and (benchmark_path / SHIM_SOURCE_MARKER).is_file():
        source = Path(
            (benchmark_path / SHIM_SOURCE_MARKER).read_text().strip()
        )
    else:
        raise ShimError(
            f"{benchmark_path} is neither the legacy symlink nor a prior "
            "shim (no .shim_source marker); refusing to guess the source"
        )
    if not source.is_dir():
        raise ShimError(f"shim source is not a directory: {source}")
    return source


def _copy_verified(source: Path, target: Path) -> int:
    """Copy a small write-prone subtree; every byte hash-verified."""
    shutil.copytree(source, target, symlinks=False)
    copied = 0
    for dirpath, _dirnames, filenames in os.walk(source):
        rel = Path(dirpath).relative_to(source)
        for name in filenames:
            source_file = Path(dirpath) / name
            target_file = target / rel / name
            if target_file.is_symlink() or not target_file.is_file():
                raise ShimError(f"copy produced a non-file at {target_file}")
            if _sha256(source_file) != _sha256(target_file):
                raise ShimError(f"copy verification failed for {target_file}")
            copied += 1
    return copied


def _link_tree(source: Path, target: Path) -> int:
    """Real directories, per-file symlinks: writes land in the shim."""
    linked = 0
    for dirpath, _dirnames, filenames in os.walk(source):
        rel = Path(dirpath).relative_to(source)
        (target / rel).mkdir(parents=True, exist_ok=True)
        for name in filenames:
            link = target / rel / name
            link.symlink_to(Path(dirpath) / name)
            if not link.exists():
                raise ShimError(f"dangling feature symlink at {link}")
            linked += 1
    return linked


def build_shim(benchmark_path: Path) -> dict[str, object]:
    source = resolve_source(benchmark_path)
    entries = {entry.name for entry in source.iterdir()}
    unknown = sorted(entries - EXPECTED_ENTRIES)
    if unknown:
        raise ShimError(
            f"{source} carries top-level entries outside the audited write "
            f"map: {unknown}; re-audit before shimming"
        )
    staging = benchmark_path.parent / f".{benchmark_path.name}.shim.new"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    stats: dict[str, object] = {"source": str(source)}
    try:
        for entry in COPY_ENTRIES:
            if (source / entry).is_dir():
                stats[f"copied:{entry}"] = _copy_verified(
                    source / entry, staging / entry
                )
        for entry in LINKED_TREE_ENTRIES:
            if (source / entry).is_dir():
                stats[f"linked:{entry}"] = _link_tree(
                    source / entry, staging / entry
                )
        for entry in SINK_ENTRIES:
            (staging / entry).mkdir()
        (staging / SHIM_SOURCE_MARKER).write_text(f"{source}\n")
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    # Swap: the retired object is kept beside the shim for one build cycle.
    retired = benchmark_path.parent / f".{benchmark_path.name}.pre_shim"
    if retired.exists() or retired.is_symlink():
        if retired.is_symlink() or retired.is_file():
            retired.unlink()
        else:
            shutil.rmtree(retired)
    if benchmark_path.exists() or benchmark_path.is_symlink():
        os.rename(benchmark_path, retired)
    os.rename(staging, benchmark_path)
    stats["retired"] = str(retired)
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--guard-roots", required=True)
    parser.add_argument(
        "--repo-root", default=str(Path(__file__).resolve().parents[2]),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--i-know-idle", action="store_true",
        help=(
            "Required confirmation that NO campaign job is running: the "
            "swap moves the benchmark tree under running workers otherwise."
        ),
    )
    args = parser.parse_args(argv)
    if not args.i_know_idle:
        print(
            "build_guard_root_shims: refusing without --i-know-idle — the "
            "swap must never run under a live campaign job",
            file=sys.stderr,
        )
        return 1
    guard_roots = Path(os.path.expanduser(args.guard_roots))
    if not guard_roots.is_dir():
        print(f"guard roots not a directory: {guard_roots}", file=sys.stderr)
        return 1
    try:
        cohorts = roster_cohorts(Path(args.repo_root).resolve())
    except ShimError as exc:
        print(f"build_guard_root_shims: {exc}", file=sys.stderr)
        return 1
    failures = 0
    for cohort in cohorts:
        benchmark_path = guard_roots / cohort / "benchmark"
        try:
            stats = build_shim(benchmark_path)
        except ShimError as exc:
            print(f"{cohort}: FAILED — {exc}", file=sys.stderr)
            failures += 1
            continue
        summary = ", ".join(
            f"{key.split(':', 1)[1]}={value}"
            for key, value in sorted(stats.items())
            if key.startswith(("copied:", "linked:"))
        )
        print(f"{cohort}: shim built from {stats['source']} ({summary})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
