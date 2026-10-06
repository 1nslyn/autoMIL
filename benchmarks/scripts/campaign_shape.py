#!/usr/bin/env python3
"""Predict a campaign cell's job wall time and pick its SLURM job shape.

For each cell root under a campaign runtime directory, this script reads the
registered baseline's five-fold elapsed time from ``campaign_state.json`` and
predicts how long the cell's job takes (see ``predict_hours``: the serial
gate attempt, the discovery batches one after another with one of them
running into the attempt timeout, plus a fixed overhead). Every job takes
one GPU, because a whole batch runs on one; the
wall is the shorter of ``WALL_OPTIONS_H`` whose ``FIT_FRACTION`` holds the
prediction.

Cells without a registered baseline (``baseline`` is ``None``), with
malformed or missing state, or whose predicted time exceeds every wall are
reported as unshaped with a reason; a bad cell never crashes the sweep over
the rest.

This file is deliberately standalone (stdlib only): it is delivered to the
cluster mid-campaign, alongside campaign_export.py, and must never import
autobench/automil.

Usage:
    campaign_shape.py --runtime <dir> [--cells a,b,...] [--json]
    campaign_shape.py --runtime <dir> --cell <id> --field gpus
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

# Per-GPU concurrent-attempt cap. The launcher hands it to the daemon as
# AUTOMIL_MAX_CONCURRENT_PER_GPU, so the packing the prediction assumes is
# the packing the job runs (the frozen cell config's own cap was written
# for another host). Sized from the four LUAD KRAS rehearsal cells
# (2026-09-13/14): packed 4 per GPU they used 18-21% of their cores (nnMIL
# 54%), 4-5 GB of RAM and 1-1.5 GB of VRAM per attempt against 12 cores,
# 128 GB and 80 GB per GPU, and each attempt ran only 1.1-1.2x its serial
# time; 8 per GPU stays inside every one of those budgets.
FIT_FRACTION = 0.85
CAP_PER_GPU = 8
OVERHEAD_H = 2.0

# A batch lasts as long as its slowest attempt, and the agent's candidates
# train longer than the baseline (later early stopping, heavier settings).
# Slowest attempt of each batch over the baseline's per-fold time, measured
# on the 2026-10 trial cells: fir H100 ABMIL 0.84-1.43; aihub RTX CLAM
# 1.08-1.26, DTFD 1.85-2.18, ABMIL 1.30-2.24. TITAN (2026-09-04) set the floor: 16 min
# attempts against a 2 min per-fold time, the process start-up and feature
# loading that no baseline time predicts. The serial gate attempt re-runs
# the baseline's own configuration alone, so only the floor applies to it.
ATTEMPT_DILATION = 2.0
ATTEMPT_FLOOR_H = 0.25
# One batch may run into the attempt timeout (autobench.campaign.
# ATTEMPT_TIMEOUT_MIN, 1020 min for a five-fold attempt): aihub DTFD's third
# batch tried heavier settings and lasted the full three-fold timeout
# (2026-10-02). A cell starts only on a wall that survives one such batch
# (Leo, 2026-10-03).
ATTEMPT_TIMEOUT_H = 17.0

# Every job takes one GPU and a quarter of a fir node's cores and memory: a
# whole batch runs on one GPU, so a second one would not shorten the cell.
# Eight packed ABMIL attempts used 5.2 cores on average and 56 GB at peak on
# fir (2026-10-03). The walls are fir's 24 h and 3-day GPU tiers; 12 h never
# holds a full cell once one batch may last 17 h.
JOB_GPUS = 1
JOB_CORES = 12
JOB_MEM_GB = 128
WALL_OPTIONS_H = (24, 72)

# The frozen protocol's stage structure (autobench.campaign STAGE_FOLDS and
# DISCOVERY_PHASING; a test pins these copies): every discovery attempt
# re-runs all 5 baseline folds, and the attempts are spent in fixed batches,
# each started only after the earlier ones have finished.
DISCOVERY_FOLDS = 5
TOTAL_FOLDS = 5
DISCOVERY_BATCHES = (8, 8, 8, 6)

SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class Shape:
    """A concrete SLURM job shape and its predicted job wall time."""

    gpus: int
    wall_hours: int
    cpus: int
    mem_gb: int
    predicted_hours: float


@dataclass(frozen=True)
class ShapeReport:
    """One cell's shaping outcome: a fitting ``Shape``, or a reason there isn't one.

    ``baseline_elapsed_seconds`` is the prediction input (the cell's 5-fold
    baseline elapsed time), carried so a submitter can record it.
    """

    cell_id: str
    shape: Shape | None
    reason: str | None
    baseline_elapsed_seconds: float | None = None
    cached_folds: int = 0
    """Folds the registered baseline loaded from its cache instead of
    training; the ledger's elapsed total excludes them, so the prediction
    input is scaled back to the full fold count (see _prediction_input)."""
    baseline_elapsed_source: str | None = None
    """Where the prediction input came from: ``ledger`` (registered total),
    ``ledger-scaled`` (cached folds scaled back), or ``operator`` (a time
    supplied at submission for a baseline whose retry cached every fold)."""


def fold_hours(e5_seconds: float, folds: int) -> float:
    """The baseline's own time for ``folds`` of its five folds, never
    below ``ATTEMPT_FLOOR_H``: the serial gate attempt's cost."""
    e5_hours = e5_seconds / SECONDS_PER_HOUR
    return max(ATTEMPT_FLOOR_H, e5_hours * (folds / TOTAL_FOLDS))


def attempt_hours(e5_seconds: float, folds: int) -> float:
    """The slowest agent attempt of a batch over ``folds`` folds: the
    baseline's per-fold time dilated by ``ATTEMPT_DILATION``, never below
    the floor."""
    e5_hours = e5_seconds / SECONDS_PER_HOUR
    return max(ATTEMPT_FLOOR_H, ATTEMPT_DILATION * e5_hours * (folds / TOTAL_FOLDS))


def rounds(attempts: int, gpus: int) -> int:
    """The rounds the daemon runs ``attempts`` submitted together in, on
    ``gpus`` GPUs at ``CAP_PER_GPU`` each."""
    return math.ceil(attempts / (CAP_PER_GPU * gpus))


def discovery_hours(e5_seconds: float, gpus: int) -> float:
    """The serial gate attempt, then the discovery batches one after another,
    each as long as its slowest attempt, with one of them running into the
    attempt timeout. A batch fits one GPU, so more GPUs do not shorten it."""
    attempt = attempt_hours(e5_seconds, DISCOVERY_FOLDS)
    batches = sum(rounds(size, gpus) for size in DISCOVERY_BATCHES)
    return (
        fold_hours(e5_seconds, DISCOVERY_FOLDS)
        + batches * attempt
        + max(0.0, ATTEMPT_TIMEOUT_H - attempt)
    )


def predict_hours(e5_seconds: float, gpus: int) -> float:
    """Predict a cell's job wall time (hours) on ``gpus`` GPUs.

    ``e5_seconds`` is the 5-fold baseline's total elapsed time, as recorded
    in ``baseline.resources.elapsed_seconds.total``.
    """
    return discovery_hours(e5_seconds, gpus) + OVERHEAD_H


def _job_shape(wall_hours: int, predicted_hours: float) -> Shape:
    return Shape(
        gpus=JOB_GPUS, wall_hours=wall_hours, cpus=JOB_CORES, mem_gb=JOB_MEM_GB,
        predicted_hours=predicted_hours,
    )


def finish_shape() -> Shape:
    """The finish-only recovery lane, on the shorter wall: the discovery
    freeze, the winner and the session close train nothing, so it needs no
    baseline time."""
    return _job_shape(WALL_OPTIONS_H[0], 0.0)


def choose_shape(e5_seconds: float) -> Shape | None:
    """The shorter wall whose ``FIT_FRACTION`` holds the predicted job time;
    ``None`` if neither does."""
    predicted = predict_hours(e5_seconds, JOB_GPUS)
    for wall_hours in WALL_OPTIONS_H:
        if predicted <= FIT_FRACTION * wall_hours:
            return _job_shape(wall_hours, predicted)
    return None


def _read_campaign_state(
    runtime: Path, cell_id: str,
) -> tuple[Mapping[str, object] | None, str | None]:
    state_path = runtime / cell_id / "campaign_state.json"
    if not state_path.is_file():
        return None, f"no campaign_state.json at {state_path}"
    try:
        raw = json.loads(state_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"cannot read campaign_state.json: {exc}"
    if not isinstance(raw, dict):
        return None, "campaign_state.json is not a JSON object"
    return raw, None


def _baseline_elapsed_seconds(
    state: Mapping[str, object],
) -> tuple[float | None, str | None]:
    baseline = state.get("baseline")
    if baseline is None:
        return None, "no baseline registered for this cell"
    try:
        total = baseline["resources"]["elapsed_seconds"]["total"]
    except (KeyError, TypeError):
        return None, "baseline.resources.elapsed_seconds.total is missing"
    if total is None:
        return None, (
            "baseline.resources.elapsed_seconds.total is null "
            "(no reported folds)"
        )
    if isinstance(total, bool) or not isinstance(total, (int, float)) or total <= 0:
        return None, "baseline.resources.elapsed_seconds.total is not a positive number"
    return float(total), None


#: Line the training code prints per fold it reuses instead of training. The
#: wording differs by framework only in case and prefix ("[fold 0] Already
#: completed, loading from disk" vs "[DTFD fold 0] already completed, loading
#: from disk"), so the match is case-insensitive on the shared tail.
CACHED_FOLD_MARKER = re.compile(r"already completed, loading from disk", re.IGNORECASE)
BASELINE_RUN_LOG = Path("baseline-execution") / "archive" / "run.log"


def _cached_fold_count(cell_root: Path) -> int:
    try:
        text = (cell_root / BASELINE_RUN_LOG).read_text(errors="replace")
    except OSError:
        return 0
    return len(CACHED_FOLD_MARKER.findall(text))


def _prediction_input(
    runtime: Path, cell_id: str, state: Mapping[str, object],
) -> tuple[float | None, int, str | None]:
    """The cell's five-fold baseline time in seconds, plus the cached-fold count.

    A baseline job that was interrupted and re-run loads its finished folds
    from the cache, and the ledger's ``elapsed_seconds.total`` then covers
    only the fresh folds (seen on tcga_luad kras hoptimus1 clam: 0.73 h
    recorded, 3.52 h when trained fresh). With ``k`` cached folds of ``n``
    the total is scaled by ``n / (n - k)``; a baseline with no fresh fold
    carries no timing at all and is refused.
    """
    e5_seconds, reason = _baseline_elapsed_seconds(state)
    if reason is not None:
        return None, 0, reason
    cached = _cached_fold_count(runtime / cell_id)
    if cached == 0:
        return e5_seconds, 0, None
    folds = (state.get("baseline") or {}).get("validation_folds") or []
    n_folds = len(folds) if isinstance(folds, list) and folds else 5
    if cached >= n_folds:
        return None, cached, (
            f"baseline elapsed covers no fresh fold ({cached} of {n_folds} cached); "
            "re-run the baseline to time it"
        )
    return e5_seconds * n_folds / (n_folds - cached), cached, None


def _shape_one_cell(
    runtime: Path, cell_id: str, e5_override: float | None = None,
) -> ShapeReport:
    state, reason = _read_campaign_state(runtime, cell_id)
    if reason is not None:
        return ShapeReport(cell_id=cell_id, shape=None, reason=reason)

    e5_seconds, cached, reason = _prediction_input(runtime, cell_id, state)
    source = "ledger-scaled" if cached else "ledger"
    if e5_override is not None:
        # The operator's time stands in only for a baseline that cannot time
        # itself; a baseline that carries a timing is never overridden.
        if reason is None:
            return ShapeReport(
                cell_id=cell_id, shape=None, cached_folds=cached,
                reason="the baseline carries its own timing; drop the operator-supplied time",
            )
        e5_seconds, reason, source = e5_override, None, "operator"
    if reason is not None:
        return ShapeReport(cell_id=cell_id, shape=None, reason=reason, cached_folds=cached)

    shape = choose_shape(e5_seconds)
    if shape is None:
        return ShapeReport(
            cell_id=cell_id, shape=None,
            reason="predicted job wall time exceeds every wall",
            baseline_elapsed_seconds=e5_seconds, cached_folds=cached,
            baseline_elapsed_source=source,
        )
    return ShapeReport(
        cell_id=cell_id, shape=shape, reason=None,
        baseline_elapsed_seconds=e5_seconds, cached_folds=cached,
        baseline_elapsed_source=source,
    )


def shape_cells(
    runtime: Path, cell_ids: Sequence[str], e5_override: float | None = None,
) -> Mapping[str, ShapeReport]:
    """Predict a SLURM shape for each cell root under ``runtime``.

    Never raises for one cell's bad or missing data; that cell's report
    carries a ``reason`` instead of a ``shape``. ``e5_override`` (seconds)
    is the operator-supplied five-fold time for a baseline whose retry
    loaded every fold from cache; it is refused for a baseline that carries
    its own timing.
    """
    return {
        cell_id: _shape_one_cell(runtime, cell_id, e5_override) for cell_id in cell_ids
    }


def _discover_cell_ids(runtime: Path) -> list[str]:
    return sorted(
        path.name for path in runtime.iterdir()
        if path.is_dir() and (path / "campaign_state.json").is_file()
    )


def _resolve_cell_ids(runtime: Path, cells_arg: str | None) -> list[str]:
    if cells_arg is None:
        return _discover_cell_ids(runtime)
    return [cell.strip() for cell in cells_arg.split(",") if cell.strip()]


def _first_unknown_cell(runtime: Path, cell_ids: Sequence[str]) -> str | None:
    for cell_id in cell_ids:
        if not (runtime / cell_id).is_dir():
            return cell_id
    return None


def _report_to_json(report: ShapeReport) -> dict:
    if report.shape is None:
        return {"unshaped": report.reason}
    return {**asdict(report.shape), "baseline_elapsed_seconds": report.baseline_elapsed_seconds,
            "cached_folds": report.cached_folds,
            "baseline_elapsed_source": report.baseline_elapsed_source}


def _table_row(runtime: Path, cell_id: str, report: ShapeReport) -> str:
    if report.shape is None:
        return f"{cell_id:<48} unshaped: {report.reason}"
    e5_hours = (report.baseline_elapsed_seconds or 0.0) / SECONDS_PER_HOUR
    shape = report.shape
    note = f"  (+{report.cached_folds} cached folds scaled)" if report.cached_folds else ""
    return (
        f"{cell_id:<48} {e5_hours:>8.3f} {shape.gpus:>3} {shape.wall_hours:>5} "
        f"{shape.predicted_hours:>12.3f}{note}"
    )


def _format_table(runtime: Path, reports: Mapping[str, ShapeReport]) -> str:
    header = f"{'cell':<48} {'e5_h':>8} {'g':>3} {'wall':>5} {'predicted_h':>12}"
    rows = [_table_row(runtime, cell_id, reports[cell_id]) for cell_id in sorted(reports)]
    return "\n".join([header, *rows])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runtime", default=None, help="campaign runtime directory")
    parser.add_argument(
        "--cells", default=None,
        help="comma-separated cell ids (default: auto-discover under --runtime)",
    )
    parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    parser.add_argument(
        "--finish", action="store_true",
        help="print the finish-only recovery lane shape as JSON and exit",
    )
    parser.add_argument(
        "--e5-seconds", type=float, default=None,
        help="operator-supplied five-fold baseline time for a baseline whose retry "
             "loaded every fold from cache (refused when the ledger carries a timing)",
    )
    parser.add_argument("--cell", default=None, help="one cell id, used together with --field")
    parser.add_argument(
        "--field", default=None,
        choices=("gpus", "wall_hours", "cpus", "mem_gb", "predicted_hours"),
        help="single Shape field to print for --cell, for shell consumption",
    )
    return parser


def _run_cell_field(
    runtime: Path, cell_id: str, field: str | None, e5_override: float | None,
) -> int:
    if not field:
        print("campaign_shape: --field is required with --cell", file=sys.stderr)
        return 2
    if not (runtime / cell_id).is_dir():
        print(f"campaign_shape: unknown cell id: {cell_id}", file=sys.stderr)
        return 2
    report = _shape_one_cell(runtime, cell_id, e5_override)
    if report.shape is None:
        print(
            f"campaign_shape: cell {cell_id} is unshaped: {report.reason}",
            file=sys.stderr,
        )
        return 2
    print(getattr(report.shape, field))
    return 0


def _run_sweep(
    runtime: Path, cells_arg: str | None, as_json: bool, e5_override: float | None,
) -> int:
    cell_ids = _resolve_cell_ids(runtime, cells_arg)
    unknown = _first_unknown_cell(runtime, cell_ids)
    if unknown is not None:
        print(f"campaign_shape: unknown cell id: {unknown}", file=sys.stderr)
        return 2

    reports = shape_cells(runtime, cell_ids, e5_override)
    if as_json:
        payload = {cell_id: _report_to_json(reports[cell_id]) for cell_id in reports}
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_format_table(runtime, reports))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.finish:
        print(json.dumps(asdict(finish_shape()), sort_keys=True))
        return 0
    if args.runtime is None:
        print("campaign_shape: --runtime is required", file=sys.stderr)
        return 2
    runtime = Path(args.runtime)
    if not runtime.is_dir():
        print(f"campaign_shape: invalid --runtime: {runtime}", file=sys.stderr)
        return 2

    if args.e5_seconds is not None and args.e5_seconds <= 0:
        print("campaign_shape: --e5-seconds must be positive", file=sys.stderr)
        return 2
    if args.e5_seconds is not None and args.cell is None and not args.cells:
        print("campaign_shape: --e5-seconds needs --cell or --cells", file=sys.stderr)
        return 2
    if args.cell is not None:
        return _run_cell_field(runtime, args.cell, args.field, args.e5_seconds)
    return _run_sweep(runtime, args.cells, args.json, args.e5_seconds)


if __name__ == "__main__":
    sys.exit(main())
