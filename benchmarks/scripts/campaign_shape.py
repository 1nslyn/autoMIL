#!/usr/bin/env python3
"""Predict discovery wall time and pick a SLURM job shape per campaign cell.

For each cell root under a campaign runtime directory, this script reads the
registered baseline's five-fold elapsed time from ``campaign_state.json`` and
predicts how long the discovery stage will take under a candidate SLURM
allocation (see ``predict_hours`` for the formula: one serial gate attempt,
30 packed discovery attempts, 10 packed promotion candidates, plus a fixed
setup/teardown overhead). It then picks the first candidate shape — tried
wall-clock first, then GPU count — whose predicted time fits inside
``FIT_FRACTION`` of that wall clock.

Cells without a registered baseline (``baseline`` is ``None``), with
malformed or missing state, or whose predicted time exceeds every candidate
shape are reported as unshaped with a reason; a bad cell never crashes the
sweep over the rest.

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
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

# Per-GPU concurrent-job cap, frozen in the cell config as
# orchestrator.max_concurrent_per_gpu: 4; efficiency is the derated packing
# factor observed in the aihub canary logs (GPU-attached job hours vs actual
# wall-clock hours for a packed batch of attempts).
FIT_FRACTION = 0.85
CAP_PER_GPU = 4
EFFICIENCY = 0.8
OVERHEAD_H = 2.0
CORES_PER_GPU = 12
MEM_GB_PER_GPU = 128
GPU_OPTIONS = (1, 2, 4)
WALL_OPTIONS_H = (12, 24)

# Stage-fold split (autobench.campaign.STAGE_FOLDS): 3 of the 5 baseline
# folds are re-run per discovery attempt, 2 per promotion candidate.
DISCOVERY_FOLDS = 3
PROMOTION_FOLDS = 2
TOTAL_FOLDS = 5
DISCOVERY_ATTEMPTS = 30
PROMOTION_CANDIDATES = 10

SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class Shape:
    """A concrete SLURM job shape and its predicted discovery wall time."""

    gpus: int
    wall_hours: int
    cpus: int
    mem_gb: int
    whole_node: bool
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


def predict_hours(e5_seconds: float, gpus: int) -> float:
    """Predict discovery-stage wall time (hours) for a ``gpus``-wide job.

    ``e5_seconds`` is the 5-fold baseline's total elapsed time, as recorded
    in ``baseline.resources.elapsed_seconds.total``. One serial gate attempt
    (``attempt``) is followed by 30 discovery attempts and 10 promotion
    candidates, each packed ``CAP_PER_GPU * gpus`` wide at ``EFFICIENCY``
    derating, plus a fixed overhead for setup/teardown.
    """
    e5_hours = e5_seconds / SECONDS_PER_HOUR
    attempt = e5_hours * (DISCOVERY_FOLDS / TOTAL_FOLDS)
    promotion = e5_hours * (PROMOTION_FOLDS / TOTAL_FOLDS)
    capacity = CAP_PER_GPU * gpus * EFFICIENCY
    return (
        attempt
        + DISCOVERY_ATTEMPTS * attempt / capacity
        + PROMOTION_CANDIDATES * promotion / capacity
        + OVERHEAD_H
    )


PREFERENCES = ("cheap", "fast")


def candidate_shapes(prefer: str = "cheap") -> tuple[tuple[int, int], ...]:
    """Ordered ``(gpus, wall_hours)`` candidates for a preference.

    ``cheap`` (default) minimizes GPU-hours: the smallest GPU count that fits
    either wall wins, so (1,12), (1,24), (2,12), (2,24), (4,12), (4,24).
    ``fast`` minimizes wall time: (1,12), (2,12), (4,12), (1,24), (2,24),
    (4,24). Fair-share bills allocated GPU-minutes, which is why cheap is the
    default; fast trades GPU-hours for the shorter-queue 12 h tier.
    """
    if prefer not in PREFERENCES:
        raise ValueError(f"unknown preference {prefer!r}; expected one of {PREFERENCES}")
    if prefer == "cheap":
        return tuple((g, w) for g in GPU_OPTIONS for w in WALL_OPTIONS_H)
    return tuple((g, w) for w in WALL_OPTIONS_H for g in GPU_OPTIONS)


def finish_shape() -> Shape:
    """The finish-only recovery lane: promotion of at most ten candidates on
    one GPU fits the shorter wall for every roster cell (worst ~9 h)."""
    gpus, wall_hours = GPU_OPTIONS[0], WALL_OPTIONS_H[0]
    return Shape(
        gpus=gpus, wall_hours=wall_hours, cpus=CORES_PER_GPU * gpus,
        mem_gb=MEM_GB_PER_GPU * gpus, whole_node=False, predicted_hours=0.0,
    )


def choose_shape(e5_seconds: float, prefer: str = "cheap") -> Shape | None:
    """Pick the first candidate shape (see ``candidate_shapes``) whose
    predicted time fits ``FIT_FRACTION`` of its wall clock; ``None`` if none.
    """
    for gpus, wall_hours in candidate_shapes(prefer):
        predicted = predict_hours(e5_seconds, gpus)
        if predicted <= FIT_FRACTION * wall_hours:
            return Shape(
                gpus=gpus,
                wall_hours=wall_hours,
                cpus=CORES_PER_GPU * gpus,
                mem_gb=MEM_GB_PER_GPU * gpus,
                whole_node=(gpus == max(GPU_OPTIONS)),
                predicted_hours=predicted,
            )
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


#: Line the training script prints per fold it reuses instead of training.
CACHED_FOLD_MARKER = "Already completed, loading from disk"
BASELINE_RUN_LOG = Path("baseline-execution") / "archive" / "run.log"


def _cached_fold_count(cell_root: Path) -> int:
    try:
        text = (cell_root / BASELINE_RUN_LOG).read_text(errors="replace")
    except OSError:
        return 0
    return text.count(CACHED_FOLD_MARKER)


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


def _shape_one_cell(runtime: Path, cell_id: str, prefer: str = "cheap") -> ShapeReport:
    state, reason = _read_campaign_state(runtime, cell_id)
    if reason is not None:
        return ShapeReport(cell_id=cell_id, shape=None, reason=reason)

    e5_seconds, cached, reason = _prediction_input(runtime, cell_id, state)
    if reason is not None:
        return ShapeReport(cell_id=cell_id, shape=None, reason=reason, cached_folds=cached)

    shape = choose_shape(e5_seconds, prefer)
    if shape is None:
        return ShapeReport(
            cell_id=cell_id, shape=None,
            reason="predicted discovery wall time exceeds every candidate shape",
            baseline_elapsed_seconds=e5_seconds, cached_folds=cached,
        )
    return ShapeReport(
        cell_id=cell_id, shape=shape, reason=None,
        baseline_elapsed_seconds=e5_seconds, cached_folds=cached,
    )


def shape_cells(
    runtime: Path, cell_ids: Sequence[str], prefer: str = "cheap",
) -> Mapping[str, ShapeReport]:
    """Predict a SLURM shape for each cell root under ``runtime``.

    Never raises for one cell's bad or missing data; that cell's report
    carries a ``reason`` instead of a ``shape``.
    """
    return {cell_id: _shape_one_cell(runtime, cell_id, prefer) for cell_id in cell_ids}


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
            "cached_folds": report.cached_folds}


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
        "--prefer", default="cheap", choices=PREFERENCES,
        help="cheap = fewest GPU-hours (default); fast = shortest wall first",
    )
    parser.add_argument("--cell", default=None, help="one cell id, used together with --field")
    parser.add_argument(
        "--field", default=None,
        choices=("gpus", "wall_hours", "cpus", "mem_gb", "whole_node", "predicted_hours"),
        help="single Shape field to print for --cell, for shell consumption",
    )
    return parser


def _run_cell_field(runtime: Path, cell_id: str, field: str | None, prefer: str) -> int:
    if not field:
        print("campaign_shape: --field is required with --cell", file=sys.stderr)
        return 2
    if not (runtime / cell_id).is_dir():
        print(f"campaign_shape: unknown cell id: {cell_id}", file=sys.stderr)
        return 2
    report = _shape_one_cell(runtime, cell_id, prefer)
    if report.shape is None:
        print(
            f"campaign_shape: cell {cell_id} is unshaped: {report.reason}",
            file=sys.stderr,
        )
        return 2
    print(getattr(report.shape, field))
    return 0


def _run_sweep(runtime: Path, cells_arg: str | None, as_json: bool, prefer: str) -> int:
    cell_ids = _resolve_cell_ids(runtime, cells_arg)
    unknown = _first_unknown_cell(runtime, cell_ids)
    if unknown is not None:
        print(f"campaign_shape: unknown cell id: {unknown}", file=sys.stderr)
        return 2

    reports = shape_cells(runtime, cell_ids, prefer)
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

    if args.cell is not None:
        return _run_cell_field(runtime, args.cell, args.field, args.prefer)
    return _run_sweep(runtime, args.cells, args.json, args.prefer)


if __name__ == "__main__":
    sys.exit(main())
