#!/usr/bin/env python3
"""Replay protocol-v4 checkpoint selection over finished protocol-v3 run logs.

v3 restored each fold's checkpoint from the epoch with the lowest validation
loss; v4 selects on the primary validation metric (``val_auc`` for
classification, ``val_c_index`` for survival) through the production
``SelectionTracker``. The ``[epoch k] ...`` lines of existing logs are fed to
that tracker, which stops receiving epochs where the v4 trainer would break
out of its loop (``early_stop`` under the arm's patience; CLAM only at epochs
> 50), and the epoch it names is compared with the logged ``[selected]`` one.
An nnMIL ``val_auc=0.0`` is that trainer's undefined-AUC sentinel, replayed as
NaN like every non-finite value. The logged trajectory is right-censored by the
v3 loss rule, so where the v4 stop never fires the epoch shift is a lower bound.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from autobench.campaign import PROTOCOL, STAGE_FOLDS
from autobench.pipeline.selection import SelectionTracker

EPOCH_LINE = re.compile(r"^\[epoch (\d+)\](?: (.*))?$")
SELECTED_LINE = re.compile(r"^\[selected\] epoch=(-?\d+) source=(best|final|untrained)$")
NAN = float("nan")


@dataclass(frozen=True)
class StopRule:
    patience: int
    floor: int = -1  # early_stop may end training only at epochs > floor


ARM_PATIENCE: Mapping[str, int] = {"clam": 20, "abmil": 20, "dtfd": 20, "titan": 10, "nnmil": 10}
#: CLAM classification's vendored stopper refuses to stop before epoch 50; the
#: CLAM survival adapter runs the plain patience rule like every other arm.
CLAM_CLASSIFICATION_FLOOR = 50
LOG_KINDS = (  # (glob under the cell root, fixed kind or None for the node directory name)
    ("baseline-execution/archive/run.log", "baseline"),
    ("baseline-reproduction/attempt-*/archive/run.log", "baseline-reproduction"),
    ("automil/orchestrator/archive/node_*/run.log", None),
)
COLUMNS = (
    "cell_id", "arm", "task_family", "kind", "fold", "epochs_run", "old_epoch", "new_epoch",
    "primary@old", "primary@new", "loss@old", "loss@new", "changed", "would_stop_epoch",
    "later_max_ignored", "n_epochs_at_new_max", "n_absent_metric_epochs",
    "n_zero_sentinel_epochs", "old_rule_recomputed", "log",
)
ARM_KIND_HEADER = (
    "arm", "kind", "folds", "changed", "median shift", "max shift", "mean primary new-old",
    "mean loss new-old", "absent epochs", "0.0 sentinels", "v4 stop unfired", "later max ignored", "tied max",
)
CELL_HEADER = (
    "cell", "baseline old", "baseline new", "reproduction old", "reproduction new",
    "best node (old rule)", "its old", "its new", "best node (new rule)", "its old", "its new",
)
CAVEATS = (
    "The replay covers only the epochs that ran under v3's loss-driven stopping. Where the v4 stop "
    "did not fire inside the log ('v4 stop unfired', would_stop_epoch blank) a v4 run would have "
    "trained on, so the epoch shift is a lower bound. ABMIL trains a fixed 20 epochs with patience "
    "20, so its unfired folds are complete trajectories, not censored ones.",
    "Per-node hparam overrides of patience, early stopping or the epoch budget are ignored: every fold "
    "is replayed with the arm default patience (or --patience-override). 'later max ignored' counts "
    "folds with a strictly higher primary value after the replayed v4 stop; a large "
    "--patience-override gives the no-stop replay.",
    "Held-out performance at the new epoch is unknowable offline: no per-epoch test predictions were logged.",
    "Ties are frequent on rank metrics (AUC and C-index on a 47-slide validation split move in coarse "
    "steps), so the earliest epoch of the maximum plateau is what the new rule picks; 'tied max' "
    "counts folds whose selected maximum recurs at a later epoch.",
)


class InputError(ValueError):
    """A cell or run.log outside the pinned contract; reported by path and skipped."""


@dataclass(frozen=True)
class Cell:
    root: Path
    cell_id: str
    arm: str
    task_family: str


@dataclass(frozen=True)
class Segment:
    epochs: tuple[tuple[int, Mapping[str, float]], ...]  # (epoch index, metrics) in logged order
    selected_epoch: int
    source: str


@dataclass(frozen=True)
class Observation:
    epoch: int
    value: float  # primary metric, NaN when absent
    loss: float  # raw val_loss, NaN when absent
    zero_sentinel: bool


@dataclass(frozen=True)
class Replay:
    best_epoch: int
    best_value: float | None
    stop_epoch: int | None
    n_seen: int  # observations fed to the tracker, the stop epoch included


@dataclass(frozen=True)
class Skipped:
    path: Path
    reason: str


def _parse_token(token: str) -> tuple[str, float]:
    key, sep, value = token.partition("=")
    try:
        if not sep or not key:
            raise ValueError
        return key, float(value)
    except ValueError:
        raise InputError(f"metric token {token!r} is not key=<number>") from None


def parse_segments(lines: Iterable[str]) -> tuple[Segment, ...]:
    """Split a run.log into fold segments, each closed by its [selected] line."""
    segments: list[Segment] = []
    pending: list[tuple[int, dict[str, float]]] = []
    for raw in lines:
        line = raw.rstrip("\r\n")
        if epoch_match := EPOCH_LINE.match(line):
            index = int(epoch_match.group(1))
            if pending and index <= pending[-1][0]:
                raise InputError(f"epoch {index} follows epoch {pending[-1][0]} without a [selected] line")
            pending.append((index, dict(_parse_token(t) for t in (epoch_match.group(2) or "").split())))
        elif selected_match := SELECTED_LINE.match(line):
            segments.append(Segment(tuple(pending), int(selected_match.group(1)), selected_match.group(2)))
            pending = []
    if pending:
        raise InputError(f"{len(pending)} epoch lines after the last [selected] line")
    return tuple(segments)


def load_cell(root: Path) -> Cell:
    meta_path = root / "automil" / "campaign_cell.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return Cell(root, str(meta["cell_id"]), str(meta["framework"]), str(meta["task_family"]))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise InputError(f"{meta_path}: {exc!r}") from None


def cell_logs(root: Path) -> tuple[tuple[str, Path], ...]:
    return tuple(
        (kind or path.parent.name, path) for pattern, kind in LOG_KINDS for path in sorted(root.glob(pattern))
    )


def stop_rule(arm: str, task_family: str, patience_override: int | None) -> StopRule:
    floor = CLAM_CLASSIFICATION_FLOOR if arm == "clam" and task_family != "survival" else -1
    if patience_override is not None:
        return StopRule(patience_override, floor)
    if arm not in ARM_PATIENCE:
        raise InputError(f"no default patience for arm {arm!r}; pass --patience-override")
    return StopRule(ARM_PATIENCE[arm], floor)


def observe_epoch(index: int, metrics: Mapping[str, float], key: str, arm: str) -> Observation:
    raw = metrics.get(key, NAN)
    sentinel = arm == "nnmil" and key == "val_auc" and raw == 0.0
    value = NAN if sentinel or not math.isfinite(raw) else raw
    return Observation(index, value, metrics.get("val_loss", NAN), sentinel)


def replay(observations: Sequence[Observation], rule: StopRule) -> Replay:
    """Feed the production tracker until the v4 trainer would have stopped."""
    tracker = SelectionTracker(rule.patience)
    for n_seen, obs in enumerate(observations, start=1):
        tracker.observe(obs.epoch, obs.value)
        if tracker.early_stop and obs.epoch > rule.floor:
            return Replay(tracker.best_epoch, tracker.best_value, obs.epoch, n_seen)
    return Replay(tracker.best_epoch, tracker.best_value, None, len(observations))


def first_loss_minimum(observations: Sequence[Observation]) -> int:
    finite = [(obs.loss, obs.epoch) for obs in observations if math.isfinite(obs.loss)]
    return min(finite)[1] if finite else -1


def fold_row(cell: Cell, kind: str, log: Path, fold: int, segment: Segment, rule: StopRule) -> dict:
    key = "val_c_index" if cell.task_family == "survival" else "val_auc"
    observations = tuple(observe_epoch(index, metrics, key, cell.arm) for index, metrics in segment.epochs)
    by_epoch = {obs.epoch: obs for obs in observations}
    old = segment.selected_epoch
    if old >= 0 and old not in by_epoch:
        raise InputError(f"fold {fold}: [selected] epoch={old} is not a validated epoch")
    result = replay(observations, rule)
    best, new = result.best_value, result.best_epoch
    seen, later = observations[: result.n_seen], observations[result.n_seen:]

    def at(epoch: int, attr: str) -> float:
        return getattr(by_epoch[epoch], attr) if epoch in by_epoch else NAN

    return {
        "cell_id": cell.cell_id, "arm": cell.arm, "task_family": cell.task_family, "kind": kind,
        "fold": fold, "epochs_run": len(observations), "old_epoch": old, "new_epoch": new,
        "primary@old": at(old, "value"), "primary@new": at(new, "value"),
        "loss@old": at(old, "loss"), "loss@new": at(new, "loss"),
        "changed": new != old, "would_stop_epoch": result.stop_epoch,
        "later_max_ignored": any(math.isfinite(o.value) and (best is None or o.value > best) for o in later),
        "n_epochs_at_new_max": sum(1 for o in seen if best is not None and o.value == best),
        "n_absent_metric_epochs": sum(1 for o in observations if math.isnan(o.value)),
        "n_zero_sentinel_epochs": sum(1 for o in observations if o.zero_sentinel),
        "old_rule_recomputed": first_loss_minimum(observations),
        "log": str(log.relative_to(cell.root)),
    }


def replay_log(cell: Cell, kind: str, path: Path, rule: StopRule) -> tuple[dict, ...]:
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            segments = parse_segments(handle)
    except OSError as exc:
        raise InputError(f"cannot read: {exc}") from None
    return tuple(fold_row(cell, kind, path, fold, seg, rule) for fold, seg in enumerate(segments))


def replay_cells(roots: Sequence[Path], patience_override: int | None) -> tuple[tuple[dict, ...], tuple[Skipped, ...]]:
    rows: list[dict] = []
    skipped: list[Skipped] = []
    for root in roots:
        try:
            cell = load_cell(root)
            rule = stop_rule(cell.arm, cell.task_family, patience_override)
        except InputError as exc:
            skipped.append(Skipped(root, str(exc)))
            continue
        for kind, path in cell_logs(root):
            try:
                fold_rows = replay_log(cell, kind, path, rule)
            except InputError as exc:
                skipped.append(Skipped(path, str(exc)))
                continue
            if fold_rows:
                rows.extend(fold_rows)
            else:
                skipped.append(Skipped(path, "no fold segments"))
    return tuple(rows), tuple(skipped)


def _fmt(value: object, for_csv: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, bool):
        return ("true" if value else "false") if for_csv else ("yes" if value else "no")
    if isinstance(value, float):
        return repr(value) if for_csv else f"{value:.4f}"
    return str(value)


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        writer.writerows([_fmt(row[column], for_csv=True) for column in COLUMNS] for row in rows)


def md_table(header: Sequence[str], body: Iterable[Sequence[object]]) -> list[str]:
    rows = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return rows + ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in body]


def kind_group(kind: str) -> str:
    return "node" if kind.startswith("node_") else kind


def _mean_delta(rows: Sequence[dict], column: str) -> float:
    deltas = [row[f"{column}@new"] - row[f"{column}@old"] for row in rows]
    finite = [d for d in deltas if math.isfinite(d)]
    return statistics.fmean(finite) if finite else NAN


def group_summary(arm: str, kind: str, rows: Sequence[dict]) -> tuple[object, ...]:
    scored = [row for row in rows if row["old_epoch"] >= 0 and row["new_epoch"] >= 0]
    shifts = [row["new_epoch"] - row["old_epoch"] for row in scored]
    return (
        arm, kind, len(rows), sum(1 for row in rows if row["changed"]) / len(rows),
        statistics.median(shifts) if shifts else NAN, max(shifts) if shifts else NAN,
        _mean_delta(scored, "primary"), _mean_delta(scored, "loss"),
        sum(row["n_absent_metric_epochs"] for row in rows), sum(row["n_zero_sentinel_epochs"] for row in rows),
        sum(1 for row in rows if row["would_stop_epoch"] is None),
        sum(1 for row in rows if row["later_max_ignored"]),
        sum(1 for row in rows if row["n_epochs_at_new_max"] > 1),
    )


def _grouped(rows: Sequence[dict], key) -> dict:
    return {k: [row for row in rows if key(row) == k] for k in sorted({key(row) for row in rows})}


def arm_kind_table(rows: Sequence[dict]) -> list[str]:
    groups = _grouped(rows, lambda row: (row["arm"], kind_group(row["kind"])))
    return md_table(ARM_KIND_HEADER, (group_summary(arm, kind, g) for (arm, kind), g in groups.items()))


def fold_mean(rows: Sequence[dict], column: str, n_required: int) -> float:
    values = [row[column] for row in rows]
    complete = values and len(values) == n_required and all(math.isfinite(v) for v in values)
    return statistics.fmean(values) if complete else NAN


def _best_node(means: Mapping[str, tuple[float, float]], index: int) -> tuple[object, ...]:
    """(node, its old-rule mean, its new-rule mean) for the node that ranks first on ``index``."""
    eligible = {node: pair for node, pair in means.items() if math.isfinite(pair[index])}
    if not eligible:
        return ("", NAN, NAN)
    node = max(eligible, key=lambda k: eligible[k][index])
    return (node, *eligible[node])


#: Fold segments a complete run of each kind carries: the baseline registers
#: every split fold; a reproduction or a discovery attempt runs the discovery
#: folds. A run with fewer segments was killed and must not rank as a winner.
FOLDS_REQUIRED = {
    "baseline": PROTOCOL["split_folds"],
    "baseline-reproduction": len(STAGE_FOLDS["discovery"]),
    "node": len(STAGE_FOLDS["discovery"]),
}


def _latest_complete_reproduction(rows: Sequence[dict]) -> list[dict]:
    """The fold rows of the latest reproduction attempt that ran every
    discovery fold: attempts are separate runs under
    ``baseline-reproduction/attempt-N/`` and must never be pooled."""
    complete = [
        group for group in _grouped(rows, lambda row: row["log"]).values()
        if len(group) == FOLDS_REQUIRED["baseline-reproduction"]
    ]
    if not complete:
        return []
    return max(complete, key=lambda group: int(re.search(r"attempt-(\d+)", group[0]["log"]).group(1)))


def cell_summary(cell_id: str, rows: Sequence[dict]) -> tuple[object, ...]:
    baseline = [row for row in rows if row["kind"] == "baseline"]
    repro = _latest_complete_reproduction([row for row in rows if row["kind"] == "baseline-reproduction"])
    nodes = _grouped([row for row in rows if kind_group(row["kind"]) == "node"], lambda row: row["kind"])
    n_folds = FOLDS_REQUIRED["node"]
    means = {node: (fold_mean(r, "primary@old", n_folds), fold_mean(r, "primary@new", n_folds)) for node, r in nodes.items()}
    return (
        cell_id,
        fold_mean(baseline, "primary@old", FOLDS_REQUIRED["baseline"]),
        fold_mean(baseline, "primary@new", FOLDS_REQUIRED["baseline"]),
        fold_mean(repro, "primary@old", FOLDS_REQUIRED["baseline-reproduction"]),
        fold_mean(repro, "primary@new", FOLDS_REQUIRED["baseline-reproduction"]),
        *_best_node(means, 0), *_best_node(means, 1),
    )


def cell_table(rows: Sequence[dict]) -> list[str]:
    return md_table(CELL_HEADER, (cell_summary(c, g) for c, g in _grouped(rows, lambda row: row["cell_id"]).items()))


def self_check_lines(rows: Sequence[dict]) -> list[str]:
    off = [row for row in rows if row["old_rule_recomputed"] != row["old_epoch"]]
    return [f"Folds whose logged [selected] epoch differs from the first strict val_loss minimum: {len(off)}"] + [
        f"- {r['cell_id']} {r['kind']} fold {r['fold']}: logged {r['old_epoch']}, recomputed {r['old_rule_recomputed']}"
        for r in off[:20]
    ]


def build_report(rows: Sequence[dict], skipped: Sequence[Skipped], patience_override: int | None) -> str:
    defaults = ", ".join(f"{arm} {patience}" for arm, patience in ARM_PATIENCE.items()) + (
        f"; clam classification stops only after epoch {CLAM_CLASSIFICATION_FLOOR}"
    )
    patience = f"override {patience_override}" if patience_override is not None else f"arm defaults: {defaults}"
    lines = [
        "# Checkpoint-selection replay: protocol v3 (val_loss minimum) -> v4 (primary-metric maximum)", "",
        f"Fold segments: {len(rows)} from {len({row['cell_id'] for row in rows})} cells; patience {patience}.", "",
        "## Per arm x kind", "", *arm_kind_table(rows), "",
        "## Per cell: fold-mean primary under each rule", "", *cell_table(rows), "",
        "## Self-check", "", *self_check_lines(rows), "",
        "## Skipped logs", "", *([f"- {s.path}: {s.reason}" for s in skipped] or ["- none"]), "",
        "## Caveats", "", *[f"- {caveat}" for caveat in CAVEATS], "",
    ]
    return "\n".join(lines)


def expand_cells(patterns: Sequence[str]) -> tuple[Path, ...]:
    matches = (Path(m) for pattern in patterns for m in sorted(glob.glob(pattern)))
    return tuple(path for path in matches if path.is_dir())


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", nargs="+", required=True, help="cell roots or globs (quote a glob)")
    parser.add_argument("--out", type=Path, required=True, help="per-fold CSV to write")
    parser.add_argument("--report", type=Path, help="Markdown summary to write (always printed)")
    parser.add_argument("--patience-override", type=int, help="replay every arm with this patience")
    args = parser.parse_args(argv)
    if args.patience_override is not None and args.patience_override < 1:
        parser.error("--patience-override must be a positive integer")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    roots = expand_cells(args.cells)
    if not roots:
        print(f"no cell roots match {args.cells}", file=sys.stderr)
        return 2
    rows, skipped = replay_cells(roots, args.patience_override)
    if not rows:
        print("no fold segments parsed; " + "; ".join(f"{s.path}: {s.reason}" for s in skipped), file=sys.stderr)
        return 2
    write_csv(args.out, rows)
    report = build_report(rows, skipped, args.patience_override)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
