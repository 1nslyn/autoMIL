#!/usr/bin/env python3
"""Search a dev cell with Optuna's TPE sampler: the menu-AutoML arm.

    python benchmarks/scripts/dev_optuna.py run <cell_dir> --n-trials 60 --in-flight 4
    python benchmarks/scripts/dev_optuna.py status <cell_dir>

Every trial is one attempt of the cell, proposed and submitted through the batch
driver's path (``dev_loop.drive_entry``), so the framework's admissibility, purity
and policy-smoke checks, the orchestrator, the readout and its winner verdict treat
it exactly as they treat an agent's attempt. The study (a journal file under
``automil/optuna/``) maximizes the attempt's primary value (the mean smoothed
validation AUC over the root's folds) and carries the cell's companion guard as a
constraint, so the sampler steers away from attempts the guard would exclude.

The driver asks the study for a trial, draws the attempt from the menu
(``dev_optuna_space``), submits it, and tells the study the score once the
orchestrator has it; at most ``--in-flight`` attempts run at once. A restarted
driver resumes: a trial whose attempt is still running is re-attached to it (the
study re-issues its draw through ``enqueue_trial``), and a trial whose submit never
completed is marked failed. ``--n-trials`` counts submitted attempts, crashed ones
included: a crash is the search's own cost, as it is for an agent.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import optuna
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import TPESampler
from optuna.trial import Trial, TrialState

STUDY_DIRNAME = "optuna"
DEFAULT_STUDY = "tpe"
ROOT_NAME = "baseline"
ROOT_DESC = "native recipe (the root)"
AXIS = "optuna"
KIND = "hp"
N_STARTUP_TRIALS = 10
GUARD_KEY = "guard"
#: The constraint a scored trial carries: at most 0 is feasible (the guard passed).
GUARD_PASS, GUARD_FAIL = -1.0, 1.0
NODE_ATTR, NAME_ATTR = "node_id", "name"
WAITING_STATUSES = frozenset({"pending", "running"})
DEFAULT_BUDGETS = (30,)


def _load_script(stem: str):
    """``benchmarks/scripts/<stem>.py`` by path (scripts are not a package), cached by name."""
    cached = sys.modules.get(stem)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(stem, Path(__file__).resolve().with_name(f"{stem}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


_LOOP = _load_script("dev_loop")
_SPACE = _load_script("dev_optuna_space")
Entry = _LOOP.Entry
COMPLETE_STATUSES = _LOOP.COMPLETE_STATUSES


class DevOptunaError(RuntimeError):
    """A cell, study or attempt the driver cannot go on from."""


@dataclass(frozen=True)
class Cell:
    """The cell as the driver sees it: the CLI binds a real one, tests bind fakes."""

    arm: str
    submit: Callable[[Entry], str]                               # propose + submit; the node id
    nodes: Callable[[], Mapping[str, Any]]                       # graph.json's nodes; {} before the first submit
    judge: Callable[[Mapping[str, Any]], tuple[float, float]]    # a complete node -> (value, guard constraint)


@dataclass(frozen=True)
class Settings:
    n_trials: int
    in_flight: int
    positive_share: float | None = None
    poll_seconds: float = 60.0

    def __post_init__(self) -> None:
        for name in ("n_trials", "in_flight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise DevOptunaError(f"{name} must be a positive integer, got {value!r}")


#: Trial number -> (the live trial, the node it runs as).
Running = Mapping[int, tuple[Trial, str]]
Log = Callable[[str], None]


def trial_name(number: int) -> str:
    return f"t{number:03d}"


def root_entry() -> Entry:
    return Entry(ROOT_NAME, None, ROOT_DESC, None, None, None, {})


def attempt_entry(name: str, sample: Any) -> Entry:
    return Entry(name, ROOT_NAME, _SPACE.describe(sample), AXIS, KIND, dict(sample.constants), dict(sample.hparams))


def node_state(node: Mapping[str, Any]) -> str:
    """``complete`` (scored), ``waiting`` (queued or running) or ``failed`` (crash, oom, timeout, ...)."""
    status = node.get("status")
    if status in COMPLETE_STATUSES:
        return "complete"
    return "waiting" if status in WAITING_STATUSES else "failed"


def _stamp() -> str:
    return datetime.now().strftime("%m-%d %H:%M")


# ---------------------------------------------------------------------------
# The study
# ---------------------------------------------------------------------------
def study_path(adir: Path, study_name: str = DEFAULT_STUDY) -> Path:
    return Path(adir) / STUDY_DIRNAME / f"{study_name}.journal"


def open_study(adir: Path, study_name: str = DEFAULT_STUDY, *, seed: int = 0) -> optuna.Study:
    """The cell's study: created on first use, reopened from its journal afterwards."""
    path = study_path(adir, study_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(str(path)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)  # multivariate TPE and the constant liar
        sampler = TPESampler(seed=seed, n_startup_trials=N_STARTUP_TRIALS, multivariate=True, constant_liar=True)
    return optuna.create_study(study_name=study_name, storage=storage, direction="maximize",
                               sampler=sampler, load_if_exists=True)


def submitted_nodes(study: optuna.Study) -> frozenset[str]:
    """The attempts the study has submitted (a resumed trial shares its predecessor's node)."""
    return frozenset(trial.user_attrs[NODE_ATTR] for trial in study.get_trials(deepcopy=False)
                     if NODE_ATTR in trial.user_attrs)


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------
def _root(nodes: Mapping[str, Any]) -> Mapping[str, Any] | None:
    return next((node for _, node in sorted(nodes.items()) if node.get("parent_id") is None), None)


def ensure_root(cell: Cell, *, sleep: Callable[[], None], log: Log) -> None:
    """Submit the native baseline unless the graph has one, then wait until it is scored."""
    if _root(cell.nodes()) is None:
        log(f"{_stamp()} root submitted: {cell.submit(root_entry())}")
    while True:
        root = _root(cell.nodes())
        if root is None:
            raise DevOptunaError("the root is no longer in the graph")
        state = node_state(root)
        if state == "complete":
            return
        if state == "failed":
            raise DevOptunaError(f"the root {root.get('id')} ended {root.get('status')!r}: nothing to search against")
        sleep()


def resume(study: optuna.Study, cell: Cell, settings: Settings, *, log: Log) -> dict[int, tuple[Trial, str]]:
    """Re-attach the trials a previous driver left running; fail the ones it never submitted."""
    running: dict[int, tuple[Trial, str]] = {}
    for frozen in study.get_trials(states=(TrialState.RUNNING,)):
        study.tell(frozen.number, state=TrialState.FAIL)  # its live trial died with the previous driver
        node_id, name = frozen.user_attrs.get(NODE_ATTR), frozen.user_attrs.get(NAME_ATTR)
        if node_id is None or name is None:
            log(f"{_stamp()} trial {frozen.number} never reached the cell: marked failed")
            continue
        study.enqueue_trial(frozen.params, user_attrs={NODE_ATTR: node_id, NAME_ATTR: name,
                                                       "resumed_from": frozen.number})
        trial = study.ask()
        _SPACE.sample(trial, cell.arm, positive_share=settings.positive_share)  # realizes the enqueued draw
        running[trial.number] = (trial, node_id)
        log(f"{_stamp()} resumed {name} on {node_id} as trial {trial.number}")
    return running


def _settle(study: optuna.Study, cell: Cell, running: Running, *, log: Log) -> dict[int, tuple[Trial, str]]:
    """Tell the study every attempt the orchestrator has finished; the rest stay running."""
    nodes = cell.nodes()
    still: dict[int, tuple[Trial, str]] = {}
    for number, (trial, node_id) in running.items():
        node = nodes.get(node_id)
        if node is None:
            raise DevOptunaError(f"{node_id} (trial {number}) is no longer in the graph")
        name, state = trial.user_attrs[NAME_ATTR], node_state(node)
        if state == "waiting":
            still[number] = (trial, node_id)
        elif state == "failed":
            study.tell(trial, state=TrialState.FAIL)
            log(f"{_stamp()} {name} {node_id} ended {node.get('status')!r}: failed")
        else:
            value, constraint = cell.judge(node)
            standing = "pass" if constraint <= 0 else "fail"
            trial.set_constraint(GUARD_KEY, constraint)
            trial.set_user_attr(GUARD_KEY, standing)
            study.tell(trial, value)
            log(f"{_stamp()} {name} {node_id} scored {value:.4f}, guard {standing}")
    return still


def _ask(study: optuna.Study, cell: Cell, settings: Settings, running: Running, *, log: Log) -> dict[int, tuple[Trial, str]]:
    """Fill the in-flight slots with fresh draws, up to the budget."""
    filled = dict(running)
    while len(filled) < settings.in_flight and len(submitted_nodes(study)) < settings.n_trials:
        trial = study.ask()
        sample = _SPACE.sample(trial, cell.arm, positive_share=settings.positive_share)
        name = trial_name(trial.number)
        trial.set_user_attr(NAME_ATTR, name)
        try:
            node_id = cell.submit(attempt_entry(name, sample))
        except Exception:
            study.tell(trial, state=TrialState.FAIL)
            raise
        trial.set_user_attr(NODE_ATTR, node_id)
        filled[trial.number] = (trial, node_id)
        log(f"{_stamp()} asked {name} -> {node_id}: {_SPACE.describe(sample)}")
    return filled


def run(study: optuna.Study, cell: Cell, settings: Settings, *,
        sleep: Callable[[float], None] = time.sleep, log: Log = print) -> None:
    """Drive the cell to ``settings.n_trials`` submitted attempts and wait for the last one."""
    def wait() -> None:
        sleep(settings.poll_seconds)

    ensure_root(cell, sleep=wait, log=log)
    running = resume(study, cell, settings, log=log)
    while True:
        running = _ask(study, cell, settings, _settle(study, cell, running, log=log), log=log)
        if not running and len(submitted_nodes(study)) >= settings.n_trials:
            break
        wait()
    log(f"{_stamp()} done: {len(submitted_nodes(study))} attempts submitted and scored")


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrialRow:
    number: int
    name: str
    node_id: str
    state: str
    value: float | None
    guard: str | None
    draw: str


def trial_rows(study: optuna.Study) -> list[TrialRow]:
    """One row per submitted attempt (the last trial attached to it), in submission order."""
    latest: dict[str, Any] = {}
    for trial in study.get_trials(deepcopy=False):
        if NODE_ATTR in trial.user_attrs:
            latest[trial.user_attrs[NODE_ATTR]] = trial
    return [
        TrialRow(trial.number, trial.user_attrs[NAME_ATTR], node_id, trial.state.name,
                 trial.values[0] if trial.values else None, trial.user_attrs.get(GUARD_KEY),
                 ", ".join(f"{key}={value:.3g}" if isinstance(value, float) else f"{key}={value!r}"
                           for key, value in trial.params.items()))
        for node_id, trial in sorted(latest.items())
    ]


def best_within(rows: Sequence[TrialRow], budget: int) -> TrialRow | None:
    """The best scored, guard-passing attempt among the first ``budget`` submitted."""
    eligible = [row for row in rows[:budget] if row.value is not None and row.guard == "pass"]
    return max(eligible, key=lambda row: row.value) if eligible else None


def format_status(study: optuna.Study, *, budgets: Sequence[int] = DEFAULT_BUDGETS) -> str:
    rows = trial_rows(study)
    lines = [f"study {study.study_name}: {len(rows)} attempts submitted", ""]
    lines.extend(
        f"{row.name}  {row.node_id}  {row.state:<8} value {'-' if row.value is None else f'{row.value:.4f}'}  "
        f"guard {row.guard or '-'}  {row.draw}"
        for row in rows
    )
    for budget in (*budgets, len(rows)):
        best = best_within(rows, budget)
        lines.append(f"best of the first {budget}: " + ("none scored and guard-passing" if best is None
                                                        else f"{best.name} {best.node_id} value {best.value:.4f}"))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# A real cell
# ---------------------------------------------------------------------------
def _finite_value(node: Mapping[str, Any]) -> float:
    value = node.get("primary_value")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise DevOptunaError(f"{node.get('id')} is complete but records no finite primary_value: {value!r}")
    return float(value)


def bind_cell(cell_dir: Path) -> tuple[Cell, Path]:
    """``(the cell as the driver sees it, its automil dir)`` for a dev cell on disk."""
    cell_dir = Path(cell_dir).resolve()
    adir, record = _LOOP.load_cell(cell_dir)
    config = _LOOP.load_config(adir)

    def nodes() -> Mapping[str, Any]:
        return _LOOP.load_graph(adir)["nodes"] if (adir / "graph.json").is_file() else {}

    def submit(entry: Entry) -> str:
        try:
            outcome = _LOOP.drive_entry(cell_dir, entry)
        except _LOOP.DevLoopError as exc:
            raise DevOptunaError(f"the cell refused {entry.name!r}: {exc}") from exc
        if outcome.action == "skip" or outcome.node_id is None:
            raise DevOptunaError(f"the cell did not take {entry.name!r}: {outcome.line}")
        return outcome.node_id

    def judge(node: Mapping[str, Any]) -> tuple[float, float]:
        standing = _LOOP.guard_standing(config, nodes(), node)
        failed = standing is not None and standing["status"] == "fail"
        return _finite_value(node), (GUARD_FAIL if failed else GUARD_PASS)

    return Cell(record["arm"], submit, nodes, judge), adir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _emit(line: str) -> None:
    print(line, flush=True)


def cmd_run(args: argparse.Namespace) -> None:
    cell, adir = bind_cell(Path(args.cell_dir))
    settings = Settings(args.n_trials, args.in_flight, args.positive_share, args.poll_seconds)
    study = open_study(adir, args.study, seed=args.seed)
    run(study, cell, settings, log=_emit)


def cmd_status(args: argparse.Namespace) -> None:
    adir, _ = _LOOP.load_cell(Path(args.cell_dir).resolve())
    if not study_path(adir, args.study).is_file():
        raise DevOptunaError(f"no study {args.study!r} under {adir / STUDY_DIRNAME}")
    print(format_status(open_study(adir, args.study), budgets=tuple(args.budget or DEFAULT_BUDGETS)))


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dev_optuna.py", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run_cmd = sub.add_parser("run", help="search the cell with TPE until the budget is spent")
    run_cmd.add_argument("cell_dir")
    run_cmd.add_argument("--n-trials", type=_positive_int, default=60, help="attempts to submit (default 60)")
    run_cmd.add_argument("--in-flight", type=_positive_int, default=4, help="attempts running at once (default 4)")
    run_cmd.add_argument("--positive-share", type=float, default=None,
                         help="share of label 1 among the training bags (two-class cells): offers class balancing")
    run_cmd.add_argument("--study", default=DEFAULT_STUDY)
    run_cmd.add_argument("--seed", type=int, default=0, help="the sampler's seed")
    run_cmd.add_argument("--poll-seconds", type=float, default=60.0)
    run_cmd.set_defaults(func=cmd_run)
    status = sub.add_parser("status", help="list the study's attempts and the best within each budget")
    status.add_argument("cell_dir")
    status.add_argument("--study", default=DEFAULT_STUDY)
    status.add_argument("--budget", type=_positive_int, action="append", help="a budget to report (repeatable)")
    status.set_defaults(func=cmd_status)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        args.func(args)
    except (DevOptunaError, _LOOP.DevLoopError, ValueError) as exc:
        print(f"dev_optuna: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
