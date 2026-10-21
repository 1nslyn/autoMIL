#!/usr/bin/env python
"""Materialize, read out and drive development cells.

A development cell is a plain autoMIL project (no ``campaign:`` block, no
``cap.phasing``, no ``activity`` block) at
``benchmarks/experiments/dev/<dataset>__<task>__<encoder>__<arm>/automil/``.
The operator submits its attempts, so training recipes can be developed on
datasets outside the paper roster without touching the campaign protocol.
A seed replicate (``materialize --seed N``, N other than the protocol seed) is the
same cell trained under another seed, in the sibling folder ``<cell>__seed<N>``.

Subcommands
    materialize  write one cell's config and scaffold, idempotently (needs the devcv
                 splits for the companion guard, or --no-guard)
    readout      per-attempt paired lift, selected epochs and the offline winner verdict
    batch        propose and submit a JSON list of recipe attempts, idempotently

The cell's config mirrors what the campaign materializer owns (metrics, scoring,
run command, timeout, policy smoke, registry), starting from the LUAD template.
Every state transition after materialization is executed by the ``automil`` CLI
as a subprocess (``dev_loop.py``); this file never writes ``graph.json``.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from dotenv import load_dotenv

from autobench.campaign import (
    ATTEMPT_TIMEOUT_MIN,
    PROTOCOL,
    SE_MULTIPLIER,
    SPLIT_STRATEGY,
    STAGE_FOLDS,
    VALIDATION_SCHEMA_BY_FAMILY,
    CampaignManifestError,
    _policy_smoke_for_cell,
    _run_command,
    _task_block,
    classification_task_family,
)
from autobench.config import load_dataset_config
from autobench.guard_margin import GuardMarginError, derive_guard

REPO_ROOT = Path(__file__).resolve().parents[2]
DEV_ROOT = Path("benchmarks/experiments/dev")
TEMPLATE = Path("benchmarks/experiments/tcga_luad/automil/config.yaml")
#: A label is a lowercase word; the ``seed`` prefix is reserved for replicate suffixes.
LABEL_PATTERN = re.compile(r"[a-z][a-z0-9]*")
DEFAULT_ENCODER = "hoptimus1"
#: arm -> (framework, model): the pairing the campaign manifest pins per roster dataset.
ARMS: Mapping[str, tuple[str, str]] = {
    "clam_mb": ("clam", "clam_mb"),
    "abmil": ("abmil", "abmil"),
    "dtfd_mil": ("dtfd", "dtfd_mil"),
    "simple_mil": ("nnmil", "simple_mil"),
}
#: Template keys that describe the template's own cohort or the campaign machinery.
DROPPED_TEMPLATE_KEYS = frozenset({"campaign", "activity", "baseline", "cohort"})
#: Wall-clock accounting with a budget the eval count always reaches first.
DEV_CAP: Mapping[str, str] = {"mode": "wall_clock", "budget": "90d", "safety_buffer": "1h"}
VIZ_PORT_BASE = 8500
VIZ_PORT_SPAN = 400
#: Same ignore list the campaign writes beside every cell's config.
GITIGNORE = (
    "graph.json\nresults.tsv\nresult.json\norchestrator/\ncells/\nsessions/\n"
    ".activity.jsonl\n.activity.samples.json\n.activity.lock\n"
    ".automil_active\n.automil_worktrees/\n*.log\n*.pid\n"
)
_ROOT_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class DevCellError(ValueError):
    """An input the dev cell cannot be built or published from."""


# ---------------------------------------------------------------------------
# Cell identity
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CellSpec:
    """Everything a dev cell's name and record are derived from."""

    cell_name: str
    dataset: str
    task: str
    task_family: str
    encoder: str
    arm: str
    framework: str
    model: str
    seed: int
    label: str | None = None

    def record(self) -> dict[str, Any]:
        """The content of ``automil/dev_cell.json``; an unlabelled cell records no label."""
        return {key: value for key, value in asdict(self).items() if not (key == "label" and value is None)}


def _read_yaml_mapping(path: Path, what: str) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(Path(path).read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise DevCellError(f"cannot read {what} {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise DevCellError(f"{what} {path} must contain a YAML mapping")
    return loaded


def _require_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DevCellError(f"{name} must be a positive integer, got {value!r}")
    return value


def _check_label(label: str | None) -> None:
    if label is None:
        return
    if not isinstance(label, str) or not LABEL_PATTERN.fullmatch(label) or label.startswith("seed"):
        raise DevCellError(f"label {label!r} must be a lowercase word (letters and digits, starting with a "
                           "letter) and must not start with 'seed', which names a replicate")


def _resolve(raw: Mapping[str, Any], path: Path, task: str, encoder: str, arm: str, seed: int,
             label: str | None = None) -> CellSpec:
    _require_positive_int(seed, "seed")
    _check_label(label)
    dataset = raw.get("name")
    if not isinstance(dataset, str) or not dataset:
        raise DevCellError(f"{path} declares no dataset name")
    if arm not in ARMS:
        raise DevCellError(f"arm {arm!r} is not one of {sorted(ARMS)}")
    tasks = raw.get("tasks") or {}
    if task not in tasks:
        raise DevCellError(f"task {task!r} is not in {path} (tasks: {sorted(tasks)})")
    task_spec = tasks[task] or {}
    if task_spec.get("task_type", "classification") == "survival":
        raise DevCellError(f"task {task!r} is a survival task; dev cells are classification-only")
    dims = (raw.get("encoders") or {}).get("dims") or {}
    if encoder not in dims:
        raise DevCellError(f"encoder {encoder!r} is not in {path} encoders.dims ({sorted(dims)})")
    try:
        family = classification_task_family(task_spec)
    except CampaignManifestError as exc:
        raise DevCellError(f"{path}: task {task!r}: {exc}") from exc
    framework, model = ARMS[arm]
    stem = f"{dataset}__{task}__{encoder}__{arm}"
    suffix = ("" if seed == PROTOCOL["seed"] else f"__seed{seed}") + ("" if label is None else f"__{label}")
    return CellSpec(
        cell_name=f"{stem}{suffix}", dataset=dataset, task=task, task_family=family, encoder=encoder,
        arm=arm, framework=framework, model=model, seed=seed, label=label,
    )


def resolve_cell(
    dataset_yaml: Path, task: str, encoder: str, arm: str, *, seed: int = PROTOCOL["seed"],
    label: str | None = None,
) -> CellSpec:
    """Validate the request against the dataset YAML and name the cell."""
    path = Path(dataset_yaml)
    return _resolve(_read_yaml_mapping(path, "dataset YAML"), path, task, encoder, arm, seed, label)


# ---------------------------------------------------------------------------
# The pure config builder
# ---------------------------------------------------------------------------
def _relative_to_repo(path: Path, repo_root: Path, what: str) -> str:
    try:
        return Path(path).resolve().relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise DevCellError(f"{what} {path} must live inside the repo {repo_root}") from exc


def _root_env_var(raw: Mapping[str, Any], path: Path) -> str:
    data_root = (raw.get("paths") or {}).get("data_root", "")
    match = _ROOT_VAR.fullmatch(data_root) if isinstance(data_root, str) else None
    if match is None:
        raise DevCellError(f"{path}: paths.data_root must be a single ${{ENV_VAR}}, got {data_root!r}")
    return match.group(1)


def _expand_paths(raw_paths: Mapping[str, str]) -> dict[str, str]:
    """Substitute references between ``paths`` entries, leaving ``${ENV_VAR}`` roots in place."""
    expanded: dict[str, str] = {}
    for key, value in raw_paths.items():
        for earlier, resolved in expanded.items():
            value = value.replace(f"${{{earlier}}}", resolved)
        expanded[key] = value
    return expanded


def _data_block(template_data: Mapping[str, Any] | None, raw: Mapping[str, Any],
                path: Path, task: str, encoder: str) -> dict[str, Any]:
    paths = _expand_paths(raw.get("paths") or {})
    missing = [key for key in ("benchmark_dir", "features_base_dir", "mapping_csv") if key not in paths]
    if missing:
        raise DevCellError(f"{path}: paths is missing {missing}")
    return {
        **(template_data or {}),
        "features_dir": f"{paths['features_base_dir']}/features_{encoder}",
        "splits_dir": f"{paths['benchmark_dir']}/splits/{SPLIT_STRATEGY}/{task}",
        "mapping_csv": paths["mapping_csv"],
        "num_folds": PROTOCOL["split_folds"],
        "seed": PROTOCOL["seed"],
    }


def _companion_guard(dataset_yaml: Path, root_var: str, task: str) -> tuple[dict | None, str | None]:
    """The guard derived from the mounted devcv splits, or ``(None, why there is none)``."""
    if not os.environ.get(root_var):
        return None, f"{root_var} is not set, so the devcv splits cannot be found"
    benchmark_dir = Path(load_dataset_config(str(dataset_yaml)).benchmark_dir)
    splits = benchmark_dir / "splits" / SPLIT_STRATEGY / task
    if not splits.is_dir():
        return None, f"{splits} does not exist (prepare the devcv splits first)"
    try:
        guard = derive_guard(benchmark_dir, SPLIT_STRATEGY, task, range(PROTOCOL["split_folds"]))
    except GuardMarginError as exc:
        raise DevCellError(f"cannot derive the companion guard from {splits}: {exc}") from exc
    return guard, None


def _viz_port(cell_name: str) -> int:
    digest = hashlib.sha256(cell_name.encode()).digest()
    return VIZ_PORT_BASE + int.from_bytes(digest[:4], "big") % VIZ_PORT_SPAN


def build_dev_config(
    dataset_yaml: Path,
    task: str,
    encoder: str,
    arm: str,
    *,
    template_path: Path,
    eval_budget: int,
    max_concurrent_per_gpu: int,
    repo_root: Path,
    out_root: Path | None = None,
    no_guard: bool = False,
    seed: int = PROTOCOL["seed"],
    label: str | None = None,
) -> dict[str, Any]:
    """The dev cell's ``automil/config.yaml`` as a dict; writes nothing.

    Starts from ``template_path`` and overwrites the blocks the campaign
    materializer owns, through the materializer's own helpers. ``out_root`` is
    the directory the cell folder will live in (default ``benchmarks/experiments/dev``).
    The cell carries the companion guard derived from the mounted devcv splits; a
    cell for which none can be derived is refused unless ``no_guard`` asks for a
    cell without one. ``seed`` is the training seed of the run command. The prepared
    splits are reused as they are, so every seed trains on the same folds; a seed other
    than the protocol's names a replicate cell (``<cell>__seed<N>``). ``label`` names a
    sibling cell with the same inputs in its own folder (``<cell>__<label>``), for a
    search method that must not share a graph with the agent's attempts.
    """
    dataset_yaml = Path(dataset_yaml)
    repo_root = Path(repo_root).resolve()
    _require_positive_int(eval_budget, "eval_budget")
    _require_positive_int(max_concurrent_per_gpu, "max_concurrent_per_gpu")
    raw = _read_yaml_mapping(dataset_yaml, "dataset YAML")
    spec = _resolve(raw, dataset_yaml, task, encoder, arm, seed, label)
    root_var = _root_env_var(raw, dataset_yaml)
    template = copy.deepcopy(_read_yaml_mapping(template_path, "config template"))

    cell_dir = (Path(out_root) if out_root is not None else repo_root / DEV_ROOT) / spec.cell_name
    adir_rel = _relative_to_repo(cell_dir / "automil", repo_root, "dev cell")
    cell = {
        "dataset": _relative_to_repo(dataset_yaml, repo_root, "dataset YAML"),
        "task": task, "task_type": "classification", "encoder": encoder,
        "framework": spec.framework, "model": spec.model, "survival_loss": None, "seed": spec.seed,
    }
    track = list(VALIDATION_SCHEMA_BY_FAMILY[spec.task_family])
    if no_guard:
        guard = None
        print("dev-cell note: --no-guard given; scoring.guard omitted", file=sys.stderr)
    else:
        guard, why_not = _companion_guard(dataset_yaml, root_var, task)
        if guard is None:
            raise DevCellError(
                f"no companion guard can be derived: {why_not}; "
                "pass --no-guard to materialize the cell without one"
            )

    registry = dict(template.get("registry") or {})
    smoke = _policy_smoke_for_cell(registry, spec.task_family, spec.framework)
    if smoke is not None:
        registry = {**registry, "policy_smoke": smoke}
    scoring = {key: value for key, value in (template.get("scoring") or {}).items() if key != "guard"}
    scoring = {**scoring, "formula": track[0], "se_multiplier": SE_MULTIPLIER}
    if guard is not None:
        scoring = {**scoring, "guard": guard}

    config = {key: value for key, value in template.items() if key not in DROPPED_TEMPLATE_KEYS}
    config.update({
        "project": {"name": spec.cell_name,
                    "description": f"development cell (not a campaign cell): {spec.cell_name}"},
        "task": _task_block(cell, raw),
        "data": _data_block(template.get("data"), raw, dataset_yaml, task, encoder),
        "encoders": {"primary": encoder, "available": dict(raw["encoders"]["dims"])},
        "metrics": {"primary": track[0], "track": track},
        "scoring": scoring,
        "files": {"editable": [f"{adir_rel}/variants/_policies/*.py"]},
        "run": {"script": None, "command": _run_command(cell, "discovery"), "mil_model": spec.model},
        "training": {"fold_count": len(STAGE_FOLDS["discovery"])},
        "orchestrator": {**(template.get("orchestrator") or {}),
                         "default_timeout_min": ATTEMPT_TIMEOUT_MIN,
                         "max_concurrent_per_gpu": max_concurrent_per_gpu},
        "registry": registry,
        "cap": {**DEV_CAP, "eval_budget": eval_budget},
        "env": {"passthrough": ["AUTOBENCH_ROOT", root_var, "AUTOBENCH_BAG_CACHE", "NNMIL_NUM_WORKERS",
                                "OMP_NUM_THREADS", "MKL_NUM_THREADS"],
                "dotenv_files": ["benchmarks/.env"]},
        "viz": {**(template.get("viz") or {}), "port": _viz_port(spec.cell_name)},
    })
    return config


# ---------------------------------------------------------------------------
# Publishing a cell
# ---------------------------------------------------------------------------
def _diff_paths(old: object, new: object, prefix: str = "") -> list[str]:
    """Dotted paths at which two parsed documents differ."""
    if isinstance(old, dict) and isinstance(new, dict):
        paths: list[str] = []
        for key in sorted(set(old) | set(new), key=str):
            if key in old and key in new:
                paths.extend(_diff_paths(old[key], new[key], f"{prefix}{key}."))
            else:
                paths.append(f"{prefix}{key}")
        return paths
    return [] if old == new else [prefix.rstrip(".")]


def _verify_existing(cell_dir: Path, config: Mapping[str, Any], record: Mapping[str, Any]) -> None:
    adir = cell_dir / "automil"
    try:
        existing_config = yaml.safe_load((adir / "config.yaml").read_text())
        existing_record = json.loads((adir / "dev_cell.json").read_text())
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise DevCellError(f"existing dev cell {cell_dir} is incomplete or corrupt: {exc}") from exc
    # Compare what a fresh write would read back, so list/tuple spelling cannot drift.
    wanted_config = yaml.safe_load(yaml.safe_dump(config, sort_keys=False))
    drift = _diff_paths(existing_config, wanted_config) + [
        f"dev_cell.json:{path}" for path in _diff_paths(existing_record, dict(record))
    ]
    if drift:
        raise DevCellError(
            f"dev cell {cell_dir} already exists with different inputs (differs at: "
            f"{', '.join(drift[:12])}); refusing to overwrite it. Remove the directory to rebuild."
        )


def _stage_cell(staging: Path, config: Mapping[str, Any], record: Mapping[str, Any]) -> None:
    adir = staging / "automil"
    policies = adir / "variants" / "_policies"
    policies.mkdir(parents=True)
    (policies / ".gitkeep").touch()
    name = record["cell_name"]
    (adir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    (adir / "dev_cell.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    (adir / ".gitignore").write_text(GITIGNORE)
    (adir / "plan.md").write_text(f"# Dev plan: {name}\n\nNo proposals queued yet.\n")
    (adir / "learnings.md").write_text(f"# Cell-local learnings: {name}\n")


def write_dev_cell(out_dir: Path, config: Mapping[str, Any], record: Mapping[str, Any]) -> str:
    """Publish the cell directory ``out_dir`` in one rename; never writes ``graph.json``.

    Returns ``"created"``, or ``"unchanged"`` when ``out_dir`` already holds this
    exact cell. A cell that exists with different inputs is refused.
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        _verify_existing(out_dir, config, record)
        return "unchanged"
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".materialize-", dir=str(out_dir.parent)))
    try:
        _stage_cell(staging, config, record)
        os.chmod(staging, 0o755)
        os.replace(staging, out_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return "created"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _fail(message: str, code: int = 2) -> None:
    print(f"dev-cell refusal: {message}", file=sys.stderr)
    sys.exit(code)


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


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}")
    return value


def cmd_materialize(args: argparse.Namespace) -> None:
    out_root = Path(args.out).resolve() if args.out is not None else REPO_ROOT / DEV_ROOT
    dataset_yaml = Path(args.dataset).resolve()
    try:
        spec = resolve_cell(dataset_yaml, args.task, args.encoder, args.arm, seed=args.seed, label=args.label)
        config = build_dev_config(
            dataset_yaml, args.task, args.encoder, args.arm,
            template_path=REPO_ROOT / TEMPLATE, eval_budget=args.eval_budget,
            max_concurrent_per_gpu=args.max_concurrent_per_gpu,
            repo_root=REPO_ROOT, out_root=out_root, no_guard=args.no_guard, seed=args.seed, label=args.label,
        )
        cell_dir = out_root / spec.cell_name
        status = write_dev_cell(cell_dir, config, spec.record())
    except DevCellError as exc:
        _fail(str(exc))
    print(f"{status}: {cell_dir}")


def cmd_readout(args: argparse.Namespace) -> None:
    loop = _load_script("dev_loop")
    try:
        data = loop.build_readout(Path(args.cell_dir))
    except loop.DevLoopError as exc:
        _fail(str(exc))
    print(json.dumps(data, indent=2, allow_nan=False) if args.json else loop.format_readout(data))


def cmd_batch(args: argparse.Namespace) -> None:
    loop = _load_script("dev_loop")
    try:
        loop.run_batch(Path(args.cell_dir), Path(args.batch_json))
    except loop.DevLoopError as exc:
        _fail(str(exc))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    materialize = sub.add_parser("materialize", help="write one dev cell, idempotently")
    materialize.add_argument("--dataset", required=True, type=Path, help="dataset YAML path")
    materialize.add_argument("--task", required=True)
    materialize.add_argument("--encoder", default=DEFAULT_ENCODER)
    materialize.add_argument("--arm", required=True, choices=sorted(ARMS))
    materialize.add_argument("--out", type=Path, default=None,
                             help=f"directory holding the cell folders (default {DEV_ROOT}); must be inside the repo")
    materialize.add_argument("--eval-budget", type=_positive_int, default=200,
                             help="attempts the cell may launch, the root baseline included")
    materialize.add_argument("--max-concurrent-per-gpu", type=_positive_int, default=4)
    materialize.add_argument("--seed", type=_positive_int, default=PROTOCOL["seed"],
                             help="training seed (default %(default)s, the protocol seed). Any other value "
                                  "makes a seed replicate in a folder with a __seed<N> suffix. The prepared "
                                  "splits directory is reused wholesale, so the replicate trains on exactly "
                                  "the folds of the seed-%(default)s cell: the seed changes only the "
                                  "training (torch seeding)")
    materialize.add_argument("--label", default=None,
                             help="name a sibling cell with the same inputs in its own folder "
                                  "(<cell>__<label>), e.g. optuna for the menu-AutoML arm")
    materialize.add_argument("--no-guard", action="store_true",
                             help="write the cell without a companion guard (without it, a cell whose "
                                  "devcv splits cannot be found is refused)")
    materialize.set_defaults(func=cmd_materialize)

    readout = sub.add_parser(
        "readout",
        help="per-attempt paired lift vs parent and root, selected epochs, offline winner verdict",
    )
    readout.add_argument("cell_dir")
    readout.add_argument("--json", action="store_true", help="dump the readout as JSON")
    readout.set_defaults(func=cmd_readout)

    batch = sub.add_parser("batch", help="propose and submit the attempts listed in a JSON file")
    batch.add_argument("cell_dir")
    batch.add_argument("batch_json")
    batch.set_defaults(func=cmd_batch)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv(REPO_ROOT / "benchmarks" / ".env")
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
