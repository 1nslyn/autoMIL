"""Read out and drive the attempts of a dev cell (see ``dev_cell.py``).

``build_readout`` is read-only: it joins ``graph.json`` with each attempt's
``run.log`` and archived ``spec.json`` and reports, per executed node, the paired lift
against its parent and against the root, the selected epochs and the shape of the
validation curve around them, the commit it ran at and its run override, plus the
offline winner verdict. The verdict holds every attempt to the cell's companion guard
(config.yaml ``scoring.guard``) as the discovery freeze does, against the root, and an
attempt that fails it stays out of the winner pool. ``run_batch`` is the only writer. It
adds no node itself (it only creates an empty ``graph.json`` when there is none): every
node goes through the ``automil`` CLI (``propose`` and ``submit``), run as a subprocess,
so the framework's admissibility, purity and policy-smoke checks apply to a dev-cell
attempt exactly as they do to an agent's.

Per-fold curve statistics, from the ``val_auc`` of every evaluated epoch:
    level           mean val_auc of the 5 evaluated epochs around the selected one
                    (the fold's reported score, ``smoothed_selection_value``)
    jitter          val_auc at the selected epoch minus that level
    hindsight_gap   the best 5-epoch level over the whole run minus the level at the
                    selected epoch: what a hindsight-best plateau would add
"""
from __future__ import annotations

import importlib.util
import json
import math
import re
import shlex
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from autobench.campaign import WINNER_FLOOR, max_lift_winner
from autobench.campaign_stages import _companion_guard_shortfall, _fold_metric_values
from autobench.pipeline.selection import smoothed_selection_value
from automil.graph import _guard_declaration, _se_multiplier, locked_update, node_fold_primary_values
from automil.runtime_helpers import atomic_write_text
from automil.scoring import paired_delta_se

#: Per-epoch key the curve statistics read (classification cells).
EPOCH_KEY = "val_auc"
#: Statuses of an attempt that ran to a scored result.
COMPLETE_STATUSES = frozenset({"keep", "discard"})
#: The one-sided error rate the winner bar is set for.
SIDAK_CONFIDENCE = 0.95
DESCRIPTION_WIDTH = 70
OVERRIDE_WIDTH = 100


class DevLoopError(RuntimeError):
    """A cell, graph or batch file the readout or the batch driver cannot work from."""


def _load_script(stem: str):
    """``benchmarks/scripts/<stem>.py`` by path (scripts are not a package), cached by name."""
    cached = sys.modules.get(stem)
    if cached is not None:
        return cached
    path = Path(__file__).resolve().with_name(f"{stem}.py")
    if not path.is_file():
        raise DevLoopError(f"{path} is missing")
    spec = importlib.util.spec_from_file_location(stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


#: The batch file's contract (entries and their validation) lives beside this script.
_BATCH_FILE = _load_script("dev_batch_file")
Entry = _BATCH_FILE.Entry


def load_cell(cell_dir: Path) -> tuple[Path, dict[str, Any]]:
    """``(automil dir, dev_cell.json)`` of a dev cell."""
    adir = Path(cell_dir) / "automil"
    record_path = adir / "dev_cell.json"
    if not record_path.is_file():
        raise DevLoopError(f"{record_path} not found: {cell_dir} is not a dev cell")
    try:
        record = json.loads(record_path.read_text())
    except (OSError, ValueError) as exc:
        raise DevLoopError(f"cannot read {record_path}: {exc}") from exc
    missing = [key for key in ("cell_name", "arm", "framework") if not isinstance(record.get(key), str)]
    if missing:
        raise DevLoopError(f"{record_path} lacks {missing}")
    return adir, record


def load_config(adir: Path) -> dict[str, Any]:
    """``automil/config.yaml`` of a dev cell."""
    path = adir / "config.yaml"
    try:
        config = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise DevLoopError(f"cannot read {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise DevLoopError(f"{path} must contain a YAML mapping")
    return config


def load_graph(adir: Path) -> dict[str, Any]:
    path = adir / "graph.json"
    if not path.is_file():
        raise DevLoopError(f"{path} not found: no attempt has been submitted yet")
    try:
        graph = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise DevLoopError(f"cannot read {path}: {exc}") from exc
    if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), dict):
        raise DevLoopError(f"{path} is not a graph: no nodes mapping")
    return graph


def find_root(nodes: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The baseline every attempt is paired against: the parentless executed node that was kept."""
    return next((node for _, node in sorted(nodes.items())
                 if node.get("type") == "executed" and node.get("parent_id") is None
                 and node.get("status") == "keep"), None)


def _archived_spec(adir: Path, node_id: str) -> dict[str, Any] | None:
    """``orchestrator/archive/<node>/spec.json`` as the daemon wrote it when it launched the node.

    ``None`` until the node has launched; an unreadable file is an error, not an absent one.
    """
    path = adir / "orchestrator" / "archive" / node_id / "spec.json"
    if not path.is_file():
        return None
    try:
        spec = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise DevLoopError(f"cannot read {path}: {exc}") from exc
    if not isinstance(spec, dict):
        raise DevLoopError(f"{path} is not a spec: no JSON object")
    return spec


# ---------------------------------------------------------------------------
# Readout
# ---------------------------------------------------------------------------
def _finite(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def _pair(child: Mapping[int, float] | None, parent: Mapping[int, float] | None) -> dict[str, float | None] | None:
    """Mean per-fold ``child - parent`` and its paired SE; ``None`` unless the fold sets agree."""
    if not child or not parent or set(child) != set(parent):
        return None
    folds = sorted(child)
    delta = math.fsum(child[fold] - parent[fold] for fold in folds) / len(folds)
    return {"delta": delta, "se": paired_delta_se(child, parent)}


def _fold_curve_row(fold: int, segment: Any, framework: str, replay: Any) -> dict[str, Any]:
    history = {
        obs.epoch: obs.value
        for obs in (replay.observe_epoch(index, metrics, EPOCH_KEY, framework)
                    for index, metrics in segment.epochs)
    }
    selected = segment.selected_epoch
    level = smoothed_selection_value(history, selected) if selected >= 0 else None
    peak = _finite(history.get(selected))
    levels = [smoothed_selection_value(history, epoch) for epoch, value in history.items() if math.isfinite(value)]
    best = max(levels) if levels else None
    both = level is not None and peak is not None
    return {
        "fold": fold,
        "selected_epoch": selected,
        "n_epochs": len(history),
        "peak": peak,
        "level": level,
        "jitter": peak - level if both else None,
        "hindsight_level": best,
        "hindsight_gap": best - level if best is not None and level is not None else None,
    }


def _log_rows(log_path: Path, framework: str) -> tuple[list[dict[str, Any]] | None, str | None]:
    """``(per-fold curve rows, None)`` for a readable log, ``(None, why)`` for a broken one."""
    if not log_path.is_file():
        return None, None
    replay = _load_script("replay_checkpoint_selection")
    try:
        with log_path.open(encoding="utf-8", errors="replace") as handle:
            segments = replay.parse_segments(handle)
    except (OSError, replay.InputError) as exc:
        return None, str(exc)
    return [_fold_curve_row(fold, segment, framework, replay) for fold, segment in enumerate(segments)], None


def _fold_entries(node: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The node's per-fold validation entries, from the two sources ``node_fold_primary_values`` reads."""
    entries = node.get("fold_primary_values")
    if not isinstance(entries, list) or not entries:
        meta = node.get("metadata")
        entries = meta.get("validation_folds") if isinstance(meta, dict) else None
    return entries if isinstance(entries, list) else []


def _declared_guard(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """The companion guard config.yaml's ``scoring`` block declares, or ``None`` when it declares none.

    Read through the two helpers the discovery freeze reads its declaration with, so a block it
    would refuse is refused here too.
    """
    scoring = config.get("scoring")
    if not isinstance(scoring, dict):
        raise DevLoopError("config.yaml has no scoring mapping")
    meta = {"scoring": scoring}
    try:
        declared = _guard_declaration(meta)
    except ValueError as exc:
        raise DevLoopError(f"config.yaml declares an unusable scoring.guard: {exc}") from exc
    if declared is None:
        return None
    metric, margin = declared
    return {"metric": metric, "margin": margin, "se_multiplier": _se_multiplier(meta)}


def _guard_floor(guard: Mapping[str, Any] | None, root: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """What an attempt's companion is held to: the floor ``campaign_stages._companion_guard_floor``
    builds from the registered baseline, here from the root's own folds.

    ``None`` when the cell declares no guard or the root has no per-fold evidence at all (the
    verdict reports that); a root that has folds but not the companion on every one cannot hold
    anyone to it, which the freeze refuses too.
    """
    entries = _fold_entries(root) if root is not None else []
    if guard is None or not entries:
        return None
    baseline_folds = _fold_metric_values(entries, guard["metric"])
    if len(baseline_folds) != len(entries):
        raise DevLoopError(f"cannot apply the companion guard: the root records no {guard['metric']} on every fold")
    return {
        "metric": guard["metric"],
        "margin": float(guard["margin"]),
        "baseline": sum(baseline_folds.values()) / len(baseline_folds),
        "baseline_folds": baseline_folds,
        "se_multiplier": guard["se_multiplier"],
    }


def _judge(floor: Mapping[str, Any], node: Mapping[str, Any]) -> dict[str, Any] | None:
    """The node under the companion guard, judged as the discovery freeze judges a candidate
    (``campaign_stages._companion_guard_shortfall``, against the root); ``None`` unless the node
    is a complete attempt with per-fold evidence, which is all the freeze judges."""
    entries = _fold_entries(node)
    if node.get("status") not in COMPLETE_STATUSES or not entries:
        return None
    metric = floor["metric"]
    shortfall = _companion_guard_shortfall(floor, entries)
    reason = None
    if shortfall is not None:
        drop, margin = shortfall
        # The wording of freeze_discovery's attempt audit; an infinite drop is a companion the
        # node did not record on every fold (the freeze fails closed on it).
        reason = (
            f"companion guard: {metric} is not recorded on every fold" if math.isinf(drop)
            else f"companion guard: {metric} fell {drop:.4f} below the baseline (margin {margin:.4f})"
        )
    return {
        "metric": metric,
        "vs_root": _pair(_fold_metric_values(entries, metric), floor["baseline_folds"]),
        "status": "pass" if shortfall is None else "fail",
        "reason": reason,
    }


def guard_standing(config: Mapping[str, Any], nodes: Mapping[str, Any],
                   node: Mapping[str, Any]) -> dict[str, Any] | None:
    """``node`` under the cell's companion guard, judged as the readout judges it.

    ``None`` when the cell declares no guard, the root cannot hold anyone to it, the node
    is the root, or the node is not a complete attempt with per-fold evidence.
    """
    root = find_root(nodes)
    floor = _guard_floor(_declared_guard(config), root)
    if floor is None or node.get("id") == root["id"]:
        return None
    return _judge(floor, node)


def _node_row(node: Mapping[str, Any], nodes: Mapping[str, Any], root: Mapping[str, Any] | None,
              adir: Path, framework: str, floor: Mapping[str, Any] | None) -> dict[str, Any]:
    folds = node_fold_primary_values(node)
    parent = nodes.get(node.get("parent_id"))
    root_folds = node_fold_primary_values(root) if root is not None else None
    is_root = root is not None and node["id"] == root["id"]
    log, log_error = _log_rows(adir / "orchestrator" / "archive" / node["id"] / "run.log", framework)
    spec = _archived_spec(adir, node["id"]) or {}
    return {
        "id": node["id"],
        "parent": node.get("parent_id"),
        "status": node.get("status"),
        "description": node.get("description") or "",
        "base_commit": spec.get("base_commit"),
        "run_command_override": spec.get("run_command_override"),
        "primary_value": _finite(node.get("primary_value")),
        "fold_indices": sorted(folds) if folds else None,
        "fold_values": [folds[fold] for fold in sorted(folds)] if folds else None,
        "vs_parent": _pair(folds, node_fold_primary_values(parent)) if parent else None,
        "vs_root": _pair(folds, root_folds) if root is not None and not is_root else None,
        "guard": _judge(floor, node) if floor is not None and not is_root else None,
        "log": log,
        "log_error": log_error,
    }


def _unavailable(reason: str) -> dict[str, Any]:
    return {"available": False, "reason": reason, "n_pool": 0, "frozen_rule": None, "sidak": None,
            "guard_excluded": []}


def _sidak(record: Mapping[str, Any], n_attempts: int) -> dict[str, Any] | None:
    """The winner bar for the best of ``n_attempts`` comparisons (the frozen rule fixes N = 30)."""
    if n_attempts < 1:
        return None
    z = statistics.NormalDist().inv_cdf(SIDAK_CONFIDENCE ** (1 / n_attempts))
    bar = max(WINNER_FLOOR, z * record["standard_error"])
    return {"n": n_attempts, "z": z, "bar": bar, "accepted": record["leader_lift"] > bar}


def _verdict(root: Mapping[str, Any] | None, attempts: list[Mapping[str, Any]],
             guard_failures: Mapping[str, str]) -> dict[str, Any]:
    """The winner rule over the complete attempts, on validation values only.

    ``guard_failures`` maps the id of each attempt that failed the companion guard to its reason:
    such an attempt is left out of the pool and listed, as the discovery freeze leaves it out.
    """
    if root is None:
        return _unavailable("no executed root: a parentless node with status keep")
    root_folds = node_fold_primary_values(root)
    if not root_folds or len(root_folds) < 2:
        return _unavailable("the root has no per-fold validation values (the rule needs at least two)")
    folds = sorted(root_folds)
    pool, excluded = [], []
    for node in attempts:
        values = node_fold_primary_values(node)
        if node.get("status") in COMPLETE_STATUSES and values and set(values) == set(root_folds):
            if node["id"] in guard_failures:
                excluded.append({"node": node["id"], "reason": guard_failures[node["id"]]})
                continue
            pool.append({
                "candidate_id": node["id"],
                "candidate_sha256": str(node.get("config_hash") or ""),
                "fold_values": [values[fold] for fold in folds],
            })
    record = max_lift_winner([root_folds[fold] for fold in folds], pool)
    return {"available": True, "reason": None, "n_pool": len(pool), "frozen_rule": record,
            "sidak": _sidak(record, len(pool)) if pool else None, "guard_excluded": excluded}


def _counts(nodes: Mapping[str, Any]) -> dict[str, int]:
    proposed = [node for node in nodes.values() if node.get("type") == "proposed"]
    return {
        "executed": sum(1 for node in nodes.values() if node.get("type") == "executed"),
        "running": sum(1 for node in proposed if node.get("status") == "running"),
        "pending": sum(1 for node in proposed if node.get("status") == "pending"),
    }


def build_readout(cell_dir: Path) -> dict[str, Any]:
    """Everything the readout reports for one dev cell, as JSON-ready data.

    ``guard`` is the companion guard config.yaml declares (``None`` when it declares none);
    each node's own ``guard`` is that node's standing under it.
    """
    adir, record = load_cell(cell_dir)
    config = load_config(adir)
    nodes = load_graph(adir)["nodes"]
    executed = [node for _, node in sorted(nodes.items()) if node.get("type") == "executed"]
    root = find_root(nodes)
    guard = _declared_guard(config)
    floor = _guard_floor(guard, root)
    rows = [_node_row(node, nodes, root, adir, record["framework"], floor) for node in executed]
    guard_failures = {row["id"]: row["guard"]["reason"] for row in rows
                      if row["guard"] is not None and row["guard"]["status"] == "fail"}
    return {
        "cell": record["cell_name"],
        "arm": record["arm"],
        "framework": record["framework"],
        "root": root["id"] if root is not None else None,
        "guard": guard,
        "counts": _counts(nodes),
        "nodes": rows,
        "verdict": _verdict(root, [node for node in executed if node is not root], guard_failures),
    }


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------
def _num(value: float | None, places: int = 4, *, signed: bool = False) -> str:
    if value is None:
        return "-"
    return f"{value:+.{places}f}" if signed else f"{value:.{places}f}"


def _delta_text(pair: Mapping[str, float | None] | None) -> str:
    if pair is None:
        return "-"
    return f"{_num(pair['delta'], signed=True)} ± {_num(pair['se'])}"


def _clip(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 3] + "..."


def _format_node(node: Mapping[str, Any]) -> list[str]:
    override = node["run_command_override"]
    lines = [
        f"{node['id']}  parent {node['parent'] or '-'}  {node['status']}  primary {_num(node['primary_value'])}",
        f"  {_clip(node['description'], DESCRIPTION_WIDTH)}",
        f"  base        {node['base_commit'] or '-'}",
        f"  run         {_clip(override, OVERRIDE_WIDTH) if override else '-'}",
        "  folds       " + (" ".join(_num(value) for value in node["fold_values"]) if node["fold_values"] else "-"),
        f"  vs parent   {_delta_text(node['vs_parent'])}",
        f"  vs root     {_delta_text(node['vs_root'])}",
    ]
    guard = node["guard"]
    if guard is not None:
        lines.append(f"  guard       {guard['metric']} vs root {_delta_text(guard['vs_root'])}  {guard['status']}")
    if node["log_error"]:
        lines.append(f"  run.log unreadable: {node['log_error']}")
    elif node["log"]:
        lines.append("  fold  sel  epochs  peak    level   jitter  hindsight-gap")
        lines.extend(
            f"  {row['fold']:<5} {row['selected_epoch']:<4} {row['n_epochs']:<7} {_num(row['peak'])}  "
            f"{_num(row['level'])}  {_num(row['jitter'])}  {_num(row['hindsight_gap'])}"
            for row in node["log"]
        )
    return lines


def _format_guard(guard: Mapping[str, Any] | None) -> str:
    if guard is None:
        return "  companion guard: none declared in config.yaml; no attempt is held to a companion metric"
    return (f"  companion guard: {guard['metric']}, margin {_num(guard['margin'])} or "
            f"{guard['se_multiplier']:g} x paired SE if larger, against the root")


def _format_verdict(verdict: Mapping[str, Any], guard: Mapping[str, Any] | None) -> list[str]:
    lines = ["verdict (offline, validation only)", _format_guard(guard)]
    lines.extend(f"  excluded by the companion guard: {excluded['node']} ({excluded['reason']})"
                 for excluded in verdict["guard_excluded"])
    if not verdict["available"]:
        return [*lines, f"  unavailable: {verdict['reason']}"]
    record = verdict["frozen_rule"]
    if record["leader"] is None:
        return [*lines, "  no attempt in the winner pool; the baseline stays"]
    lines.append(
        f"  {verdict['n_pool']} complete attempts; leader {record['leader']} lift "
        f"{_num(record['leader_lift'], signed=True)}, pooled SE {_num(record['standard_error'])}"
    )
    lines.append(
        f"  frozen rule (z {record['se_multiplier']}): bar {_num(record['bar'])} -> "
        f"{'accepted' if record['accepted'] else 'not accepted'}"
    )
    sidak = verdict["sidak"]
    lines.append(
        f"  Sidak bar for the best of {sidak['n']} (z {sidak['z']:.3f}): bar {_num(sidak['bar'])} -> "
        f"{'accepted' if sidak['accepted'] else 'not accepted'}"
    )
    return [*lines, f"  winner: {record['winner']}"]


def format_readout(data: Mapping[str, Any]) -> str:
    counts = data["counts"]
    lines = [
        f"cell {data['cell']}  arm {data['arm']}  root {data['root'] or '-'}  executed {counts['executed']}  "
        f"running {counts['running']}  pending {counts['pending']}",
        "",
    ]
    for node in data["nodes"]:
        lines.extend([*_format_node(node), ""])
    return "\n".join([*lines, *_format_verdict(data["verdict"], data["guard"])])


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------
ROOT_NODE_ID = "node_0001"
LEDGER_NAME = "dev_batch.json"
#: The recipe template, relative to the repository root.
RECIPE_TEMPLATE = Path("benchmarks/experiments/dev/_recipes/composite.py")
PROPOSAL_PATTERN = re.compile(r"Added proposal (node_\d+)")


@dataclass(frozen=True)
class Outcome:
    """What the driver did with one entry: ``submitted``, ``resumed`` or ``skip``."""

    name: str
    action: str
    node_id: str | None = None
    detail: str = ""

    @property
    def line(self) -> str:
        return f"{self.action} {self.name}: {self.detail or self.node_id}"


def _state(node: Mapping[str, Any]) -> str:
    return f"{node.get('type')}/{node.get('status')}"


def _git_root(start: Path) -> Path:
    """The repository the cell lives in, found the way ``automil submit`` finds it."""
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    raise DevLoopError(f"{start} is not inside a git repository")


def _git_head(repo: Path, *, short: bool = False) -> str:
    """HEAD of ``repo``: the full sha ``submit`` records as ``base_commit``, or git's short form."""
    command = ["git", "rev-parse", *(["--short"] if short else []), "HEAD"]
    result = subprocess.run(command, cwd=repo, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        raise DevLoopError(f"cannot read HEAD of {repo}: {result.stderr.strip() or 'no output'}")
    return result.stdout.strip()


def _run_automil(adir: Path, repo: Path, *args: str) -> str:
    """``python -m automil --project <cell> <args>`` from the repo root; stdout, or a refusal naming stderr."""
    command = [sys.executable, "-m", "automil", "--project", str(adir), *args]
    result = subprocess.run(command, cwd=repo, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no output"
        raise DevLoopError(f"automil {args[0]} failed (exit {result.returncode}): {detail}")
    return result.stdout


def _override(hparams: Mapping[str, Any], policy: str | None = None) -> str:
    """The ``submit --override`` string: the policy to run (attempts only) and its hparams."""
    tokens = ["--policy-variant", policy] if policy is not None else []
    return shlex.join([*tokens, "--hparams", json.dumps(hparams, sort_keys=True)])


def _instantiate_recipe(template_path: Path, **fields: Any) -> str:
    """``dev_recipe.instantiate_recipe``, loaded by path; its refusals surface as batch errors."""
    recipe = _load_script("dev_recipe")
    try:
        return recipe.instantiate_recipe(template_path, **fields)
    except ValueError as exc:
        raise DevLoopError(f"recipe {fields.get('name')!r}: {exc}") from exc


def _write_policy(adir: Path, repo: Path, arm: str, entry: Entry, node_id: str) -> Path:
    text = _instantiate_recipe(
        repo / RECIPE_TEMPLATE, name=entry.name, constants={**entry.constants, "ARM": arm},
        node_id=node_id, base_commit=_git_head(repo, short=True),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    path = adir / "variants" / "_policies" / f"{entry.name}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, text)
    return path


def _propose(adir: Path, repo: Path, entry: Entry, parent_id: str) -> str:
    out = _run_automil(
        adir, repo, "propose", "--parent", parent_id, "--desc", entry.title, "--kind", entry.kind,
        "--techniques", entry.tag, "--axis", entry.axis, "--predicted-delta", "0.0",
    )
    found = PROPOSAL_PATTERN.search(out)
    if found is None:
        raise DevLoopError(f"propose printed no node id for {entry.name!r}: {out.strip()!r}")
    return found.group(1)


def _submit_root(adir: Path, repo: Path, entry: Entry, nodes: Mapping[str, Any]) -> Outcome:
    root = next((node for node in nodes.values() if node.get("parent_id") is None), None)
    if root is not None:
        raise DevLoopError(f"the graph already has a root ({root['id']}: {root.get('description')!r}); "
                           f"root entry {entry.name!r} would be a second one")
    _run_automil(adir, repo, "submit", "--node", ROOT_NODE_ID, "--desc", entry.title,
                 "--techniques", entry.tag, "--override", _override(entry.hparams))
    return Outcome(entry.name, "submitted", ROOT_NODE_ID)


def _require_root_code(adir: Path, repo: Path, nodes: Mapping[str, Any]) -> None:
    """Refuse unless HEAD is the commit the root ran at: an attempt's delta is only a comparison
    when it runs the root's code."""
    root = find_root(nodes)
    if root is None:
        raise DevLoopError("the graph has no executed root to run attempts against; let the baseline finish first")
    ran_at = (_archived_spec(adir, root["id"]) or {}).get("base_commit")
    if not isinstance(ran_at, str) or not ran_at:
        raise DevLoopError(f"cannot tell which code the root {root['id']} ran: "
                           f"orchestrator/archive/{root['id']}/spec.json is missing or records no base_commit")
    head = _git_head(repo)
    if head != ran_at:
        raise DevLoopError(f"the root ran at {ran_at}; HEAD is {head}; attempts must run the root's code")


def _submit_attempt(adir: Path, repo: Path, arm: str, entry: Entry, parent_id: str,
                    stranded: Mapping[str, Any] | None, nodes: Mapping[str, Any]) -> Outcome:
    """Check the code is the root's, then propose (unless a proposal is stranded), write the policy, submit."""
    _require_root_code(adir, repo, nodes)
    node_id = stranded["id"] if stranded is not None else _propose(adir, repo, entry, parent_id)
    policy = _write_policy(adir, repo, arm, entry, node_id)
    _run_automil(adir, repo, "submit", "--node", node_id, "--parent", parent_id, "--desc", entry.title,
                 "--files", policy.relative_to(repo).as_posix(),
                 "--override", _override(entry.hparams, policy=entry.name))
    return Outcome(entry.name, "resumed" if stranded is not None else "submitted", node_id)


def _already(entry: Entry, node: Mapping[str, Any]) -> Outcome:
    return Outcome(entry.name, "skip", node["id"], f"already {node['id']} ({_state(node)})")


def _drive(adir: Path, repo: Path, arm: str, entry: Entry) -> Outcome:
    nodes = load_graph(adir)["nodes"]
    known = _BATCH_FILE.find_node(nodes, entry.name)
    if entry.parent is None:
        return _already(entry, known) if known is not None else _submit_root(adir, repo, entry, nodes)
    stranded = None  # a proposal an earlier run left pending: its policy write or its submit failed
    if known is not None:
        if _state(known) != "proposed/pending":
            return _already(entry, known)
        stranded = known
    parent = nodes.get(entry.parent) or _BATCH_FILE.find_node(nodes, entry.parent)
    if parent is None:
        return Outcome(entry.name, "skip", None, f"parent {entry.parent!r} is not in the graph")
    if parent.get("type") != "executed" or parent.get("status") not in COMPLETE_STATUSES:
        return Outcome(entry.name, "skip", None,
                       f"parent {entry.parent} ({parent['id']}) is {_state(parent)}, not complete")
    return _submit_attempt(adir, repo, arm, entry, parent["id"], stranded, nodes)


def _remember(adir: Path, outcome: Outcome) -> None:
    """Keep ``dev_batch.json`` ({name: node id}) in step with the graph; atomic, rewritten only on change."""
    if outcome.node_id is None:
        return
    path = adir / LEDGER_NAME
    try:
        ledger = json.loads(path.read_text()) if path.is_file() else {}
    except (OSError, ValueError) as exc:
        raise DevLoopError(f"cannot read {path}: {exc}") from exc
    if not isinstance(ledger, dict):
        raise DevLoopError(f"{path} is not a name -> node id mapping")
    if ledger.get(outcome.name) != outcome.node_id:
        atomic_write_text(path, json.dumps({**ledger, outcome.name: outcome.node_id}, indent=2, sort_keys=True) + "\n")


def _num_classes(adir: Path) -> int:
    """The class count of the cell's task, from config.yaml's ``task.num_classes``."""
    task = load_config(adir).get("task")
    count = task.get("num_classes") if isinstance(task, dict) else None
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise DevLoopError(
            f"{adir / 'config.yaml'}: task.num_classes must be an integer of at least 2, got {count!r}"
        )
    return count


def drive_entry(cell_dir: Path, entry: Entry) -> Outcome:
    """One entry through the driver's path (propose, policy, submit), recorded in the ledger.

    An entry already in the graph is skipped, an entry whose parent has not finished
    is skipped and reported, and a proposal an earlier run stranded (policy write or
    submit failed) is resumed rather than proposed again.
    """
    adir, record = load_cell(Path(cell_dir).resolve())
    repo = _git_root(adir)
    if not (adir / "graph.json").exists():
        with locked_update(adir / "graph.json"):
            pass  # ``submit`` registers a node only in an existing graph; its meta seeds from config.yaml
    outcome = _drive(adir, repo, record["arm"], entry)
    _remember(adir, outcome)
    return outcome


def run_batch(cell_dir: Path, batch_path: Path, *, emit: Callable[[str], None] = print) -> list[Outcome]:
    """Propose and submit the entries of ``batch_path`` in order; idempotent (see ``drive_entry``).

    The first refusal from the framework stops the batch with its message.
    """
    adir, record = load_cell(Path(cell_dir).resolve())
    try:
        entries = _BATCH_FILE.parse_batch(Path(batch_path), arm=record["arm"], n_classes=_num_classes(adir))
    except _BATCH_FILE.BatchFileError as exc:
        raise DevLoopError(str(exc)) from exc
    outcomes: list[Outcome] = []
    for entry in entries:
        outcome = drive_entry(cell_dir, entry)
        emit(outcome.line)
        outcomes.append(outcome)
    return outcomes
