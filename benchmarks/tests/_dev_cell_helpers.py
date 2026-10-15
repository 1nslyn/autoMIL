"""Shared builders for the dev-cell tests (a helper module, not a test module)."""
from __future__ import annotations

import importlib.util
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "benchmarks" / "scripts"
LUAD_YAML = Path("benchmarks/datasets/tcga/tcga_luad.yaml")
LUAD_TEMPLATE = Path("benchmarks/experiments/tcga_luad/automil/config.yaml")
DEV_ROOT = Path("benchmarks/experiments/dev")


def load_script(stem: str):
    """``benchmarks/scripts/<stem>.py`` by path: the scripts are not a package."""
    cached = sys.modules.get(stem)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(stem, SCRIPTS / f"{stem}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def init_git_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for command in (
        ["git", "init"],
        ["git", "config", "user.name", "test"],
        ["git", "config", "user.email", "test@test.com"],
    ):
        subprocess.run(command, cwd=path, capture_output=True, check=True)
    (path / "README.md").write_text("# test\n")
    subprocess.run(["git", "add", "."], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, capture_output=True, check=True)
    return path


def git_head(repo: Path) -> str:
    """The full sha of ``repo``'s HEAD."""
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.strip()


def commit_file(repo: Path, name: str) -> str:
    """Commit one new file to ``repo``: HEAD moves; returns the new HEAD."""
    (repo / name).write_text(f"{name}\n")
    subprocess.run(["git", "add", name], cwd=repo, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", f"add {name}"], cwd=repo, capture_output=True, check=True)
    return git_head(repo)


def write_archived_spec(cell_dir: Path, node_id: str, spec: dict) -> Path:
    """``orchestrator/archive/<node>/spec.json``, which the daemon writes when it launches a node."""
    path = cell_dir / "automil" / "orchestrator" / "archive" / node_id / "spec.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"id": node_id, **spec}, indent=2))
    return path


def copy_luad_inputs(repo: Path) -> Path:
    """The two repo files ``build_dev_config`` reads, copied to the same relative paths."""
    for rel in (LUAD_YAML, LUAD_TEMPLATE):
        destination = repo / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / rel, destination)
    return repo


def build_luad_cell(
    dev_cell, repo: Path, *, arm: str = "abmil", eval_budget: int = 200, no_guard: bool = True,
    seed: int | None = None,
):
    """``(config, spec, cell_dir)`` of the LUAD KRAS dev cell for ``arm`` inside ``repo``.

    Guard-less by default: no test host has the devcv splits mounted, and a cell
    without them is refused unless ``no_guard`` says it is wanted. ``seed`` reaches
    the builder only when given, so leaving it out exercises the builder's own default.
    """
    dataset = repo / LUAD_YAML
    seeded = {} if seed is None else {"seed": seed}
    config = dev_cell.build_dev_config(
        dataset, "kras", "hoptimus1", arm,
        template_path=repo / LUAD_TEMPLATE, eval_budget=eval_budget,
        max_concurrent_per_gpu=4, repo_root=repo, no_guard=no_guard, **seeded,
    )
    spec = dev_cell.resolve_cell(dataset, "kras", "hoptimus1", arm, **seeded)
    return config, spec, repo / DEV_ROOT / spec.cell_name


# --- a synthetic cell for the readout tests ------------------------------------------

ROOT_FOLDS = [0.70, 0.72, 0.68, 0.71, 0.69]
EMA_FOLDS = [0.75, 0.75, 0.74, 0.75, 0.74]
LS_FOLDS = [0.74, 0.76, 0.73, 0.75, 0.74]
#: Fold 0 of the EMA attempt: the argmax (epoch 3) sits on a spike; the best
#: 5-epoch window (epochs 5-9) comes later.
SPIKY_CURVE = [0.60, 0.61, 0.62, 0.74, 0.60, 0.70, 0.71, 0.72, 0.71, 0.70]
RISING_CURVE = [0.50, 0.55, 0.60, 0.65, 0.70]


#: The companion guard the synthetic cell declares: val_bacc may fall one point below the root.
DEFAULT_GUARD = {"metric": "val_bacc", "margin": 0.01}
ROOT_COMMIT = "a" * 40
#: What the daemon archived for the synthetic nodes that ran: the run override of each.
ARCHIVED_OVERRIDES = {
    "node_0001": shlex.join(["--hparams", "{}"]),
    "node_0002": shlex.join(["--policy-variant", "ema", "--hparams", "{}"]),
    "node_0003": shlex.join(["--policy-variant", "ls", "--hparams", json.dumps({"weights": [0.123456789] * 12})]),
}


def fold_entries(values: list[float], companion: float | list[float | None] = 0.6) -> list[dict]:
    """The per-fold validation entries; ``companion`` is each fold's val_bacc (one number, or one per fold)."""
    per_fold = companion if isinstance(companion, list) else [companion] * len(values)
    return [
        {"fold_index": index, "primary_value": value,
         "metrics": {"val_auc_smooth": value, "val_auc": value, "val_bacc": bacc}}
        for index, (value, bacc) in enumerate(zip(values, per_fold))
    ]


def epoch_log(curves: list[list[float]]) -> str:
    """A run.log in the trainers' format: epoch lines, then selected and smoothed per fold."""
    lines = ["  hoptimus1:  50%|#####     | 2/4 [00:00<00:00, 38.16slide/s]"]
    for curve in curves:
        lines += [f"[epoch {index}] val_auc={value!r} val_loss=0.5" for index, value in enumerate(curve)]
        best = curve.index(max(curve))
        lines += [f"[selected] epoch={best} source=best", f"[smoothed] epoch={best} val_auc_smooth=0.5"]
    return "\n".join(lines) + "\n"


def write_synthetic_cell(
    cell_dir: Path, *, framework: str = "abmil", root_in_metadata: bool = False,
    first_curves: list[list[float]] | None = None, guard: dict | None = DEFAULT_GUARD,
    companions: dict[str, float | list[float | None]] | None = None,
) -> Path:
    """A root, two complete attempts, a crash and two queued nodes, with run logs, archived specs
    and a config.yaml declaring ``guard`` (none when ``guard`` is None).

    ``companions`` sets a node's per-fold val_bacc (every node's is 0.6 otherwise).
    """
    adir = cell_dir / "automil"
    archive = adir / "orchestrator" / "archive"
    companion_of = {} if companions is None else companions

    def executed(node_id, parent, status, description, values, config_hash, in_metadata=False):
        node = {"id": node_id, "parent_id": parent, "type": "executed", "status": status,
                "description": description, "techniques": [], "config_hash": config_hash,
                "primary_value": sum(values) / len(values) if values else 0.0}
        entries = fold_entries(values, companion_of.get(node_id, 0.6))
        if values and in_metadata:
            node["metadata"] = {"validation_folds": entries}
        elif values:
            node["fold_primary_values"] = entries
        return node

    nodes = {
        "node_0001": executed("node_0001", None, "keep", "native baseline", ROOT_FOLDS, "c0" * 8, root_in_metadata),
        "node_0002": executed("node_0002", "node_0001", "keep", "ema: EMA weight averaging with decay 0.998 "
                              "swapped in for validation and selection", EMA_FOLDS, "c2" * 8),
        "node_0003": executed("node_0003", "node_0002", "discard", "ls: label smoothing 0.2", LS_FOLDS, "c3" * 8),
        "node_0004": executed("node_0004", "node_0001", "crash", "crashed attempt", [], "c4" * 8),
        "node_0005": {"id": "node_0005", "parent_id": "node_0001", "type": "proposed", "status": "running",
                      "description": "queued", "techniques": []},
        "node_0006": {"id": "node_0006", "parent_id": "node_0001", "type": "proposed", "status": "pending",
                      "description": "not submitted", "techniques": []},
    }
    adir.mkdir(parents=True)
    (adir / "graph.json").write_text(json.dumps({"meta": {"next_id": 7}, "nodes": nodes}, indent=2))
    (adir / "dev_cell.json").write_text(json.dumps({
        "cell_name": cell_dir.name, "dataset": "tcga_luad", "task": "kras", "task_family": "binary",
        "encoder": "hoptimus1", "arm": framework, "framework": framework, "model": framework,
    }))
    first = first_curves or [SPIKY_CURVE, *[RISING_CURVE] * 4]
    logs = {
        "node_0001": [RISING_CURVE] * 5,
        "node_0002": first,
        "node_0003": [[0.5, 0.6, 0.55, 0.65, 0.62, 0.7]] * 5,
    }
    for node_id, curves in logs.items():
        (archive / node_id).mkdir(parents=True)
        (archive / node_id / "run.log").write_text(epoch_log(curves))
        write_archived_spec(cell_dir, node_id, {
            "base_commit": ROOT_COMMIT, "run_command_override": ARCHIVED_OVERRIDES[node_id],
        })
    scoring = {"formula": "val_auc_smooth", "se_multiplier": 1.0, **({"guard": guard} if guard else {})}
    (adir / "config.yaml").write_text(yaml.safe_dump({"task": {"name": "kras", "num_classes": 2}, "scoring": scoring}))
    return cell_dir


# --- a git-backed dev cell for the batch tests ----------------------------------------

POLICY_TEXT = '''"""{name}: a test policy that changes nothing."""
from automil.registry import register, VariantSpec, PolicyVariant


@register(VariantSpec(
    name="{name}", kind="policy", parent=None,
    base_commit="{base_commit}", primary_value=0.0, node_id="{node_id}",
    created_at="{created_at}",
))
class Recipe(PolicyVariant):
    def wrap_optimizer(self, opt):
        return opt
'''


def fake_instantiate_recipe(calls: list[dict], text: str | None = None):
    """A stand-in for ``dev_recipe.instantiate_recipe`` that records its arguments."""

    def instantiate(template_path, *, name, constants, node_id, base_commit, created_at):
        calls.append({"template_path": template_path, "name": name, "constants": constants,
                      "node_id": node_id, "base_commit": base_commit, "created_at": created_at})
        return text if text is not None else POLICY_TEXT.format(
            name=name, base_commit=base_commit, node_id=node_id, created_at=created_at)

    return instantiate


def build_git_dev_cell(dev_cell, tmp_path: Path, *, arm: str = "abmil") -> Path:
    """The LUAD KRAS dev cell inside a fresh git repo, with a trivial policy smoke."""
    repo = copy_luad_inputs(init_git_repo(tmp_path / "repo"))
    config, spec, cell_dir = build_luad_cell(dev_cell, repo, arm=arm)
    smoke = {"command": ["{python}", "-c", "pass"], "timeout_s": 60}  # the real smoke needs torch
    config = {**config, "registry": {**config["registry"], "policy_smoke": smoke}}
    dev_cell.write_dev_cell(cell_dir, config, spec.record())
    return cell_dir


def seed_executed_root(cell_dir: Path, folds: list[float] = ROOT_FOLDS) -> None:
    """A finished root baseline in graph.json, as the daemon would leave it: it ran at HEAD."""
    from automil.graph import locked_update

    with locked_update(cell_dir / "automil" / "graph.json") as graph:
        node_id = graph.add_executed(
            None, "baseline: native baseline", ["recipe:baseline"],
            {"primary_value": sum(folds) / len(folds)}, status="keep", config_hash="c0" * 8)
        graph.nodes[node_id]["fold_primary_values"] = fold_entries(folds)
    write_archived_spec(cell_dir, node_id, {"base_commit": git_head(cell_dir.parents[3])})


def finish_node(cell_dir: Path, node_id: str, folds: list[float], status: str = "keep") -> None:
    """Complete a submitted node the way the daemon does: archive its queued spec, then the terminal writer."""
    from automil.graph import locked_update

    queued = cell_dir / "automil" / "orchestrator" / "queue" / f"{node_id}.json"
    if queued.is_file():
        write_archived_spec(cell_dir, node_id, json.loads(queued.read_text()))
    with locked_update(cell_dir / "automil" / "graph.json") as graph:
        graph.promote(node_id, {"status": status, "primary_value": sum(folds) / len(folds),
                                "fold_primary_values": fold_entries(folds)})
