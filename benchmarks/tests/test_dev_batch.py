"""Dev cells: the batch driver, run against the real automil CLI in a throwaway git repo."""
from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from _dev_cell_helpers import (
    POLICY_TEXT,
    REPO_ROOT,
    ROOT_FOLDS,
    SCRIPTS,
    build_git_dev_cell,
    commit_file,
    fake_instantiate_recipe,
    finish_node,
    git_head,
    load_script,
    seed_executed_root,
)

from automil.graph import locked_update

ROOT = {"name": "baseline", "parent": None, "desc": "native baseline", "axis": None, "kind": None,
        "constants": None, "hparams": {}}


def attempt(name: str, parent: str = "baseline", **fields) -> dict:
    return {"name": name, "parent": parent, "desc": f"{name} attempt", "axis": "weight_averaging",
            "kind": "regularization", "constants": {"EMA_DECAY": 0.998}, "hparams": {}, **fields}


@pytest.fixture(scope="module")
def dev_cell():
    return load_script("dev_cell")


@pytest.fixture(scope="module")
def dev_loop():
    return load_script("dev_loop")


@pytest.fixture
def cell(dev_cell, tmp_path) -> Path:
    return build_git_dev_cell(dev_cell, tmp_path)


@pytest.fixture
def recipes(dev_loop, monkeypatch) -> list[dict]:
    """The instantiate_recipe calls the driver made; the real template is another module's job."""
    calls: list[dict] = []
    monkeypatch.setattr(dev_loop, "_instantiate_recipe", fake_instantiate_recipe(calls))
    return calls


def _repo(cell: Path) -> Path:
    return cell.parents[3]


def _run(dev_loop, cell: Path, entries, emit=lambda line: None):
    path = cell / "batch.json"
    path.write_text(json.dumps(entries))
    return dev_loop.run_batch(cell, path, emit=emit)


def _graph(cell: Path) -> dict:
    return json.loads((cell / "automil" / "graph.json").read_text())


def _queued(cell: Path, node_id: str) -> dict:
    return json.loads((cell / "automil" / "orchestrator" / "queue" / f"{node_id}.json").read_text())


def _ledger(cell: Path) -> dict:
    return json.loads((cell / "automil" / "dev_batch.json").read_text())


def _submitted_child(dev_loop, cell: Path, batch: list[dict]) -> None:
    """Root submitted, finished by the daemon, then the child proposed and submitted."""
    _run(dev_loop, cell, batch)
    finish_node(cell, "node_0001", ROOT_FOLDS)
    _run(dev_loop, cell, batch)


def test_a_fresh_cell_gets_its_graph_and_root_and_the_children_wait_for_the_root(dev_loop, recipes, cell):
    outcomes = _run(dev_loop, cell, [ROOT, attempt("ema")])
    assert [(o.name, o.action) for o in outcomes] == [("baseline", "submitted"), ("ema", "skip")]
    assert "node_0001" in outcomes[1].detail and "running" in outcomes[1].detail

    graph = _graph(cell)
    scoring = graph["meta"]["scoring"]  # frozen from the cell config at the graph's first write
    assert (scoring["accept_margin"], scoring["formula"]) == (0.015, "val_auc_smooth")
    root = graph["nodes"]["node_0001"]
    assert (root["type"], root["status"], root["parent_id"]) == ("proposed", "running", None)
    assert root["techniques"] == ["recipe:baseline"] and root["description"] == "baseline: native baseline"
    spec = _queued(cell, "node_0001")
    assert shlex.split(spec["run_command_override"]) == ["--hparams", "{}"]
    assert spec["admissibility"]["candidate_class"] == "config-only"
    assert recipes == [] and not list((cell / "automil/variants/_policies").glob("*.py"))
    assert _ledger(cell) == {"baseline": "node_0001"}


def test_a_child_is_proposed_written_and_submitted_with_the_arm_baked_into_its_constants(dev_loop, recipes, cell):
    batch = [ROOT, attempt("ema", hparams={"dropout": 0.7})]
    _submitted_child(dev_loop, cell, batch)

    node = _graph(cell)["nodes"]["node_0002"]
    assert (node["parent_id"], node["kind"], node["type"], node["status"]) == (
        "node_0001", "regularization", "proposed", "running")
    assert node["techniques"] == ["recipe:ema"] and node["description"] == "ema: ema attempt"
    assert node["metadata"]["axis"] == "weight_averaging" and node["metadata"]["predicted_delta"] == 0.0

    (call,) = recipes
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=_repo(cell), capture_output=True,
                          text=True, check=True).stdout.strip()
    assert (call["name"], call["node_id"], call["base_commit"]) == ("ema", "node_0002", head)
    assert call["constants"] == {"EMA_DECAY": 0.998, "ARM": "abmil"}
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00", call["created_at"])
    assert call["template_path"] == _repo(cell) / "benchmarks/experiments/dev/_recipes/composite.py"

    policy = cell / "automil/variants/_policies/ema.py"
    assert policy.read_text() == POLICY_TEXT.format(
        name="ema", base_commit=head, node_id="node_0002", created_at=call["created_at"])
    spec = _queued(cell, "node_0002")
    assert list(spec["overlay_manifest"]) == [policy.relative_to(_repo(cell)).as_posix()]
    assert spec["run_command_override"] == shlex.join(
        ["--policy-variant", "ema", "--hparams", json.dumps({"dropout": 0.7})])
    assert spec["graph_metadata"]["parent_id"] == "node_0001"
    assert spec["admissibility"]["candidate_class"] == "train-only-source"
    assert _ledger(cell) == {"baseline": "node_0001", "ema": "node_0002"}


def test_running_the_same_batch_again_changes_nothing_and_reports_each_skip(dev_loop, recipes, cell):
    batch = [ROOT, attempt("ema")]
    _submitted_child(dev_loop, cell, batch)
    queue = cell / "automil/orchestrator/queue"
    before = {path.name: path.stat().st_mtime_ns for path in queue.iterdir()}
    graph_before = _graph(cell)

    lines: list[str] = []
    outcomes = _run(dev_loop, cell, batch, emit=lines.append)
    assert [o.action for o in outcomes] == ["skip", "skip"] and len(recipes) == 1
    assert len(lines) == 2 and all(line.startswith("skip ") for line in lines)
    assert {path.name: path.stat().st_mtime_ns for path in queue.iterdir()} == before
    assert _graph(cell)["nodes"] == graph_before["nodes"]


def test_a_proposal_whose_policy_or_submit_failed_is_resumed_not_proposed_again(dev_loop, cell, monkeypatch):
    seed_executed_root(cell)

    def broken(*args, **kwargs):
        raise dev_loop.DevLoopError("the recipe template is broken")

    monkeypatch.setattr(dev_loop, "_instantiate_recipe", broken)
    with pytest.raises(dev_loop.DevLoopError, match="template is broken"):
        _run(dev_loop, cell, [attempt("ema")])
    stranded = _graph(cell)["nodes"]["node_0002"]
    assert (stranded["type"], stranded["status"]) == ("proposed", "pending")

    calls: list[dict] = []
    monkeypatch.setattr(dev_loop, "_instantiate_recipe", fake_instantiate_recipe(calls))
    (outcome,) = _run(dev_loop, cell, [attempt("ema")])
    assert (outcome.action, outcome.node_id) == ("resumed", "node_0002")
    graph = _graph(cell)
    assert sorted(graph["nodes"]) == ["node_0001", "node_0002"]  # no second proposal
    assert graph["nodes"]["node_0002"]["status"] == "running" and calls[0]["node_id"] == "node_0002"


@pytest.mark.skipif(
    not (SCRIPTS / "dev_recipe.py").is_file() or not (REPO_ROOT / "benchmarks/experiments/dev/_recipes/composite.py").is_file(),
    reason="dev_recipe.py and the recipe template are not in this tree",
)
def test_the_real_recipe_template_goes_through_the_whole_path(dev_loop, cell):
    """No fake: dev_recipe's own instantiate_recipe, the framework's validators and the submit."""
    template = _repo(cell) / dev_loop.RECIPE_TEMPLATE
    template.parent.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / dev_loop.RECIPE_TEMPLATE, template)
    seed_executed_root(cell)

    (outcome,) = _run(dev_loop, cell, [attempt("ema", constants={"EMA_DECAY": 0.998})])
    assert (outcome.action, outcome.node_id) == ("submitted", "node_0002")
    spec = _queued(cell, "node_0002")
    assert spec["admissibility"]["candidate_class"] == "train-only-source"
    assert (cell / "automil/variants/_policies/ema.py").is_file()


def test_a_refused_submission_surfaces_the_framework_message_and_stops_the_batch(dev_loop, cell, monkeypatch):
    seed_executed_root(cell)
    monkeypatch.setattr(dev_loop, "_instantiate_recipe", fake_instantiate_recipe([], text="import os\n"))
    with pytest.raises(dev_loop.DevLoopError, match="submit") as raised:
        _run(dev_loop, cell, [attempt("bad"), attempt("later")])
    assert "Refusing to submit" in str(raised.value)
    nodes = _graph(cell)["nodes"]
    assert sorted(nodes) == ["node_0001", "node_0002"]  # "later" was never proposed
    assert nodes["node_0002"]["status"] == "pending"
    assert not (cell / "automil/orchestrator/queue/node_0002.json").exists()


def test_children_of_an_unknown_or_failed_parent_are_skipped_with_a_reason(dev_loop, recipes, cell):
    seed_executed_root(cell)
    with locked_update(cell / "automil" / "graph.json") as graph:
        graph.add_executed("node_0001", "crashy: a crashed attempt", ["recipe:crashy"],
                           {"primary_value": 0.0}, status="crash")
    outcomes = _run(dev_loop, cell, [attempt("a", parent="nowhere"), attempt("b", parent="crashy")])
    assert [o.action for o in outcomes] == ["skip", "skip"]
    assert "not in the graph" in outcomes[0].detail and "crash" in outcomes[1].detail
    assert recipes == [] and sorted(_graph(cell)["nodes"]) == ["node_0001", "node_0002"]


def test_hparams_reach_the_queue_spec_intact_through_the_override_quoting(dev_loop, recipes, cell):
    hparams = {"batch_size": 4, "learning_rate": 0.0002}
    _submitted_child(dev_loop, cell, [ROOT, attempt("ema", hparams=hparams)])
    tokens = shlex.split(_queued(cell, "node_0002")["run_command_override"])
    assert tokens[:3] == ["--policy-variant", "ema", "--hparams"]
    assert json.loads(tokens[3]) == hparams and len(tokens) == 4


def test_a_batch_file_that_is_not_json_is_refused_before_anything_runs(dev_loop, recipes, cell):
    path = cell / "batch.json"
    path.write_text("[{")
    with pytest.raises(dev_loop.DevLoopError, match="cannot read batch file"):
        dev_loop.run_batch(cell, path, emit=lambda line: None)
    assert not (cell / "automil" / "graph.json").exists()


def test_a_cell_outside_a_git_repository_is_refused(dev_loop, recipes, cell):
    shutil.rmtree(_repo(cell) / ".git")
    with pytest.raises(dev_loop.DevLoopError, match="not inside a git repository"):
        _run(dev_loop, cell, [ROOT])
    assert not (cell / "automil" / "graph.json").exists()


def test_a_second_root_is_refused(dev_loop, recipes, cell):
    seed_executed_root(cell)
    with pytest.raises(dev_loop.DevLoopError, match="already has a root"):
        _run(dev_loop, cell, [{**ROOT, "name": "another_root"}])


@pytest.mark.parametrize(("entries", "message"), [
    ([ROOT, ROOT], "duplicate"),
    ([attempt("_hidden")], "name"),
    ([attempt("node_0003")], "name"),
    ([ROOT, {**ROOT, "name": "other_root"}], "one root"),
    ([{**ROOT, "constants": {"EMA_DECAY": 0.9}}], "root entry"),
    ([{**ROOT, "hparams": {"lr": 0.001}}], "root entry.*hparams must be"),  # the root is the native recipe
    ([attempt("a", constants=None)], "needs constants"),
    ([attempt("a", kind="architecture")], "kind"),
    ([attempt("a", parent="b"), attempt("b")], "earlier"),
    ([attempt("a", constants={"ARM": "clam_mb"})], "ARM"),
    ([{**attempt("a"), "hparam": {}}], "unknown key"),
    ([attempt("a", hparams=[1])], "hparams"),
    ([], "empty"),
    ({"name": "x"}, "list"),
])
def test_an_invalid_batch_is_refused_before_anything_runs(dev_loop, recipes, cell, entries, message):
    with pytest.raises(dev_loop.DevLoopError, match=message):
        _run(dev_loop, cell, entries)
    assert not (cell / "automil" / "graph.json").exists() and recipes == []


def _set_class_count(cell: Path, count: int | None) -> None:
    """Rewrite the cell's ``task.num_classes``; ``None`` removes it."""
    path = cell / "automil" / "config.yaml"
    config = yaml.safe_load(path.read_text())
    task = {key: value for key, value in config["task"].items() if key != "num_classes"}
    path.write_text(yaml.safe_dump({**config, "task": task if count is None else {**task, "num_classes": count}},
                                   sort_keys=False))


@pytest.mark.parametrize("constants", [{"POSITIVE_SHARE": 0.3}, {"CLASS_WEIGHTS": [1.0, 2.0, 3.0]}])
def test_class_weights_are_refused_on_a_cell_with_more_than_two_classes(dev_loop, recipes, cell, constants):
    """The submit-time policy smoke drives two classes, so a multi-class weight table cannot pass it and
    POSITIVE_SHARE would crash at the first real step."""
    _set_class_count(cell, 6)
    (name,) = constants
    with pytest.raises(dev_loop.DevLoopError, match=f"{name}.*6 classes"):
        _run(dev_loop, cell, [ROOT, attempt("w", constants=constants)])
    assert not (cell / "automil" / "graph.json").exists() and recipes == []


def test_class_weights_are_accepted_on_a_two_class_cell_and_a_weight_that_is_off_anywhere(dev_loop, recipes, cell):
    weights = {"POSITIVE_SHARE": 0.3, "CLASS_WEIGHTS": [1.0, 2.0]}
    assert [o.action for o in _run(dev_loop, cell, [ROOT, attempt("w", constants=weights)])] == ["submitted", "skip"]
    _set_class_count(cell, 6)  # the root is in the graph now, so the batch parses and both entries skip
    off = {"POSITIVE_SHARE": None, "CLASS_WEIGHTS": None}
    assert [o.action for o in _run(dev_loop, cell, [ROOT, attempt("w", constants=off)])] == ["skip", "skip"]


def test_a_cell_whose_config_names_no_class_count_cannot_run_a_batch(dev_loop, recipes, cell):
    _set_class_count(cell, None)
    with pytest.raises(dev_loop.DevLoopError, match="task.num_classes"):
        _run(dev_loop, cell, [ROOT])
    assert not (cell / "automil" / "graph.json").exists()


# --- F3: attempts run the root's code ----------------------------------------------------------------


def test_an_attempt_is_refused_when_head_moved_since_the_root_ran_and_nothing_is_proposed(dev_loop, recipes, cell):
    seed_executed_root(cell)
    root_sha = git_head(_repo(cell))
    head = commit_file(_repo(cell), "later.txt")
    with pytest.raises(dev_loop.DevLoopError) as raised:
        _run(dev_loop, cell, [attempt("ema")])
    assert str(raised.value) == (
        f"the root ran at {root_sha}; HEAD is {head}; attempts must run the root's code"
    )
    assert sorted(_graph(cell)["nodes"]) == ["node_0001"]  # no proposal was left behind
    assert recipes == [] and not list((cell / "automil/variants/_policies").glob("*.py"))
    assert not (cell / "automil" / "dev_batch.json").exists()


def test_a_stranded_proposal_is_not_resumed_on_other_code_than_the_roots(dev_loop, cell, monkeypatch):
    seed_executed_root(cell)

    def broken(*args, **kwargs):
        raise dev_loop.DevLoopError("the recipe template is broken")

    monkeypatch.setattr(dev_loop, "_instantiate_recipe", broken)
    with pytest.raises(dev_loop.DevLoopError, match="template is broken"):
        _run(dev_loop, cell, [attempt("ema")])
    commit_file(_repo(cell), "later.txt")
    with pytest.raises(dev_loop.DevLoopError, match="the root ran at"):
        _run(dev_loop, cell, [attempt("ema")])
    assert _graph(cell)["nodes"]["node_0002"]["status"] == "pending"  # still stranded, not submitted


def test_the_root_is_the_parentless_executed_node_whatever_its_id(dev_loop, recipes, cell):
    seed_executed_root(cell)
    archive = cell / "automil" / "orchestrator" / "archive"
    graph = _graph(cell)
    renamed = {**graph["nodes"]["node_0001"], "id": "node_0007"}
    (cell / "automil" / "graph.json").write_text(json.dumps({**graph, "nodes": {"node_0007": renamed}}))
    (archive / "node_0001").rename(archive / "node_0007")
    root_sha = git_head(_repo(cell))
    commit_file(_repo(cell), "later.txt")
    with pytest.raises(dev_loop.DevLoopError, match=f"the root ran at {root_sha}"):
        _run(dev_loop, cell, [attempt("ema", parent="node_0007")])


def test_a_root_without_an_archived_spec_cannot_vouch_for_its_code(dev_loop, recipes, cell):
    seed_executed_root(cell)
    (cell / "automil/orchestrator/archive/node_0001/spec.json").unlink()
    with pytest.raises(dev_loop.DevLoopError, match=r"node_0001.*spec\.json"):
        _run(dev_loop, cell, [attempt("ema")])
    assert sorted(_graph(cell)["nodes"]) == ["node_0001"]


def test_batch_cli_prints_a_line_per_attempt_resolves_node_id_parents_and_exits_2_on_a_refusal(
    dev_cell, recipes, cell, capsys,
):
    seed_executed_root(cell)
    path = cell / "batch.json"
    path.write_text(json.dumps([attempt("ema", parent="node_0001")]))
    dev_cell.main(["batch", str(cell), str(path)])
    assert "submitted ema: node_0002" in capsys.readouterr().out
    dev_cell.main(["batch", str(cell), str(path)])
    assert "skip ema" in capsys.readouterr().out

    path.write_text("[]")
    with pytest.raises(SystemExit) as raised:
        dev_cell.main(["batch", str(cell), str(path)])
    assert raised.value.code == 2 and "empty" in capsys.readouterr().err
