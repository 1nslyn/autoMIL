"""Graph nodes must carry their budget-cell identity (CELL-1).

The experiment tree lives in ``graph.json`` and the budget cells live in
``automil/cells/<cell_id>.json``, but nothing joined them: ``cell_id`` appeared
nowhere in ``graph.py``. "How many evaluations did cell X get?" was therefore
unanswerable from the graph, which blocks both the eval-count budget (H-2) and
per-cell keying of the paper's figures.

These tests pin: submit stamps the node, the completion writer backfills nodes
created by other submission paths, and the accessors tolerate legacy nodes that
have no cell identity at all.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from automil.cli import main
from automil.graph import ExperimentGraph


@pytest.fixture
def cli_runner():
    return CliRunner()


def _init_git_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=path, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=path, capture_output=True)
    (path / "README.md").write_text("# Test\n")
    subprocess.run(["git", "add", "."], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, capture_output=True, check=True)


def _use_wall_clock(tmp_path: Path) -> None:
    config_path = tmp_path / "automil" / "config.yaml"
    config = yaml.safe_load(config_path.read_text()) or {}
    config.setdefault("cap", {})["mode"] = "wall_clock"
    config_path.write_text(yaml.safe_dump(config))


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------


def _graph_with_nodes(tmp_path: Path, nodes: dict) -> ExperimentGraph:
    g = ExperimentGraph(path=tmp_path / "graph.json")
    g.nodes.update(nodes)
    return g


def test_nodes_in_cell_returns_only_that_cells_nodes(tmp_path: Path):
    g = _graph_with_nodes(tmp_path, {
        "node_0001": {"id": "node_0001", "type": "executed", "cell_id": "cellA"},
        "node_0002": {"id": "node_0002", "type": "executed", "cell_id": "cellB"},
        "node_0003": {"id": "node_0003", "type": "proposed", "cell_id": "cellA"},
    })

    ids = [n["id"] for n in g.nodes_in_cell("cellA")]
    assert ids == ["node_0001", "node_0003"], ids
    assert g.count_in_cell("cellA") == 2
    assert g.count_in_cell("cellB") == 1


def test_count_in_cell_can_restrict_to_executed_nodes(tmp_path: Path):
    """The evaluation count is the executed subset — proposals are not evals."""
    g = _graph_with_nodes(tmp_path, {
        "node_0001": {"id": "node_0001", "type": "executed", "cell_id": "cellA"},
        "node_0002": {"id": "node_0002", "type": "proposed", "cell_id": "cellA"},
    })

    assert g.count_in_cell("cellA") == 2
    assert g.count_in_cell("cellA", executed_only=True) == 1


def test_legacy_nodes_without_cell_id_never_match_and_never_crash(tmp_path: Path):
    """Backward compat: pre-CELL-1 nodes have no cell_id at all."""
    g = _graph_with_nodes(tmp_path, {
        "node_0001": {"id": "node_0001", "type": "executed"},          # legacy
        "node_0002": {"id": "node_0002", "type": "executed", "cell_id": None},
        "node_0003": {"id": "node_0003", "type": "executed", "cell_id": "cellA"},
    })

    assert [n["id"] for n in g.nodes_in_cell("cellA")] == ["node_0003"]
    assert g.count_in_cell("cellA") == 1
    # A lookup keyed on the legacy absence must not sweep every untagged node in.
    assert g.nodes_in_cell("") == []
    assert g.count_in_cell("nonexistent") == 0


def test_nodes_in_cell_also_resolves_metadata_cell_id(tmp_path: Path):
    """Gate-eval children carry their cell under metadata (gate/evaluate.py)."""
    g = _graph_with_nodes(tmp_path, {
        "node_0001": {"id": "node_0001", "type": "executed", "cell_id": "cellA"},
        "node_0002": {
            "id": "node_0002", "type": "gate_eval",
            "metadata": {"cell_id": "cellA", "gate_eval": True},
        },
    })

    assert [n["id"] for n in g.nodes_in_cell("cellA")] == ["node_0001", "node_0002"]


def test_nodes_in_cell_tolerates_a_non_dict_metadata_value(tmp_path: Path):
    g = _graph_with_nodes(tmp_path, {
        "node_0001": {"id": "node_0001", "type": "executed", "metadata": "corrupt"},
    })
    assert g.nodes_in_cell("cellA") == []


# ---------------------------------------------------------------------------
# submit stamps the node
# ---------------------------------------------------------------------------


def _submit(cli_runner, tmp_path: Path, node_id: str, contents: str) -> str:
    """Submit `node_id` and return the cell_id stamped on its queue spec."""
    (tmp_path / "model.py").write_text(contents)
    result = cli_runner.invoke(
        main,
        ["submit", "--node", node_id, "--desc", f"desc {node_id}", "--files", "model.py",
         "--mil-model", "clam_sb"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    spec = json.loads(
        (tmp_path / "automil" / "orchestrator" / "queue" / f"{node_id}.json").read_text()
    )
    return spec["metadata"]["cell_id"]


def test_submit_stamps_cell_id_on_a_pre_existing_proposal(cli_runner, tmp_path, monkeypatch):
    """propose → submit (the agent's normal flow): the proposal gets tagged."""
    _init_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    cli_runner.invoke(main, ["init"])
    _use_wall_clock(tmp_path)

    proposed = cli_runner.invoke(
        main, ["propose", "--parent", "root", "--desc", "a proposal",
               "--techniques", "baseline"],
        catch_exceptions=False,
    )
    assert proposed.exit_code == 0, proposed.output
    node_id = next(iter(ExperimentGraph(path=tmp_path / "automil" / "graph.json").nodes))

    cell_id = _submit(cli_runner, tmp_path, node_id, "print('v1')\n")

    graph = ExperimentGraph(path=tmp_path / "automil" / "graph.json")
    assert graph.get_node(node_id).get("cell_id") == cell_id, (
        "CELL-1: the graph node must carry the same cell_id as its queue spec, "
        "otherwise per-cell evaluation counts are unanswerable from the graph."
    )
    assert graph.count_in_cell(cell_id) == 1


def test_submit_stamps_cell_id_on_a_node_it_registers_itself(cli_runner, tmp_path, monkeypatch):
    """Submitting an id the graph has never seen registers AND tags it."""
    _init_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    cli_runner.invoke(main, ["init"])
    _use_wall_clock(tmp_path)
    # A prior propose is what creates graph.json; the second submit below then
    # exercises submit's own add_proposed branch.
    cli_runner.invoke(main, ["propose", "--parent", "root", "--desc", "seed"],
                      catch_exceptions=False)

    cell_id = _submit(cli_runner, tmp_path, "node_0002", "print('v2')\n")

    graph = ExperimentGraph(path=tmp_path / "automil" / "graph.json")
    node = graph.get_node("node_0002")
    assert node is not None, "submit must register an unknown node id"
    assert node.get("cell_id") == cell_id
    assert [n["id"] for n in graph.nodes_in_cell(cell_id)] == ["node_0002"]


# ---------------------------------------------------------------------------
# terminal_writer backfill (covers Backend.submit paths that bypass the CLI)
# ---------------------------------------------------------------------------


def test_terminal_writer_backfills_cell_id_from_the_spec(tmp_path: Path):
    """A node created outside `automil submit` is tagged at completion."""
    from automil.terminal_writer import write_terminal_state

    graph_path = tmp_path / "graph.json"
    graph = ExperimentGraph(path=graph_path)
    graph.nodes["node_0001"] = {
        "id": "node_0001", "type": "running", "status": "running",
        "description": "backend-submitted", "techniques": [], "primary_value": 0.0,
    }
    graph.save()

    write_terminal_state(
        node_id="node_0001",
        result={"status": "completed", "primary_value": 0.7, "metrics": {"val_auc": 0.7}},
        graph=graph,
        completed_dir=tmp_path / "completed",
        archive_dir=tmp_path / "archive" / "node_0001",
        results_tsv_writer=lambda *a, **k: None,
        spec={"description": "backend-submitted", "metadata": {"cell_id": "cellZ"}},
        elapsed_s=1.0,
        gpu_id=0,
    )

    reloaded = ExperimentGraph(path=graph_path)
    assert reloaded.get_node("node_0001").get("cell_id") == "cellZ"
    assert reloaded.count_in_cell("cellZ", executed_only=True) == 1


def test_terminal_writer_does_not_overwrite_an_existing_cell_id(tmp_path: Path):
    """Submit-time identity wins — completion only fills a gap."""
    from automil.terminal_writer import write_terminal_state

    graph_path = tmp_path / "graph.json"
    graph = ExperimentGraph(path=graph_path)
    graph.nodes["node_0001"] = {
        "id": "node_0001", "type": "running", "status": "running",
        "description": "x", "techniques": [], "primary_value": 0.0,
        "cell_id": "original",
    }
    graph.save()

    write_terminal_state(
        node_id="node_0001",
        result={"status": "completed", "primary_value": 0.7, "metrics": {"val_auc": 0.7}},
        graph=graph,
        completed_dir=tmp_path / "completed",
        archive_dir=tmp_path / "archive" / "node_0001",
        results_tsv_writer=lambda *a, **k: None,
        spec={"description": "x", "metadata": {"cell_id": "different"}},
        elapsed_s=1.0,
        gpu_id=0,
    )

    reloaded = ExperimentGraph(path=graph_path)
    assert reloaded.get_node("node_0001")["cell_id"] == "original"
