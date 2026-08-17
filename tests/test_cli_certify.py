"""certify command — the one sanctioned held-out TEST read (val-firewall).

certify reveals the sealed archive/<node>/certify.json for the val-selected
node(s). It must surface test (the honest final number) without that number ever
having driven selection.
"""
from __future__ import annotations

import json

from click.testing import CliRunner

from automil.cli import main


def _setup(tmp_path):
    """A project with automil/graph.json + a sealed certify.json for one keep node."""
    adir = tmp_path / "automil"
    adir.mkdir()
    (adir / "config.yaml").write_text("run:\n  script: train.py\n")
    graph = {
        "schema_version": 2,
        "meta": {
            "best_node_id": "node_0001",
            "best_primary_value": 0.87,
            "scoring": {"accept_margin": 0.0},
        },
        "nodes": {
            "node_0001": {
                "id": "node_0001", "type": "executed", "status": "keep",
                "primary_value": 0.87, "metrics": {"val_auc": 0.9, "val_bacc": 0.84},
            },
        },
        "technique_stats": {},
    }
    (adir / "graph.json").write_text(json.dumps(graph))
    node_arch = adir / "orchestrator" / "archive" / "node_0001" / "certify"
    node_arch.mkdir(parents=True)
    (node_arch / "certify.json").write_text(
        json.dumps({"held_out": {"test_auc": 0.83, "test_bacc": 0.80}})
    )
    return adir


def test_certify_reveals_sealed_held_out_test(tmp_path, monkeypatch):
    adir = _setup(tmp_path)
    monkeypatch.chdir(adir.parent)
    result = CliRunner().invoke(main, ["certify"])
    assert result.exit_code == 0, result.output
    assert "node_0001" in result.output
    assert "primary_value=0.8700" in result.output
    assert "test_auc=0.8300" in result.output
    assert "test_bacc=0.8000" in result.output


def test_certify_missing_sidecar_is_graceful(tmp_path, monkeypatch):
    adir = _setup(tmp_path)
    (adir / "orchestrator" / "archive" / "node_0001" / "certify" / "certify.json").unlink()
    monkeypatch.chdir(adir.parent)
    result = CliRunner().invoke(main, ["certify"])
    assert result.exit_code == 0, result.output
    assert "no certify.json" in result.output


def _setup_many(tmp_path, primary_values=(0.90, 0.88, 0.86)):
    """A project with several keep nodes, each carrying a sealed certify.json."""
    adir = tmp_path / "automil"
    adir.mkdir()
    (adir / "config.yaml").write_text("run:\n  script: train.py\n")
    nodes = {}
    for rank, comp in enumerate(primary_values, start=1):
        nid = f"node_{rank:04d}"
        nodes[nid] = {
            "id": nid, "type": "executed", "status": "keep",
            "primary_value": comp, "metrics": {"val_auc": comp},
        }
        na = adir / "orchestrator" / "archive" / nid / "certify"
        na.mkdir(parents=True)
        (na / "certify.json").write_text(
            json.dumps({"held_out": {"test_auc": 0.5 + rank / 100}})
        )
    (adir / "graph.json").write_text(json.dumps({
        "schema_version": 2,
        "meta": {"best_node_id": "node_0001", "scoring": {"accept_margin": 0.0}},
        "nodes": nodes,
        "technique_stats": {},
    }))
    return adir


class TestTopKIsSelectionOnTest:
    """M-8 (audit 2026-07-23): ``--top-k`` > 1 re-opens test-set selection.

    The val-firewall's whole point is that test is read once, for the node
    validation already chose. Unsealing K nodes and reading their test numbers
    is selection on test by another name — it was permitted with nothing but a
    log warning. K > 1 now requires an explicit acknowledgement flag.
    """

    def test_top_k_above_one_is_refused_without_the_flag(self, tmp_path, monkeypatch):
        adir = _setup_many(tmp_path)
        monkeypatch.chdir(adir.parent)
        result = CliRunner().invoke(main, ["certify", "--top-k", "3"])
        assert result.exit_code != 0
        # Refusing must not leak while refusing.
        assert "test_auc" not in result.output
        assert "0.5100" not in result.output

    def test_refusal_names_the_cost_and_the_escape_hatch(self, tmp_path, monkeypatch):
        adir = _setup_many(tmp_path)
        monkeypatch.chdir(adir.parent)
        result = CliRunner().invoke(main, ["certify", "--top-k", "2"])
        assert "selection on test" in result.output
        assert "--unseal-multiple" in result.output

    def test_top_k_above_one_reveals_each_node_with_the_flag(self, tmp_path, monkeypatch):
        adir = _setup_many(tmp_path)
        monkeypatch.chdir(adir.parent)
        result = CliRunner().invoke(
            main, ["certify", "--top-k", "2", "--unseal-multiple"],
        )
        assert result.exit_code == 0, result.output
        assert "node_0001" in result.output
        assert "node_0002" in result.output
        assert "node_0003" not in result.output

    def test_default_reveals_exactly_one_node(self, tmp_path, monkeypatch):
        adir = _setup_many(tmp_path)
        monkeypatch.chdir(adir.parent)
        result = CliRunner().invoke(main, ["certify"])
        assert result.exit_code == 0, result.output
        assert "node_0001" in result.output
        assert "node_0002" not in result.output
        assert "node_0003" not in result.output

    def test_flag_alone_does_not_widen_the_default(self, tmp_path, monkeypatch):
        """``--unseal-multiple`` is an acknowledgement, not a request for K>1."""
        adir = _setup_many(tmp_path)
        monkeypatch.chdir(adir.parent)
        result = CliRunner().invoke(main, ["certify", "--unseal-multiple"])
        assert result.exit_code == 0, result.output
        assert "node_0002" not in result.output

    def test_explicit_single_node_still_works(self, tmp_path, monkeypatch):
        """The targeted one-node read is legitimate and must stay unblocked."""
        adir = _setup_many(tmp_path)
        monkeypatch.chdir(adir.parent)
        result = CliRunner().invoke(main, ["certify", "--node", "node_0002"])
        assert result.exit_code == 0, result.output
        assert "node_0002" in result.output

    def test_node_combined_with_top_k_is_a_usage_error(self, tmp_path, monkeypatch):
        """``--top-k`` was silently ignored alongside ``--node``; an operator
        who passes both must not believe they received K nodes."""
        adir = _setup_many(tmp_path)
        monkeypatch.chdir(adir.parent)
        result = CliRunner().invoke(
            main, ["certify", "--node", "node_0002", "--top-k", "3"],
        )
        assert result.exit_code != 0
        assert "test_auc" not in result.output

    def test_non_positive_top_k_is_a_usage_error(self, tmp_path, monkeypatch):
        """``max(1, top_k)`` silently rewrote ``--top-k 0`` to 1."""
        adir = _setup_many(tmp_path)
        monkeypatch.chdir(adir.parent)
        result = CliRunner().invoke(main, ["certify", "--top-k", "0"])
        assert result.exit_code != 0


def test_certify_default_matches_canonical_best_on_tie(tmp_path, monkeypatch):
    """M1: on tied primary_values, certify's default node matches best_node (D-12: smaller id)."""
    adir = tmp_path / "automil"
    adir.mkdir()
    (adir / "config.yaml").write_text("run:\n  script: train.py\n")
    (adir / "graph.json").write_text(json.dumps({
        "schema_version": 2,
        "meta": {"best_node_id": "node_0001", "scoring": {"accept_margin": 0.0}},
        "nodes": {
            "node_0005": {"id": "node_0005", "type": "executed", "status": "keep",
                          "primary_value": 0.87, "metrics": {"val_auc": 0.9}},
            "node_0001": {"id": "node_0001", "type": "executed", "status": "keep",
                          "primary_value": 0.87, "metrics": {"val_auc": 0.9}},
        },
        "technique_stats": {},
    }))
    for nid in ("node_0001", "node_0005"):
        na = adir / "orchestrator" / "archive" / nid / "certify"
        na.mkdir(parents=True)
        (na / "certify.json").write_text(json.dumps({"held_out": {"test_auc": 0.8}}))
    monkeypatch.chdir(adir.parent)
    result = CliRunner().invoke(main, ["certify"])   # default: top-1
    assert result.exit_code == 0, result.output
    assert "node_0001" in result.output          # D-12: smaller id wins the tie
    assert "node_0005" not in result.output
