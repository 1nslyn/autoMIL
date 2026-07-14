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
            "best_composite": 0.87,
            "scoring": {"accept_margin": 0.0},
        },
        "nodes": {
            "node_0001": {
                "id": "node_0001", "type": "executed", "status": "keep",
                "composite": 0.87, "metrics": {"val_auc": 0.9, "val_bacc": 0.84},
            },
        },
        "technique_stats": {},
    }
    (adir / "graph.json").write_text(json.dumps(graph))
    node_arch = adir / "orchestrator" / "archive" / "node_0001"
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
    assert "val_composite=0.8700" in result.output
    assert "test_auc=0.8300" in result.output
    assert "test_bacc=0.8000" in result.output


def test_certify_missing_sidecar_is_graceful(tmp_path, monkeypatch):
    adir = _setup(tmp_path)
    (adir / "orchestrator" / "archive" / "node_0001" / "certify.json").unlink()
    monkeypatch.chdir(adir.parent)
    result = CliRunner().invoke(main, ["certify"])
    assert result.exit_code == 0, result.output
    assert "no certify.json" in result.output
