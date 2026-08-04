"""The daemon revalidates candidate policy before creating a worktree."""
from __future__ import annotations

import json
import hashlib
from unittest.mock import MagicMock

import yaml


def _valid_preserving_spec(orch, node_id: str) -> dict:
    from automil.admissibility import load_candidate_policy

    archive = orch.archive_dir / node_id
    (archive / "recipes").mkdir(parents=True)
    (archive / "recipes" / "cosine.py").write_text("NAME = 'cosine'\n")
    policy = load_candidate_policy(orch.automil_dir)
    verdict = policy.classify(["recipes/cosine.py"])
    queue_file = orch.queue_dir / f"{node_id}.json"
    queue_file.parent.mkdir(parents=True, exist_ok=True)
    spec = {
        "id": node_id,
        "base_commit": "deadbeef",
        "overlay_dir": f"archive/{node_id}",
        "overlay_manifest": {"recipes/cosine.py": "sha256:placeholder"},
        "deletions": [],
        "admissibility": verdict.to_dict(),
        "base_run_command_sha256": hashlib.sha256(b"python train.py").hexdigest(),
        "_file": str(queue_file),
    }
    queue_file.write_text(json.dumps(spec))
    return spec


def test_architecture_preserving_legacy_spec_fails_before_worktree(tmp_path):
    from automil.orchestrator import ExperimentOrchestrator

    adir = tmp_path / "automil"
    adir.mkdir()
    (tmp_path / ".git").mkdir()
    (adir / "config.yaml").write_text(yaml.safe_dump({
        "registry": {
            "mode": "architecture-preserving",
            "protected": ["models/**"],
            "allowed_override_options": ["--hparams"],
            "allowed_variant_kinds": ["policy"],
        },
        "files": {"editable": ["recipes/**"]},
        "run": {"command": "python train.py"},
    }))
    (adir / "graph.json").write_text('{"nodes": {}, "meta": {"next_id": 1}}')
    orch = ExperimentOrchestrator(project_root=tmp_path, automil_dir=adir)

    node_id = "node_0001"
    archive = orch.archive_dir / node_id
    (archive / "recipes").mkdir(parents=True)
    (archive / "recipes" / "cosine.py").write_text("NAME = 'cosine'\n")
    queue_file = orch.queue_dir / f"{node_id}.json"
    queue_file.parent.mkdir(parents=True, exist_ok=True)
    spec = {
        "id": node_id,
        "base_commit": "deadbeef",
        "overlay_dir": f"archive/{node_id}",
        "overlay_manifest": {"recipes/cosine.py": "sha256:placeholder"},
        "deletions": [],
        # Deliberately no admissibility record: old CLI/manual queue bypass.
        "_file": str(queue_file),
    }
    queue_file.write_text(json.dumps(spec))

    orch.runner = MagicMock()
    orch._mark_crashed = MagicMock()
    orch._launch(spec, gpu_id=0)

    orch.runner.create_worktree.assert_not_called()
    orch._mark_crashed.assert_called_once()
    assert "admissibility" in orch._mark_crashed.call_args.args[2]
    assert not queue_file.exists()


def test_base_command_drift_fails_before_worktree(tmp_path):
    from automil.orchestrator import ExperimentOrchestrator

    adir = tmp_path / "automil"
    adir.mkdir()
    (tmp_path / ".git").mkdir()
    (adir / "config.yaml").write_text(yaml.safe_dump({
        "registry": {
            "mode": "architecture-preserving",
            "protected": ["models/**"],
            "allowed_override_options": ["--hparams"],
            "allowed_variant_kinds": ["policy"],
        },
        "files": {"editable": ["recipes/**"]},
        "run": {"command": "python train.py"},
    }))
    (adir / "graph.json").write_text('{"nodes": {}, "meta": {"next_id": 1}}')
    orch = ExperimentOrchestrator(project_root=tmp_path, automil_dir=adir)
    spec = _valid_preserving_spec(orch, "node_0001")
    spec["base_run_command_sha256"] = hashlib.sha256(b"python other.py").hexdigest()

    orch.runner = MagicMock()
    orch._mark_crashed = MagicMock()
    orch._launch(spec, gpu_id=0)

    orch.runner.create_worktree.assert_not_called()
    assert "base run command changed" in orch._mark_crashed.call_args.args[2]


def test_campaign_manifest_drift_fails_before_worktree(tmp_path):
    from automil.orchestrator import ExperimentOrchestrator

    adir = tmp_path / "automil"
    adir.mkdir()
    (tmp_path / ".git").mkdir()
    (adir / "config.yaml").write_text(yaml.safe_dump({
        "registry": {
            "mode": "architecture-preserving",
            "protected": ["models/**"],
            "allowed_override_options": ["--hparams"],
            "allowed_variant_kinds": ["policy"],
        },
        "files": {"editable": ["recipes/**"]},
        "run": {"command": "python train.py"},
    }))
    (adir / "graph.json").write_text('{"nodes": {}, "meta": {"next_id": 1}}')
    manifest = tmp_path / "campaign" / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text('{"version": 2}\n')
    orch = ExperimentOrchestrator(project_root=tmp_path, automil_dir=adir)
    spec = _valid_preserving_spec(orch, "node_0001")
    spec["metadata"] = {"campaign": {
        "manifest": "campaign/manifest.json",
        "manifest_sha256": hashlib.sha256(b'{"version": 1}\n').hexdigest(),
    }}

    orch.runner = MagicMock()
    orch._mark_crashed = MagicMock()
    orch._launch(spec, gpu_id=0)

    orch.runner.create_worktree.assert_not_called()
    assert "campaign manifest changed" in orch._mark_crashed.call_args.args[2]


def test_campaign_cell_binding_drift_fails_before_worktree(tmp_path):
    from automil.orchestrator import ExperimentOrchestrator

    adir = tmp_path / "automil"
    adir.mkdir()
    (tmp_path / ".git").mkdir()
    (adir / "config.yaml").write_text(yaml.safe_dump({
        "registry": {
            "mode": "architecture-preserving",
            "protected": ["models/**"],
            "allowed_override_options": ["--hparams"],
            "allowed_variant_kinds": ["policy"],
        },
        "files": {"editable": ["recipes/**"]},
        "run": {"command": "python train.py"},
    }))
    (adir / "graph.json").write_text('{"nodes": {}, "meta": {"next_id": 1}}')
    cell = {
        "cell_id": "cell-1",
        "budget_identity": {"cell_id": "budget-1"},
        "commands": {"discovery": "python train.py"},
    }
    canonical = json.dumps(
        cell, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()
    cell["cell_sha256"] = hashlib.sha256(canonical).hexdigest()
    manifest_payload = json.dumps({
        "campaign_id": "campaign-v1", "cells": [cell],
    })
    manifest = tmp_path / "campaign" / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text(manifest_payload)

    orch = ExperimentOrchestrator(project_root=tmp_path, automil_dir=adir)
    spec = _valid_preserving_spec(orch, "node_0001")
    spec["metadata"] = {
        "cell_id": "budget-1",
        "campaign": {
            "campaign_id": "campaign-v1",
            "manifest": "campaign/manifest.json",
            "manifest_sha256": hashlib.sha256(manifest_payload.encode()).hexdigest(),
            "cell_id": "cell-1",
            "cell_sha256": "0" * 64,
            "budget_cell_id": "budget-1",
            "stage": "discovery",
        },
    }

    orch.runner = MagicMock()
    orch._mark_crashed = MagicMock()
    orch._launch(spec, gpu_id=0)

    orch.runner.create_worktree.assert_not_called()
    assert "cell hash" in orch._mark_crashed.call_args.args[2]
