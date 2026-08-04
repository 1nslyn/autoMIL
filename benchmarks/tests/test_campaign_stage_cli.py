"""Operator CLI must stay validation-only unless certify is explicit."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from autobench.campaign import CAMPAIGN_ID, PROTOCOL, content_sha256
from autobench.campaign_stages import initialize_stage_state


def _load_cli():
    path = Path(__file__).resolve().parents[1] / "scripts" / "campaign_stage.py"
    spec = importlib.util.spec_from_file_location("campaign_stage_cli_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _state(tmp_path: Path):
    root = tmp_path / "cell"
    cell = {"cell_id": "cell", "cell_sha256": "a" * 64}
    return root, initialize_stage_state(
        root, cell=cell, manifest_sha256="b" * 64,
        base_commit="d" * 40,
    )


def test_public_status_contains_no_held_out_surface(tmp_path):
    module = _load_cli()
    _, state = _state(tmp_path)
    rendered = module.public_status(state)
    assert rendered["phase"] == "discovery"
    assert "held" not in json.dumps(rendered).lower()
    assert "test" not in json.dumps(rendered).lower()


def test_advance_never_auto_certifies(tmp_path, monkeypatch):
    module = _load_cli()
    root, state = _state(tmp_path)
    state["phase"] = "winner-frozen"
    state["winner"] = {
        "kind": "baseline", "candidate_id": "baseline",
        "validation_mean": 0.6, "lift_over_baseline": 0.0,
        "selection_sha256": "c" * 64,
    }
    state["state_sha256"] = content_sha256({
        key: value for key, value in state.items() if key != "state_sha256"
    })
    (root / "campaign_state.json").write_text(json.dumps(state))
    monkeypatch.setattr(module, "certify_winner", lambda *_: pytest.fail(
        "advance must never call certify_winner",
    ))

    advanced = module.advance(root, tmp_path)
    assert advanced["phase"] == "winner-frozen"
    assert advanced["certification"] is None


def test_status_schema_tracks_the_frozen_protocol(tmp_path):
    module = _load_cli()
    _, state = _state(tmp_path)
    rendered = module.public_status(state)
    assert rendered["campaign_id"] == CAMPAIGN_ID
    assert rendered["discovery"]["attempt_budget"] == PROTOCOL["discovery_attempts"]
    assert rendered["promotion"]["jobs"] == 0


def test_baseline_command_is_an_explicit_non_agentic_fivefold_run(tmp_path):
    module = _load_cli()
    root, _ = _state(tmp_path)
    adir = root / "automil"
    adir.mkdir()
    (adir / "campaign_cell.json").write_text(json.dumps({
        "commands": {"baseline": "python train.py --folds 0,1,2,3,4"},
    }))
    assert module.baseline_command(root).endswith("--folds 0,1,2,3,4")
