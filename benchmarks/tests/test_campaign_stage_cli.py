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


def test_run_baseline_cli_forwards_physical_gpu(tmp_path, monkeypatch, capsys):
    module = _load_cli()
    root, state = _state(tmp_path)
    observed = {}
    monkeypatch.setattr(module, "_cell_root", lambda *_: root)

    def fake_run(cell_root, *, repo_root, gpu_id):
        observed.update({
            "cell_root": cell_root,
            "repo_root": repo_root,
            "gpu_id": gpu_id,
        })
        return state

    monkeypatch.setattr(module, "run_native_baseline", fake_run)
    module.main(["run-baseline", "--cell-root", "ignored", "--gpu", "3"])

    assert observed["cell_root"] == root
    assert observed["gpu_id"] == 3
    assert json.loads(capsys.readouterr().out)["phase"] == "discovery"


@pytest.mark.parametrize(
    "action,function_name,payload",
    [
        (
            "open-agent-session",
            "open_agent_session",
            {
                "session_id": "session-v1",
                "started_at": "2026-08-04T00:00:00+00:00",
            },
        ),
        (
            "finalize-agent-session",
            "finalize_agent_session",
            {
                "session_id": "session-v1",
                "ended_at": "2026-08-04T01:00:00+00:00",
                "termination_reason": "budget-complete",
                "usage": {
                    "status": "unavailable",
                    "input_tokens": None,
                    "output_tokens": None,
                    "cached_input_tokens": None,
                    "cost_usd": None,
                    "basis": "fixture runtime does not expose usage",
                },
            },
        ),
    ],
)
def test_agent_session_cli_actions_load_relative_payload_and_emit_status(
    tmp_path, monkeypatch, capsys, action, function_name, payload,
):
    module = _load_cli()
    root, state = _state(tmp_path)
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    session_path = fake_repo / "session.json"
    session_path.write_text(json.dumps(payload))
    module.__file__ = str(
        fake_repo / "benchmarks/scripts/campaign_stage.py"
    )
    observed = {}
    monkeypatch.setattr(module, "_cell_root", lambda *_: root)
    monkeypatch.setattr(
        module,
        function_name,
        lambda cell_root, loaded: observed.update({
            "cell_root": cell_root,
            "payload": loaded,
        }),
    )

    module.main([
        action,
        "--cell-root", "ignored",
        "--agent-session", "session.json",
    ])

    assert observed == {"cell_root": root, "payload": payload}
    assert json.loads(capsys.readouterr().out)["phase"] == state["phase"]


def test_agent_session_cli_requires_payload_path(tmp_path, monkeypatch, capsys):
    module = _load_cli()
    root, _ = _state(tmp_path)
    monkeypatch.setattr(module, "_cell_root", lambda *_: root)

    with pytest.raises(SystemExit, match="2"):
        module.main(["open-agent-session", "--cell-root", "ignored"])

    assert "requires --agent-session" in capsys.readouterr().err
