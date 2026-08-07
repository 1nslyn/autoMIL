"""CLI adapter and strict identity tests for the activity journal."""
from __future__ import annotations

import json

import pytest
import yaml
from click.testing import CliRunner

from automil.activity_hooks import (
    claude_activity_environment,
    claude_activity_hooks,
)
from automil.cells.identity import CellIdentityError, resolve_cell_identity
from automil.cells.state import make_cell_id
from automil.cli import main


def _write_config(tmp_path, config: dict) -> None:
    automil_dir = tmp_path / "automil"
    automil_dir.mkdir()
    (automil_dir / "config.yaml").write_text(yaml.safe_dump(config))


def _config(**overrides) -> dict:
    config = {
        "project": {"name": "tcga_luad"},
        "encoders": {"primary": "UNI_v2"},
        "task": {"name": "egfr"},
        "run": {"mil_model": " CLAM_SB "},
    }
    config.update(overrides)
    return config


def test_canonical_claude_observer_uses_native_metric_and_session_hooks():
    hooks = claude_activity_hooks("custom ingest")

    assert set(hooks) == {"SessionStart", "SessionEnd"}
    for event, entries in hooks.items():
        assert len(entries) == 1
        assert entries[0]["hooks"] == [
            {"type": "command", "command": "custom ingest"}
        ]
        assert "async" not in str(entries[0])
    assert hooks["SessionStart"][0]["matcher"] == "startup"
    assert "matcher" not in hooks["SessionEnd"][0]
    assert claude_activity_environment() == {
        "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
        "OTEL_METRICS_EXPORTER": "prometheus",
        "OTEL_METRICS_INCLUDE_SESSION_ID": "true",
    }


def test_resolve_cell_identity_uses_only_current_schema():
    identity = resolve_cell_identity(_config())

    assert identity.dataset == "tcga_luad"
    assert identity.encoder == "UNI_v2"
    assert identity.mil_model == "clam sb"
    assert identity.task == "egfr"
    assert identity.cell_id == make_cell_id("tcga_luad", "UNI_v2", "clam sb", "egfr")


def test_resolve_cell_identity_omits_only_redundant_task():
    identity = resolve_cell_identity(_config(task={"name": "tcga_luad"}), "ABMIL")

    assert identity.task is None
    assert identity.mil_model == "abmil"
    assert identity.cell_id == make_cell_id("tcga_luad", "UNI_v2", "abmil")


@pytest.mark.parametrize(
    "config",
    [
        _config(project={}, dataset={"name": "legacy"}),
        _config(encoders={}, encoder={"name": "legacy"}),
        _config(task={}),
        _config(run={}, mil_model="legacy"),
    ],
)
def test_resolve_cell_identity_rejects_missing_current_schema(config):
    with pytest.raises(CellIdentityError):
        resolve_cell_identity(config)


def test_activity_ingest_real_cli_accepts_claude_payloads(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "automil.activity_metrics.refresh_activity_metrics",
        lambda *_: ("session-123",),
    )
    config = _config(task={"name": "tcga_luad"})
    _write_config(tmp_path, config)
    monkeypatch.chdir(tmp_path)
    payloads = [
        {
            "hook_event_name": "SessionStart",
            "session_id": "session-123",
            "cwd": str(tmp_path),
            "source": "startup",
        },
        {
            "hook_event_name": "SessionEnd",
            "session_id": "session-123",
            "cwd": str(tmp_path),
            "reason": "clear",
        },
    ]

    runner = CliRunner()
    for payload in payloads:
        result = runner.invoke(main, ["activity", "ingest"], input=json.dumps(payload))
        assert result.exit_code == 0, result.output

    records = [
        json.loads(line)
        for line in (tmp_path / "automil" / ".activity.jsonl").read_text().splitlines()
    ]
    assert [record["event"] for record in records] == [
        "session_open",
        "session_end",
    ]
    assert {record["cell_id"] for record in records} == {
        resolve_cell_identity(config).cell_id
    }
    assert {record["session_id"] for record in records} == {"session-123"}


@pytest.mark.parametrize(
    "raw",
    ["", "[]", "null", "{broken", json.dumps({"session_id": "session-123"})],
)
def test_activity_ingest_rejects_invalid_payload(tmp_path, monkeypatch, raw):
    _write_config(tmp_path, _config())
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(main, ["activity", "ingest"], input=raw)

    assert result.exit_code != 0
    assert not (tmp_path / "automil" / ".activity.jsonl").exists()


def test_activity_ingest_rejects_tool_payload(tmp_path, monkeypatch):
    _write_config(tmp_path, _config())
    monkeypatch.chdir(tmp_path)
    payload = {
        "hook_event_name": "PreToolUse",
        "session_id": "session-123",
        "tool_name": "Bash",
        "tool_use_id": "tool-123",
        "tool_input": {"command": "true"},
    }

    result = CliRunner().invoke(main, ["activity", "ingest"], input=json.dumps(payload))

    assert result.exit_code != 0
    assert "unsupported hook event" in result.output


def test_session_end_requires_its_final_native_metric(tmp_path, monkeypatch):
    config = _config()
    _write_config(tmp_path, config)
    monkeypatch.chdir(tmp_path)
    from automil.cells.activity import record_hook_event

    record_hook_event(
        tmp_path / "automil",
        resolve_cell_identity(config).cell_id,
        {
            "hook_event_name": "SessionStart",
            "session_id": "session-123",
            "source": "startup",
        },
    )
    monkeypatch.setattr(
        "automil.activity_metrics.refresh_activity_metrics", lambda *_: None
    )
    payload = {
        "hook_event_name": "SessionEnd",
        "session_id": "session-123",
    }

    result = CliRunner().invoke(
        main, ["activity", "ingest"], input=json.dumps(payload)
    )

    assert result.exit_code != 0
    assert "final Claude active-time metric" in result.output
    records = [
        json.loads(line)
        for line in (tmp_path / "automil" / ".activity.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [record["event"] for record in records] == ["session_open"]
