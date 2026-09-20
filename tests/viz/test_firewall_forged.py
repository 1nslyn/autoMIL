"""Forged violations: held-out data planted everywhere must never reach a payload.

Each test plants the thing the firewall forbids and then builds every record
payload, checking the payloads and the files that were opened.
"""
from __future__ import annotations

import builtins
import io
import json
import os
from pathlib import Path

import pytest

from automil.firewall import REDACTION, is_held_out_metric_key
from automil.session_record import store_session_record
from automil.viz.clock import host_clock
from automil.viz.record import RunSource
from automil.viz.record_graph import walk_keys

from tests.viz.conftest import SID, write_mini_session, write_project

PLANT = {"test_auc": 0.9137, "held_out_bacc": 0.8813}
PLANTED_VALUES = ("0.9137", "0.8813", "0.93")


def _everything(source: RunSource) -> dict[str, object]:
    payloads: dict[str, object] = {
        "index": source.index_entry(),
        "graph": source.build_graph(),
        "sessions": source.build_sessions(),
        "links": source.build_links(),
        "timeline": source.build_timeline(),
        "notes": source.build_notes(),
    }
    for node_id in source.build_graph()["nodes"]:
        payloads[f"node:{node_id}"] = source.build_node(node_id)
    for session in payloads["sessions"]["sessions"]:  # type: ignore[index]
        sid = session["session_id"]
        for chunk in range(session["n_chunks"]):
            payloads[f"chunk:{sid}:{chunk}"] = source.build_chunk(sid, chunk)
    return payloads


@pytest.fixture
def planted(tmp_path: Path) -> RunSource:
    automil = write_project(tmp_path / "proj", plant=PLANT)
    transcript = write_mini_session(tmp_path / "claude" / "projects" / "-data-project")
    store_session_record(automil, SID, transcript)
    return RunSource(automil, host_clock(tz_name="UTC"), config_dir=tmp_path / "nohome")


def test_planted_held_out_keys_reach_no_payload(planted: RunSource):
    payloads = _everything(planted)
    for name, payload in payloads.items():
        for path, key in walk_keys(payload):
            assert not is_held_out_metric_key(key), f"{name}: held-out key at {path}"
        text = json.dumps(payload)
        for value in PLANTED_VALUES:
            assert value not in text, f"{name} carries the planted value {value}"
    # the terminal run log had a planted line: it is redacted and counted
    log = payloads["node:node_0002"]["run_log"]  # type: ignore[index]
    assert log["redacted_lines"] == 1 and REDACTION in log["tail"]


def test_nothing_under_certify_is_opened_and_the_archive_is_never_listed(planted: RunSource, monkeypatch):
    opened: list[str] = []
    listed: list[str] = []
    real_open = builtins.open
    real_scandir = os.scandir
    real_listdir = os.listdir

    def spy_open(file, *args, **kwargs):
        if isinstance(file, (str, Path)):
            opened.append(str(file))
        return real_open(file, *args, **kwargs)

    def spy_scandir(path=".", *args, **kwargs):
        listed.append(str(path))
        return real_scandir(path, *args, **kwargs)

    def spy_listdir(path=".", *args, **kwargs):
        listed.append(str(path))
        return real_listdir(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(io, "open", spy_open)
    monkeypatch.setattr(os, "scandir", spy_scandir)
    monkeypatch.setattr(os, "listdir", spy_listdir)
    _everything(planted)
    planted.build_overlay_file("node_0003", "automil/variants/_policies/dropout.py")

    archive = planted.paths.archive_dir
    assert opened, "the spies saw no file reads"
    for path in opened:
        assert "certify" not in Path(path).parts, f"opened a sealed file: {path}"
    for path in listed:
        assert not str(path).startswith(str(archive)), f"listed the archive: {path}"


def test_a_running_node_log_is_refused_even_when_present(tmp_path: Path):
    automil = write_project(tmp_path / "proj", plant=PLANT, running=("node_0002",))
    source = RunSource(automil, host_clock(tz_name="UTC"), config_dir=tmp_path / "nohome")
    detail = source.build_node("node_0002")
    assert detail["run_log"] == {"available": False, "reason": "not terminal", "tail": []}
    for value in PLANTED_VALUES:
        assert value not in json.dumps(detail)


def test_secrets_in_tool_results_are_masked_everywhere(planted: RunSource):
    chunk = planted.build_chunk(SID, 0)
    full = planted.build_full_result(SID, "t7")
    assert "hf_abcdefghijklmnopqrstuvwxyz1234" not in json.dumps(chunk)
    assert "hf_abcdefghijklmnopqrstuvwxyz1234" not in full["text"]
    assert "HF_TOKEN=[REDACTED]" in full["text"]
