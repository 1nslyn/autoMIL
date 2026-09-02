"""Behavioral tests for benchmarks/scripts/campaign_scan.py (stdlib CLI)."""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "campaign_scan.py"


def _load():
    spec = importlib.util.spec_from_file_location("campaign_scan", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def scan_mod():
    return _load()


def _cell(runtime: Path, name: str, **state) -> Path:
    root = runtime / name
    (root / "automil").mkdir(parents=True)
    payload = {"phase": "discovery", "baseline": {"archive": "baseline/archive"}, **state}
    (root / "campaign_state.json").write_text(json.dumps(payload))
    return root


def _runtime(tmp_path: Path, names) -> tuple[Path, Path]:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    roster = tmp_path / "active_roster.json"
    roster.write_text(json.dumps({"cohorts": ["tcga_luad"], "cells": len(names)}))
    for name in names:
        _cell(runtime, name)
    return runtime, roster


def test_classes_are_mutually_exclusive_and_ordered(scan_mod, tmp_path, monkeypatch):
    names = [f"tcga_luad__t{i}__enc__arm__s42__v" for i in range(6)]
    runtime, roster = _runtime(tmp_path, names)
    # done
    (runtime / names[0] / "campaign_state.json").write_text(json.dumps({"phase": "certified", "baseline": {}}))
    # claimed by a live job
    (runtime / names[1] / ".discovery_claim").write_text("111\n")
    # finishable: ended session + 30 charged
    (runtime / names[2] / "campaign_state.json").write_text(json.dumps(
        {"phase": "discovery", "baseline": {}, "discovery": {"attempts_charged": 30}}))
    (runtime / names[2] / "automil" / ".activity.jsonl").write_text(
        '{"event":"session_open"}\n{"event":"session_end"}\n')
    # stranded: evidence, no end
    (runtime / names[3] / "agent_session.json").write_text('{"status":"open"}')
    # blocked
    (runtime / names[4] / "campaign_state.json").write_text(json.dumps(
        {"phase": "discovery", "baseline": {}, "baseline_reproduction": {"mode": "gate", "verdict": "fail"}}))
    monkeypatch.setattr(scan_mod, "live_job_ids", lambda: {"111"})
    result = scan_mod.scan(runtime, roster, "999")
    assert result["done"] == [names[0]]
    assert result["claimed"] == [names[1]]
    assert result["finishable"] == [names[2]]
    assert result["stranded"] == [names[3]]
    assert result["blocked"] == [names[4]]
    assert result["pending"] == [names[5]]
    assert result["squeue_ok"] is True


def test_stale_claim_from_dead_job_is_pending_and_never_unlinked(scan_mod, tmp_path, monkeypatch):
    names = ["tcga_luad__a__e__m__s42__v"]
    runtime, roster = _runtime(tmp_path, names)
    claim = runtime / names[0] / ".discovery_claim"
    claim.write_text("424242\n")
    monkeypatch.setattr(scan_mod, "live_job_ids", lambda: {"1"})
    result = scan_mod.scan(runtime, roster, "999")
    assert result["pending"] == names
    assert claim.is_file(), "scan must never reap a claim; take_claim owns replacement"


def test_squeue_failure_treats_every_claim_as_live(scan_mod, tmp_path, monkeypatch):
    names = ["tcga_luad__a__e__m__s42__v"]
    runtime, roster = _runtime(tmp_path, names)
    (runtime / names[0] / ".discovery_claim").write_text("424242\n")
    monkeypatch.setattr(scan_mod, "live_job_ids", lambda: None)
    result = scan_mod.scan(runtime, roster, "999")
    assert result["claimed"] == names
    assert result["squeue_ok"] is False


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file modes")
def test_unreadable_state_is_reported_as_claimed_not_a_crash(scan_mod, tmp_path, monkeypatch):
    names = ["tcga_luad__a__e__m__s42__v", "tcga_luad__b__e__m__s42__v"]
    runtime, roster = _runtime(tmp_path, names)
    locked = runtime / names[0] / "campaign_state.json"
    locked.chmod(0)
    try:
        monkeypatch.setattr(scan_mod, "live_job_ids", lambda: set())
        result = scan_mod.scan(runtime, roster, "999")
    finally:
        locked.chmod(stat.S_IRUSR | stat.S_IWUSR)
    assert result["claimed"] == [names[0]]
    assert "unreadable" in result["notes"][names[0]]
    assert result["pending"] == [names[1]]


def test_missing_baseline_fails_loudly(scan_mod, tmp_path, monkeypatch):
    names = ["tcga_luad__a__e__m__s42__v"]
    runtime, roster = _runtime(tmp_path, names)
    (runtime / names[0] / "campaign_state.json").write_text(json.dumps({"phase": "discovery", "baseline": None}))
    monkeypatch.setattr(scan_mod, "live_job_ids", lambda: set())
    with pytest.raises(SystemExit, match="no registered baseline"):
        scan_mod.scan(runtime, roster, "999")


def test_roster_census_mismatch_fails_loudly(scan_mod, tmp_path):
    runtime, roster = _runtime(tmp_path, ["tcga_luad__a__e__m__s42__v"])
    roster.write_text(json.dumps({"cohorts": ["tcga_luad"], "cells": 2}))
    with pytest.raises(SystemExit, match="roster declares 2"):
        scan_mod.scan(runtime, roster, "999")


def test_cli_class_filter_prints_one_cell_per_line(tmp_path, monkeypatch):
    names = ["tcga_luad__a__e__m__s42__v", "tcga_luad__b__e__m__s42__v"]
    runtime, roster = _runtime(tmp_path, names)
    env = {**os.environ, "PATH": str(tmp_path / "bin") + os.pathsep + os.environ["PATH"]}
    (tmp_path / "bin").mkdir()
    fake = tmp_path / "bin" / "squeue"
    fake.write_text("#!/bin/sh\necho 7\n")
    fake.chmod(0o755)
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--runtime", str(runtime), "--roster", str(roster),
         "--job-id", "5", "--class", "pending"],
        capture_output=True, text=True, env=env, check=True,
    ).stdout
    assert out.split() == names
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--runtime", str(runtime), "--roster", str(roster)],
        capture_output=True, text=True, env=env, check=True,
    ).stdout
    assert json.loads(out)["pending"] == names


def test_cli_missing_runtime_exits_2(tmp_path):
    rc = subprocess.run(
        [sys.executable, str(SCRIPT), "--runtime", str(tmp_path / "nope"), "--roster", str(tmp_path / "r.json")],
        capture_output=True, text=True,
    ).returncode
    assert rc == 2
