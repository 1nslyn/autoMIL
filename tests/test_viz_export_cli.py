"""``automil viz export`` writes the site for the current project."""
from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from automil.cli import main
from automil.session_record import store_session_record

from tests.viz.conftest import SID, write_mini_session, write_project


def test_viz_export_writes_a_site_and_a_single_file(tmp_path: Path, monkeypatch):
    automil = write_project(tmp_path / "proj")
    store_session_record(automil, SID, write_mini_session(tmp_path / "home" / "projects" / "-x"))
    monkeypatch.chdir(tmp_path / "proj")
    out = tmp_path / "site"
    result = CliRunner().invoke(
        main, ["viz", "export", "--out", str(out), "--run-id", "kras-abmil", "--tz", "America/Vancouver",
               "--single-file", str(tmp_path / "one.html")],
    )
    assert result.exit_code == 0, result.output
    assert "exported kras-abmil: 5 nodes, 1 session(s)" in result.output
    index = json.loads((out / "record" / "index.json").read_text())
    assert index["mode"] == "static" and index["runs"][0]["run_id"] == "kras-abmil"
    assert index["runs"][0]["clock"]["tz_name"] == "America/Vancouver"
    assert (out / "index.html").is_file() and (out / "static").is_dir()
    assert (tmp_path / "one.html").is_file()

    bad_zone = CliRunner().invoke(main, ["viz", "export", "--out", str(tmp_path / "x"), "--tz", "Mars/Olympus"])
    assert bad_zone.exit_code != 0 and "unknown time zone" in bad_zone.output

    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("x")
    refused = CliRunner().invoke(main, ["viz", "export", "--out", str(foreign)])
    assert refused.exit_code != 0 and "--force" in refused.output
    forced = CliRunner().invoke(main, ["viz", "export", "--out", str(foreign), "--force", "--no-frontend"])
    assert forced.exit_code == 0, forced.output
    assert (foreign / "record" / "index.json").is_file() and not (foreign / "index.html").exists()
