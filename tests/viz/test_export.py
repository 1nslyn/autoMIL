"""viz.export: the record tree on disk, index merging, the single-file form."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from automil.session_record import store_session_record
from automil.viz.clock import host_clock
from automil.viz.export import (
    ExportError,
    check_target,
    export_run,
    export_site,
    merge_index,
    single_file,
)
from automil.viz.record import RunSource

from tests.viz.conftest import SID, write_mini_session, write_project


def _source(tmp_path: Path, name: str, run_id: str) -> RunSource:
    automil = write_project(tmp_path / name)
    transcript = write_mini_session(tmp_path / f"home-{name}" / "projects" / "-data-project")
    store_session_record(automil, SID, transcript)
    return RunSource(automil, host_clock(tz_name="UTC"), run_id=run_id, config_dir=tmp_path / "nohome")


def _fake_static(tmp_path: Path) -> Path:
    static = tmp_path / "static"
    (static / "css").mkdir(parents=True)
    (static / "js").mkdir()
    (static / "fonts").mkdir()
    (static / "fonts" / "f.woff2").write_bytes(b"\x00font")
    (static / "css" / "a.css").write_text("@font-face{src:url(../fonts/f.woff2)}\nbody{margin:0}\n")
    (static / "js" / "main.js").write_text("console.log('</script>');\n")
    (static / "index.html").write_text(
        '<!doctype html><html><head><link rel="stylesheet" href="./static/css/a.css"></head>'
        '<body><script src="./static/js/main.js"></script></body></html>'
    )
    return static


def test_export_run_writes_the_record_tree_and_is_byte_stable(tmp_path: Path):
    source = _source(tmp_path, "a", "run-a")
    out = tmp_path / "out"
    summary = export_run(source, out)
    run_dir = out / "record" / "runs" / "run-a"
    assert summary.nodes == 5 and summary.sessions == 1 and summary.files > 10
    for name in ("graph.json", "timeline.json", "sessions.json", "agent_links.json", "notes.json"):
        assert (run_dir / name).is_file()
    assert not (run_dir / "certified.json").exists()
    assert (run_dir / "nodes" / "node_0003.json").is_file()
    assert (run_dir / "nodes" / "node_0003" / "files" / "automil" / "variants" / "_policies" / "dropout.py.json").is_file()
    assert (run_dir / "sessions" / SID / "turns" / "0.json").is_file()
    assert (run_dir / "sessions" / SID / "results" / "t7.json").is_file()  # the only truncated result
    assert not (run_dir / "sessions" / SID / "results" / "t1.json").exists()
    assert (run_dir / "sessions" / SID / "agents" / "agent1.json").is_file()
    snapshot = {p: p.read_bytes() for p in run_dir.rglob("*.json")}
    export_run(source, out)
    assert {p: p.read_bytes() for p in run_dir.rglob("*.json")} == snapshot


def test_index_merges_runs_and_drops_stale_entries(tmp_path: Path):
    a = _source(tmp_path, "a", "run-a")
    b = _source(tmp_path, "b", "run-b")
    out = tmp_path / "out"
    export_run(a, out)
    merge_index(out, [a])
    export_run(b, out)
    index = merge_index(out, [b])
    assert [r["run_id"] for r in index["runs"]] == ["run-a", "run-b"] and index["mode"] == "static"
    # a run whose files were removed disappears from the index on the next merge
    import shutil

    shutil.rmtree(out / "record" / "runs" / "run-a")
    index = merge_index(out, [b])
    assert [r["run_id"] for r in index["runs"]] == ["run-b"]


def test_check_target_refuses_foreign_non_empty_directories(tmp_path: Path):
    out = tmp_path / "out"
    check_target(out, force=False)  # missing is fine
    out.mkdir()
    (out / "notes.txt").write_text("x")
    with pytest.raises(ExportError):
        check_target(out, force=False)
    check_target(out, force=True)
    (out / "record").mkdir()
    (out / "record" / "index.json").write_text("{}")
    check_target(out, force=False)  # an existing record is a valid target


def test_export_site_copies_the_frontend_and_single_file_inlines_everything(tmp_path: Path):
    static = _fake_static(tmp_path)
    source = _source(tmp_path, "a", "run-a")
    out = tmp_path / "site"
    export_site([source], out, static_dir=static)
    assert (out / "index.html").is_file() and (out / "static" / "js" / "main.js").is_file()
    assert not (out / "static" / "index.html").exists()
    assert json.loads((out / "record" / "index.json").read_text())["runs"][0]["run_id"] == "run-a"

    html_path = single_file(tmp_path / "one.html", out, static_dir=static)
    html = html_path.read_text()
    assert "<style>" in html and "body{margin:0}" in html
    assert "url(data:font/woff2;base64," in html
    assert "<\\/script>" in html and 'src="./static' not in html
    start = html.index('id="automil-record">') + len('id="automil-record">')
    blob = html[start: html.index("</script>", start)]
    record = json.loads(blob.replace("<\\/script", "</script"))
    assert "index.json" in record and "runs/run-a/graph.json" in record
    assert record["index.json"]["runs"][0]["run_id"] == "run-a"
