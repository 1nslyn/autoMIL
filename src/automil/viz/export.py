"""Write a run's record to disk, and the whole site around it.

The output is the same tree the live server serves, so the frontend does not
know which one it is reading. Per-run files carry no generation time and are
written with sorted keys, so exporting an unchanged run again changes nothing.
"""
from __future__ import annotations

import base64
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from automil.viz.record import RunSource, build_index

STATIC_DIR = Path(__file__).parent / "static"
RECORD_DIRNAME = "record"
_SAFE_PATH = re.compile(r"^[A-Za-z0-9._/-]+$")


class ExportError(RuntimeError):
    """The export target cannot be used as asked."""


@dataclass
class ExportSummary:
    run_id: str
    out: Path
    files: int = 0
    sessions: int = 0
    nodes: int = 0
    warnings: list[str] = field(default_factory=list)


def dump_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _safe_relative(rel_path: str) -> str | None:
    parts = Path(rel_path).parts
    if not rel_path or rel_path.startswith("/") or ".." in parts or not _SAFE_PATH.match(rel_path):
        return None
    return rel_path


def export_run(source: RunSource, out: Path) -> ExportSummary:
    """Write ``out/record/runs/<run_id>/...`` for one run."""
    run_dir = Path(out) / RECORD_DIRNAME / "runs" / source.run_id
    summary = ExportSummary(run_id=source.run_id, out=run_dir)

    def put(rel: str, payload: Any) -> None:
        _write(run_dir / rel, dump_json(payload))
        summary.files += 1

    graph = source.build_graph()
    put("graph.json", graph)
    timeline = source.build_timeline()
    summary.warnings.extend(timeline.get("warnings", []))
    put("timeline.json", timeline)
    sessions = source.build_sessions()
    put("sessions.json", sessions)
    put("agent_links.json", source.build_links())
    put("notes.json", source.build_notes())
    certified = source.build_certified()
    if certified is not None:
        put("certified.json", certified)

    for node_id in graph["nodes"]:
        detail = source.build_node(node_id)
        if detail is None:
            continue
        put(f"nodes/{node_id}.json", detail)
        summary.nodes += 1
        for entry in detail["overlay"]["files"]:
            rel = _safe_relative(entry["path"])
            if rel is None:
                continue
            file = source.build_overlay_file(node_id, rel)
            if file is not None:
                put(f"nodes/{node_id}/files/{rel}.json", file)

    for session in sessions["sessions"]:
        sid = session["session_id"]
        summary.sessions += 1
        for chunk in range(session["n_chunks"]):
            payload = source.build_chunk(sid, chunk)
            if payload is None:
                continue
            put(f"sessions/{sid}/turns/{chunk}.json", payload)
            for turn in payload["turns"]:
                for call in turn.get("tool_calls", ()):
                    result = call.get("result") or {}
                    if result.get("truncated"):
                        full = source.build_full_result(sid, call["tool_use_id"])
                        if full is not None:
                            put(f"sessions/{sid}/results/{call['tool_use_id']}.json", full)
        for agent in session.get("subagents", ()):
            payload = source.build_agent(sid, agent["agent_id"])
            if payload is not None:
                put(f"sessions/{sid}/agents/{agent['agent_id']}.json", payload)
    return summary


def merge_index(out: Path, sources: Iterable[RunSource], *, mode: str = "static") -> dict[str, Any]:
    """Rewrite ``record/index.json`` keeping runs exported earlier into ``out``."""
    index_path = Path(out) / RECORD_DIRNAME / "index.json"
    previous: dict[str, Any] = {}
    if index_path.is_file():
        try:
            previous = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
    fresh = build_index(list(sources), mode)
    fresh_ids = {run["run_id"] for run in fresh["runs"]}
    kept = [run for run in previous.get("runs", []) if isinstance(run, dict) and run.get("run_id") not in fresh_ids]
    runs_dir = Path(out) / RECORD_DIRNAME / "runs"
    kept = [run for run in kept if (runs_dir / str(run.get("run_id")) / "graph.json").is_file()]
    fresh["runs"] = kept + fresh["runs"]
    _write(index_path, json.dumps(fresh, indent=2, sort_keys=True) + "\n")
    return fresh


def copy_frontend(out: Path, *, static_dir: Path = STATIC_DIR) -> None:
    """Copy ``index.html`` and ``static/`` next to the record."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(static_dir / "index.html", out / "index.html")
    target = out / "static"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(static_dir, target, ignore=shutil.ignore_patterns("index.html", "__pycache__"))


def check_target(out: Path, *, force: bool) -> None:
    """Refuse a non-empty directory that is not already a record, unless forced."""
    out = Path(out)
    if not out.exists():
        return
    if any(out.iterdir()) and not (out / RECORD_DIRNAME / "index.json").is_file() and not force:
        raise ExportError(f"{out} is not empty and holds no record; pass --force to write into it")


def export_site(
    sources: list[RunSource],
    out: Path,
    *,
    force: bool = False,
    static_dir: Path = STATIC_DIR,
    with_frontend: bool = True,
) -> list[ExportSummary]:
    check_target(out, force=force)
    summaries = [export_run(source, out) for source in sources]
    merge_index(out, sources)
    if with_frontend:
        copy_frontend(out, static_dir=static_dir)
    return summaries


_FONT_TYPES = {".woff2": "font/woff2", ".woff": "font/woff"}


def _inline_fonts(css: str, css_dir: Path) -> str:
    def replace(match: re.Match[str]) -> str:
        rel = match.group(1)
        target = (css_dir / rel).resolve()
        if not target.is_file() or target.suffix not in _FONT_TYPES:
            return match.group(0)
        data = base64.b64encode(target.read_bytes()).decode("ascii")
        return f"url(data:{_FONT_TYPES[target.suffix]};base64,{data})"

    return re.sub(r"url\((?:'|\")?([^'\")]+\.woff2?)(?:'|\")?\)", replace, css)


def single_file(out_html: Path, record_root: Path, *, static_dir: Path = STATIC_DIR) -> Path:
    """One HTML file: stylesheets, scripts, fonts and the whole record inlined.

    The frontend reads the inlined record map before fetching anything, so
    the file works from disk without a server.
    """
    html = (static_dir / "index.html").read_text(encoding="utf-8")

    def inline_link(match: re.Match[str]) -> str:
        rel = match.group(1)
        css_path = static_dir / rel.removeprefix("./static/")
        css = _inline_fonts(css_path.read_text(encoding="utf-8"), css_path.parent)
        return f"<style>\n{css}\n</style>"

    def inline_script(match: re.Match[str]) -> str:
        rel = match.group(1)
        js_path = static_dir / rel.removeprefix("./static/")
        return "<script>\n" + js_path.read_text(encoding="utf-8").replace("</script", "<\\/script") + "\n</script>"

    html = re.sub(r'<link rel="stylesheet" href="(\./static/[^"]+)">', inline_link, html)
    html = re.sub(r'<script src="(\./static/[^"]+)"></script>', inline_script, html)
    record: dict[str, Any] = {}
    root = Path(record_root) / RECORD_DIRNAME
    for path in sorted(root.rglob("*.json")):
        rel = path.relative_to(root).as_posix()
        record[rel] = json.loads(path.read_text(encoding="utf-8"))
    blob = json.dumps(record, separators=(",", ":"), ensure_ascii=False).replace("</script", "<\\/script")
    tag = f'<script type="application/json" id="automil-record">{blob}</script>'
    html = html.replace("</head>", tag + "\n</head>", 1)
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    return out_html
