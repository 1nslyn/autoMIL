"""viz.record_routes + server: the record over HTTP, CORS, and the SPA fallback."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer

from automil.session_record import store_session_record
from automil.viz import server as srv
from automil.viz.clock import host_clock
from automil.viz.record import RunSource
from automil.viz.record_routes import origin_allowed

from tests.viz.conftest import SID, write_mini_session, write_project


def _run(coro):
    return asyncio.run(coro)


def _source(tmp_path: Path) -> RunSource:
    automil = write_project(tmp_path / "proj")
    transcript = write_mini_session(tmp_path / "home" / "projects" / "-data-project")
    store_session_record(automil, SID, transcript)
    return RunSource(automil, host_clock(tz_name="UTC"), run_id="demo", config_dir=tmp_path / "nohome")


async def _client(source: RunSource | None, **kwargs) -> TestClient:
    app = srv.create_app(run_source=source, **kwargs)
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


def test_every_record_url_matches_the_builders(tmp_path: Path):
    source = _source(tmp_path)

    async def scenario():
        client = await _client(source)
        try:
            index = await (await client.get("/record/index.json")).json()
            assert index["mode"] == "live" and index["runs"][0]["run_id"] == "demo"
            for name, builder in (
                ("graph", source.build_graph), ("timeline", source.build_timeline),
                ("sessions", source.build_sessions), ("agent_links", source.build_links),
                ("notes", source.build_notes),
            ):
                response = await client.get(f"/record/runs/demo/{name}.json")
                assert response.status == 200 and response.headers["Cache-Control"].startswith("no-store")
                assert await response.json() == json.loads(json.dumps(builder()))
            node = await (await client.get("/record/runs/demo/nodes/node_0003.json")).json()
            assert node["node_id"] == "node_0003"
            file = await client.get("/record/runs/demo/nodes/node_0003/files/automil/variants/_policies/dropout.py.json")
            assert (await file.json())["text"] == "DROPOUT = 0.5\n"
            chunk = await (await client.get(f"/record/runs/demo/sessions/{SID}/turns/0.json")).json()
            assert chunk["turns"][0]["kind"] == "human"
            full = await (await client.get(f"/record/runs/demo/sessions/{SID}/results/t7.json")).json()
            assert full["text"].endswith("TAIL-MARKER")
            agent = await (await client.get(f"/record/runs/demo/sessions/{SID}/agents/agent1.json")).json()
            assert agent["tool_use_id"] == "t3"
            for bad in (
                "/record/runs/other/graph.json", "/record/runs/demo/certified.json",
                "/record/runs/demo/nodes/node_1.json", "/record/runs/demo/nodes/node_9999.json",
                f"/record/runs/demo/sessions/{SID}/turns/7.json", "/record/runs/demo/sessions/zz/turns/0.json",
                "/record/runs/demo/nodes/node_0003/files/../../secret.json",
                "/record/runs/demo/nodes/node_0003/files/certify/certify.json.json",
            ):
                response = await client.get(bad)
                assert response.status == 404, bad
            # the old routes still exist
            assert (await client.get("/api/promotion-rate")).status == 200
            page = await client.get("/")
            assert page.status == 200 and "<html" in (await page.text()).lower()
            # unknown paths fall back to the page (hash routes)
            fallback = await client.get("/run/demo/tree")
            assert fallback.status == 200 and "<html" in (await fallback.text()).lower()
            assert (await client.get("/static/nope.js")).status == 404
        finally:
            await client.close()

    _run(scenario())


def test_cors_allows_listed_and_loopback_origins_only(tmp_path: Path):
    source = _source(tmp_path)

    async def scenario():
        client = await _client(source, cors_origins=("https://automil.org",))
        try:
            ok = await client.get("/record/index.json", headers={"Origin": "https://automil.org"})
            assert ok.headers["Access-Control-Allow-Origin"] == "https://automil.org"
            assert ok.headers["Vary"] == "Origin"
            local = await client.get("/record/index.json", headers={"Origin": "http://localhost:8000"})
            assert local.headers["Access-Control-Allow-Origin"] == "http://localhost:8000"
            denied = await client.get("/record/index.json", headers={"Origin": "https://evil.example"})
            assert "Access-Control-Allow-Origin" not in denied.headers and denied.status == 200
            preflight = await client.options(
                "/record/runs/demo/graph.json",
                headers={"Origin": "https://automil.org", "Access-Control-Request-Method": "GET",
                         "Access-Control-Request-Private-Network": "true"},
            )
            assert preflight.status == 204
            assert preflight.headers["Access-Control-Allow-Private-Network"] == "true"
            assert preflight.headers["Access-Control-Allow-Methods"] == "GET, OPTIONS"
            events_preflight = await client.options("/events", headers={"Origin": "https://automil.org"})
            assert events_preflight.status == 204 and events_preflight.headers["Access-Control-Allow-Origin"] == "https://automil.org"
            # the stream itself sends its headers before any frame, so the grant must be on them
            stream = await client.get("/events", headers={"Origin": "https://automil.org"})
            assert stream.status == 200 and stream.headers["Access-Control-Allow-Origin"] == "https://automil.org"
            assert stream.headers["Content-Type"].startswith("text/event-stream")
            stream.close()
            denied_stream = await client.get("/events", headers={"Origin": "https://evil.example"})
            assert "Access-Control-Allow-Origin" not in denied_stream.headers
            denied_stream.close()
            # the page itself is not a cross-origin resource
            page = await client.get("/", headers={"Origin": "https://automil.org"})
            assert "Access-Control-Allow-Origin" not in page.headers
        finally:
            await client.close()

    _run(scenario())
    assert origin_allowed("http://127.0.0.1:5173", ()) and not origin_allowed("http://evil.localhost.example", ())
    assert not origin_allowed(None, ("https://automil.org",))


def test_sse_first_frame_is_the_projected_graph(tmp_path: Path, monkeypatch):
    from tests.viz.conftest import graph_payload

    source = _source(tmp_path)
    monkeypatch.setattr(srv, "GRAPH_FILE", source.paths.graph)
    monkeypatch.setattr(srv, "GPU_STATE_FILE", source.paths.gpu_state)
    planted = graph_payload(plant={"test_auc": 0.9137})
    source.paths.graph.write_text(json.dumps(planted))
    watcher = srv.GraphWatcher()
    watcher.set_clock(host_clock(tz_name="UTC"))
    frame = json.loads(_run(watcher.get_initial()))
    assert frame["type"] == "graph_update" and "node_0001" in frame["added"]
    text = json.dumps(frame)
    assert "test_auc" not in text and "0.9137" not in text
    assert frame["full_graph"]["nodes"]["node_0002"]["created_at"].endswith("Z")
