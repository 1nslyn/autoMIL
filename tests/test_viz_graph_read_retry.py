"""viz/server.py must not read graph.json with zero defence against a
transient parse failure (L-8b).

graph.py's ExperimentGraph.save() writes graph.json via tempfile + os.rename,
which is atomic at the filesystem level, so a true torn read should not be
possible on a local POSIX filesystem. This is defence-in-depth for anything
that could still surface as a transient JSONDecodeError.

Deliberately NOT fcntl-locked (see graph.locked_update's <path>.lock
sidecar): the SSE broadcast loop must never be able to block on a lock the
orchestrator's writers also want -- a hung or slow dashboard read must never
be able to wedge the daemon. So the fix is a bounded, non-blocking
(asyncio.sleep) retry on parse failure, entirely local to this process.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
from pathlib import Path

from aiohttp.test_utils import loop_context


def _run(coro):
    with loop_context() as loop:
        return loop.run_until_complete(coro)


_EMPTY_GRAPH = {"nodes": {}, "meta": {}, "technique_stats": {}}


class TestReadGraphJsonRetry:
    def test_returns_parsed_data_on_a_clean_read(self, tmp_path):
        import automil.viz.server as srv

        graph_file = tmp_path / "graph.json"
        payload = {"nodes": {"node_0001": {"id": "node_0001"}}, "meta": {}, "technique_stats": {}}
        graph_file.write_text(json.dumps(payload))

        result = _run(srv._read_graph_json(graph_file))

        assert result == payload

    def test_returns_none_when_the_file_is_missing(self, tmp_path):
        import automil.viz.server as srv

        result = _run(srv._read_graph_json(tmp_path / "does_not_exist.json"))

        assert result is None

    def test_retries_a_transient_parse_failure_then_succeeds(self, tmp_path, monkeypatch):
        """The core reproduction: the first read(s) hit a parse failure (simulating
        a read racing a write), and the retry recovers instead of giving up
        immediately."""
        import automil.viz.server as srv

        graph_file = tmp_path / "graph.json"
        graph_file.write_text("irrelevant — the loader below is patched")
        good_data = {"nodes": {}, "meta": {}, "technique_stats": {}}
        calls = {"n": 0}

        def flaky_loader(path):
            calls["n"] += 1
            if calls["n"] < 3:
                raise json.JSONDecodeError("boom", "doc", 0)
            return good_data

        monkeypatch.setattr(srv, "_load_graph_json_text", flaky_loader)

        result = _run(srv._read_graph_json(graph_file, retries=5, retry_delay_s=0.001))

        assert result == good_data
        assert calls["n"] == 3, "must have retried exactly twice before succeeding"

    def test_gives_up_after_the_retry_budget_and_logs_a_warning(self, tmp_path, monkeypatch, caplog):
        import automil.viz.server as srv

        graph_file = tmp_path / "graph.json"
        graph_file.write_text("irrelevant — the loader below is patched")

        def always_bad(path):
            raise json.JSONDecodeError("boom", "doc", 0)

        monkeypatch.setattr(srv, "_load_graph_json_text", always_bad)

        with caplog.at_level(logging.WARNING):
            result = _run(srv._read_graph_json(graph_file, retries=3, retry_delay_s=0.001))

        assert result is None
        assert any("graph.json" in r.getMessage() for r in caplog.records), (
            "a persistent parse failure must be logged, not silently swallowed"
        )

    def test_does_not_retry_on_a_missing_file(self, tmp_path, monkeypatch):
        """FileNotFoundError is the ordinary case (no graph.json yet) — must
        return None immediately, not burn through the retry budget."""
        import automil.viz.server as srv

        calls = {"n": 0}

        def never_called(path):
            calls["n"] += 1
            raise FileNotFoundError()

        monkeypatch.setattr(srv, "_load_graph_json_text", never_called)

        result = _run(srv._read_graph_json(tmp_path / "missing.json", retries=5, retry_delay_s=0.001))

        assert result is None
        assert calls["n"] == 1


class TestNoLockingTradeOff:
    """Guard the deliberate choice: retry, never fcntl-lock, in the read path."""

    def test_read_graph_json_source_has_no_locking_primitives(self):
        """Checks actual usage (``fcntl.``/``flock(``), not the word "fcntl" —
        the docstring itself names ``fcntl`` in prose to explain the trade-off,
        so a bare substring check would flag its own documentation."""
        import automil.viz.server as srv

        src = inspect.getsource(srv._read_graph_json) + inspect.getsource(srv._load_graph_json_text)
        assert "fcntl." not in src
        assert "flock(" not in src

    def test_viz_server_module_does_not_import_fcntl(self):
        import automil.viz.server as srv

        source = Path(srv.__file__).read_text()
        assert "import fcntl" not in source, (
            "viz/server.py must never take graph.json's fcntl lock — a hung or "
            "slow SSE read must not be able to block the orchestrator's writers"
        )


class TestNotifyAndGetInitialUseTheRetryingReader:
    def test_get_initial_returns_the_graph_when_readable(self, tmp_path, monkeypatch):
        import automil.viz.server as srv

        graph_file = tmp_path / "graph.json"
        graph_file.write_text(json.dumps(
            {"nodes": {"node_0001": {"id": "node_0001"}}, "meta": {}, "technique_stats": {}}
        ))
        monkeypatch.setattr(srv, "GRAPH_FILE", graph_file)
        monkeypatch.setattr(srv, "GPU_STATE_FILE", tmp_path / "gpu_state.json")

        watcher = srv.GraphWatcher()
        payload = json.loads(_run(watcher.get_initial()))

        assert payload["added"] == ["node_0001"]

    def test_get_initial_falls_back_to_empty_skeleton_when_graph_missing(self, tmp_path, monkeypatch):
        import automil.viz.server as srv

        monkeypatch.setattr(srv, "GRAPH_FILE", tmp_path / "does_not_exist.json")
        monkeypatch.setattr(srv, "GPU_STATE_FILE", tmp_path / "gpu_state.json")

        watcher = srv.GraphWatcher()
        payload = json.loads(_run(watcher.get_initial()))

        assert payload["full_graph"] == _EMPTY_GRAPH

    def test_get_initial_recovers_from_a_transient_parse_failure(self, tmp_path, monkeypatch):
        import automil.viz.server as srv

        graph_file = tmp_path / "graph.json"
        graph_file.write_text("irrelevant — the loader below is patched")
        monkeypatch.setattr(srv, "GPU_STATE_FILE", tmp_path / "gpu_state.json")
        monkeypatch.setattr(srv, "GRAPH_FILE", graph_file)

        calls = {"n": 0}

        def flaky_loader(path):
            calls["n"] += 1
            if calls["n"] < 2:
                raise json.JSONDecodeError("boom", "doc", 0)
            return {"nodes": {"node_0001": {"id": "node_0001"}}, "meta": {}, "technique_stats": {}}

        monkeypatch.setattr(srv, "_load_graph_json_text", flaky_loader)

        watcher = srv.GraphWatcher()
        payload = json.loads(_run(watcher.get_initial()))

        assert payload["added"] == ["node_0001"], (
            "get_initial must recover a torn/transient read instead of "
            "falling back to the empty skeleton"
        )

    def test_notify_skips_the_broadcast_when_graph_unreadable(self, tmp_path, monkeypatch):
        import automil.viz.server as srv

        monkeypatch.setattr(srv, "GRAPH_FILE", tmp_path / "does_not_exist.json")
        monkeypatch.setattr(srv, "GPU_STATE_FILE", tmp_path / "gpu_state.json")

        watcher = srv.GraphWatcher()
        queue: asyncio.Queue = asyncio.Queue()
        watcher.subscribers.append(queue)

        _run(watcher._notify())

        assert queue.empty(), "nothing should be broadcast when graph.json cannot be read"


def test_running_overlay_reads_typed_cpu_execution_slots(tmp_path, monkeypatch):
    import automil.viz.server as srv

    state_file = tmp_path / "gpu_state.json"
    state_file.write_text(json.dumps({
        "gpus": {},
        "execution_slots": {
            "cpu:0": {
                "accelerator": "cpu",
                "device_index": None,
                "running": ["node_0001"],
                "capacity": 1,
            }
        },
    }))
    monkeypatch.setattr(srv, "GPU_STATE_FILE", state_file)
    data = {"nodes": {"node_0001": {"status": "pending"}}}

    srv.GraphWatcher._overlay_running_status(data)

    assert data["nodes"]["node_0001"]["status"] == "running"
