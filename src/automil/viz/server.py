#!/usr/bin/env python3
"""Experiment graph visualization server.

Watches graph.json for changes via inotify, pushes updates to browser via SSE.
Serves static D3.js dashboard.

Usage:
    uv run python autoMIL/viz/server.py start [--port 8420]
    uv run python autoMIL/viz/server.py status
    uv run python autoMIL/viz/server.py stop
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
from pathlib import Path

try:
    from aiohttp import web
except ImportError:
    print("aiohttp required: uv add aiohttp")
    sys.exit(1)

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    print("watchdog required: uv add watchdog")
    sys.exit(1)

VIZ_DIR = Path(__file__).parent
STATIC_DIR = VIZ_DIR / "static"

# These are set at runtime by cmd_start() based on project_root
GRAPH_FILE: Path = Path("graph.json")
GPU_STATE_FILE: Path = Path("gpu_state.json")
PID_FILE: Path = Path("viz_server.pid")
LOG_FILE: Path = Path("viz_server.log")

DEFAULT_PORT = 8420


class GraphWatcher(FileSystemEventHandler):
    def __init__(self):
        self.subscribers: list[asyncio.Queue] = []
        self._prev_data: dict | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def _maybe_notify(self, path: str):
        name = Path(path).name
        if name in (GRAPH_FILE.name, GPU_STATE_FILE.name) and self._loop:
            self._loop.call_soon_threadsafe(
                asyncio.ensure_future, self._notify()
            )

    @staticmethod
    def _overlay_running_status(data: dict) -> None:
        """Mark nodes as 'running' based on orchestrator gpu_state.json.

        graph.json is only updated by submit/reconcile, so in-flight
        experiments still show as 'pending' there. The orchestrator
        rewrites gpu_state.json every poll cycle with the running node
        IDs per GPU; we merge that in so the viz reflects live state.
        """
        try:
            state = json.loads(GPU_STATE_FILE.read_text())
        except (json.JSONDecodeError, FileNotFoundError, OSError):
            return
        running_ids: set[str] = set()
        for gpu in state.get("gpus", {}).values():
            for nid in gpu.get("running", []) or []:
                running_ids.add(nid)
        nodes = data.get("nodes", {})
        for nid in running_ids:
            node = nodes.get(nid)
            if node is not None:
                node["status"] = "running"

    def on_modified(self, event):
        self._maybe_notify(event.src_path)

    def on_moved(self, event):
        self._maybe_notify(event.dest_path)

    def on_created(self, event):
        self._maybe_notify(event.src_path)

    async def _notify(self):
        try:
            data = json.loads(GRAPH_FILE.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            return

        self._overlay_running_status(data)

        changed, added, removed = [], [], []
        meta_changed = False
        if self._prev_data is not None:
            prev_nodes = set(self._prev_data.get("nodes", {}).keys())
            curr_nodes = set(data.get("nodes", {}).keys())
            added = list(curr_nodes - prev_nodes)
            removed = list(prev_nodes - curr_nodes)
            for nid in curr_nodes & prev_nodes:
                if data["nodes"][nid] != self._prev_data["nodes"].get(nid):
                    changed.append(nid)
            meta_changed = (
                data.get("meta") != self._prev_data.get("meta")
                or data.get("technique_stats") != self._prev_data.get("technique_stats")
            )
            # gpu_state.json is rewritten every orchestrator poll cycle. If the
            # overlaid payload is byte-identical to the previous broadcast, the
            # client would only see a wasteful d3 force-layout reheat. Skip.
            if not (changed or added or removed or meta_changed):
                return

        self._prev_data = data
        event = json.dumps({
            "type": "graph_update",
            "changed": changed,
            "added": added,
            "removed": removed,
            "full_graph": data,
        })

        dead = []
        for q in self.subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self.subscribers.remove(q)

    async def get_initial(self) -> str:
        try:
            data = json.loads(GRAPH_FILE.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            data = {"nodes": {}, "meta": {}, "technique_stats": {}}
        self._overlay_running_status(data)
        self._prev_data = data
        return json.dumps({
            "type": "graph_update",
            "changed": [],
            "added": list(data.get("nodes", {}).keys()),
            "full_graph": data,
        })


watcher = GraphWatcher()


async def sse_handler(request):
    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )
    await response.prepare(request)

    initial = await watcher.get_initial()
    await response.write(f"data: {initial}\n\n".encode())

    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    watcher.subscribers.append(queue)

    try:
        while True:
            event = await queue.get()
            if event is None:  # shutdown sentinel
                break
            await response.write(f"data: {event}\n\n".encode())
    except (asyncio.CancelledError, ConnectionResetError):
        pass
    finally:
        if queue in watcher.subscribers:
            watcher.subscribers.remove(queue)

    return response


async def _on_shutdown(app):
    """Wake all SSE handlers so they exit cleanly instead of blocking shutdown."""
    for q in list(watcher.subscribers):
        try:
            q.put_nowait(None)
        except asyncio.QueueFull:
            pass


async def index_handler(request):
    return web.FileResponse(STATIC_DIR / "index.html")


async def promotion_rate_handler(request):
    """GTE-06 / D-144: serve current promotion_rate + gate-health diagnostic.

    Reads graph.json on every request (no internal counter that can drift).
    Soft-fails to safe zeros when graph.json is absent or contains no nominations.
    """
    from automil.gate.stats import diagnose_gate_health
    from automil.graph import ExperimentGraph

    response_data = {
        "promotion_rate": 0.0,
        "nominated": 0,
        "promoted": 0,
        "health_diagnostic": "no data — graph.json not found",
        "window_days": 30,
    }
    try:
        if GRAPH_FILE.exists():
            graph = ExperimentGraph(path=str(GRAPH_FILE))
            rate = graph.promotion_rate(days=30)
            nominated_nodes = graph.nominations_in_window(days=30)
            nominated_count = len(nominated_nodes)
            promoted_count = sum(
                1 for n in nominated_nodes if n.get("status") == "registered"
            )
            if nominated_count > 0:
                diagnostic = diagnose_gate_health(rate)
            else:
                diagnostic = "no data — zero nominations in 30-day window"
            response_data = {
                "promotion_rate": rate,
                "nominated": nominated_count,
                "promoted": promoted_count,
                "health_diagnostic": diagnostic,
                "window_days": 30,
            }
    except Exception:
        # Soft-fail: return safe defaults (Pitfall 5a parallel discipline)
        pass
    return web.json_response(response_data)


@web.middleware
async def _no_cache_static(request, handler):
    response = await handler(request)
    if request.path.startswith("/static/") or request.path == "/":
        response.headers["Cache-Control"] = "no-store, must-revalidate"
    return response


def create_app() -> web.Application:
    app = web.Application(middlewares=[_no_cache_static])
    app.router.add_get("/", index_handler)
    app.router.add_get("/events", sse_handler)
    app.router.add_get("/api/promotion-rate", promotion_rate_handler)
    app.router.add_static("/static", STATIC_DIR)
    app.on_shutdown.append(_on_shutdown)
    return app


def cmd_start(
    port: int | None = None,
    project_root: Path | None = None,
    host: str | None = None,
):
    global GRAPH_FILE, GPU_STATE_FILE, PID_FILE, LOG_FILE
    if project_root is None:
        project_root = Path.cwd()
    automil_dir = project_root / "automil"
    GRAPH_FILE = automil_dir / "graph.json"
    GPU_STATE_FILE = automil_dir / "orchestrator" / "gpu_state.json"
    PID_FILE = automil_dir / "orchestrator" / "viz_server.pid"
    LOG_FILE = automil_dir / "orchestrator" / "viz_server.log"

    # Bind host + port resolution — load config once for both.
    # Host resolution order: explicit arg > automil/config.yaml viz.host
    # > AUTOMIL_VIZ_HOST env var > 127.0.0.1 (loopback default).
    # Port resolution order: explicit arg > automil/config.yaml viz.port
    # > DEFAULT_PORT (8420).
    # Loopback default keeps the dashboard off the LAN unless the operator
    # opts in explicitly. The SSE stream and gpu_state.json carry PIDs,
    # GPU utilization, and node descriptions; on a shared workstation those
    # should not be browseable by every host on the subnet.
    if host is None or port is None:
        cfg_loaded: dict = {}
        config_path = automil_dir / "config.yaml"
        if config_path.exists():
            try:
                import yaml as _yaml  # noqa: PLC0415
                cfg_loaded = _yaml.safe_load(config_path.read_text()) or {}
            except Exception:
                cfg_loaded = {}

        if host is None:
            cfg_host: str | None = (cfg_loaded.get("viz") or {}).get("host")
            host = cfg_host or os.environ.get("AUTOMIL_VIZ_HOST") or "127.0.0.1"

        if port is None:
            raw_port = (cfg_loaded.get("viz") or {}).get("port")
            try:
                port = int(raw_port) if raw_port is not None else DEFAULT_PORT
            except (TypeError, ValueError):
                port = DEFAULT_PORT

    if PID_FILE.exists():
        pid = int(PID_FILE.read_text().strip())
        try:
            os.kill(pid, 0)
            print(f"Viz server already running (PID {pid})")
            return
        except OSError:
            PID_FILE.unlink()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
    )

    PID_FILE.write_text(str(os.getpid()) + "\n")

    observer = Observer()
    observer.schedule(watcher, str(automil_dir), recursive=False)
    orch_dir = automil_dir / "orchestrator"
    if orch_dir.exists():
        observer.schedule(watcher, str(orch_dir), recursive=False)
    observer.start()

    async def run_server():
        app = create_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port, shutdown_timeout=2.0)
        await site.start()
        if host in ("0.0.0.0", "::"):
            logging.warning(
                "Viz server bound to %s:%d — reachable from any host on the "
                "network. The SSE stream exposes graph.json and gpu_state.json "
                "(PIDs, GPU utilization, node descriptions). Set viz.host to "
                "'127.0.0.1' in automil/config.yaml unless remote access is "
                "intended.", host, port,
            )
        logging.info(f"Viz server running on http://{host}:{port}")

        # Wait for shutdown signal
        stop_event = asyncio.Event()
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop_event.set)

        await stop_event.wait()
        logging.info("Shutting down...")
        await runner.cleanup()

    loop = asyncio.new_event_loop()
    watcher.set_loop(loop)

    try:
        loop.run_until_complete(run_server())
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        observer.stop()
        observer.join(timeout=2)
        if PID_FILE.exists():
            PID_FILE.unlink()
        logging.info("Viz server stopped.")


def _resolve_pid_file(project_root: Path | None = None) -> Path:
    """Get the PID file path for the viz server."""
    if project_root is None:
        project_root = Path.cwd()
    return project_root / "automil" / "orchestrator" / "viz_server.pid"


def cmd_status(project_root: Path | None = None):
    pid_file = _resolve_pid_file(project_root)
    if pid_file.exists():
        pid = int(pid_file.read_text().strip())
        try:
            os.kill(pid, 0)
            print(f"Viz server: RUNNING (PID {pid})")
        except OSError:
            print("Viz server: DEAD (stale PID file)")
    else:
        print("Viz server: NOT RUNNING")


def cmd_stop(project_root: Path | None = None):
    pid_file = _resolve_pid_file(project_root)
    if not pid_file.exists():
        print("Viz server not running")
        return
    pid = int(pid_file.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Sent SIGTERM to PID {pid}")
    except OSError as e:
        print(f"Failed to stop: {e}")
        pid_file.unlink()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "start":
        port = DEFAULT_PORT
        if "--port" in sys.argv:
            idx = sys.argv.index("--port")
            port = int(sys.argv[idx + 1])
        cmd_start(port)
    elif cmd == "status":
        cmd_status()
    elif cmd == "stop":
        cmd_stop()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
