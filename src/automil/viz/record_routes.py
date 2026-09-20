"""HTTP routes for the record, and the CORS policy for a hosted site.

The live server answers the same paths the export writes, so one frontend
reads both. Cross-origin reads are allowed only from the listed site origins
and from loopback dev origins; the dashboard binds loopback and has no auth,
so a wildcard would let any open tab read the record through the tunnel.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable
from urllib.parse import urlsplit

from aiohttp import web

RECORD_PREFIX = "/record"
DEFAULT_CORS_ORIGINS = ("https://automil.org", "https://www.automil.org")
_RUN = r"[A-Za-z0-9._-]{1,120}"
_NODE = r"node_\d{4,}"
_SESSION = r"[0-9a-f-]{8,64}"
_TOKEN = r"[A-Za-z0-9_-]{1,120}"
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "[::1]"})


def origin_allowed(origin: str | None, allowed: Iterable[str]) -> bool:
    """Listed origins exactly; any loopback origin on any port."""
    if not origin:
        return False
    if origin in set(allowed):
        return True
    parts = urlsplit(origin)
    return parts.scheme in ("http", "https") and parts.hostname in _LOOPBACK_HOSTS


def cors_headers(request: web.Request, allowed: Iterable[str]) -> dict[str, str]:
    """The CORS headers a record or events response gets for this request's origin."""
    origin = request.headers.get("Origin")
    if not origin or not origin_allowed(origin, allowed):
        return {}
    if not (request.path.startswith(RECORD_PREFIX) or request.path == "/events"):
        return {}
    return {
        "Access-Control-Allow-Origin": origin,
        "Vary": "Origin",
        "Access-Control-Allow-Methods": "GET, OPTIONS",
        "Access-Control-Allow-Headers": "Cache-Control, Last-Event-ID",
        "Access-Control-Allow-Private-Network": "true",
        "Access-Control-Max-Age": "3600",
    }


def cors_middleware(allowed: Iterable[str]) -> Callable[..., Any]:
    allowed = tuple(allowed)

    @web.middleware
    async def middleware(request: web.Request, handler: Callable[..., Any]) -> web.StreamResponse:
        if request.method == "OPTIONS":
            response: web.StreamResponse = web.Response(status=204)
        else:
            response = await handler(request)
        # A streaming response (SSE) has already sent its headers: the handler
        # merged them itself through cors_headers before preparing.
        if not response.prepared:
            for name, value in cors_headers(request, allowed).items():
                response.headers[name] = value
        return response

    return middleware


def _source(request: web.Request, run_id: str):
    source = request.app.get("run_source")
    if source is None or source.run_id != run_id:
        raise web.HTTPNotFound()
    return source


def _json(payload: Any) -> web.Response:
    if payload is None:
        raise web.HTTPNotFound()
    response = web.json_response(payload)
    response.headers["Cache-Control"] = "no-store, must-revalidate"
    return response


async def index_json(request: web.Request) -> web.Response:
    from automil.viz.record import build_index

    source = request.app.get("run_source")
    return _json(build_index([source] if source else [], "live"))


async def run_file(request: web.Request) -> web.Response:
    source = _source(request, request.match_info["run"])
    name = request.match_info["name"]
    builders = {
        "graph": source.build_graph,
        "timeline": source.build_timeline,
        "sessions": source.build_sessions,
        "agent_links": source.build_links,
        "notes": source.build_notes,
        "certified": source.build_certified,
    }
    builder = builders.get(name)
    if builder is None:
        raise web.HTTPNotFound()
    return _json(builder())


async def node_json(request: web.Request) -> web.Response:
    source = _source(request, request.match_info["run"])
    return _json(source.build_node(request.match_info["node"]))


async def node_file(request: web.Request) -> web.Response:
    source = _source(request, request.match_info["run"])
    rel = request.match_info["path"]
    if not rel.endswith(".json"):
        raise web.HTTPNotFound()
    return _json(source.build_overlay_file(request.match_info["node"], rel[: -len(".json")]))


async def turns_chunk(request: web.Request) -> web.Response:
    source = _source(request, request.match_info["run"])
    return _json(source.build_chunk(request.match_info["session"], int(request.match_info["chunk"])))


async def full_result(request: web.Request) -> web.Response:
    source = _source(request, request.match_info["run"])
    return _json(source.build_full_result(request.match_info["session"], request.match_info["tool"]))


async def agent_json(request: web.Request) -> web.Response:
    source = _source(request, request.match_info["run"])
    return _json(source.build_agent(request.match_info["session"], request.match_info["agent"]))


def register_record_routes(app: web.Application) -> None:
    p = RECORD_PREFIX
    app.router.add_get(f"{p}/index.json", index_json)
    app.router.add_get(f"{p}/runs/{{run:{_RUN}}}/{{name:[a-z_]+}}.json", run_file)
    app.router.add_get(f"{p}/runs/{{run:{_RUN}}}/nodes/{{node:{_NODE}}}.json", node_json)
    app.router.add_get(f"{p}/runs/{{run:{_RUN}}}/nodes/{{node:{_NODE}}}/files/{{path:.+}}", node_file)
    app.router.add_get(f"{p}/runs/{{run:{_RUN}}}/sessions/{{session:{_SESSION}}}/turns/{{chunk:\\d+}}.json", turns_chunk)
    app.router.add_get(f"{p}/runs/{{run:{_RUN}}}/sessions/{{session:{_SESSION}}}/results/{{tool:{_TOKEN}}}.json", full_result)
    app.router.add_get(f"{p}/runs/{{run:{_RUN}}}/sessions/{{session:{_SESSION}}}/agents/{{agent:{_TOKEN}}}.json", agent_json)
    app.router.add_route("OPTIONS", f"{p}/{{tail:.*}}", _preflight)
    app.router.add_route("OPTIONS", "/events", _preflight)


async def _preflight(request: web.Request) -> web.Response:
    return web.Response(status=204)

