"""Validation-only projections of graph and result payloads, and the verdict.

This is the record's one firewall enforcement point: every node, result and
graph the dashboard serves passes through here, and nothing here reads a
file. Held-out-named keys are dropped wherever the framework's own leak rule
(``automil.firewall``) looks for them, and error text is redacted line by line.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

from automil.firewall import is_held_out_metric_key, redact_held_out
from automil.graph import (
    KEEP_CLASS,
    effective_accept_margin,
    guard_basis,
    keep_or_discard,
    margin_se_basis,
    node_primary_se,
)
from automil.viz.clock import HostClock, to_utc_iso

RECORD_SCHEMA = 1
_SEALED_BLOCKS = ("held_out", "summary")
_JUDGED = frozenset(KEEP_CLASS | {"discard"})


def project_metrics(metrics: object) -> dict[str, Any]:
    """A metrics block without any held-out-named key."""
    if not isinstance(metrics, Mapping):
        return {}
    return {str(k): v for k, v in metrics.items() if not is_held_out_metric_key(str(k))}


def _project_folds(folds: object) -> list[Any] | None:
    if not isinstance(folds, list):
        return None
    out = []
    for entry in folds:
        if isinstance(entry, Mapping):
            clean = {k: v for k, v in entry.items() if k != "metrics"}
            if "metrics" in entry:
                clean["metrics"] = project_metrics(entry.get("metrics"))
            out.append(clean)
        else:
            out.append(entry)
    return out


def project_node(node: Mapping[str, Any], clock: HostClock) -> dict[str, Any]:
    """One graph node, validation-only, with stamps in UTC and a gate-child flag."""
    metadata = node.get("metadata") if isinstance(node.get("metadata"), Mapping) else {}
    clean_meta = {k: v for k, v in metadata.items() if k not in _SEALED_BLOCKS}
    if "validation_folds" in metadata:
        clean_meta["validation_folds"] = _project_folds(metadata.get("validation_folds"))
    projected = {k: v for k, v in node.items() if k not in _SEALED_BLOCKS}
    projected["metrics"] = project_metrics(node.get("metrics"))
    projected["metadata"] = clean_meta
    projected["gate_child"] = bool(metadata.get("held_out", False))
    projected["created_at"] = to_utc_iso(node.get("created_at"), clock)
    if isinstance(node.get("error"), str):
        projected["error"] = redact_held_out(node["error"])
    return projected


def running_node_ids(gpu_state: Mapping[str, Any] | None) -> tuple[str, ...]:
    """Node ids the orchestrator reports in flight (typed slots, legacy ``gpus``)."""
    if not isinstance(gpu_state, Mapping):
        return ()
    slots = gpu_state.get("execution_slots")
    if not isinstance(slots, Mapping):
        slots = gpu_state.get("gpus") if isinstance(gpu_state.get("gpus"), Mapping) else {}
    ids: set[str] = set()
    for slot in slots.values():
        if isinstance(slot, Mapping):
            ids.update(str(n) for n in slot.get("running") or [] if isinstance(n, str))
    return tuple(sorted(ids))


def project_graph(
    raw: Mapping[str, Any], clock: HostClock, running: Iterable[str] = ()
) -> dict[str, Any]:
    """The graph the dashboard sees: projected nodes plus the running overlay."""
    running_ids = tuple(sorted(set(running)))
    nodes_in = raw.get("nodes") if isinstance(raw.get("nodes"), Mapping) else {}
    nodes = {}
    for node_id, node in nodes_in.items():
        if not isinstance(node, Mapping):
            continue
        projected = project_node(node, clock)
        if node_id in running_ids:
            projected["status"] = "running"
        nodes[str(node_id)] = projected
    return {
        "schema": RECORD_SCHEMA,
        "nodes": nodes,
        "meta": dict(raw.get("meta") or {}),
        "technique_stats": dict(raw.get("technique_stats") or {}),
        "running": list(running_ids),
    }


def project_result(result: Mapping[str, Any] | None, clock: HostClock) -> dict[str, Any] | None:
    """A terminal result (``completed/<node>.json`` or ``result.json``), val-only."""
    if not isinstance(result, Mapping):
        return None
    projected = {k: v for k, v in result.items() if k not in _SEALED_BLOCKS}
    projected["metrics"] = project_metrics(result.get("metrics"))
    if "validation_folds" in result:
        projected["validation_folds"] = _project_folds(result.get("validation_folds"))
    if isinstance(result.get("error"), str):
        projected["error"] = redact_held_out(result["error"])
    if "completed_at" in result:
        projected["completed_at"] = to_utc_iso(result.get("completed_at"), clock)
    return projected


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def verdict_unavailable(node: Mapping[str, Any]) -> str | None:
    """Why no keep/discard verdict applies to this node, or ``None`` if one does."""
    if node.get("type") != "executed":
        return "not executed"
    status = node.get("status")
    if status in _JUDGED:
        return None
    if status in ("partial", "crash", "cancelled", "oom", "timeout"):
        return str(status)
    return "not terminal"


def node_verdict(
    meta: Mapping[str, Any] | None,
    parent: Mapping[str, Any] | None,
    node: Mapping[str, Any],
) -> dict[str, Any] | None:
    """The keep/discard decision, reproduced with the graph's own helpers.

    Mirrors what ``automil rank`` prints: the bar is
    ``max(δ, k × SE)`` with the paired SE when both nodes carry the same folds,
    and the companion guard can only veto. ``None`` when no verdict applies.
    """
    if verdict_unavailable(node) is not None:
        return None
    meta = dict(meta or {})
    scoring = meta.get("scoring") if isinstance(meta.get("scoring"), Mapping) else {}
    decision = keep_or_discard(meta, dict(parent) if parent else None, dict(node))
    stored = str(node.get("status"))
    stored_class = "keep" if stored in KEEP_CLASS else "discard"
    child_value = _number(node.get("primary_value"))
    if parent is None:
        explanation = (
            f"Root node: kept because its value {child_value:.4f} is above zero."
            if decision == "keep" else "Root node: discarded, its value is not above zero."
        )
        return {
            "decision": decision,
            "stored_status": stored,
            "consistent": decision == stored_class,
            "parent_id": None,
            "parent_primary_value": None,
            "child_primary_value": child_value,
            "delta": None,
            "bar": None,
            "accept_margin": _number(scoring.get("accept_margin")),
            "se_multiplier": _number(scoring.get("se_multiplier", 1.0)),
            "basis": "none",
            "basis_se": None,
            "guard": {"verdict": "none", "delta": None, "metric": None, "margin": None, "decisive": False},
            "explanation": explanation,
        }
    parent_value = _number(parent.get("primary_value"))
    delta = child_value - parent_value
    basis, basis_se = margin_se_basis(meta, dict(parent), dict(node))
    bar = effective_accept_margin(meta, dict(parent), dict(node))
    g_verdict, g_delta, g_metric, g_margin = guard_basis(meta, dict(parent), dict(node))
    decisive = g_verdict == "fail" and delta > bar
    se_text = {
        "paired": f"paired SE {basis_se:.4f}" if basis_se is not None else "paired SE unavailable",
        "marginal": f"marginal SE {basis_se:.4f}" if basis_se is not None else "marginal SE unavailable",
        "none": "no SE",
    }[basis]
    if delta > bar:
        head = f"{delta:+.4f} over the parent, above the bar of {bar:.4f} ({se_text}, floor {_number(scoring.get('accept_margin')):.4f})."
    else:
        head = f"{delta:+.4f} over the parent, not above the bar of {bar:.4f} ({se_text}, floor {_number(scoring.get('accept_margin')):.4f})."
    if g_verdict == "none":
        tail = ""
    else:
        g_value = f"{g_delta:+.4f}" if g_delta is not None else "unreported"
        g_bar = f" against a bar of {g_margin:.4f}" if g_margin is not None else ""
        tail = f" Guard {g_metric or 'companion'} {g_value}{g_bar}: {g_verdict}."
        if decisive:
            tail += " The guard decided."
    return {
        "decision": decision,
        "stored_status": stored,
        "consistent": decision == stored_class,
        "parent_id": parent.get("id"),
        "parent_primary_value": parent_value,
        "child_primary_value": child_value,
        "delta": delta,
        "bar": bar,
        "accept_margin": _number(scoring.get("accept_margin")),
        "se_multiplier": _number(scoring.get("se_multiplier", 1.0)),
        "basis": basis,
        "basis_se": basis_se,
        "child_se": node_primary_se(dict(node)),
        "guard": {"verdict": g_verdict, "delta": g_delta, "metric": g_metric, "margin": g_margin, "decisive": decisive},
        "explanation": ("Kept: " if decision == "keep" else "Discarded: ") + head + tail,
    }


def walk_keys(payload: object, path: str = "") -> Iterable[tuple[str, str]]:
    """Every (path, key) pair in a nested payload; used by the firewall tests."""
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            here = f"{path}.{key}" if path else str(key)
            yield here, str(key)
            yield from walk_keys(value, here)
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            yield from walk_keys(value, f"{path}[{index}]")
