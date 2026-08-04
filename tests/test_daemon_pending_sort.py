"""``_get_pending`` must not raise on mixed ``submitted_at`` shapes (L-7).

The pending queue is sorted by ``(priority, submitted_at)``. ``submitted_at``
can be a proper ISO string (the normal case written by ``automil submit``),
an explicitly-null JSON value (a present key with a ``None`` value), or
absent entirely (an older or hand-written spec). Python 3 raises
``TypeError`` comparing ``None`` to ``str``, and that exception propagates
out of ``list.sort()`` and out of ``_get_pending`` itself -- taking down the
whole poll tick (``tick()`` -> ``_get_pending()``), not just the offending
spec. One malformed queue file therefore stalls scheduling for every pending
experiment.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _make_orch(tmp_path: Path) -> Any:
    from automil.orchestrator import ExperimentOrchestrator

    automil_dir = tmp_path / "automil"
    automil_dir.mkdir(parents=True, exist_ok=True)
    (automil_dir / "config.yaml").write_text("orchestrator: {}\n")
    (tmp_path / ".git").mkdir(exist_ok=True)
    orch = ExperimentOrchestrator(project_root=tmp_path, automil_dir=automil_dir)
    orch.queue_dir.mkdir(parents=True, exist_ok=True)
    return orch


def _write_spec(orch: Any, node_id: str, **fields: Any) -> None:
    spec: dict = {
        "id": node_id,
        "description": f"spec {node_id}",
        "base_commit": "deadbeef",
        "overlay_dir": f"archive/{node_id}",
        "overlay_manifest": {},
        "deletions": [],
        "estimated_vram_gb": 0.5,
        "metadata": {"backend": "local"},
        **fields,
    }
    (orch.queue_dir / f"{node_id}.json").write_text(json.dumps(spec, indent=2))


def test_get_pending_sorts_mixed_submitted_at_without_raising(tmp_path):
    """L-7 reproducer: string / absent / explicit-None submitted_at together."""
    orch = _make_orch(tmp_path)
    _write_spec(orch, "node_0001", submitted_at="2026-07-01T00:00:00")
    _write_spec(orch, "node_0002")  # absent entirely
    _write_spec(orch, "node_0003", submitted_at=None)  # present but null
    _write_spec(orch, "node_0004", submitted_at=12345)  # malformed: wrong type

    pending = orch._get_pending()  # must not raise TypeError

    assert {p["id"] for p in pending} == {
        "node_0001", "node_0002", "node_0003", "node_0004",
    }


def test_get_pending_treats_untimestamped_specs_as_earliest(tmp_path):
    """Deterministic ordering: no-timestamp specs sort before any real timestamp.

    All four specs tie on priority (default 2), so the tiebreak is
    ``submitted_at``. A spec with no usable timestamp coerces to ``""``,
    which sorts before any real ISO-8601 string, and Python's sort is
    stable, so untimestamped specs fall back to queue (filename) order
    among themselves.
    """
    orch = _make_orch(tmp_path)
    _write_spec(orch, "node_0001", submitted_at="2026-07-01T00:00:00")
    _write_spec(orch, "node_0002")  # absent -> ""
    _write_spec(orch, "node_0003", submitted_at=None)  # None -> ""

    pending = orch._get_pending()

    assert [p["id"] for p in pending] == ["node_0002", "node_0003", "node_0001"]
