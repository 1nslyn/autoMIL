"""A queued spec that no GPU on the host can ever hold is refused at launch,
not left in the queue forever.

``_find_best_gpu`` returns ``None`` whenever ``free - safety - allocated`` is
below the request and the tick simply skips the spec; nothing archived or
refused it, its node stayed ``running`` in the graph (so ``dequeue`` and
``cancel`` both refused it), and a phased cell held every later batch on it.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from automil.backends._orchestrator_daemon import ExperimentOrchestrator, GPUInfo


def _orch(tmp_path):
    automil_dir = tmp_path / "automil"
    automil_dir.mkdir()
    (automil_dir / "config.yaml").write_text("orchestrator: {safety_margin_gb: 1.0}\n")
    (tmp_path / ".git").mkdir()
    return ExperimentOrchestrator(project_root=tmp_path, automil_dir=automil_dir)


def _queued(orch, node_id: str, vram_gb: float) -> dict:
    orch.graph.add_proposed(parent_id=None, description=node_id, techniques=[])
    orch.graph.mark_running(node_id)
    orch.graph.save()
    spec = {"id": node_id, "description": node_id, "estimated_vram_gb": vram_gb,
            "overlay_manifest": {}, "deletions": [], "metadata": {}}
    path = orch.queue_dir / f"{node_id}.json"
    path.write_text(json.dumps(spec))
    return {**spec, "_file": str(path)}


ONE_48GB_GPU = [GPUInfo(index=0, total_mb=48 * 1024, free_mb=40 * 1024, utilization=0)]


class TestUnplaceableSpec:
    def test_a_request_above_the_largest_gpu_is_refused_and_recorded(self, tmp_path):
        orch = _orch(tmp_path)
        spec = _queued(orch, "node_0001", 999.0)
        with patch("automil.backends._orchestrator_daemon.query_gpus", return_value=ONE_48GB_GPU):
            assert orch._refuse_if_unplaceable(spec, 999.0) is True
        assert not (orch.queue_dir / "node_0001.json").exists()
        archived = json.loads((orch.archive_dir / "node_0001" / "spec.json").read_text())
        assert archived["metadata"]["cap_refused"] is True
        assert archived["metadata"]["cancel_reason"] == "unplaceable"
        assert not (orch.archive_dir / "node_0001" / "result.json").exists()   # never charged
        node = json.loads((orch.automil_dir / "graph.json").read_text())["nodes"]["node_0001"]
        assert node["status"] == "cancelled"

    @pytest.mark.parametrize("vram", [2.0, 47.0])
    def test_a_request_that_fits_the_largest_gpu_stays_queued(self, tmp_path, vram):
        """Fitting the host is enough: free memory is a matter of time."""
        orch = _orch(tmp_path)
        spec = _queued(orch, "node_0001", vram)
        with patch("automil.backends._orchestrator_daemon.query_gpus", return_value=ONE_48GB_GPU):
            assert orch._refuse_if_unplaceable(spec, vram) is False
        assert (orch.queue_dir / "node_0001.json").exists()

    def test_no_visible_gpu_refuses_nothing(self, tmp_path):
        orch = _orch(tmp_path)
        spec = _queued(orch, "node_0001", 999.0)
        with patch("automil.backends._orchestrator_daemon.query_gpus", return_value=[]):
            assert orch._refuse_if_unplaceable(spec, 999.0) is False

    def test_the_tick_refuses_before_looking_for_a_gpu(self, tmp_path):
        orch = _orch(tmp_path)
        _queued(orch, "node_0001", 999.0)
        with patch("automil.backends._orchestrator_daemon.query_gpus", return_value=ONE_48GB_GPU), \
                patch.object(orch, "_find_best_gpu") as find:
            orch.tick()
        find.assert_not_called()
        assert not (orch.queue_dir / "node_0001.json").exists()
        assert (orch.archive_dir / "node_0001" / "spec.json").exists()
