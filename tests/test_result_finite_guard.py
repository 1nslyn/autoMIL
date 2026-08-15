"""CR-1a (audit 2026-07-23): non-finite composite/metric values must never
influence val-firewall selection, and graph.json must stay standards-valid JSON.

Covers three coordinated guards:
- ``validate_result`` rejects non-finite ``composite`` and metric values.
- ``Runner.collect_result`` degrades an ``Infinity``/``NaN`` result.json to a
  crash result (no keep/best influence).
- ``ExperimentGraph.save`` writes finite JSON (``allow_nan=False``).
"""
from __future__ import annotations

import json
import math

import pytest

from automil.schemas import validate_result, ValidationError


def _valid_payload(composite=0.83):
    return {
        "status": "completed",
        "composite": composite,
        "metrics": {"val_auc": 0.85, "val_bacc": 0.81},
        "held_out": {"test_auc": 0.80},
        "elapsed_seconds": 12.0,
        "peak_vram_mb": 4096,
    }


def test_validate_result_accepts_finite():
    validate_result(_valid_payload())  # must not raise


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_validate_result_rejects_nonfinite_composite(bad):
    payload = _valid_payload(composite=bad)
    with pytest.raises(ValidationError):
        validate_result(payload)


@pytest.mark.parametrize("block", ["metrics", "held_out"])
@pytest.mark.parametrize("bad", [math.inf, math.nan])
def test_validate_result_rejects_nonfinite_metric(block, bad):
    payload = _valid_payload()
    payload[block] = {"x": bad}
    with pytest.raises(ValidationError):
        validate_result(payload)


@pytest.mark.parametrize("token", ["Infinity", "-Infinity", "NaN"])
def test_collect_result_degrades_nonfinite_token_to_crash(tmp_path, token):
    from automil.runner import Runner

    worktree = tmp_path / "wt"
    worktree.mkdir()
    archive = tmp_path / "archive" / "node_0001"
    archive.mkdir(parents=True)
    # Hand-write the raw JSON so the non-finite arrives as a literal token, exactly
    # as an agent-authored train.py would emit via json.dump.
    (worktree / "result.json").write_text(
        '{"status": "completed", "composite": %s, "metrics": {}}' % token
    )

    runner = Runner(project_root=tmp_path, automil_dir=tmp_path / "automil")
    out = runner.collect_result(worktree, archive)

    assert out is not None
    assert out["status"] == "crash"
    assert out["composite"] == 0.0
    assert "rejected" in out.get("error", "")


def test_collect_result_parses_finite_normally(tmp_path):
    from automil.runner import Runner

    worktree = tmp_path / "wt"
    worktree.mkdir()
    archive = tmp_path / "archive" / "node_0002"
    archive.mkdir(parents=True)
    (worktree / "result.json").write_text(
        json.dumps({"status": "completed", "composite": 0.7, "metrics": {"val_auc": 0.7}})
    )

    runner = Runner(project_root=tmp_path, automil_dir=tmp_path / "automil")
    out = runner.collect_result(worktree, archive)

    assert out["status"] == "completed"
    assert out["composite"] == 0.7


def test_graph_save_writes_finite_json(tmp_path):
    from automil.graph import ExperimentGraph

    path = tmp_path / "graph.json"
    g = ExperimentGraph(path=str(path))
    g.save()
    text = path.read_text()
    # No bare non-finite tokens that would break non-Python JSON readers.
    assert "NaN" not in text
    assert "Infinity" not in text
    json.loads(text)  # round-trips as standards-valid JSON


def test_graph_save_raises_on_nonfinite(tmp_path):
    from automil.graph import ExperimentGraph

    path = tmp_path / "graph.json"
    g = ExperimentGraph(path=str(path))
    # Inject a non-finite into the persisted structure to prove save() fails loud
    # rather than emitting an invalid-JSON token (M-3).
    g.meta["best_composite"] = math.inf
    with pytest.raises(ValueError):
        g.save()
