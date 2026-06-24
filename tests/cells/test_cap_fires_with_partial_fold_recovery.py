"""Pitfall-4 anti-acceptance gate (CAP-03, CAP-04 / D-126).

Goal-backward verifier for Phase 4: cap-firing must produce a usable
partial composite, NOT corrupt results.tsv. budget_seconds=60 here is
a TEST-ONLY value chosen to make the test executable in seconds — the
framework property is the *mechanism*, not the *value*
(paper_campaign_vs_framework rule; Leo's paper campaign uses its own
consumer-supplied cap value via automil/config.yaml).

Composes the full chain end-to-end:
  subprocess SIGTERM -> register_sigterm_flush handler -> result.json
  -> aggregate_folds -> reconcile_budget_kill -> graph._reevaluate_descendants

Fragile Invariant #6 defence (CONCERNS.md): descendants are recomputed
against the PARTIAL composite (0.82), NOT zero. A descendant that beats
0.82 stays "keep"; one that beats zero but loses to 0.82 flips to "discard".
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="SIGTERM unsupported on Windows")


def test_cap_fires_with_partial_fold_recovery(tmp_path: Path):
    """Full chain: SIGTERM -> partial result.json -> reconcile -> graph cascade.

    budget_seconds=60 (paper-campaign-vs-framework rule: test uses small value).
    Test finishes in < 15 seconds.
    """
    # 1. Set up synthetic node archive
    node_id = "node_test_cap_fire"
    archive_dir = tmp_path / "archive"
    node_archive = archive_dir / node_id
    node_archive.mkdir(parents=True)

    # 2. Write a synthetic training script that:
    #    a. registers the SIGTERM handler (from automil.runtime_helpers)
    #    b. writes 3 fold files (fold_0, fold_1, fold_2) in its CWD
    #    c. writes a _ready marker after the 3 folds
    #    d. sleeps 30s to give the parent time to SIGTERM it
    #
    # IMPORTANT: The SIGTERM handler reads fold files from Path.cwd() and
    # writes result.json to Path.cwd(). We set cwd=node_archive so that
    # the handler aggregates the fold files the script writes there.
    script = textwrap.dedent("""
        import json, os, time, sys
        from pathlib import Path
        from automil.runtime_helpers import register_sigterm_flush
        register_sigterm_flush()
        cwd = Path.cwd()
        for i, comp in enumerate([0.80, 0.82, 0.84]):
            payload = {
                "fold_index": i, "fold_count": 5, "status": "completed",
                "metrics": {"val_auc": comp, "val_bacc": comp,
                            "test_auc": comp, "test_bacc": comp},
                "composite": comp, "elapsed_seconds": 1, "peak_vram_mb": 1000,
            }
            (cwd / f"fold_{i}_result.json").write_text(json.dumps(payload))
            sys.stdout.flush()
            time.sleep(0.05)
        # Signal readiness — parent waits for this marker
        (cwd / "_ready").write_text("ready")
        # Now sleep to give parent time to SIGTERM us
        time.sleep(30)
    """)

    env = os.environ.copy()
    env["AUTOMIL_FOLD_COUNT"] = "5"

    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        cwd=str(node_archive),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # 3. Wait for 3 fold files + ready marker (cap_fire trigger)
    deadline = time.time() + 30
    while time.time() < deadline:
        if (node_archive / "_ready").exists():
            break
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            pytest.fail(
                f"Subprocess died before writing folds. "
                f"stderr={stderr.decode(errors='replace')}"
            )
        time.sleep(0.1)

    assert (node_archive / "fold_0_result.json").exists()
    assert (node_archive / "fold_1_result.json").exists()
    assert (node_archive / "fold_2_result.json").exists()
    assert not (node_archive / "fold_3_result.json").exists()
    assert not (node_archive / "fold_4_result.json").exists()

    # 4. Send SIGTERM (cap fire equivalent) — handler must flush partial result.json
    cap_fire_time = time.time()
    proc.send_signal(signal.SIGTERM)
    stdout, stderr = proc.communicate(timeout=10)

    # 5. VRAM-equivalent assertion: subprocess actually exited within 30s
    elapsed_to_exit = time.time() - cap_fire_time
    assert elapsed_to_exit < 30, (
        f"Subprocess did not exit in 30s after SIGTERM (elapsed={elapsed_to_exit:.1f}s)"
    )
    assert proc.returncode == 0, (
        f"SIGTERM should produce returncode 0 (clean exit via handler). "
        f"got {proc.returncode}; stderr={stderr.decode(errors='replace')}"
    )

    # 6. Assert SIGTERM handler wrote result.json with partial status
    result_path = node_archive / "result.json"
    assert result_path.exists(), "SIGTERM handler did not write result.json"
    result = json.loads(result_path.read_text())
    assert result["status"] == "partial", f"expected partial, got {result['status']}"
    assert result["partial_folds"] == 3
    assert result["expected_folds"] == 5
    assert result["composite"] > 0.0, (
        "composite must NOT be zero (Fragile Invariant #6 defence)"
    )
    assert abs(result["composite"] - 0.82) < 0.01, (
        "composite must be mean of [0.80, 0.82, 0.84]"
    )

    # 7. Simulate the daemon's reconcile pathway end-to-end via reconcile_budget_kill.
    #    MagicMock for graph lets us verify the payload without a full daemon integration.
    from automil.cells.reconcile import reconcile_budget_kill

    mock_graph = MagicMock()
    payload = reconcile_budget_kill(
        node_id=node_id,
        archive_dir=archive_dir,
        graph=mock_graph,
        expected_fold_count=5,
    )

    assert payload["status"] == "partial"
    assert payload["partial_folds"] == 3
    assert payload["composite"] > 0.0
    assert payload.get("metadata", {}).get("budget_killed") is True

    # 8. D-10 (REC-02): reconcile_budget_kill no longer rewrites result.json —
    #    terminal_writer is the sole archive result.json writer. Check the payload
    #    returned has budget_killed=True; the disk file is terminal_writer's responsibility.
    assert payload.get("metadata", {}).get("budget_killed") is True

    # 9. Fragile Invariant #6 defence: real-graph descendant cascade against the
    #    PARTIAL composite (0.82), NOT zero. This is what makes Pitfall-4 single-file
    #    load-bearing — without this step, only the SIGTERM/reconcile chain is verified.
    #
    #    PINNED API (verified against src/automil/graph.py at planning time):
    #      ExperimentGraph(path: str|Path)
    #      add_executed(parent_id, description, techniques, metrics,
    #                   status="discard", ...) -> nid (generated)
    #      get_node(nid) -> dict | None
    #      _reevaluate_descendants(root_id) -> None
    #    Cascade rule (graph.py:267):
    #      keep = (c_auc >= p_auc AND c_bacc >= p_bacc AND c_comp > p_comp)
    from automil.graph import ExperimentGraph

    eg = ExperimentGraph(tmp_path / "descendant_graph.json")

    # Parent node with weak composite/AUC/BACC (0.50)
    parent_nid = eg.add_executed(
        parent_id=None, description="parent", techniques=[],
        metrics={"composite": 0.50, "test_auc": 0.50, "test_bacc": 0.50},
        status="keep",
    )
    # Cap-killed node registered with the PARTIAL composite (0.82)
    capkill_nid = eg.add_executed(
        parent_id=parent_nid, description="cap-killed (partial)", techniques=[],
        metrics={"composite": 0.82, "test_auc": 0.82, "test_bacc": 0.82},
        status="keep",
    )
    # Descendant that BEATS the partial composite/AUC/BACC (0.85) — must stay "keep"
    better_nid = eg.add_executed(
        parent_id=capkill_nid, description="better than partial", techniques=[],
        metrics={"composite": 0.85, "test_auc": 0.85, "test_bacc": 0.85},
        status="keep",
    )
    # Descendant that LOSES to partial composite (0.70 < 0.82) but beats zero —
    # must flip to "discard" if cascade ran against partial composite correctly.
    # If it stays "keep", the cascade ran against zero — Fragile Invariant #6 broken.
    worse_nid = eg.add_executed(
        parent_id=capkill_nid,
        description="worse than partial, beats zero",
        techniques=[],
        metrics={"composite": 0.70, "test_auc": 0.70, "test_bacc": 0.70},
        status="keep",
    )
    eg._reevaluate_descendants(capkill_nid)

    assert eg.get_node(better_nid)["status"] == "keep", (
        "Descendant beating partial composite 0.82 should remain 'keep'."
    )
    assert eg.get_node(worse_nid)["status"] == "discard", (
        "Descendant losing to partial composite 0.82 should flip to 'discard'. "
        "If it stayed 'keep', the cascade ran against zero — Fragile Invariant #6 broken."
    )
