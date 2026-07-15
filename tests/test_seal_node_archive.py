"""val-firewall B2: terminal_writer._seal_node_archive quarantines the whole
per-fold test subtree into archive/<node>/certify/, leaving the agent-visible
node archive test-free."""
from __future__ import annotations

import json

from automil.terminal_writer import _seal_node_archive


def test_seal_moves_fold_and_results_into_certify(tmp_path):
    node = tmp_path
    # The agent-visible node archive after a run: test-bearing per-fold detritus
    # sits directly under the node dir alongside val-only result.json / run.log.
    (node / "result.json").write_text('{"composite": 0.84}')        # val-only, stays
    (node / "run.log").write_text("Val error: 0.2")                  # stays
    (node / "fold_0_result.json").write_text('{"held_out": {"test_auc": 0.8}}')
    (node / "fold_1_result.json").write_text('{"held_out": {"test_auc": 0.7}}')
    res = node / "results" / "fold_0"
    res.mkdir(parents=True)
    (res / "predictions.csv").write_text("y_true,y_prob\n1,0.9\n")
    (node / "results" / "summary.json").write_text('{"test": {"auc": 0.8}}')

    _seal_node_archive(node, {"held_out": {"test_auc": 0.8}, "summary": {"test": 1}})

    # Everything test-bearing is now under certify/ and gone from the node dir.
    assert (node / "certify" / "certify.json").exists()
    assert (node / "certify" / "fold_0_result.json").exists()
    assert (node / "certify" / "fold_1_result.json").exists()
    assert (node / "certify" / "results" / "summary.json").exists()
    assert (node / "certify" / "results" / "fold_0" / "predictions.csv").exists()
    assert not (node / "fold_0_result.json").exists()
    assert not (node / "fold_1_result.json").exists()
    assert not (node / "results").exists()
    # Val-only artifacts remain agent-visible.
    assert (node / "result.json").exists()
    assert (node / "run.log").exists()


def test_seal_handles_no_test_gracefully(tmp_path):
    # No fold files / results, empty sealed dict -> just creates certify/, no crash.
    _seal_node_archive(tmp_path, {})
    assert (tmp_path / "certify").is_dir()
    assert not (tmp_path / "certify" / "certify.json").exists()  # nothing to seal


def test_seal_backstop_leaves_root_test_free_when_certify_already_born_sealed(tmp_path):
    """F1 regression: under born-sealing, certify/results already exists (the CLAM
    detail tree is born there). If a bypassing writer ALSO leaks results/ + a fold
    file to the agent-visible node-archive ROOT, the backstop must still leave the
    root test-free — Path.replace onto the non-empty certify/results raises
    ENOTEMPTY, so the naive move silently failed and the leak survived."""
    node = tmp_path
    # Born-sealed reality: certify/ already holds the real per-fold + results tree.
    sealed = node / "certify"
    (sealed / "results").mkdir(parents=True)
    (sealed / "results" / "summary.json").write_text('{"test": {"auc": 0.9}}')   # born-sealed
    (sealed / "fold_0_result.json").write_text('{"held_out": {"test_auc": 0.9}}')  # born-sealed
    # A bypassing/legacy writer leaks test-bearing artifacts to the AGENT-VISIBLE root.
    (node / "results").mkdir()
    (node / "results" / "summary.json").write_text('{"test": {"auc": 0.8}}')       # LEAK
    (node / "fold_0_result.json").write_text('{"held_out": {"test_auc": 0.8}}')     # LEAK

    _seal_node_archive(node, {"held_out": {"test_auc": 0.9}})

    # The agent-visible root is left test-free — no stray survives.
    assert not (node / "results").exists(), "stray results/ leaked at the agent-visible root"
    assert not list(node.glob("fold_*_result.json")), "stray fold file leaked at the root"
    # The born-sealed certify/ copies are preserved (not clobbered by the strays).
    assert json.loads((sealed / "results" / "summary.json").read_text())["test"]["auc"] == 0.9
    assert json.loads((sealed / "fold_0_result.json").read_text())["held_out"]["test_auc"] == 0.9
