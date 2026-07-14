"""val-firewall B2: terminal_writer._seal_node_archive quarantines the whole
per-fold test subtree into archive/<node>/certify/, leaving the agent-visible
node archive test-free."""
from __future__ import annotations

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
