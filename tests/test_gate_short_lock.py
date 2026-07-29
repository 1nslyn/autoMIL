"""CR-2b: the gate must not hold graph state across the held-out evaluations.

`promote` loaded an ``ExperimentGraph``, ran Stage B — backend jobs that can take
many minutes — and then called ``graph.save()``. That is a whole-snapshot write
of state captured *before* the evaluations, so every daemon completion that
landed in the window was silently discarded. The lock genuinely cannot be held
across the evaluations, so the transaction is split: evaluate unlocked, then
re-acquire, RE-READ, and apply only the status transition.

The re-read is the load-bearing half. Re-acquiring the lock but writing back the
pre-evaluation copy of the node would reintroduce the same clobber, just at one
node's granularity instead of the whole file.
"""
from __future__ import annotations

import pytest

from automil.gate.promote import _apply_gate_outcome
from automil.graph import ExperimentGraph


@pytest.fixture
def graph(tmp_path):
    g = ExperimentGraph(tmp_path / "graph.json")
    g.candidate_id = g.add_proposed(parent_id=None, description="cand", techniques=[])
    g.save()
    return g


def test_the_transition_lands_on_disk(graph):
    _apply_gate_outcome(graph, graph.candidate_id, lambda n: n.__setitem__("status", "registered"))
    assert ExperimentGraph(graph.path).get_node(graph.candidate_id)["status"] == "registered"


def test_a_concurrent_write_during_evaluation_survives(graph):
    """THE DEFECT: a completion that lands while Stage B runs must not be lost."""
    from automil.graph import locked_update

    # Simulate the daemon completing an unrelated node mid-evaluation, i.e. after
    # `graph` was loaded and before the gate applies its outcome.
    with locked_update(str(graph.path)) as g:
        g.add_executed(parent_id=None, description="daemon result", techniques=[],
                       status="keep", metrics={"composite": 0.9})

    _apply_gate_outcome(graph, graph.candidate_id, lambda n: n.__setitem__("status", "registered"))

    on_disk = ExperimentGraph(graph.path)
    assert on_disk.get_node(graph.candidate_id)["status"] == "registered"
    assert any(n.get("description") == "daemon result" for n in on_disk.nodes.values()), (
        "the daemon's completion was clobbered by the gate's write"
    )


def test_it_reads_the_node_fresh_rather_than_writing_a_stale_copy(graph):
    """Re-acquiring the lock is not enough on its own."""
    from automil.graph import locked_update

    with locked_update(str(graph.path)) as g:
        g.get_node(graph.candidate_id)["metadata"] = {"touched_by": "daemon"}

    _apply_gate_outcome(graph, graph.candidate_id, lambda n: n.__setitem__("status", "registered"))

    node = ExperimentGraph(graph.path).get_node(graph.candidate_id)
    assert node["status"] == "registered"
    assert node.get("metadata", {}).get("touched_by") == "daemon"


def test_a_vanished_node_is_reported_not_recreated(graph, caplog):
    with caplog.at_level("WARNING"):
        _apply_gate_outcome(graph, "9999", lambda n: n.__setitem__("status", "registered"))
    assert "9999" in " ".join(r.getMessage() for r in caplog.records)
    assert ExperimentGraph(graph.path).get_node("9999") is None


def test_promote_no_longer_calls_graph_save(graph):
    """Guard: a reintroduced whole-snapshot save would silently restore the bug."""
    import inspect

    import automil.gate.promote as promote_mod

    code_lines = [
        ln for ln in inspect.getsource(promote_mod).splitlines()
        if not ln.lstrip().startswith("#")          # the fix's own comment mentions it
    ]
    assert not any("graph.save()" in ln for ln in code_lines), (
        "promote calls graph.save() again — that writes state captured before "
        "the held-out evaluations and discards concurrent daemon completions"
    )
