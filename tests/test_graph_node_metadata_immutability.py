"""``graph.merged_metadata`` — copy-on-write node metadata updates (L-8a).

Several call sites read a node via ``get_node()`` and then wrote into its
``metadata`` sub-dict IN PLACE (``gnode.setdefault("metadata", {}).update(...)``
or ``gnode.setdefault("metadata", {})[k] = v``). That is reachable from two
writers: ``gate/evaluate.py`` creates a gate-eval child node via a SHALLOW
copy of a node dict (``dict(node)``), which leaves the child's ``metadata``
key pointing at the SAME dict object as its source. An in-place mutation
through either alias silently corrupts the other -- a plain dict is not
copy-on-write.

``merged_metadata`` is the shared fix: it always returns a NEW dict, so a
call site does ``gnode["metadata"] = merged_metadata(gnode, {...})`` instead
of mutating whatever object happened to be stored there.
"""
from __future__ import annotations


def test_merged_metadata_does_not_mutate_the_input_node():
    from automil.graph import merged_metadata

    node = {"id": "node_0001", "metadata": {"a": 1}}
    snapshot_metadata = dict(node["metadata"])

    result = merged_metadata(node, {"b": 2})

    assert result == {"a": 1, "b": 2}
    assert node["metadata"] == snapshot_metadata, (
        "merged_metadata must not mutate the node's existing metadata dict"
    )
    assert result is not node["metadata"], "must return a fresh dict, not the same object"


def test_merged_metadata_breaks_aliasing_between_two_nodes_sharing_one_metadata_dict():
    """The core L-8a reproduction: two node dicts share ONE metadata object,
    mirroring what gate/evaluate.py's ``dict(node)`` shallow copy produces.
    Merging into one node's metadata must leave the other's completely alone.
    """
    from automil.graph import merged_metadata

    shared_metadata = {"gate_eval": True, "cell_id": "abc12345"}
    node_a = {"id": "node_a", "metadata": shared_metadata}
    node_b = {"id": "node_b", "metadata": shared_metadata}  # aliased, like a shallow dict(node) copy

    node_a["metadata"] = merged_metadata(node_a, {"cancelled_at": "2026-07-29T00:00:00"})

    assert node_a["metadata"] == {
        "gate_eval": True, "cell_id": "abc12345", "cancelled_at": "2026-07-29T00:00:00",
    }
    assert node_b["metadata"] == {"gate_eval": True, "cell_id": "abc12345"}, (
        "updating node_a's metadata must not leak into node_b's aliased dict"
    )
    assert shared_metadata == {"gate_eval": True, "cell_id": "abc12345"}, (
        "the originally-shared dict object must be untouched"
    )


def test_merged_metadata_with_no_existing_metadata_key():
    from automil.graph import merged_metadata

    node = {"id": "node_0002"}  # no "metadata" key at all (e.g. a freshly proposed node)

    result = merged_metadata(node, {"mil_model": "clam_mb"})

    assert result == {"mil_model": "clam_mb"}
    assert "metadata" not in node, "merged_metadata must not mutate the input node itself"


def test_merged_metadata_tolerates_a_none_node():
    from automil.graph import merged_metadata

    assert merged_metadata(None, {"k": "v"}) == {"k": "v"}


def test_merged_metadata_tolerates_a_non_dict_metadata_value():
    """Defensive: a corrupt/legacy node with metadata set to a non-dict must not raise."""
    from automil.graph import merged_metadata

    node = {"id": "node_0003", "metadata": "not-a-dict"}

    result = merged_metadata(node, {"k": "v"})

    assert result == {"k": "v"}
