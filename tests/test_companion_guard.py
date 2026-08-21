"""Companion non-inferiority guard: a veto without a vote.

Selection is single-metric because a threshold-quantized companion (val_bacc on
a few-dozen-slide validation split) cannot vote without injecting lattice noise
at the decision scale. The guard restores the companion's ability to REJECT a
child without giving it a say in the argmax, which is what makes it noise-free:
it never promotes anything, so it never steers the search onto its own jitter.
"""
from __future__ import annotations

import json

import pytest
import yaml

from automil.graph import (ExperimentGraph, _config_scoring_guard,
                           _guard_declaration, guard_basis, keep_or_discard)

#: One quantization step of balanced accuracy on the campaign's binary cells:
#: 1 / (3 discovery folds x 2 classes x 17 minority slides).
QUANTUM = 1.0 / (3 * 2 * 17)

META = {
    "scoring": {
        "formula": "val_auc",
        "accept_margin": 0.0,
        "se_multiplier": 0.0,
        "guard": {"metric": "val_bacc", "margin": QUANTUM},
    }
}


def _node(primary: float, bacc: float | None = None, **extra) -> dict:
    metrics = {"val_auc": primary}
    if bacc is not None:
        metrics["val_bacc"] = bacc
    return {"primary_value": primary, "metrics": metrics, **extra}


class TestGuardSemantics:
    def test_undeclared_guard_leaves_the_gate_untouched(self):
        """No declaration → a companion collapse cannot block a primary win."""
        meta = {"scoring": {"formula": "val_auc", "accept_margin": 0.0,
                            "se_multiplier": 0.0}}
        parent = _node(0.70, bacc=0.70)
        child = _node(0.75, bacc=0.40)
        assert keep_or_discard(meta, parent, child) == "keep"
        assert guard_basis(meta, parent, child) == ("none", None, None)

    def test_companion_gain_keeps(self):
        assert keep_or_discard(META, _node(0.70, 0.70), _node(0.75, 0.72)) == "keep"

    def test_drop_of_exactly_one_quantum_keeps(self):
        """A drop the size of a single slide flip is not evidence of harm.

        `margin` IS the largest change one validation sample can make, so a
        drop of exactly that much is arithmetically explainable by one
        borderline slide changing side. Rejection must need strictly more.
        """
        parent, child = _node(0.70, 0.70), _node(0.75, 0.70 - QUANTUM)
        assert guard_basis(META, parent, child)[0] == "pass"
        assert keep_or_discard(META, parent, child) == "keep"

    def test_drop_past_one_quantum_discards_a_primary_winner(self):
        """The whole point: a big AUC gain does not buy a bACC collapse."""
        parent = _node(0.70, 0.70)
        child = _node(0.85, 0.70 - 2 * QUANTUM)   # +0.15 val_auc, -2 slides
        verdict, delta, metric = guard_basis(META, parent, child)
        assert verdict == "fail"
        assert delta == pytest.approx(-2 * QUANTUM)
        assert keep_or_discard(META, parent, child) == "discard"

    def test_a_drop_of_exactly_the_declared_margin_survives_float_subtraction(self):
        """The promise has to hold on the RECORDED decimals, not in theory.

        Both the margin and the metrics are decimals, and binary floats put a
        drop of exactly the margin a few ulps on the wrong side of it
        (0.5408 - 0.5507 == -0.00990000000000002). A bare `delta < -margin`
        therefore still rejected 122 of the 182 exact-margin one-slide drops
        on the campaign's own tcga_luad/kras lattice — two thirds of the
        cases the grid-aligned margin was introduced to rescue.
        """
        meta = {"scoring": {"formula": "val_auc", "accept_margin": 0.0,
                            "se_multiplier": 0.0,
                            "guard": {"metric": "val_bacc", "margin": 0.0099}}}
        parent, child = _node(0.70, 0.5507), _node(0.80, 0.5408)
        assert (child["metrics"]["val_bacc"] - parent["metrics"]["val_bacc"]) < -0.0099
        assert guard_basis(meta, parent, child)[0] == "pass"
        assert keep_or_discard(meta, parent, child) == "keep"
        # ...and the slack is far too small to admit a genuinely larger drop.
        assert keep_or_discard(meta, parent, _node(0.80, 0.5407)) == "discard"

    def test_guard_never_rescues_a_primary_loser(self):
        """Veto without a vote: the companion can only ever subtract."""
        parent, child = _node(0.70, 0.70), _node(0.65, 0.99)
        assert guard_basis(META, parent, child)[0] == "pass"
        assert keep_or_discard(META, parent, child) == "discard"

    def test_root_has_no_guard(self):
        assert keep_or_discard(META, None, _node(0.75, 0.10)) == "keep"
        assert guard_basis(META, None, _node(0.75, 0.10)) == ("none", None, None)


class TestGuardFailsClosed:
    def test_child_without_the_metric_is_discarded(self):
        """`metrics` is agent-editable: dropping the key must not be an escape.

        If a missing companion opened the guard, deleting it from result.json
        would be the dominant strategy for every candidate that hurt it.
        """
        parent, child = _node(0.70, 0.70), _node(0.85)   # child reports no bacc
        assert guard_basis(META, parent, child) == ("fail", None, "val_bacc")
        assert keep_or_discard(META, parent, child) == "discard"

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), "0.7", True, None])
    def test_unusable_child_value_is_discarded(self, bad):
        child = {"primary_value": 0.85, "metrics": {"val_auc": 0.85, "val_bacc": bad}}
        assert keep_or_discard(META, _node(0.70, 0.70), child) == "discard"

    def test_parent_without_the_metric_opens_the_guard(self):
        """Nothing to be non-inferior TO — a legacy or pre-guard incumbent.

        The child has already been required to carry the metric (checked
        first), so this exempts ONE comparison, never a lineage.
        """
        parent, child = _node(0.70), _node(0.75, 0.10)
        assert guard_basis(META, parent, child) == ("none", None, "val_bacc")
        assert keep_or_discard(META, parent, child) == "keep"

    def test_the_parent_exemption_is_not_hereditary(self):
        """The escape the child-first ordering closes.

        Checking the parent FIRST returned "none" before the child was ever
        examined, so a metric-less child under a metric-less parent was kept —
        and became a metric-less parent itself. A trainer that simply never
        wrote the key then disabled the guard for its entire lineage, which is
        exactly the dominant strategy the child-side rule exists to prevent.
        """
        parent, child = _node(0.70), _node(0.80)     # neither reports val_bacc
        assert guard_basis(META, parent, child) == ("fail", None, "val_bacc")
        assert keep_or_discard(META, parent, child) == "discard"
        # ...so omitting the key cannot propagate down a lineage either: the
        # grandchild is judged on its OWN evidence, not on what its parent
        # happened to be missing.
        assert keep_or_discard(META, child, _node(0.85)) == "discard"

    def test_companion_read_from_fold_evidence_when_aggregate_lacks_it(self):
        """The campaign's discovery BASELINE ROOT topology.

        That root is created with the framework scalars only — its `metrics`
        block holds no companion — and it is the dominant parent of the whole
        cell. Reading only `metrics` left the guard open for every
        first-generation candidate: exactly the comparisons it exists to make,
        including the one that first becomes best_node.
        """
        root = {
            "primary_value": 0.70,
            "metrics": {"primary_value": 0.70},          # no val_bacc here
            "metadata": {"validation_folds": [
                {"fold_index": i, "primary_value": 0.70,
                 "metrics": {"val_auc": 0.70, "val_bacc": 0.70}}
                for i in range(3)
            ]},
        }
        verdict, delta, _ = guard_basis(META, root, _node(0.80, 0.70 - 3 * QUANTUM))
        assert verdict == "fail"
        assert delta == pytest.approx(-3 * QUANTUM)
        assert keep_or_discard(META, root, _node(0.80, 0.40)) == "discard"
        assert keep_or_discard(META, root, _node(0.80, 0.72)) == "keep"

    def test_partial_fold_evidence_is_not_averaged(self):
        """A mean over a subset is a different statistic from the aggregate."""
        root = {
            "primary_value": 0.70,
            "metrics": {"primary_value": 0.70},
            "metadata": {"validation_folds": [
                {"fold_index": 0, "metrics": {"val_auc": 0.70, "val_bacc": 0.70}},
                {"fold_index": 1, "metrics": {"val_auc": 0.70}},   # lost companion
            ]},
        }
        assert guard_basis(META, root, _node(0.80, 0.40)) == ("none", None, "val_bacc")

    @pytest.mark.parametrize("bad", [
        {"metric": "val_bacc"},                    # margin missing
        {"margin": 0.01},                          # metric missing
        {"metric": "", "margin": 0.01},            # empty metric name
        {"metric": "val_bacc", "margin": -0.01},   # negative margin inverts it
        {"metric": "val_bacc", "margin": "0.01"},  # string margin
        {"metric": "val_bacc", "margin": True},    # bool is an int subclass
        {"metric": "val_bacc", "margin": float("nan")},
        "val_bacc",                                # not a mapping
    ])
    def test_malformed_declaration_fails_closed(self, bad):
        """One typo must not silently switch a declared protection off."""
        meta = {"scoring": {"formula": "val_auc", "accept_margin": 0.0,
                            "se_multiplier": 0.0, "guard": bad}}
        with pytest.raises(ValueError):
            _guard_declaration(meta)
        assert guard_basis(meta, _node(0.70, 0.70), _node(0.99, 0.99)) == ("fail", None, None)
        assert keep_or_discard(meta, _node(0.70, 0.70), _node(0.99, 0.99)) == "discard"


class TestGuardDeclarationIsFrozen:
    def _write_config(self, tmp_path, scoring: dict) -> None:
        (tmp_path / "config.yaml").write_text(yaml.safe_dump({"scoring": scoring}))

    def test_seeded_from_config_and_frozen_against_later_edits(self, tmp_path):
        """Same freeze semantics as δ: a mid-campaign edit cannot widen it."""
        self._write_config(tmp_path, {
            "formula": "val_auc",
            "guard": {"metric": "val_bacc", "margin": QUANTUM,
                      "basis": "1/(3 folds x 2 classes x 17 minority slides)"},
        })
        graph_path = tmp_path / "graph.json"
        ExperimentGraph(graph_path).save()

        frozen = json.loads(graph_path.read_text())["meta"]["scoring"]["guard"]
        assert frozen["metric"] == "val_bacc"
        assert frozen["margin"] == pytest.approx(QUANTUM)
        # Provenance travels with the number so the frozen graph records WHY.
        assert "17" in frozen["basis"]

        self._write_config(tmp_path, {
            "formula": "val_auc",
            "guard": {"metric": "val_bacc", "margin": 0.5},   # a much looser edit
        })
        assert ExperimentGraph(graph_path).meta["scoring"]["guard"]["margin"] == \
            pytest.approx(QUANTUM)

    def test_undeclared_guard_writes_no_key(self, tmp_path):
        """A project without a guard keeps a graph.json free of a null."""
        self._write_config(tmp_path, {"formula": "val_auc"})
        graph_path = tmp_path / "graph.json"
        ExperimentGraph(graph_path).save()
        assert "guard" not in json.loads(graph_path.read_text())["meta"]["scoring"]

    def test_malformed_config_declaration_raises_at_load(self, tmp_path):
        """Fail at config load, not by degrading into 'no guard' at gate time."""
        self._write_config(tmp_path, {"guard": {"metric": "val_bacc"}})
        with pytest.raises(ValueError, match="scoring.guard"):
            _config_scoring_guard(tmp_path / "graph.json")


class TestGuardAcrossIngestPaths:
    """Every keep/discard mouth must hand the guard the CHILD's own metrics."""

    def test_descendant_reevaluation_applies_the_guard(self, tmp_path):
        graph = ExperimentGraph(tmp_path / "graph.json")
        graph.meta["scoring"].update(META["scoring"])
        root = graph.add_executed(None, "baseline", [],
                                  {"primary_value": 0.70, "val_auc": 0.70,
                                   "val_bacc": 0.70}, status="keep")
        child = graph.add_executed(root, "companion collapse", [],
                                   {"primary_value": 0.85, "val_auc": 0.85,
                                    "val_bacc": 0.70 - 3 * QUANTUM}, status="keep")
        graph._reevaluate_descendants(root)
        assert graph.get_node(child)["status"] == "discard"

    def test_terminal_writer_gates_on_the_new_metrics_not_the_stale_ones(
            self, tmp_path):
        """THE ORDERING DEFECT: `metrics` used to be assigned AFTER the accept.

        A re-ingested node still carries its previous run's metrics when the
        gate runs, so the guard would have read the OLD companion value — a
        node whose balanced accuracy just collapsed would pass on the strength
        of the value it used to have.
        """
        from automil.graph import ExperimentGraph
        from automil.terminal_writer import write_terminal_state

        graph = ExperimentGraph(path=str(tmp_path / "graph.json"))
        graph.meta["scoring"].update(META["scoring"])
        parent = graph.add_executed(
            None, "parent", [], {"primary_value": 0.70, "val_auc": 0.70,
                                 "val_bacc": 0.70}, status="keep")
        child = graph.add_proposed(parent, "child", [])
        # A previous ingest left a healthy companion value on the node.
        graph.nodes[child].update({"status": "running",
                                   "metrics": {"val_auc": 0.90, "val_bacc": 0.90}})
        graph.save()

        (tmp_path / "completed").mkdir()
        (tmp_path / "archive" / child).mkdir(parents=True)
        write_terminal_state(
            node_id=child,
            result={"status": "completed", "primary_value": 0.85,
                    "metrics": {"val_auc": 0.85,
                                "val_bacc": 0.70 - 3 * QUANTUM}},
            graph=graph,
            completed_dir=tmp_path / "completed",
            archive_dir=tmp_path / "archive" / child,
            results_tsv_writer=lambda *a, **k: None,
            spec={"description": "d", "graph_metadata": {}},
            elapsed_s=1.0,
            gpu_id=0,
        )
        reloaded = ExperimentGraph(path=str(tmp_path / "graph.json"))
        assert reloaded.get_node(child)["status"] == "discard"

    def test_reconcile_completed_path_applies_the_guard(self, tmp_path):
        from automil.graph import ExperimentGraph

        for sub in ("queue", "running", "completed", "archive"):
            (tmp_path / sub).mkdir()
        graph = ExperimentGraph(path=str(tmp_path / "graph.json"))
        graph.meta["scoring"].update(META["scoring"])
        parent = graph.add_executed(
            None, "parent", [], {"primary_value": 0.70, "val_auc": 0.70,
                                 "val_bacc": 0.70}, status="keep")
        child = graph.add_proposed(parent, "child", [])
        graph.save()
        (tmp_path / "completed" / f"{child}.json").write_text(json.dumps({
            "id": child, "status": "completed", "primary_value": 0.85,
            "metrics": {"val_auc": 0.85, "val_bacc": 0.70 - 3 * QUANTUM},
            "graph_metadata": {"parent_id": parent},
        }))

        graph.reconcile(str(tmp_path / "queue"), str(tmp_path / "running"),
                        str(tmp_path / "completed"), str(tmp_path / "archive"))
        assert graph.get_node(child)["status"] == "discard"

    def test_archive_recovery_path_applies_the_guard(self, tmp_path):
        from automil.graph import ExperimentGraph

        for sub in ("queue", "running", "completed", "archive"):
            (tmp_path / sub).mkdir()
        graph = ExperimentGraph(path=str(tmp_path / "graph.json"))
        graph.meta["scoring"].update(META["scoring"])
        parent = graph.add_executed(
            None, "parent", [], {"primary_value": 0.70, "val_auc": 0.70,
                                 "val_bacc": 0.70}, status="keep")
        graph.save()
        recovered = tmp_path / "archive" / "node_0099"
        recovered.mkdir()
        (recovered / "spec.json").write_text(json.dumps({
            "description": "recovered", "graph_metadata": {"parent_id": parent},
        }))
        (recovered / "result.json").write_text(json.dumps({
            "status": "completed", "primary_value": 0.85,
            "metrics": {"val_auc": 0.85, "val_bacc": 0.70 - 3 * QUANTUM},
        }))

        graph.reconcile(str(tmp_path / "queue"), str(tmp_path / "running"),
                        str(tmp_path / "completed"), str(tmp_path / "archive"))
        assert graph.get_node("node_0099")["status"] == "discard"


class TestCheckCatchesGuardConfigErrors:
    """`automil check` runs before a graph exists — the earliest place a bad
    declaration can be caught, and the only place one specific mistake is
    caught at all: a guard on a metric the trainer never emits fails EVERY
    child closed, and the operator would otherwise see 30/30 discards with
    nothing pointing at the cause."""

    def _project(self, tmp_path, monkeypatch, scoring: dict, track: list[str],
                 nodes: dict | None = None, frozen_scoring: dict | None = None):
        import subprocess

        from click.testing import CliRunner

        from automil.cli import main

        subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
        monkeypatch.chdir(tmp_path)
        CliRunner().invoke(main, ["init"])
        adir = tmp_path / "automil"
        cfg = yaml.safe_load((adir / "config.yaml").read_text())
        cfg.setdefault("scoring", {}).update(scoring)
        cfg.setdefault("metrics", {})["track"] = track
        (adir / "config.yaml").write_text(yaml.safe_dump(cfg))
        if nodes is not None or frozen_scoring is not None:
            (adir / "graph.json").write_text(json.dumps({
                "schema_version": 3,
                "meta": {"scoring": frozen_scoring or {"formula": "val_auc"}},
                "nodes": nodes or {},
            }))
        return CliRunner().invoke(main, ["check"])

    def test_guard_on_an_untracked_metric_is_reported(self, tmp_path, monkeypatch):
        result = self._project(
            tmp_path, monkeypatch,
            {"formula": "val_auc", "guard": {"metric": "val_bacc", "margin": 0.01}},
            track=["val_auc"],
        )
        assert "is not in metrics.track" in result.output

    def test_malformed_guard_is_reported(self, tmp_path, monkeypatch):
        result = self._project(
            tmp_path, monkeypatch,
            {"formula": "val_auc", "guard": {"metric": "val_bacc"}},
            track=["val_auc", "val_bacc"],
        )
        assert "scoring.guard is declared but unusable" in result.output

    def test_adding_a_guard_to_a_graph_with_history_is_flagged(self, tmp_path, monkeypatch):
        """The one direction the freeze does not cover — and it rewrites history.

        Seeding a new guard makes the next re-evaluation discard every node
        that never recorded the metric. That is correct under the declaration
        (a node that cannot be shown non-inferior fails closed) but it is
        silent, so `check` has to say it before anything runs.
        """
        result = self._project(
            tmp_path, monkeypatch,
            {"formula": "val_auc", "guard": {"metric": "val_bacc", "margin": 0.0099}},
            track=["val_auc", "val_bacc"],
            nodes={"node_0001": {"id": "node_0001", "type": "executed",
                                 "status": "keep", "primary_value": 0.7,
                                 "metrics": {"val_auc": 0.7}}},
        )
        assert "scoring.guard is new to a graph that already has" in result.output

    def test_a_held_out_metric_cannot_be_the_guard(self, tmp_path, monkeypatch):
        """The guard reads the AGENT-FACING validation block.

        A held-out key there is a val-firewall violation that fails the node
        closed; in `held_out` the guard would never see it. Either way the
        declaration is unusable, so it must not survive preflight.
        """
        result = self._project(
            tmp_path, monkeypatch,
            {"formula": "val_auc", "guard": {"metric": "test_auc", "margin": 0.01}},
            track=["val_auc", "test_auc"],
        )
        assert "is held-out-named" in result.output

    def test_a_malformed_FROZEN_guard_is_an_issue_not_a_warning(
            self, tmp_path, monkeypatch):
        """The frozen declaration is the one every keep/discard consults.

        Config-side validation says nothing about it, and a graph frozen with
        a broken guard discards every non-root child at run time — so a
        preflight that only reported drift would be describing a graph that
        cannot gate at all.
        """
        result = self._project(
            tmp_path, monkeypatch,
            {"formula": "val_auc", "guard": {"metric": "val_bacc", "margin": 0.0099}},
            track=["val_auc", "val_bacc"],
            frozen_scoring={"formula": "val_auc", "guard": {"metric": "val_bacc"}},
        )
        assert "froze an unusable scoring.guard" in result.output

    def test_a_well_formed_guard_is_silent(self, tmp_path, monkeypatch):
        result = self._project(
            tmp_path, monkeypatch,
            {"formula": "val_auc", "guard": {"metric": "val_bacc", "margin": 0.0098}},
            track=["val_auc", "val_bacc"],
        )
        assert "scoring.guard" not in result.output
