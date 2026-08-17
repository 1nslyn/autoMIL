import json
import os
import tempfile
import pytest
from pathlib import Path

from automil.graph import ExperimentGraph


class TestGraphBasics:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.graph_path = os.path.join(self.tmpdir, "graph.json")

    def test_create_empty_graph(self):
        g = ExperimentGraph(path=self.graph_path)
        assert g.meta["total_executed"] == 0
        assert g.meta["total_proposed"] == 0
        assert g.meta["next_id"] == 1

    def test_add_root_executed(self):
        g = ExperimentGraph(path=self.graph_path)
        nid = g.add_executed(
            parent_id=None, description="baseline", techniques=[],
            metrics={"primary_value": 0.814, "test_auc": 0.836, "test_bacc": 0.792,
                     "val_auc": 0.869, "val_bacc": 0.810,
                     "vram_gb": 0.4, "elapsed_min": 56.6, "gpu": 0},
            status="keep",
        )
        assert nid == "node_0001"
        node = g.get_node(nid)
        assert node["type"] == "executed"
        assert node["status"] == "keep"
        assert node["primary_value"] == 0.814
        assert node["parent_id"] is None
        assert g.meta["total_executed"] == 1
        assert g.meta["best_primary_value"] == 0.814

    def test_add_proposed(self):
        g = ExperimentGraph(path=self.graph_path)
        root = g.add_executed(
            parent_id=None, description="baseline", techniques=[],
            metrics={"primary_value": 0.814, "test_auc": 0.836, "test_bacc": 0.792,
                     "val_auc": 0.869, "val_bacc": 0.810,
                     "vram_gb": 0.4, "elapsed_min": 56.6, "gpu": 0},
            status="keep",
        )
        pid = g.add_proposed(
            parent_id=root, description="try focal loss",
            techniques=["focal_g1"], rationale="may help with class imbalance",
            tier=1,
        )
        assert pid == "node_0002"
        node = g.get_node(pid)
        assert node["type"] == "proposed"
        assert node["status"] == "pending"
        assert node["parent_id"] == root
        assert g.meta["total_proposed"] == 1

    def test_get_nonexistent_node_returns_none(self):
        g = ExperimentGraph(path=self.graph_path)
        assert g.get_node("node_9999") is None


class TestNodeLifecycle:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.graph_path = os.path.join(self.tmpdir, "graph.json")
        self.g = ExperimentGraph(path=self.graph_path)
        self.root = self.g.add_executed(
            parent_id=None, description="baseline", techniques=[],
            metrics={"primary_value": 0.814, "test_auc": 0.836, "test_bacc": 0.792,
                     "val_auc": 0.869, "val_bacc": 0.810,
                     "vram_gb": 0.4, "elapsed_min": 56.6, "gpu": 0},
            status="keep",
        )

    def test_proposed_to_running_to_executed(self):
        pid = self.g.add_proposed(self.root, "try focal", ["focal_g1"],
                                  rationale="test")
        assert self.g.get_node(pid)["status"] == "pending"
        self.g.mark_running(pid)
        assert self.g.get_node(pid)["status"] == "running"
        self.g.promote(pid, {
            "primary_value": 0.832, "test_auc": 0.867, "test_bacc": 0.797,
            "val_auc": 0.866, "val_bacc": 0.768,
            "vram_gb": 0.4, "elapsed_min": 69.5, "gpu": 1,
            "status": "keep",
        })
        node = self.g.get_node(pid)
        assert node["type"] == "executed"
        assert node["status"] == "keep"
        assert node["primary_value"] == 0.832
        assert abs(node["parent_delta"] - (0.832 - 0.814)) < 1e-6

    def test_mark_failed(self):
        pid = self.g.add_proposed(self.root, "try oom thing", ["big_model"],
                                  rationale="test")
        self.g.mark_running(pid)
        self.g.mark_failed(pid, "oom", "CUDA out of memory")
        node = self.g.get_node(pid)
        assert node["type"] == "executed"
        assert node["status"] == "oom"
        assert node["error"] == "CUDA out of memory"

    def test_cancel(self):
        pid = self.g.add_proposed(self.root, "nevermind", ["x"], rationale="test")
        self.g.cancel(pid)
        assert self.g.get_node(pid)["status"] == "cancelled"

    def test_mark_running_wrong_state_returns_false(self):
        pid = self.g.add_proposed(self.root, "test", ["x"], rationale="test")
        assert self.g.mark_running(pid) is True
        assert self.g.mark_running(pid) is False
        assert self.g.get_node(pid)["status"] == "running"

    def test_best_node_updates_on_promote(self):
        pid = self.g.add_proposed(self.root, "better", ["x"], rationale="test")
        self.g.mark_running(pid)
        self.g.promote(pid, {"primary_value": 0.900, "status": "keep",
                             "test_auc": 0.9, "test_bacc": 0.9,
                             "val_auc": 0.9, "val_bacc": 0.9,
                             "vram_gb": 0.4, "elapsed_min": 60, "gpu": 0})
        assert self.g.meta["best_primary_value"] == 0.900
        assert self.g.meta["best_node_id"] == pid


class TestScoring:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.graph_path = os.path.join(self.tmpdir, "graph.json")
        self.g = ExperimentGraph(path=self.graph_path)
        self.root = self.g.add_executed(
            parent_id=None, description="baseline", techniques=[],
            metrics={"primary_value": 0.814, "test_auc": 0.836, "test_bacc": 0.792,
                     "val_auc": 0.869, "val_bacc": 0.810,
                     "vram_gb": 0.4, "elapsed_min": 56.6, "gpu": 0},
            status="keep",
        )

    def test_recalculate_scores_executed(self):
        self.g.add_executed(
            parent_id=self.root, description="second", techniques=["x"],
            metrics={"primary_value": 0.790, "test_auc": 0.80, "test_bacc": 0.78,
                     "val_auc": 0.80, "val_bacc": 0.78,
                     "vram_gb": 0.4, "elapsed_min": 60, "gpu": 0},
            status="discard",
        )
        self.g.recalculate_scores()
        node = self.g.get_node(self.root)
        assert node["potential"] > node["primary_value"]

    def test_recalculate_scores_proposed(self):
        pid = self.g.add_proposed(self.root, "try focal", ["focal_g1"],
                                  rationale="test")
        self.g.recalculate_scores()
        node = self.g.get_node(pid)
        assert node["potential"] > 0
        assert node["potential"] >= self.g.get_node(self.root)["primary_value"]

    def test_rank_proposals_diversity(self):
        self.g.add_proposed(self.root, "a1", ["x"], rationale="test")
        self.g.add_proposed(self.root, "a2", ["y"], rationale="test")
        self.g.add_proposed(self.root, "a3", ["z"], rationale="test")

        child = self.g.add_executed(
            parent_id=self.root, description="child", techniques=["a"],
            metrics={"primary_value": 0.820, "test_auc": 0.84, "test_bacc": 0.80,
                     "val_auc": 0.86, "val_bacc": 0.78,
                     "vram_gb": 0.4, "elapsed_min": 60, "gpu": 0},
            status="keep",
        )
        self.g.add_proposed(child, "b1", ["w"], rationale="test")

        self.g.recalculate_scores()
        ranked = self.g.rank_proposals(n=4, max_per_branch=2)
        parent_ids = [p["parent_id"] for p in ranked]

        assert self.root in parent_ids, "root branch missing from ranked proposals"
        assert child in parent_ids, "child branch missing from ranked proposals"
        assert parent_ids.count(self.root) <= 2
        assert parent_ids.count(child) <= 2

    def test_technique_stats_updated(self):
        self.g.add_executed(
            parent_id=self.root, description="focal", techniques=["focal_g1"],
            metrics={"primary_value": 0.832, "test_auc": 0.867, "test_bacc": 0.797,
                     "val_auc": 0.866, "val_bacc": 0.768,
                     "vram_gb": 0.4, "elapsed_min": 69, "gpu": 0},
            status="keep",
        )
        stats = self.g.technique_stats("focal_g1")
        assert stats["times_tried"] == 1
        assert stats["best_parent_delta"] == pytest.approx(0.832 - 0.814, abs=1e-6)


class TestPersistence:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.graph_path = os.path.join(self.tmpdir, "graph.json")

    def test_save_and_load(self):
        g = ExperimentGraph(path=self.graph_path)
        g.add_executed(
            parent_id=None, description="baseline", techniques=[],
            metrics={"primary_value": 0.814, "test_auc": 0.836, "test_bacc": 0.792,
                     "val_auc": 0.869, "val_bacc": 0.810,
                     "vram_gb": 0.4, "elapsed_min": 56.6, "gpu": 0},
            status="keep",
        )
        g.save()
        g2 = ExperimentGraph.load(self.graph_path)
        assert g2.meta["total_executed"] == 1
        assert g2.get_node("node_0001")["primary_value"] == 0.814

    def test_config_hash(self):
        script = "# comment\nx = 1\ny = 2\n"
        script2 = "# different comment\nx = 1\ny = 2\n"
        h1 = ExperimentGraph.compute_config_hash(script)
        h2 = ExperimentGraph.compute_config_hash(script2)
        assert h1 == h2

        script3 = "x = 1\ny = 3\n"
        h3 = ExperimentGraph.compute_config_hash(script3)
        assert h1 != h3

    def test_config_hash_preserves_hash_in_strings(self):
        script_a = 'x = "hello # world"\n'
        script_b = 'x = "hello # world"\n# actual comment\n'
        h_a = ExperimentGraph.compute_config_hash(script_a)
        h_b = ExperimentGraph.compute_config_hash(script_b)
        assert h_a == h_b

    def test_has_config(self):
        g = ExperimentGraph(path=self.graph_path)
        g.add_executed(
            parent_id=None, description="baseline", techniques=[],
            metrics={"primary_value": 0.814, "test_auc": 0.836, "test_bacc": 0.792,
                     "val_auc": 0.869, "val_bacc": 0.810,
                     "vram_gb": 0.4, "elapsed_min": 56.6, "gpu": 0},
            status="keep", config_hash="abc123",
        )
        assert g.has_config("abc123")
        assert not g.has_config("xyz789")


class TestReconciliation:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.graph_path = os.path.join(self.tmpdir, "graph.json")
        self.queue_dir = os.path.join(self.tmpdir, "queue")
        self.running_dir = os.path.join(self.tmpdir, "running")
        self.completed_dir = os.path.join(self.tmpdir, "completed")
        self.archive_dir = os.path.join(self.tmpdir, "archive")
        for d in (self.queue_dir, self.running_dir, self.completed_dir, self.archive_dir):
            os.makedirs(d, exist_ok=True)

        self.g = ExperimentGraph(path=self.graph_path)
        self.root = self.g.add_executed(
            parent_id=None, description="baseline", techniques=[],
            metrics={"primary_value": 0.814, "test_auc": 0.836, "test_bacc": 0.792,
                     "val_auc": 0.869, "val_bacc": 0.810,
                     "vram_gb": 0.4, "elapsed_min": 56.6, "gpu": 0},
            status="keep",
        )

    def test_promote_from_completed(self):
        pid = self.g.add_proposed(self.root, "focal", ["focal_g1"], rationale="test")
        self.g.mark_running(pid)
        completion = {
            "id": pid, "status": "completed",
            "primary_value": 0.832,
            # Contract shape: `metrics` is validation-only (B6 ingest guard
            # crashes a completion carrying held-out-named keys).
            "metrics": {"val_auc": 0.867, "val_bacc": 0.797},
            "elapsed_seconds": 4170, "peak_vram_mb": 500, "gpu": 1,
            "graph_metadata": {"parent_id": self.root},
        }
        with open(os.path.join(self.completed_dir, f"{pid}.json"), "w") as f:
            json.dump(completion, f)
        self.g.reconcile(self.queue_dir, self.running_dir,
                         self.completed_dir, self.archive_dir)
        node = self.g.get_node(pid)
        assert node["type"] == "executed"
        assert node["status"] == "keep"

    def test_recover_missing_node(self):
        node_id = "node_0099"
        spec = {
            "id": node_id,
            "graph_metadata": {
                "parent_id": self.root,
                "techniques": ["rdrop"],
                "config_hash": "abc",
            },
        }
        os.makedirs(os.path.join(self.archive_dir, node_id), exist_ok=True)
        with open(os.path.join(self.archive_dir, node_id, "spec.json"), "w") as f:
            json.dump(spec, f)
        completion = {
            "id": node_id, "status": "completed", "tsv_status": "discard",
            "primary_value": 0.790, "test_auc_roc": 0.80, "test_bacc": 0.78,
            "description": "rdrop attempt",
            "elapsed_min": 60, "gpu": 0,
        }
        with open(os.path.join(self.completed_dir, f"{node_id}.json"), "w") as f:
            json.dump(completion, f)
        self.g.reconcile(self.queue_dir, self.running_dir,
                         self.completed_dir, self.archive_dir)
        node = self.g.get_node(node_id)
        assert node is not None
        assert node["type"] == "executed"
        assert node["parent_id"] == self.root

    def test_reset_stale_running(self):
        pid = self.g.add_proposed(self.root, "stale", ["x"], rationale="test")
        self.g.mark_running(pid)
        self.g.reconcile(self.queue_dir, self.running_dir,
                         self.completed_dir, self.archive_dir)
        assert self.g.get_node(pid)["status"] == "pending"

    def test_running_in_queue_not_reset(self):
        pid = self.g.add_proposed(self.root, "queued", ["x"], rationale="test")
        self.g.mark_running(pid)
        with open(os.path.join(self.queue_dir, f"{pid}.json"), "w") as f:
            json.dump({"id": pid}, f)
        self.g.reconcile(self.queue_dir, self.running_dir,
                         self.completed_dir, self.archive_dir)
        assert self.g.get_node(pid)["status"] == "running"

    def test_stale_pending_proposal_cancelled(self):
        """Guard 2: old proposed/pending nodes with no orchestrator state and
        no archive result are cancelled by the reconcile zombie sweep."""
        pid = self.g.add_proposed(self.root, "stale zombie", ["x"],
                                  rationale="test")
        # Back-date its created_at to older than the 6h threshold.
        self.g.get_node(pid)["created_at"] = "2020-01-01T00:00:00"
        self.g.reconcile(self.queue_dir, self.running_dir,
                         self.completed_dir, self.archive_dir)
        node = self.g.get_node(pid)
        assert node["status"] == "cancelled"
        assert "stale" in node.get("cancel_reason", "")

    def test_fresh_pending_proposal_not_cancelled(self):
        """Guard 2: recently-created proposals must not be swept."""
        pid = self.g.add_proposed(self.root, "fresh", ["y"],
                                  rationale="test")
        # created_at is set by add_proposed to datetime.now(), so it's fresh.
        self.g.reconcile(self.queue_dir, self.running_dir,
                         self.completed_dir, self.archive_dir)
        node = self.g.get_node(pid)
        assert node["status"] == "pending"

    def test_stale_pending_with_archive_result_not_cancelled(self):
        """Guard 2: a stale proposal that has an archive result must be
        left alone (it should get promoted via archive recovery, not
        cancelled as a zombie)."""
        pid = self.g.add_proposed(self.root, "stale but archived", ["z"],
                                  rationale="test")
        self.g.get_node(pid)["created_at"] = "2020-01-01T00:00:00"
        # Plant an archive result and spec for it. The archive recovery
        # path only recovers nodes NOT yet in self.nodes — so for a node
        # that's already proposed/pending, the recovery branch won't fire.
        # The sweep must still skip it because a result exists on disk.
        os.makedirs(os.path.join(self.archive_dir, pid), exist_ok=True)
        with open(os.path.join(self.archive_dir, pid, "result.json"), "w") as f:
            json.dump({"status": "completed", "primary_value": 0.82,
                       "metrics": {"test_auc": 0.85, "test_bacc": 0.79,
                                   "val_auc": 0.85, "val_bacc": 0.79}}, f)
        with open(os.path.join(self.archive_dir, pid, "spec.json"), "w") as f:
            json.dump({"id": pid,
                       "graph_metadata": {"parent_id": self.root}}, f)
        self.g.reconcile(self.queue_dir, self.running_dir,
                         self.completed_dir, self.archive_dir)
        node = self.g.get_node(pid)
        # It should NOT be cancelled. (The current node remains pending; the
        # archive recovery branch only rebuilds missing nodes, so this node
        # simply isn't touched — the important assertion is "not cancelled".)
        assert node["status"] != "cancelled"

    def test_stale_threshold_configurable(self):
        """Guard 2: proposal_stale_hours kwarg controls the cutoff."""
        pid = self.g.add_proposed(self.root, "recentish", ["w"],
                                  rationale="test")
        # Back-date by ~2h.
        import datetime as dt
        self.g.get_node(pid)["created_at"] = (
            dt.datetime.now() - dt.timedelta(hours=2)
        ).isoformat()
        # With default 6h threshold, it stays pending.
        self.g.reconcile(self.queue_dir, self.running_dir,
                         self.completed_dir, self.archive_dir)
        assert self.g.get_node(pid)["status"] == "pending"
        # With a tighter 1h threshold, it gets cancelled.
        self.g.reconcile(self.queue_dir, self.running_dir,
                         self.completed_dir, self.archive_dir,
                         proposal_stale_hours=1.0)
        assert self.g.get_node(pid)["status"] == "cancelled"


class TestMigration:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.graph_path = os.path.join(self.tmpdir, "graph.json")

        self.tsv_path = os.path.join(self.tmpdir, "results.tsv")
        with open(self.tsv_path, "w") as f:
            f.write("commit\tval_auc\tval_bacc\ttest_auc\ttest_bacc\tprimary_value\tdelta\tvram_gb\telapsed_min\tstatus\tdescription\n")
            f.write("abc123\t0.869\t0.810\t0.836\t0.792\t0.814\t+0.814\t0.4\t56.6\tkeep\tbaseline\n")
            f.write("abc123\t0.846\t0.690\t0.820\t0.681\t0.751\t-0.063\t0.5\t68.3\tdiscard\tL2_norm\n")
            f.write("abc123\t0.864\t0.797\t0.843\t0.810\t0.827\t+0.013\t0.4\t53.4\tkeep\tno_inst_eval\n")
            f.write("def456\t0.866\t0.768\t0.875\t0.788\t0.832\t+0.004\t0.4\t69.5\tkeep\tfocal+gamma1\n")

        self.strategies_path = os.path.join(self.tmpdir, "strategies.json")
        with open(self.strategies_path, "w") as f:
            json.dump({
                "strategies": [
                    {"id": "poly_loss", "name": "Poly-1 loss", "tier": 2,
                     "status": "not_started", "description": "try poly loss",
                     "experiments": []},
                    {"id": "rdrop", "name": "R-Drop", "tier": 1,
                     "status": "exhausted", "description": "already tried",
                     "experiments": [{"desc": "rdrop_a1", "primary_value": 0.849}]},
                ],
                "meta": {"best_primary_value": 0.832},
            }, f)

    def test_import_from_tsv(self):
        g = ExperimentGraph.import_from_tsv(
            self.tsv_path, self.strategies_path, graph_path=self.graph_path
        )
        executed = [n for n in g.nodes.values() if n["type"] == "executed"]
        assert len(executed) == 4

        root = [n for n in executed if n["description"] == "baseline"][0]
        assert root["parent_id"] is None
        assert root["status"] == "keep"

        l2 = [n for n in executed if n["description"] == "L2_norm"][0]
        assert l2["parent_id"] == root["id"]

        no_inst = [n for n in executed if n["description"] == "no_inst_eval"][0]
        focal = [n for n in executed if n["description"] == "focal+gamma1"][0]
        assert focal["parent_id"] == no_inst["id"]

        proposed = [n for n in g.nodes.values() if n["type"] == "proposed"]
        assert len(proposed) == 1
        assert proposed[0]["description"] == "try poly loss"

        assert all(n.get("bootstrapped") for n in executed)


class TestNewFeatures:
    def test_multi_file_config_hash(self, tmp_path):
        content = {
            "train.py": "print('hello')",
            "models/clam.py": "class CLAM: pass",
        }
        h1 = ExperimentGraph.compute_config_hash(content)
        # Same content, different order - should produce same hash
        content2 = {
            "models/clam.py": "class CLAM: pass",
            "train.py": "print('hello')",
        }
        h2 = ExperimentGraph.compute_config_hash(content2)
        assert h1 == h2
        # Different content - different hash
        content3 = {
            "train.py": "print('world')",
            "models/clam.py": "class CLAM: pass",
        }
        h3 = ExperimentGraph.compute_config_hash(content3)
        assert h1 != h3

    def test_multi_file_hash_includes_base_commit(self, tmp_path):
        content = {"train.py": "print('hello')"}
        h1 = ExperimentGraph.compute_config_hash(content, base_commit="abc123")
        h2 = ExperimentGraph.compute_config_hash(content, base_commit="def456")
        assert h1 != h2

    def test_technique_map_parameterized(self, tmp_path):
        custom_map = {"my_tech": "my_technique"}
        graph = ExperimentGraph(
            path=str(tmp_path / "graph.json"),
            technique_map=custom_map,
        )
        assert graph._technique_map == custom_map

    def test_technique_map_defaults(self, tmp_path):
        graph = ExperimentGraph(path=str(tmp_path / "graph.json"))
        assert graph._technique_map == ExperimentGraph.DEFAULT_TECHNIQUE_MAP


# ---------------------------------------------------------------------------
# DBT-01: legacy schema round-trip tests
# These must be top-level (not inside a class) — pytest collects them directly.
# ---------------------------------------------------------------------------


def test_legacy_schema_round_trip(tmp_path):
    """DBT-01: pre-D-200 flat-key graph.json migrates on load — no KeyError."""
    import json
    from automil.graph import ExperimentGraph

    # Build a real pre-D-200 graph.json fixture (flat metric keys, no metrics dict)
    legacy_graph = {
        "schema_version": 1,          # legacy schema version
        "meta": {
            "best_primary_value": 0.87,
            "best_node_id": "node_0001",
            "total_executed": 1,
            "total_proposed": 0,
            "next_id": 2,
            "baseline_primary_value": 0.0,
        },
        "nodes": {
            "node_0001": {
                "id": "node_0001",
                "parent_id": None,
                "type": "executed",
                "status": "keep",
                "description": "baseline",
                "techniques": [],
                "primary_value": 0.87,
                "global_delta": 0.0,
                "parent_delta": 0.0,
                # PRE-D-200 flat keys — NO "metrics" dict:
                "val_auc": 0.85,
                "val_bacc": 0.80,
                "test_auc": 0.87,
                "test_bacc": 0.83,
                "vram_gb": 4.5,
                "elapsed_min": 68.3,
                "gpu": 0,
            }
        },
        "technique_stats": {},
    }
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(legacy_graph))

    # Load through the REAL ExperimentGraph (not hand-constructed result)
    g = ExperimentGraph(graph_path)

    # Assert migration happened
    node = g.nodes["node_0001"]
    assert "metrics" in node, "migration must add 'metrics' dict to legacy node"
    assert node["metrics"]["val_auc"] == 0.85
    assert node["metrics"]["val_bacc"] == 0.80
    assert node["metrics"]["test_auc"] == 0.87
    assert node["metrics"]["test_bacc"] == 0.83
    # Top-level keys preserved (keep-flat strategy)
    assert node["val_auc"] == 0.85
    # Schema version bumped in-memory
    assert g._data["schema_version"] == 3


def test_legacy_schema_absent_version(tmp_path):
    """DBT-01: graph.json with NO schema_version key also migrates."""
    import json
    from automil.graph import ExperimentGraph

    legacy_graph = {
        # NO schema_version key — truly legacy
        "meta": {
            "best_primary_value": 0.0,
            "best_node_id": None,
            "total_executed": 1,
            "total_proposed": 0,
            "next_id": 2,
            "baseline_primary_value": 0.0,
        },
        "nodes": {
            "node_0001": {
                "id": "node_0001", "parent_id": None, "type": "executed",
                "status": "keep", "description": "x", "techniques": [],
                "primary_value": 0.5, "global_delta": 0.0, "parent_delta": 0.0,
                "val_auc": 0.5, "val_bacc": 0.5, "test_auc": 0.5, "test_bacc": 0.5,
                "vram_gb": 1.0, "elapsed_min": 10.0, "gpu": 0,
            }
        },
        "technique_stats": {},
    }
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(legacy_graph))
    g = ExperimentGraph(graph_path)
    assert "metrics" in g.nodes["node_0001"], "absent schema_version must trigger migration"
    assert g._data["schema_version"] == 3


def test_post_d200_graph_not_remigrated(tmp_path):
    """DBT-01 idempotency: post-D-200 graph (schema_version=2) is not re-migrated."""
    import json
    from automil.graph import ExperimentGraph

    post_graph = {
        "schema_version": 2,
        "meta": {
            "best_primary_value": 0.87,
            "best_node_id": "node_0001",
            "total_executed": 1,
            "total_proposed": 0,
            "next_id": 2,
            "baseline_primary_value": 0.0,
        },
        "nodes": {
            "node_0001": {
                "id": "node_0001", "parent_id": None, "type": "executed",
                "status": "keep", "description": "baseline", "techniques": [],
                "primary_value": 0.87, "global_delta": 0.0, "parent_delta": 0.0,
                "metrics": {
                    "val_auc": 0.85,
                    "val_bacc": 0.80,
                    "test_auc": 0.87,
                    "test_bacc": 0.83,
                    "primary_value": 0.87,
                },
                "vram_gb": 4.5, "elapsed_min": 68.3, "gpu": 0,
            }
        },
        "technique_stats": {},
    }
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(post_graph))
    g = ExperimentGraph(graph_path)
    # metrics dict unchanged — same object values, no extra keys injected by migration
    assert g.nodes["node_0001"]["metrics"]["val_auc"] == 0.85
    assert g._data["schema_version"] == 3
    # Confirm migration did NOT add spurious keys to the metrics dict
    expected_keys = {"val_auc", "val_bacc", "test_auc", "test_bacc", "primary_value"}
    assert set(g.nodes["node_0001"]["metrics"].keys()) == expected_keys


class TestSchema3CompositeRetirementMigration:
    """A pre-rename graph must migrate on read, never silently zero out.

    Before schema 3, every node carried `composite`/`composite_se` and meta
    carried `best_composite`; the renamed readers default missing keys to
    0.0, which would strand the real best and let the next keep become
    "best" on a live project.
    """

    def _legacy(self, tmp_path):
        import json
        legacy = {
            "schema_version": 2,
            "meta": {"best_composite": 0.8074, "best_node_id": "node_0005",
                     "baseline_composite": 0.61, "total_executed": 1,
                     "total_proposed": 1, "next_id": 6, "scoring": {}},
            "nodes": {"node_0005": {
                "id": "node_0005", "type": "executed", "status": "keep",
                "composite": 0.8074, "composite_se": 0.02,
                "fold_composites": [{"fold_index": 0, "composite": 0.80},
                                     {"fold_index": 1, "composite": 0.82}],
                "metrics": {"val_auc": 0.8074},
                "metadata": {"composite_disagreement": {"reported": 0.9}},
            }},
            "technique_stats": {},
        }
        path = tmp_path / "graph.json"
        path.write_text(json.dumps(legacy))
        return path

    def test_nodes_meta_and_fold_entries_migrate(self, tmp_path):
        from automil.graph import ExperimentGraph, node_fold_primary_values
        g = ExperimentGraph(path=str(self._legacy(tmp_path)))
        node = g.get_node("node_0005")
        assert node["primary_value"] == 0.8074
        assert node["primary_se"] == 0.02
        assert "composite" not in node and "composite_se" not in node
        assert node_fold_primary_values(node) == {0: 0.80, 1: 0.82}
        assert g.meta["best_primary_value"] == 0.8074
        assert g.meta["baseline_primary_value"] == 0.61
        assert "primary_value_disagreement" in node["metadata"]
        assert g._data["schema_version"] == 3

    def test_migrated_graph_persists_on_save(self, tmp_path):
        import json
        from automil.graph import ExperimentGraph
        path = self._legacy(tmp_path)
        g = ExperimentGraph(path=str(path))
        g.save()
        raw = json.loads(path.read_text())
        assert raw["schema_version"] == 3
        assert raw["nodes"]["node_0005"]["primary_value"] == 0.8074
        assert "composite" not in raw["nodes"]["node_0005"]
