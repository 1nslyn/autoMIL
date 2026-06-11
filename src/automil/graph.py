"""Experiment graph: directed tree tracking for multi-branch exploration.

Provides atomic read/write to graph.json. Concurrent writers (daemon +
CLI) coordinate through an advisory ``flock`` on a sidecar ``.lock``
file; see ``locked_update``.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import io
import json
import logging
import math
import os
import tempfile
import tokenize
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def locked_update(graph_path: str | Path, *, technique_map: dict[str, str] | None = None):
    """Read-modify-write context manager for graph.json under a fcntl lock.

    Use this whenever a process needs to mutate ``graph.json`` to prevent
    lost updates between the daemon and CLI:

        with locked_update(path) as graph:
            graph.add_proposed(...)
            # graph.save() runs on context exit

    Acquires an exclusive POSIX advisory lock on ``<graph_path>.lock``
    BEFORE constructing the in-memory ExperimentGraph, so the snapshot
    read by the constructor cannot be invalidated by another writer
    until the block exits.

    Atomic-rename in ``save()`` alone prevented torn writes but not
    lost updates; this context manager is the fix for the race the
    audit flagged.

    ``technique_map`` is forwarded to the constructed ExperimentGraph so
    consumer-supplied vocabularies (declared in ``automil/config.yaml``:
    ``scoring.technique_map``) drive auto-extraction inside the locked
    block. None preserves the framework's empty default.
    """
    path = Path(graph_path)
    lock_path = path.with_suffix(path.suffix + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_f = open(lock_path, "w")
    try:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        graph = ExperimentGraph(path=path, technique_map=technique_map)
        yield graph
        graph.save()
    finally:
        try:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
        finally:
            lock_f.close()


class ExperimentGraph:
    # Generic by default. Consumers that want technique-name normalisation
    # supply their own dict via ``technique_map=`` on the constructor (or
    # to ``import_from_tsv``). The framework ships no domain-specific
    # vocabulary here — the empty default is the contract.
    DEFAULT_TECHNIQUE_MAP: dict[str, str] = {}

    def __init__(self, path: str | Path, technique_map: dict[str, list[str]] | None = None, data: dict | None = None):
        self.path = Path(path)
        self._technique_map = technique_map if technique_map is not None else self.DEFAULT_TECHNIQUE_MAP
        loaded_from_disk = False
        if data is not None:
            self._data = data
        elif self.path.exists():
            self._data = json.loads(self.path.read_text())
            loaded_from_disk = True
        else:
            self._data = {}
        # Normalize: fill in missing top-level / meta keys with defaults
        # so legacy schemas and fresh-init paths both work. When loading
        # an existing file that's missing keys, log a warning — partial-
        # write corruption silently filled in with defaults would mask
        # real data loss, and operators need a paper trail.
        defaults = {
            "schema_version": 1,
            "meta": {
                "best_composite": 0.0,
                "best_node_id": None,
                "total_executed": 0,
                "total_proposed": 0,
                "next_id": 1,
                "baseline_composite": 0.0,
                "scoring": {
                    "exploration_weight": 0.005,
                    "novelty_weight": 0.003,
                },
            },
            "nodes": {},
            "technique_stats": {},
        }
        missing_top = [k for k in defaults if k not in self._data]
        for k, v in defaults.items():
            self._data.setdefault(k, v if not isinstance(v, dict) else dict(v))
        missing_meta = [k for k in defaults["meta"] if k not in self._data["meta"]]
        for mk, mv in defaults["meta"].items():
            self._data["meta"].setdefault(mk, mv if not isinstance(mv, dict) else dict(mv))
        if loaded_from_disk and (missing_top or missing_meta):
            # Top-level missing keys are the more alarming signal (file
            # exists but is structurally incomplete). Meta-only gaps are
            # usually schema-version drift from an old graph and are
            # safe to fill silently — but we still report the meta keys
            # so a schema migration audit can pick them up.
            logger.warning(
                "graph.json at %s loaded with missing keys "
                "(top-level=%r, meta=%r); filled with defaults. If this "
                "is not a known schema migration, check for partial-"
                "write corruption.",
                self.path, missing_top, missing_meta,
            )

    @staticmethod
    def load(path: str | Path, technique_map: dict[str, str] | None = None) -> ExperimentGraph:
        return ExperimentGraph(path=path, technique_map=technique_map)

    @property
    def meta(self) -> dict:
        return self._data["meta"]

    @property
    def nodes(self) -> dict:
        return self._data["nodes"]

    @property
    def technique_stats_data(self) -> dict:
        return self._data["technique_stats"]

    # --- ID generation ---
    def next_id(self) -> str:
        nid = self.meta["next_id"]
        self.meta["next_id"] = nid + 1
        return f"node_{nid:04d}"

    # --- Reading ---
    def get_node(self, node_id: str) -> dict | None:
        return self.nodes.get(node_id)

    def best_node(self) -> dict | None:
        best_id = self.meta.get("best_node_id")
        node = self.nodes.get(best_id) if best_id else None
        # D-01: partial results are quarantined — excluded from best_node
        if node and node.get("status") == "partial":
            return None
        return node

    def children(self, node_id: str) -> list[dict]:
        return [n for n in self.nodes.values() if n.get("parent_id") == node_id]

    def lineage(self, node_id: str) -> list[dict]:
        path = []
        current = node_id
        while current:
            node = self.get_node(current)
            if node is None:
                break
            path.append(node)
            current = node.get("parent_id")
        path.reverse()
        return path

    def technique_stats(self, technique: str) -> dict:
        return self.technique_stats_data.get(technique, {
            "times_tried": 0, "best_parent_delta": 0.0, "avg_parent_delta": 0.0,
        })

    # --- Writing ---
    def _auto_extract_if_empty(self, description: str, techniques: list[str]) -> list[str]:
        """If techniques is empty and a consumer technique_map is configured,
        auto-extract tags from the description.

        Backward-compatible: when the technique_map is empty (framework default)
        OR techniques is non-empty (explicit caller input), this is a no-op.
        Consumers opt in by populating ``scoring.technique_map`` in
        ``automil/config.yaml`` and threading it through via
        ``cli/_helpers._load_technique_map``.
        """
        if techniques:
            return techniques
        if not self._technique_map:
            return techniques
        return self._extract_techniques(description)

    def add_executed(self, parent_id: str | None, description: str,
                     techniques: list[str], metrics: dict,
                     status: str = "discard", commit: str | None = None,
                     config_hash: str | None = None,
                     bootstrapped: bool = False) -> str:
        nid = self.next_id()
        parent = self.get_node(parent_id) if parent_id else None
        parent_composite = parent.get("composite", 0.0) if parent else 0.0
        composite = metrics.get("composite", 0.0)
        techniques = self._auto_extract_if_empty(description, techniques)

        node = {
            "id": nid,
            "parent_id": parent_id,
            "type": "executed",
            "status": status,
            "description": description,
            "techniques": techniques,
            # Framework-owned scalars (D-200): preserved at top level.
            "composite": composite,
            "global_delta": metrics.get("global_delta", metrics.get("delta", 0.0)),
            "parent_delta": composite - parent_composite,
            # Consumer metrics stored as opaque dict (D-200 / DEC-04).
            "metrics": dict(metrics),
            # Orchestrator-measured scalars (kept top-level for ergonomics; read
            # by init.py for empirical default_vram_estimate_gb).
            "vram_gb": metrics.get("vram_gb", 0.0),
            "elapsed_min": metrics.get("elapsed_min", 0.0),
            "gpu": metrics.get("gpu", -1),
            "commit": commit,
            "archive_id": nid,
            "config_hash": config_hash,
            "potential": 0.0,
            "child_count": 0,
            "created_at": datetime.now().isoformat(),
        }
        if bootstrapped:
            node["bootstrapped"] = True

        self.nodes[nid] = node
        self.meta["total_executed"] += 1

        if composite > self.meta["best_composite"]:
            self.meta["best_composite"] = composite
            self.meta["best_node_id"] = nid

        if parent_id and parent_id in self.nodes:
            self.nodes[parent_id]["child_count"] = len(self.children(parent_id))

        self._update_technique_stats(techniques, composite - parent_composite)
        return nid

    def add_proposed(self, parent_id: str, description: str,
                     techniques: list[str], rationale: str = "",
                     reference: str | None = None,
                     expected_gain: str = "low", effort: str = "low",
                     tier: int = 2, kind: str = "unspecified") -> str:
        nid = self.next_id()
        techniques = self._auto_extract_if_empty(description, techniques)
        node = {
            "id": nid,
            "parent_id": parent_id,
            "type": "proposed",
            "status": "pending",
            "description": description,
            "techniques": techniques,
            "tier": tier,
            # kind classifies the experiment for the architecture-vs-HP portfolio
            # (P1.2): architecture | regularization | hp | data | ensemble |
            # unspecified. Drives `automil portfolio` so the loop stays
            # structurally exploratory, not a pure hyperparameter sweep.
            "kind": kind,
            "rationale": rationale,
            "reference": reference,
            "expected_gain": expected_gain,
            "effort": effort,
            "potential": 0.0,
            "created_at": datetime.now().isoformat(),
        }
        self.nodes[nid] = node
        self.meta["total_proposed"] += 1
        return nid

    def mark_running(self, node_id: str) -> bool:
        node = self.nodes[node_id]
        if node["type"] != "proposed" or node["status"] != "pending":
            logger.warning(
                "mark_running skipped for %s: type=%s status=%s",
                node_id, node["type"], node["status"],
            )
            return False
        node["status"] = "running"
        return True

    def promote(self, node_id: str, metrics: dict):
        node = self.nodes[node_id]
        parent = self.get_node(node.get("parent_id")) if node.get("parent_id") else None
        parent_composite = parent.get("composite", 0.0) if parent else 0.0
        composite = metrics.get("composite", 0.0)
        status = metrics.get("status", "discard")

        node["type"] = "executed"
        node["status"] = status
        node["composite"] = composite
        node["global_delta"] = metrics.get("global_delta", metrics.get("delta", 0.0))
        node["parent_delta"] = composite - parent_composite
        # D-200: store consumer metrics as opaque dict.
        node["metrics"] = dict(metrics)
        # Orchestrator-measured scalars stay top-level.
        node["vram_gb"] = metrics.get("vram_gb", 0.0)
        node["elapsed_min"] = metrics.get("elapsed_min", 0.0)
        node["gpu"] = metrics.get("gpu", -1)
        node["commit"] = metrics.get("commit")
        node["archive_id"] = node_id
        node["config_hash"] = metrics.get("config_hash")
        node["child_count"] = 0

        self.meta["total_executed"] += 1
        self.meta["total_proposed"] = max(0, self.meta["total_proposed"] - 1)

        if composite > self.meta["best_composite"]:
            self.meta["best_composite"] = composite
            self.meta["best_node_id"] = node_id

        pid = node.get("parent_id")
        if pid and pid in self.nodes:
            self.nodes[pid]["child_count"] = len([
                n for n in self.nodes.values()
                if n.get("parent_id") == pid and n["type"] == "executed"
            ])

        self._update_technique_stats(node.get("techniques", []),
                                     composite - parent_composite)

        self._reevaluate_descendants(node_id)

    def _reevaluate_descendants(self, root_id: str) -> None:
        """Recompute keep/discard for executed descendants of root_id.

        Children can be promoted before their parent completes, in which case
        parent metrics default to 0 and the Pareto check spuriously yields
        'keep'. Re-run the check now that root_id has real metrics.
        """
        stack = [root_id]
        while stack:
            pid = stack.pop()
            parent = self.nodes.get(pid)
            if not parent or parent.get("type") != "executed":
                continue
            p_comp = parent.get("composite", 0)
            for child in self.nodes.values():
                if child.get("parent_id") != pid:
                    continue
                if child.get("type") != "executed":
                    continue
                if child.get("status") == "partial":
                    continue   # D-01: partial nodes are not keep/discard candidates
                if child.get("status") not in ("keep", "discard"):
                    continue
                c_comp = child.get("composite", 0)
                # D-200 Option B: composite-only dominance. Framework no longer
                # encodes a named-key monotonicity guard; the composite is
                # consumer-computed and is the single dominance signal.
                keep = c_comp > p_comp
                child["status"] = "keep" if keep else "discard"
                child["parent_delta"] = c_comp - p_comp
                stack.append(child["id"])

    def mark_failed(self, node_id: str, status: str, error: str = "",
                    config_hash: str | None = None):
        node = self.nodes[node_id]
        node["type"] = "executed"
        node["status"] = status
        node["composite"] = 0.0
        node["parent_delta"] = 0.0
        node["global_delta"] = 0.0
        node["error"] = error
        node["child_count"] = 0
        node["archive_id"] = node_id
        if config_hash:
            node["config_hash"] = config_hash
        self.meta["total_executed"] += 1
        self.meta["total_proposed"] = max(0, self.meta["total_proposed"] - 1)

    def cancel(self, node_id: str):
        node = self.nodes[node_id]
        node["status"] = "cancelled"
        self.meta["total_proposed"] = max(0, self.meta["total_proposed"] - 1)

    # --- Technique stats ---
    def _update_technique_stats(self, techniques: list[str], parent_delta: float):
        for tech in techniques:
            if tech not in self.technique_stats_data:
                self.technique_stats_data[tech] = {
                    "times_tried": 0,
                    "best_parent_delta": float("-inf"),
                    "avg_parent_delta": 0.0,
                    "_total_delta": 0.0,
                }
            stats = self.technique_stats_data[tech]
            stats["times_tried"] += 1
            stats["_total_delta"] = stats.get("_total_delta", 0.0) + parent_delta
            stats["avg_parent_delta"] = stats["_total_delta"] / stats["times_tried"]
            if parent_delta > stats["best_parent_delta"]:
                stats["best_parent_delta"] = parent_delta

    # --- Scoring ---
    def recalculate_scores(self):
        total = max(1, self.meta["total_executed"])
        w_e = self.meta["scoring"]["exploration_weight"]
        w_n = self.meta["scoring"]["novelty_weight"]

        for node in self.nodes.values():
            if node["type"] == "executed":
                child_count = len([
                    n for n in self.nodes.values()
                    if n.get("parent_id") == node["id"] and n["type"] == "executed"
                ])
                node["child_count"] = child_count
                node["potential"] = round(
                    node.get("composite", 0) +
                    w_e * math.sqrt(math.log(total) / (1 + child_count)),
                    6,
                )
            elif node["type"] == "proposed" and node["status"] != "cancelled":
                parent = self.get_node(node.get("parent_id"))
                parent_composite = parent.get("composite", 0.0) if parent else 0.0
                siblings_tried = len([
                    n for n in self.nodes.values()
                    if n.get("parent_id") == node.get("parent_id")
                    and n["type"] == "executed"
                ])
                tech_novelty = 0.0
                for tech in node.get("techniques", []):
                    stats = self.technique_stats_data.get(tech, {})
                    tech_novelty += 1.0 / (1 + stats.get("times_tried", 0))
                if node.get("techniques"):
                    tech_novelty /= len(node["techniques"])

                node["potential"] = round(
                    parent_composite +
                    w_e * math.sqrt(math.log(total) / (1 + siblings_tried)) +
                    w_n * tech_novelty,
                    6,
                )

    def recompute_best(self) -> tuple[str | None, float, str | None, float]:
        """Walk executed/keep nodes; pick max-composite node as best (CLI-07 / D-10..D-12).

        Returns ``(old_node_id, old_composite, new_node_id, new_composite)``.
        Mutates ``self._data["meta"]`` in place. The caller decides whether to
        call ``self.save()`` — recompute_best does NOT persist (so the CLI
        ``--dry-run`` flag can skip save).

        Walk semantics (D-10): only nodes where ``type == "executed"`` AND
        ``status == "keep"``. Discarded / crashed / cancelled / budget-killed /
        proposed nodes are excluded.

        Composite formula (D-11): uses the existing per-node ``composite`` field
        as already populated by train.py → result.json → orchestrator pipeline.
        Phase 0 does NOT redefine the formula — that's Phase 8 / DEC-04.

        Tie-break (D-12): equal composites resolve to lexicographic min on
        ``node_id``. Stable and deterministic.
        """
        old_id = self.meta.get("best_node_id")
        old_c = float(self.meta.get("best_composite", 0.0))

        keep_nodes: list[tuple[str, float]] = []
        for node_id, node in self.nodes.items():
            if node.get("type") == "executed" and node.get("status") == "keep":
                keep_nodes.append((node_id, float(node.get("composite", 0.0))))

        if not keep_nodes:
            new_id: str | None = None
            new_c = 0.0
        else:
            # Sort: composite DESC, node_id ASC (lex tie-break — D-12).
            keep_nodes.sort(key=lambda x: (-x[1], x[0]))
            new_id, new_c = keep_nodes[0]

        self.meta["best_node_id"] = new_id
        self.meta["best_composite"] = new_c
        return old_id, old_c, new_id, new_c

    def rank_proposals(self, n: int = 6, max_per_branch: int = 2) -> list[dict]:
        proposals = [
            nd for nd in self.nodes.values()
            if nd["type"] == "proposed" and nd["status"] == "pending"
        ]
        proposals.sort(key=lambda x: x.get("potential", 0), reverse=True)

        result = []
        branch_counts: dict[str, int] = {}
        for p in proposals:
            pid = p.get("parent_id", "")
            if branch_counts.get(pid, 0) >= max_per_branch:
                continue
            result.append(p)
            branch_counts[pid] = branch_counts.get(pid, 0) + 1
            if len(result) >= n:
                break
        return result

    # --- Gate helpers (D-144, GTE-06) ---

    def nominations_in_window(self, days: int = 30) -> list[dict]:
        """Return nodes whose history contains a 'nominated' event in the last ``days`` days.

        A node may have multiple 'nominated' events (e.g. retire+re-nominate).
        The first matching event within the window is sufficient to include the
        node in the result — each node appears at most once.

        Legacy nodes without a ``history`` key are silently skipped (D-147).
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        result = []
        for node in self.nodes.values():
            for event in node.get("history", []):
                if event.get("event") != "nominated":
                    continue
                try:
                    ts = datetime.fromisoformat(event["timestamp"])
                except (ValueError, KeyError, TypeError):
                    continue
                # Normalise naive timestamps to UTC for comparison
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts > cutoff:
                    result.append(node)
                    break
        return result

    def promotion_rate(self, days: int = 30) -> float:
        """Return promoted / nominated over a rolling window (D-144).

        Returns 0.0 when no nominations exist in the window (zero-division guard).
        Promoted nodes are those whose current status is ``'registered'``.
        """
        nominated = self.nominations_in_window(days)
        if not nominated:
            return 0.0
        promoted = [n for n in nominated if n.get("status") == "registered"]
        return len(promoted) / len(nominated)

    # --- Deduplication ---
    @staticmethod
    def compute_config_hash(content: str | dict[str, str], base_commit: str = "") -> str:
        """Hash experiment config. Single script or {path: content} dict."""
        if isinstance(content, dict):
            parts = []
            for path in sorted(content.keys()):
                file_hash = hashlib.sha256(content[path].encode()).hexdigest()
                parts.append(f"{path}:{file_hash}")
            combined = base_commit + "\n" + "\n".join(parts)
            return hashlib.sha256(combined.encode()).hexdigest()[:16]
        else:
            # Keep existing tokenizer-based hash logic for single file
            try:
                tokens = tokenize.generate_tokens(io.StringIO(content).readline)
                code_tokens = [
                    tok.string for tok in tokens
                    if tok.type not in (tokenize.COMMENT, tokenize.NL,
                                        tokenize.NEWLINE, tokenize.INDENT,
                                        tokenize.DEDENT, tokenize.ENCODING)
                ]
                normalized = " ".join(code_tokens)
            except tokenize.TokenError:
                normalized = content
            return hashlib.sha256(normalized.encode()).hexdigest()

    def has_config(self, config_hash: str) -> bool:
        return any(
            n.get("config_hash") == config_hash
            for n in self.nodes.values()
            if n.get("config_hash")
        )

    # --- Technique extraction ---
    def _extract_techniques(self, description: str) -> list[str]:
        """Extract technique tags from a description string."""
        techniques = []
        desc_lower = description.lower()
        for pattern, tag in self._technique_map.items():
            if pattern in desc_lower and tag not in techniques:
                techniques.append(tag)
        return techniques

    # --- Reconciliation ---
    def reconcile(self, queue_dir: str, running_dir: str,
                  completed_dir: str, archive_dir: str,
                  proposal_stale_hours: float = 6.0):
        queue_path = Path(queue_dir)
        running_path = Path(running_dir)
        completed_path = Path(completed_dir)
        archive_path = Path(archive_dir)

        orch_ids = set()
        # queue/ is flat (no subdirs); running/ is namespaced per D-169 (Phase 6):
        # running/local/*.json, running/slurm/*.json, running/ray/*.json.
        # Use rglob for running_path to find entries across all backend subdirs.
        for d, glob_fn in ((queue_path, "glob"), (running_path, "rglob")):
            if d.exists():
                for f in getattr(d, glob_fn)("*.json"):
                    try:
                        spec = json.loads(f.read_text())
                        orch_ids.add(spec.get("id", f.stem))
                    except (json.JSONDecodeError, Exception):
                        orch_ids.add(f.stem)

        if completed_path.exists():
            for f in completed_path.glob("*.json"):
                try:
                    completion = json.loads(f.read_text())
                except (json.JSONDecodeError, Exception):
                    continue
                node_id = completion.get("id", f.stem)
                orch_ids.add(node_id)

                node = self.get_node(node_id)
                if node and node["type"] == "executed":
                    continue

                orch_status = completion.get("status", "")
                if orch_status in ("oom", "crash", "timeout"):
                    graph_status = orch_status
                elif orch_status == "completed":
                    composite = completion.get("composite", 0.0)
                    comp_metrics = completion.get("metrics", {})
                    gm = completion.get("graph_metadata", {})
                    if not gm:
                        spec_file = archive_path / node_id / "spec.json"
                        if spec_file.exists():
                            try:
                                gm = json.loads(spec_file.read_text()).get("graph_metadata", {})
                            except Exception:
                                pass
                    parent_id_check = gm.get("parent_id")
                    # Fall back to existing node's parent if metadata is missing
                    if not parent_id_check and node:
                        parent_id_check = node.get("parent_id")
                    parent_node = self.get_node(parent_id_check) if parent_id_check else None
                    if parent_node:
                        p_comp = parent_node.get("composite", 0)
                        # D-200 Option B: composite-only dominance.
                        keep = composite > p_comp
                        graph_status = "keep" if keep else "discard"
                    else:
                        graph_status = "keep" if composite > 0 else "discard"
                else:
                    graph_status = "discard"

                comp_metrics = completion.get("metrics", {})
                metrics = dict(comp_metrics)  # D-200: spread consumer metrics
                metrics["composite"] = completion.get("composite", 0.0)
                metrics["vram_gb"] = completion.get("peak_vram_mb", 0) / 1024
                metrics["elapsed_min"] = completion.get("elapsed_seconds", 0) / 60
                metrics["gpu"] = completion.get("gpu", -1)
                metrics["status"] = graph_status
                metrics["global_delta"] = completion.get("composite", 0) - self.meta.get("best_composite", 0)

                config_hash = completion.get("config_hash")
                if not config_hash:
                    spec_file = archive_path / node_id / "spec.json"
                    if spec_file.exists():
                        try:
                            spec_data = json.loads(spec_file.read_text())
                            config_hash = spec_data.get("graph_metadata", {}).get("config_hash")
                        except (json.JSONDecodeError, Exception):
                            pass
                metrics["config_hash"] = config_hash

                if node:
                    if graph_status in ("keep", "discard"):
                        self.promote(node_id, metrics)
                    else:
                        self.mark_failed(node_id, graph_status,
                                         completion.get("error", ""),
                                         config_hash=config_hash)
                else:
                    parent_id = None
                    techniques = []
                    spec_file = archive_path / node_id / "spec.json"
                    if spec_file.exists():
                        try:
                            spec = json.loads(spec_file.read_text())
                            gm = spec.get("graph_metadata", {})
                            parent_id = gm.get("parent_id")
                            techniques = gm.get("techniques", [])
                            if not config_hash:
                                config_hash = gm.get("config_hash")
                                metrics["config_hash"] = config_hash
                        except (json.JSONDecodeError, Exception):
                            pass

                    self.nodes[node_id] = {
                        "id": node_id,
                        "parent_id": parent_id,
                        "type": "executed",
                        "status": graph_status,
                        "description": completion.get("description", "recovered"),
                        "techniques": techniques,
                        "composite": metrics["composite"],
                        "global_delta": metrics["global_delta"],
                        "parent_delta": 0.0,
                        # D-200: consumer metrics opaque dict.
                        "metrics": dict(comp_metrics),
                        "vram_gb": metrics["vram_gb"],
                        "elapsed_min": metrics["elapsed_min"],
                        "gpu": metrics["gpu"],
                        "commit": None,
                        "archive_id": node_id,
                        "config_hash": metrics.get("config_hash"),
                        "potential": 0.0,
                        "child_count": 0,
                        "created_at": datetime.now().isoformat(),
                        "recovered": True,
                    }
                    if parent_id and parent_id in self.nodes:
                        parent_comp = self.nodes[parent_id].get("composite", 0)
                        self.nodes[node_id]["parent_delta"] = metrics["composite"] - parent_comp
                    self.meta["total_executed"] += 1
                    if metrics["composite"] > self.meta["best_composite"]:
                        self.meta["best_composite"] = metrics["composite"]
                        self.meta["best_node_id"] = node_id

                    self._update_technique_stats(
                        techniques, self.nodes[node_id]["parent_delta"])

                    if node_id.startswith("node_"):
                        try:
                            recovered_num = int(node_id.split("_")[1])
                            if recovered_num >= self.meta["next_id"]:
                                self.meta["next_id"] = recovered_num + 1
                        except (ValueError, IndexError):
                            pass

        # Archive-based recovery: scan for result.json in archive dirs
        if archive_path.exists():
            for node_dir in archive_path.iterdir():
                if not node_dir.is_dir():
                    continue
                node_id_r = node_dir.name
                result_file = node_dir / "result.json"
                if node_id_r not in self.nodes and result_file.exists():
                    try:
                        result = json.loads(result_file.read_text())
                        spec_file = node_dir / "spec.json"
                        spec = json.loads(spec_file.read_text()) if spec_file.exists() else {}
                        gm = spec.get("graph_metadata", {})
                        r_metrics = result.get("metrics", {})
                        composite = result.get("composite", 0.0)
                        num = int(node_id_r.split("_")[1])
                        if num >= self.meta["next_id"]:
                            self.meta["next_id"] = num + 1

                        parent_id = gm.get("parent_id")
                        parent = self.get_node(parent_id) if parent_id else None
                        parent_composite = parent.get("composite", 0.0) if parent else 0.0
                        raw_status = result.get("status", "completed")
                        if raw_status == "completed":
                            if parent:
                                p_comp = parent.get("composite", 0)
                                # D-200 Option B: composite-only dominance.
                                keep = composite > p_comp
                                status = "keep" if keep else "discard"
                            else:
                                status = "keep" if composite > 0 else "discard"
                        else:
                            status = raw_status

                        techniques = gm.get("techniques", [])
                        self.nodes[node_id_r] = {
                            "id": node_id_r, "parent_id": parent_id,
                            "type": "executed", "status": status,
                            "description": spec.get("description", f"recovered {node_id_r}"),
                            "techniques": techniques, "composite": composite,
                            "global_delta": composite - self.meta.get("best_composite", 0),
                            "parent_delta": composite - parent_composite,
                            # D-200: consumer metrics opaque dict.
                            "metrics": dict(r_metrics),
                            "vram_gb": result.get("peak_vram_mb", 0) / 1024,
                            "elapsed_min": result.get("elapsed_seconds", 0) / 60,
                            "gpu": -1,
                            "config_hash": gm.get("config_hash"),
                            "archive_id": node_id_r, "recovered": True,
                            "created_at": datetime.now().isoformat(),
                        }
                        self.meta["total_executed"] += 1
                        if composite > self.meta.get("best_composite", 0):
                            self.meta["best_composite"] = composite
                            self.meta["best_node_id"] = node_id_r
                        parent_delta = composite - parent_composite
                        self._update_technique_stats(techniques, parent_delta)
                    except (json.JSONDecodeError, Exception):
                        continue

        for node in list(self.nodes.values()):
            if node["type"] == "proposed" and node["status"] == "running":
                if node["id"] not in orch_ids:
                    node["status"] = "pending"

        # Zombie sweep: proposed/pending nodes that have no presence in
        # orchestrator state (queue/running/completed) and no archive result,
        # and whose created_at is older than proposal_stale_hours, are
        # cancelled. This cleans up stale proposals left behind by agent
        # resubmissions and orchestrator restarts — the class of zombies
        # that accumulated as 0018/0047/0048/0049 in the ccrcc run.
        now = datetime.now()
        stale_sec = proposal_stale_hours * 3600
        archive_path_obj = Path(archive_dir)
        for node in list(self.nodes.values()):
            if node.get("type") != "proposed":
                continue
            if node.get("status") != "pending":
                continue
            if node["id"] in orch_ids:
                continue
            result_file = archive_path_obj / node["id"] / "result.json"
            if result_file.exists():
                continue
            created = node.get("created_at")
            if not created:
                continue
            try:
                age_s = (now - datetime.fromisoformat(created)).total_seconds()
            except (ValueError, TypeError):
                continue
            if age_s <= stale_sec:
                continue
            node["status"] = "cancelled"
            node["cancel_reason"] = (
                f"stale: no orchestrator state, no archive result, "
                f"age {age_s / 3600:.1f}h > {proposal_stale_hours}h"
            )
            self.meta["total_proposed"] = max(
                0, self.meta["total_proposed"] - 1
            )

        self.recalculate_scores()

    # --- Migration ---
    @staticmethod
    def import_from_tsv(tsv_path: str, strategies_path: str | None = None,
                        graph_path: str | Path = "graph.json",
                        technique_map: dict[str, str] | None = None) -> ExperimentGraph:
        """Bootstrap a graph from a TSV produced by ``_append_results_tsv``.

        Column order is read from the header row, not hardcoded — any
        columns beyond ``node_id``, ``composite``, ``vram_gb``,
        ``elapsed_min``, ``status``, ``description`` are mapped into the
        node's ``metrics`` dict by their header name. Any consumer's TSV
        round-trips without framework changes.

        ``technique_map`` is the optional consumer-specific shorthand
        dict for tagging techniques from the description; default empty
        (no tagging). Pass the consumer's own map to recover
        domain-shorthand behaviour.
        """
        g = ExperimentGraph(path=graph_path, technique_map=technique_map or {})

        with open(tsv_path) as f:
            lines = f.readlines()

        if not lines or len(lines) < 2:
            return g

        header_cols = lines[0].strip().split("\t")
        # First column accepted as identifier under either name: post-v1.0
        # is "node_id"; pre-v1.0 was "commit". Both round-trip.
        if header_cols and header_cols[0] in ("node_id", "commit"):
            i_node = 0
        else:
            try:
                i_node = header_cols.index("node_id")
            except ValueError:
                raise ValueError(
                    f"TSV {tsv_path} has no 'node_id' or 'commit' column "
                    "as the identifier."
                )
        try:
            i_composite = header_cols.index("composite")
            i_vram = header_cols.index("vram_gb")
            i_elapsed = header_cols.index("elapsed_min")
            i_status = header_cols.index("status")
            i_desc = header_cols.index("description")
        except ValueError as exc:
            raise ValueError(
                f"TSV {tsv_path} is missing one of the required columns "
                f"(composite, vram_gb, elapsed_min, status, description): "
                f"{exc}"
            )
        _RESERVED = {header_cols[i_node], "composite", "vram_gb",
                     "elapsed_min", "status", "description", "delta"}
        # All other columns are treated as metrics.
        metric_idx = [
            (col, idx) for idx, col in enumerate(header_cols)
            if col not in _RESERVED
        ]

        rows = lines[1:]
        current_best_id = None

        for row in rows:
            parts = row.strip().split("\t")
            if len(parts) < len(header_cols):
                continue

            commit = parts[i_node]
            try:
                composite = float(parts[i_composite])
            except ValueError:
                continue
            try:
                vram_gb = float(parts[i_vram])
                elapsed_min = float(parts[i_elapsed])
            except ValueError:
                vram_gb, elapsed_min = 0.0, 0.0
            status = parts[i_status]
            description = parts[i_desc]

            metrics: dict[str, float] = {
                "composite": composite,
                "vram_gb": vram_gb,
                "elapsed_min": elapsed_min,
                "gpu": -1,
            }
            # Carry the optional pre-v1.0 `delta` column into metrics for
            # round-trip fidelity, parsing "+0.013"-style strings.
            if "delta" in header_cols:
                try:
                    metrics["delta"] = float(parts[header_cols.index("delta")].replace("+", ""))
                except (ValueError, IndexError):
                    pass
            for col_name, idx in metric_idx:
                cell = parts[idx]
                if cell == "":
                    continue
                try:
                    metrics[col_name] = float(cell)
                except ValueError:
                    # non-numeric metric column — store raw
                    metrics[col_name] = cell  # type: ignore[assignment]

            techniques: list[str] = []
            desc_lower = description.lower()
            for pattern, tag in (g._technique_map or {}).items():
                if pattern in desc_lower and tag not in techniques:
                    techniques.append(tag)

            nid = g.add_executed(
                parent_id=current_best_id,
                description=description,
                techniques=techniques,
                metrics=metrics,
                status=status,
                commit=commit,
                bootstrapped=True,
            )

            if status == "keep":
                current_best_id = nid

        if strategies_path and os.path.exists(strategies_path):
            with open(strategies_path) as f:
                strat_data = json.loads(f.read())
            best_id = g.meta.get("best_node_id")
            for strat in strat_data.get("strategies", []):
                if strat.get("status") == "not_started" and best_id:
                    g.add_proposed(
                        parent_id=best_id,
                        description=strat.get("description", strat.get("name", "")),
                        techniques=[strat["id"]],
                        rationale=strat.get("description", ""),
                        reference=strat.get("reference"),
                        expected_gain=strat.get("expected_gain", "low"),
                        effort=strat.get("effort", "low"),
                        tier=strat.get("tier", 2),
                    )

        g.recalculate_scores()
        return g

    # --- Persistence ---
    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(self.path.parent), suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(self._data, f, indent=2)
                f.write("\n")
            os.rename(tmp_path, str(self.path))
            os.utime(str(self.path))
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def to_dict(self) -> dict:
        return self._data
