"""Experiment graph: directed tree tracking for multi-branch exploration.

Provides atomic read/write to graph.json. Concurrent writers (daemon +
CLI) coordinate through an advisory ``flock`` on a sidecar ``.lock``
file; see ``locked_update``.

Immutability (L-8a, audit 2026-07-23): this module's own methods
(``add_executed``, ``promote``, ``mark_failed``, ``reconcile``, ...)
mutate node dicts stored in ``self._data`` in place, field by field —
they do NOT rebuild and reassign a fresh dict per update. This is a
deliberate, pragmatic choice, not an oversight: ``self._data`` is
single-owner for the lifetime of one ``locked_update`` transaction (the
flock above serializes every writer), and it is serialized wholesale on
``save()``, so there is no aliasing surface between transactions — every
``locked_update`` call constructs a brand-new ``ExperimentGraph`` from a
fresh ``json.loads`` of the file, so no Python object outlives its lock.
Converting every one of those in-place field assignments to copy-on-write
would be a sweeping rewrite of this module for no correctness gain, so it
is deliberately NOT done.

The one nested structure that genuinely IS reachable from two writers is
a node's ``metadata`` sub-dict: ``gate/evaluate.py`` creates a gate-eval
child node via a SHALLOW ``dict(node)`` copy, which leaves the child's
``metadata`` key aliased to the same dict object as its source. A caller
that then mutates ``gnode["metadata"]`` in place (``.setdefault(...)
.update(...)``) could silently corrupt whichever node it is aliased
with. ``merged_metadata`` below is the copy-on-write fix for exactly that
structure, used by ``terminal_writer``, ``cli/cancel``, ``cli/propose``,
``cli/reconcile``, and the daemon's cap-refusal path — the sites where a
node read via ``get_node()`` needs to add or change ``metadata`` keys.
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

from automil.scoring import DEFAULT_FORMULA as _DEFAULT_SCORING_FORMULA

logger = logging.getLogger(__name__)


def node_cell_id(node: dict | None) -> str | None:
    """Return the budget-cell id a graph node belongs to, or ``None`` (CELL-1).

    Two shapes are accepted because two writers exist: ``automil submit`` stamps
    a top-level ``cell_id`` (parallel to ``config_hash``), while gate-eval
    children are created with the id under ``metadata`` (``gate/evaluate.py``).

    ``None`` is the legacy answer — nodes created before cells existed carry no
    identity and must simply never match a lookup.
    """
    if not isinstance(node, dict):
        return None
    cell_id = node.get("cell_id")
    if not cell_id:
        meta = node.get("metadata")
        cell_id = meta.get("cell_id") if isinstance(meta, dict) else None
    return cell_id if isinstance(cell_id, str) and cell_id else None


def merged_metadata(node: dict | None, updates: dict) -> dict:
    """Copy-on-write merge into a node's ``metadata`` sub-dict (L-8a).

    Several call sites (``terminal_writer``, ``cli/cancel``, ``cli/propose``,
    ``cli/reconcile``, the daemon's cap-refusal path) read a node via
    ``get_node()`` and then need to add or change a few ``metadata`` keys.
    The naive way — ``gnode.setdefault("metadata", {}).update(updates)`` or
    ``gnode.setdefault("metadata", {})[k] = v`` — mutates whatever dict
    object is already stored at ``node["metadata"]``, in place.

    That is reachable from two writers: ``gate/evaluate.py`` creates a
    gate-eval child node via a SHALLOW copy of a node dict (``dict(node)``),
    which leaves the child's ``metadata`` key pointing at the exact same
    dict object as its source node's. An in-place mutation through either
    alias would silently corrupt the other — a plain dict has no
    copy-on-write semantics of its own.

    Callers use this as ``gnode["metadata"] = merged_metadata(gnode,
    {...})``: the OUTER node dict is still updated by direct key assignment
    (matching every other field mutation in this codebase — self._data is
    single-owner per flock-guarded ``locked_update`` transaction and
    serialized wholesale on save, so that part has no aliasing surface and
    converting it too would be a much larger, unrelated rewrite). Only this
    specific NESTED, cross-writer-reachable structure is made copy-on-write.

    Tolerates ``node=None`` and a non-dict ``metadata`` value (legacy/corrupt
    data) by treating the base as empty, matching ``node_cell_id``'s
    defensiveness above.
    """
    base = (node or {}).get("metadata")
    if not isinstance(base, dict):
        base = {}
    return {**base, **updates}


def _accept_margin(meta: dict | None) -> float:
    """Predeclared Ladder keep-margin δ from ``meta.scoring.accept_margin``.

    δ=0.0 (the default) reproduces plain composite dominance. A δ>0 requires a
    child to beat its parent's validation composite by more than the margin
    before it is kept — a Ladder-style gate against promoting within-noise
    improvements over a long agentic search.
    """
    try:
        raw = ((meta or {}).get("scoring") or {}).get("accept_margin", 0.0)
        return max(0.0, float(raw or 0.0))   # clamp: a negative δ would invert the gate
    except (TypeError, ValueError, AttributeError):
        return 0.0


def _accept(child_composite: float, parent_composite: float, margin: float = 0.0) -> bool:
    """Keep a child iff its composite beats the parent's by more than ``margin``.

    The single keep/discard predicate, shared by every decision site (the live
    terminal writer, descendant re-evaluation, and both reconcile paths) so the
    Ladder margin is applied uniformly. margin=0.0 → strict dominance.
    """
    return child_composite > parent_composite + margin


#: One SE. The point of CR-4 is that the bar is the measured noise; a default of
#: 0 would ship the feature switched off, which is how it got missed the first time.
DEFAULT_SE_MULTIPLIER = 1.0


def node_composite_se(node: dict | None) -> float | None:
    """Cross-fold SE of a node's composite, or ``None`` if it was never measured.

    ``None`` covers three real cases and they must not be conflated with zero:
    a legacy node written before CR-4, a partial run with fewer than two finite
    folds (H-8 / M-15), and a corrupt or negative value. A caller seeing ``None``
    falls back to the predeclared δ; a caller seeing 0.0 is being told the folds
    genuinely agreed.
    """
    if not isinstance(node, dict):
        return None
    raw = node.get("composite_se")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    val = float(raw)
    if not math.isfinite(val) or val < 0:
        return None
    return val


def _se_multiplier(meta: dict | None) -> float:
    """How many SEs a child must clear, from ``meta.scoring.se_multiplier``.

    Clamped at 0: a negative multiplier would turn the noise floor into a
    discount, letting a noisy parent be beaten by *less* than nothing.
    """
    try:
        raw = ((meta or {}).get("scoring") or {}).get("se_multiplier", DEFAULT_SE_MULTIPLIER)
        if raw is None:
            return DEFAULT_SE_MULTIPLIER
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return DEFAULT_SE_MULTIPLIER


def effective_accept_margin(meta: dict | None, parent_node: dict | None) -> float:
    """The margin actually applied: ``max(predeclared δ, k × parent SE)`` (CR-4).

    Two invariants, both load-bearing:

    **Monotone.** The measured noise can only ever RAISE the bar. A campaign that
    predeclared δ=0.05 must not silently drop to 0.01 because one parent happened
    to have a tight CV — the predeclared value is a pre-registration commitment,
    not an opening bid.

    **The bar belongs to the incumbent.** It is derived from the PARENT's SE, not
    the child's. If it came from the child's, then taking the argmax over ~60
    screened candidates would simultaneously be taking the argmin over their
    margins: the search would be selecting on the gate itself.

    This is a conservative single-arm screen, **not a test**. Parent and child
    share folds, so the SE of their difference is not the SE of either one. The
    honest paired inference happens at the Stage-B gate (``gate/stats.py``:
    paired Wilcoxon + BCa on per-cell deltas). Do not report this margin as
    significance.
    """
    delta = _accept_margin(meta)
    se = node_composite_se(parent_node)
    if se is None:
        return delta
    return max(delta, _se_multiplier(meta) * se)


def _config_accept_margin(graph_path) -> float | None:
    """Best-effort read of ``scoring.accept_margin`` from the sibling config.yaml.

    Lets an operator predeclare the Ladder keep-margin δ per-dataset in
    ``automil/config.yaml`` (``scoring.accept_margin``); a fresh graph seeds its
    ``meta.scoring.accept_margin`` from it. Returns None when there is no config,
    the key is absent, or it cannot be parsed as a number (callers fall back to
    0.0). The graph stays config-agnostic everywhere else.
    """
    config_path = Path(graph_path).parent / "config.yaml"
    if not config_path.exists():
        return None
    try:
        import yaml
        cfg = yaml.safe_load(config_path.read_text()) or {}
        raw = (cfg.get("scoring") or {}).get("accept_margin")
        return max(0.0, float(raw)) if raw is not None else None   # clamp negative δ
    except Exception as exc:  # noqa: BLE001 — best-effort seed; bad config → default
        logger.warning("Could not read scoring.accept_margin from %s: %s", config_path, exc)
        return None


def _config_se_multiplier(graph_path) -> float | None:
    """Best-effort read of ``scoring.se_multiplier`` from the sibling config.yaml (CR-4).

    Predeclared per-dataset alongside δ, and clamped at 0 for the same reason
    ``_se_multiplier`` clamps: a negative multiplier would turn the measured
    noise floor into a discount. Returns None when there is no config or the key
    is absent, so the caller falls back to ``DEFAULT_SE_MULTIPLIER`` (one SE)
    rather than to 0, which would ship the gate switched off.
    """
    config_path = Path(graph_path).parent / "config.yaml"
    if not config_path.exists():
        return None
    try:
        import yaml
        cfg = yaml.safe_load(config_path.read_text()) or {}
        raw = (cfg.get("scoring") or {}).get("se_multiplier")
        return max(0.0, float(raw)) if raw is not None else None
    except Exception as exc:  # noqa: BLE001 — best-effort seed; bad config → default
        logger.warning("Could not read scoring.se_multiplier from %s: %s", config_path, exc)
        return None


def _config_scoring_formula(graph_path) -> str | None:
    """Best-effort read of ``scoring.formula`` from the sibling config.yaml (CR-1b).

    Lets an operator predeclare the composite reducer per-dataset. Returns None
    when there is no config or the key is absent (callers fall back to the
    framework default). The graph stays config-agnostic everywhere else.
    """
    config_path = Path(graph_path).parent / "config.yaml"
    if not config_path.exists():
        return None
    try:
        import yaml
        cfg = yaml.safe_load(config_path.read_text()) or {}
        raw = (cfg.get("scoring") or {}).get("formula")
        return str(raw) if raw else None
    except Exception as exc:  # noqa: BLE001 — best-effort seed; bad config → default
        logger.warning("Could not read scoring.formula from %s: %s", config_path, exc)
        return None


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
        # Capture the on-disk schema_version BEFORE setdefault fills in the new
        # default of 2.  Absent key → 1 (legacy); present → whatever was stored.
        # Used by the DBT-01 migration gate below.
        _on_disk_schema_version = self._data.get("schema_version", 1)
        # Normalize: fill in missing top-level / meta keys with defaults
        # so legacy schemas and fresh-init paths both work. When loading
        # an existing file that's missing keys, log a warning — partial-
        # write corruption silently filled in with defaults would mask
        # real data loss, and operators need a paper trail.
        # Ladder keep-margin δ: a fresh (or legacy) graph seeds accept_margin from
        # the sibling config.yaml so an operator can predeclare it per-dataset.
        # Once persisted in graph.json, the stored value wins over config.
        _meta = self._data.get("meta")
        _has_margin = (
            isinstance(_meta, dict)
            and isinstance(_meta.get("scoring"), dict)
            and "accept_margin" in _meta["scoring"]
        )
        _cfg_margin = None if _has_margin else _config_accept_margin(self.path)
        _default_margin = _cfg_margin if _cfg_margin is not None else 0.0
        # CR-4: the SE multiplier is predeclared alongside δ and frozen the same
        # way — once in graph.json the stored value wins, so a campaign cannot
        # loosen its own gate halfway through.
        _has_mult = (
            isinstance(_meta, dict)
            and isinstance(_meta.get("scoring"), dict)
            and "se_multiplier" in _meta["scoring"]
        )
        _cfg_mult = None if _has_mult else _config_se_multiplier(self.path)
        _default_mult = _cfg_mult if _cfg_mult is not None else DEFAULT_SE_MULTIPLIER
        # CR-1b: the composite reducer, predeclarable per-dataset in config.yaml.
        _cfg_formula = _config_scoring_formula(self.path)
        _default_formula = _cfg_formula if _cfg_formula else _DEFAULT_SCORING_FORMULA
        defaults = {
            "schema_version": 2,
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
                    "accept_margin": _default_margin,
                    "se_multiplier": _default_mult,
                    "formula": _default_formula,
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
        # Backfill accept_margin into a pre-existing scoring block (legacy graphs
        # that predate the Ladder gate) so a predeclared config δ still applies.
        if not isinstance(self._data["meta"].get("scoring"), dict):
            self._data["meta"]["scoring"] = dict(defaults["meta"]["scoring"])
        # M-1 (audit 2026-07-23): backfill EVERY scoring key (not only
        # accept_margin) so a legacy / hand-edited scoring block missing
        # exploration_weight or novelty_weight cannot KeyError in
        # recalculate_scores() and silently turn every reconcile() into a no-op.
        for _sk, _sv in defaults["meta"]["scoring"].items():
            self._data["meta"]["scoring"].setdefault(_sk, _sv)
        self._data["meta"]["scoring"].setdefault("accept_margin", _default_margin)
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
        # DBT-01: migrate pre-D-200 nodes (flat metric keys) to metrics-dict layout on read.
        # Gate: on-disk schema_version < 2 AND node lacks "metrics" — idempotent on post-D-200.
        # Uses _on_disk_schema_version (captured before setdefault filled in the new default of 2)
        # so that graphs written without a schema_version key are correctly treated as legacy.
        # Migration is in-memory only; caller decides when to save.
        if _on_disk_schema_version < 2:
            _LEGACY_METRIC_KEYS = ("val_auc", "val_bacc", "test_auc", "test_bacc")
            _migrated = 0
            for _node in self._data.get("nodes", {}).values():
                if "metrics" not in _node:
                    _node["metrics"] = {
                        k: _node.get(k, 0.0) for k in _LEGACY_METRIC_KEYS
                    }
                    _migrated += 1
            self._data["schema_version"] = 2
            if loaded_from_disk and _migrated > 0:
                logger.warning(
                    "graph.json at %s: legacy schema (pre-D-200) detected; "
                    "migrated %d node(s) to metrics-dict layout on read. "
                    "Re-save to persist the migration.",
                    self.path,
                    _migrated,
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
        visited: set[str] = set()  # M-4 (audit 2026-07-23): guard a parent_id cycle
        while current:
            if current in visited:
                logger.warning("lineage: parent_id cycle detected at %s; truncating", current)
                break
            visited.add(current)
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

    # --- Budget-cell membership (CELL-1) ---
    def nodes_in_cell(self, cell_id: str) -> list[dict]:
        """Return the nodes belonging to budget cell ``cell_id``, ordered by id.

        The join between the experiment tree (``graph.json``) and the budget
        cells (``automil/cells/<cell_id>.json``). Legacy nodes carry no cell
        identity and never match — including for a falsy ``cell_id`` query,
        which must not sweep every untagged node in.
        """
        if not cell_id:
            return []
        return [
            node for _, node in sorted(self.nodes.items())
            if node_cell_id(node) == cell_id
        ]

    def count_in_cell(self, cell_id: str, *, executed_only: bool = False) -> int:
        """Count the nodes in a budget cell.

        ``executed_only=True`` counts evaluations (proposals are not evaluations),
        which is the graph-side cross-check for ``Cell.consumed_evals``. The cell
        counter remains authoritative — it also bills nodes the graph never got
        (e.g. a spec launched by a non-CLI submission path).
        """
        nodes = self.nodes_in_cell(cell_id)
        if executed_only:
            nodes = [n for n in nodes if n.get("type") == "executed"]
        return len(nodes)

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
        # CR-4: the cross-fold SE is a framework-owned scalar like `composite`, so
        # it is lifted to the top level rather than left inside the opaque consumer
        # metrics dict — where CR-1b's mean-of-metrics reducer would average it in.
        composite_se = node_composite_se({"composite_se": metrics.get("composite_se")})
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
            "composite_se": composite_se,
            "global_delta": metrics.get("global_delta", metrics.get("delta", 0.0)),
            "parent_delta": composite - parent_composite,
            # Consumer metrics stored as opaque dict (D-200 / DEC-04).
            "metrics": {k: v for k, v in metrics.items() if k != "composite_se"},
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

        # H-6 (audit 2026-07-23): only a keep node may become best (this path has
        # no descendant re-evaluation, so a keep-gated inline update suffices and
        # avoids an O(N) recompute per insert).
        if status == "keep" and composite > self.meta["best_composite"]:
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
        # CR-4: keep the measured noise attached when a node is promoted from a
        # reconcile artifact, or the recovered incumbent would set its children's
        # bar from the bare predeclared margin instead of its own CV spread.
        _se = node_composite_se({"composite_se": metrics.get("composite_se")})
        if _se is not None or "composite_se" not in node:
            node["composite_se"] = _se
        node["global_delta"] = metrics.get("global_delta", metrics.get("delta", 0.0))
        node["parent_delta"] = composite - parent_composite
        # D-200: store consumer metrics as opaque dict. `composite_se` is a
        # framework-owned scalar (lifted above), so it is excluded here for the
        # same reason as in add_executed: CR-1b recomputes the composite as the
        # mean of `metrics`, and an SE averaged in would corrupt it.
        node["metrics"] = {k: v for k, v in metrics.items() if k != "composite_se"}
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

        pid = node.get("parent_id")
        if pid and pid in self.nodes:
            self.nodes[pid]["child_count"] = len([
                n for n in self.nodes.values()
                if n.get("parent_id") == pid and n["type"] == "executed"
            ])

        self._update_technique_stats(node.get("techniques", []),
                                     composite - parent_composite)

        self._reevaluate_descendants(node_id)
        # H-6 (audit 2026-07-23): recompute best from keep nodes only, AFTER
        # _reevaluate_descendants may have flipped nodes to discard. Replaces the
        # status-agnostic inline update that could leave best on a discarded node.
        self.recompute_best()

    def _reevaluate_descendants(self, root_id: str) -> None:
        """Recompute keep/discard for executed descendants of root_id.

        Children can be promoted before their parent completes, in which case
        parent metrics default to 0 and the Pareto check spuriously yields
        'keep'. Re-run the check now that root_id has real metrics.
        """
        stack = [root_id]
        visited: set[str] = set()  # M-4 (audit 2026-07-23): guard a parent/child cycle
        while stack:
            pid = stack.pop()
            if pid in visited:
                continue
            visited.add(pid)
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
                # D-200 Option B: composite-only dominance, gated by the Ladder
                # keep-margin (δ=0.0 → strict dominance). The composite is the
                # consumer-computed validation selection signal (val-firewall).
                keep = _accept(c_comp, p_comp, effective_accept_margin(self.meta, parent))
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
                    composite_se = node_composite_se(completion)   # CR-4
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
                        # D-200 Option B: composite-only dominance + Ladder margin.
                        keep = _accept(composite, p_comp,
                                       effective_accept_margin(self.meta, parent_node))
                        graph_status = "keep" if keep else "discard"
                    else:
                        graph_status = "keep" if composite > 0 else "discard"  # root: no parent, δ N/A
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
                metrics["composite_se"] = composite_se   # CR-4: lifted by add_executed

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
                    # H-6 (audit 2026-07-23): only a keep node may become best
                    # (keep-gated inline update preserves the D-14 no-full-recompute
                    # contract of default reconcile while never selecting a discard).
                    if graph_status == "keep" and metrics["composite"] > self.meta["best_composite"]:
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
                        composite_se = node_composite_se(result)   # CR-4
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
                                # D-200 Option B: composite-only dominance + Ladder margin.
                                keep = _accept(composite, p_comp,
                                              effective_accept_margin(self.meta, parent))
                                status = "keep" if keep else "discard"
                            else:
                                status = "keep" if composite > 0 else "discard"  # root: no parent, δ N/A
                        else:
                            status = raw_status

                        techniques = gm.get("techniques", [])
                        self.nodes[node_id_r] = {
                            "id": node_id_r, "parent_id": parent_id,
                            "type": "executed", "status": status,
                            "description": spec.get("description", f"recovered {node_id_r}"),
                            "techniques": techniques, "composite": composite,
                            "composite_se": composite_se,   # CR-4
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
                        # H-6 (audit 2026-07-23): only a keep node may become best.
                        if status == "keep" and composite > self.meta.get("best_composite", 0):
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
                # CR-1a / M-3 (audit 2026-07-23): allow_nan=False guarantees
                # graph.json is standards-valid JSON. A NaN/Infinity would
                # otherwise serialize as a bare token that breaks every non-Python
                # reader (viz SSE JSON.parse, jq, serde). Non-finite values are
                # rejected upstream at result ingestion (validate_result), so this
                # raises only on a genuine internal invariant violation — loudly,
                # instead of persisting silent corruption.
                json.dump(self._data, f, indent=2, allow_nan=False)
                f.write("\n")
            os.rename(tmp_path, str(self.path))
            os.utime(str(self.path))
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def to_dict(self) -> dict:
        return self._data
