"""Gate promotion orchestrator (D-141 / D-143 / GTE-04).

Composes:
  gate.evaluate_candidate  (Wave 3 / plan 05-06)
  gate.stats.bonferroni_correct + paired_wilcoxon_with_bootstrap (Wave 1 / plan 05-01)
  manifest.load_manifest   (Wave 2 / plan 05-02)

Outcomes
--------
  pass         -> status 'registered'     (returns True)
  fail         -> status 'keep'           (returns False)
  inconclusive -> status stays 'candidate' (returns False, D-150)

D-143 Two-Stage Gate composition:
  Stage A (Pareto on search cells) is enforced upstream — nominate() requires
  status='keep' before transitioning to 'candidate'. This module enforces Stage B.

D-151 calibrate mode: dry-run; runs evaluate + stats, writes archive log,
  but does NOT mutate node status or append to parent gate_log.

BCK-04 clean: no os.kill / os.killpg / Popen / .pid references.
Framework purity: no benchmark-specific references (D-148).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from automil.gate.evaluate import evaluate_candidate
from automil.gate.manifest import load_manifest
from automil.gate.stats import (
    bonferroni_correct,
    paired_wilcoxon_with_bootstrap,
)

if TYPE_CHECKING:
    from automil.backends.base import Backend
    from automil.graph import ExperimentGraph

logger = logging.getLogger(__name__)


def _apply_gate_outcome(graph, node_id: str, mutate) -> None:
    """Apply ONLY the gate's status transition, under a short lock (CR-2b).

    ``promote`` used to hold an ``ExperimentGraph`` loaded *before* the held-out
    evaluations and then call ``graph.save()`` afterwards — a whole-snapshot
    write of state that was minutes stale, silently discarding every daemon
    completion that landed while the evaluations ran.

    The lock genuinely cannot be held across the evaluations (they are backend
    jobs taking many minutes), so the transaction is split instead: evaluate
    unlocked, then re-acquire, RE-READ the node, and apply only the transition.
    The re-read matters — the node may have changed while we were gone, and
    writing the pre-evaluation copy back would reintroduce the same clobber at
    one node's granularity.
    """
    from automil.graph import locked_update

    with locked_update(
        str(graph.path),
        technique_map=getattr(graph, "_technique_map", None),
    ) as g:
        fresh = g.get_node(node_id)
        if fresh is None:
            logger.warning(
                "promote: %s vanished from the graph during held-out evaluation; "
                "gate outcome not applied", node_id,
            )
            return
        mutate(fresh)


def promote(
    candidate_node_id: str,
    backend: "Backend",
    graph: "ExperimentGraph",
    manifests_dir: Path,
    archive_dir: Path,
    K_floor: int = 2,
    calibrate: bool = False,
) -> bool:
    """Run Stage B held-out paired test on a nominated candidate.

    Args:
        candidate_node_id: graph node id with status='candidate'.
        backend: Backend instance for gate-eval job submission.
        graph: live ExperimentGraph; mutated in-place on pass/fail.
        manifests_dir: directory containing <parent_id>.gate_manifest.json files.
        archive_dir: root for per-candidate forensic logs
                     (<archive_dir>/<candidate_id>/gate_evaluation.jsonl).
        K_floor: minimum K_effective required to avoid inconclusive result (D-150).
        calibrate: D-151 dry-run mode — runs eval + stats, writes archive log,
                   but does NOT mutate node status or append to parent gate_log.

    Returns:
        True  if candidate promoted to 'registered'.
        False if reverted to 'keep' (fail) OR stayed 'candidate' (inconclusive).

    Raises:
        ValueError: candidate not found, status not 'candidate'.
        FileNotFoundError: no manifest registered for parent.
    """
    # ------------------------------------------------------------------ #
    # Validate inputs
    # ------------------------------------------------------------------ #
    node = graph.nodes.get(candidate_node_id)
    if node is None:
        raise ValueError(
            f"candidate {candidate_node_id!r} not found in graph"
        )
    if node.get("status") != "candidate":
        raise ValueError(
            f"Cannot promote {candidate_node_id!r}: status={node.get('status')!r}; "
            f"expected 'candidate'. Run `automil nominate {candidate_node_id}` first "
            f"(D-136 status flow: keep -> candidate -> registered)."
        )

    parent_id = node.get("parent_id")
    if not parent_id:
        raise ValueError(
            f"candidate {candidate_node_id!r} has no parent_id; cannot find manifest"
        )

    # ------------------------------------------------------------------ #
    # Load manifest (raises FileNotFoundError if absent)
    # ------------------------------------------------------------------ #
    manifest = load_manifest(parent_id, manifests_dir)

    # ------------------------------------------------------------------ #
    # Stage B: held-out evaluations
    # ------------------------------------------------------------------ #
    per_cell_results, skipped = evaluate_candidate(
        candidate_node_id, manifest, backend, graph,
    )

    K_effective = manifest.K - len(skipped)
    timestamp = datetime.now(timezone.utc).isoformat()

    # Per-candidate forensic log — written regardless of outcome (incl. calibrate)
    cand_log_dir = archive_dir / candidate_node_id
    cand_log_dir.mkdir(parents=True, exist_ok=True)
    cand_log_path = cand_log_dir / "gate_evaluation.jsonl"

    # ------------------------------------------------------------------ #
    # D-150 / Pitfall 1: K_effective floor — inconclusive path
    # ------------------------------------------------------------------ #
    if K_effective < K_floor:
        logger.warning(
            "promote: %s INCONCLUSIVE — K_effective=%d < K_floor=%d "
            "(skipped due to cap: %s)",
            candidate_node_id, K_effective, K_floor, skipped,
        )
        decision = {
            "event": "decision",
            "result": "inconclusive",
            "reason": "K_effective_below_floor",
            "K": manifest.K,
            "K_effective": K_effective,
            "K_floor": K_floor,
            "skipped_cells_due_to_cap": skipped,
            "per_cell_results": per_cell_results,
            "timestamp": timestamp,
        }
        _write_archive_log(cand_log_path, per_cell_results, decision)

        if not calibrate:
            _inconclusive_entry = {
                "event": "gate_result",
                "result": "inconclusive",
                "K_effective": K_effective,
                "K_floor": K_floor,
                "skipped": skipped,
                "timestamp": timestamp,
            }

            def _apply_inconclusive(n: dict) -> None:
                n.setdefault("history", []).append(dict(_inconclusive_entry))

            _apply_gate_outcome(graph, candidate_node_id, _apply_inconclusive)
            _apply_inconclusive(node)   # mirror onto the caller's view
        return False

    # ------------------------------------------------------------------ #
    # Bonferroni correction + statistical test
    # ------------------------------------------------------------------ #
    deltas = np.array([r["delta"] for r in per_cell_results])
    p_corrected = bonferroni_correct(manifest.p_threshold, K_effective)
    passes_test, p_value, (ci_low, ci_high), wins = paired_wilcoxon_with_bootstrap(deltas, p_corrected, manifest.bootstrap_reps)  # noqa: E501

    # Win iff: Wilcoxon p <= alpha_corrected AND CI lower > 0 AND >= K_effective wins
    gate_pass = bool(passes_test and (wins >= K_effective))

    decision = {
        "event": "decision",
        "result": "pass" if gate_pass else "fail",
        "p_value": p_value,
        "p_corrected": p_corrected,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "wins": wins,
        "K": manifest.K,
        "K_effective": K_effective,
        "skipped_cells_due_to_cap": skipped,
        "per_cell_results": per_cell_results,
        "win_definition": manifest.win_definition,
        "timestamp": timestamp,
    }
    _write_archive_log(cand_log_path, per_cell_results, decision)

    # ------------------------------------------------------------------ #
    # D-151 calibrate: informational only — no mutations, no parent log
    # ------------------------------------------------------------------ #
    if calibrate:
        logger.info(
            "promote --calibrate: %s would-%s (no status change, no parent_gate_log)",
            candidate_node_id, "PASS" if gate_pass else "FAIL",
        )
        return gate_pass

    # ------------------------------------------------------------------ #
    # Mutate status + history
    # ------------------------------------------------------------------ #
    _new_status = "registered" if gate_pass else "keep"
    _history_entry = {
        "event": "gate_result",
        "result": "pass" if gate_pass else "fail",
        "p_value": p_value,
        "p_corrected": p_corrected,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "wins": wins,
        "K_effective": K_effective,
        "skipped_cells_due_to_cap": skipped,
        "timestamp": timestamp,
    }

    # Per-parent gate log — append-only jsonl for promotion_rate computation (D-144)
    parent_log_path = manifests_dir / f"{parent_id}.gate_log.jsonl"
    parent_record = {
        "candidate_node_id": candidate_node_id,
        "parent_id": parent_id,
        "result": "pass" if gate_pass else "fail",
        "p_value": p_value,
        "K_effective": K_effective,
        "wins": wins,
        "timestamp": timestamp,
    }
    with parent_log_path.open("a") as fh:
        fh.write(json.dumps(parent_record) + "\n")

    # CR-2b: re-acquire briefly and apply ONLY the transition. The whole-snapshot
    # graph.save() that used to live here wrote state loaded before the held-out
    # evaluations began, discarding any daemon completion from that window.
    def _apply(n: dict) -> None:
        n["status"] = _new_status
        n.setdefault("history", []).append(_history_entry)

    _apply_gate_outcome(graph, candidate_node_id, _apply)
    # Mirror onto the caller's in-memory node so its view matches disk. The
    # locked copy is authoritative; this is only so a caller that already holds
    # the graph object does not read a node that appears not to have been gated.
    _apply(node)

    logger.info(
        "promote: %s -> %s (p=%.4f, ci=[%.4f, %.4f], wins=%d/%d)",
        candidate_node_id, node["status"], p_value, ci_low, ci_high, wins, K_effective,
    )
    return gate_pass


def _write_archive_log(
    path: Path,
    per_cell_results: list[dict],
    decision: dict,
) -> None:
    """Forensic per-candidate log: per-cell results first, then decision summary.

    Both lines are JSON objects on separate lines (JSONL format).
    Written regardless of calibrate mode (operator inspection).
    """
    with path.open("a") as fh:
        fh.write(
            json.dumps({"event": "per_cell_results", "per_cell_results": per_cell_results})
            + "\n"
        )
        fh.write(json.dumps(decision) + "\n")
