#!/usr/bin/env python
"""Operate one frozen preprint-campaign cell without touching paper files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from autobench.campaign_stages import (
    CampaignStageError,
    certify_winner,
    freeze_discovery,
    freeze_promotion,
    load_stage_state,
    materialize_promotion,
    register_baseline,
    register_agent_session,
    run_native_baseline,
    select_winner,
)


def _cell_root(raw: str, repo_root: Path) -> Path:
    path = Path(raw)
    path = path.resolve() if path.is_absolute() else (repo_root / path).resolve()
    try:
        path.relative_to(repo_root)
    except ValueError as exc:
        raise CampaignStageError("--cell-root must live inside the git repository") from exc
    return path


def public_status(state: dict[str, Any]) -> dict[str, Any]:
    """Return a validation-only status surface; certification values stay separate."""
    winner = state.get("winner") or {}
    certification = state.get("certification") or {}
    return {
        "campaign_id": state["campaign_id"],
        "cell_id": state["cell_id"],
        "base_commit": state["base_commit"],
        "phase": state["phase"],
        "revision": state["revision"],
        "baseline_registered": state.get("baseline") is not None,
        "discovery_root_node_id": (
            (state.get("baseline") or {}).get("discovery_root_node_id")
        ),
        "discovery": {
            "attempts_charged": state["discovery"]["attempts_charged"],
            "attempt_budget": state["discovery"]["attempt_budget"],
            "complete_candidates": state["discovery"]["complete_candidates"],
            "promoted_candidates": len(
                state["discovery"]["promoted_candidates"]
            ),
            "frozen": state["discovery"]["frozen"],
        },
        "promotion": {
            "jobs": len(state["promotion"]["jobs"]),
            "materialized": state["promotion"]["materialized"],
            "frozen": state["promotion"]["frozen"],
            "eligible_candidates": len(
                state["promotion"].get("eligible_candidates", [])
            ),
        },
        "winner": ({
            "kind": winner.get("kind"),
            "candidate_id": winner.get("candidate_id"),
            "validation_mean": winner.get("validation_mean"),
            "lift_over_baseline": winner.get("lift_over_baseline"),
            "selection_sha256": winner.get("selection_sha256"),
        } if winner else None),
        "certification": ({
            "bundle": certification.get("bundle"),
            "bundle_sha256": certification.get("bundle_sha256"),
            "certified_at": certification.get("certified_at"),
        } if certification else None),
    }


def baseline_command(cell_root: Path) -> str:
    """Return the manifest-locked native five-fold incumbent command."""
    cell = json.loads((cell_root / "automil" / "campaign_cell.json").read_text())
    return str(cell["commands"]["baseline"])


def advance(cell_root: Path, repo_root: Path) -> dict[str, Any]:
    """Advance exactly one safe transition, never the held-out reveal."""
    state = load_stage_state(cell_root)
    phase = state["phase"]
    if phase == "discovery":
        return freeze_discovery(cell_root)
    if phase == "promotion-ready":
        return materialize_promotion(cell_root, repo_root=repo_root)
    if phase == "promotion":
        return freeze_promotion(cell_root)
    if phase == "selection-ready":
        return select_winner(cell_root)
    if phase in {"winner-frozen", "certified"}:
        return state
    raise CampaignStageError(f"unknown campaign phase {phase!r}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Operate one immutable 60→top-10→five-fold campaign cell.",
    )
    parser.add_argument(
        "action",
        choices=(
            "status", "register-baseline", "freeze-discovery",
            "materialize-promotion", "freeze-promotion", "select-winner",
            "certify", "baseline-command", "run-baseline", "advance",
            "register-agent-session",
        ),
    )
    parser.add_argument("--cell-root", required=True)
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="Physical GPU id used by run-baseline (default: 0).",
    )
    parser.add_argument(
        "--baseline-archive",
        help="Agent-facing native-baseline archive; required by register-baseline.",
    )
    parser.add_argument(
        "--agent-session",
        help="Runtime/resource attestation JSON; required by register-agent-session.",
    )
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    try:
        cell_root = _cell_root(args.cell_root, repo_root)
        if args.action == "status":
            state = load_stage_state(cell_root)
        elif args.action == "baseline-command":
            print(baseline_command(cell_root))
            return
        elif args.action == "register-baseline":
            if not args.baseline_archive:
                parser.error("register-baseline requires --baseline-archive")
            baseline = Path(args.baseline_archive)
            baseline = (
                baseline.resolve()
                if baseline.is_absolute() else (repo_root / baseline).resolve()
            )
            state = register_baseline(cell_root, baseline)
        elif args.action == "run-baseline":
            state = run_native_baseline(
                cell_root, repo_root=repo_root, gpu_id=args.gpu,
            )
        elif args.action == "register-agent-session":
            if not args.agent_session:
                parser.error("register-agent-session requires --agent-session")
            session_path = Path(args.agent_session)
            if not session_path.is_absolute():
                session_path = (repo_root / session_path).resolve()
            try:
                attestation = json.loads(session_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                parser.error(f"cannot read --agent-session: {exc}")
            register_agent_session(cell_root, attestation)
            state = load_stage_state(cell_root)
        elif args.action == "freeze-discovery":
            state = freeze_discovery(cell_root)
        elif args.action == "materialize-promotion":
            state = materialize_promotion(cell_root, repo_root=repo_root)
        elif args.action == "freeze-promotion":
            state = freeze_promotion(cell_root)
        elif args.action == "select-winner":
            state = select_winner(cell_root)
        elif args.action == "certify":
            state = certify_winner(cell_root)
            bundle = cell_root / state["certification"]["bundle"]
            print(bundle.read_text(), end="")
            return
        else:
            state = advance(cell_root, repo_root)
        print(json.dumps(public_status(state), indent=2, sort_keys=True))
        if args.action == "advance" and state["phase"] == "winner-frozen":
            print(
                "Winner is frozen. Held-out data remains sealed until all 130 "
                "cells are frozen by campaign_manifest.py freeze-selections."
            )
    except CampaignStageError as exc:
        parser.exit(2, f"campaign-stage error: {exc}\n")


if __name__ == "__main__":
    main()
