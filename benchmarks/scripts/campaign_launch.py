#!/usr/bin/env python
"""Launch the formal per-cell discovery session under the locked protocol.

``preflight`` verifies every launch precondition and prints the derived
plan without side effects.  ``launch-command`` additionally prints the
exact command line for inspection (runs nothing).  ``launch`` renders the
locked instruction surface into the cell and replaces this process with
the pinned runtime, cwd at the cell root.
"""
from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from autobench.campaign_launch import (
    CampaignLaunchError,
    launch,
    preflight,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "launch-command", "launch"))
    parser.add_argument("--cell-root", required=True)
    parser.add_argument(
        "--claude-bin", default="claude",
        help="Runtime executable to version-check and exec.",
    )
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    cell_root = Path(args.cell_root)
    if not cell_root.is_absolute():
        cell_root = (repo_root / cell_root).resolve()
    try:
        plan = preflight(cell_root, repo_root, claude_bin=args.claude_bin)
    except CampaignLaunchError as exc:
        raise SystemExit(f"campaign-launch refusal: {exc}")
    print(f"agent_protocol_sha256 {plan.agent_protocol_sha256}")
    print(f"cwd {plan.cwd}")
    print(f"instruction {plan.instruction_path}")
    for name, value in sorted(plan.env.items()):
        print(f"env {name}={value}")
    if args.action == "preflight":
        print("preflight ok — every launch precondition holds on this host")
        return
    print(f"command {shlex.join(plan.argv)}")
    if args.action == "launch-command":
        return
    launch(plan)


if __name__ == "__main__":
    main()
