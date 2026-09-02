#!/usr/bin/env python3
"""Classify every roster cell of the preprint discovery campaign.

Stdlib only: this file is read by the SLURM launchers on a running cluster and
must import nothing from the workspace packages.

Classes (mutually exclusive, in evaluation order):

- ``done``        phase ``certified``, or ``winner-frozen`` with a finalized session
- ``claimed``     a ``.discovery_claim`` whose holder is a live SLURM job (or is
                  this job); an UNREADABLE state file also lands here, because a
                  cell that another member is driving right now is the one case
                  where a read can fail mid-write, and refusing to touch it is
                  the safe answer
- ``finishable``  session ended cleanly after the full 30-attempt budget; the
                  finish ladder is idempotent and may be resumed by anyone
- ``stranded``    any other session evidence (live elsewhere, or dead mid-run)
- ``blocked``     reproduction gate recorded ``verdict: fail``
- ``pending``     clean, unclaimed, gate-eligible

Claims are never unlinked here: ``take_claim`` in the launcher library is the
only place allowed to replace a claim (compare-and-swap under flock).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

DISCOVERY_ATTEMPTS = 30
CLASSES = ("pending", "done", "stranded", "blocked", "claimed", "finishable")


def roster_cells(runtime: Path, roster: Mapping[str, object]) -> list[str]:
    cohorts = set(roster["cohorts"])
    cells = sorted(
        entry.name for entry in runtime.iterdir()
        if entry.is_dir() and entry.name.split("__")[0] in cohorts
    )
    if len(cells) != int(roster["cells"]):
        raise SystemExit(
            f"runtime holds {len(cells)} roster cells, roster declares {roster['cells']}"
        )
    return cells


def live_job_ids() -> set[str] | None:
    """Cluster-wide job ids, or None when squeue cannot be trusted (fail safe)."""
    try:
        output = subprocess.run(
            ["squeue", "-h", "-o", "%i"], capture_output=True, text=True,
            timeout=30, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return set(output.split())


def _read_json(path: Path) -> Mapping[str, object] | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _session_ended(journal: Path) -> bool:
    try:
        lines = journal.read_text().splitlines()
    except OSError:
        return False
    for line in lines:
        if not line.strip().startswith("{"):
            continue
        try:
            if json.loads(line).get("event") == "session_end":
                return True
        except ValueError:
            continue
    return False


def _has_session_evidence(root: Path) -> bool:
    session = root / "agent_session.json"
    journal = root / "automil" / ".activity.jsonl"
    if session.exists():
        return True
    try:
        return journal.is_file() and bool(journal.read_text().strip())
    except OSError:
        return True  # unreadable evidence is still evidence


def classify_cell(
    root: Path, alive: set[str] | None, job_id: str,
) -> tuple[str, str]:
    """Return ``(class, note)`` for one cell root."""
    state = _read_json(root / "campaign_state.json")
    if state is None:
        return "claimed", "campaign_state.json unreadable (in use by another member?)"
    phase = state.get("phase")
    session = _read_json(root / "agent_session.json")
    session_status = session.get("status") if session else None
    if phase == "certified" or (phase == "winner-frozen" and session_status == "finalized"):
        return "done", ""
    claim = root / ".discovery_claim"
    if claim.is_file():
        try:
            holder = claim.read_text().strip()
        except OSError:
            holder = ""
        stale = alive is not None and holder and holder not in alive and holder != job_id
        if not stale:
            return "claimed", f"held by {holder or '?'}"
    if _has_session_evidence(root):
        charged = (state.get("discovery") or {}).get("attempts_charged")
        if _session_ended(root / "automil" / ".activity.jsonl") and charged == DISCOVERY_ATTEMPTS:
            return "finishable", ""
        return "stranded", "session evidence without a finished ladder"
    reproduction = state.get("baseline_reproduction") or {}
    if reproduction.get("mode") == "gate" and reproduction.get("verdict") == "fail":
        return "blocked", "reproduction gate failed"
    if state.get("baseline") is None:
        raise SystemExit(f"{root.name}: no registered baseline — discovery cannot start")
    return "pending", ""


def scan(runtime: Path, roster_path: Path, job_id: str) -> dict[str, object]:
    roster = json.loads(roster_path.read_text())
    alive = live_job_ids()
    classes: dict[str, list[str]] = {name: [] for name in CLASSES}
    notes: dict[str, str] = {}
    for cell in roster_cells(runtime, roster):
        kind, note = classify_cell(runtime / cell, alive, job_id)
        classes[kind].append(cell)
        if note:
            notes[cell] = note
    return {**classes, "notes": notes, "squeue_ok": alive is not None}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--roster", default=Path("/dev/null"), type=Path)
    parser.add_argument("--job-id", default="manual")
    parser.add_argument("--class", dest="only", choices=CLASSES,
                        help="print just this class, one cell per line")
    parser.add_argument("--session-ended", metavar="CELL",
                        help="exit 0 if CELL's activity journal records session_end, else 1")
    args = parser.parse_args(argv)
    if args.session_ended:
        journal = args.runtime / args.session_ended / "automil" / ".activity.jsonl"
        return 0 if _session_ended(journal) else 1
    if not args.runtime.is_dir() or not args.roster.is_file():
        print("campaign_scan: runtime or roster missing", file=sys.stderr)
        return 2
    result = scan(args.runtime, args.roster, args.job_id)
    if args.only:
        print("\n".join(result[args.only]))
        return 0
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
