"""Executable contracts for the agentic-campaign operator runbook."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNBOOK = REPO_ROOT / "docs/tutorials/run_agentic_campaign.md"


def test_runbook_paths_are_rooted_and_local_references_resolve() -> None:
    text = RUNBOOK.read_text()

    assert 'export REPO_ROOT="$(git rev-parse --show-toplevel)"' in text
    assert 'export CELL="$REPO_ROOT/benchmarks/campaigns/' in text
    assert "python benchmarks/scripts/" not in text
    assert 'python "$REPO_ROOT/benchmarks/scripts/' in text
    assert 'uv run automil --project "$CELL"' not in text
    assert 'uv run --project "$REPO_ROOT" automil --project "$CELL"' in text

    for raw_target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
        if raw_target.startswith(("http://", "https://", "#")):
            continue
        target = raw_target.split("#", 1)[0]
        assert (RUNBOOK.parent / target).resolve().exists(), raw_target

    shell_probe = """
set -eu
REPO_ROOT="$1"
CELL="$REPO_ROOT/benchmarks/campaigns/preprint_130/runtime/example-cell"
test -f "$REPO_ROOT/benchmarks/scripts/campaign_stage.py"
test -f "$REPO_ROOT/benchmarks/scripts/campaign_manifest.py"
case "$CELL" in "$REPO_ROOT"/*) ;; *) exit 1 ;; esac
"""
    subprocess.run(
        ["bash", "-c", shell_probe, "runbook-probe", str(REPO_ROOT)],
        check=True,
    )


def test_runbook_orders_session_end_before_promotion_and_attestation_after_winner() -> None:
    text = RUNBOOK.read_text()

    freeze = text.index("freeze-discovery --cell-root")
    session_end = text.index("/exit\n", freeze)
    promotion = text.index("materialize-promotion --cell-root", session_end)
    winner = text.index("select-winner --cell-root", promotion)
    attestation = text.index("finalize-agent-session", winner)

    assert freeze < session_end < promotion < winner < attestation
    assert "durable final active-time sample" in text
    assert "60 CLAM/ABMIL baseline reruns" not in text


def test_progress_is_the_only_runbook_launch_gate_source() -> None:
    text = RUNBOOK.read_text()
    progress = (
        REPO_ROOT / "benchmarks/campaigns/preprint_130/PROGRESS.md"
    ).read_text()

    assert "single source of truth for launch gates" in text
    assert "130/130 exact manifest coverage audit" in progress
    assert "Allocation request" in progress
    assert "P-SHARE-1" in progress
