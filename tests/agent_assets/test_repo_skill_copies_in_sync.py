"""A3 (claims-alignment): the repo's checked-in skill copies cannot drift.

This repository is both the framework source and a live agent workspace:
Claude Code loads `.claude/skills/` and other runtimes read `.agents/skills/`,
while the canonical content lives in `src/automil/agent_assets/_shared/`.
The `.claude` copy silently pre-dated `registry.mode` and instructed agents to
propose architecture experiments inside an architecture-preserving campaign —
the drift class this test kills.

The expected bytes are the per-runtime *render* (`merge_skill`), so if a
runtime overlay is ever added for these skills, the checked-in copies must be
re-rendered, not hand-edited.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from automil.agent_assets._overlay import merge_skill

REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_SKILLS = REPO_ROOT / "src" / "automil" / "agent_assets" / "_shared" / "skills"
AGENT_ASSETS = REPO_ROOT / "src" / "automil" / "agent_assets"

#: (checked-in copy dir, runtime whose overlay applies to that dir)
COPY_DIRS = (
    (REPO_ROOT / ".claude" / "skills", "claude"),
    (REPO_ROOT / ".agents" / "skills", "claude"),
)
SKILLS = ("automil", "automil-setup")


def _canonical_render(runtime: str, skill: str) -> str:
    shared = SHARED_SKILLS / skill / "SKILL.md"
    overlay = AGENT_ASSETS / runtime / "skills" / skill / "SKILL.md"
    return merge_skill(runtime, shared, overlay if overlay.exists() else None)


@pytest.mark.parametrize("copy_dir,runtime", COPY_DIRS, ids=[".claude", ".agents"])
@pytest.mark.parametrize("skill", SKILLS)
def test_checked_in_copy_matches_canonical_render(copy_dir, runtime, skill):
    copy_path = copy_dir / skill / "SKILL.md"
    assert copy_path.exists(), f"missing checked-in copy: {copy_path}"
    expected = _canonical_render(runtime, skill)
    actual = copy_path.read_text(encoding="utf-8")
    assert actual == expected, (
        f"{copy_path.relative_to(REPO_ROOT)} has drifted from the canonical "
        f"render of agent_assets/_shared/skills/{skill}/SKILL.md — sync it "
        f"(cp from _shared, or re-render if an overlay was added). Agents in "
        f"this repo read the checked-in copy, not the canonical source."
    )
