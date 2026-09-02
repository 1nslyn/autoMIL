"""The campaign's instruction surface is hash-pinned into every cell.

`toolset.json` pins the repo-root CLAUDE.md by sha256 (`ancestor_memory`),
`campaign_launch.preflight` refuses any drift, and the protocol carrying that
pin is bound into every cell's config, so the file cannot be rebuilt around a
change. This test keeps the tree honest: the CLAUDE.md that ships is the one
the 78 cells will launch against.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOOLSET = REPO / "benchmarks" / "campaigns" / "preprint_130" / "toolset.json"


def test_repo_root_claude_md_matches_the_pinned_hash():
    pinned = json.loads(TOOLSET.read_text())["ancestor_memory"]["CLAUDE.md"]
    observed = hashlib.sha256((REPO / "CLAUDE.md").read_text().encode()).hexdigest()
    assert observed == pinned, (
        "CLAUDE.md drifted from the campaign pin; every cell's launch preflight "
        "would refuse. Restore the pinned bytes (git checkout -- CLAUDE.md) "
        "rather than editing the protocol."
    )
