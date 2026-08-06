"""AutoBench — MIL benchmark suite for computational pathology."""

import os
from pathlib import Path

# Root of the benchmarks/ directory. Normally derived from this file's path,
# but autoMIL overrides it via AUTOBENCH_ROOT so experiments running inside
# a git worktree pick up the worktree's lib/ and scripts/ instead of the
# editable-installed main repo copies. Without this override, overlays that
# modify files under benchmarks/lib/ or benchmarks/src/autobench/ are silently
# ignored because the uv workspace pins imports to the main-repo source.
_override = os.environ.get("AUTOBENCH_ROOT")
if _override:
    BENCHMARKS_ROOT = Path(_override).resolve()
else:
    BENCHMARKS_ROOT = Path(__file__).resolve().parents[2]

# External library directory (benchmarks/lib/)
LIB_ROOT = BENCHMARKS_ROOT / "lib"
