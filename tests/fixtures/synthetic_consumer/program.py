"""Synthetic mini-consumer for Phase 1 acceptance (REG-09 / D-50).

Deterministic torch-free training stub that:
  1. Reads the variant selection from automil/config.yaml.
  2. Resolves the variant via the registry (proves the registry path works).
  3. "Trains" by running the variant's forward on a fixed seed.
  4. Writes result.json with a deterministic primary_value.

The primary_value formula is `0.5 + 0.001 * sum(features)` over a fixed
feature vector — same input -> same output, byte-identical re-runs.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml


def main() -> int:
    project_root = Path(os.environ.get("PYTHONPATH", "").split(":")[0] or os.getcwd())
    # In the worktree, automil/ lives at <project_root>/automil/.
    here = Path.cwd()
    automil_dir = here / "automil"
    if not automil_dir.exists():
        # Walk up from cwd.
        p = here
        while p != p.parent:
            if (p / "automil" / "config.yaml").exists():
                automil_dir = p / "automil"
                break
            p = p.parent

    cfg = yaml.safe_load((automil_dir / "config.yaml").read_text()) or {}

    # Add automil/ to sys.path so `from <variants_pkg> import ...` works.
    sys.path.insert(0, str(automil_dir.parent))

    # Trigger registry refresh by importing the variants package.
    # We use the scanner directly (cheaper than re-running the CLI).
    from automil.registry._state import _clear_registry
    from automil.registry.scanner import scan_variants
    _clear_registry()
    scan_variants(automil_dir / "variants")

    # Resolve the active variant.
    from automil.registry import resolve_model

    model_cfg = cfg.get("model", {}) or {}
    parent = model_cfg.get("parent")
    name = model_cfg.get("variant")

    primary_value: float
    if name and parent:
        cls = resolve_model(parent, name)
        # The synthetic variant's forward returns the deterministic primary_value directly.
        instance = cls()
        # Fixed input vector: features = [1.0, 2.0, 3.0, 4.0]
        features = [1.0, 2.0, 3.0, 4.0]
        out = instance.forward(features=features)
        # Convention: synthetic variant returns a float that IS the primary_value.
        primary_value = float(out) if out is not None else 0.5
    else:
        # No variant configured: use the baseline primary_value.
        primary_value = 0.5

    # Write result.json. Val-firewall contract: `metrics` is validation-only
    # and must SUPPORT the recompute (CR-1b derives the primary_value from it
    # — mean or val_auc selector both reproduce it here); test metrics live
    # in the sealed `held_out` block, never in `metrics`.
    result = {
        "status": "completed",
        "metrics": {"val_auc": primary_value, "val_bacc": primary_value},
        "held_out": {"test_auc": 0.0, "test_bacc": 0.0},
        "primary_value": primary_value,
        "elapsed_seconds": 0.001,
        "peak_vram_mb": 0,
    }
    Path("result.json").write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
