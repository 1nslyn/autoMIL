"""APL-02 translation layer: apply automil variant selection to ExperimentConfig.

This module is the consumer-side bridge between the autoMIL variant registry
and the autobench CLAM pipeline.  It reads the active variant from
``automil/config.yaml`` (written by ``automil apply``) and patches
``ExperimentConfig.model`` / ``ExperimentConfig.train`` in-place before
``_make_clam_args`` is called.

Design (D-03 from 10-CONTEXT.md):
  - Lives in autobench (consumer), NOT in src/automil/ (framework stays generic).
  - Imports FROM automil.registry (autobench → automil direction is allowed).
  - Does NOT edit lib/ — no loop opening.

Status: STUB — raises NotImplementedError until Plan 10-03 implements the body.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from automil.registry.scanner import scan_variants  # noqa: F401 — validates import chain
from automil.registry._state import MODEL_VARIANTS   # noqa: F401 — validates import chain
from automil.registry.spec import VariantSpec        # noqa: F401 — validates import chain

__all__ = ["apply_model_variant_to_exp_cfg"]


def apply_model_variant_to_exp_cfg(exp_cfg, automil_dir: Path) -> None:
    """Mutate *exp_cfg*.model (and optionally *exp_cfg*.train) in-place.

    Reads ``automil/config.yaml`` for the active ``model.variant`` selection,
    looks up the registered variant class in ``MODEL_VARIANTS`` (after scanning
    the variants directory), and applies the class-level ``CLAM_ARGS`` dict to
    the appropriate ``ExperimentConfig`` fields.

    No-op when:
      - ``automil_dir / "config.yaml"`` does not exist, or
      - the config has no ``model.variant`` key, or
      - the variant class has an empty / missing ``CLAM_ARGS`` dict.

    Raises:
      ValueError: if a variant name is specified but not found in the registry
        after scanning the variants directory.

    Args:
      exp_cfg: An ``ExperimentConfig`` instance whose ``.model`` and ``.train``
        fields will be patched in-place (Pitfall 6: mutable dataclass, safe).
      automil_dir: Path to the ``automil/`` directory (contains ``config.yaml``
        and ``variants/`` subdirectory).
    """
    raise NotImplementedError("APL-02: not yet implemented — implement in Plan 10-03")
