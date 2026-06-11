"""APL-02 translation layer: apply automil variant selection to ExperimentConfig.

This module is the consumer-side bridge between the autoMIL variant registry
and the autobench CLAM pipeline.  It reads the active variant selection from
``automil/applied_variant.json`` (written by ``automil apply`` and propagated
into the worktree by ``apply_overlay``) and patches ``ExperimentConfig.model``
/ ``ExperimentConfig.train`` in-place before ``_make_clam_args`` is called.

Design (D-03 from 10-CONTEXT.md):
  - Lives in autobench (consumer), NOT in src/automil/ (framework stays generic).
  - Imports FROM automil.registry (autobench → automil direction is allowed).
  - Does NOT edit lib/ — no loop opening.

Read path priority (APL-02 A1-closure):
  1. PRIMARY:  ``automil_dir / "applied_variant.json"``  — written by apply.py
               Task 2 of plan 10-02 and propagated into the worktree by
               ``apply_overlay`` from ``archive/<node_id>/``.  This is the only
               path that works in real orchestrated runs (config.yaml is
               gitignored and never in the worktree).
  2. FALLBACK: ``automil_dir / "config.yaml"`` reading ``model.variant`` key.
               DEPRECATED — supported only for pre-10-02 experiments.
               New tests MUST NOT rely on this path.
  3. LAST RESORT: ``os.environ["AUTOMIL_VARIANT_MODEL"]`` environment variable.

Security: variant_name is validated (Path(variant_name).name == variant_name)
against path traversal before any filesystem access (T-10-05 mitigation).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import yaml

from automil.registry.scanner import scan_variants
from automil.registry._state import MODEL_VARIANTS

__all__ = ["apply_model_variant_to_exp_cfg"]

logger = logging.getLogger(__name__)


def apply_model_variant_to_exp_cfg(exp_cfg, automil_dir: Path) -> None:
    """Mutate *exp_cfg*.model (and optionally *exp_cfg*.train) in-place.

    Reads the active ``model.variant`` selection from ``applied_variant.json``
    (PRIMARY path), ``config.yaml`` (DEPRECATED fallback), or
    ``AUTOMIL_VARIANT_MODEL`` env var (last resort).  Looks up the registered
    variant class in ``MODEL_VARIANTS`` (after scanning the variants directory)
    and applies the class-level ``CLAM_ARGS`` dict to the appropriate
    ``ExperimentConfig`` fields.

    No-op when:
      - No variant selection is found in any of the three read paths, or
      - the variant name resolves to None / empty string, or
      - the variant class has an empty / missing ``CLAM_ARGS`` dict.

    Raises:
      ValueError: if a variant name is specified but not found in the registry
        after scanning the variants directory, or if the variant name fails
        path-traversal validation (T-10-05).

    Args:
      exp_cfg: An ``ExperimentConfig`` instance whose ``.model`` and ``.train``
        fields will be patched in-place (Pitfall 6: mutable dataclass, safe).
      automil_dir: Path to the ``automil/`` directory (contains
        ``applied_variant.json`` and/or ``variants/`` subdirectory).
    """
    automil_dir = Path(automil_dir)

    # ------------------------------------------------------------------
    # Step 1: locate the variant selection — three-path priority chain
    # ------------------------------------------------------------------
    variant_name: str | None = None
    parent_name: str | None = None

    # PRIMARY: applied_variant.json (APL-02 A1-closure)
    applied_json_path = automil_dir / "applied_variant.json"
    if applied_json_path.exists():
        try:
            selection = json.loads(applied_json_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Failed to parse %s: %s — falling through to config.yaml fallback",
                applied_json_path,
                exc,
            )
        else:
            model_sel = selection.get("model") or {}
            variant_name = model_sel.get("variant") or None
            parent_name = model_sel.get("parent") or None
            logger.debug(
                "apply_model_variant_to_exp_cfg: loaded from applied_variant.json"
                " variant=%r parent=%r",
                variant_name,
                parent_name,
            )

    # FALLBACK (deprecated): config.yaml model.variant key
    if variant_name is None:
        config_yaml_path = automil_dir / "config.yaml"
        if config_yaml_path.exists():
            try:
                config = yaml.safe_load(config_yaml_path.read_text()) or {}
            except (yaml.YAMLError, OSError) as exc:
                logger.warning(
                    "Failed to parse %s: %s — falling through to env-var fallback",
                    config_yaml_path,
                    exc,
                )
            else:
                model_section = config.get("model") or {}
                variant_name = model_section.get("variant") or None
                parent_name = model_section.get("parent") or None
                if variant_name:
                    logger.warning(
                        "apply_model_variant_to_exp_cfg: reading variant from config.yaml"
                        " (DEPRECATED — use applied_variant.json).  variant=%r",
                        variant_name,
                    )

    # LAST RESORT: environment variable
    if variant_name is None:
        env_val = os.environ.get("AUTOMIL_VARIANT_MODEL")
        if env_val:
            variant_name = env_val.strip() or None
            if variant_name:
                logger.warning(
                    "apply_model_variant_to_exp_cfg: reading variant from"
                    " AUTOMIL_VARIANT_MODEL env var.  variant=%r",
                    variant_name,
                )

    # No variant selected — return without mutation.
    if not variant_name:
        return

    # ------------------------------------------------------------------
    # Step 2: validate variant_name against path traversal (T-10-05)
    # ------------------------------------------------------------------
    if Path(variant_name).name != variant_name:
        raise ValueError(
            f"Variant name {variant_name!r} contains path traversal characters"
            f" (e.g. '/' or '..').  Rejected for security (T-10-05)."
        )

    # ------------------------------------------------------------------
    # Step 3: scan variants directory and look up the variant class
    # ------------------------------------------------------------------
    variants_root = automil_dir / "variants"
    scan_variants(variants_root)  # populates MODEL_VARIANTS via @register decorators

    key = (parent_name, variant_name)
    variant_cls = MODEL_VARIANTS.get(key)
    if variant_cls is None:
        raise ValueError(
            f"Variant {variant_name!r} (parent={parent_name!r}) not found in registry"
            f" after scanning {variants_root}."
            f" Run `automil refresh-registry` to re-scan the variants directory."
        )

    # ------------------------------------------------------------------
    # Step 4: apply CLAM_ARGS to ExperimentConfig fields (T-10-06: safe,
    # CLAM_ARGS is committed to git — registry-first invariant)
    # ------------------------------------------------------------------
    clam_args: dict = getattr(variant_cls, "CLAM_ARGS", {})
    for field, value in clam_args.items():
        if hasattr(exp_cfg.model, field):
            setattr(exp_cfg.model, field, value)
            logger.debug(
                "apply_model_variant_to_exp_cfg: set exp_cfg.model.%s = %r", field, value
            )
        elif hasattr(exp_cfg.train, field):
            setattr(exp_cfg.train, field, value)
            logger.debug(
                "apply_model_variant_to_exp_cfg: set exp_cfg.train.%s = %r", field, value
            )
        else:
            logger.warning(
                "CLAM_ARGS field %r not found in ModelConfig or TrainConfig;"
                " skipped (variant=%r)",
                field,
                variant_name,
            )

    logger.info(
        "apply_model_variant_to_exp_cfg: applied variant %r (parent=%r),"
        " %d CLAM_ARGS field(s) patched",
        variant_name,
        parent_name,
        len(clam_args),
    )
