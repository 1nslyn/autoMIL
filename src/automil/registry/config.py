"""Consumer-side config reader for the registry section (REG-04 / REG-06 / REG-07).

The framework ships no defaults for `protected` (D-33 + D-49) — autoMIL is
generic, the consumer's automil/config.yaml is the source of truth for
project-specific paths.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

logger = logging.getLogger(__name__)

Mode = Literal["free", "architecture-preserving"]
_VALID_MODES: tuple[str, ...] = ("free", "architecture-preserving")


@dataclass(frozen=True)
class RegistryConfig:
    """Typed view onto automil/config.yaml registry section.

    Frozen because each call to load_registry_config returns a fresh value;
    callers reload after mutation. The Mode literal is enforced at load
    time (load_registry_config raises ValueError on unknown modes).

    Defaults (all configurable in consumer automil/config.yaml):
        protected: ()           — no framework defaults (D-33 + D-49)
        mode: "free"            — D-31: default search scope
        repro_tolerance: 0.005  — D-39: CCRCC ±0.005 carried as framework default
        identity_constraints: () — D-31: per-project structural rules
    """

    protected: tuple[str, ...] = ()              # glob patterns (relative to project root)
    mode: Mode = "free"                          # D-31: default free
    repro_tolerance: float = 0.005               # D-39: default ±0.005
    identity_constraints: tuple[str, ...] = ()   # D-31: per-project identity rules


def _coerce_str_tuple(raw: object, key: str) -> tuple[str, ...]:
    """Coerce a YAML list to a tuple of strings; hard-fail on wrong type.

    Hard-fail (TypeError) named after the offending key so the operator
    sees exactly what to fix in config.yaml.
    """
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise TypeError(
            f"automil/config.yaml: {key!r} must be a list of strings; "
            f"got {type(raw).__name__}. Edit the file to use list syntax."
        )
    out: list[str] = []
    for i, item in enumerate(raw):
        if not isinstance(item, str):
            raise TypeError(
                f"automil/config.yaml: {key}[{i}] must be a string; "
                f"got {type(item).__name__}."
            )
        out.append(item)
    return tuple(out)


def load_registry_config(automil_dir: Path) -> RegistryConfig:
    """Load the registry section from automil/config.yaml.

    Returns RegistryConfig with all defaults if:
        - automil/config.yaml does not exist, OR
        - the registry: section is missing, OR
        - the registry: section is an empty dict.

    Raises:
        TypeError: if any registry field is the wrong type (e.g.,
                   protected: 42 instead of a list).
        ValueError: if registry.mode is not in {"free", "architecture-preserving"}.
    """
    config_path = automil_dir / "config.yaml"
    if not config_path.exists():
        return RegistryConfig()

    raw_yaml = yaml.safe_load(config_path.read_text()) or {}
    registry_section = raw_yaml.get("registry") or {}
    if not isinstance(registry_section, dict):
        raise TypeError(
            f"automil/config.yaml: 'registry' must be a mapping; "
            f"got {type(registry_section).__name__}."
        )

    protected = _coerce_str_tuple(registry_section.get("protected"), "registry.protected")
    identity_constraints = _coerce_str_tuple(
        registry_section.get("identity_constraints"), "registry.identity_constraints"
    )

    mode_raw = registry_section.get("mode", "free")
    if mode_raw not in _VALID_MODES:
        raise ValueError(
            f"automil/config.yaml: registry.mode must be one of {_VALID_MODES}; "
            f"got {mode_raw!r}. Set to 'free' (default) or 'architecture-preserving'."
        )

    # H-4: the mode used to be parsed, validated, and then read by no code at
    # all. Requiring a non-empty protected list makes it self-consistent — a
    # project cannot announce that it preserves a substrate while naming nothing
    # to preserve. This matters because submit's gate is written
    # `if reg_cfg.protected and ...`, so an empty tuple short-circuits the whole
    # enforcement path (and check.py's drift detection, and revert-baseline) to a
    # silent no-op.
    if mode_raw == "architecture-preserving" and not protected:
        raise ValueError(
            "automil/config.yaml: registry.mode is 'architecture-preserving' but "
            "registry.protected is empty. An empty list disables the protected-file "
            "gate entirely, so the substrate would be declared frozen and not "
            "enforced. List the paths that must not change (splits/prep/evaluate, "
            "the composite writer, the feature-extraction entry point), or set "
            "mode: free."
        )

    repro_tolerance_raw = registry_section.get("repro_tolerance", 0.005)
    try:
        repro_tolerance = float(repro_tolerance_raw)
    except (TypeError, ValueError) as e:
        raise TypeError(
            f"automil/config.yaml: registry.repro_tolerance must be a float; "
            f"got {type(repro_tolerance_raw).__name__} ({repro_tolerance_raw!r})."
        ) from e

    return RegistryConfig(
        protected=protected,
        mode=mode_raw,  # type: ignore[arg-type]  # validated above
        repro_tolerance=repro_tolerance,
        identity_constraints=identity_constraints,
    )
