"""Consumer-side config reader for the ``gate:`` section (CFG-1).

**Why this module exists.** The scaffold has shipped a ``gate:`` block
(``auto_nominate`` / ``K`` / ``p_threshold`` / ``bootstrap_reps``) since the gate
was built, annotated "consumer-supplied (NOT framework constants)" — and no code
ever read it. Every value came from ``automil gate register-manifest``'s own CLI
defaults, so the block was decorative: an operator could set ``K: 5`` and get 2.

For a **pre-registered** statistical gate that is the wrong direction of travel.
The whole point of ``write_manifest_committed`` is that the thresholds are fixed
in git before any held-out evaluation runs; taking them from whatever was typed
at the moment someone decided to run the gate leaves the one number that must be
decided in advance being decided last. Reading them from the committed
``automil/config.yaml`` puts the decision where the discipline already assumes it
is, and the CLI flags remain available as an explicit override.

Defaults here reproduce the previous CLI defaults exactly, so a project whose
config omits the block sees no change.
"""
from __future__ import annotations

from dataclasses import dataclass

from automil.gate.manifest import BOOTSTRAP_REPS_FLOOR

__all__ = ["GateConfig", "load_gate_config"]

DEFAULT_K = 2
DEFAULT_P_THRESHOLD = 0.05
DEFAULT_BOOTSTRAP_REPS = 1000
DEFAULT_AUTO_NOMINATE = False


@dataclass(frozen=True)
class GateConfig:
    """Typed view onto the ``gate:`` section of automil/config.yaml."""

    K: int = DEFAULT_K
    p_threshold: float = DEFAULT_P_THRESHOLD
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS
    #: RESERVED (claims-alignment C-e): parsed and validated, but no automatic
    #: nomination path exists anywhere — nomination is always the operator's
    #: `automil nominate`. Kept so the CFG-1 config surface stays stable; the
    #: template comment says the same so nobody expects behavior from it.
    auto_nominate: bool = DEFAULT_AUTO_NOMINATE


def _require_int(section: dict, key: str, default: int) -> int:
    raw = section.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(
            f"automil/config.yaml: gate.{key} must be an integer; got {raw!r}"
        )
    return raw


def load_gate_config(automil_cfg: object) -> GateConfig:
    """Resolve the gate's pre-registered parameters from a parsed config dict.

    Validation is the same as ``manifest.validate_manifest_fields`` and happens
    here as well as there, deliberately: a bad value in the committed config
    should be rejected when the manifest is written, not when the gate finally
    runs against it hours later.

    Raises:
        ValueError: any value is out of range or the wrong type.
        TypeError: ``gate`` is present but is not a mapping.
    """
    if not isinstance(automil_cfg, dict):
        return GateConfig()
    section = automil_cfg.get("gate")
    if section is None:
        return GateConfig()
    if not isinstance(section, dict):
        raise TypeError(
            f"automil/config.yaml: 'gate' must be a mapping; "
            f"got {type(section).__name__}."
        )

    K = _require_int(section, "K", DEFAULT_K)
    if K < 1:
        raise ValueError(f"automil/config.yaml: gate.K must be >= 1; got {K}")

    reps = _require_int(section, "bootstrap_reps", DEFAULT_BOOTSTRAP_REPS)
    if reps < BOOTSTRAP_REPS_FLOOR:
        raise ValueError(
            f"automil/config.yaml: gate.bootstrap_reps must be >= "
            f"{BOOTSTRAP_REPS_FLOOR}; got {reps}"
        )

    p_raw = section.get("p_threshold", DEFAULT_P_THRESHOLD)
    if isinstance(p_raw, bool) or not isinstance(p_raw, (int, float)):
        raise ValueError(
            f"automil/config.yaml: gate.p_threshold must be a number; got {p_raw!r}"
        )
    p_threshold = float(p_raw)
    if not 0 < p_threshold <= 1:
        raise ValueError(
            f"automil/config.yaml: gate.p_threshold must be in (0, 1]; "
            f"got {p_threshold}"
        )

    auto = section.get("auto_nominate", DEFAULT_AUTO_NOMINATE)
    if not isinstance(auto, bool):
        raise ValueError(
            f"automil/config.yaml: gate.auto_nominate must be a boolean; got {auto!r}"
        )

    return GateConfig(
        K=K, p_threshold=p_threshold, bootstrap_reps=reps, auto_nominate=auto,
    )
