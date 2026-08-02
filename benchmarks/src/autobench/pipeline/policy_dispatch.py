"""Consumer-side dispatch for autoMIL's train-only ``PolicyVariant`` seam.

The autoMIL framework only records and transports a selected policy.  This
module is the benchmark consumer's single adapter: it resolves that selection,
instantiates the registered policy, and exposes three guarded operations to all
five MIL arms.  Model construction, forward paths, defining losses, validation,
and result writing remain in protected trainers and are never passed in.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


def runtime_automil_dir() -> Path:
    """Return this worktree's configured autoMIL directory.

    The orchestrator launches from the git-worktree root even when the consumer
    project is nested.  ``AUTOMIL_DIR_REL`` is therefore the only unambiguous
    runtime locator; the root-level fallback preserves standalone projects.
    """
    raw = os.environ.get("AUTOMIL_DIR_REL", "automil")
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"invalid AUTOMIL_DIR_REL {raw!r}")
    return path


def _policy_from_applied_variant(automil_dir: Path) -> str | None:
    path = automil_dir / "applied_variant.json"
    if not path.exists():
        return None
    try:
        selection = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(selection, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    section = selection.get("policy") or {}
    if not isinstance(section, Mapping):
        raise ValueError(f"{path}: policy must be a JSON object")
    name = section.get("variant")
    if name is None:
        return None
    if not isinstance(name, str) or not name.strip() or Path(name).name != name:
        raise ValueError(f"{path}: invalid policy variant name {name!r}")
    return name


def resolve_policy_name(exp_cfg: Any, automil_dir: Path | None = None) -> str | None:
    """Resolve explicit and archived selection, rejecting disagreement."""
    automil_dir = automil_dir or runtime_automil_dir()
    explicit = getattr(exp_cfg, "policy_variant", None)
    archived = _policy_from_applied_variant(automil_dir)
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit.strip() or Path(explicit).name != explicit:
            raise ValueError(f"invalid policy variant name {explicit!r}")
    if explicit and archived and explicit != archived:
        raise ValueError(
            "policy selection disagrees between ExperimentConfig and "
            f"applied_variant.json ({explicit!r} != {archived!r})"
        )
    return explicit or archived


@dataclass(frozen=True)
class PolicyRuntime:
    """Fail-loud runtime wrapper around one optional ``PolicyVariant``."""

    name: str | None = None
    policy: Any | None = None

    @classmethod
    def from_experiment(
        cls,
        exp_cfg: Any,
        automil_dir: Path | None = None,
    ) -> "PolicyRuntime":
        automil_dir = automil_dir or runtime_automil_dir()
        name = resolve_policy_name(exp_cfg, automil_dir)
        if name is None:
            return cls()

        from automil.registry import resolve_policy
        from automil.registry.scanner import scan_variants

        variants_root = automil_dir / "variants"
        scan_result = scan_variants(variants_root)
        try:
            policy_cls = resolve_policy(name)
        except KeyError as exc:
            failures = "; ".join(
                f"{path.name}: {reason}" for path, reason in scan_result.failed
            )
            detail = f" Import failures: {failures}" if failures else ""
            raise ValueError(
                f"selected policy variant {name!r} was not registered from "
                f"{variants_root}.{detail}"
            ) from exc
        return cls(name=name, policy=policy_cls())

    def wrap_optimizer(self, optimizer: Any, *, role: str = "main") -> Any:
        if self.policy is None:
            return optimizer
        wrapped = self.policy.wrap_optimizer_for(optimizer, role=role)
        if wrapped is None:
            raise TypeError(f"policy {self.name!r} returned None for optimizer role {role!r}")
        for method in ("zero_grad", "step"):
            if not callable(getattr(wrapped, method, None)):
                raise TypeError(
                    f"policy {self.name!r} returned an invalid optimizer for role "
                    f"{role!r}: missing callable {method}()"
                )
        return wrapped

    def wrap_scheduler(self, scheduler: Any, *, role: str = "main") -> Any:
        if self.policy is None:
            return scheduler
        wrapped = self.policy.wrap_scheduler_for(scheduler, role=role)
        if wrapped is None or not callable(getattr(wrapped, "step", None)):
            raise TypeError(
                f"policy {self.name!r} returned an invalid scheduler for role {role!r}"
            )
        return wrapped

    def should_stop(
        self,
        default: bool,
        *,
        epoch: int,
        metrics: Mapping[str, float] | None = None,
    ) -> bool:
        if self.policy is None:
            return bool(default)
        safe_metrics: dict[str, float] = {}
        for key, value in (metrics or {}).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"policy metric {key!r} must be a real scalar")
            safe_metrics[str(key)] = float(value)
        decision = self.policy.should_stop(
            default=bool(default), epoch=int(epoch), metrics=safe_metrics,
        )
        if type(decision) is not bool:
            raise TypeError(f"policy {self.name!r} should_stop() must return bool")
        return decision
