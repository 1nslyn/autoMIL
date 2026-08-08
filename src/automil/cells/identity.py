"""Strict budget-cell identity resolution for runtime integrations.

Runtime hooks cannot rely on submit-time proposal metadata.  They therefore
resolve the cell directly from the current ``config.yaml`` schema and fail
closed when that identity is incomplete.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from automil.cells.state import make_cell_id, normalize_mil_model


class CellIdentityError(ValueError):
    """Raised when the current config cannot identify one budget cell."""


@dataclass(frozen=True, slots=True)
class CellIdentity:
    """Canonical identity of one budget cell."""

    dataset: str
    encoder: str
    mil_model: str
    task: str | None
    cell_id: str


def _required_string(config: Mapping[str, object], section: str, key: str) -> str:
    value = config.get(section)
    if not isinstance(value, Mapping):
        raise CellIdentityError(f"config.{section} must be a mapping")
    resolved = value.get(key)
    if not isinstance(resolved, str) or not resolved.strip():
        raise CellIdentityError(f"config.{section}.{key} must be a non-empty string")
    return resolved.strip()


def resolve_cell_identity(
    config: Mapping[str, object],
    mil_model: str | None = None,
) -> CellIdentity:
    """Resolve a canonical cell identity from the current config schema.

    ``mil_model`` is an explicit override when supplied; otherwise
    ``run.mil_model`` is required.  No historical config keys or proposal
    metadata are consulted.
    """
    if not isinstance(config, Mapping):
        raise CellIdentityError("config must be a mapping")

    dataset = _required_string(config, "project", "name")
    encoder = _required_string(config, "encoders", "primary")
    task_name = _required_string(config, "task", "name")

    model_raw = mil_model
    if model_raw is None:
        model_raw = _required_string(config, "run", "mil_model")
    elif not isinstance(model_raw, str) or not model_raw.strip():
        raise CellIdentityError("mil_model must be a non-empty string")

    model = normalize_mil_model(model_raw)
    task = None if task_name == dataset else task_name
    return CellIdentity(
        dataset=dataset,
        encoder=encoder,
        mil_model=model,
        task=task,
        cell_id=make_cell_id(dataset, encoder, model, task),
    )


__all__ = ["CellIdentity", "CellIdentityError", "resolve_cell_identity"]
