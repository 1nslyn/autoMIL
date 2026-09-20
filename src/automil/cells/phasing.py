"""Fixed batches and the phasing rule of a budgeted cell (``cap.phasing``).

A cell that declares ``cap.phasing`` spends its evaluation budget in fixed,
non-overlapping batches, opens on a declared number of distinct axes, never
spends more than a declared number of consecutive attempts on one axis
without a kept result, and closes with a declared number of pre-registered
robustness neighbours of the best node. ``automil propose`` records the axis,
the predicted delta and the role on each proposal; ``automil submit`` asks
:func:`phasing_refusal` before it writes a queue spec, so a refusal costs
nothing. Attempt order is node-id order, the order the campaign's freeze
census uses.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

NEIGHBOUR = "neighbour"
ROLES = (NEIGHBOUR,)

_KEYS = ("batches", "opening_axes_min", "max_consecutive_per_axis", "reserve_neighbours_min")


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"cap.phasing.{name} must be a positive integer, got {value!r}")
    return value


@dataclass(frozen=True)
class PhasingPolicy:
    batches: tuple[int, ...]
    opening_axes_min: int
    max_consecutive_per_axis: int
    reserve_neighbours_min: int

    @classmethod
    def from_config(cls, cap: Mapping | None) -> PhasingPolicy | None:
        """The declared policy, or ``None`` when ``cap.phasing`` is absent.

        Raises ``ValueError`` on a malformed declaration: a policy that is
        declared but unreadable must refuse, never degrade into "no phasing".
        """
        raw = (cap or {}).get("phasing") if isinstance(cap, Mapping) else None
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise ValueError("cap.phasing must be a mapping")
        missing = [key for key in _KEYS if key not in raw]
        if missing:
            raise ValueError(f"cap.phasing is missing {missing}")
        batches = raw["batches"]
        if not isinstance(batches, list) or not batches:
            raise ValueError("cap.phasing.batches must be a non-empty list")
        policy = cls(
            batches=tuple(_positive_int(size, "batches[]") for size in batches),
            opening_axes_min=_positive_int(raw["opening_axes_min"], "opening_axes_min"),
            max_consecutive_per_axis=_positive_int(
                raw["max_consecutive_per_axis"], "max_consecutive_per_axis"),
            reserve_neighbours_min=_positive_int(
                raw["reserve_neighbours_min"], "reserve_neighbours_min"),
        )
        budget = cap.get("eval_budget")
        if budget is not None and policy.total != budget:
            raise ValueError(
                f"cap.phasing.batches sum to {policy.total}, cap.eval_budget is {budget}"
            )
        return policy

    @property
    def total(self) -> int:
        return sum(self.batches)

    def batch_of(self, attempt: int) -> int:
        """1-based batch of the 1-based ``attempt``."""
        end = 0
        for index, size in enumerate(self.batches, start=1):
            end += size
            if attempt <= end:
                return index
        return len(self.batches)

    def batch_bounds(self, batch: int) -> tuple[int, int]:
        """(first, last) 1-based attempts of the 1-based ``batch``."""
        first = 1 + sum(self.batches[: batch - 1])
        return first, first + self.batches[batch - 1] - 1


@dataclass(frozen=True)
class Attempt:
    """One submitted attempt of the cell, in submission order."""

    node_id: str
    axis: str | None
    role: str | None
    status: str | None


def cell_attempts(nodes: Mapping[str, Mapping], cell_id: str) -> tuple[Attempt, ...]:
    """The cell's submitted attempts in node-id order: every node tagged with
    the cell that has left ``pending``, except the bootstrapped root and a
    spec the cap refused at launch (it was never charged)."""
    from automil.graph import node_cell_id

    attempts = []
    for node_id, node in sorted(nodes.items()):
        if node_cell_id(node) != cell_id or node.get("bootstrapped"):
            continue
        if node.get("status") in (None, "pending", "cancelled"):
            continue
        meta = node.get("metadata") if isinstance(node.get("metadata"), Mapping) else {}
        attempts.append(Attempt(
            node_id=node_id, axis=meta.get("axis"), role=meta.get("role"),
            status=node.get("status"),
        ))
    return tuple(attempts)


def in_flight_node_ids(adir: Path, cell_id: str) -> frozenset[str]:
    """Node ids of the cell's specs still in ``orchestrator/queue`` or
    ``orchestrator/running/<backend>/``."""
    orchestrator = adir / "orchestrator"
    paths = list((orchestrator / "queue").glob("*.json")) + \
        list((orchestrator / "running").glob("*/*.json"))
    ids = set()
    for path in paths:
        try:
            spec = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (spec.get("metadata") or {}).get("cell_id") == cell_id:
            ids.add(path.stem)
    return frozenset(ids)


def phasing_refusal(
    policy: PhasingPolicy,
    attempts: tuple[Attempt, ...],
    *,
    axis: str | None,
    role: str | None,
    parent_id: str | None,
    best_node_id: str | None,
    in_flight: frozenset[str],
) -> str | None:
    """Why the next submission would break the declared phasing, or ``None``.

    The candidate is attempt ``len(attempts) + 1``. Past the budget the
    phasing says nothing: the budget gate refuses that.
    """
    if not axis:
        return ("propose with --axis (and --predicted-delta): cap.phasing "
                "judges every attempt by its axis")
    if role == NEIGHBOUR and parent_id != best_node_id:
        return (f"a robustness neighbour must be a child of the current best node "
                f"({best_node_id}), not of {parent_id}")
    k = len(attempts) + 1
    if k > policy.total:
        return None
    batch = policy.batch_of(k)
    first, last = policy.batch_bounds(batch)
    if batch > 1:
        earlier = {a.node_id for a in attempts[: first - 1]}
        still_running = sorted(earlier & in_flight)
        if still_running:
            return (f"attempt {k} opens batch {batch}; it can start only after every "
                    f"attempt of batches 1-{batch - 1} has finished (in flight: "
                    f"{', '.join(still_running)})")
    if k == policy.batch_bounds(1)[1]:
        axes = {a.axis for a in attempts[:k - 1]} | {axis}
        if len(axes) < policy.opening_axes_min:
            return (f"the opening batch would close with {len(axes)} distinct axes "
                    f"({', '.join(sorted(axes))}); {policy.opening_axes_min} are required")
    m = policy.max_consecutive_per_axis
    recent = attempts[-m:] if m <= len(attempts) else ()
    if len(recent) == m and all(a.axis == axis for a in recent) \
            and not any(a.status == "keep" for a in recent):
        return (f"attempt {k} would be the {m + 1}th consecutive attempt on axis "
                f"{axis!r} without a kept result; change axis")
    if k == policy.total:
        final = [a for a in attempts[first - 1:] if a.role == NEIGHBOUR]
        n = len(final) + (1 if role == NEIGHBOUR else 0)
        if n < policy.reserve_neighbours_min:
            return (f"the final batch would close with {n} robustness neighbour(s) of "
                    f"the best node; {policy.reserve_neighbours_min} are required "
                    f"(propose --role neighbour under {best_node_id})")
    return None


def batch_position(
    policy: PhasingPolicy, attempts: tuple[Attempt, ...], in_flight: frozenset[str],
) -> str:
    """One status line: where the cell stands in its declared batches."""
    submitted = len(attempts)
    running = len({a.node_id for a in attempts} & in_flight)
    if submitted >= policy.total:
        return f"phasing: all {policy.total} attempts submitted"
    k = submitted + 1
    batch = policy.batch_of(k)
    first, last = policy.batch_bounds(batch)
    return (f"phasing: {submitted}/{policy.total} submitted, next is attempt {k} in "
            f"batch {batch} of {len(policy.batches)} (attempts {first}-{last}), "
            f"{running} in flight")
