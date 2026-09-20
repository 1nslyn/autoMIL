"""Find ``automil`` commands inside Bash tool calls and the node ids they touch.

The agent drives the framework through its CLI, so the transcript's Bash
calls are where proposals and submissions happen. ``propose`` and ``submit``
print the node they created (``cli/propose.py``, ``cli/submit.py``,
``cli/resubmit.py``); every other subcommand only refers to ids given on its
command line.
"""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import Iterable

NODE_ID = re.compile(r"\bnode_\d{4,}\b")

# ``automil <sub> ...`` anywhere in a shell line, however it is launched
# (``uv run --project . automil ...``, ``python -m automil ...``). The
# ``--project`` group option may sit between the program and the subcommand.
_COMMAND = re.compile(
    r"\bautomil\s+(?:--project(?:=\S+|\s+\S+)\s+)?(?P<sub>[a-z][a-z-]*)(?P<rest>[^\n;&|)]*)"
)

_CREATED = {
    "propose": re.compile(r"^Added proposal (node_\d{4,}) \[", re.MULTILINE),
    "submit": re.compile(r"^Submitted (node_\d{4,}):", re.MULTILINE),
    "resubmit": re.compile(r"^(node_\d{4,})\s*$", re.MULTILINE),
}


@dataclass(frozen=True)
class AutomilCall:
    """One ``automil`` invocation found in a shell command."""

    sub: str
    argv: tuple[str, ...]
    node_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {"sub": self.sub, "argv": list(self.argv), "node_ids": list(self.node_ids)}


def _split(text: str) -> tuple[str, ...]:
    try:
        return tuple(shlex.split(text))
    except ValueError:
        return tuple(text.split())


def detect_automil(command: str) -> tuple[AutomilCall, ...]:
    """Every ``automil`` invocation in ``command``, in order of appearance."""
    calls = []
    for match in _COMMAND.finditer(command):
        argv = (match["sub"],) + _split(match["rest"])
        node_ids = tuple(sorted({m.group(0) for token in argv[1:] for m in NODE_ID.finditer(token)}))
        calls.append(AutomilCall(sub=match["sub"], argv=argv, node_ids=node_ids))
    return tuple(calls)


def created_node_ids(subs: Iterable[str], output: str) -> tuple[tuple[str, str], ...]:
    """``(node_id, sub)`` for every node the output reports as created.

    Only the subcommands present in the shell line are consulted, so a
    ``rank`` listing never reads as a creation.
    """
    found: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for sub in subs:
        pattern = _CREATED.get(sub)
        if pattern is None:
            continue
        for match in pattern.finditer(output):
            pair = (match.group(1), sub)
            if pair not in seen:
                seen.add(pair)
                found.append(pair)
    return tuple(found)


def summarize_calls(
    calls: Iterable[AutomilCall], output: str | None = None
) -> dict[str, object]:
    """The ``automil`` field of a Bash tool call: its invocations and node ids.

    ``created`` lists the nodes the output reported creating (empty until the
    result arrives); ``node_ids`` is every id the call touched.
    """
    calls = tuple(calls)
    created = created_node_ids([c.sub for c in calls], output) if output else ()
    ids = {node_id for call in calls for node_id in call.node_ids}
    ids.update(node_id for node_id, _ in created)
    return {
        "calls": [call.as_dict() for call in calls],
        "node_ids": sorted(ids),
        "created": [{"node_id": node_id, "sub": sub} for node_id, sub in created],
    }
