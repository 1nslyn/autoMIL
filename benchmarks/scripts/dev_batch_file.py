"""The batch file of a dev cell: a JSON list of recipe attempts, validated up front.

    {"name": "ema", "parent": "baseline", "desc": "EMA of the weights, decay 0.998",
     "axis": "weight_averaging", "kind": "regularization",
     "constants": {"EMA_DECAY": 0.998}, "hparams": {}}

``parent`` is null for the root baseline (the one entry that runs the trainer's own
recipe), else the name of an earlier entry, the name of an attempt an earlier batch
already registered, or a node id. ``constants`` are the recipe template's class
constants; the cell supplies ``ARM`` itself. ``hparams`` go to the trainer unchanged.
A child inherits nothing from its parent: it restates every constant and hparam it
wants, and ``parent`` only names the node its delta is paired against.
Every entry is checked before the driver touches the cell, so a malformed batch
never leaves a half-registered one.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

KINDS = ("hp", "regularization")
ENTRY_KEYS = ("name", "parent", "desc", "axis", "kind", "constants", "hparams")
REQUIRED_KEYS = ("name", "parent", "desc")
#: dev_recipe's own name rule: no leading ``_`` (the registry scanner skips such files).
NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
NODE_ID_PATTERN = re.compile(r"node_\d+")
#: Recipe constants that weight two classes: the submit-time policy smoke drives two classes only.
TWO_CLASS_CONSTANTS = ("POSITIVE_SHARE", "CLASS_WEIGHTS")


class BatchFileError(ValueError):
    """A batch file the driver refuses to run."""


@dataclass(frozen=True)
class Entry:
    """One validated attempt; ``parent`` None marks the root baseline."""

    name: str
    parent: str | None
    desc: str
    axis: str | None
    kind: str | None
    constants: Mapping[str, Any] | None
    hparams: Mapping[str, Any]

    @property
    def title(self) -> str:
        """The node description: led by the name, which is how a re-run finds the node again."""
        return f"{_title_prefix(self.name)}{self.desc}"

    @property
    def tag(self) -> str:
        return _tag(self.name)


def _tag(name: str) -> str:
    return f"recipe:{name}"


def _title_prefix(name: str) -> str:
    return f"{name}: "


def find_node(nodes: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    """The node an entry name labels: its ``recipe:<name>`` tag, else its ``<name>: `` description prefix."""
    for node_id in sorted(nodes):
        node = nodes[node_id]
        if _tag(name) in (node.get("techniques") or []) or str(node.get("description") or "").startswith(
            _title_prefix(name)
        ):
            return node
    return None


def _check_root(entry: Mapping[str, Any], name: str) -> None:
    stray = [key for key in ("constants", "axis", "kind") if entry.get(key) is not None]
    if stray:
        raise BatchFileError(f"root entry {name!r} (parent null) runs the native recipe: {stray} must be null")
    if entry.get("hparams"):
        raise BatchFileError(f"root entry {name!r} (parent null) runs the native recipe: hparams must be {{}}")


def _check_attempt(entry: Mapping[str, Any], name: str, arm: str, n_classes: int) -> None:
    where = f"entry {name!r}"
    constants = entry.get("constants")
    if not isinstance(constants, dict):
        raise BatchFileError(f"{where} needs constants: a JSON object, {{}} for the native recipe")
    if entry.get("kind") not in KINDS:
        raise BatchFileError(f"{where}: kind {entry.get('kind')!r} is not one of {list(KINDS)}")
    axis = entry.get("axis")
    if not isinstance(axis, str) or not axis.strip():
        raise BatchFileError(f"{where}: axis must be a non-empty string, got {axis!r}")
    if constants.get("ARM", arm) != arm:
        raise BatchFileError(f"{where}: constants ARM {constants['ARM']!r} conflicts with this cell's arm "
                             f"{arm!r}; the cell supplies ARM")
    weighted = [key for key in TWO_CLASS_CONSTANTS if constants.get(key)]
    if weighted and n_classes != 2:
        raise BatchFileError(f"{where}: class weights ({', '.join(weighted)}) need a two-class cell; this cell has "
                             f"{n_classes} classes (the submit-time policy smoke drives two classes only)")


def _parse_entry(item: object, position: int, arm: str, n_classes: int) -> Entry:
    where = f"batch entry {position}"
    if not isinstance(item, dict):
        raise BatchFileError(f"{where} must be a JSON object, got {type(item).__name__}")
    unknown = sorted(set(item) - set(ENTRY_KEYS))
    if unknown:
        raise BatchFileError(f"{where}: unknown key(s) {unknown}; an entry has {list(ENTRY_KEYS)}")
    absent = [key for key in REQUIRED_KEYS if key not in item]
    if absent:
        raise BatchFileError(f"{where}: missing key(s) {absent}")
    name = item["name"]
    if not isinstance(name, str) or not NAME_PATTERN.fullmatch(name) or NODE_ID_PATTERN.fullmatch(name):
        raise BatchFileError(f"{where}: name {name!r} must be letters, digits, '_' or '-', start with a "
                             "letter or digit and not look like a node id")
    parent, desc, hparams = item["parent"], item["desc"], item.get("hparams", {})
    if parent is not None and not (isinstance(parent, str) and parent.strip()):
        raise BatchFileError(f"entry {name!r}: parent must be null (the root), an earlier entry's name or a node id")
    if not isinstance(desc, str) or not desc.strip():
        raise BatchFileError(f"entry {name!r}: desc must be a non-empty string")
    if not isinstance(hparams, dict):
        raise BatchFileError(f"entry {name!r}: hparams must be a JSON object, got {type(hparams).__name__}")
    if parent is None:
        _check_root(item, name)
    else:
        _check_attempt(item, name, arm, n_classes)
    axis = item.get("axis")
    return Entry(name, parent.strip() if parent else None, desc.strip(), axis.strip() if axis else None,
                 item.get("kind"), item.get("constants"), hparams)


def parse_batch(path: Path, *, arm: str, n_classes: int) -> list[Entry]:
    """The entries of a batch file for a cell running ``arm`` on ``n_classes`` classes, every one validated."""
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise BatchFileError(f"cannot read batch file {path}: {exc}") from exc
    if not isinstance(raw, list):
        raise BatchFileError(f"{path}: a batch is a JSON list of entries, got {type(raw).__name__}")
    if not raw:
        raise BatchFileError(f"{path}: the batch is empty")
    entries = [_parse_entry(item, position, arm, n_classes) for position, item in enumerate(raw, start=1)]
    names = [entry.name for entry in entries]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise BatchFileError(f"{path}: duplicate entry name(s) {repeated}")
    roots = [entry.name for entry in entries if entry.parent is None]
    if len(roots) > 1:
        raise BatchFileError(f"{path}: a batch has at most one root entry (parent null); found {roots}")
    for position, entry in enumerate(entries):
        if entry.parent in names and names.index(entry.parent) >= position:
            raise BatchFileError(f"entry {entry.name!r}: parent {entry.parent!r} is listed at or after it; "
                                 "a parent must come earlier in the batch")
    return entries
