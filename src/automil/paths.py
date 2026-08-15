"""Path probing shared across CLI, daemon, and hook entrypoints.

The project-root walk probes ``<ancestor>/automil/config.yaml`` for every
ancestor of cwd. Any ancestor may contain an ``automil`` entry the current
user cannot stat — concretely, a host with an ``automil`` service user makes
``/home/automil`` unreadable, and every walk from a cwd under ``/home``
crashed with PermissionError instead of walking on. A probe target the user
cannot read is not that user's project root; it must read as "not found".
"""
from __future__ import annotations

from pathlib import Path


def probe_exists(path: Path) -> bool:
    """``path.exists()`` that treats an unreadable path as absent."""
    try:
        return path.exists()
    except OSError:
        return False
