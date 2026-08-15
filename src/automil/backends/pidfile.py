"""PID-file starttime cross-check (CLN-04 / D-17) — public surface.

PID reuse on Linux can cause a stale PID file to claim ownership of an
unrelated process. Compare both pid AND /proc/<pid>/stat starttime_ticks
before signalling. Linux-only is acceptable per PROJECT.md Constraints.

Extracted from ``_orchestrator_daemon`` so that operator tooling (e.g.
``benchmarks/scripts/campaign_operate.py``) can check daemon liveness with
the daemon's own semantics without importing the daemon module. Ad-hoc
parsing of these JSON pid files has already caused one wrong "all daemons
stale" verdict; this module is the single authority.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_starttime_from_stat_line(line: str) -> int:
    """Parse field 22 (1-indexed) — process starttime in clock ticks — from a /proc/<pid>/stat line.

    The `comm` field (#2) is wrapped in parentheses and CAN contain spaces.
    Find the LAST ')' to skip past comm, then split the suffix on whitespace.
    """
    end_comm = line.rfind(")")
    if end_comm == -1:
        raise ValueError(f"Malformed /proc/<pid>/stat line: {line!r}")
    # After the ')' there's a space, then field 3 (state) onwards.
    suffix = line[end_comm + 1:].strip()
    fields = suffix.split()
    # suffix starts at field 3; starttime is field 22 (1-indexed) -> suffix index 22 - 3 = 19.
    if len(fields) < 20:
        raise ValueError(f"/proc/<pid>/stat has fewer fields than expected: {len(fields)}")
    return int(fields[19])


def read_proc_starttime(pid: int) -> int | None:
    """Read /proc/<pid>/stat field 22 (starttime_ticks). Returns None if pid not found or /proc unavailable."""
    try:
        line = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, PermissionError, OSError):
        return None
    try:
        return parse_starttime_from_stat_line(line)
    except ValueError as e:
        logger.warning("Could not parse /proc/%d/stat: %s", pid, e)
        return None


def is_pid_alive_with_starttime(pid: int, expected_starttime_ticks: int) -> bool:
    """True iff the process at *pid* is running AND its starttime matches the recorded value.

    The starttime check defends against PID reuse: a previous daemon's PID
    could be reassigned to an unrelated process; signalling that PID would
    be wrong. See CONCERNS.md §"PID-file stale-detection uses os.kill(pid, 0)".
    """
    actual = read_proc_starttime(pid)
    if actual is None:
        return False
    return actual == expected_starttime_ticks


def write_pid_file(pid_file: Path) -> None:
    """Write PID file as JSON with pid + starttime_ticks + starttime_iso (D-17 shape)."""
    my_pid = os.getpid()
    starttime = read_proc_starttime(my_pid)
    if starttime is None:
        # /proc unavailable (non-Linux test env); record what we can.
        starttime = 0
    payload = {
        "pid": my_pid,
        "starttime_ticks": starttime,
        "starttime_iso": datetime.now().isoformat(),
    }
    pid_file.write_text(json.dumps(payload) + "\n")


def load_pid_file(pid_file: Path) -> dict | None:
    """Load pid_file as JSON. Returns None on legacy plain-int, invalid JSON, or missing keys.

    None means "treat as stale" — the caller should unlink and proceed as
    if no daemon were running. Documented for plain-int compat: an in-flight
    daemon started before this change uses the legacy format; on first
    post-upgrade cmd_start, the legacy file is treated as stale and
    unlinked, the operator restarts and gets the new format.
    """
    try:
        data = json.loads(pid_file.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    if not {"pid", "starttime_ticks", "starttime_iso"}.issubset(data.keys()):
        return None
    return data
