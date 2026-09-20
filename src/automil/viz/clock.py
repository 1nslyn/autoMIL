"""The run host's clock: put naive host-local stamps and UTC stamps on one axis.

Transcripts carry UTC timestamps. ``graph.json`` ``created_at``,
``completed/<node>.json`` ``completed_at`` and the orchestrator log carry
naive host-local time. A record converts every stamp to UTC once, using the
run host's zone, so the frontend never has to guess.
"""
from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Iterable, Literal, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

ClockSource = Literal["host", "explicit"]

_QUARTER_HOUR_S = 15 * 60
_LOG_ASCTIME = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2}) (?P<time>\d{2}:\d{2}:\d{2}),(?P<ms>\d{3})"
)


@dataclass(frozen=True)
class HostClock:
    """Where a record's naive stamps come from."""

    tz_name: str | None
    utc_offset_s: int
    source: ClockSource

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _zone(name: str) -> ZoneInfo | None:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        return None


def _offset_of(zone: ZoneInfo, now: datetime) -> int:
    offset = now.astimezone(zone).utcoffset() or timedelta(0)
    return int(offset.total_seconds())


def _zone_from_link(link: Path) -> str | None:
    """The IANA key named by an ``/etc/localtime`` symlink, if any."""
    try:
        target = os.readlink(link)
    except OSError:
        return None
    marker = "zoneinfo/"
    index = target.rfind(marker)
    if index < 0:
        return None
    name = target[index + len(marker):]
    return name if _zone(name) is not None else None


def host_clock(
    *,
    tz_name: str | None = None,
    utc_offset_s: int | None = None,
    env: Mapping[str, str] | None = None,
    localtime_link: Path = Path("/etc/localtime"),
    now: datetime | None = None,
) -> HostClock:
    """Resolve the clock: an explicit zone or offset, else the host's own.

    Host resolution order: ``TZ`` (when it names a zone), the
    ``/etc/localtime`` symlink, then the process offset without a name.
    """
    at = now or datetime.now(timezone.utc)
    if tz_name is not None:
        zone = _zone(tz_name)
        if zone is None:
            raise ValueError(f"unknown time zone {tz_name!r}")
        return HostClock(tz_name=tz_name, utc_offset_s=_offset_of(zone, at), source="explicit")
    if utc_offset_s is not None:
        return HostClock(tz_name=None, utc_offset_s=int(utc_offset_s), source="explicit")

    environment = os.environ if env is None else env
    env_name = environment.get("TZ") or ""
    zone = _zone(env_name) if env_name and not env_name.startswith(":") else None
    name = env_name if zone is not None else _zone_from_link(localtime_link)
    if name is not None:
        zone = _zone(name)
        if zone is not None:
            return HostClock(tz_name=name, utc_offset_s=_offset_of(zone, at), source="host")
    local_offset = at.astimezone().utcoffset() or timedelta(0)
    return HostClock(tz_name=None, utc_offset_s=int(local_offset.total_seconds()), source="host")


def _parse_iso(value: str) -> datetime | None:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _localize(naive: datetime, clock: HostClock) -> datetime:
    zone = _zone(clock.tz_name) if clock.tz_name else None
    if zone is not None:
        return naive.replace(tzinfo=zone)
    return naive.replace(tzinfo=timezone(timedelta(seconds=clock.utc_offset_s)))


def format_utc(moment: datetime) -> str:
    """UTC ISO with millisecond precision and a ``Z`` suffix."""
    utc = moment.astimezone(timezone.utc)
    return utc.strftime("%Y-%m-%dT%H:%M:%S.") + f"{utc.microsecond // 1000:03d}Z"


def to_utc_iso(value: object, clock: HostClock) -> str | None:
    """Convert an aware ISO string, a naive host-local ISO string or an epoch.

    Anything unparseable yields ``None``; this never raises on content.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            return format_utc(datetime.fromtimestamp(float(value), tz=timezone.utc))
        except (OverflowError, OSError, ValueError):
            return None
    if not isinstance(value, str):
        return None
    parsed = _parse_iso(value)
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = _localize(parsed, clock)
    return format_utc(parsed)


def parse_log_asctime(text: str) -> datetime | None:
    """The naive stamp at the start of an orchestrator log line."""
    match = _LOG_ASCTIME.match(text)
    if match is None:
        return None
    return datetime.fromisoformat(
        f"{match['date']}T{match['time']}.{match['ms']}000"
    )


def offset_cross_check(pairs: Iterable[tuple[str, str]]) -> int | None:
    """Infer the host offset from (naive local stamp, UTC stamp) pairs.

    Each pair comes from one event seen by both clocks (a node's ``created_at``
    and the transcript turn that created it). The median difference, rounded
    to a quarter hour, is the offset; ``None`` without a usable pair.
    """
    deltas: list[float] = []
    for local_text, utc_text in pairs:
        local = _parse_iso(local_text) if isinstance(local_text, str) else None
        utc = _parse_iso(utc_text) if isinstance(utc_text, str) else None
        if local is None or utc is None or local.tzinfo is not None or utc.tzinfo is None:
            continue
        deltas.append((local - utc.astimezone(timezone.utc).replace(tzinfo=None)).total_seconds())
    if not deltas:
        return None
    return int(round(median(deltas) / _QUARTER_HOUR_S) * _QUARTER_HOUR_S)
