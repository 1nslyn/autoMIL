"""viz.clock: put naive host-local stamps and UTC stamps on one axis."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from automil.viz.clock import (
    HostClock,
    host_clock,
    offset_cross_check,
    parse_log_asctime,
    to_utc_iso,
)


def test_explicit_zone_is_validated_and_marked_explicit():
    clock = host_clock(tz_name="America/Vancouver", now=datetime(2026, 9, 7, tzinfo=timezone.utc))
    assert clock.source == "explicit"
    assert clock.tz_name == "America/Vancouver"
    assert clock.utc_offset_s == -7 * 3600  # PDT in September
    with pytest.raises(ValueError):
        host_clock(tz_name="Mars/Olympus")


def test_explicit_fixed_offset():
    clock = host_clock(utc_offset_s=3600)
    assert clock == HostClock(tz_name=None, utc_offset_s=3600, source="explicit")


def test_host_zone_comes_from_tz_env_then_localtime_link(tmp_path: Path):
    from_env = host_clock(env={"TZ": "Europe/Berlin"}, now=datetime(2026, 1, 15, tzinfo=timezone.utc))
    assert (from_env.tz_name, from_env.source, from_env.utc_offset_s) == ("Europe/Berlin", "host", 3600)

    link = tmp_path / "localtime"
    link.symlink_to("/usr/share/zoneinfo/Asia/Tokyo")
    from_link = host_clock(env={}, localtime_link=link, now=datetime(2026, 1, 15, tzinfo=timezone.utc))
    assert (from_link.tz_name, from_link.utc_offset_s) == ("Asia/Tokyo", 9 * 3600)

    # An unusable TZ value falls through to the link, and a missing link to the
    # process offset (tz_name None, still source "host").
    fallback = host_clock(env={"TZ": "not-a-zone"}, localtime_link=tmp_path / "missing")
    assert fallback.source == "host" and fallback.tz_name is None


def test_to_utc_iso_accepts_aware_naive_epoch_and_garbage():
    clock = host_clock(tz_name="America/Vancouver")
    assert to_utc_iso("2026-09-07T00:22:29.657Z", clock) == "2026-09-07T00:22:29.657Z"
    assert to_utc_iso("2026-09-07T00:36:23.214441+00:00", clock) == "2026-09-07T00:36:23.214Z"
    # naive host-local (PDT, -7) in September
    assert to_utc_iso("2026-09-06T17:30:28.759919", clock) == "2026-09-07T00:30:28.759Z"
    # naive in January is PST (-8): the IANA zone handles the DST change
    assert to_utc_iso("2026-01-06T17:30:28", clock) == "2026-01-07T01:30:28.000Z"
    assert to_utc_iso(1788740513.9171042, clock) == "2026-09-07T00:21:53.917Z"
    assert to_utc_iso(None, clock) is None
    assert to_utc_iso("yesterday", clock) is None
    assert to_utc_iso({"at": 1}, clock) is None


def test_to_utc_iso_uses_fixed_offset_without_a_zone_name():
    clock = HostClock(tz_name=None, utc_offset_s=-7 * 3600, source="explicit")
    assert to_utc_iso("2026-09-06T17:30:28", clock) == "2026-09-07T00:30:28.000Z"


def test_parse_log_asctime():
    assert parse_log_asctime("2026-09-06 17:36:40,814") == datetime(2026, 9, 6, 17, 36, 40, 814000)
    assert parse_log_asctime("not a stamp") is None


def test_offset_cross_check_rounds_to_quarter_hours_and_needs_pairs():
    pairs = [
        ("2026-09-06T17:30:28.759919", "2026-09-07T00:30:27.100Z"),
        ("2026-09-06T17:30:29.899969", "2026-09-07T00:30:29.500Z"),
    ]
    assert offset_cross_check(pairs) == -7 * 3600
    assert offset_cross_check([]) is None
    assert offset_cross_check([("garbage", "2026-09-07T00:30:27Z")]) is None
