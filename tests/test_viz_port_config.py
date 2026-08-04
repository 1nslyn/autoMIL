"""RED stubs for viz port config fallback (OPS-05).

Wave-0 Nyquist compliance — all stubs xfail until 13-02 implements the
port resolution chain in viz.py / server.py cmd_start.

Resolution order (once implemented): explicit --port flag > viz.port in
config.yaml > DEFAULT_PORT (8420).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from automil.cli import main


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


# ---------------------------------------------------------------------------
# OPS-05 RED stubs (Wave 0 — Nyquist compliance)
# All xfail until plan 13-02 changes viz.py --port default=None and adds
# port-resolution logic in server.py cmd_start.
# ---------------------------------------------------------------------------


def test_viz_port_default(tmp_path: Path, monkeypatch, cli_runner: CliRunner) -> None:
    """viz start with no --port and no config key uses DEFAULT_PORT 8420.

    OPS-05: when neither --port nor viz.port in config is set, cmd_start must
    receive port=8420 (DEFAULT_PORT). The current implementation hard-codes
    port=8420 as the default, so this regression guard already passes.
    NOT xfail — this behaviour is already correct.
    """
    # Config has no viz.port key.
    adir = tmp_path / "automil"
    adir.mkdir()
    (adir / "config.yaml").write_text("run:\n  script: train.py\n")
    monkeypatch.chdir(tmp_path)

    captured: dict = {}

    def fake_cmd_start(**kwargs):
        captured.update(kwargs)

    with patch("automil.viz.server.cmd_start", side_effect=fake_cmd_start):
        result = cli_runner.invoke(main, ["viz", "start"], catch_exceptions=False)

    assert captured.get("port") == 8420, (
        f"expected port=8420 (DEFAULT_PORT), got port={captured.get('port')!r}\n"
        f"viz start output: {result.output!r}"
    )


def test_viz_port_from_config(tmp_path: Path, monkeypatch, cli_runner: CliRunner) -> None:
    """viz start with viz.port in config uses the config value.

    OPS-05: when config.yaml contains `viz:\n  port: 9000`, cmd_start must
    receive port=9000 (config wins over DEFAULT_PORT).
    """
    adir = tmp_path / "automil"
    adir.mkdir()
    (adir / "config.yaml").write_text("run:\n  script: train.py\nviz:\n  port: 9000\n")
    monkeypatch.chdir(tmp_path)

    captured: dict = {}

    def fake_cmd_start(**kwargs):
        captured.update(kwargs)

    with patch("automil.viz.server.cmd_start", side_effect=fake_cmd_start):
        result = cli_runner.invoke(main, ["viz", "start"], catch_exceptions=False)

    assert captured.get("port") == 9000, (
        f"expected port=9000 (from config), got port={captured.get('port')!r}\n"
        f"viz start output: {result.output!r}"
    )


def test_viz_port_explicit_overrides_config(
    tmp_path: Path, monkeypatch, cli_runner: CliRunner
) -> None:
    """Explicit --port flag overrides viz.port from config.

    OPS-05: resolution order — explicit --port flag > config viz.port > DEFAULT_PORT.
    When --port 7777 is passed and config has viz.port: 9000, cmd_start must
    receive port=7777. The current implementation passes --port through directly,
    so this regression guard already passes.
    NOT xfail — this behaviour is already correct.
    """
    adir = tmp_path / "automil"
    adir.mkdir()
    (adir / "config.yaml").write_text("run:\n  script: train.py\nviz:\n  port: 9000\n")
    monkeypatch.chdir(tmp_path)

    captured: dict = {}

    def fake_cmd_start(**kwargs):
        captured.update(kwargs)

    with patch("automil.viz.server.cmd_start", side_effect=fake_cmd_start):
        result = cli_runner.invoke(
            main, ["viz", "start", "--port", "7777"], catch_exceptions=False
        )

    assert captured.get("port") == 7777, (
        f"expected port=7777 (explicit flag), got port={captured.get('port')!r}\n"
        f"viz start output: {result.output!r}"
    )


# ---------------------------------------------------------------------------
# WR-06: server-side resolution test (NO cmd_start mock)
# ---------------------------------------------------------------------------


def test_server_cmd_start_resolves_config_port(tmp_path: Path) -> None:
    """server.cmd_start(port=None) resolves viz.port from config (server-side).

    WR-06: the three tests above all patch `automil.viz.server.cmd_start`, so
    server.py's OWN `port is None` fallback branch is never exercised — the path
    that direct callers (and the legacy `viz.server.main()` shim) actually hit.
    This test calls cmd_start DIRECTLY with port=None against a config carrying
    `viz.port: 9001` and asserts the server resolves to 9001, intercepting at the
    aiohttp TCPSite construction so no server is ever bound/left running.
    """
    from automil.viz import server as viz_server  # noqa: PLC0415

    adir = tmp_path / "automil"
    adir.mkdir()
    (adir / "config.yaml").write_text(
        "run:\n  script: train.py\nviz:\n  port: 9001\n"
    )
    # orchestrator/ dir so cmd_start can place its PID/log files.
    (adir / "orchestrator").mkdir()

    captured: dict = {}

    class _StopBeforeBind(Exception):
        """Sentinel raised to abort cmd_start at TCPSite construction."""

    def fake_tcpsite(runner, host, port, **kwargs):  # noqa: ANN001
        captured["host"] = host
        captured["port"] = port
        raise _StopBeforeBind

    # Patch TCPSite so the resolved (host, port) is captured at the moment the
    # server would bind, then abort before any socket is opened. The filesystem
    # observer must not start before a successful bind.
    with (
        patch("automil.viz.server.web.TCPSite", side_effect=fake_tcpsite),
        patch("automil.viz.server.Observer") as observer_cls,
    ):
        observer_cls.return_value.is_alive.return_value = False
        try:
            viz_server.cmd_start(
                port=None, project_root=tmp_path, host="127.0.0.1"
            )
        except _StopBeforeBind:
            pass
        observer_cls.return_value.start.assert_not_called()

    assert captured.get("port") == 9001, (
        f"server-side resolution should read viz.port=9001 from config, "
        f"got port={captured.get('port')!r}"
    )
    assert captured.get("host") == "127.0.0.1"
    # No server left running: cmd_start's finally removed the PID file.
    assert not (adir / "orchestrator" / "viz_server.pid").exists(), (
        "viz_server.pid must not linger after the aborted cmd_start"
    )
