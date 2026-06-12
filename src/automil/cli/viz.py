"""viz subgroup: start, stop, status."""
from __future__ import annotations

import click

from automil.cli import main
from automil.cli._helpers import _find_automil_dir


@main.group(name="viz")
def viz_group():
    """Manage the visualization dashboard."""
    pass


@viz_group.command("start")
@click.option(
    "--port",
    default=None,
    type=int,
    help="Server port (default: viz.port in automil/config.yaml, then 8420).",
)
@click.option(
    "--host", default=None,
    help="Bind address (default: 127.0.0.1; falls back to viz.host in "
         "automil/config.yaml then AUTOMIL_VIZ_HOST env var). Pass 0.0.0.0 "
         "only on trusted networks — the dashboard exposes PIDs and node "
         "descriptions and has no auth.",
)
def viz_start(port: int | None, host: str | None):
    """Start the 3D visualization dashboard."""
    from automil.viz.server import DEFAULT_PORT, cmd_start  # noqa: PLC0415
    adir = _find_automil_dir()
    # Port resolution order: explicit --port > viz.port in config > DEFAULT_PORT (8420).
    # Resolution happens here (CLI layer) so cmd_start receives a resolved int
    # regardless of whether it is called directly or via the CLI.
    if port is None:
        config_path = adir / "config.yaml"
        cfg_port: int | None = None
        if config_path.exists():
            try:
                import yaml as _yaml  # noqa: PLC0415
                _cfg = _yaml.safe_load(config_path.read_text()) or {}
                raw = (_cfg.get("viz") or {}).get("port")
                if raw is not None:
                    cfg_port = int(raw)
            except Exception:  # noqa: BLE001
                cfg_port = None
        port = cfg_port if cfg_port is not None else DEFAULT_PORT
    cmd_start(port=port, project_root=adir.parent, host=host)


@viz_group.command("stop")
def viz_stop():
    """Stop the visualization dashboard."""
    adir = _find_automil_dir()
    from automil.viz.server import cmd_stop
    cmd_stop(project_root=adir.parent)


@viz_group.command("status")
def viz_status():
    """Show visualization server status."""
    adir = _find_automil_dir()
    from automil.viz.server import cmd_status
    cmd_status(project_root=adir.parent)
