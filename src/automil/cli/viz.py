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
@click.option(
    "--tz", "tz_name", default=None,
    help="IANA zone of the host that ran the project, for a record copied from elsewhere (default: this host's zone).",
)
def viz_start(port: int | None, host: str | None, tz_name: str | None):
    """Start the dashboard: the site with this project as its run."""
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
    cmd_start(port=port, project_root=adir.parent, host=host, tz_name=tz_name)


@viz_group.command("export")
@click.option("--out", "out", required=True, type=click.Path(), help="Directory for the site (index.html, static/, record/).")
@click.option("--run-id", "run_id", default=None, help="Run id in the record (default: the project name).")
@click.option("--title", default=None, help="Run title shown on the site (default: the project description).")
@click.option("--tz", "tz_name", default=None, help="IANA zone of the host that ran the project (default: this host's zone).")
@click.option("--force", is_flag=True, help="Write into a non-empty directory that holds no record.")
@click.option("--frontend/--no-frontend", "with_frontend", default=True, help="Copy the page and static assets next to record/ (default) or write only record/.")
@click.option("--single-file", "single", default=None, type=click.Path(), help="Also write one self-contained HTML file.")
def viz_export(out: str, run_id: str | None, title: str | None, tz_name: str | None, force: bool, with_frontend: bool, single: str | None):
    """Write this project's record (and the site around it) to a directory."""
    from pathlib import Path  # noqa: PLC0415

    from automil.viz.clock import host_clock  # noqa: PLC0415
    from automil.viz.export import ExportError, export_site, single_file  # noqa: PLC0415
    from automil.viz.record import RunSource  # noqa: PLC0415

    adir = _find_automil_dir()
    try:
        clock = host_clock(tz_name=tz_name)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    source = RunSource(adir, clock, run_id=run_id, title=title, mode="static")
    try:
        summaries = export_site([source], Path(out), force=force, with_frontend=with_frontend)
    except ExportError as exc:
        raise click.ClickException(str(exc)) from exc
    for summary in summaries:
        click.echo(
            f"exported {summary.run_id}: {summary.nodes} nodes, {summary.sessions} session(s), "
            f"{summary.files} files -> {summary.out}"
        )
        for warning in summary.warnings:
            click.echo(f"warning: {warning}", err=True)
    if single:
        click.echo(f"single file -> {single_file(Path(single), Path(out))}")


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
