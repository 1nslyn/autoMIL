"""trajectory subgroup: record and export trajectory events (TRJ-04, TRJ-05 / D-94)."""
from __future__ import annotations

import click

from automil.paths import probe_exists

from automil.cli import main


@main.group("trajectory")
def trajectory_group() -> None:
    """Trajectory capture and export commands."""
    pass


@trajectory_group.command("record")
@click.argument("event_json")
def record(event_json: str) -> None:
    """Record one trajectory event (runtime-agnostic CLI fallback).

    EVENT_JSON: JSON string of the event dict, or @/path/to/file to read from file.

    Exit code 0 -- success (event recorded).
    Exit code 0 -- soft-fail (recorder error; WARNING logged to stderr).
    Exit code 1 -- hard error (JSON parse error OR missing AUTOMIL_NODE_ID env).
    """
    import json
    import os
    from pathlib import Path

    from automil.trajectory import record_event  # lazy import

    # @filepath convention: read event from file
    if event_json.startswith("@"):
        filepath = event_json[1:]
        try:
            event_json = Path(filepath).read_text(encoding="utf-8")
        except OSError as exc:
            raise click.ClickException(
                f"Cannot read event file {filepath!r}: {exc}"
            )

    # Parse JSON -- hard fail on parse error (exit 1)
    try:
        event = json.loads(event_json)
    except json.JSONDecodeError as exc:
        raise click.ClickException(
            f"JSON parse error: {exc}. "
            f"Event must be a valid JSON object. "
            f"Check the output from your hook script."
        )

    if not isinstance(event, dict):
        raise click.ClickException(
            f"Event must be a JSON object (got {type(event).__name__})"
        )

    # Missing AUTOMIL_NODE_ID -- hard fail (exit 1)
    node_id = os.environ.get("AUTOMIL_NODE_ID")
    if not node_id:
        raise click.ClickException(
            "AUTOMIL_NODE_ID environment variable is not set. "
            "trajectory record must be called within an autoMIL orchestrated session."
        )

    # Resolve archive_dir from AUTOMIL_DIR or walk up to find automil/config.yaml
    automil_dir_env = os.environ.get("AUTOMIL_DIR")
    if automil_dir_env:
        archive_dir = Path(automil_dir_env) / "archive"
    else:
        # Walk up from cwd to find automil/ directory
        cwd = Path.cwd()
        automil_dir = None
        for parent in [cwd] + list(cwd.parents):
            candidate = parent / "automil"
            if probe_exists(candidate / "config.yaml"):
                automil_dir = candidate
                break
        if automil_dir is None:
            # Soft-fail: cannot find automil dir; log warning and exit 0
            click.echo(
                f"WARNING: Cannot locate automil/ directory from {cwd}; "
                "event not recorded.",
                err=True,
            )
            return  # exit 0 (soft-fail)
        archive_dir = automil_dir / "archive"

    archive_dir.mkdir(parents=True, exist_ok=True)

    # record_event returns False on soft-fail -- we still exit 0 (D-94)
    success = record_event(
        node_id=node_id,
        event=event,
        archive_dir=archive_dir,
    )
    if not success:
        click.echo(
            "WARNING: trajectory event not recorded (recorder soft-fail); "
            "check trajectory.err.log for details.",
            err=True,
        )
    # Always exit 0 for success AND soft-fail (D-94)


@trajectory_group.command("export")
@click.argument("node_id")
@click.option(
    "--out",
    default=None,
    help="Output path for the bundle .tar.gz (default: <node_id>.trajectory.tar.gz in cwd)",
)
def export(node_id: str, out: str | None) -> None:
    """Produce a redacted, schema-validated trajectory bundle.

    Bundle: <node_id>.trajectory.tar.gz containing trajectory.jsonl +
    rotated siblings + manifest.json listing schema version + line counts +
    redaction-rule-set hash.
    """
    import os
    from pathlib import Path

    from automil.trajectory.export import export_bundle  # lazy import

    # T-03-09-S03 (Phase 3 security audit): node_id is a free-form click argument.
    # `archive_dir / node_id` with `node_id == "../../etc"` resolves to /etc; the
    # bundler would then package any `trajectory*.jsonl`-named files found there.
    # Reject path-like inputs at the CLI boundary — node_ids are graph identifiers
    # (e.g. "node_0176"), never paths.
    if not node_id or node_id in (".", "..") or os.path.isabs(node_id) \
       or ".." in Path(node_id).parts or "/" in node_id or "\\" in node_id:
        raise click.ClickException(
            f"Invalid node_id {node_id!r}: must be a graph identifier "
            f"(e.g. 'node_0176'), not a path."
        )

    automil_dir_env = os.environ.get("AUTOMIL_DIR")
    if automil_dir_env:
        archive_dir = Path(automil_dir_env) / "archive"
    else:
        cwd = Path.cwd()
        automil_dir = None
        for parent in [cwd] + list(cwd.parents):
            candidate = parent / "automil"
            if probe_exists(candidate / "config.yaml"):
                automil_dir = candidate
                break
        if automil_dir is None:
            raise click.ClickException(
                "Cannot locate automil/ directory. "
                "Run from within an autoMIL-initialized project."
            )
        archive_dir = automil_dir / "archive"

    out_path = Path(out) if out else None

    try:
        bundle_path = export_bundle(node_id, archive_dir, out_path)
        click.echo(f"Bundle created: {bundle_path}")
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc))
    except Exception as exc:
        raise click.ClickException(f"Export failed: {exc}")
