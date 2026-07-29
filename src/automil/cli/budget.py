"""budget: view and set the cell budget cap (P2.3).

``automil budget show`` resolves the effective cap (budget / safety-buffer /
mode / idle-grace) from config.yaml and lists each cell's consumed vs remaining.

``automil budget set 6h`` writes ``cap.budget`` into config.yaml using a
comment-preserving line edit (so the template's annotations survive). Per D-134
the new value applies only to cells opened AFTER the change — existing cells keep
their budget.
"""
from __future__ import annotations

import re

import click

from automil.cli import main
from automil.cli._helpers import _find_automil_dir

_CAP_LINE = re.compile(r"^cap:\s*(#.*)?$")
_KEY_LINE = re.compile(r"^(\s+)([A-Za-z_][\w-]*):(.*)$")


def _fmt_hms(seconds: float) -> str:
    h, rem = divmod(int(max(0, seconds)), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _apply_cap_updates(text: str, updates: dict[str, str],
                       drop_keys: frozenset[str] = frozenset()) -> str:
    """Set ``cap.<key>: <value>`` lines in config.yaml text, preserving comments.

    Replaces existing keys in place; inserts new keys just after ``cap:``; drops
    any key in *drop_keys* (used to retire a legacy ``*_seconds`` twin). Appends a
    fresh ``cap:`` block if none exists.
    """
    lines = text.splitlines()
    cap_idx = next((i for i, ln in enumerate(lines) if _CAP_LINE.match(ln)), None)

    if cap_idx is None:
        block = ["", "cap:"] + [f"  {k}: {v}" for k, v in updates.items()]
        out = "\n".join(lines + block)
        return out + ("" if out.endswith("\n") else "\n")

    applied: set[str] = set()
    key_indent = "  "
    out_block: list[str] = []
    j = cap_idx + 1
    while j < len(lines):
        line = lines[j]
        if line.strip() == "" or line.lstrip().startswith("#"):
            out_block.append(line)
            j += 1
            continue
        if len(line) - len(line.lstrip()) == 0:
            break  # dedent → end of cap block
        m = _KEY_LINE.match(line)
        if m:
            key_indent, key = m.group(1), m.group(2)
            if key in drop_keys:
                j += 1
                continue
            if key in updates and key not in applied:
                out_block.append(f"{m.group(1)}{key}: {updates[key]}")
                applied.add(key)
                j += 1
                continue
        out_block.append(line)
        j += 1

    inserts = [f"{key_indent}{k}: {v}" for k, v in updates.items() if k not in applied]
    new_lines = lines[:cap_idx + 1] + inserts + out_block + lines[j:]
    out = "\n".join(new_lines)
    return out + ("" if out.endswith("\n") else "\n")


@main.group("budget")
def budget_group() -> None:
    """Inspect or set the cell budget cap."""
    pass


@budget_group.command("show")
def budget_show() -> None:
    """Show the resolved cap and per-cell consumed/remaining."""
    import yaml

    from automil.cells import consumed_seconds, format_duration, list_cells
    from automil.cells.capconfig import resolve_cap_config

    adir = _find_automil_dir()
    cfg_path = adir / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    try:
        cap = resolve_cap_config(cfg)
    except ValueError as exc:
        raise click.ClickException(str(exc))

    click.echo(f"budget:        {format_duration(cap.budget_seconds)} ({cap.budget_seconds}s)")
    click.echo(f"safety_buffer: {format_duration(cap.safety_buffer_seconds)} ({cap.safety_buffer_seconds}s)")
    click.echo(f"mode:          {cap.mode}")
    if cap.mode == "agent_active":
        click.echo(f"idle_grace:    {format_duration(cap.idle_grace_seconds)} ({cap.idle_grace_seconds}s)")
    # H-2: the eval axis is the primary equal-effort comparison axis; time is the
    # safety wall. Report it first-class so an operator can see which one binds.
    click.echo(
        f"eval_budget:   {cap.eval_budget if cap.eval_budget is not None else 'none (time-only)'}"
    )

    cells = list_cells()
    if not cells:
        return
    click.echo("")
    click.echo(
        f"{'cell_id':<10}{'status':<14}{'consumed':<11}{'remaining':<11}"
        f"{'evals':<10}{'usable':<8}"
    )
    click.echo("-" * 64)
    for c in cells:
        consumed = consumed_seconds(c)
        remaining = max(0.0, c.budget_seconds - consumed)
        evals = f"{c.consumed_evals}/{c.eval_budget if c.eval_budget is not None else '-'}"
        click.echo(
            f"{c.cell_id[:8]:<10}{c.status.value:<14}"
            f"{_fmt_hms(consumed):<11}{_fmt_hms(remaining):<11}"
            f"{evals:<10}{c.completed_evals:<8}"
        )


@budget_group.command("set")
@click.argument("duration")
@click.option("--safety-buffer", default=None, help="Also set the safety buffer (e.g. 30m).")
@click.option("--mode", type=click.Choice(["agent_active", "wall_clock"]), default=None,
              help="Billing mode for newly-opened cells.")
@click.option("--idle-grace", default=None,
              help="agent_active idle grace, e.g. 5m (how long after the agent's "
                   "last action the clock keeps running).")
@click.option("--eval-budget", default=None,
              help="Evaluations a cell may launch — the primary equal-effort axis "
                   "(H-2). An integer count, or 'none' for time-only.")
def budget_set(duration: str, safety_buffer: str | None, mode: str | None,
               idle_grace: str | None, eval_budget: str | None) -> None:
    """Set cap.budget in config.yaml, e.g. `automil budget set 6h`.

    Applies to cells opened AFTER this change (existing cells keep their budget
    per D-134). 6h is just the autoMIL-paper default — set whatever fits.
    """
    from automil.cells.capconfig import parse_duration, parse_eval_budget

    adir = _find_automil_dir()
    cfg_path = adir / "config.yaml"
    if not cfg_path.exists():
        raise click.ClickException(f"No config.yaml at {cfg_path}")

    updates: dict[str, str] = {}
    drop: set[str] = set()

    try:
        budget_s = parse_duration(duration)
        updates["budget"] = duration
        drop.add("budget_seconds")
        if safety_buffer is not None:
            buffer_s = parse_duration(safety_buffer)
            updates["safety_buffer"] = safety_buffer
            drop.add("safety_buffer_seconds")
            if not (0 < buffer_s < budget_s):
                raise click.ClickException(
                    f"safety buffer must satisfy 0 < buffer < budget "
                    f"(got buffer={buffer_s}s, budget={budget_s}s)"
                )
        if idle_grace is not None:
            updates["idle_grace"] = idle_grace
            drop.add("idle_grace_seconds")
            parse_duration(idle_grace)
        if eval_budget is not None:
            if eval_budget.strip().lower() in ("none", "null", ""):
                updates["eval_budget"] = "null"
            else:
                updates["eval_budget"] = str(parse_eval_budget(eval_budget))
    except ValueError as exc:
        raise click.ClickException(str(exc))

    if budget_s <= 0:
        raise click.ClickException(f"budget must be > 0 (got {budget_s}s)")
    if mode is not None:
        updates["mode"] = mode

    new_text = _apply_cap_updates(cfg_path.read_text(), updates, frozenset(drop))
    cfg_path.write_text(new_text)

    summary = ", ".join(f"{k}={v}" for k, v in updates.items())
    click.echo(f"Updated config.yaml cap: {summary}.")
    click.echo("Applies to newly-opened cells (existing cells keep their budget, D-134).")
