"""init command: scaffold autoMIL into an existing project."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import click

from automil.activity_hooks import (
    claude_activity_environment,
    claude_activity_hooks,
)
from automil.backends.base import HealthReport  # noqa: F401
from automil.cli import main
from automil.cli._helpers import _find_git_root


def _scaffold_variants_skeleton(automil_dir: Path) -> None:
    """Create automil/variants/<_losses|_policies|_candidates>/ + .gitkeep files.

    Per D-25 + REG-04 + REG-08 (deferred portion): each kind subdirectory
    gets a .gitkeep so the empty directory commits cleanly. The per-parent
    <parent>/ subdirectory is created by `port-variant` on first use
    (Plan 01-11), no parent name to assume at init time.

    Idempotent: re-running on an existing skeleton is a no-op (mkdir
    parents=True exist_ok=True; .gitkeep .touch is safe to repeat).
    """
    variants_root = automil_dir / "variants"
    variants_root.mkdir(parents=True, exist_ok=True)
    (variants_root / ".gitkeep").touch()
    for sub in ("_losses", "_policies", "_candidates"):
        sub_dir = variants_root / sub
        sub_dir.mkdir(parents=True, exist_ok=True)
        (sub_dir / ".gitkeep").touch()


def _register_claude_hooks(
    project_root: Path,
    project_claude: Path,
    package_dir: Path,
) -> None:
    """Register the Stop guard and exact Claude active-time observer.

    autoMIL-owned legacy/activity entries are normalized on every update while
    unrelated user settings are preserved.
    """
    settings_path = project_claude / "settings.json"
    stop_cmd = f"bash {project_root / '.claude' / 'hooks' / 'on_stop.sh'}"
    activity_hooks = claude_activity_hooks()

    if settings_path.exists():
        settings = json.loads(settings_path.read_text())
    else:
        project_claude.mkdir(parents=True, exist_ok=True)
        settings = {}

    original = json.dumps(settings, sort_keys=True)
    hooks = settings.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise click.ClickException(f"{settings_path}: hooks must be an object")
    env = settings.setdefault("env", {})
    if not isinstance(env, dict):
        raise click.ClickException(f"{settings_path}: env must be an object")
    required_env = claude_activity_environment()
    preserved = {
        key: env[key]
        for key in required_env
        if key in env and env[key] != required_env[key]
    }
    for key, value in required_env.items():
        env.setdefault(key, value)
    if preserved:
        click.echo(
            "warning: keeping existing telemetry env "
            + ", ".join(f"{k}={v!r}" for k, v in sorted(preserved.items()))
            + "; native active-time metering requires "
            + ", ".join(f"{k}={v}" for k, v in sorted(required_env.items()))
            + " — agent_active cells hold admission until the exporter matches",
            err=True,
        )

    # Remove every obsolete on_tool.sh registration and normalize any prior
    # activity-ingest registration to the exact event/matcher matrix below.
    for event, entries in list(hooks.items()):
        if not isinstance(entries, list):
            continue
        kept_entries: list[dict] = []
        for entry in entries:
            if not isinstance(entry, dict):
                kept_entries.append(entry)
                continue
            commands = entry.get("hooks")
            if not isinstance(commands, list):
                kept_entries.append(entry)
                continue
            kept_commands = [
                hook for hook in commands
                if not (
                    isinstance(hook, dict)
                    and isinstance(hook.get("command"), str)
                    and (
                        "on_tool.sh" in hook["command"]
                        or hook["command"].endswith("automil activity ingest")
                    )
                )
            ]
            if kept_commands:
                normalized = dict(entry)
                normalized["hooks"] = kept_commands
                kept_entries.append(normalized)
        if kept_entries:
            hooks[event] = kept_entries
        else:
            del hooks[event]

    def _ensure(event: str, command: str, *, matcher: str | None = None) -> None:
        """Append a hook entry for *event* if *command* isn't already present."""
        entries = hooks.setdefault(event, [])
        if any(command in str(e) for e in entries):
            return
        hook_obj = {"type": "command", "command": command}
        entry: dict = {"hooks": [hook_obj]}
        if matcher is not None:
            entry["matcher"] = matcher
        entries.append(entry)

    _ensure("Stop", stop_cmd)
    for event, entries in activity_hooks.items():
        hooks.setdefault(event, []).extend(entries)

    if json.dumps(settings, sort_keys=True) != original:
        settings_path.write_text(json.dumps(settings, indent=2) + "\n")


def _install_runtime_assets(
    rt: str,
    project_root: Path,
    package_dir: Path,
    merge_skill,  # callable: (runtime, shared_path, overlay_path) -> str
    update: bool = False,
) -> None:
    """Install per-runtime overlay assets (skill, hooks, native files).

    - Skill files into .claude/skills/ (for claude runtime)
    - Runtime hook scripts and settings registrations
    - Native runtime instruction files (CLAUDE.md, .opencode/AGENTS.md, etc.)
    """
    shared_dir = package_dir / "agent_assets" / "_shared"
    overlay_dir = package_dir / "agent_assets" / rt

    # Copy skill files from _shared/skills into runtime-appropriate location
    skills_src = shared_dir / "skills"

    if rt == "claude":
        project_claude = project_root / ".claude"
        project_claude.mkdir(parents=True, exist_ok=True)

        # Install skills into .claude/skills/ using merge_skill for overlay
        if skills_src.exists():
            for skill_dir in skills_src.iterdir():
                if skill_dir.is_dir():
                    dst_dir = project_claude / "skills" / skill_dir.name
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    shared_skill = skill_dir / "SKILL.md"
                    if shared_skill.exists():
                        overlay_skill_path = overlay_dir / "skills" / skill_dir.name / "SKILL.md"
                        overlay_arg = overlay_skill_path if overlay_skill_path.exists() else None
                        merged = merge_skill(rt, shared_skill, overlay_arg)
                        dst = dst_dir / "SKILL.md"
                        # WR-04: skills are framework-managed (users customize
                        # program.md, not SKILL.md). Honor --update so updated
                        # framework guidance reaches existing projects; without
                        # this, a stale skill could never refresh (the original
                        # bug: anti-sweep directives never reached the agent).
                        existed = dst.exists()
                        if not existed or update:
                            dst.write_text(merged, encoding="utf-8")
                            verb = "Updated" if (existed and update) else "Installed"
                            click.echo(f"  {verb}: .claude/skills/{skill_dir.name}/SKILL.md")

        # Copy hook scripts from agent_assets/claude/hooks
        hooks_src = package_dir / "agent_assets" / "claude" / "hooks"
        if hooks_src.exists():
            hooks_dst = project_claude / "hooks"
            hooks_dst.mkdir(parents=True, exist_ok=True)
            for f in hooks_src.iterdir():
                if f.is_file():
                    dst = hooks_dst / f.name
                    # Hooks are framework-managed; honor --update so new/changed
                    # hook scripts (e.g. the Stop guard) reach existing
                    # projects without a manual reinstall.
                    if not dst.exists() or update:
                        shutil.copy2(f, dst)
                        if f.suffix == ".sh":
                            dst.chmod(dst.stat().st_mode | 0o111)

        # on_tool.sh was the lossy timestamp adapter. Native activity metrics
        # need only lifecycle hooks, so --update removes any installed copy.
        obsolete_hook = project_claude / "hooks" / "on_tool.sh"
        if update and obsolete_hook.exists():
            obsolete_hook.unlink()

        # .claude/CLAUDE.md: first line @AGENTS.md (D-90)
        claude_md = project_claude / "CLAUDE.md"
        if not claude_md.exists():
            claude_md.write_text(
                "@AGENTS.md\n\n# Claude Code: autoMIL\n\nSee AGENTS.md above for universal instructions.\n",
                encoding="utf-8",
            )
            click.echo("  Created: .claude/CLAUDE.md")

        # Register the Stop guard and activity journal in settings.json.
        _register_claude_hooks(project_root, project_claude, package_dir)
        click.echo("  Runtime: claude, assets installed")

    elif rt == "opencode":
        opencode_dir = project_root / ".opencode"
        opencode_dir.mkdir(parents=True, exist_ok=True)
        # .opencode/AGENTS.md: same universal content as project-root AGENTS.md
        agents_src = shared_dir / "AGENTS.md"
        if agents_src.exists():
            dst = opencode_dir / "AGENTS.md"
            if not dst.exists():
                dst.write_text(agents_src.read_text(encoding="utf-8"), encoding="utf-8")
        # .opencode/plugins/automil-trajectory.ts (Bun TypeScript plugin)
        plugins_dir = opencode_dir / "plugins"
        plugins_dir.mkdir(exist_ok=True)
        ts_src = package_dir / "agent_assets" / "opencode" / "plugins" / "automil-trajectory.ts"
        if ts_src.exists():
            (plugins_dir / "automil-trajectory.ts").write_text(ts_src.read_text(encoding="utf-8"), encoding="utf-8")
            click.echo("  Plugin: .opencode/plugins/automil-trajectory.ts")
        click.echo("  Runtime: opencode, assets installed (.opencode/AGENTS.md + plugin)")

    elif rt == "codex":
        codex_dir = project_root / ".codex"
        codex_dir.mkdir(parents=True, exist_ok=True)
        # .codex/instructions.md
        instructions = codex_dir / "instructions.md"
        if not instructions.exists():
            agents_src = shared_dir / "AGENTS.md"
            content = (
                agents_src.read_text(encoding="utf-8")
                if agents_src.exists()
                else "# autoMIL\n\nSee AGENTS.md.\n"
            )
            instructions.write_text(content, encoding="utf-8")
        click.echo("  Runtime: codex, assets installed (.codex/instructions.md)")

    elif rt in ("deepseek-via-opencode", "deepseek-via-codex"):
        base_rt = "opencode" if "opencode" in rt else "codex"
        click.echo(f"  Runtime: {rt}. DeepSeek is a model; installing {base_rt} overlay")
        _install_runtime_assets(base_rt, project_root, package_dir, merge_skill, update)


def _format_health_report(report: "HealthReport") -> str:
    """Render HealthReport as a human-readable stdout block (D-191)."""
    vram_str = ", ".join(f"{v:.1f}" for v in report.gpu_vram_gb) or "(none)"
    lines = [
        "automil hardware probe:",
        f"  accelerator:        {report.accelerator}",
        f"  gpu_count:          {report.gpu_count}",
        f"  gpu_vram_gb:        {vram_str}",
        f"  python_version:     {report.python_version}",
        f"  automil_version:    {report.automil_version}",
        f"  detection_status:   {report.detection_status}",
    ]
    for w in report.detection_warnings:
        lines.append(f"  warning: {w}")
    return "\n".join(lines)


def _stamp_healthcheck_defaults(
    automil_dir: Path,
    report: "HealthReport | None",
) -> dict:
    """Compute Jinja2 context for config.yaml.j2 stamping (D-191 / STP-02 / OQ-2).

    Behavior matrix:
      - report is None  (--no-healthcheck): conservative defaults.
      - report.gpu_count == 0: conservative defaults; CPU shape.
      - report.gpu_count > 0 + results.tsv missing or <10 rows: conservative
        from min(gpu_vram_gb) / 8.0 floor 8.0.
      - report.gpu_count > 0 + results.tsv >=10 rows: empirical
        numpy.quantile(vram_gb_col, 0.95).

    vram_gb is the column header in results.tsv per
    src/automil/backends/_orchestrator_daemon.py:1289 (orchestrator converts
    peak_vram_mb to vram_gb at write time).
    """
    if report is None or report.gpu_count == 0:
        gpu_count = 0 if report is None else report.gpu_count
        accelerator = "cpu" if report is None else report.accelerator
        return {
            "healthcheck_gpu_count": gpu_count,
            "healthcheck_vram_gb": 0.0,
            "healthcheck_accelerator": accelerator,
            "max_concurrent_per_gpu": 4,
            "default_vram_estimate_gb": 8.0,
        }

    min_vram = min(report.gpu_vram_gb) if report.gpu_vram_gb else 8.0
    conservative_vram = max(8.0, min_vram / 8.0)
    max_concurrent = max(1, int(min_vram // conservative_vram)) if min_vram >= conservative_vram else 1

    results_tsv = automil_dir / "results.tsv"
    empirical_vram: float | None = None
    if results_tsv.exists():
        try:
            with open(results_tsv, "r") as f:
                reader = csv.DictReader(f, delimiter="\t")
                vram_values: list[float] = []
                for row in reader:
                    raw = row.get("vram_gb")
                    if raw is None or raw == "":
                        continue
                    try:
                        v = float(raw)
                        if v > 0:
                            vram_values.append(v)
                    except ValueError:
                        continue
            if len(vram_values) >= 10:
                try:
                    import numpy  # noqa: PLC0415
                    empirical_vram = float(numpy.quantile(vram_values, 0.95))
                except ImportError:
                    import statistics  # noqa: PLC0415
                    qs = statistics.quantiles(vram_values, n=20)
                    empirical_vram = qs[18] if len(qs) >= 19 else max(vram_values)
        except (OSError, ValueError):
            empirical_vram = None

    default_vram = empirical_vram if empirical_vram is not None else conservative_vram

    return {
        "healthcheck_gpu_count": report.gpu_count,
        "healthcheck_vram_gb": min_vram,
        "healthcheck_accelerator": report.accelerator,
        "max_concurrent_per_gpu": max_concurrent,
        "default_vram_estimate_gb": float(default_vram),
    }


@main.command()
@click.argument("path", default="automil")
@click.option("--task", default="binary", help="Task type: binary or multiclass")
@click.option("--encoder", default="hoptimus1", help="Primary encoder name")
@click.option(
    "--runtime",
    default=None,
    type=click.Choice(
        ["claude", "opencode", "codex", "deepseek-via-opencode", "deepseek-via-codex", "all"]
    ),
    help=(
        "Runtime to install assets for "
        "(default: auto-detect from existing .claude/.opencode/.codex dirs)"
    ),
)
@click.option(
    "--update",
    is_flag=True,
    default=False,
    help="Re-render skills/hooks/AGENTS.md for installed runtimes without re-scaffolding",
)
@click.option(
    "--no-healthcheck",
    is_flag=True,
    default=False,
    help="Skip hardware probe (CI / smoke-test path); use conservative defaults.",
)
def init(path: str, task: str, encoder: str, runtime: str | None, update: bool, no_healthcheck: bool) -> None:
    """Add autoMIL to an existing project."""
    from jinja2 import Environment, FileSystemLoader

    project_root = Path.cwd()
    automil_dir = project_root / path

    # Verify we're inside a git repo (can be a subdirectory)
    try:
        _find_git_root(project_root)
    except click.ClickException:
        raise click.ClickException(
            "Not inside a git repository. Run 'git init' or cd into your project."
        )

    # D-92: --update bypasses the already-initialized guard
    if automil_dir.exists() and (automil_dir / "config.yaml").exists():
        if not update:
            raise click.ClickException(f"autoMIL already initialized at {automil_dir}")
        # --update: skip scaffold, proceed to asset re-install

    # D-191 / STP-02 / STP-03: hardware probe between --update guard and template render.
    # The probe surfaces report values; CLI prompts user override on detection failure
    # (NEVER silently uses wrong defaults).
    report: "HealthReport | None"
    if no_healthcheck:
        report = None
        click.echo("automil init: hardware probe skipped (--no-healthcheck).")
    else:
        from automil.backends.local import LocalBackend  # noqa: PLC0415
        report = LocalBackend(project_root=project_root, automil_dir=automil_dir).healthcheck()
        click.echo(_format_health_report(report))
        if report.detection_status == "failed":
            if not click.confirm(
                "Detection failed; use conservative defaults?",
                default=False,
            ):
                raise click.ClickException(
                    "Healthcheck failed and operator declined conservative defaults. "
                    "Run `automil init --no-healthcheck` to skip the probe, or fix "
                    "GPU drivers and retry."
                )
    healthcheck_context = _stamp_healthcheck_defaults(automil_dir, report)

    # Prepare template environment and context (used for both first-time init and --update re-stamp).
    templates_dir = Path(__file__).parent.parent / "templates"
    env = Environment(loader=FileSystemLoader(str(templates_dir)))
    context: dict = {
        "task_type": task,
        "encoder": encoder,
        "project_name": project_root.name,
    }
    context.update(healthcheck_context)

    if not update:
        # Create directory structure
        automil_dir.mkdir(parents=True, exist_ok=True)
        for subdir in [
            "orchestrator/queue",
            "orchestrator/running",
            "orchestrator/archive",
            "orchestrator/completed",
        ]:
            (automil_dir / subdir).mkdir(parents=True, exist_ok=True)

        # Scaffold the registry variants/ skeleton (D-25, REG-04).
        _scaffold_variants_skeleton(automil_dir)

        # Render templates. Templates live alongside the package (not the cli/
        # subpackage), so resolve relative to ``automil/`` rather than ``cli/``.
        for template_name, target_name in [
            ("config.yaml.j2", "config.yaml"),
            ("program.md.j2", "program.md"),
            ("plan.md.j2", "plan.md"),
            ("learnings.md.j2", "learnings.md"),
            (".gitignore.j2", ".gitignore"),
        ]:
            template = env.get_template(template_name)
            (automil_dir / target_name).write_text(template.render(**context))
    elif not no_healthcheck:
        # --update with healthcheck: re-render config.yaml to stamp fresh detected values (D-191).
        # Note: this replaces the file; user edits in other sections are not preserved.
        # Use --no-healthcheck with --update to preserve existing config.yaml verbatim.
        config_template = env.get_template("config.yaml.j2")
        (automil_dir / "config.yaml").write_text(config_template.render(**context))

    # D-91: auto-detect runtime from existing dirs if --runtime not specified
    if runtime is None:
        detected = []
        if (project_root / ".claude").exists():
            detected.append("claude")
        if (project_root / ".opencode").exists():
            detected.append("opencode")
        if (project_root / ".codex").exists():
            detected.append("codex")
        if detected:
            runtimes_to_install = detected
        else:
            runtimes_to_install = ["claude"]  # default: Claude Code
            click.echo(
                "No runtime config detected; installing Claude Code overlay. "
                "Use --runtime to override or --runtime all for full multi-runtime support."
            )
    elif runtime == "all":
        runtimes_to_install = ["claude", "opencode", "codex"]
    else:
        runtimes_to_install = [runtime]

    # D-92: lazy import of overlay merger inside command body
    from automil.agent_assets._overlay import merge_skill  # noqa: E402

    package_dir = Path(__file__).parent.parent

    # Render project-root AGENTS.md once per init invocation (D-90)
    # WR-04 (Phase 3 review): never silently overwrite a user-customised
    # project-root AGENTS.md. Match the `if not dst.exists()` guard used for
    # every other asset; only --update is allowed to overwrite.
    agents_shared = package_dir / "agent_assets" / "_shared" / "AGENTS.md"
    agents_dst = project_root / "AGENTS.md"
    if agents_shared.exists():
        if not agents_dst.exists() or update:
            agents_content = agents_shared.read_text(encoding="utf-8")
            agents_dst.write_text(agents_content, encoding="utf-8")
            verb = "Updated" if (agents_dst.exists() and update) else "Created"
            click.echo(f"  {verb}: AGENTS.md")
        else:
            click.echo("  Skipped: AGENTS.md (exists; use --update to overwrite)")

    # Install per-runtime assets
    for rt in runtimes_to_install:
        _install_runtime_assets(rt, project_root, package_dir, merge_skill, update)

    click.echo(f"autoMIL initialized at {automil_dir}/")
    click.echo("Next steps:")
    click.echo(f"  1. Edit {automil_dir}/config.yaml with your project settings")
    click.echo(f"  2. Run: uv run automil orchestrator start")
    runtimes_display = ", ".join(runtimes_to_install)
    click.echo(f"  3. Start your coding agent ({runtimes_display} -> /automil-setup)")
