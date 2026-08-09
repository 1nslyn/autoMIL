"""orchestrator subgroup: start, stop, status."""
from __future__ import annotations

import click

from automil.cli import main
from automil.cli._helpers import _find_automil_dir, _find_git_root


@main.group(name="orchestrator")
def orchestrator_group():
    """Manage the GPU scheduler daemon."""
    pass


@orchestrator_group.command("start")
def orch_start():
    """Start the orchestrator daemon."""
    from automil.orchestrator import ExperimentOrchestrator
    try:
        orch = ExperimentOrchestrator(
            project_root=_find_git_root(), automil_dir=_find_automil_dir(),
        )
        orch.cmd_start()
    except ValueError as exc:
        # e.g. a malformed or empty AUTOMIL_VISIBLE_GPUS partition — a
        # refusal, not a traceback.
        raise click.ClickException(str(exc)) from exc


@orchestrator_group.command("stop")
def orch_stop():
    """Stop the orchestrator daemon."""
    from automil.orchestrator import ExperimentOrchestrator
    orch = ExperimentOrchestrator(
        project_root=_find_git_root(), automil_dir=_find_automil_dir(),
    )
    orch.cmd_stop()


@orchestrator_group.command("status")
def orch_status():
    """Show orchestrator status."""
    from automil.orchestrator import ExperimentOrchestrator
    orch = ExperimentOrchestrator(
        project_root=_find_git_root(), automil_dir=_find_automil_dir(),
    )
    orch.cmd_status()
