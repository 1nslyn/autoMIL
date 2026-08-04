"""resubmit command: re-queue a terminal node as a new node (CLI-04 / D-67).

Workflow:
  1. Look up node_id in graph.json (hard-fail if unknown).
  2. Hard-fail if node is not in a terminal state (completed | crashed | cancelled).
  3. Read overlay files from archive/<node_id>/ (exclude result.json + *_running_spec.json).
  4. Generate a NEW node_id via graph.next_id() — never reuse the old one (D-67 step 3).
  5. Read archived spec to reconstruct JobSpec fields.
  6. Resolve backend name from node.metadata.backend (default 'local', D-76).
  7. Instantiate backend; call backend.submit(new_spec) to get a handle.
  8. Insert new graph node: parent_id = <old_node>.parent_id,
     metadata.resubmitted_from = <old_node_id>.
  9. graph.save().
 10. Echo new_node_id to stdout (operator capture, D-67 step 7).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import click

from automil.cli import main
from automil.cli._helpers import _find_automil_dir, _find_git_root, _load_technique_map
from automil.cli.lifecycle._shared import _get_node_or_die

logger = logging.getLogger(__name__)

_TERMINAL_STATES = frozenset({"completed", "crashed", "cancelled", "budget_killed"})

# Overlay files to exclude when re-submitting: result data and running specs.
_EXCLUDE_SUFFIXES = frozenset({".tmp"})
_EXCLUDE_NAMES = frozenset({"result.json"})


def _is_overlay_file(p: Path) -> bool:
    """Return True if this archive file is an overlay source (not metadata/result)."""
    name = p.name
    if name in _EXCLUDE_NAMES:
        return False
    if name.endswith("_running_spec.json"):
        return False
    if p.suffix in _EXCLUDE_SUFFIXES:
        return False
    # Exclude spec.json and completion.json — these are orchestrator metadata.
    if name in ("spec.json", "completion.json"):
        return False
    return True


@main.command("resubmit")
@click.argument("node_id")
def resubmit(node_id: str) -> None:
    """Re-queue a terminal experiment as a new node with the same overlay.

    Creates a NEW node_id (never reuses the old one — preserves graph history).
    The new node carries metadata.resubmitted_from = <old_node_id> for traceability.
    The new node's parent_id is inherited from the original node.

    Reads the overlay from archive/<node_id>/. Calls Backend.submit(new_spec)
    via the backend registered for this node's metadata.backend field.

    Prints the new node_id to stdout for operator capture.

    Hard-fails if:
      - node_id is not in graph.json.
      - node is not in a terminal state (completed | crashed | cancelled).
      - archive/<node_id>/ does not exist or has no overlay files.
      - backend name is not in the BACKENDS registry.
    """
    # Lazy imports inside function body (PATTERNS.md §8 / D-69).
    from automil.backends import BACKENDS, JobSpec  # noqa: PLC0415
    from automil.backends.local import LocalBackend  # noqa: F401,PLC0415
    from automil.graph import ExperimentGraph  # noqa: PLC0415

    adir = _find_automil_dir()

    # Step 1: look up node — hard-fail if unknown.
    node = _get_node_or_die(adir, node_id)

    # Step 2: hard-fail if node is not in a terminal state.
    state = node.get("status", "")
    if state not in _TERMINAL_STATES:
        raise click.ClickException(
            f"Refusing to resubmit: node {node_id!r} is in state {state!r}. "
            f"Only terminal nodes (completed | crashed | cancelled) can be resubmitted. "
            f"Use `automil status` to check current state."
        )

    orch_dir = adir / "orchestrator"
    archive_node_dir = orch_dir / "archive" / node_id

    if not archive_node_dir.exists():
        raise click.ClickException(
            f"Refusing to resubmit: archive directory {archive_node_dir} does not exist. "
            f"The node's overlay may have been purged. "
            f"Use `automil propose` to create a new proposal instead."
        )

    # A direct backend resubmit does not pass through `automil submit`, so it
    # cannot mint a fresh admissibility record or a stage-budget identity. In a
    # fail-closed campaign mode, require a new proposal+submit instead of
    # silently creating a policy bypass for terminal overlays.
    from automil.admissibility import load_candidate_policy
    candidate_policy = load_candidate_policy(adir)
    if candidate_policy.mode == "architecture-preserving":
        raise click.ClickException(
            "Refusing to resubmit in architecture-preserving mode: this legacy "
            "path bypasses submit-time admissibility and staged attempt accounting. "
            "Create a new proposal and run `automil submit` so the candidate is "
            "reclassified, rehashed, and billed as a new attempt."
        )

    # Step 3: discover overlay files (exclude metadata and result files).
    overlay_paths: list[Path] = [
        p for p in archive_node_dir.iterdir()
        if p.is_file() and _is_overlay_file(p)
    ]

    if not overlay_paths:
        logger.warning(
            "resubmit: no overlay files found in %s — resubmitting with empty overlay",
            archive_node_dir,
        )

    overlay_file_names: tuple[str, ...] = tuple(
        p.name for p in sorted(overlay_paths)
    )

    # Read archived spec to reconstruct JobSpec fields.
    spec_path = archive_node_dir / "spec.json"
    archived_spec: dict = {}
    if spec_path.exists():
        try:
            archived_spec = json.loads(spec_path.read_text())
        except json.JSONDecodeError as exc:
            logger.warning("resubmit: could not parse spec.json for %s: %s", node_id, exc)

    # Recover fields from archived spec with sensible defaults.
    base_commit: str = archived_spec.get("base_commit", "HEAD")
    command_raw = archived_spec.get("command", ["python", "train.py"])
    command: tuple[str, ...] = tuple(command_raw) if isinstance(command_raw, list) else (command_raw,)
    env_raw = archived_spec.get("env", {})
    if isinstance(env_raw, dict):
        env: tuple[tuple[str, str], ...] = tuple(
            (k, v) for k, v in env_raw.items()
        )
    else:
        env = tuple()
    working_subdir: str = archived_spec.get("working_subdir", "")
    gpu_estimate_gb: float = float(archived_spec.get("estimated_vram_gb", 2.0))
    walltime_seconds: int = int(archived_spec.get("timeout_min", 60)) * 60

    # Step 4: generate a NEW node_id.
    graph_path = adir / "graph.json"
    graph = ExperimentGraph(path=str(graph_path), technique_map=_load_technique_map(adir))
    new_node_id: str = graph.next_id()

    # Step 5–6: resolve backend name.
    backend_name: str = node.get("metadata", {}).get("backend", "local")

    # Step 7: resolve and instantiate backend; submit.
    BackendClass = BACKENDS.get(backend_name)
    if BackendClass is None:
        raise click.ClickException(
            f"Unknown backend {backend_name!r}; available: {sorted(BACKENDS.keys())}. "
            f"Check automil/config.yaml or import the backend module first."
        )

    try:
        git_root = _find_git_root()
    except click.ClickException:
        git_root = adir.parent

    backend = BackendClass(project_root=git_root, automil_dir=adir)

    new_spec = JobSpec(
        node_id=new_node_id,
        base_commit=base_commit,
        overlay_files=overlay_file_names,
        overlay_dir=archive_node_dir,
        command=command,
        env=env,
        working_subdir=working_subdir,
        gpu_estimate_gb=gpu_estimate_gb,
        walltime_seconds=walltime_seconds,
    )
    handle = backend.submit(new_spec)
    logger.debug(
        "resubmit: submitted %s → handle.opaque_id=%s", new_node_id, handle.opaque_id
    )

    # Step 8: insert new graph node with resubmitted_from metadata.
    parent_id = node.get("parent_id")
    description = node.get("description", f"resubmit of {node_id}")
    techniques = node.get("techniques", [])

    new_node: dict = {
        "id": new_node_id,
        "parent_id": parent_id,
        "type": "proposed",
        "status": "running",
        "description": description,
        "techniques": techniques,
        "potential": 0.0,
        "metadata": {
            "backend": backend_name,
            "resubmitted_from": node_id,
            "resubmitted_at": __import__("datetime").datetime.now().isoformat(),
        },
        "created_at": __import__("datetime").datetime.now().isoformat(),
    }
    graph.nodes[new_node_id] = new_node

    # Step 9: save graph atomically.
    graph.save()

    # Step 10: echo new node_id for operator capture (D-67 step 7).
    click.echo(new_node_id)
