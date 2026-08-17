"""port-variant command: convert overlay to registered variant module + manifest (CLI-05 / D-43, D-44)."""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import click

from automil.cli import main
from automil.cli._helpers import _find_automil_dir, _load_technique_map
from automil.cli.lifecycle._shared import (
    _atomic_write_text, _get_node_or_die, _load_registry_or_die,
)

logger = logging.getLogger(__name__)


def _read_spec_json(adir: Path, node_id: str) -> dict:
    """Read orchestrator/archive/<node_id>/spec.json. Hard-fail if missing."""
    spec_path = adir / "orchestrator" / "archive" / node_id / "spec.json"
    if not spec_path.exists():
        raise click.ClickException(
            f"orchestrator/archive/{node_id}/spec.json not found at {spec_path}. "
            f"Either {node_id} was never submitted or its archive was cleaned up. "
            f"Run `automil submit --node {node_id} ...` to recreate."
        )
    try:
        return json.loads(spec_path.read_text())
    except json.JSONDecodeError as e:
        raise click.ClickException(
            f"orchestrator/archive/{node_id}/spec.json is malformed JSON: {e}"
        ) from e


def _derive_kind(overlay_paths: list[str], protected: tuple[str, ...]) -> str | None:
    """Heuristic: if any overlay path matches a 'model' shape (`model`, `aggregator`,
    `clam_mb`, etc.) -> 'model'; if 'loss' or 'core_utils' -> 'loss'; if 'optimizer'
    or 'training' or 'sam' -> 'policy'.

    Returns None if ambiguous (multiple kinds matched) — caller falls back to --kind.
    """
    kinds_matched: set[str] = set()
    model_signals = ("model", "aggregator", "clam_mb", "ab_mil", "dsmil", "transmil")
    loss_signals = ("loss", "core_utils")  # CLAM hides smoothing inside core_utils
    policy_signals = ("optimizer", "policy", "training", "sam", "lookahead")

    for path in overlay_paths:
        path_lower = path.lower()
        if any(sig in path_lower for sig in model_signals):
            kinds_matched.add("model")
        if any(sig in path_lower for sig in loss_signals):
            kinds_matched.add("loss")
        if any(sig in path_lower for sig in policy_signals):
            kinds_matched.add("policy")

    if len(kinds_matched) == 1:
        return next(iter(kinds_matched))
    return None  # ambiguous or none


def _node_id_short(node_id: str) -> str:
    """`node_0176` -> `0176`. Falls back to the full node_id if no number found."""
    m = re.search(r"_(\d+)$", node_id)
    return m.group(1) if m else node_id


def _detect_parent(spec_json: dict, overlay_paths: list[str]) -> Optional[str]:
    """Best-effort parent detection from spec.json.graph_metadata or overlay paths.

    For Phase 1, accept `--parent` override; auto-detect is best-effort only.
    """
    parent = spec_json.get("graph_metadata", {}).get("parent_id")
    if parent:
        return parent
    # Heuristic from overlay path.
    for path in overlay_paths:
        for known in ("clam_mb", "ab_mil", "dsmil", "transmil", "dtfd_mil"):
            if known in path.lower():
                return known
    return None


def _write_variant_module(
    target_path: Path,
    spec,
    source_overlay_files: list[str],
) -> None:
    """Write the .py module body. For Phase 1, this is a minimal stub that
    subclasses the appropriate ABC; the consumer is responsible for pasting in
    the actual variant code (D-37: CCRCC byte-identical-port deferred).

    The body:
      - imports the matching ABC and @register
      - has a docstring with the D-44 header schema
      - declares the class with a stub forward / __call__ / wrap_optimizer
      - has a TODO comment pointing to the operator's task
    """
    abc_map = {"model": "ModelVariant", "loss": "LossVariant", "policy": "PolicyVariant"}
    abc_name = abc_map[spec.kind]

    if spec.kind == "model":
        body_method = (
            "    def forward(self, features, coords=None):\n"
            "        # TODO: paste the variant's forward body here.\n"
            "        # See sources: " + ", ".join(source_overlay_files) + "\n"
            "        raise NotImplementedError(\"variant body not yet ported\")\n"
        )
    elif spec.kind == "loss":
        body_method = (
            "    def __call__(self, logits, targets, *, instance_logits=None, instance_labels=None):\n"
            "        # TODO: paste the variant's loss body here.\n"
            "        # See sources: " + ", ".join(source_overlay_files) + "\n"
            "        raise NotImplementedError(\"variant body not yet ported\")\n"
        )
    else:  # policy
        body_method = (
            "    def wrap_optimizer(self, opt):\n"
            "        # TODO: paste the variant's policy body here.\n"
            "        # See sources: " + ", ".join(source_overlay_files) + "\n"
            "        return opt\n"
        )

    class_name = "".join(part.capitalize() for part in spec.name.split("_"))
    parent_field = f'parent="{spec.parent}"' if spec.parent else "parent=None"
    # Render as a valid tuple literal: () for empty, ("x", "y") for non-empty.
    if spec.mutations:
        mutations_field = "(" + ", ".join(f'"{m}"' for m in spec.mutations) + ",)"
    else:
        mutations_field = "()"

    body = f'''"""{spec.name} variant.

Parent: {spec.parent or 'None'}
Base commit: {spec.base_commit}
Primary_value: {spec.primary_value}
Node ID: {spec.node_id}
Mutations: {", ".join(spec.mutations) if spec.mutations else ""}
"""
from automil.registry import register, VariantSpec, {abc_name}


@register(VariantSpec(
    name="{spec.name}", kind="{spec.kind}", {parent_field},
    base_commit="{spec.base_commit}", primary_value={spec.primary_value},
    node_id="{spec.node_id}", created_at="{spec.created_at}",
    mutations={mutations_field},
))
class {class_name}({abc_name}):
{body_method}
'''
    _atomic_write_text(target_path, body)


@main.command("port-variant")
@click.argument("node_id")
@click.option("--name", default=None, help="Override auto-name (default: <parent>_v<short>).")
@click.option("--kind", default=None,
              type=click.Choice(["model", "loss", "policy"]),
              help="Override auto-detected kind.")
@click.option("--parent", default=None, help="Override auto-detected parent (model variants only).")
def port_variant(node_id: str, name: str | None, kind: str | None, parent: str | None):
    """Convert a node's overlay into a registered variant module + manifest.

    Workflow: after an experiment produced a good primary_value, run
    `automil port-variant <node_id>` to convert its dirty diff into a
    committed variant module under automil/variants/<kind_dir>/<name>.py
    plus a sibling <name>.json manifest. Auto-names as <parent>_v<short>;
    auto-detects kind from the overlay paths.

    Idempotent: re-porting a node with matching node_id is a no-op.
    Mismatched-node-id same-name is a hard-fail (won't silently overwrite).

    Calls `automil refresh-registry` at the end (D-43).
    """
    from automil.registry.manifest import Manifest
    from automil.registry.spec import VariantSpec
    from automil.registry.scanner import scan_variants, regenerate_init_py
    from automil.registry._state import _clear_registry

    adir = _find_automil_dir()
    cfg = _load_registry_or_die(adir)

    # 1. Find the spec.json.
    spec_json = _read_spec_json(adir, node_id)
    overlay_paths = list(spec_json.get("overlay_manifest", {}).keys())
    if not overlay_paths:
        raise click.ClickException(
            f"orchestrator/archive/{node_id}/spec.json has no overlay_manifest. "
            f"Cannot derive variant code. Was this an empty submit?"
        )

    # 2. Determine kind.
    final_kind = kind or _derive_kind(overlay_paths, cfg.protected)
    if final_kind is None:
        raise click.ClickException(
            f"Could not auto-detect kind from overlay paths {overlay_paths}. "
            f"Pass --kind model | loss | policy explicitly."
        )

    # 3. Determine parent (for model kind only).
    final_parent: Optional[str] = None
    if final_kind == "model":
        final_parent = parent or _detect_parent(spec_json, overlay_paths)
        if final_parent is None:
            raise click.ClickException(
                f"Could not auto-detect parent for model variant. "
                f"Pass --parent <parent_name> (e.g., 'clam_mb')."
            )

    # 4. Determine name.
    if name:
        final_name = name
    else:
        prefix = final_parent if final_kind == "model" else final_kind
        final_name = f"{prefix}_v{_node_id_short(node_id)}"

    # 5. Resolve target paths.
    if final_kind == "model":
        kind_dir = adir / "variants" / final_parent
    elif final_kind == "loss":
        kind_dir = adir / "variants" / "_losses"
    else:
        kind_dir = adir / "variants" / "_policies"

    kind_dir.mkdir(parents=True, exist_ok=True)
    module_path = kind_dir / f"{final_name}.py"
    manifest_path = kind_dir / f"{final_name}.json"

    # 6. Build VariantSpec from inputs.
    primary_value = spec_json.get("primary_value") or 0.0
    # If spec.json doesn't carry primary_value, look up from graph.json node.
    if not primary_value:
        node = _get_node_or_die(adir, node_id)
        primary_value = float(node.get("primary_value", 0.0))

    base_commit = spec_json.get("base_commit", "unknown")
    techniques = spec_json.get("graph_metadata", {}).get("techniques", []) or []

    spec = VariantSpec(
        name=final_name,
        kind=final_kind,
        parent=final_parent,
        base_commit=base_commit,
        primary_value=float(primary_value),
        node_id=node_id,
        created_at=datetime.now(tz=timezone.utc).isoformat(),
        mutations=tuple(techniques),
    )

    # 7. Idempotence check: if module exists with matching node_id, no-op.
    if module_path.exists() and manifest_path.exists():
        try:
            existing = Manifest.read(manifest_path)
        except Exception:
            existing = None
        if existing is not None and existing.spec.node_id == node_id:
            click.echo(f"port-variant: {final_name} already ported (node_id match); no-op.")
            return
        if existing is not None and existing.spec.node_id != node_id:
            raise click.ClickException(
                f"Refusing to port: {module_path} already exists with "
                f"node_id={existing.spec.node_id!r}, but you're porting "
                f"node_id={node_id!r}. Names collide. Use `--name <other_name>` "
                f"to disambiguate."
            )

    # 8. Write module + manifest atomically.
    _write_variant_module(module_path, spec, overlay_paths)
    manifest = Manifest(
        spec=spec,
        source_node=node_id,
        source_overlay_files=tuple(overlay_paths),
        ported_at=datetime.now(tz=timezone.utc).isoformat(),
        tool_version="automil 0.1.0",
    )
    manifest.write(manifest_path)

    # 9. Write variant_spec into graph.json so `automil apply <node_id>` can
    # read it (Plan 01-09 contract — closes the lifecycle integration loop).
    # Use ExperimentGraph.save() (PATTERNS.md anti-pattern #3 forbids bypassing
    # save()'s atomic tempfile+rename).
    from automil.graph import ExperimentGraph

    graph_path = adir / "graph.json"
    graph = ExperimentGraph.load(graph_path, technique_map=_load_technique_map(adir))
    node_record = graph._data["nodes"].get(node_id)
    if node_record is None:
        raise click.ClickException(
            f"Node {node_id!r} not in graph.json after port-variant write — "
            f"refusing to leave the variant module on disk without a graph mutation. "
            f"Inspect graph.json and re-submit if needed."
        )
    node_record["variant_spec"] = {
        "kind": spec.kind,
        "name": spec.name,
        "parent": spec.parent,
    }
    graph.save()  # atomic tempfile+rename

    # 10. Refresh registry (D-43).
    _clear_registry()
    scan_result = scan_variants(adir / "variants")
    for sub in (adir / "variants").iterdir():
        if sub.is_dir() and sub.name != "__pycache__":
            regenerate_init_py(sub)

    click.echo(f"Ported {node_id} -> {module_path.relative_to(adir.parent)}")
    click.echo(f"Manifest: {manifest_path.relative_to(adir.parent)}")
    if scan_result.failed:
        for path, err in scan_result.failed:
            click.echo(f"WARNING: {path}: {err}", err=True)
