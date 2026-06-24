"""apply command: copy a node's variant selection into the active config (CLI-01 / D-41)."""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Optional

import click
import yaml

from automil.cli import main
from automil.cli._helpers import _find_automil_dir
from automil.cli.lifecycle._shared import (
    _atomic_write_text,
    _get_node_or_die,
)

logger = logging.getLogger(__name__)


def _classify_variant_route(selection: dict, variants_root: Path) -> None:
    """Raise ClickException if any selected variant requires loop opening.

    Classification rule (APL-03, D-05, RESEARCH.md §APL-03):
    - A loss variant is loop-opening iff it is registered in LOSS_VARIANTS
      as a custom LossVariant subclass (not a plain string bag_loss selector).
    - A policy variant is loop-opening iff it is registered in POLICY_VARIANTS
      as a custom PolicyVariant subclass.
    - Model variants are always seam-expressible — never raise.
    - String bag_loss selectors (e.g. "svm", "ce") are not in LOSS_VARIANTS — no raise.

    Security (T-10-02): variant names from selection are only used for dict
    lookup, not Path construction — no path traversal risk here.

    Args:
        selection: dict with "model", "loss", "policy" sub-dicts from _derive_variant_selection.
        variants_root: the automil/variants directory; passed to scan_variants if needed.
    """
    # Lazy imports to avoid circular dependency at module load time.
    from automil.registry.scanner import scan_variants
    from automil.registry._state import LOSS_VARIANTS, POLICY_VARIANTS

    loss_name = (selection.get("loss") or {}).get("variant")
    policy_name = (selection.get("policy") or {}).get("variant")

    # Only scan (and guard) when there is a loss or policy variant to classify.
    # WR-01: scan once rather than once per branch; also warn when the variants
    # directory does not yet exist so the operator knows the guard cannot fire.
    if loss_name is not None or policy_name is not None:
        if not variants_root.exists():
            logger.warning(
                "_classify_variant_route: variants directory %s does not exist; "
                "loss/policy loop-opening guard cannot fire. Run `automil refresh-registry` "
                "after committing variant modules.",
                variants_root,
            )
        else:
            scan_variants(variants_root)

    if loss_name is not None:
        if loss_name in LOSS_VARIANTS:
            raise click.ClickException(
                f"Loss variant '{loss_name}' is a custom LossVariant callable that "
                f"requires injecting into a closed MIL training loop "
                f"(ISSUE-007 / RTA). It cannot be applied through the open "
                f"_make_clam_args seam. This variant has NOT been applied. "
                f"Deferred to the RTA milestone."
            )

    if policy_name is not None:
        if policy_name in POLICY_VARIANTS:
            raise click.ClickException(
                f"Policy variant '{policy_name}' is a custom PolicyVariant callable that "
                f"requires injecting into a closed MIL training loop "
                f"(ISSUE-007 / RTA). It cannot be applied through the open "
                f"_make_clam_args seam. This variant has NOT been applied. "
                f"Deferred to the RTA milestone."
            )


def _derive_variant_selection(node: dict) -> dict[str, dict[str, Optional[str]]]:
    """From a graph node, derive {model, loss, policy} variant selection.

    Honours two formats:
      - ``recipe``: a list of ``{kind, name, parent?}`` dicts for multi-kind nodes.
      - ``variant_spec``: a single ``{kind, name, parent?}`` dict.

    Both formats are honoured; if a node has both, the ``variant_spec`` value
    takes precedence (it was written last by ``port-variant``).

    Returns a dict with keys ``"model"``, ``"loss"``, ``"policy"`` and
    optional string values:
        {
            "model": {"variant": "clam_mb_v0176", "parent": "clam_mb"},
            "loss":  {"variant": "ce_smooth008"},
            "policy":{"variant": "sam_lookahead"},
        }
    All values default to ``None`` when the node does not specify that kind.
    """
    sel: dict[str, dict[str, Optional[str]]] = {
        "model": {"variant": None, "parent": None},
        "loss": {"variant": None},
        "policy": {"variant": None},
    }

    # Recipe path: list of {kind, name, parent?} dicts.
    recipe = node.get("recipe")
    if isinstance(recipe, list):
        for entry in recipe:
            if not isinstance(entry, dict):
                continue
            kind = entry.get("kind")
            if kind in sel:
                sel[kind]["variant"] = entry.get("name")
                if kind == "model":
                    sel["model"]["parent"] = entry.get("parent")

    # variant_spec path: single {kind, name, parent?}.
    spec = node.get("variant_spec")
    if isinstance(spec, dict):
        kind = spec.get("kind")
        if kind in sel:
            sel[kind]["variant"] = spec.get("name")
            if kind == "model":
                sel["model"]["parent"] = spec.get("parent")

    return sel


@main.command("apply")
@click.argument("node_id")
def apply(node_id: str):
    """Apply a node's variant selection to automil/config.yaml.

    Workflow: after running an experiment that produced a good composite,
    use `automil apply <node_id>` to set that node's variant choices
    (model.variant, loss.variant, policy.variant) as the active config
    for the next submit. Edits config.yaml only — never modifies the
    codebase (registry-first invariant: variant code is committed).

    Backup: writes a single rolling automil/config.yaml.bak before mutation.
    Atomic write via tempfile+rename. Idempotent.

    Hard-fails if:
      - node_id is not in graph.json (lists available nodes).
      - the node has no recorded variant_spec or recipe (run port-variant first).
      - automil/config.yaml does not exist (run automil init first).
    """
    adir = _find_automil_dir()
    config_path = adir / "config.yaml"
    backup_path = adir / "config.yaml.bak"

    if not config_path.exists():
        raise click.ClickException(
            f"automil/config.yaml not found at {config_path}. "
            f"Run `automil init` first."
        )

    node = _get_node_or_die(adir, node_id)
    selection = _derive_variant_selection(node)

    # Hard-fail if the node has no variant selection recorded at all.
    if (
        selection["model"]["variant"] is None
        and selection["loss"]["variant"] is None
        and selection["policy"]["variant"] is None
    ):
        raise click.ClickException(
            f"Node {node_id} has no recorded variant_spec or recipe. "
            f"Run `automil port-variant {node_id}` first to register the "
            f"variant, then `automil apply {node_id}` again."
        )

    # APL-03 / D-05: classify variant route BEFORE any config mutation.
    # Raises ClickException for loop-opening LossVariant / PolicyVariant callables.
    # Must fire before raw_yaml load, backup, and write (Pitfall 3 / RESEARCH.md).
    variants_root = adir / "variants"
    _classify_variant_route(selection, variants_root)

    raw_yaml = yaml.safe_load(config_path.read_text()) or {}

    # Patch the three sections.
    for kind in ("model", "loss", "policy"):
        section = raw_yaml.setdefault(kind, {})
        if not isinstance(section, dict):
            raise click.ClickException(
                f"automil/config.yaml: `{kind}:` is not a mapping. "
                f"Fix the file or restore from a recent commit."
            )
        v = selection[kind].get("variant")
        if v is not None:
            section["variant"] = v
        if kind == "model":
            p = selection["model"].get("parent")
            if p is not None:
                section["parent"] = p

    # Roll backup THEN atomic write.
    shutil.copy2(config_path, backup_path)
    new_text = yaml.safe_dump(raw_yaml, sort_keys=False, default_flow_style=False)
    _atomic_write_text(config_path, new_text)

    click.echo(
        f"Applied node {node_id}: "
        f"model.variant={selection['model'].get('variant')}, "
        f"loss.variant={selection['loss'].get('variant')}, "
        f"policy.variant={selection['policy'].get('variant')}"
    )
    click.echo(f"Backup: {backup_path}")

    # A1 fix (CR-01 / D-01): write a framework-level active_variant.json that
    # survives across node boundaries.  `apply` runs BEFORE the next `submit`,
    # so there is no "next node id" yet.  Writing to archive/<node_id>/ (the
    # already-completed node) was the original bug: apply_overlay copies from
    # archive/<NEW node_id>/, not the old one, so the file never reached the
    # new worktree.
    #
    # Correct two-part fix:
    #   1. Write to automil/active_variant.json (framework-level, node-agnostic).
    #   2. submit.py reads this file and copies it into archive/<new_node>/ as
    #      applied_variant.json so apply_overlay carries it into every future
    #      worktree (apply_overlay copies all files from overlay_dir except the
    #      three metadata files: spec.json, run.log, result.json).
    #
    # We ALSO keep the archive/<node_id>/ write for backward compatibility with
    # tests that assert on that location and for the env-injection fallback.
    active_variant_path = adir / "active_variant.json"
    _atomic_write_text(
        active_variant_path,
        json.dumps(selection, indent=2),
    )
    click.echo(
        f"Wrote active_variant.json to automil/ "
        f"(submit will propagate into the next experiment's overlay)."
    )

    # Also write to archive/<node_id>/ for backward compat + env-injection path.
    archive_dir = adir / "orchestrator" / "archive" / node_id
    archive_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        archive_dir / "applied_variant.json",
        json.dumps(selection, indent=2),
    )
    click.echo(
        f"Wrote applied_variant.json to archive/{node_id}/ "
        f"(backward compat; orchestrator picks up active_variant.json on next submit)."
    )

    # Inject AUTOMIL_VARIANT_MODEL into the queue spec env for runtime fallback.
    queue_file = adir / "orchestrator" / "queue" / f"{node_id}.json"
    if queue_file.exists():
        try:
            spec_data = json.loads(queue_file.read_text())
            if not isinstance(spec_data.get("env"), dict):
                spec_data["env"] = {}
            spec_data["env"]["AUTOMIL_VARIANT_MODEL"] = (
                selection["model"].get("variant") or ""
            )
            _atomic_write_text(queue_file, json.dumps(spec_data, indent=2))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Could not inject AUTOMIL_VARIANT_MODEL into queue spec %s: %s",
                queue_file,
                exc,
            )
