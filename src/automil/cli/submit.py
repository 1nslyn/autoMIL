"""submit command: snapshot changed files and queue an experiment."""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import click
import yaml

from automil.cli import main
from automil.cli._helpers import (
    _find_automil_dir,
    _find_git_root,
    _load_technique_map,
    _matches_scope,
)


@main.command()
@click.option("--node", required=True, help="Node ID (e.g., node_0042)")
@click.option("--desc", required=True, help="Experiment description")
@click.option("--files", multiple=True, help="Files to snapshot (auto-detect if omitted)")
@click.option("--priority", default=1, help="Priority (lower = higher)")
@click.option("--vram", default=0.5, help="Estimated VRAM in GB")
@click.option("--timeout", default=None, type=int, help="Timeout in minutes (default: orchestrator.default_timeout_min from config.yaml)")
@click.option("--max-time", "max_time_seconds", type=int, default=None,
              help="Override --timeout with seconds-precision (rounded up to 1 min minimum, D-195).")
@click.option("--parent", default=None, help="Parent node ID")
@click.option("--techniques", multiple=True, help="Technique tags")
@click.option("--budget-seconds", default=None, type=int,
              help="Override cap.budget_seconds for this cell (D-134; honored only on cell creation; ignored on subsequent submits joining an existing cell with logged INFO).")
@click.option("--safety-buffer-seconds", default=None, type=int,
              help="Override cap.safety_buffer_seconds for this cell (D-134; same scoping as --budget-seconds).")
@click.option("--mil-model", default=None,
              help="MIL model identifier for budget cell keying (D-12, REC-04). "
                   "Resolved: --mil-model flag → run.mil_model in config → "
                   "propose-time node metadata → ClickException if none found.")
@click.option("--override", default=None,
              help="Extra args appended to run.command in the worktree "
                   "(e.g. '--seed 42 --lr 1e-4'). Suffix-append only — config "
                   "run.command remains the authoritative base. (D-04, CFG-03)")
def submit(node: str, desc: str, files: tuple, priority: int, vram: float,
           timeout: int | None, max_time_seconds: int | None, parent: str | None, techniques: tuple,
           budget_seconds: int | None, safety_buffer_seconds: int | None, mil_model: str | None,
           override: str | None):
    """Snapshot changed files and queue an experiment.

    Variant modules under ``automil/variants/<parent>/<name>.py`` are
    validated at submit time: PurityValidator (no top-level I/O / network /
    mutable globals) runs first, then InterfaceValidator (subclass of the
    matching ABC, required-method signatures match). Files matching
    ``registry.protected`` glob patterns are hard-rejected (D-34).
    """
    # D-195 / RESEARCH.md OQ-5: --max-time SECONDS overrides --timeout MINUTES via ceil-div.
    if max_time_seconds is not None:
        if max_time_seconds <= 0:
            raise click.ClickException(
                f"--max-time must be > 0 seconds, got {max_time_seconds}"
            )
        translated = max(1, (max_time_seconds + 59) // 60)
        if timeout is not None:  # caller passed --timeout explicitly
            click.echo(
                f"submit: both --max-time {max_time_seconds}s and --timeout {timeout}m "
                f"provided; --max-time wins (timeout_min={translated})."
            )
        timeout = translated

    import hashlib

    git_root = _find_git_root()
    adir = _find_automil_dir()

    # Guard against overwriting an already-executed node. Submitting against
    # an id that already has recorded results would cause the orchestrator to
    # re-run it and clobber its archive/result.json — destroying prior data
    # and corrupting graph state. The only valid targets for submit are:
    #   (a) an unused id (new node), or
    #   (b) an existing proposal that has not yet been executed.
    graph_path_preflight = adir / "graph.json"
    graph_json: dict = {"nodes": {}}
    if graph_path_preflight.exists():
        try:
            graph_json = json.loads(graph_path_preflight.read_text())
        except (json.JSONDecodeError, OSError):
            graph_json = {"nodes": {}}
        existing = graph_json.get("nodes", {}).get(node)
        if existing is not None:
            ntype = existing.get("type")
            nstatus = existing.get("status")
            if ntype == "executed" or nstatus in {
                "keep", "discard", "crash", "completed", "running",
            }:
                raise click.ClickException(
                    f"Refusing to submit: {node} is already {ntype}/{nstatus}. "
                    f"Submitting would overwrite its archive and destroy prior "
                    f"results. Use 'automil propose' to create a new proposal, "
                    f"then submit against that new node id."
                )
    # Also refuse if a spec for this node is already in queue/ or running/.
    # WR-03 fix: since D-169 (Phase 6) running specs are namespaced under
    # running/<backend>/. The flat path orchestrator/running/<node>.json never
    # exists, making the running-spec conflict check permanently ineffective for
    # all experiments and allowing resubmit to silently overwrite completed nodes.
    # Fix: check queue/ with the flat path (unchanged), then iterate all backend
    # subdirs under running/ for the running-spec check.
    queue_conflict = adir / "orchestrator" / "queue" / f"{node}.json"
    if queue_conflict.exists():
        raise click.ClickException(
            f"Refusing to submit: {node} is already present in "
            f"orchestrator/queue/. Wait for it to finish or remove "
            f"the stale spec file before resubmitting."
        )
    running_root = adir / "orchestrator" / "running"
    if running_root.exists():
        for backend_dir in running_root.iterdir():
            if backend_dir.is_dir():
                running_conflict = backend_dir / f"{node}.json"
                if running_conflict.exists():
                    raise click.ClickException(
                        f"Refusing to submit: {node} is currently running in "
                        f"orchestrator/running/{backend_dir.name}/. Wait for it "
                        f"to finish or remove the stale spec file before resubmitting."
                    )

    # Guard against submitting a child before its parent has completed.
    # If the parent is still a pending/running proposal, the Pareto-dominance
    # keep/discard computed at reconcile time has no basis (parent.composite
    # is 0). This was the root cause of orphan subtrees like 0051-0055→0048
    # where the child was submitted before 0048 had ever run. Failed parents
    # (crash/oom/timeout) are allowed but warned: the child's comparison will
    # be against composite=0, which the agent should know.
    if parent:
        parent_node = graph_json.get("nodes", {}).get(parent)
        if parent_node is None:
            raise click.ClickException(
                f"Refusing to submit: --parent {parent} does not exist in "
                f"graph.json. Either propose the parent first or omit --parent "
                f"for a root-level submission."
            )
        p_type = parent_node.get("type")
        p_status = parent_node.get("status")
        if p_type == "proposed":
            raise click.ClickException(
                f"Refusing to submit: --parent {parent} has type=proposed "
                f"(status={p_status}) and has not been executed yet. "
                f"Submitting a child now means the keep/discard Pareto check "
                f"will compare against composite=0. Wait for {parent} to "
                f"finish, or pick a different --parent."
            )
        if p_type == "executed" and p_status == "running":
            raise click.ClickException(
                f"Refusing to submit: --parent {parent} is still running. "
                f"Wait for it to finish before submitting a child."
            )
        if p_type == "executed" and p_status in ("crash", "oom", "timeout"):
            click.echo(
                f"Warning: --parent {parent} has status={p_status}; the "
                f"child's keep/discard will compare against composite=0 "
                f"for the parent."
            )

    # Compute automil dir prefix relative to git root for exclusion filtering
    try:
        automil_rel = adir.resolve().relative_to(git_root.resolve()).as_posix() + "/"
    except ValueError:
        automil_rel = "automil/"

    # --- Phase 1 registry: load registry config + helper for variant-module detection ---
    from automil.registry.config import load_registry_config
    reg_cfg = load_registry_config(adir)

    def _is_variant_module_path(rel_path: str) -> bool:
        """True if rel_path is a variant module under <consumer>/automil/variants/<*>/."""
        parts = Path(rel_path).parts
        if "variants" not in parts:
            return False
        idx = parts.index("variants")
        return (
            idx + 2 < len(parts)
            and parts[idx + 2].endswith(".py")
            and not parts[idx + 2].startswith("_")
            and parts[idx + 2] != "__init__.py"
        )

    # Determine files to snapshot
    if files:
        file_list = list(files)
        # Warn (but allow) if explicit --files includes readonly files
        config_path = adir / "config.yaml"
        if config_path.exists():
            config = yaml.safe_load(config_path.read_text())
            readonly = set(config.get("files", {}).get("readonly", []))
            for f in file_list:
                if _matches_scope(f, readonly):
                    click.echo(f"Warning: {f} is marked readonly in config.yaml (submitting anyway)")
    else:
        # Auto-detect: use files.editable from config as the default scope,
        # intersected with actually changed files. This prevents capturing
        # unrelated changes in a dirty repo.
        config_path = adir / "config.yaml"
        if config_path.exists():
            config = yaml.safe_load(config_path.read_text())
            editable = set(config.get("files", {}).get("editable", []))
        else:
            editable = set()

        # Get all changed files from git (paths relative to git root)
        tracked = subprocess.run(
            ["git", "diff", "--name-only"],
            cwd=git_root, capture_output=True, text=True,
        ).stdout.strip().splitlines()
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=git_root, capture_output=True, text=True,
        ).stdout.strip().splitlines()
        # Exclude automil, runtime-config dirs, and AGENTS.md from auto-detect.
        # AGENTS.md is framework-managed (rendered by `automil init`), not user code.
        # .claude/, .opencode/, .codex/ are runtime overlay dirs installed by init.
        _FRAMEWORK_PREFIXES = (automil_rel, ".claude/", ".opencode/", ".codex/")
        all_changed = [
            f for f in tracked + untracked
            if f
            and f != "AGENTS.md"
            and not any(f.startswith(p) for p in _FRAMEWORK_PREFIXES)
        ]

        if editable:
            # Only capture files that are both editable AND changed
            file_list = [f for f in all_changed if _matches_scope(f, editable)]
            skipped = [f for f in all_changed if not _matches_scope(f, editable)]
            if skipped:
                click.echo(f"Skipping {len(skipped)} non-editable changed file(s). "
                           f"Use --files to override.")
        else:
            # No editable list configured, fall back to all changed
            file_list = all_changed

    if not file_list:
        raise click.ClickException("No changed files to snapshot")

    # Get base commit
    base_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=git_root, capture_output=True, text=True, check=True,
    ).stdout.strip()

    # Create archive directory and copy files
    archive = adir / "orchestrator" / "archive" / node
    archive.mkdir(parents=True, exist_ok=True)

    overlay_manifest = {}
    deletions = []
    for f in file_list:
        # Phase 1 protected-files reject (REG-04 / D-33 / D-34): hard-fail with
        # named pattern + fix suggestion.  Runs BEFORE path-validation so the
        # protected message wins when a path matches both a protected glob AND a
        # path-validation rule (T-01-28 mitigation; no --force escape in Phase 1).
        if reg_cfg.protected and _matches_scope(f, list(reg_cfg.protected)):
            matched: list[str] = []
            for pattern in reg_cfg.protected:
                if _matches_scope(f, [pattern]):
                    matched.append(pattern)
            raise click.ClickException(
                f"Refusing to submit: file {f!r} matches registry.protected "
                f"pattern(s) {matched}. Protected files only change via committed "
                f"variant modules. Use `automil revert-baseline` to reset working "
                f"tree, or edit automil/config.yaml: registry.protected if this "
                f"path should NOT be protected."
            )

        # Phase 1 variant-module validator chain (REG-03 / Plan 01-04 T-01-14:
        # purity FIRST, then interface).
        if _is_variant_module_path(f):
            abs_path = git_root / f
            if abs_path.exists():
                from automil.registry.validators import (
                    InterfaceValidator,
                    PurityValidator,
                )
                from automil.registry.errors import ValidationError
                try:
                    PurityValidator().check(abs_path)       # 1. AST-only, no import
                    InterfaceValidator().check(abs_path)    # 2. imports for reflection
                except ValidationError as e:
                    raise click.ClickException(
                        f"Refusing to submit: variant module {f!r} failed "
                        f"{e.validator_name} validation. {e.reason} "
                        f"Fix: {e.fix_suggestion}"
                    ) from e

        # Reject absolute paths and directory traversal
        if os.path.isabs(f) or ".." in Path(f).parts:
            raise click.ClickException(f"Invalid path (must be relative, no ..): {f}")
        src = git_root / f
        if not src.exists():
            # File was deleted - record as deletion
            deletions.append(f)
            click.echo(f"  {f}: deleted (will be removed in worktree)")
            continue
        # Verify resolved path is inside the git root
        try:
            src.resolve().relative_to(git_root.resolve())
        except ValueError:
            raise click.ClickException(f"Path escapes repository root: {f}")
        dst = archive / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        content_hash = hashlib.sha256(src.read_bytes()).hexdigest()
        overlay_manifest[f] = f"sha256:{content_hash}"

    if not overlay_manifest and not deletions:
        raise click.ClickException("No files to snapshot or delete")

    # Compute config_hash from manifest + deletions
    parts = [f"{p}:{h}" for p, h in sorted(overlay_manifest.items())]
    parts.extend(f"DELETE:{d}" for d in sorted(deletions))
    config_hash = hashlib.sha256(
        (base_commit + "\n" + "\n".join(parts)).encode()
    ).hexdigest()[:16]

    # D-76: read backend name from automil/config.yaml (default "local" if absent).
    # Written here so cancel.py / resubmit.py know which BACKENDS[name] to use.
    # opaque_id is NOT written at submit time — the daemon writes it on launch.
    _automil_cfg = yaml.safe_load((adir / "config.yaml").read_text()) if (adir / "config.yaml").exists() else {}
    _backend_name: str = _automil_cfg.get("backend", {}).get("name", "local")

    # D-134 + P2.3: Resolve cap config — CLI flag > cap.<key> duration >
    # legacy cap.<key>_seconds int > framework fallback. Honored only on the
    # submit that opens the cell.
    from automil.cells.capconfig import resolve_cap_config  # noqa: E402
    try:
        _cap = resolve_cap_config(
            _automil_cfg,
            budget_override=budget_seconds,
            buffer_override=safety_buffer_seconds,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc))
    if _cap.budget_seconds <= 0:
        raise click.ClickException(f"budget must be > 0 (got {_cap.budget_seconds}s)")
    if not (0 < _cap.safety_buffer_seconds < _cap.budget_seconds):
        raise click.ClickException(
            f"safety buffer must satisfy 0 < buffer < budget "
            f"(got buffer={_cap.safety_buffer_seconds}s, budget={_cap.budget_seconds}s)"
        )

    # D-116: Cell refusal hook — call get_or_create_cell BEFORE writing the queue spec.
    # metadata.cell_id is the cap-membership tag the daemon reads to count in-cell experiments.
    from automil.cells import get_or_create_cell, is_refusing_new, consumed_seconds  # noqa: E402

    # Cell identity: key the cap-cell by the optimization target. Real configs
    # expose this via project.name (dataset+task) and encoders.primary — NOT
    # top-level dataset/encoder, which never existed in any shipped config and
    # silently collapsed every cell to (unknown, unknown), defeating per-lineage
    # budget enforcement. Prefer the real schema; keep legacy keys as fallback.
    def _cfg_path(cfg: object, *keys: str) -> str | None:
        cur = cfg
        for k in keys:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(k)
        return cur if isinstance(cur, str) and cur.strip() else None

    _dataset_name = (
        _cfg_path(_automil_cfg, "project", "name")
        or _cfg_path(_automil_cfg, "dataset", "name")  # legacy schema
        or _cfg_path(_automil_cfg, "task", "name")
        or "unknown"
    )
    _encoder_name = (
        _cfg_path(_automil_cfg, "encoders", "primary")
        or _cfg_path(_automil_cfg, "encoder", "name")  # legacy schema
        or "unknown"
    )
    if _dataset_name == "unknown" or _encoder_name == "unknown":
        click.echo(
            f"  warning: cell identity falling back to (dataset={_dataset_name}, "
            f"encoder={_encoder_name}); set project.name and encoders.primary in "
            f"config.yaml so per-lineage budgets key correctly.",
            err=True,
        )
    # D-12 (REC-04): resolve mil_model — flag → config → propose node metadata → error.
    _mil_model_raw = (
        mil_model                                                          # --mil-model flag
        or (_automil_cfg.get("run") or {}).get("mil_model")               # config fallback
        or (graph_json.get("nodes", {}).get(node) or {})
           .get("metadata", {}).get("mil_model")                          # propose-time metadata
    )
    if not _mil_model_raw:
        raise click.ClickException(
            "--mil-model is required (or set run.mil_model in config.yaml, or pass it "
            "at propose time with automil propose --mil-model). This pins the budget cell "
            "to a specific MIL model so re-parenting does not open a fresh budget. (D-12, REC-04)"
        )
    from automil.cells.state import normalize_mil_model
    _mil_model_norm = normalize_mil_model(_mil_model_raw)

    _cell = get_or_create_cell(
        dataset=_dataset_name,
        encoder=_encoder_name,
        mil_model=_mil_model_norm,
        budget_seconds=_cap.budget_seconds,
        safety_buffer_seconds=_cap.safety_buffer_seconds,
        idle_grace_seconds=_cap.idle_grace_seconds,
        mode=_cap.mode,
    )
    if is_refusing_new(_cell):
        raise click.ClickException(
            f"Cell {_cell.cell_id[:8]} is {_cell.status.value}: budget exhausted "
            f"({consumed_seconds(_cell):.0f}/{_cell.budget_seconds}s consumed). "
            f"Wait for cell to finalize, or submit with a different "
            f"(dataset={_dataset_name}, encoder={_encoder_name}, mil_model={_mil_model_norm}) tuple."
        )

    # CR-01 fix: propagate active_variant.json into this node's archive directory
    # as applied_variant.json so apply_overlay carries it into the worktree.
    # apply_overlay copies every file from overlay_dir (= archive/<node>/),
    # excluding only spec.json, run.log, and result.json (runner.py line 70).
    # active_variant.json is written by `automil apply` BEFORE `automil submit`
    # runs, so by the time we reach here the file is either present (user ran
    # `apply` before this submit) or absent (no variant selected — no-op).
    _active_variant_path = adir / "active_variant.json"
    if _active_variant_path.exists():
        import shutil as _shutil
        _applied_dst = archive / "applied_variant.json"
        _shutil.copy2(str(_active_variant_path), str(_applied_dst))
        click.echo(
            f"  [variant] Copied active_variant.json → archive/{node}/applied_variant.json"
            f" (will be overlaid into worktree by orchestrator)."
        )

    # Write spec to queue
    spec = {
        "id": node,
        "description": desc,
        "base_commit": base_commit,
        "overlay_dir": f"archive/{node}",
        "overlay_manifest": overlay_manifest,
        "deletions": deletions,
        "priority": priority,
        "estimated_vram_gb": vram,
        "graph_metadata": {
            "parent_id": parent,
            "techniques": list(techniques),
            "config_hash": config_hash,
        },
        "submitted_at": datetime.now().isoformat(),
    }
    # D-02: only write timeout_min when explicitly supplied; daemon falls back to
    # orchestrator.default_timeout_min (config.yaml) when the key is absent.
    if timeout is not None:
        spec["timeout_min"] = timeout
    # D-04 (CFG-03): write per-node run-command override suffix into spec.
    if override is not None:
        spec["run_command_override"] = override
    spec.setdefault("metadata", {})["backend"] = _backend_name
    # D-97: write metadata.runtime so orchestrator + cancel.py know which
    # runtime made this submission. AUTOMIL_RUNTIME is set by the agent runtime
    # (never inferred — D-87). Falls back to "unknown" if unset.
    spec.setdefault("metadata", {})["runtime"] = os.environ.get("AUTOMIL_RUNTIME", "unknown")
    # D-117: stamp metadata.cell_id — symmetric to metadata.backend and metadata.runtime.
    # The daemon's _running_in_cell() filters in-cell experiments by this field.
    # Backward compat: legacy nodes without metadata.cell_id are treated as cell-less (no cap enforcement).
    spec.setdefault("metadata", {})["cell_id"] = _cell.cell_id

    queue_file = adir / "orchestrator" / "queue" / f"{node}.json"
    queue_file.write_text(json.dumps(spec, indent=2))

    # Register the node in the graph so next_id is bumped and proposals
    # don't collide with submitted experiment IDs. Route through
    # add_proposed + mark_running rather than direct dict-mutation so the
    # state-machine counter (meta.total_proposed) stays consistent;
    # otherwise every submit increments running without a matching
    # proposed-counter bump, and the counter drifts negative as nodes
    # complete and mark_executed decrements.
    #
    # locked_update serializes this read-modify-write against the daemon
    # (which also writes graph.json from _handle_completion) so we don't
    # lose either side's update.
    graph_path = adir / "graph.json"
    if graph_path.exists():
        from automil.graph import locked_update
        with locked_update(graph_path, technique_map=_load_technique_map(adir)) as graph:
            if not graph.get_node(node):
                # Force the next-allocated id to match `node`. add_proposed
                # calls next_id() internally; pre-bump meta.next_id to the
                # numeric component of node so add_proposed returns this id.
                if node.startswith("node_"):
                    try:
                        num = int(node.split("_")[1])
                        if num > graph.meta["next_id"]:
                            graph.meta["next_id"] = num
                    except (ValueError, IndexError):
                        pass
                allocated = graph.add_proposed(
                    parent_id=parent,
                    description=desc,
                    techniques=list(techniques),
                )
                # Carry over the config_hash that was computed for this
                # submit. add_proposed doesn't take it as an argument.
                graph.nodes[allocated]["config_hash"] = config_hash
                # Transition to running through the official state-machine
                # path so the counter math stays consistent.
                graph.mark_running(allocated)

    n_snap = len(overlay_manifest)
    n_del = len(deletions)
    parts_msg = []
    if n_snap:
        parts_msg.append(f"{n_snap} file(s) snapshotted")
    if n_del:
        parts_msg.append(f"{n_del} file(s) deleted")
    click.echo(f"Submitted {node}: {', '.join(parts_msg)}")
    click.echo(f"  base_commit: {base_commit[:8]}")
    click.echo(f"  config_hash: {config_hash}")
