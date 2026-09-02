"""submit command: snapshot changed files and queue an experiment."""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
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
              help="Override cap.budget for this cell in seconds (honored only on cell creation; ignored on later submits joining the cell).")
@click.option("--safety-buffer-seconds", default=None, type=int,
              help="Override cap.safety_buffer for this cell in seconds (same scoping as --budget-seconds).")
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
    # keep/discard computed at reconcile time has no basis (parent.primary_value
    # is 0). This was the root cause of orphan subtrees like 0051-0055→0048
    # where the child was submitted before 0048 had ever run. Failed parents
    # (crash/oom/timeout) are allowed but warned: the child's comparison will
    # be against primary_value=0, which the agent should know.
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
                f"will compare against primary_value=0. Wait for {parent} to "
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
                f"child's keep/discard will compare against primary_value=0 "
                f"for the parent."
            )

    # Compute automil dir prefix relative to git root for exclusion filtering
    try:
        automil_rel = adir.resolve().relative_to(git_root.resolve()).as_posix() + "/"
    except ValueError:
        automil_rel = "automil/"

    # One candidate policy owns path matching, mode semantics, command-override
    # scope, classification, and the launch-time policy hash (LCH-1/LCH-3).
    from automil.admissibility import (
        CandidateClass,
        load_candidate_policy,
    )
    candidate_policy = load_candidate_policy(adir)

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

        # Get all changed files from git (paths relative to git root).
        # Fail closed on a git error instead of treating stderr as "no
        # output" -- a silent empty changed-file list here would archive an
        # empty overlay and burn a charged attempt for nothing.
        tracked_result = subprocess.run(
            ["git", "diff", "--name-only"],
            cwd=git_root, capture_output=True, text=True,
        )
        if tracked_result.returncode != 0:
            raise click.ClickException(
                f"git diff --name-only failed (exit {tracked_result.returncode}): "
                f"{tracked_result.stderr.strip()}"
            )
        tracked = tracked_result.stdout.strip().splitlines()
        untracked_result = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=git_root, capture_output=True, text=True,
        )
        if untracked_result.returncode != 0:
            raise click.ClickException(
                f"git ls-files --others --exclude-standard failed "
                f"(exit {untracked_result.returncode}): {untracked_result.stderr.strip()}"
            )
        untracked = untracked_result.stdout.strip().splitlines()
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

    _active_variant_path = adir / "active_variant.json"
    _active_variant_selection: dict | None = None
    if _active_variant_path.exists():
        try:
            _active_variant_selection = json.loads(_active_variant_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            raise click.ClickException(
                f"Refusing to submit [invalid]: cannot read active_variant.json: {exc}"
            ) from exc
        if not isinstance(_active_variant_selection, dict):
            raise click.ClickException(
                "Refusing to submit [invalid]: active_variant.json must contain a JSON object"
            )

    verdict = candidate_policy.classify(
        file_list,
        override=override,
        variant_selection=_active_variant_selection,
    )
    if not verdict.accepted:
        fix = ""
        if verdict.candidate_class is CandidateClass.PROTECTED_SURFACE_VIOLATION:
            fix = (
                " Fix the candidate to use the declared train-only surface, or "
                "run `automil revert-baseline` if a protected file was edited."
            )
        raise click.ClickException(
            f"Refusing to submit [{verdict.candidate_class.value}]: "
            f"{verdict.reason}.{fix}"
        )

    # A policy module without a selector is inert: the protected trainers would
    # run the baseline while the graph recorded a source candidate.  Conversely,
    # two disagreeing selectors make the executed policy ambiguous.  Close both
    # cases before archiving so every accepted train-only source candidate is
    # actually activated by the exact command/selection covered by its verdict.
    if candidate_policy.mode == "architecture-preserving":
        policy_files = tuple(f for f in file_list if _is_variant_module_path(f))
        explicit_policy: str | None = None
        if override is not None:
            override_tokens = shlex.split(override)
            explicit_values = []
            for index, token in enumerate(override_tokens):
                if token == "--policy-variant" and index + 1 < len(override_tokens):
                    explicit_values.append(override_tokens[index + 1])
                elif token.startswith("--policy-variant="):
                    explicit_values.append(token.split("=", 1)[1])
            if len(explicit_values) > 1:
                raise click.ClickException(
                    "Refusing to submit [invalid]: --policy-variant appears more than once"
                )
            explicit_policy = explicit_values[0] if explicit_values else None

        archived_policy: str | None = None
        if isinstance(_active_variant_selection, dict):
            section = _active_variant_selection.get("policy") or {}
            if isinstance(section, dict):
                value = section.get("variant")
                archived_policy = value if isinstance(value, str) else None

        if policy_files and not (explicit_policy or archived_policy):
            raise click.ClickException(
                "Refusing to submit [invalid]: a train-only policy source file "
                "requires an explicit --policy-variant selector (or an active "
                "policy selection); otherwise the candidate would execute as a no-op"
            )
        if explicit_policy and archived_policy and explicit_policy != archived_policy:
            raise click.ClickException(
                "Refusing to submit [invalid]: --policy-variant disagrees with "
                f"active_variant.json ({explicit_policy!r} != {archived_policy!r})"
            )

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
    framework_overlay_files: list[str] = []
    for f in file_list:
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
                    PurityValidator(
                        strict_policy=(
                            candidate_policy.mode == "architecture-preserving"
                        ),
                    ).check(abs_path)                       # 1. AST-only, no import
                    InterfaceValidator().check(abs_path)    # 2. static interface proof
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

    # Framework-managed variant selection is not part of --files, but it still
    # changes the live candidate. Include its exact bytes in the overlay digest
    # after the policy has classified the selected variant kinds.
    if _active_variant_path.exists():
        _applied_rel = f"{automil_rel}applied_variant.json"
        _applied_dst = archive / _applied_rel
        _applied_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(_active_variant_path), str(_applied_dst))
        _applied_hash = hashlib.sha256(_applied_dst.read_bytes()).hexdigest()
        overlay_manifest[_applied_rel] = f"sha256:{_applied_hash}"
        framework_overlay_files.append(_applied_rel)
        click.echo(
            f"  [variant] Copied active_variant.json → archive/{node}/{_applied_rel}"
            f" (will be overlaid into worktree by orchestrator)."
        )

    if (
        not overlay_manifest
        and not deletions
        and verdict.candidate_class is not CandidateClass.CONFIG_ONLY
    ):
        raise click.ClickException("No files to snapshot or delete")

    # Compute config_hash from manifest + deletions
    parts = [f"{p}:{h}" for p, h in sorted(overlay_manifest.items())]
    parts.extend(f"DELETE:{d}" for d in sorted(deletions))
    if override is not None:
        parts.append(f"OVERRIDE:{override}")
    config_hash = hashlib.sha256(
        (base_commit + "\n" + "\n".join(parts)).encode()
    ).hexdigest()[:16]

    # D-76: read backend name from automil/config.yaml (default "local" if absent).
    # Written here so cancel.py / resubmit.py know which BACKENDS[name] to use.
    # opaque_id is NOT written at submit time — the daemon writes it on launch.
    _automil_cfg = yaml.safe_load((adir / "config.yaml").read_text()) if (adir / "config.yaml").exists() else {}
    _backend_name: str = _automil_cfg.get("backend", {}).get("name", "local")
    _base_run_command = (_automil_cfg.get("run") or {}).get("command")
    if _base_run_command is not None and not isinstance(_base_run_command, str):
        raise click.ClickException("run.command must be a string or null")
    _base_run_command_sha256 = hashlib.sha256(
        (_base_run_command or "").encode()
    ).hexdigest()

    # D-134 + P2.3: Resolve cap config — CLI flag > cap.<key> duration >
    # framework fallback. Honored only on the submit that opens the cell.
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

    # Resolve one exact budget identity from the current config schema.  The
    # The observer uses the same resolver, so accounting and submission cannot
    # silently choose different cells.
    from automil.cells import (  # noqa: E402
        ActivityError,
        CellSchemaError,
        blocks_new_work,
        consumed_seconds,
        get_cell,
        get_or_create_cell,
        read_activity_report,
        resolve_cell_identity,
    )

    # D-12 (REC-04): explicit flag, then current config, then proposal metadata.
    _mil_model_raw = (
        mil_model
        or (_automil_cfg.get("run") or {}).get("mil_model")
        or (graph_json.get("nodes", {}).get(node) or {})
           .get("metadata", {}).get("mil_model")
    )
    if not _mil_model_raw:
        raise click.ClickException(
            "--mil-model is required (or set run.mil_model in config.yaml, or pass it "
            "at propose time with automil propose --mil-model). This pins the budget cell "
            "to a specific MIL model so re-parenting does not open a fresh budget. (D-12, REC-04)"
        )
    try:
        _identity = resolve_cell_identity(_automil_cfg, mil_model=_mil_model_raw)
    except ValueError as exc:
        raise click.ClickException(f"cannot resolve budget cell: {exc}") from exc
    _dataset_name = _identity.dataset
    _encoder_name = _identity.encoder
    _mil_model_norm = _identity.mil_model

    # Existing cells own their immutable accounting mode (D-134). Resolve the
    # target journal before choosing a time source: consulting current config
    # first can either demand activity evidence from a wall-clock cell or bypass
    # an agent-active journal after a config edit.
    cells_dir = adir / "cells"
    try:
        _persisted_cell = get_cell(_identity.cell_id, cells_dir=cells_dir)
    except CellSchemaError as exc:
        raise click.ClickException(str(exc)) from exc
    _effective_mode = (
        _persisted_cell.mode if _persisted_cell is not None else _cap.mode
    )
    _campaign_cfg = _automil_cfg.get("campaign")
    if _campaign_cfg is not None and not isinstance(_campaign_cfg, dict):
        raise click.ClickException("campaign must be a mapping in config.yaml")
    if _campaign_cfg is not None and _persisted_cell is None:
        raise click.ClickException(
            "campaign budget cell is missing; open the campaign agent session "
            "before the first submit"
        )

    _activity_report = None
    _active_seconds = None
    if _effective_mode == "agent_active":
        from automil.cells.activity import (  # noqa: PLC0415
            assess_activity,
            bind_activity_session,
            read_unbound_activity_report,
        )
        from automil.activity_metrics import observe_activity_metrics  # noqa: PLC0415

        try:
            # Refresh durable cumulative evidence immediately before admission;
            # this distinguishes an unavailable endpoint from an empty or
            # foreign scrape without fabricating consumed seconds.
            _activity_observation = observe_activity_metrics(adir)
            _activity_report = read_activity_report(adir, _identity.cell_id)

            # A normal project SessionStart is intentionally unbound: submit is
            # the first point where the final identity precedence
            # (--mil-model -> config -> proposal) is known. Campaign sessions
            # are bound earlier to their stronger launch digest and must never
            # be silently rebound here.
            if (
                not _activity_report.sessions
                and _campaign_cfg is None
                and _persisted_cell is None
            ):
                unbound = read_unbound_activity_report(adir)
                unbound_assessment = assess_activity(
                    unbound, _activity_observation,
                )
                if (
                    unbound.sessions != unbound.open_sessions
                    or len(unbound.sessions) != 1
                ):
                    raise ActivityError(
                        "agent_active accounting requires exactly one open, "
                        "project-local SessionStart"
                    )
                if not unbound_assessment.admissible:
                    raise ActivityError(
                        unbound_assessment.reason
                        or "project-local activity evidence is not admissible"
                    )
                bind_activity_session(
                    adir,
                    _identity.cell_id,
                    unbound.sessions[0],
                )
                _activity_report = read_activity_report(adir, _identity.cell_id)

            assessment = assess_activity(
                _activity_report, _activity_observation,
            )
        except ActivityError as exc:
            raise click.ClickException(
                f"activity accounting is invalid: {exc}"
            ) from exc
        if assessment.complete:
            raise click.ClickException(
                "this cell's bound session has ended and agent_active cells "
                "are single-session: a new Claude session cannot rebind an "
                "existing cell. Continue in the original session, or use "
                "cap.mode: wall_clock for multi-session work"
            )
        if (
            len(_activity_report.sessions) != 1
            or len(_activity_report.open_sessions) != 1
        ):
            raise click.ClickException(
                "agent_active accounting requires exactly one bound open "
                "session; this cell accepts work only from its one bound "
                "session (a new session cannot rebind an existing cell)"
            )
        if not assessment.admissible:
            raise click.ClickException(
                "agent_active accounting is degraded and new work is paused: "
                f"{assessment.reason or assessment.health.value}"
            )
        _active_seconds = assessment.active_seconds

    # The manifest payload is consumer-owned; the binding contract is generic.
    # Verify its bytes, then prove command, budget, and cell hashes all resolve
    # from the same unique manifest row before stamping them into the queue spec.
    _campaign_spec: dict[str, object] | None = None
    _campaign_agent_session: dict[str, str] | None = None
    if _campaign_cfg is not None:
        _required_campaign = (
            "campaign_id", "manifest", "manifest_sha256", "cell_id",
            "cell_sha256", "budget_cell_id", "stage",
        )
        _missing_campaign = [
            key for key in _required_campaign
            if not isinstance(_campaign_cfg.get(key), str)
            or not str(_campaign_cfg.get(key)).strip()
        ]
        if _missing_campaign:
            raise click.ClickException(
                f"campaign metadata is missing non-empty string field(s) {_missing_campaign}"
            )
        _manifest_rel = Path(str(_campaign_cfg["manifest"]))
        if _manifest_rel.is_absolute() or ".." in _manifest_rel.parts:
            raise click.ClickException("campaign.manifest must be a safe git-root-relative path")
        _manifest_path = git_root / _manifest_rel
        if not _manifest_path.is_file():
            raise click.ClickException(f"campaign manifest not found: {_manifest_rel}")
        _manifest_actual = hashlib.sha256(_manifest_path.read_bytes()).hexdigest()
        if _manifest_actual != _campaign_cfg["manifest_sha256"]:
            raise click.ClickException(
                "campaign manifest hash differs from config.yaml; regenerate or "
                "rematerialize the cell before submitting"
            )
        try:
            from automil.admissibility import validate_campaign_binding

            _campaign_binding = {
                key: _campaign_cfg[key] for key in _required_campaign
            }
            if "protocol_version" in _campaign_cfg:
                _campaign_binding["protocol_version"] = _campaign_cfg[
                    "protocol_version"
                ]
            from automil.admissibility import enforce_attempt_timeout_cap

            # Timeout cap: --timeout above the audited cell default would
            # unbind the hash-locked failure-containment constant. RAW config
            # value — same reference the daemon's launch revalidation uses.
            enforce_attempt_timeout_cap(
                timeout,
                (_automil_cfg.get("orchestrator") or {}).get("default_timeout_min"),
            )
            _campaign_spec = validate_campaign_binding(
                _manifest_path,
                _campaign_binding,
                base_run_command=_base_run_command,
                budget_cell_id=_identity.cell_id,
            )
            _protocol_sha256 = _campaign_cfg.get("agent_protocol_sha256")
            if (
                not isinstance(_protocol_sha256, str)
                or len(_protocol_sha256) != 64
                or any(char not in "0123456789abcdef" for char in _protocol_sha256)
            ):
                raise ValueError("campaign agent protocol binding is missing")
            _session_path = adir.parent / "agent_session.json"
            try:
                _session = json.loads(_session_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    "open the campaign agent session before the first submit"
                ) from exc
            from automil.launch_binding import validate_launch_binding

            _launch_binding = validate_launch_binding(
                _session,
                campaign_id=str(_campaign_cfg["campaign_id"]),
                cell_id=str(_campaign_cfg["cell_id"]),
                agent_protocol_sha256=_protocol_sha256,
                require_open=True,
            )
            _campaign_agent_session = {
                "session_id": _launch_binding["session_id"],
                "agent_protocol_sha256": _protocol_sha256,
                "binding_sha256": _launch_binding["binding_sha256"],
            }
            if _activity_report is None:
                raise ValueError("campaign discovery requires agent_active accounting")
            expected_session = (_launch_binding["session_id"],)
            expected_binding = (
                (_launch_binding["session_id"], _launch_binding["binding_sha256"]),
            )
            if _activity_report.sessions != expected_session:
                raise ValueError(
                    "activity journal is not exclusive to the bound campaign session"
                )
            if _activity_report.bindings != expected_binding:
                raise ValueError(
                    "activity journal is not bound to agent_session.json"
                )
            if _activity_report.complete:
                raise ValueError("the bound campaign agent session has already ended")
        except ValueError as exc:
            raise click.ClickException(
                f"campaign config is not bound to its manifest: {exc}"
            ) from exc

    if _campaign_spec is not None:
        _cell = _persisted_cell
        if _cell is None:
            raise click.ClickException(
                "campaign budget cell is missing; open the campaign agent session "
                "before the first submit"
            )
        expected_cap = (
            _cap.budget_seconds,
            _cap.safety_buffer_seconds,
            _cap.mode,
            _cap.eval_budget,
        )
        actual_cap = (
            _cell.budget_seconds,
            _cell.safety_buffer_seconds,
            _cell.mode,
            _cell.eval_budget,
        )
        if actual_cap != expected_cap:
            raise click.ClickException(
                "campaign budget cell differs from the frozen config"
            )
    else:
        _cell = get_or_create_cell(
            dataset=_identity.dataset,
            encoder=_identity.encoder,
            mil_model=_identity.mil_model,
            budget_seconds=_cap.budget_seconds,
            safety_buffer_seconds=_cap.safety_buffer_seconds,
            mode=_cap.mode,
            task=_identity.task,
            eval_budget=_cap.eval_budget,
            cells_dir=cells_dir,
        )

    _consumed = consumed_seconds(
        _cell, agent_active_seconds=_active_seconds,
    )
    _time_refusing = (
        _cell.budget_seconds - _consumed <= _cell.safety_buffer_seconds
    )
    if blocks_new_work(_cell) or _time_refusing:
        # H-2: name whichever axis is binding. The eval axis can bind while the
        # status still reads ACTIVE (status only advances on the next daemon tick).
        _evals_msg = (
            f", {_cell.consumed_evals}/{_cell.eval_budget} evaluations consumed"
            if _cell.eval_budget is not None else ""
        )
        raise click.ClickException(
            f"Cell {_cell.cell_id[:8]} is {_cell.status.value}: budget exhausted "
            f"({_consumed:.0f}/{_cell.budget_seconds}s consumed"
            f"{_evals_msg}). "
            f"Wait for cell to finalize, or submit with a different "
            f"(dataset={_dataset_name}, encoder={_encoder_name}, mil_model={_mil_model_norm}) tuple."
        )

    # Write spec to queue
    spec = {
        "id": node,
        "description": desc,
        "base_commit": base_commit,
        "overlay_dir": f"archive/{node}",
        "overlay_manifest": overlay_manifest,
        "deletions": deletions,
        "framework_overlay_files": framework_overlay_files,
        "admissibility": verdict.to_dict(),
        "base_run_command_sha256": _base_run_command_sha256,
        "priority": priority,
        "estimated_vram_gb": vram,
        "graph_metadata": {
            "parent_id": parent,
            "techniques": list(techniques),
            "config_hash": config_hash,
        },
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }
    # D-02: only write timeout_min when explicitly supplied; daemon falls back to
    # orchestrator.default_timeout_min (config.yaml) when the key is absent.
    if timeout is not None:
        spec["timeout_min"] = timeout
    # D-04 (CFG-03): write per-node run-command override suffix into spec.
    # WR-01 fix: validate shlex.split() at submit time so malformed quotes
    # raise a ClickException immediately rather than crashing the daemon at
    # launch time (after the spec has already been dequeued).
    if override is not None:
        try:
            shlex.split(override)
        except ValueError as exc:
            raise click.ClickException(
                f"--override contains unbalanced quotes and cannot be parsed: {exc}"
            ) from exc
        spec["run_command_override"] = override
    spec.setdefault("metadata", {})["backend"] = _backend_name
    # D-97: write metadata.runtime so orchestrator + cancel.py know which
    # runtime made this submission. AUTOMIL_RUNTIME is set by the agent runtime
    # (never inferred — D-87). Falls back to "unknown" if unset.
    spec.setdefault("metadata", {})["runtime"] = os.environ.get("AUTOMIL_RUNTIME", "unknown")
    # D-117: stamp metadata.cell_id — symmetric to metadata.backend and metadata.runtime.
    # The daemon's _running_in_cell() filters in-cell experiments by this field.
    # Direct backend specs may be cell-less; every CLI submit is explicitly metered.
    spec.setdefault("metadata", {})["cell_id"] = _cell.cell_id
    if _campaign_spec is not None:
        spec.setdefault("metadata", {})["campaign"] = _campaign_spec
        spec.setdefault("metadata", {})["agent_session"] = _campaign_agent_session

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
                # CELL-1: record budget-cell membership on the node itself, not
                # only on the queue spec, so per-cell evaluation counts are
                # answerable from graph.json alone.
                graph.nodes[allocated]["cell_id"] = _cell.cell_id
                from automil.graph import merged_metadata
                graph.nodes[allocated]["metadata"] = merged_metadata(
                    graph.nodes[allocated],
                    {
                        "candidate_class": verdict.candidate_class.value,
                        "candidate_policy_hash": verdict.policy_hash,
                    },
                )
                # Transition to running through the official state-machine
                # path so the counter math stays consistent.
                graph.mark_running(allocated)
            else:
                # OPS-03 (D-06): existing node that already exists as
                # type=proposed, status=pending transitions to running here.
                # Without this else branch, mark_running is never called for
                # pre-existing pending proposals, leaving the graph stuck in
                # pending state while the queue spec is already written.
                # mark_running is already type/status-guarded (graph.py:280) —
                # it logs a warning and returns False for any other state, so
                # this branch is safe to call unconditionally on any existing node.
                existing = graph.get_node(node)
                if existing is not None:
                    # CELL-1: tag the pre-existing proposal too — `propose` runs
                    # before the cell is known, so this is the first chance.
                    existing["cell_id"] = _cell.cell_id
                    from automil.graph import merged_metadata
                    existing["metadata"] = merged_metadata(
                        existing,
                        {
                            "candidate_class": verdict.candidate_class.value,
                            "candidate_policy_hash": verdict.policy_hash,
                        },
                    )
                if (
                    existing
                    and existing.get("type") == "proposed"
                    and existing.get("status") == "pending"
                ):
                    graph.mark_running(node)

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
