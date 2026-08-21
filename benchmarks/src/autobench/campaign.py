"""Immutable campaign manifests and isolated per-cell runtime materialization.

The preprint campaign must not derive its command from one source and its
budget identity from another.  This module makes one checked-in manifest the
source for both, then materializes one independent ``automil/`` state root per
cell.  It contains no scheduler or ranking policy; stage transitions live in
``campaign_stages.py``.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shlex
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import yaml

from automil.activity_hooks import (
    ACTIVITY_METRICS_PORT,
    claude_activity_settings,
)
from automil.cells.state import make_cell_id, normalize_mil_model

#: 7 adds the per-cell companion guard. The CAMPAIGN_ID deliberately does NOT
#: move with it: this is the same 130-cell campaign under the same training
#: protocol, and bumping it would mean editing an analysis plan whose whole
#: value is being frozen before certification.
SCHEMA_VERSION = 7
CAMPAIGN_ID = "automil-preprint-130-v6"
PROTOCOL_VERSION = "preprint-v3"
ANALYSIS_PLAN_PATH = "benchmarks/campaigns/preprint_130/analysis_plan.json"
#: Per-dataset+task companion-guard margins, derived from the frozen validation
#: splits by derive_guard_margins.py and checked in so the number in the paper
#: is in git rather than recomputed. Hashed into the manifest.
GUARD_MARGINS_PATH = "benchmarks/campaigns/preprint_130/guard_margins.json"
AGENT_PROTOCOL_FILE = "agent_protocol.json"
DATASETS = (
    "tcga_luad",
    "tcga_lgg",
    "cptac_gbm",
    "cptac_pdac",
    "tcga_hnsc",
)
ENCODERS = ("uni_v2", "virchow2", "hoptimus1")
TILE_ARMS = (
    ("clam", "clam_models"),
    ("nnmil", "nnmil_models"),
    ("abmil", "abmil_models"),
    ("dtfd", "dtfd_models"),
)
STAGE_FOLDS = {
    "discovery": (0, 1, 2),
    "promotion": (3, 4),
}
CERTIFICATION_FOLDS = (0, 1, 2, 3, 4)
BASELINE_FOLDS = CERTIFICATION_FOLDS
#: The frozen held-out evidence schema, per task FAMILY — the one authority the
#: sealed fold writers, the certification reader, and the analysis stage all
#: answer to. The FIRST key of each tuple is that family's REPORTING primary
#: (the field the analysis plan's ``aggregation.primary_by_task_family``
#: declares; ``build_publication_report`` cross-checks the two and fails
#: closed on drift). Reporting is per family, Patho-Bench style — binary and
#: nominal multiclass on AUROC, ordinal grading on quadratic-weighted kappa,
#: survival on the concordance index — while SELECTION everywhere stays the
#: primary validation metric (``scoring.formula: val_auc`` / ``val_c_index``);
#: the val-firewall keeps these two axes from ever touching.
HELD_OUT_SCHEMA_BY_FAMILY = {
    "binary": ("test_auc", "test_bacc"),
    "multiclass": ("test_auc", "test_bacc"),
    "ordinal": ("test_qwk", "test_auc", "test_bacc"),
    "survival": ("test_c_index",),
}
#: The validation-evidence twin: the exact recorded `metrics` key set per
#: family (`metrics.track` in every materialized cell config, and the
#: fold-schema lock at every campaign ingest). NOTE the first-key convention
#: differs deliberately from the held-out side: here the first key is the
#: SELECTION primary (`scoring.formula`), which for ordinal cells is still
#: val_auc — qwk is a recorded companion on the validation side and the
#: REPORTING primary only on the sealed side.
VALIDATION_SCHEMA_BY_FAMILY = {
    "binary": ("val_auc", "val_bacc"),
    "multiclass": ("val_auc", "val_bacc"),
    "ordinal": ("val_auc", "val_bacc", "val_qwk"),
    "survival": ("val_c_index",),
}
#: 39 binary (kras, idh1, tp53) + 13 nominal multiclass (immune_class,
#: deliberately non-ordinal per its dataset YAML) + 13 ordinal (grade,
#: ``ordinal: true``) + 65 survival (os) = 130.
TASK_FAMILY_CENSUS = {
    "binary": 39, "multiclass": 13, "ordinal": 13, "survival": 65,
}


def classification_task_family(spec: Mapping[str, Any]) -> str:
    """Family of a classification task, from its dataset-YAML declaration.

    Keyed on the DECLARED ``ordinal`` flag and ``n_classes`` — never sniffed
    from data or metric availability (the same rule as the trainer's qwk
    handling). Fails closed on a missing ``n_classes`` rather than assuming
    binary.
    """
    if spec.get("ordinal"):
        return "ordinal"
    n_classes = spec.get("n_classes")
    if isinstance(n_classes, bool) or not isinstance(n_classes, int):
        raise CampaignManifestError(
            "classification task must declare an integer n_classes"
        )
    return "multiclass" if n_classes > 2 else "binary"


DISCOVERY_ATTEMPTS = 30
DISCOVERY_AGENT_ACTIVE_BUDGET = "12h"
AGENT_TIME_ACCOUNTING = {
    "source": "claude-native-active-time-v1",
    "metric": "claude_code.active_time.total",
    "observer": "localhost Prometheus scrape",
    "session_binding": "synchronous SessionStart/SessionEnd hooks",
}
# Promotion runs with no coding agent in the loop, so its wall-clock cap is
# pure runaway containment, never a search budget: the eval axis (exact frozen
# candidate count) is the only binding limit. Sized far above the worst case
# (10 candidates x 2 folds x the 6h attempt timeout ~= 5d serial) so the cap
# can only ever catch a hung job, not kill legitimate in-flight promotion.
PROMOTION_WALL_CLOCK_CONTAINMENT = "7d"
PROMOTION_CANDIDATES = 10
ATTEMPT_OUTCOME_CLASSES = (
    "completed", "budget-killed", "timeout", "oom", "cancelled",
    "partial", "crash", "missing-result", "unknown",
)


def classify_attempt_outcome(
    result_status: object,
    termination_reason: object,
    budget_killed: object,
) -> str:
    """Derive one predeclared terminal class from frozen terminal facts."""
    status = result_status if isinstance(result_status, str) else "missing"
    reason = termination_reason if isinstance(termination_reason, str) else "unspecified"
    if budget_killed is True or status == "budget_killed":
        return "budget-killed"
    if reason == "timeout":
        return "timeout"
    if reason == "oom":
        return "oom"
    if status == "cancelled" or reason == "cancelled_by_operator":
        return "cancelled"
    if status == "completed":
        return "completed"
    if status == "partial":
        return "partial"
    if status == "crash":
        return "crash"
    if status in {"missing", "missing-result"}:
        return "missing-result"
    return "unknown"


def expected_promotion_sources(
    attempts: list[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Recompute the stable top-10 unique discovery roster from its census."""
    eligible = sorted(
        (row for row in attempts if row.get("eligible") is True),
        key=lambda row: (-float(row["validation_mean"]), str(row["node_id"])),
    )
    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in eligible:
        candidate_sha256 = str(row["candidate_sha256"])
        if candidate_sha256 in seen:
            continue
        seen.add(candidate_sha256)
        selected.append({
            "source_node_id": str(row["node_id"]),
            "source_candidate_sha256": candidate_sha256,
            "source_spec_sha256": str(row["source_spec_sha256"]),
            "candidate_class": str(row["candidate_class"]),
            "policy_hash": str(row["policy_hash"]),
        })
        if len(selected) == PROMOTION_CANDIDATES:
            break
    return selected
# This is a failure-containment wall clock for one submitted multi-fold attempt,
# not an optimization budget.  Three CLAM classification folds take about
# 206 minutes in the committed timing census. The runtime-canary rehearsal
# showed 360 punishing exactly the recipes that train longer than the native
# defaults (three charged partials, 23.6 wasted GPU-hours, and a re-run
# attempt in one cell; every such config completed within 600). Ten hours
# still contains a hung attempt at ~2.4x the longest completed rehearsal run
# while no longer clipping the search surface — and submit refuses any
# per-spec attempt to raise it further (lowering stays free).
ATTEMPT_TIMEOUT_MIN = 600
MAXIMUM_AGENTIC_FOLD_TRAININGS_PER_CELL = (
    DISCOVERY_ATTEMPTS * len(STAGE_FOLDS["discovery"])
    + PROMOTION_CANDIDATES * len(STAGE_FOLDS["promotion"])
)
# C-j (claims-alignment): submit hosts and the controller host are not the same
# machine on a cluster, and the freeze used to abort PERMANENTLY on a
# submitted_at even one second before bound_at — timestamps live in hashed
# archived specs and cannot be legitimately corrected. Ordinary NTP-level skew
# is tolerated (declared here, hash-locked via PROTOCOL); anything beyond it
# still fails closed.
SUBMIT_CLOCK_SKEW_TOLERANCE_SECONDS = 120
# A4 (claims-alignment): the campaign's identity locks, asserted by the
# materialization audit so a template that silently lost a lock cannot pass.
# Flat union across arms — each name exists on exactly one arm's search space
# and no hparams.FIELD_ALIASES entry maps onto any of them (asserted in
# benchmarks/tests/test_campaign_identity_locks.py), so the union cannot
# false-lock another arm's legitimate knob.
EXPECTED_ALLOWED_OVERRIDE_OPTIONS = ("--hparams", "--policy-variant")
EXPECTED_IDENTITY_LOCKED_HPARAMS = (
    "no_inst_cluster", "bag_weight",  # clam defining-loss switches
    "model_size",                     # clam attention-width preset
    "M", "L",                         # abmil attention/embedding widths
    "mDim", "numLayer_Res",           # dtfd width + residual depth
    "hidden_dim",                     # nnmil model width
)
PROTOCOL = {
    "protocol_version": PROTOCOL_VERSION,
    "seed": 42,
    "split_folds": 5,
    "discovery_attempts": DISCOVERY_ATTEMPTS,
    "discovery_agent_active_budget": DISCOVERY_AGENT_ACTIVE_BUDGET,
    "agent_time_accounting": AGENT_TIME_ACCOUNTING,
    "promotion_candidates": PROMOTION_CANDIDATES,
    "promotion_wall_clock_containment": PROMOTION_WALL_CLOCK_CONTAINMENT,
    "attempt_outcome_classes": list(ATTEMPT_OUTCOME_CLASSES),
    "frozen_winners": 1,
    "stage_folds": {key: list(value) for key, value in STAGE_FOLDS.items()},
    "baseline": {
        "folds": list(BASELINE_FOLDS),
        "incumbent": True,
        "counts_toward_agentic_budget": False,
    },
    "winner_selection": {
        "metric_source": "validation",
        "aggregation": "mean",
        "folds": list(CERTIFICATION_FOLDS),
    },
    "certification": {
        "mode": "unseal-existing-held-out",
        "folds": list(CERTIFICATION_FOLDS),
        "retrain": False,
    },
    "attempt_timeout": {
        "minutes": ATTEMPT_TIMEOUT_MIN,
        "role": "failure-containment-not-search-budget",
        "scope": "one-multi-fold-attempt",
    },
    "submit_clock_skew_tolerance_seconds": SUBMIT_CLOCK_SKEW_TOLERANCE_SECONDS,
    "identity_locked_hparams": list(EXPECTED_IDENTITY_LOCKED_HPARAMS),
    "agentic_fold_trainings_per_cell": {
        "discovery": DISCOVERY_ATTEMPTS * len(STAGE_FOLDS["discovery"]),
        "promotion_per_candidate": len(STAGE_FOLDS["promotion"]),
        "promotion_candidates_min": 0,
        "promotion_candidates_max": PROMOTION_CANDIDATES,
        "minimum": DISCOVERY_ATTEMPTS * len(STAGE_FOLDS["discovery"]),
        "maximum": MAXIMUM_AGENTIC_FOLD_TRAININGS_PER_CELL,
    },
}
_CANARY_PROPOSAL_POLICY = "canary proposal policy"
_CANARY_TOOLSET = "canary toolset"
CANARY_AGENT_PROTOCOL = {
    "schema_version": 2,
    "campaign_id": CAMPAIGN_ID,
    "purpose": "canary",
    "provider": "canary",
    "runtime": "canary",
    "runtime_version": "canary-1",
    "model": "canary",
    "model_version": "canary-1",
    "effort": "max",
    "network_access": "enabled",
    "fallback_model": None,
    "proposal_policy_content": _CANARY_PROPOSAL_POLICY,
    "proposal_policy_sha256": hashlib.sha256(
        _CANARY_PROPOSAL_POLICY.encode()
    ).hexdigest(),
    "toolset_content": _CANARY_TOOLSET,
    "toolset_sha256": hashlib.sha256(_CANARY_TOOLSET.encode()).hexdigest(),
    "max_sessions_per_cell": 1,
}


class CampaignManifestError(ValueError):
    """A campaign artifact is malformed or has drifted from its lock."""


def validate_agent_protocol(
    raw: Mapping[str, Any], *, allow_canary: bool = False,
) -> dict[str, Any]:
    """Validate the coding-agent policy that must be locked before search."""
    required_strings = (
        "provider", "runtime", "runtime_version", "model", "model_version",
    )
    required_contents = ("proposal_policy_content", "toolset_content")
    required_hashes = ("proposal_policy_sha256", "toolset_sha256")
    expected_keys = {
        "schema_version", "campaign_id", "purpose", *required_strings,
        *required_contents, *required_hashes, "effort", "network_access",
        "fallback_model", "max_sessions_per_cell",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected_keys:
        raise CampaignManifestError("agent protocol field set is not exact")
    purpose = raw.get("purpose")
    if purpose not in ({"publication", "canary"} if allow_canary else {"publication"}):
        raise CampaignManifestError("agent protocol purpose is not allowed here")
    if raw.get("schema_version") != 2 or raw.get("campaign_id") != CAMPAIGN_ID:
        raise CampaignManifestError("agent protocol identity is invalid")
    for key in required_strings:
        value = raw.get(key)
        if (
            not isinstance(value, str)
            or not value.strip()
            or value.upper().startswith("REPLACE")
            or value.lower() == "unknown"
        ):
            raise CampaignManifestError(f"agent protocol {key} is not publication-ready")
    for key in required_contents:
        value = raw.get(key)
        if (
            not isinstance(value, str)
            or not value.strip()
            or value.upper().startswith("REPLACE")
        ):
            raise CampaignManifestError(f"agent protocol {key} is not archived")
    for key in required_hashes:
        value = raw.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise CampaignManifestError(f"agent protocol {key} is not a SHA-256")
    for stem in ("proposal_policy", "toolset"):
        observed = hashlib.sha256(str(raw[f"{stem}_content"]).encode()).hexdigest()
        if raw[f"{stem}_sha256"] != observed:
            raise CampaignManifestError(
                f"agent protocol {stem} content/hash binding mismatch"
            )
    if raw.get("effort") != "max":
        raise CampaignManifestError("preprint protocol requires max agent effort")
    if raw.get("network_access") != "enabled":
        raise CampaignManifestError("preprint protocol requires external network access")
    if raw.get("fallback_model") is not None:
        raise CampaignManifestError("preprint protocol forbids model fallback")
    if raw.get("max_sessions_per_cell") != 1:
        raise CampaignManifestError("preprint protocol requires one agent session per cell")
    return json.loads(json.dumps(raw))


def build_agent_protocol(
    *,
    proposal_policy: str,
    toolset: str,
    model: str,
    model_version: str,
    runtime_version: str,
    provider: str = "Anthropic",
    runtime: str = "Claude Code",
) -> dict[str, Any]:
    """Assemble the publication agent protocol from its source payloads.

    The two content payloads are embedded verbatim and hashed here, so the
    caller can never produce a content/hash mismatch; everything else is
    either a campaign constant or an identity string the operator must pin.
    The result is returned only if it passes ``validate_agent_protocol``.
    """
    raw = {
        "schema_version": 2,
        "campaign_id": CAMPAIGN_ID,
        "purpose": "publication",
        "provider": provider,
        "runtime": runtime,
        "runtime_version": runtime_version,
        "model": model,
        "model_version": model_version,
        "effort": "max",
        "network_access": "enabled",
        "fallback_model": None,
        "max_sessions_per_cell": 1,
        "proposal_policy_content": proposal_policy,
        "proposal_policy_sha256": hashlib.sha256(
            proposal_policy.encode()
        ).hexdigest(),
        "toolset_content": toolset,
        "toolset_sha256": hashlib.sha256(toolset.encode()).hexdigest(),
    }
    return validate_agent_protocol(raw)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")


def content_sha256(value: object) -> str:
    """Hash a JSON-compatible value independently of whitespace/key order."""
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset_config_path(repo_root: Path, dataset: str) -> Path:
    source = "cptac" if dataset.startswith("cptac_") else "tcga"
    return repo_root / "benchmarks" / "datasets" / source / f"{dataset}.yaml"


def _policy_template_path(repo_root: Path, dataset: str) -> Path:
    return (
        repo_root / "benchmarks" / "experiments" / dataset
        / "automil" / "config.yaml"
    )


def _run_command(cell: Mapping[str, Any], stage: str) -> str:
    command_folds = {"baseline": BASELINE_FOLDS, **STAGE_FOLDS}
    if stage not in command_folds:
        raise CampaignManifestError(f"unknown campaign stage {stage!r}")
    tokens = [
        "python", "benchmarks/scripts/run_experiment.py",
        "--dataset", str(cell["dataset"]),
        "--task", str(cell["task"]),
        "--encoder", str(cell["encoder"]),
        "--model", str(cell["model"]),
        "--framework", str(cell["framework"]),
        "--strategy", "standard",
        "--seed", str(PROTOCOL["seed"]),
        "--n_folds", str(PROTOCOL["split_folds"]),
        "--folds", ",".join(str(i) for i in command_folds[stage]),
    ]
    if cell["survival_loss"] is not None:
        tokens.extend(["--survival_loss", str(cell["survival_loss"])])
    tokens.append("--no_wandb")
    return shlex.join(tokens)


def _cell_record(
    *,
    dataset: str,
    task: str,
    task_type: str,
    task_family: str,
    encoder: str,
    framework: str,
    model: str,
    survival_loss: str | None,
    dataset_config: str,
    dataset_config_sha256: str,
    policy_template: str,
    policy_template_sha256: str,
    guard: Mapping[str, Any] | None,
) -> dict[str, Any]:
    identity = {
        "dataset": dataset,
        "task": task,
        "encoder": encoder,
        "arm": framework,
        "seed": PROTOCOL["seed"],
        "protocol_version": PROTOCOL_VERSION,
    }
    experiment_id = (
        f"{dataset}__{task}__{encoder}__{framework}"
        f"__s{PROTOCOL['seed']}__{PROTOCOL_VERSION}"
    )
    normalized_model = normalize_mil_model(model)
    cell = {
        "cell_id": experiment_id,
        "identity": identity,
        "dataset": dataset,
        "task": task,
        "task_type": task_type,
        # The reporting family (HELD_OUT_SCHEMA_BY_FAMILY / the analysis
        # plan's primary_by_task_family key), frozen into the cell identity
        # so no downstream stage re-derives it from a YAML at read time.
        "task_family": task_family,
        "encoder": encoder,
        "framework": framework,
        "model": model,
        "survival_loss": survival_loss,
        "regime": "slide" if framework == "titan" else "tile",
        "strategy": "standard",
        "seed": PROTOCOL["seed"],
        "dataset_config": dataset_config,
        "dataset_config_sha256": dataset_config_sha256,
        "policy_template": policy_template,
        "policy_template_sha256": policy_template_sha256,
        # The companion non-inferiority guard: balanced accuracy may not
        # regress by more than one validation slide's worth. Derived per
        # dataset+task from the frozen split composition (see
        # autobench.guard_margin) and carried here so the margin is frozen
        # alongside the counts that justify it. Explicitly null for survival,
        # which reports no balanced accuracy — a stated fact, not an omission.
        "guard": dict(guard) if guard is not None else None,
        "budget_identity": {
            "dataset": dataset,
            "task": task,
            "encoder": encoder,
            "mil_model": normalized_model,
            "cell_id": make_cell_id(dataset, encoder, normalized_model, task),
        },
    }
    # There is deliberately no ``final`` training command.  The frozen winner
    # already owns folds 0-2 from discovery and folds 3-4 from promotion; final
    # reporting unseals only that candidate's existing five-fold held-out data.
    cell["commands"] = {
        stage: _run_command(cell, stage)
        for stage in ("baseline", *STAGE_FOLDS)
    }
    cell["cell_sha256"] = content_sha256(cell)
    return cell


def _validate_guard(guard: Any, label: str) -> dict[str, Any]:
    """Shape-check one companion-guard declaration, or raise naming ``label``."""
    from autobench.guard_margin import GUARD_METRIC

    margin = guard.get("margin") if isinstance(guard, Mapping) else None
    if not isinstance(guard, Mapping) or guard.get("metric") != GUARD_METRIC \
            or isinstance(margin, bool) or not isinstance(margin, (int, float)) \
            or not 0 < float(margin) < 1:
        raise CampaignManifestError(
            f"{label}: companion-guard declaration is malformed ({guard!r}); "
            f"expected metric {GUARD_METRIC!r} and a margin in (0, 1)"
        )
    counts = guard.get("validation_class_counts")
    if not counts:
        raise CampaignManifestError(
            f"{label}: companion-guard margin carries no validation class "
            "counts; the number would not be checkable by hand"
        )
    # "Re-derivable by hand from the published counts" has to be ENFORCED, not
    # merely asserted: without this, a hand-edited margins file carrying honest
    # counts beside a margin of 0.5 sails through manifest construction,
    # hashing, materialization and graph seeding as "derived", and a child
    # could shed 0.4 balanced accuracy and still pass.
    from autobench.guard_margin import derived_margin_for_counts

    try:
        expected = derived_margin_for_counts(counts)
    except Exception as exc:  # noqa: BLE001 — any unusable counts block is fatal here
        raise CampaignManifestError(
            f"{label}: companion-guard validation class counts do not support "
            f"a margin ({exc})"
        ) from exc
    if float(margin) != expected:
        raise CampaignManifestError(
            f"{label}: companion-guard margin {float(margin)!r} is not the "
            f"margin its own published counts imply ({expected!r})"
        )
    return dict(guard)


def _frozen_guard(
    margins: Mapping[str, Any], dataset: str, task: str,
) -> dict[str, Any] | None:
    """One classification cell's guard, out of the frozen derivation artifact.

    ``None`` when this cohort's margin has not been derived yet — deriving it
    needs the dataset MOUNTED, so a manifest built on a partial host records
    the gap honestly instead of inventing a number that would be published as
    if it came from the split. Nothing runs on the gap: materialization
    refuses a classification cell whose guard is null, so an under-derived
    manifest can exist but can never produce a runnable cell.

    A PRESENT but malformed entry still raises: that is a defect in the
    artifact, not an undone step.
    """
    entry = margins.get(f"{dataset}__{task}")
    if entry is None:
        return None
    return _validate_guard(entry, f"{dataset}__{task}")


def build_preprint_manifest(repo_root: Path) -> dict[str, Any]:
    """Build the exact 130-cell manifest from the five pinned dataset YAMLs."""
    repo_root = repo_root.resolve()
    guard_margins_path = repo_root / GUARD_MARGINS_PATH
    try:
        guard_margins = json.loads(guard_margins_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignManifestError(
            f"cannot read frozen companion-guard margins {guard_margins_path}: "
            f"{exc}"
        ) from exc
    cells: list[dict[str, Any]] = []
    sources: dict[str, str] = {}
    policy_sources: dict[str, str] = {}
    for dataset in DATASETS:
        config_path = _dataset_config_path(repo_root, dataset)
        policy_path = _policy_template_path(repo_root, dataset)
        raw = yaml.safe_load(config_path.read_text()) or {}
        tasks = raw.get("tasks") or {}
        classification = [
            name for name, spec in tasks.items()
            if (spec or {}).get("task_type", "classification") != "survival"
        ]
        survival = [
            name for name, spec in tasks.items()
            if (spec or {}).get("task_type", "classification") == "survival"
        ]
        if len(classification) != 1 or survival != ["os"]:
            raise CampaignManifestError(
                f"{dataset}: expected one classification task plus os, got "
                f"classification={classification}, survival={survival}"
            )
        losses = list((tasks["os"] or {}).get("survival_losses") or [])
        if losses != ["nllsurv"]:
            raise CampaignManifestError(
                f"{dataset}: preprint survival_losses must be exactly ['nllsurv'], "
                f"got {losses}"
            )
        encoder_dims = ((raw.get("encoders") or {}).get("dims") or {})
        if not set(ENCODERS).issubset(encoder_dims):
            raise CampaignManifestError(
                f"{dataset}: missing roster encoder(s) "
                f"{sorted(set(ENCODERS) - set(encoder_dims))}"
            )
        config_rel = config_path.relative_to(repo_root).as_posix()
        config_hash = file_sha256(config_path)
        sources[config_rel] = config_hash
        policy_rel = policy_path.relative_to(repo_root).as_posix()
        policy_hash = file_sha256(policy_path)
        policy_sources[policy_rel] = policy_hash
        task_pairs = (
            (classification[0], "classification", None,
             classification_task_family(tasks[classification[0]] or {})),
            ("os", "survival", "nllsurv", "survival"),
        )
        for task, task_type, loss, task_family in task_pairs:
            guard = None if task_family == "survival" else _frozen_guard(
                guard_margins, dataset, task
            )
            for framework, roster_key in TILE_ARMS:
                models = list(raw.get(roster_key) or [])
                if len(models) != 1:
                    raise CampaignManifestError(
                        f"{dataset}: {roster_key} must pin exactly one model, got {models}"
                    )
                for encoder in ENCODERS:
                    cells.append(_cell_record(
                        dataset=dataset, task=task, task_type=task_type,
                        task_family=task_family,
                        encoder=encoder, framework=framework, model=models[0],
                        survival_loss=loss, dataset_config=config_rel,
                        dataset_config_sha256=config_hash,
                        policy_template=policy_rel,
                        policy_template_sha256=policy_hash,
                        guard=guard,
                    ))
            cells.append(_cell_record(
                dataset=dataset, task=task, task_type=task_type,
                task_family=task_family,
                encoder="titan", framework="titan", model="titan",
                survival_loss=loss, dataset_config=config_rel,
                dataset_config_sha256=config_hash,
                policy_template=policy_rel,
                policy_template_sha256=policy_hash,
                guard=guard,
            ))
    analysis_plan_path = repo_root / ANALYSIS_PLAN_PATH
    try:
        analysis_plan = json.loads(analysis_plan_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignManifestError(
            f"cannot read frozen analysis plan {analysis_plan_path}: {exc}"
        ) from exc
    if (
        not isinstance(analysis_plan, dict)
        or analysis_plan.get("schema_version") != 2
        or analysis_plan.get("campaign_id") != CAMPAIGN_ID
        or analysis_plan.get("status") != "frozen-before-held-out-certification"
    ):
        raise CampaignManifestError("frozen analysis plan contract is invalid")
    declared_primary = (analysis_plan.get("aggregation") or {}).get(
        "primary_by_task_family"
    )
    if declared_primary != {
        family: keys[0] for family, keys in HELD_OUT_SCHEMA_BY_FAMILY.items()
    }:
        raise CampaignManifestError(
            "analysis plan primary_by_task_family disagrees with "
            "HELD_OUT_SCHEMA_BY_FAMILY"
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "protocol": PROTOCOL,
        "analysis_plan": {
            "path": ANALYSIS_PLAN_PATH,
            "sha256": file_sha256(analysis_plan_path),
        },
        # The companion-guard derivation, hashed like the analysis plan: the
        # per-cell margins below are only as trustworthy as the split-derived
        # counts they came from, so the artifact holding those counts is
        # pinned rather than merely consulted.
        "guard_margins": {
            "path": GUARD_MARGINS_PATH,
            "sha256": file_sha256(guard_margins_path),
        },
        "dataset_sources": sources,
        "policy_sources": policy_sources,
        "cells": cells,
    }
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Fail closed on roster, identity, hash, or command drift."""
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise CampaignManifestError("unsupported campaign manifest schema")
    if manifest.get("campaign_id") != CAMPAIGN_ID:
        raise CampaignManifestError("unexpected campaign_id")
    if manifest.get("protocol") != PROTOCOL:
        raise CampaignManifestError("campaign protocol differs from the frozen contract")
    analysis_plan = manifest.get("analysis_plan")
    if (
        not isinstance(analysis_plan, dict)
        or analysis_plan.get("path") != ANALYSIS_PLAN_PATH
        or not isinstance(analysis_plan.get("sha256"), str)
        or len(analysis_plan["sha256"]) != 64
    ):
        raise CampaignManifestError("campaign analysis-plan lock is invalid")
    cells = manifest.get("cells")
    if not isinstance(cells, list) or len(cells) != 130:
        raise CampaignManifestError(f"campaign must contain exactly 130 cells, got {len(cells or [])}")
    ids: set[str] = set()
    budgets: set[str] = set()
    dataset_sources = manifest.get("dataset_sources")
    policy_sources = manifest.get("policy_sources")
    if not isinstance(dataset_sources, dict) or not isinstance(policy_sources, dict):
        raise CampaignManifestError("campaign source locks must be objects")
    per_dataset: dict[str, int] = {}
    per_task_type: dict[str, int] = {}
    per_task_family: dict[str, int] = {}
    for raw in cells:
        if not isinstance(raw, dict):
            raise CampaignManifestError("every campaign cell must be an object")
        cell = dict(raw)
        recorded_hash = cell.pop("cell_sha256", None)
        if recorded_hash != content_sha256(cell):
            raise CampaignManifestError(f"cell hash mismatch for {cell.get('cell_id')}")
        cell_id = str(cell["cell_id"])
        identity = cell.get("identity")
        expected_identity = {
            "dataset": cell.get("dataset"),
            "task": cell.get("task"),
            "encoder": cell.get("encoder"),
            "arm": cell.get("framework"),
            "seed": PROTOCOL["seed"],
            "protocol_version": PROTOCOL_VERSION,
        }
        expected_cell_id = (
            f"{expected_identity['dataset']}__{expected_identity['task']}"
            f"__{expected_identity['encoder']}__{expected_identity['arm']}"
            f"__s{expected_identity['seed']}__{PROTOCOL_VERSION}"
        )
        if identity != expected_identity or cell_id != expected_cell_id:
            raise CampaignManifestError(
                f"declared cell identity mismatch for {cell_id}"
            )
        if cell.get("seed") != PROTOCOL["seed"]:
            raise CampaignManifestError(f"cell seed differs from identity for {cell_id}")
        budget_id = str((cell.get("budget_identity") or {})["cell_id"])
        if cell_id in ids or budget_id in budgets:
            raise CampaignManifestError(f"duplicate cell or budget identity: {cell_id}")
        ids.add(cell_id)
        budgets.add(budget_id)
        dataset_config = cell.get("dataset_config")
        policy_template = cell.get("policy_template")
        if dataset_sources.get(dataset_config) != cell.get("dataset_config_sha256"):
            raise CampaignManifestError(f"dataset source lock mismatch for {cell_id}")
        if policy_sources.get(policy_template) != cell.get("policy_template_sha256"):
            raise CampaignManifestError(f"policy source lock mismatch for {cell_id}")
        per_dataset[cell["dataset"]] = per_dataset.get(cell["dataset"], 0) + 1
        per_task_type[cell["task_type"]] = per_task_type.get(cell["task_type"], 0) + 1
        family = cell.get("task_family")
        if family not in HELD_OUT_SCHEMA_BY_FAMILY:
            raise CampaignManifestError(f"unknown task_family for {cell_id}")
        if (family == "survival") != (cell["task_type"] == "survival"):
            raise CampaignManifestError(
                f"task_family/task_type disagree for {cell_id}"
            )
        per_task_family[family] = per_task_family.get(family, 0) + 1
        # A survival cell reports no balanced accuracy, so a guard on one is
        # incoherent. A classification cell may legitimately carry null — its
        # cohort was not mounted when the manifest was built — and the
        # requirement is enforced at materialization, where the gap can no
        # longer be closed by anything but deriving it.
        _guard = cell.get("guard")
        if family == "survival":
            if _guard is not None:
                raise CampaignManifestError(
                    f"{cell_id}: survival cells report no balanced accuracy and "
                    "must carry no companion guard"
                )
        elif _guard is not None:
            _validate_guard(_guard, cell_id)
        expected_commands = {
            stage: _run_command(cell, stage)
            for stage in ("baseline", *STAGE_FOLDS)
        }
        if cell.get("commands") != expected_commands:
            raise CampaignManifestError(f"command drift for {cell_id}")
    if per_dataset != {dataset: 26 for dataset in DATASETS}:
        raise CampaignManifestError(f"per-dataset census mismatch: {per_dataset}")
    if per_task_type != {"classification": 65, "survival": 65}:
        raise CampaignManifestError(f"task-axis census mismatch: {per_task_type}")
    if per_task_family != TASK_FAMILY_CENSUS:
        raise CampaignManifestError(
            f"task-family census mismatch: {per_task_family}"
        )


def write_manifest(manifest: Mapping[str, Any], path: Path) -> str:
    """Write a deterministic manifest plus adjacent SHA-256 lock."""
    validate_manifest(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.write_text(payload)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n"
    )
    return digest


def load_manifest(path: Path, *, verify_lock: bool = True) -> dict[str, Any]:
    """Load, schema-check, and optionally verify the adjacent byte lock."""
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignManifestError(f"cannot read campaign manifest {path}: {exc}") from exc
    validate_manifest(manifest)
    if verify_lock:
        lock_path = path.with_suffix(path.suffix + ".sha256")
        try:
            expected = lock_path.read_text().split()[0]
        except (OSError, IndexError) as exc:
            raise CampaignManifestError(f"cannot read manifest lock {lock_path}") from exc
        actual = file_sha256(path)
        if actual != expected:
            raise CampaignManifestError(
                f"manifest byte hash mismatch ({actual} != {expected})"
            )
    return manifest


def _task_block(cell: Mapping[str, Any], dataset_raw: Mapping[str, Any]) -> dict[str, Any]:
    source = dict((dataset_raw.get("tasks") or {})[cell["task"]])
    if cell["task_type"] == "survival":
        return {
            "name": cell["task"], "type": "survival",
            "event_column": source["event_col"],
            "time_column": source["time_col"],
            "survival_loss": "nllsurv", "nll_bins": source.get("nll_bins", 4),
        }
    return {
        "name": cell["task"], "type": "classification",
        "num_classes": source["n_classes"], "label_column": source["label_col"],
    }


def materialize_discovery_cells(
    manifest_path: Path,
    output_root: Path,
    repo_root: Path,
    *,
    agent_protocol: Mapping[str, Any],
    allow_canary_protocol: bool = False,
) -> list[Path]:
    """Create 130 isolated discovery roots from the immutable manifest.

    Each root has its own graph/plan/learnings/orchestrator namespace.  The
    generated config's run command, budget identity, fold subset, and source
    hashes all come from the same cell record.
    """
    manifest = load_manifest(manifest_path)
    manifest_hash = file_sha256(manifest_path)
    locked_agent_protocol = validate_agent_protocol(
        agent_protocol, allow_canary=allow_canary_protocol,
    )
    agent_protocol_sha256 = content_sha256(locked_agent_protocol)
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    try:
        output_root.relative_to(repo_root)
    except ValueError as exc:
        raise CampaignManifestError("campaign output_root must live inside the git repo") from exc
    output_root.mkdir(parents=True, exist_ok=True)
    agent_protocol_path = output_root / AGENT_PROTOCOL_FILE
    if agent_protocol_path.exists():
        try:
            existing_agent_protocol = json.loads(agent_protocol_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise CampaignManifestError("existing agent protocol is unreadable") from exc
        if existing_agent_protocol != locked_agent_protocol:
            raise CampaignManifestError("existing campaign uses a different agent protocol")
    else:
        fd, temporary = tempfile.mkstemp(
            dir=str(output_root), prefix=".agent-protocol-", suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w") as stream:
                stream.write(
                    json.dumps(locked_agent_protocol, indent=2, sort_keys=True) + "\n"
                )
            os.replace(temporary, agent_protocol_path)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
    written: list[Path] = []
    for cell_index, cell in enumerate(manifest["cells"]):
        # One deterministic exporter port per manifest row, so any number of
        # cells can meter concurrently on one host without contending for a
        # single endpoint. The port is part of the audited cell config and
        # of the settings the runtime is started with.
        exporter_port = ACTIVITY_METRICS_PORT + cell_index
        activity_settings = claude_activity_settings(exporter_port)
        cell_root = output_root / cell["cell_id"]
        adir = cell_root / "automil"
        template_path = repo_root / cell["policy_template"]
        if not template_path.exists():
            raise CampaignManifestError(f"missing cohort template {template_path}")
        if file_sha256(template_path) != cell["policy_template_sha256"]:
            raise CampaignManifestError(
                f"policy template drift for {cell['dataset']}; regenerate the manifest"
            )
        config = copy.deepcopy(yaml.safe_load(template_path.read_text()) or {})
        dataset_path = repo_root / cell["dataset_config"]
        if file_sha256(dataset_path) != cell["dataset_config_sha256"]:
            raise CampaignManifestError(
                f"dataset config drift for {cell['dataset']}; regenerate the manifest"
            )
        dataset_raw = yaml.safe_load(dataset_path.read_text()) or {}
        config["project"] = {
            "name": cell["dataset"],
            "description": f"{CAMPAIGN_ID}: {cell['cell_id']}",
        }
        config["task"] = _task_block(cell, dataset_raw)
        config.setdefault("data", {})["num_folds"] = PROTOCOL["split_folds"]
        config["data"]["seed"] = PROTOCOL["seed"]
        config.setdefault("encoders", {})["primary"] = cell["encoder"]
        # The agent reads this and hill-climbs on it, so it must match what
        # run_experiment.py actually computes (_primary_components) and what
        # the framework recomputes at ingest (scoring.formula selector below).
        # Selection is the PRIMARY validation metric ALONE: on few-dozen-slide
        # validation splits bacc/qwk are threshold-quantized companions whose
        # single-count jitter is the size of the accept-margin floor — they
        # stay tracked and recorded, but no longer vote.
        # From the FROZEN cell identity (task_family), never a YAML re-parse
        # at materialization time — one authority, first key = selection
        # primary.
        track = list(VALIDATION_SCHEMA_BY_FAMILY[cell["task_family"]])
        primary = track[0]
        config["metrics"] = {
            "primary": primary,
            "track": track,
        }
        # The framework-side selector (CR-1b recompute + per-fold projection):
        # campaign-owned, template-independent, audited below.
        config.setdefault("scoring", {})["formula"] = primary
        # The companion guard rides the same rail: campaign-owned, frozen in
        # the cell, audited below. Selection stays single-metric — the guard
        # can only reject a child, never promote one, so nothing about the
        # argmax changes; it just cannot be won by trading balanced accuracy
        # away. Survival cells declare none and the key stays absent.
        if cell["task_family"] != "survival":
            if cell.get("guard") is None:
                raise CampaignManifestError(
                    f"{cell['cell_id']}: classification cell has no companion-"
                    f"guard margin. Derive it where {cell['dataset']} is "
                    "mounted (benchmarks/campaigns/preprint_130/"
                    "derive_guard_margins.py --write) and regenerate the "
                    "manifest; a cell must not search without the guard its "
                    "protocol declares."
                )
            config["scoring"]["guard"] = copy.deepcopy(cell["guard"])
        else:
            config["scoring"].pop("guard", None)
        adir_rel = adir.relative_to(repo_root).as_posix()
        config["files"] = {
            "editable": [f"{adir_rel}/variants/_policies/*.py"],
        }
        config["run"] = {
            "script": None,
            "command": cell["commands"]["discovery"],
            "mil_model": cell["model"],
        }
        config.setdefault("cap", {})["budget"] = PROTOCOL[
            "discovery_agent_active_budget"
        ]
        config["cap"]["mode"] = "agent_active"
        config["cap"]["eval_budget"] = PROTOCOL["discovery_attempts"]
        config["activity"] = {"exporter_port": exporter_port}
        config["training"] = {"fold_count": len(STAGE_FOLDS["discovery"])}
        config.setdefault("orchestrator", {})["default_timeout_min"] = (
            ATTEMPT_TIMEOUT_MIN
        )
        config["campaign"] = {
            "campaign_id": CAMPAIGN_ID,
            "manifest": manifest_path.relative_to(repo_root).as_posix(),
            "manifest_sha256": manifest_hash,
            "cell_id": cell["cell_id"],
            "cell_sha256": cell["cell_sha256"],
            "budget_cell_id": cell["budget_identity"]["cell_id"],
            "stage": "discovery",
            "protocol_version": PROTOCOL_VERSION,
            "agent_protocol_sha256": agent_protocol_sha256,
        }

        # Materialization is a restart-safe initializer, never a reset command.
        # A repeated invocation verifies the immutable inputs but preserves the
        # agent-owned plan/learnings/policies and the progressed stage ledger.
        if cell_root.exists():
            from autobench.campaign_stages import load_stage_state

            try:
                state = load_stage_state(cell_root)
                existing_cell = json.loads((adir / "campaign_cell.json").read_text())
                existing_config = yaml.safe_load((adir / "config.yaml").read_text()) or {}
                existing_settings = json.loads(
                    (cell_root / ".claude/settings.json").read_text()
                )
            except (OSError, json.JSONDecodeError, yaml.YAMLError, ValueError) as exc:
                raise CampaignManifestError(
                    f"existing discovery root is incomplete or corrupt: {cell_root}"
                ) from exc
            expected_state = (
                cell["cell_id"], cell["cell_sha256"], manifest_hash,
                PROTOCOL_VERSION,
            )
            actual_state = (
                state.get("cell_id"), state.get("cell_sha256"),
                state.get("manifest_sha256"), state.get("protocol_version"),
            )
            if (
                existing_cell != cell
                or existing_config != config
                or existing_settings != activity_settings
                or actual_state != expected_state
            ):
                raise CampaignManifestError(
                    f"existing discovery root is bound to different inputs: {cell_root}"
                )
            written.append(adir)
            continue

        # Publish a fully initialized cell directory in one rename.  A crash
        # before os.replace leaves only a hidden temporary directory and never
        # exposes a half-created campaign root as resumable state.
        staging_root = Path(tempfile.mkdtemp(prefix=".materialize-", dir=str(output_root)))
        try:
            staging_adir = staging_root / "automil"
            staging_adir.mkdir(parents=True)
            (staging_adir / "config.yaml").write_text(
                yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
            )
            (staging_adir / "campaign_cell.json").write_text(
                json.dumps(cell, indent=2, sort_keys=True) + "\n"
            )
            (staging_adir / ".gitignore").write_text(
                "graph.json\nresults.tsv\nresult.json\norchestrator/\ncells/\n"
                ".activity.jsonl\n.activity.samples.json\n.activity.lock\n"
                ".automil_active\n.automil_worktrees/\n*.log\n*.pid\n"
            )
            settings_path = staging_root / ".claude" / "settings.json"
            settings_path.parent.mkdir(parents=True)
            settings_path.write_text(
                json.dumps(activity_settings, indent=2, sort_keys=True) + "\n"
            )
            (staging_adir / "plan.md").write_text(
                f"# Discovery plan — {cell['cell_id']}\n\nNo proposals queued yet.\n"
            )
            (staging_adir / "learnings.md").write_text(
                f"# Cell-local learnings — {cell['cell_id']}\n"
            )
            policy_dir = staging_adir / "variants" / "_policies"
            policy_dir.mkdir(parents=True)
            (policy_dir / ".gitkeep").touch()
            from autobench.campaign_stages import initialize_stage_state

            initialize_stage_state(
                staging_root,
                cell=cell,
                manifest_sha256=manifest_hash,
            )
            os.replace(staging_root, cell_root)
        finally:
            if staging_root.exists():
                shutil.rmtree(staging_root)
        written.append(adir)
    return written


def audit_materialized_campaign(
    *,
    roots: list[Path],
    manifest_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Audit all 130 launch roots without executing a GPU training process."""
    from automil.admissibility import (
        load_candidate_policy,
        validate_campaign_binding,
    )
    from autobench.campaign_stages import load_stage_state

    manifest = load_manifest(manifest_path)
    repo_root = repo_root.resolve()
    if len(roots) != len(manifest["cells"]):
        raise CampaignManifestError(
            f"materialized root count mismatch: {len(roots)} != {len(manifest['cells'])}"
        )
    runtime_roots = {adir.parent.parent.resolve() for adir in roots}
    if len(runtime_roots) != 1:
        raise CampaignManifestError("materialized cells do not share one runtime root")
    runtime_root = next(iter(runtime_roots))
    try:
        agent_protocol = validate_agent_protocol(
            json.loads((runtime_root / AGENT_PROTOCOL_FILE).read_text()),
            allow_canary=True,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignManifestError("cannot read locked campaign agent protocol") from exc
    agent_protocol_sha256 = content_sha256(agent_protocol)
    by_id = {cell["cell_id"]: cell for cell in manifest["cells"]}
    port_by_id = {
        cell["cell_id"]: ACTIVITY_METRICS_PORT + index
        for index, cell in enumerate(manifest["cells"])
    }
    seen: set[str] = set()
    regimes: dict[tuple[str, str], str] = {}
    manifest_hash = file_sha256(manifest_path)
    for adir in roots:
        if not adir.is_dir():
            raise CampaignManifestError(f"missing materialized root: {adir}")
        try:
            cell = json.loads((adir / "campaign_cell.json").read_text())
            config = yaml.safe_load((adir / "config.yaml").read_text()) or {}
            settings = json.loads((adir.parent / ".claude/settings.json").read_text())
        except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
            raise CampaignManifestError(f"cannot read materialized root {adir}: {exc}") from exc
        cell_id = cell.get("cell_id")
        if cell_id in seen or cell_id not in by_id or cell != by_id[cell_id]:
            raise CampaignManifestError(f"materialized cell identity drift: {cell_id}")
        seen.add(cell_id)
        campaign = config.get("campaign") or {}
        if campaign.get("manifest_sha256") != manifest_hash:
            raise CampaignManifestError(f"manifest binding drift for {cell_id}")
        validate_campaign_binding(
            manifest_path,
            campaign,
            base_run_command=(config.get("run") or {}).get("command"),
            budget_cell_id=cell["budget_identity"]["cell_id"],
        )
        if campaign.get("stage") != "discovery":
            raise CampaignManifestError(f"{cell_id}: initial root is not discovery")
        if campaign.get("agent_protocol_sha256") != agent_protocol_sha256:
            raise CampaignManifestError(f"{cell_id}: agent protocol binding drift")
        if settings != claude_activity_settings(port_by_id[cell_id]):
            raise CampaignManifestError(f"{cell_id}: activity observer contract drift")
        if (config.get("activity") or {}).get("exporter_port") != port_by_id[cell_id]:
            raise CampaignManifestError(f"{cell_id}: activity exporter port drift")
        if (config.get("cap") or {}).get("eval_budget") != DISCOVERY_ATTEMPTS:
            raise CampaignManifestError(f"{cell_id}: discovery attempt cap drift")
        if (config.get("cap") or {}).get(
            "budget"
        ) != DISCOVERY_AGENT_ACTIVE_BUDGET:
            raise CampaignManifestError(f"{cell_id}: agent-active budget drift")
        if (config.get("cap") or {}).get("mode") != "agent_active":
            raise CampaignManifestError(f"{cell_id}: activity clock mode drift")
        if (config.get("training") or {}).get("fold_count") != len(
            STAGE_FOLDS["discovery"]
        ):
            raise CampaignManifestError(f"{cell_id}: discovery fold count drift")
        if (config.get("orchestrator") or {}).get(
            "default_timeout_min"
        ) != ATTEMPT_TIMEOUT_MIN:
            raise CampaignManifestError(f"{cell_id}: attempt timeout drift")
        _expected_track = list(VALIDATION_SCHEMA_BY_FAMILY[cell["task_family"]])
        _expected_formula = _expected_track[0]
        # Exact-block lock: a stale materialization carrying a retired key
        # (e.g. the removed metrics.composite_formula) must fail the audit
        # loudly, not slide through per-key equality checks.
        if config.get("metrics") != {
            "primary": _expected_formula, "track": _expected_track,
        }:
            raise CampaignManifestError(
                f"{cell_id}: metrics block drift (expected exactly "
                f"primary={_expected_formula!r} + track={_expected_track})"
            )
        if (config.get("scoring") or {}).get("formula") != _expected_formula:
            raise CampaignManifestError(
                f"{cell_id}: selection-formula drift (expected "
                f"{_expected_formula})"
            )
        # Exact-match, like the formula: a materialized cell whose guard was
        # edited (widened, retargeted, deleted) would gate on a margin the
        # manifest does not record, and the frozen counts would no longer
        # justify the number actually applied.
        if (config.get("scoring") or {}).get("guard") != cell.get("guard"):
            raise CampaignManifestError(
                f"{cell_id}: companion-guard drift (config declares "
                f"{(config.get('scoring') or {}).get('guard')!r}, manifest "
                f"froze {cell.get('guard')!r})"
            )
        # (metrics.primary is covered by the exact-block lock above.)
        # A graph.json seeded under a DIFFERENT formula silently wins over
        # config.yaml forever after (graph meta uses setdefault freeze
        # semantics — deliberate for accept_margin, inherited by formula).
        # Materialized cells must start graph-less; an existing graph that
        # froze another formula would recompute every primary_value on the wrong
        # estimand while passing the config audit above.
        graph_path = adir / "graph.json"
        if graph_path.exists():
            _frozen_scoring = (
                (json.loads(graph_path.read_text()).get("meta") or {})
                .get("scoring") or {}
            )
            frozen_formula = _frozen_scoring.get("formula")
            if frozen_formula != _expected_formula:
                raise CampaignManifestError(
                    f"{cell_id}: graph.json froze scoring.formula "
                    f"{frozen_formula!r}; the campaign selects on "
                    f"{_expected_formula}"
                )
            # Same mechanism, same lock: the FROZEN guard is the one that
            # governs every keep/discard, so a hand-edited margin would run
            # the cell under a tolerance the manifest does not record while
            # the config audit above still reports it clean.
            if _frozen_scoring.get("guard") != cell.get("guard"):
                raise CampaignManifestError(
                    f"{cell_id}: graph.json froze scoring.guard "
                    f"{_frozen_scoring.get('guard')!r}; the manifest records "
                    f"{cell.get('guard')!r}"
                )
        policy = load_candidate_policy(adir)
        expected_editable = (
            f"{adir.relative_to(repo_root).as_posix()}/variants/_policies/*.py",
        )
        if (
            policy.mode != "architecture-preserving"
            or policy.editable != expected_editable
            or policy.allowed_variant_kinds != ("policy",)
            # A4: assert the VALUES, not template fidelity — materialize already
            # guarantees fidelity; the hole worth closing is a template that
            # silently lost a lock (or an override option) before the regen.
            or sorted(policy.allowed_override_options)
            != sorted(EXPECTED_ALLOWED_OVERRIDE_OPTIONS)
            or sorted(policy.identity_locked_hparams)
            != sorted(EXPECTED_IDENTITY_LOCKED_HPARAMS)
        ):
            raise CampaignManifestError(f"{cell_id}: candidate boundary drift")
        state = load_stage_state(adir.parent)
        if (
            state["phase"] != "discovery"
            or state["cell_id"] != cell_id
            or state["manifest_sha256"] != manifest_hash
            or state["protocol_version"] != campaign.get("protocol_version")
        ):
            raise CampaignManifestError(f"{cell_id}: stage ledger drift")
        for command_name, expected_folds in {
            "baseline": BASELINE_FOLDS,
            **STAGE_FOLDS,
        }.items():
            tokens = shlex.split(cell["commands"][command_name])
            try:
                actual_folds = tokens[tokens.index("--folds") + 1]
            except (ValueError, IndexError) as exc:
                raise CampaignManifestError(
                    f"{cell_id}: {command_name} command lacks --folds"
                ) from exc
            if actual_folds != ",".join(map(str, expected_folds)):
                raise CampaignManifestError(
                    f"{cell_id}: {command_name} fold command drift"
                )
        regimes.setdefault((cell["framework"], cell["task_type"]), cell_id)

    expected_regimes = {
        (framework, task_type)
        for framework in ("clam", "nnmil", "abmil", "dtfd", "titan")
        for task_type in ("classification", "survival")
    }
    if set(regimes) != expected_regimes:
        raise CampaignManifestError(
            f"arm/task canary coverage mismatch: {sorted(set(regimes))}"
        )
    return {
        "campaign_id": CAMPAIGN_ID,
        "manifest_sha256": manifest_hash,
        "agent_protocol_sha256": agent_protocol_sha256,
        "cells": len(seen),
        "protocol_version": PROTOCOL_VERSION,
        "regimes": {
            f"{framework}/{task_type}": regimes[(framework, task_type)]
            for framework, task_type in sorted(regimes)
        },
        "gpu_processes_started": 0,
    }


def run_materialization_canary(
    manifest_path: Path, *, repo_root: Path,
) -> dict[str, Any]:
    """Materialize, audit, and automatically remove one full dry-run campaign."""
    parent = manifest_path.resolve().parent
    with tempfile.TemporaryDirectory(prefix=".canary-", dir=str(parent)) as raw:
        roots = materialize_discovery_cells(
            manifest_path, Path(raw) / "runtime", repo_root,
            agent_protocol=CANARY_AGENT_PROTOCOL,
            allow_canary_protocol=True,
        )
        return audit_materialized_campaign(
            roots=roots, manifest_path=manifest_path, repo_root=repo_root,
        )
