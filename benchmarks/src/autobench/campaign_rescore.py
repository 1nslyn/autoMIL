"""Winner re-score: the frozen winner and the baseline, trained again on two
fresh development partitions.

Discovery picks its winner on one cut of the development slides, so the
held-out gain the winner shows on that cut alone cannot be told from a gain
the search found by chance on it. This stage trains the winner and the
baseline on ``devcv_r1`` and ``devcv_r2`` as well, with the same frozen
command, the same code and the same GPU type, and certification then reads
all three partitions (15 fold models per candidate).

The stage records validation evidence only. The sealed fold files it creates
are hashed and left unread until certification.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import shlex
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

import yaml

from automil.runtime_helpers import group_mkdtemp

from autobench.campaign import (
    CERTIFICATION_FOLDS,
    RESCORE_STRATEGIES,
    SPLIT_STRATEGY,
    STAGE_FOLDS,
    VALIDATION_SCHEMA_BY_FAMILY,
    content_sha256,
)
from autobench.campaign_launch import CampaignLaunchError, _check_execution_identity
from autobench.campaign_stages import (
    CampaignStageError,
    _apply_archived_overlay,
    _cell_task_family,
    _commit_state,
    _execute_frozen_command,
    _head_commit,
    _mean,
    _rescored_sources,
    _searched_winner_sources,
    _sealed_fold_hashes,
    _stage_lock,
    _utc_now,
    _validation_folds,
    _verify_baseline_unchanged,
    load_stage_state,
)

BASELINE_ID = "baseline"
RESCORE_LOCK_FILE = ".winner_rescore.lock"
RESCORE_DIR = "rescore"


class _Candidate(NamedTuple):
    """What one re-score trains: the baseline, or the searched winner."""

    candidate_id: str
    extra_args: tuple[str, ...]
    overlay: tuple[Path, Mapping[str, str], Sequence[str]] | None


class _Plan(NamedTuple):
    commit: str
    command: str
    candidates: tuple[_Candidate, ...]


def run_winner_rescore(
    cell_root: Path,
    *,
    repo_root: Path,
    gpu_id: int = 0,
) -> dict[str, Any]:
    """Re-score the frozen winner and the baseline on the fresh partitions.

    Needs phase ``winner-frozen``; success moves the cell to
    ``winner-rescored``. Any failure leaves the state untouched. A recorded
    re-score is byte-verified and returned without running anything.
    """
    if gpu_id < 0:
        raise CampaignStageError("re-score gpu_id must be non-negative")
    cell_root = cell_root.resolve()
    repo_root = repo_root.resolve()
    try:
        cell_root.relative_to(repo_root)
    except ValueError as exc:
        raise CampaignStageError(
            "campaign cell root must live inside repo_root"
        ) from exc
    lock_path = cell_root / RESCORE_LOCK_FILE
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as run_lock:
        try:
            fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CampaignStageError("winner re-score is already running") from exc
        try:
            return _run_locked(cell_root, repo_root, gpu_id)
        finally:
            fcntl.flock(run_lock.fileno(), fcntl.LOCK_UN)


def _run_locked(cell_root: Path, repo_root: Path, gpu_id: int) -> dict[str, Any]:
    state = load_stage_state(cell_root)
    if state["phase"] == "winner-rescored":
        _verify_recorded_rescore(cell_root, state)
        return state
    if state["phase"] != "winner-frozen":
        raise CampaignStageError(
            f"the winner re-score needs phase winner-frozen, got {state['phase']!r}"
        )
    plan = _plan(cell_root, repo_root, state)
    attempt_dir = _train_partitions(cell_root, repo_root, gpu_id, plan)
    records = _ingest_attempt(cell_root, state, attempt_dir, plan)
    winner = state["winner"]
    rescore = {
        "attempt": attempt_dir.relative_to(cell_root).as_posix(),
        "commit": plan.commit,
        "strategies": records,
        "validation_lift": _validation_lift(records, winner),
        "completed_at": _utc_now(),
    }
    return _record(cell_root, state, rescore)


# --- before anything runs ----------------------------------------------


def _plan(cell_root: Path, repo_root: Path, state: Mapping[str, Any]) -> _Plan:
    """Verify everything the re-score stands on, then say what to train."""
    winner, baseline = state.get("winner"), state.get("baseline")
    if not isinstance(winner, dict) or not isinstance(baseline, dict):
        raise CampaignStageError(
            "the winner re-score needs a registered baseline and a frozen winner"
        )
    _verify_baseline_unchanged(cell_root, state, baseline)
    searched = winner["kind"] != "baseline"
    if searched:
        _searched_winner_sources(cell_root, state, winner)
    commit = _launch_anchored_head(cell_root, repo_root)
    command = _frozen_discovery_command(cell_root, state)
    for strategy in RESCORE_STRATEGIES:
        _partition_command(command, strategy)
    candidates = [_Candidate(BASELINE_ID, ("--skip-prep",), None)]
    if searched:
        candidates.append(
            _winner_candidate(cell_root, repo_root, winner, command, commit)
        )
    return _Plan(commit, command, tuple(candidates))


def _launch_anchored_head(cell_root: Path, repo_root: Path) -> str:
    """The commit the re-score trains at: HEAD, held to the launch's own gate.

    The trainers import ``autobench`` and ``automil`` from the editable main
    checkout, as they do under the orchestrator daemon, so the code that
    trains is HEAD's whatever the worktree holds. HEAD must therefore be a
    commit a session could launch at (the baseline's execution commit or a
    passing reproduction's) with a clean training tree.
    """
    try:
        _check_execution_identity(cell_root, repo_root)
    except CampaignLaunchError as exc:
        raise CampaignStageError(f"cannot re-score at this HEAD: {exc}") from exc
    return _head_commit(repo_root)


def _frozen_discovery_command(cell_root: Path, state: Mapping[str, Any]) -> str:
    """The discovery command, once the config the daemon runs agrees with it."""
    _cell_task_family(cell_root, state)  # binds campaign_cell.json to the state
    try:
        cell = json.loads((cell_root / "automil" / "campaign_cell.json").read_text())
        config = yaml.safe_load((cell_root / "automil" / "config.yaml").read_text())
        configured = config["run"]["command"]
        frozen = cell["commands"]["discovery"]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise CampaignStageError(
            f"cannot read the cell's base run command: {exc}"
        ) from exc
    if configured != frozen:
        raise CampaignStageError(
            "the cell's base run command (config run.command) differs from "
            "its frozen discovery command"
        )
    return str(frozen)


def _partition_command(command_string: str, strategy: str) -> str:
    """The discovery command with its split strategy swapped for ``strategy``."""
    tokens = shlex.split(command_string)
    at = [index for index, token in enumerate(tokens) if token == "--strategy"]
    if (
        len(at) != 1
        or at[0] + 1 >= len(tokens)
        or tokens[at[0] + 1] != SPLIT_STRATEGY
    ):
        raise CampaignStageError(
            f"the frozen discovery command must carry `--strategy "
            f"{SPLIT_STRATEGY}` exactly once to be re-scored on {strategy}"
        )
    return shlex.join([*tokens[: at[0] + 1], strategy, *tokens[at[0] + 2:]])


def _winner_candidate(
    cell_root: Path,
    repo_root: Path,
    winner: Mapping[str, Any],
    command: str,
    commit: str,
) -> _Candidate:
    archive = cell_root / "automil" / "orchestrator" / "archive" / winner["candidate_id"]
    try:
        spec = json.loads((archive / "spec.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignStageError(f"cannot read the winner's spec: {exc}") from exc
    if spec.get("base_run_command_sha256") != hashlib.sha256(
        command.encode()
    ).hexdigest():
        raise CampaignStageError(
            "the winner was submitted against a different base run command"
        )
    if spec.get("base_commit") != commit:
        raise CampaignStageError(
            f"the winner ran at base commit {spec.get('base_commit')!r} but HEAD "
            f"is {commit!r}; check that commit out to re-score it"
        )
    manifest = spec.get("overlay_manifest") or {}
    deletions = spec.get("deletions") or []
    if not isinstance(manifest, dict) or not isinstance(deletions, list):
        raise CampaignStageError("the winner's overlay record is malformed")
    _preflight_overlay(cell_root, repo_root, archive, manifest, deletions)
    override = spec.get("run_command_override")
    return _Candidate(
        str(winner["candidate_id"]),
        ("--skip-prep", *shlex.split(override or "")),
        (archive, manifest, deletions),
    )


def _preflight_overlay(
    cell_root: Path,
    repo_root: Path,
    archive: Path,
    manifest: Mapping[str, str],
    deletions: Sequence[str],
) -> None:
    """Apply the winner's overlay once onto an empty directory, so a tampered
    file is refused before any partition is prepared or trained."""
    with tempfile.TemporaryDirectory(dir=cell_root, prefix=".rescore-preflight-") as scratch:
        worktree = Path(scratch) / "worktree"
        worktree.mkdir()
        try:
            _apply_archived_overlay(
                repo_root=repo_root,
                cell_root=cell_root,
                worktree=worktree,
                scratch=Path(scratch),
                archive=archive,
                manifest=manifest,
                deletions=deletions,
            )
        except (OSError, ValueError) as exc:
            raise CampaignStageError(f"the winner's overlay is refused: {exc}") from exc


# --- training ----------------------------------------------------------


def _train_partitions(
    cell_root: Path, repo_root: Path, gpu_id: int, plan: _Plan,
) -> Path:
    """Prepare each partition, then train every candidate on both at once."""
    rescore_root = cell_root / RESCORE_DIR
    rescore_root.mkdir(exist_ok=True)
    # A fresh directory per invocation: a trainer's fold-level resume cache
    # would otherwise hand a crashed earlier attempt's folds back as results.
    attempt_dir = Path(group_mkdtemp(dir=str(rescore_root), prefix="attempt-"))
    extra_env = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "PYTHONUNBUFFERED": "1",
        "AUTOMIL_DIR_REL": (cell_root / "automil").relative_to(repo_root).as_posix(),
    }

    def execute(
        strategy: str, *, node_id: str, archive_dir: Path, worktree_prefix: str,
        extra_args: Sequence[str], overlay=None,
    ) -> None:
        sealed_dir = archive_dir / "certify"
        sealed_dir.mkdir(parents=True)
        executed = _execute_frozen_command(
            cell_root=cell_root,
            repo_root=repo_root,
            command_string=_partition_command(plan.command, strategy),
            worktree_prefix=worktree_prefix,
            node_id=node_id,
            fold_count=len(CERTIFICATION_FOLDS),
            gpu_id=gpu_id,
            results_dir=sealed_dir,
            archive_dir=archive_dir,
            inject_worktree_pythonpath=False,
            extra_env=extra_env,
            overlay=overlay,
            extra_args=extra_args,
        )
        if executed != plan.commit:
            raise CampaignStageError(f"{node_id} ran at {executed}, not {plan.commit}")

    for strategy in RESCORE_STRATEGIES:
        execute(
            strategy,
            node_id=f"winner-rescore-prep-{strategy}",
            archive_dir=attempt_dir / strategy / "prep",
            worktree_prefix=".rescore-prep-worktree-",
            extra_args=("--prepare-only",),
        )
    runs = [
        (strategy, candidate)
        for strategy in RESCORE_STRATEGIES
        for candidate in plan.candidates
    ]
    with ThreadPoolExecutor(max_workers=len(runs)) as pool:
        futures = [
            pool.submit(
                execute,
                strategy,
                node_id=f"winner-rescore-{strategy}-{candidate.candidate_id}",
                archive_dir=attempt_dir / strategy / candidate.candidate_id / "archive",
                worktree_prefix=".rescore-worktree-",
                extra_args=candidate.extra_args,
                overlay=candidate.overlay,
            )
            for strategy, candidate in runs
        ]
    errors = [future.exception() for future in futures if future.exception()]
    if errors:
        raise CampaignStageError(
            "winner re-score failed and recorded nothing: "
            + "; ".join(str(error) for error in errors)
        ) from errors[0]
    return attempt_dir


# --- recording ---------------------------------------------------------


def _ingest_archive(
    cell_root: Path, archive: Path, expected_metrics: tuple[str, ...],
) -> dict[str, Any]:
    """One run's validation evidence; the sealed folds are hashed, not read."""
    try:
        result = json.loads((archive / "result.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignStageError(f"cannot read re-score result: {exc}") from exc
    folds = _validation_folds(
        result, STAGE_FOLDS["discovery"], expected_metrics=expected_metrics,
    )
    return {
        "archive": archive.relative_to(cell_root).as_posix(),
        "validation_folds": folds,
        "validation_mean": _mean(folds),
        "sealed_fold_sha256": _sealed_fold_hashes(archive, CERTIFICATION_FOLDS),
    }


def _ingest_attempt(
    cell_root: Path, state: Mapping[str, Any], attempt_dir: Path, plan: _Plan,
) -> dict[str, dict[str, dict[str, Any]]]:
    expected_metrics = VALIDATION_SCHEMA_BY_FAMILY[_cell_task_family(cell_root, state)]
    return {
        strategy: {
            candidate.candidate_id: _ingest_archive(
                cell_root,
                attempt_dir / strategy / candidate.candidate_id / "archive",
                expected_metrics,
            )
            for candidate in plan.candidates
        }
        for strategy in RESCORE_STRATEGIES
    }


def _validation_lift(
    records: Mapping[str, Mapping[str, Mapping[str, Any]]],
    winner: Mapping[str, Any],
) -> float:
    """Mean paired fold gain of the winner over the baseline, across the
    re-score partitions; zero when the baseline is the winner."""
    if winner["kind"] == "baseline":
        return 0.0
    deltas: list[float] = []
    for strategy in RESCORE_STRATEGIES:
        by_candidate = records[strategy]
        won, base = (
            {
                fold["fold_index"]: fold["primary_value"]
                for fold in by_candidate[candidate_id]["validation_folds"]
            }
            for candidate_id in (winner["candidate_id"], BASELINE_ID)
        )
        deltas.extend(won[fold] - base[fold] for fold in CERTIFICATION_FOLDS)
    return math.fsum(deltas) / len(deltas)


def _record(
    cell_root: Path, planned: Mapping[str, Any], rescore: Mapping[str, Any],
) -> dict[str, Any]:
    """Commit the re-score, provided the state is still the one it was run for."""
    with _stage_lock(cell_root):
        state = load_stage_state(cell_root)
        if state["state_sha256"] != planned["state_sha256"]:
            raise CampaignStageError(
                "the stage state changed while the winner was re-scored; "
                "nothing was recorded"
            )
        return _commit_state(cell_root, {
            **state,
            "phase": "winner-rescored",
            "rescore": dict(rescore),
            "revision": state["revision"] + 1,
            "updated_at": rescore["completed_at"],
            "history": [*state["history"], {
                "event": "winner-rescored",
                "candidate_id": state["winner"]["candidate_id"],
                "commit": rescore["commit"],
                "validation_lift": rescore["validation_lift"],
                "at": rescore["completed_at"],
            }],
        })


def _verify_recorded_rescore(cell_root: Path, state: Mapping[str, Any]) -> None:
    """A recorded re-score is evidence: its bytes must still be the recorded ones."""
    winner, rescore = state.get("winner"), state.get("rescore")
    expected_ids = {BASELINE_ID}
    if isinstance(winner, dict) and winner.get("kind") != "baseline":
        expected_ids.add(str(winner.get("candidate_id")))
    strategies = rescore.get("strategies") if isinstance(rescore, dict) else None
    if (
        not isinstance(strategies, dict)
        or set(strategies) != set(RESCORE_STRATEGIES)
        or any(
            not isinstance(per_candidate, dict) or set(per_candidate) != expected_ids
            for per_candidate in strategies.values()
        )
    ):
        raise CampaignStageError(
            "the recorded re-score does not cover the winner and the baseline "
            "on every re-score partition"
        )
    for candidate_id in expected_ids:
        _rescored_sources(cell_root, state, candidate_id)
    expected_metrics = VALIDATION_SCHEMA_BY_FAMILY[_cell_task_family(cell_root, state)]
    for strategy, per_candidate in strategies.items():
        for candidate_id, record in per_candidate.items():
            observed = _ingest_archive(
                cell_root, cell_root / str(record.get("archive")), expected_metrics,
            )
            if content_sha256(observed) != content_sha256(record):
                raise CampaignStageError(
                    f"recorded re-score evidence changed: {strategy}/{candidate_id}"
                )
