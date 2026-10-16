"""The winner re-score stage trains the frozen winner and the baseline again on
the two fresh development partitions and records validation evidence only.

The training subprocess is faked at ``subprocess.run`` with the same shape the
reproduction-gate tests use, so the real executor (worktree lifecycle, overlay
staging, environment) runs under every test here.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from autobench.campaign import (
    CERTIFICATION_FOLDS,
    CERTIFICATION_PARTITIONS,
    CERTIFICATION_SOURCES,
    RESCORE_STRATEGIES,
    SPLIT_STRATEGY,
)
from autobench.campaign_rescore import run_winner_rescore
from autobench.campaign_stages import (
    CampaignStageError,
    _commit_state,
    _execute_frozen_command,
    _stage_manifest_files,
    load_stage_state,
)

from _helpers import (
    full_h100_nvidia_smi, write_fake_rescore, write_reproduction_policy,
)
from test_campaign_stages import (  # noqa: F401  (staged_cell is a fixture)
    CLEAR_WINNER_LIFTS,
    EXECUTION_COMMIT,
    TRAILING_POOL_LIFTS,
    _folds,
    _make_staged_cell,
    _selected,
    staged_cell,
)

WINNER_ID = "node_0007"
#: The winner's policy file: the one file its overlay manifest claims.
OVERLAY_FILE = (
    "dataset__arm__task/automil/variants/_policies/candidate_6.py"
)
#: What the winner adds to the baseline's validation value on every fold of a
#: re-score partition; the mean of the ten paired deltas is 0.03.
WINNER_VALIDATION_LIFT = {"devcv_r1": 0.04, "devcv_r2": 0.02}
#: The passing reproduction verdict the staged cell records (test_campaign_stages
#: ``_reproduced``): the second commit a session may launch at.
REPRODUCTION_COMMIT = "f" * 40


class FakeTraining:
    """``subprocess.run`` for the re-score.

    Answers the identity git calls and the worktree lifecycle, and plays
    ``run_experiment.py``: it writes the public (validation-only) result to
    its working directory and the sealed per-fold files, held-out values
    included, to ``AUTOMIL_RESULTS_DIR``, as the real script does.
    """

    def __init__(self, *, fail=None, barrier_parties=None, head=EXECUTION_COMMIT):
        self.head = head
        self.runs: list[dict] = []
        self.worktree_refs: list[str] = []
        self.commands: list[list[str]] = []
        self._lock = threading.Lock()
        self._fail = fail
        self._barrier = (
            threading.Barrier(barrier_parties, timeout=10)
            if barrier_parties else None
        )

    def __call__(self, command, **kwargs):
        with self._lock:
            self.commands.append(list(command))
        if (gpu := full_h100_nvidia_smi(command)) is not None:
            return gpu
        if command[:2] == ["git", "rev-parse"]:
            return SimpleNamespace(returncode=0, stdout=self.head + "\n", stderr="")
        if command[:3] == ["git", "diff", "--quiet"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if command[:3] == ["git", "worktree", "add"]:
            Path(command[4]).mkdir(parents=True)
            with self._lock:
                self.worktree_refs.append(command[5])
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if command[:3] == ["git", "worktree", "remove"]:
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        return self._run_experiment(command, **kwargs)

    def _run_experiment(self, command, *, cwd, env, stdout, **_):
        tokens = list(command[2:])
        worktree = Path(cwd)
        strategy = tokens[tokens.index("--strategy") + 1]
        searched = "--policy-variant" in tokens or "--hparams" in tokens
        run = {
            "kind": "prep" if "--prepare-only" in tokens else "train",
            "tokens": tokens,
            "strategy": strategy,
            "searched": searched,
            "env": dict(env),
            "worktree": worktree,
            "worktree_files": sorted(
                path.relative_to(worktree).as_posix()
                for path in worktree.rglob("*") if path.is_file()
            ),
            "results_dir_before": sorted(
                path.name for path in Path(env["AUTOMIL_RESULTS_DIR"]).iterdir()
            ),
        }
        with self._lock:
            self.runs.append(run)
        if self._barrier is not None and run["kind"] == "train":
            self._barrier.wait()
        if self._fail is not None and self._fail(run):
            return SimpleNamespace(returncode=3)
        if run["kind"] == "train":
            self._write_results(run)
        return SimpleNamespace(returncode=0)

    @staticmethod
    def _write_results(run):
        strategy, searched = run["strategy"], run["searched"]
        shift = 0.02 if strategy == "devcv_r2" else 0.0
        lift = WINNER_VALIDATION_LIFT[strategy] if searched else 0.0
        folds = _folds(CERTIFICATION_FOLDS, 0.60 + shift)
        for fold in folds:
            fold["primary_value"] += lift
            fold["metrics"]["val_auc_smooth"] += lift
        (run["worktree"] / "result.json").write_text(json.dumps({
            "status": "completed",
            "primary_value": 0.62,
            "metrics": {"val_auc_smooth": 0.62, "val_auc": 0.64, "val_bacc": 0.60},
            "validation_folds": folds,
        }))
        sealed = Path(run["env"]["AUTOMIL_RESULTS_DIR"])
        for fold in CERTIFICATION_FOLDS:
            held_out = 0.50 + fold / 100 + (0.05 if searched else 0.0) + shift / 2
            (sealed / f"fold_{fold}_result.json").write_text(json.dumps({
                "fold_index": fold,
                "held_out": {"test_auc": held_out, "test_bacc": held_out},
            }))

    def of_kind(self, kind):
        return [run for run in self.runs if run["kind"] == kind]


def _ready(staged_cell, monkeypatch, *, searched=True, **fake_options):
    """A cell frozen on a winner, with the fake training installed.

    ``searched`` selects node_0007, a policy candidate whose overlay claims
    one file; otherwise no candidate beats the baseline.
    """
    cell_root, _, _, _, repo_root = staged_cell
    if searched:
        _selected(staged_cell, CLEAR_WINNER_LIFTS, source_at=6)
    else:
        _selected(staged_cell, TRAILING_POOL_LIFTS)
    write_reproduction_policy(repo_root, runtimes=(cell_root.parent.name,))
    fake = FakeTraining(**fake_options)
    monkeypatch.setattr("autobench.campaign_stages.subprocess.run", fake)
    return cell_root, repo_root, fake


def _walk_keys(value):
    if isinstance(value, dict):
        for key, inner in value.items():
            yield key
            yield from _walk_keys(inner)
    elif isinstance(value, list):
        for inner in value:
            yield from _walk_keys(inner)


def test_the_constants_name_three_partitions_of_five_folds():
    assert RESCORE_STRATEGIES == ("devcv_r1", "devcv_r2")
    assert CERTIFICATION_PARTITIONS == (SPLIT_STRATEGY, *RESCORE_STRATEGIES)
    assert len(CERTIFICATION_SOURCES) == 15
    assert CERTIFICATION_SOURCES[:2] == ((SPLIT_STRATEGY, 0), (SPLIT_STRATEGY, 1))
    assert CERTIFICATION_SOURCES[-1] == ("devcv_r2", 4)


def test_every_rescore_strategy_is_a_registered_split_strategy():
    from autobench.pipeline.splits import DEVCV_PARTITION_SEEDS

    assert set(CERTIFICATION_PARTITIONS) <= set(DEVCV_PARTITION_SEEDS)


def test_a_searched_winner_is_prepared_then_trained_on_both_fresh_partitions(
    staged_cell, monkeypatch,
):
    cell_root, repo_root, fake = _ready(staged_cell, monkeypatch)

    state = run_winner_rescore(cell_root, repo_root=repo_root, gpu_id=2)

    kinds = [run["kind"] for run in fake.runs]
    assert kinds == ["prep", "prep", "train", "train", "train", "train"]
    preps = fake.of_kind("prep")
    assert [run["strategy"] for run in preps] == list(RESCORE_STRATEGIES)
    assert all("--prepare-only" in run["tokens"] for run in preps)
    assert all(run["worktree_files"] == [] for run in preps)
    trains = fake.of_kind("train")
    assert sorted((run["strategy"], run["searched"]) for run in trains) == [
        ("devcv_r1", False), ("devcv_r1", True),
        ("devcv_r2", False), ("devcv_r2", True),
    ]
    for run in trains:
        tokens = run["tokens"]
        assert "--skip-prep" in tokens and "--prepare-only" not in tokens
        assert SPLIT_STRATEGY not in tokens
        assert tokens[tokens.index("--folds") + 1] == "0,1,2,3,4"
        if run["searched"]:
            # The winner's own override tokens follow the manifest command.
            assert tokens[-2:] == ["--policy-variant", "candidate_6"]
        else:
            assert "--policy-variant" not in tokens
    assert state["phase"] == "winner-rescored"
    assert state["history"][-1]["event"] == "winner-rescored"


def test_every_run_executes_at_head_with_loop_parity(staged_cell, monkeypatch):
    cell_root, repo_root, fake = _ready(staged_cell, monkeypatch)

    state = run_winner_rescore(cell_root, repo_root=repo_root, gpu_id=2)

    assert fake.worktree_refs == ["HEAD"] * 6
    assert state["rescore"]["commit"] == EXECUTION_COMMIT
    for run in fake.runs:
        env = run["env"]
        assert env["AUTOMIL_DIR_REL"] == "dataset__arm__task/automil"
        assert env["CUDA_VISIBLE_DEVICES"] == "2" and env["AUTOMIL_GPU"] == "0"
        assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        assert env["PYTHONUNBUFFERED"] == "1"
        assert env["AUTOMIL_FOLD_COUNT"] == "5"
        assert str(run["worktree"]) not in env.get("PYTHONPATH", "")


def test_the_winner_overlay_reaches_only_the_winners_worktrees(
    staged_cell, monkeypatch,
):
    cell_root, repo_root, fake = _ready(staged_cell, monkeypatch)

    run_winner_rescore(cell_root, repo_root=repo_root)

    winner_runs = [run for run in fake.of_kind("train") if run["searched"]]
    baseline_runs = [run for run in fake.of_kind("train") if not run["searched"]]
    assert len(winner_runs) == len(baseline_runs) == 2
    # The sealed discovery folds and logs sit beside the overlay in the winner's
    # archive; the worktree gets the one claimed file and nothing else.
    assert all(run["worktree_files"] == [OVERLAY_FILE] for run in winner_runs)
    assert all(run["worktree_files"] == [] for run in baseline_runs)


def test_staging_copies_the_manifest_claimed_files_and_nothing_else(tmp_path):
    archive = tmp_path / "archive"
    for relative in ("pkg/claimed.py", "certify.json", "summary.json",
                     "certify/fold_0_result.json", "pkg/stray.py"):
        (archive / relative).parent.mkdir(parents=True, exist_ok=True)
        (archive / relative).write_text(relative)
    staging = tmp_path / "staging"
    staging.mkdir()

    _stage_manifest_files(archive, {"pkg/claimed.py": "sha256:0"}, staging)

    assert [
        path.relative_to(staging).as_posix()
        for path in staging.rglob("*") if path.is_file()
    ] == ["pkg/claimed.py"]


@pytest.mark.parametrize("escape", ["../outside.py", "/etc/passwd", "pkg/../../x"])
def test_staging_refuses_a_manifest_path_that_leaves_the_archive(tmp_path, escape):
    (tmp_path / "archive").mkdir()
    (tmp_path / "staging").mkdir()

    with pytest.raises(CampaignStageError, match="escapes the archive"):
        _stage_manifest_files(
            tmp_path / "archive", {escape: "sha256:0"}, tmp_path / "staging",
        )


def test_a_tampered_overlay_is_refused_before_any_run(staged_cell, monkeypatch):
    cell_root, repo_root, fake = _ready(staged_cell, monkeypatch)
    before = load_stage_state(cell_root)
    (cell_root / "automil/orchestrator/archive" / WINNER_ID / OVERLAY_FILE).write_text(
        "# edited after submit\n"
    )

    with pytest.raises(CampaignStageError, match="digest mismatch"):
        run_winner_rescore(cell_root, repo_root=repo_root)

    assert fake.runs == []
    assert not any(command[:3] == ["git", "worktree", "add"] for command in fake.commands)
    assert load_stage_state(cell_root) == before


def test_a_baseline_winner_trains_only_the_baseline_and_gains_nothing(
    staged_cell, monkeypatch,
):
    cell_root, repo_root, fake = _ready(staged_cell, monkeypatch, searched=False)

    state = run_winner_rescore(cell_root, repo_root=repo_root)

    assert [run["kind"] for run in fake.runs] == ["prep", "prep", "train", "train"]
    assert not any(run["searched"] for run in fake.runs)
    assert state["winner"]["kind"] == "baseline"
    assert state["rescore"]["validation_lift"] == 0.0
    assert set(state["rescore"]["strategies"]) == set(RESCORE_STRATEGIES)
    for per_candidate in state["rescore"]["strategies"].values():
        assert set(per_candidate) == {"baseline"}


def test_the_record_pairs_the_winner_with_the_baseline_on_each_partition(
    staged_cell, monkeypatch,
):
    cell_root, repo_root, fake = _ready(staged_cell, monkeypatch)

    state = run_winner_rescore(cell_root, repo_root=repo_root)

    rescore = state["rescore"]
    assert rescore["attempt"].startswith("rescore/attempt-")
    assert list(rescore["strategies"]) == list(RESCORE_STRATEGIES)
    for strategy, per_candidate in rescore["strategies"].items():
        assert set(per_candidate) == {"baseline", WINNER_ID}
        for candidate_id, record in per_candidate.items():
            assert set(record) == {
                "archive", "validation_folds", "validation_mean",
                "sealed_fold_sha256",
            }
            assert record["archive"] == (
                f"{rescore['attempt']}/{strategy}/{candidate_id}/archive"
            )
            assert [f["fold_index"] for f in record["validation_folds"]] == list(
                CERTIFICATION_FOLDS
            )
            assert sorted(record["sealed_fold_sha256"]) == [
                f"fold_{fold}_result.json" for fold in CERTIFICATION_FOLDS
            ]
            for filename in record["sealed_fold_sha256"]:
                assert (cell_root / record["archive"] / "certify" / filename).is_file()
        lift = WINNER_VALIDATION_LIFT[strategy]
        assert per_candidate[WINNER_ID]["validation_mean"] == pytest.approx(
            per_candidate["baseline"]["validation_mean"] + lift
        )
    assert rescore["validation_lift"] == pytest.approx(0.03)
    assert rescore["completed_at"] == state["updated_at"]


def test_the_shared_fixture_writes_the_block_the_stage_records(
    staged_cell, monkeypatch,
):
    """Selection-freeze and analysis fixtures stand in for a finished re-score
    with ``write_fake_rescore``; its block must not drift from the real one."""
    cell_root, repo_root, _ = _ready(staged_cell, monkeypatch)
    recorded = run_winner_rescore(cell_root, repo_root=repo_root)

    fake = write_fake_rescore(
        cell_root, recorded,
        held_out=lambda strategy, candidate_id, fold: {"test_auc": 0.5},
    )

    real = recorded["rescore"]
    assert set(fake) == set(real)
    assert list(fake["strategies"]) == list(real["strategies"])
    for strategy, per_candidate in real["strategies"].items():
        assert set(fake["strategies"][strategy]) == set(per_candidate)
        for candidate_id, record in per_candidate.items():
            assert set(fake["strategies"][strategy][candidate_id]) == set(record)


def test_the_state_holds_validation_evidence_only(staged_cell, monkeypatch):
    """The sealed folds the fake writes carry held-out values; the record the
    stage keeps must not mention them in any form."""
    cell_root, repo_root, _ = _ready(staged_cell, monkeypatch)

    state = run_winner_rescore(cell_root, repo_root=repo_root)

    keys = set(_walk_keys(state["rescore"]))
    assert "held_out" not in keys
    assert not any("test" in str(key) for key in keys)
    on_disk = (cell_root / "campaign_state.json").read_text()
    assert "held_out" not in on_disk and "test_auc" not in on_disk


def test_a_failed_run_leaves_the_phase_at_winner_frozen(staged_cell, monkeypatch):
    cell_root, repo_root, fake = _ready(
        staged_cell, monkeypatch,
        fail=lambda run: run["kind"] == "train"
        and run["strategy"] == "devcv_r2" and run["searched"],
    )
    before = load_stage_state(cell_root)

    with pytest.raises(CampaignStageError, match="exited with code 3"):
        run_winner_rescore(cell_root, repo_root=repo_root)

    after = load_stage_state(cell_root)
    assert after == before
    assert after["phase"] == "winner-frozen" and "rescore" not in after


def test_a_missing_public_result_fails_the_stage(staged_cell, monkeypatch):
    class NoPublicResult(FakeTraining):
        @staticmethod
        def _write_results(run):
            return None

    cell_root, repo_root, _ = _ready(staged_cell, monkeypatch)
    monkeypatch.setattr(
        "autobench.campaign_stages.subprocess.run", NoPublicResult(),
    )

    with pytest.raises(CampaignStageError, match="result"):
        run_winner_rescore(cell_root, repo_root=repo_root)

    assert load_stage_state(cell_root)["phase"] == "winner-frozen"


def test_the_four_training_runs_are_in_flight_together(staged_cell, monkeypatch):
    """The barrier only opens when all four runs are waiting at once; a stage
    that trains them one after another breaks it and fails."""
    cell_root, repo_root, fake = _ready(staged_cell, monkeypatch, barrier_parties=4)

    state = run_winner_rescore(cell_root, repo_root=repo_root)

    assert state["phase"] == "winner-rescored"
    assert len(fake.of_kind("train")) == 4


def test_a_retry_after_a_failure_never_reuses_the_failed_attempt_directory(
    staged_cell, monkeypatch,
):
    cell_root, repo_root, fake = _ready(
        staged_cell, monkeypatch,
        fail=lambda run: run["kind"] == "train" and run["searched"],
    )
    with pytest.raises(CampaignStageError):
        run_winner_rescore(cell_root, repo_root=repo_root)
    first_attempts = sorted((cell_root / "rescore").iterdir())
    assert len(first_attempts) == 1

    healthy = FakeTraining()
    monkeypatch.setattr("autobench.campaign_stages.subprocess.run", healthy)
    state = run_winner_rescore(cell_root, repo_root=repo_root)

    attempts = sorted((cell_root / "rescore").iterdir())
    assert len(attempts) == 2
    assert state["rescore"]["attempt"] == f"rescore/{[a for a in attempts if a != first_attempts[0]][0].name}"
    # Every run of the retry started on an empty sealed directory, so a trainer
    # resume cache could not short-circuit a fold.
    assert all(run["results_dir_before"] == [] for run in healthy.of_kind("train"))


def test_a_recorded_rescore_is_verified_and_returned_without_running(
    staged_cell, monkeypatch,
):
    cell_root, repo_root, fake = _ready(staged_cell, monkeypatch)
    first = run_winner_rescore(cell_root, repo_root=repo_root)
    runs = len(fake.runs)

    again = run_winner_rescore(cell_root, repo_root=repo_root)

    assert again == first
    assert len(fake.runs) == runs


@pytest.mark.parametrize("tamper", ["sealed-fold", "public-result"])
def test_a_recorded_rescore_is_byte_verified_on_every_call(
    staged_cell, monkeypatch, tamper,
):
    cell_root, repo_root, _ = _ready(staged_cell, monkeypatch)
    state = run_winner_rescore(cell_root, repo_root=repo_root)
    archive = cell_root / state["rescore"]["strategies"]["devcv_r1"][WINNER_ID]["archive"]
    if tamper == "sealed-fold":
        (archive / "certify/fold_3_result.json").write_text("{}")
    else:
        result = json.loads((archive / "result.json").read_text())
        result["validation_folds"][0]["primary_value"] = 0.99
        result["validation_folds"][0]["metrics"]["val_auc_smooth"] = 0.99
        (archive / "result.json").write_text(json.dumps(result))

    with pytest.raises(CampaignStageError, match="re-score"):
        run_winner_rescore(cell_root, repo_root=repo_root)


@pytest.mark.parametrize("phase", ["discovery", "selection-ready", "certified"])
def test_the_stage_needs_a_frozen_winner(staged_cell, monkeypatch, phase):
    cell_root, repo_root, fake = _ready(staged_cell, monkeypatch)
    state = load_stage_state(cell_root)
    _commit_state(cell_root, {**state, "phase": phase})

    with pytest.raises(CampaignStageError, match="winner-frozen"):
        run_winner_rescore(cell_root, repo_root=repo_root)

    assert fake.runs == []


def test_a_second_invocation_is_refused_while_one_is_running(
    staged_cell, monkeypatch,
):
    import fcntl

    cell_root, repo_root, fake = _ready(staged_cell, monkeypatch)
    with (cell_root / ".winner_rescore.lock").open("a+") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX)
        with pytest.raises(CampaignStageError, match="already running"):
            run_winner_rescore(cell_root, repo_root=repo_root)
    assert fake.runs == []


def test_a_spec_changed_after_the_freeze_is_refused(staged_cell, monkeypatch):
    cell_root, repo_root, fake = _ready(staged_cell, monkeypatch)
    spec_path = (
        cell_root / "automil/orchestrator/archive" / WINNER_ID / "spec.json"
    )
    spec = json.loads(spec_path.read_text())
    spec["description"] = "edited after the discovery freeze"
    spec_path.write_text(json.dumps(spec))

    with pytest.raises(CampaignStageError, match="spec changed"):
        run_winner_rescore(cell_root, repo_root=repo_root)

    assert fake.runs == []


def test_a_head_no_session_could_launch_at_is_refused(staged_cell, monkeypatch):
    """The trainers import the main checkout, so a moved HEAD would train other
    code under the recorded commit."""
    cell_root, repo_root, fake = _ready(staged_cell, monkeypatch, head="9" * 40)

    with pytest.raises(CampaignStageError, match="code identity drift"):
        run_winner_rescore(cell_root, repo_root=repo_root)

    assert fake.runs == []


def test_a_winner_submitted_at_another_commit_than_head_is_refused(
    staged_cell, monkeypatch,
):
    cell_root, repo_root, fake = _ready(
        staged_cell, monkeypatch, head=REPRODUCTION_COMMIT,
    )

    with pytest.raises(CampaignStageError, match="check that commit out"):
        run_winner_rescore(cell_root, repo_root=repo_root)

    assert fake.runs == []


def test_a_baseline_winner_is_rescored_at_the_reproduction_anchor(
    staged_cell, monkeypatch,
):
    cell_root, repo_root, fake = _ready(
        staged_cell, monkeypatch, searched=False, head=REPRODUCTION_COMMIT,
    )

    state = run_winner_rescore(cell_root, repo_root=repo_root)

    assert state["rescore"]["commit"] == REPRODUCTION_COMMIT
    assert len(fake.of_kind("train")) == 2


def test_a_cell_without_a_code_identity_anchor_is_refused(
    staged_cell, monkeypatch,
):
    cell_root, repo_root, fake = _ready(staged_cell, monkeypatch, searched=False)
    state = load_stage_state(cell_root)
    anonymous = {
        key: value for key, value in state["baseline"].items()
        if key != "execution_identity"
    }
    _commit_state(cell_root, {
        **state, "baseline": anonymous, "baseline_reproduction": None,
    })

    with pytest.raises(CampaignStageError, match="code-identity anchor"):
        run_winner_rescore(cell_root, repo_root=repo_root)

    assert fake.runs == []


def test_a_changed_base_run_command_is_refused(staged_cell, monkeypatch):
    cell_root, repo_root, fake = _ready(staged_cell, monkeypatch)
    config_path = cell_root / "automil/config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["run"]["command"] += " --no_wandb"
    config_path.write_text(yaml.safe_dump(config))

    with pytest.raises(CampaignStageError, match="base run command"):
        run_winner_rescore(cell_root, repo_root=repo_root)

    assert fake.runs == []


@pytest.mark.parametrize("command", [
    "python benchmarks/scripts/run_experiment.py --folds 0,1,2,3,4",
    "python benchmarks/scripts/run_experiment.py --strategy standard --folds 0,1,2,3,4",
    "python benchmarks/scripts/run_experiment.py --strategy devcv --strategy devcv",
])
def test_the_discovery_command_must_name_the_discovery_strategy_once(
    tmp_path, monkeypatch, command,
):
    staged = _make_staged_cell(tmp_path, discovery_command=command)
    cell_root, repo_root = staged[0], staged[4]
    _selected(staged, CLEAR_WINNER_LIFTS, source_at=6)
    write_reproduction_policy(repo_root, runtimes=(cell_root.parent.name,))
    fake = FakeTraining()
    monkeypatch.setattr("autobench.campaign_stages.subprocess.run", fake)

    with pytest.raises(CampaignStageError, match="--strategy"):
        run_winner_rescore(cell_root, repo_root=repo_root)

    assert fake.runs == []


def test_a_dirty_training_tree_is_refused(staged_cell, monkeypatch):
    cell_root, repo_root, fake = _ready(staged_cell, monkeypatch)
    clean = fake.__call__

    def dirty(command, **kwargs):
        if command[:3] == ["git", "diff", "--quiet"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="")
        return clean(command, **kwargs)

    monkeypatch.setattr("autobench.campaign_stages.subprocess.run", dirty)

    with pytest.raises(CampaignStageError, match="tracked modifications"):
        run_winner_rescore(cell_root, repo_root=repo_root)

    assert fake.runs == []


def test_the_declared_gpu_type_is_enforced_before_any_run(staged_cell, monkeypatch):
    cell_root, repo_root, fake = _ready(staged_cell, monkeypatch)
    inner = fake.__call__

    def slice_node(command, **kwargs):
        if command and command[0] == "nvidia-smi":
            return SimpleNamespace(
                returncode=0, stdout="0, NVIDIA H100 80GB HBM3, Enabled\n", stderr="",
            )
        return inner(command, **kwargs)

    monkeypatch.setattr("autobench.campaign_stages.subprocess.run", slice_node)

    with pytest.raises(CampaignStageError, match="MIG enabled"):
        run_winner_rescore(cell_root, repo_root=repo_root)

    assert fake.runs == []
    assert load_stage_state(cell_root)["phase"] == "winner-frozen"


# --- the shared executor -------------------------------------------------


def _execute(staged_cell, **options):
    cell_root, _, _, _, repo_root = staged_cell
    archive_dir = cell_root / "exec-archive"
    sealed = archive_dir / "certify"
    sealed.mkdir(parents=True, exist_ok=True)
    return _execute_frozen_command(
        cell_root=cell_root,
        repo_root=repo_root,
        command_string=(
            "python benchmarks/scripts/run_experiment.py --strategy devcv"
        ),
        worktree_prefix=".exec-worktree-",
        node_id="executor-test",
        fold_count=5,
        gpu_id=0,
        results_dir=sealed,
        archive_dir=archive_dir,
        inject_worktree_pythonpath=False,
        **options,
    )


def test_the_executor_defaults_to_head_and_appends_nothing(staged_cell, monkeypatch):
    cell_root, _, _, _, repo_root = staged_cell
    write_reproduction_policy(repo_root, runtimes=(cell_root.parent.name,))
    fake = FakeTraining()
    monkeypatch.setattr("autobench.campaign_stages.subprocess.run", fake)

    executed = _execute(staged_cell)

    assert executed == fake.head
    assert fake.worktree_refs == ["HEAD"]
    assert fake.runs[0]["tokens"] == ["--strategy", "devcv"]


def test_the_executor_appends_extra_args_after_the_manifest_command(
    staged_cell, monkeypatch,
):
    cell_root, _, _, _, repo_root = staged_cell
    write_reproduction_policy(repo_root, runtimes=(cell_root.parent.name,))
    fake = FakeTraining()
    monkeypatch.setattr("autobench.campaign_stages.subprocess.run", fake)

    executed = _execute(staged_cell, extra_args=("--skip-prep", "--no_wandb"))

    assert executed == fake.head
    assert fake.worktree_refs == ["HEAD"]
    assert fake.runs[0]["tokens"] == [
        "--strategy", "devcv", "--skip-prep", "--no_wandb",
    ]


def test_worktree_creation_and_removal_never_overlap_across_threads(
    staged_cell, monkeypatch,
):
    cell_root, _, _, _, repo_root = staged_cell
    write_reproduction_policy(repo_root, runtimes=(cell_root.parent.name,))
    fake = FakeTraining()
    inside = {"now": 0, "max": 0}
    guard = threading.Lock()

    def tracked(command, **kwargs):
        if command[:2] != ["git", "worktree"]:
            return fake(command, **kwargs)
        with guard:
            inside["now"] += 1
            inside["max"] = max(inside["max"], inside["now"])
        try:
            threading.Event().wait(0.02)
            return fake(command, **kwargs)
        finally:
            with guard:
                inside["now"] -= 1

    monkeypatch.setattr("autobench.campaign_stages.subprocess.run", tracked)
    threads = [
        threading.Thread(target=_execute, args=(staged_cell,)) for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(fake.runs) == 4
    assert inside["max"] == 1
