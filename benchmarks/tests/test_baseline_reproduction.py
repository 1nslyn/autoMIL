"""Contracts for the loop-start baseline reproduction gate.

The search loop's root is a ledger import, never a second measurement.
run_baseline_reproduction is the double-check that the setup the loop is
about to search from still reproduces the registered grid baseline; the
passing verdict is what open_agent_session requires. Every gate rule here
is exercised by a forged violation that tries to defeat it.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from autobench.campaign import REPRODUCTION_POLICY_PATH, STAGE_FOLDS
from autobench.campaign_stages import (
    CampaignStageError,
    load_stage_state,
    open_agent_session,
    register_baseline,
    run_baseline_reproduction,
)

from test_campaign_stages import (  # noqa: F401  (staged_cell is a fixture)
    _baseline,
    _folds,
    _record_session_start,
    staged_cell,
)

DISCOVERY_FOLDS = list(STAGE_FOLDS["discovery"])


def _declare_policy(repo_root: Path, epsilon=0.005) -> Path:
    path = repo_root / REPRODUCTION_POLICY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"epsilon": epsilon}))
    return path


def _fake_execution(
    monkeypatch, *, fold_values, fold_hashes=None, observed=None, head="c" * 40,
):
    """Fake subprocess for the reproduction run.

    Handles the identity git calls, the worktree lifecycle, and the frozen
    discovery command — which writes a stripped result.json into its cwd
    exactly like run_experiment.py does.
    """
    observed = observed if observed is not None else {}

    def fake_run(command, **kwargs):
        if command[:2] == ["git", "rev-parse"]:
            return SimpleNamespace(returncode=0, stdout=head + "\n", stderr="")
        if command[:3] == ["git", "diff", "--quiet"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if command[:3] == ["git", "worktree", "add"]:
            Path(command[4]).mkdir(parents=True)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if command[:3] == ["git", "worktree", "remove"]:
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

        env = kwargs["env"]
        observed["command"] = command
        observed["env"] = dict(env)
        results_dir = Path(env["AUTOMIL_RESULTS_DIR"])
        observed.setdefault("results_dirs", []).append(results_dir)
        observed["results_dir_preexisting"] = sorted(
            entry.name for entry in results_dir.iterdir()
        )
        folds = _folds(DISCOVERY_FOLDS, 0.60)
        for fold_entry, value in zip(folds, fold_values):
            fold_entry["metrics"]["val_auc"] = value
            fold_entry["primary_value"] = value
            if fold_hashes is not None:
                fold_entry["val_predictions_sha256"] = fold_hashes.get(
                    fold_entry["fold_index"]
                )
        result = {
            "status": "completed",
            "primary_value": sum(fold_values) / len(fold_values),
            "metrics": {"val_auc": 0.62, "val_bacc": 0.60},
            "validation_folds": folds,
        }
        (Path(kwargs["cwd"]) / "result.json").write_text(json.dumps(result))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "autobench.campaign_stages.subprocess.run", fake_run,
    )
    return observed


def _baseline_fold_values(cell_root: Path) -> dict[int, float]:
    state = load_stage_state(cell_root)
    return {
        fold["fold_index"]: fold["primary_value"]
        for fold in state["baseline"]["validation_folds"]
        if fold["fold_index"] in set(DISCOVERY_FOLDS)
    }


def _open_session(cell_root: Path, adir: Path, *, record: bool = True):
    if record:
        _record_session_start(adir)
    return open_agent_session(cell_root, {
        "session_id": "fixture-session",
        "started_at": (
            datetime.now(timezone.utc) - timedelta(minutes=1)
        ).isoformat(),
    })


def test_committed_policy_declares_the_signed_off_epsilon():
    """The predeclared tolerance is a committed, reviewable artifact."""
    from pathlib import Path

    from autobench.campaign_stages import _load_reproduction_policy

    repo_root = Path(__file__).resolve().parents[2]
    policy = _load_reproduction_policy(repo_root)
    assert policy["epsilon"] == 0.025


def test_reproduction_requires_a_registered_baseline(staged_cell):
    cell_root, _, _, _, repo_root = staged_cell
    _declare_policy(repo_root)
    with pytest.raises(CampaignStageError, match="registered native baseline"):
        run_baseline_reproduction(cell_root, repo_root=repo_root)


def test_gate_refuses_to_run_without_a_declared_policy(staged_cell):
    """Forged violation: no default epsilon exists anywhere."""
    cell_root, _, _, _, repo_root = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    with pytest.raises(CampaignStageError, match="no default tolerance"):
        run_baseline_reproduction(cell_root, repo_root=repo_root)


@pytest.mark.parametrize("epsilon", [0, -0.1, "0.005", None, True])
def test_gate_refuses_an_invalid_epsilon(staged_cell, epsilon):
    cell_root, _, _, _, repo_root = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    path = repo_root / REPRODUCTION_POLICY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"epsilon": epsilon}))
    with pytest.raises(CampaignStageError, match="epsilon"):
        run_baseline_reproduction(cell_root, repo_root=repo_root)


def test_passing_reproduction_unblocks_the_session(staged_cell, monkeypatch):
    cell_root, adir, _, _, repo_root = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _declare_policy(repo_root, epsilon=0.005)
    base = _baseline_fold_values(cell_root)
    observed = _fake_execution(
        monkeypatch,
        fold_values=[base[fold] + 0.004 for fold in DISCOVERY_FOLDS],
    )

    with pytest.raises(CampaignStageError, match="passing baseline reproduction"):
        _open_session(cell_root, adir)

    state = run_baseline_reproduction(cell_root, repo_root=repo_root)
    block = state["baseline_reproduction"]
    assert block["mode"] == "gate"
    assert block["verdict"] == "pass"
    assert block["epsilon"] == 0.005
    assert block["commit"] == "c" * 40
    assert [fold["fold_index"] for fold in block["folds"]] == DISCOVERY_FOLDS
    assert all(
        fold["delta"] == pytest.approx(0.004) for fold in block["folds"]
    )
    # Loop parity: the discovery command ran, on discovery folds, without
    # worktree PYTHONPATH injection.
    assert "0,1,2" in observed["command"]
    assert "0,1,2,3,4" not in observed["command"]
    assert observed["env"]["AUTOMIL_FOLD_COUNT"] == str(len(DISCOVERY_FOLDS))
    assert observed["env"]["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert "PYTHONPATH" not in observed["env"] or "repo" not in observed["env"].get(
        "PYTHONPATH", ""
    )
    assert observed["results_dir_preexisting"] == []

    assert _open_session(cell_root, adir, record=False)["status"] == "open"


def test_fold_delta_beyond_epsilon_fails_closed(staged_cell, monkeypatch):
    """Forged violation: one drifted fold blocks discovery, and the verdict
    survives in state so the session gate refuses afterward too."""
    cell_root, adir, _, _, repo_root = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _declare_policy(repo_root, epsilon=0.005)
    base = _baseline_fold_values(cell_root)
    values = [base[fold] for fold in DISCOVERY_FOLDS]
    values[2] += 0.02
    _fake_execution(monkeypatch, fold_values=values)

    with pytest.raises(CampaignStageError, match="FAILED.*folds \\[2\\]"):
        run_baseline_reproduction(cell_root, repo_root=repo_root)
    state = load_stage_state(cell_root)
    assert state["baseline_reproduction"]["verdict"] == "fail"
    assert state["baseline_reproduction"]["exceeding_folds"] == [2]
    with pytest.raises(CampaignStageError, match="passing baseline reproduction"):
        _open_session(cell_root, adir)


def test_wrong_fold_set_fails_closed(staged_cell, monkeypatch):
    """Forged violation: verifying a subset is worse than not verifying."""
    cell_root, _, _, _, repo_root = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _declare_policy(repo_root)
    base = _baseline_fold_values(cell_root)

    def truncating(monkeypatched_values):
        observed = {}

        def fake_run(command, **kwargs):
            if command[:2] == ["git", "rev-parse"]:
                return SimpleNamespace(
                    returncode=0, stdout="c" * 40 + "\n", stderr="",
                )
            if command[:3] == ["git", "diff", "--quiet"]:
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            if command[:3] == ["git", "worktree", "add"]:
                Path(command[4]).mkdir(parents=True)
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            if command[:3] == ["git", "worktree", "remove"]:
                return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
            result = {
                "status": "completed",
                "primary_value": 0.6,
                "metrics": {"val_auc": 0.62, "val_bacc": 0.60},
                "validation_folds": _folds(DISCOVERY_FOLDS[:-1], 0.60),
            }
            (Path(kwargs["cwd"]) / "result.json").write_text(json.dumps(result))
            return SimpleNamespace(returncode=0)

        return fake_run, observed

    fake_run, _ = truncating(base)
    import autobench.campaign_stages as stages
    orig = stages.subprocess.run
    monkeypatch.setattr("autobench.campaign_stages.subprocess.run", fake_run)
    try:
        with pytest.raises(CampaignStageError, match="folds must be exactly"):
            run_baseline_reproduction(cell_root, repo_root=repo_root)
    finally:
        monkeypatch.setattr("autobench.campaign_stages.subprocess.run", orig)
    assert load_stage_state(cell_root).get("baseline_reproduction") is None


def test_recorded_verdict_is_immutable_without_force(staged_cell, monkeypatch):
    """No silent retry-until-pass: a second run must be explicit and the
    superseded verdict stays in history."""
    cell_root, _, _, _, repo_root = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _declare_policy(repo_root, epsilon=0.005)
    base = _baseline_fold_values(cell_root)
    values = [base[fold] for fold in DISCOVERY_FOLDS]
    values[0] += 0.02
    _fake_execution(monkeypatch, fold_values=values)
    with pytest.raises(CampaignStageError, match="FAILED"):
        run_baseline_reproduction(cell_root, repo_root=repo_root)

    _fake_execution(
        monkeypatch, fold_values=[base[fold] for fold in DISCOVERY_FOLDS],
    )
    with pytest.raises(CampaignStageError, match="--force"):
        run_baseline_reproduction(cell_root, repo_root=repo_root)

    state = run_baseline_reproduction(
        cell_root, repo_root=repo_root, force=True,
    )
    assert state["baseline_reproduction"]["verdict"] == "pass"
    superseded = [
        event for event in state["history"]
        if event.get("superseded_verdict") is not None
    ]
    assert superseded and superseded[-1]["superseded_verdict"] == "fail"


def test_passing_verdict_is_idempotent(staged_cell, monkeypatch):
    cell_root, _, _, _, repo_root = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _declare_policy(repo_root)
    base = _baseline_fold_values(cell_root)
    observed = _fake_execution(
        monkeypatch, fold_values=[base[fold] for fold in DISCOVERY_FOLDS],
    )
    first = run_baseline_reproduction(cell_root, repo_root=repo_root)
    again = run_baseline_reproduction(cell_root, repo_root=repo_root)
    assert again["baseline_reproduction"] == first["baseline_reproduction"]
    assert len(observed["results_dirs"]) == 1


def test_tampered_baseline_archive_refuses_before_any_run(
    staged_cell, monkeypatch,
):
    """Forged violation: a recorded pass must not survive baseline tamper."""
    cell_root, _, _, _, repo_root = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _declare_policy(repo_root)
    base = _baseline_fold_values(cell_root)
    _fake_execution(
        monkeypatch, fold_values=[base[fold] for fold in DISCOVERY_FOLDS],
    )
    run_baseline_reproduction(cell_root, repo_root=repo_root)

    archive_result = cell_root / "baseline" / "archive" / "result.json"
    payload = json.loads(archive_result.read_text())
    payload["metrics"]["val_auc"] = 0.99
    archive_result.write_text(json.dumps(payload))

    with pytest.raises(CampaignStageError):
        run_baseline_reproduction(cell_root, repo_root=repo_root)


def test_measurement_mode_records_spread_but_never_satisfies_the_gate(
    staged_cell, monkeypatch,
):
    cell_root, adir, _, _, repo_root = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    base = _baseline_fold_values(cell_root)
    _fake_execution(
        monkeypatch,
        fold_values=[base[fold] + 0.03 for fold in DISCOVERY_FOLDS],
    )
    state = run_baseline_reproduction(
        cell_root, repo_root=repo_root, measure=True,
    )
    block = state["baseline_reproduction"]
    assert block["mode"] == "measurement"
    assert block["verdict"] == "measured"
    assert block["epsilon"] is None
    assert all(
        fold["delta"] == pytest.approx(0.03) for fold in block["folds"]
    )
    with pytest.raises(CampaignStageError, match="passing baseline reproduction"):
        _open_session(cell_root, adir)


def test_prediction_hashes_are_diagnosis_never_a_gate(staged_cell, monkeypatch):
    """Hash inequality with matching metrics still passes; the mismatch is
    recorded. ~100/130 arms are not bit-deterministic by design."""
    cell_root, _, _, _, repo_root = staged_cell
    register_baseline(cell_root, _baseline(
        cell_root,
        val_hashes={fold: "a" * 64 for fold in range(5)},
    ))
    _declare_policy(repo_root)
    base = _baseline_fold_values(cell_root)
    _fake_execution(
        monkeypatch,
        fold_values=[base[fold] for fold in DISCOVERY_FOLDS],
        fold_hashes={0: "a" * 64, 1: "b" * 64, 2: "a" * 64},
    )
    state = run_baseline_reproduction(cell_root, repo_root=repo_root)
    block = state["baseline_reproduction"]
    assert block["verdict"] == "pass"
    matches = {
        fold["fold_index"]: fold["prediction_hash_match"]
        for fold in block["folds"]
    }
    assert matches == {0: True, 1: False, 2: True}


def test_forged_session_verdict_bound_to_other_baseline_is_refused(
    staged_cell, monkeypatch,
):
    """Forged violation: a pass copied from another cell/baseline must not
    open the session."""
    cell_root, adir, _, _, repo_root = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _declare_policy(repo_root)
    base = _baseline_fold_values(cell_root)
    _fake_execution(
        monkeypatch, fold_values=[base[fold] for fold in DISCOVERY_FOLDS],
    )
    run_baseline_reproduction(cell_root, repo_root=repo_root)

    from autobench.campaign_stages import _commit_state
    state = load_stage_state(cell_root)
    state["baseline_reproduction"]["candidate_sha256"] = "0" * 64
    _commit_state(cell_root, state)
    with pytest.raises(CampaignStageError, match="passing baseline reproduction"):
        _open_session(cell_root, adir)


def test_pass_at_another_head_does_not_short_circuit(staged_cell, monkeypatch):
    """The recovery the launch preflight prescribes must actually run: a
    recorded pass earned at a different commit refuses (pointing at --force)
    instead of returning a success that ran nothing."""
    cell_root, _, _, _, repo_root = staged_cell
    register_baseline(cell_root, _baseline(cell_root))
    _declare_policy(repo_root)
    base = _baseline_fold_values(cell_root)
    _fake_execution(
        monkeypatch, fold_values=[base[fold] for fold in DISCOVERY_FOLDS],
    )
    run_baseline_reproduction(cell_root, repo_root=repo_root)

    def moved_head(command, **kwargs):
        if command[:2] == ["git", "rev-parse"]:
            return SimpleNamespace(returncode=0, stdout="d" * 40 + "\n", stderr="")
        raise AssertionError(f"no execution expected, got {command[:3]}")

    monkeypatch.setattr("autobench.campaign_stages.subprocess.run", moved_head)
    with pytest.raises(CampaignStageError, match="--force"):
        run_baseline_reproduction(cell_root, repo_root=repo_root)

    _fake_execution(
        monkeypatch,
        fold_values=[base[fold] for fold in DISCOVERY_FOLDS],
        head="d" * 40,
    )
    state = run_baseline_reproduction(
        cell_root, repo_root=repo_root, force=True,
    )
    assert state["baseline_reproduction"]["verdict"] == "pass"
    assert state["baseline_reproduction"]["commit"] == "d" * 40
    # And now the same-head re-run is a true idempotent no-op again.
    assert run_baseline_reproduction(
        cell_root, repo_root=repo_root,
    )["baseline_reproduction"]["commit"] == "d" * 40
