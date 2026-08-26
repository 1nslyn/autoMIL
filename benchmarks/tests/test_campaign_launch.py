"""Contracts for the frozen agent protocol builder and the per-cell launcher.

The launcher is the executor of the locked protocol: everything it enforces
(instruction surface, tool surface, runtime identity, exclusivity) must be
derived from ``runtime/agent_protocol.json`` and nothing else.  These tests
pin that derivation and every fail-closed refusal.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

from automil.activity_hooks import claude_activity_settings

from autobench.campaign import (
    AGENT_PROTOCOL_FILE,
    CampaignManifestError,
    build_agent_protocol,
    content_sha256,
)
from autobench.campaign_launch import (
    CampaignLaunchError,
    parse_toolset,
    preflight,
    render_instruction,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_DIR = REPO_ROOT / "benchmarks/campaigns/preprint_130"
POLICY_SOURCE = CAMPAIGN_DIR / "proposal_policy.md"
TOOLSET_SOURCE = CAMPAIGN_DIR / "toolset.json"
RUNTIME_VERSION = "2.1.220"
MODEL_VERSION = "claude-test-5-20260801"


def _load_script(filename: str, name: str):
    path = REPO_ROOT / "benchmarks/scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_claude(path: Path, version: str = RUNTIME_VERSION) -> str:
    path.write_text(f"#!/bin/sh\necho '{version} (Claude Code)'\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def _toolset_for(repo_root: Path) -> str:
    data = json.loads(TOOLSET_SOURCE.read_text())
    data["ancestor_memory"] = {
        "CLAUDE.md": hashlib.sha256(
            (repo_root / "CLAUDE.md").read_text().encode()
        ).hexdigest()
    }
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


@pytest.fixture
def launch_host(tmp_path):
    """A fake repository with one materialized-enough cell, ready to launch."""
    repo_root = tmp_path / "repo"
    runtime_root = repo_root / "runtime"
    cell_root = runtime_root / "dataset__task__enc__arm__s42__preprint-v3"
    adir = cell_root / "automil"
    adir.mkdir(parents=True)
    (repo_root / "CLAUDE.md").write_text("# repo dev instructions\n")
    (adir / "config.yaml").write_text("project:\n  name: dataset\n")
    (adir / "campaign_cell.json").write_text("{}")
    settings_dir = cell_root / ".claude"
    settings_dir.mkdir()
    (settings_dir / "settings.json").write_text(
        json.dumps(claude_activity_settings(), indent=2, sort_keys=True) + "\n"
    )
    orch_dir = adir / "orchestrator"
    orch_dir.mkdir()
    (orch_dir / "orchestrator.pid").write_text(
        json.dumps({"pid": os.getpid(), "starttime_ticks": 0}) + "\n"
    )
    protocol = build_agent_protocol(
        proposal_policy=POLICY_SOURCE.read_text(),
        toolset=_toolset_for(repo_root),
        model="Claude Opus 5",
        model_version=MODEL_VERSION,
        runtime_version=RUNTIME_VERSION,
    )
    (runtime_root / AGENT_PROTOCOL_FILE).write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    )
    home = tmp_path / "home"
    home.mkdir()
    claude_bin = _fake_claude(tmp_path / "claude")
    # A real git repo with one commit: the identity gate re-derives HEAD and
    # tree cleanliness from git at every launch. Everything the fixture
    # writes stays untracked, which doubles as the standing no-false-positive
    # check (untracked files are not dirt).
    subprocess.run(["git", "init", "-q", str(repo_root)], check=True)
    subprocess.run(
        ["git", "-C", str(repo_root), "-c", "user.email=t@test",
         "-c", "user.name=t", "commit", "--allow-empty", "-q", "-m", "init"],
        check=True,
    )
    head = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    (cell_root / "campaign_state.json").write_text(json.dumps({
        "baseline": {"execution_identity": {"commit": head, "clean": True}},
    }))
    return {
        "repo_root": repo_root,
        "cell_root": cell_root,
        "protocol": protocol,
        "home": home,
        "claude_bin": claude_bin,
        "head": head,
    }


def _preflight(host, **overrides):
    kwargs = {
        "claude_bin": host["claude_bin"],
        "environ": {},
        "home": host["home"],
        "probe_port": False,
        "require_orchestrator": True,
    }
    kwargs.update(overrides)
    return preflight(host["cell_root"], host["repo_root"], **kwargs)


# --- the committed sources are a coherent publication protocol -----------


def test_committed_sources_build_a_publication_protocol():
    protocol = build_agent_protocol(
        proposal_policy=POLICY_SOURCE.read_text(),
        toolset=TOOLSET_SOURCE.read_text(),
        model="Claude Opus 5",
        model_version=MODEL_VERSION,
        runtime_version=RUNTIME_VERSION,
    )
    assert protocol["proposal_policy_content"] == POLICY_SOURCE.read_text()
    assert protocol["toolset_content"] == TOOLSET_SOURCE.read_text()
    toolset = parse_toolset(protocol)
    assert "--dangerously-skip-permissions" in toolset["claude_flags"]
    assert toolset["ancestor_memory"].keys() == {"CLAUDE.md"}


def test_committed_policy_carries_the_load_bearing_rules():
    text = " ".join(POLICY_SOURCE.read_text().split())
    for anchor in (
        "read-only until bound",
        "30",
        "automil certify",
        "--include-held-out",
        "variants/_policies",
        "git-root-relative",
        # v6: the editable-path authority is config.yaml files.editable read
        # VERBATIM — a literal runtime/ prefix broke every non-runtime root.
        "`files.editable` — read the value and use it VERBATIM",
        "<cell-id>/automil/variants/_policies",
        "wrap_scheduler",
        "only on the DTFD arm",
        "never end the session while unspent attempts remain",
        "regularization",
    ):
        assert anchor in text, f"proposal policy lost its {anchor!r} rule"


def test_builder_rejects_a_placeholder_model_version():
    with pytest.raises(CampaignManifestError):
        build_agent_protocol(
            proposal_policy=POLICY_SOURCE.read_text(),
            toolset=TOOLSET_SOURCE.read_text(),
            model="Claude Opus 5",
            model_version="unknown",
            runtime_version=RUNTIME_VERSION,
        )


# --- preflight derives everything from the locked protocol ---------------


def test_preflight_derives_the_plan_from_the_locked_protocol(launch_host):
    plan = _preflight(launch_host)
    protocol = launch_host["protocol"]
    toolset = parse_toolset(protocol)
    assert plan.argv == (
        launch_host["claude_bin"], "--model", MODEL_VERSION,
        *toolset["claude_flags"],
    )
    assert plan.env["DISABLE_AUTOUPDATER"] == "1"
    assert plan.env["REPO_ROOT"] == str(launch_host["repo_root"])
    assert plan.cwd == launch_host["cell_root"]
    assert plan.agent_protocol_sha256 == content_sha256(protocol)
    assert plan.instruction_content == protocol["proposal_policy_content"]


def test_render_instruction_is_byte_exact_and_idempotent(launch_host):
    plan = _preflight(launch_host)
    path = render_instruction(plan)
    assert path.read_text() == launch_host["protocol"]["proposal_policy_content"]
    render_instruction(plan)
    assert _preflight(launch_host).instruction_path == path
    path.write_text(path.read_text() + "\nEDITED")
    with pytest.raises(CampaignLaunchError, match="differs from the locked"):
        _preflight(launch_host)


def test_preflight_rejects_runtime_version_drift(launch_host, tmp_path):
    drifted = _fake_claude(tmp_path / "claude-drift", version="2.1.999")
    with pytest.raises(CampaignLaunchError, match="runtime version drift"):
        _preflight(launch_host, claude_bin=drifted)


def test_preflight_rejects_settings_drift(launch_host):
    settings = launch_host["cell_root"] / ".claude" / "settings.json"
    payload = json.loads(settings.read_text())
    payload["permissions"] = {"allow": ["Bash"]}
    settings.write_text(json.dumps(payload))
    with pytest.raises(CampaignLaunchError, match="activity observer contract"):
        _preflight(launch_host)


def test_preflight_rejects_forbidden_environment(launch_host):
    with pytest.raises(CampaignLaunchError, match="ANTHROPIC_MODEL"):
        _preflight(launch_host, environ={"ANTHROPIC_MODEL": "something-else"})


def test_preflight_rejects_pinned_memory_drift(launch_host):
    (launch_host["repo_root"] / "CLAUDE.md").write_text("# edited mid-campaign\n")
    with pytest.raises(CampaignLaunchError, match="instruction surface drift"):
        _preflight(launch_host)


def test_preflight_rejects_unpinned_memory_on_the_path(launch_host):
    (launch_host["cell_root"].parent / "CLAUDE.md").write_text("stray\n")
    with pytest.raises(CampaignLaunchError, match="unpinned CLAUDE.md"):
        _preflight(launch_host)


def test_preflight_rejects_memory_above_the_repository(launch_host):
    above = launch_host["repo_root"].parent / "CLAUDE.md"
    above.write_text("memory above the repo still loads\n")
    try:
        with pytest.raises(CampaignLaunchError, match="unpinned CLAUDE.md"):
            _preflight(launch_host)
    finally:
        above.unlink()


def test_preflight_rejects_memory_variants_on_the_path(launch_host):
    local = launch_host["repo_root"] / "CLAUDE.local.md"
    local.write_text("local memory\n")
    with pytest.raises(CampaignLaunchError, match="memory variant"):
        _preflight(launch_host)
    local.unlink()
    scoped = launch_host["cell_root"] / ".claude" / "CLAUDE.md"
    scoped.write_text("scoped memory\n")
    with pytest.raises(CampaignLaunchError, match="memory variant"):
        _preflight(launch_host)


def test_preflight_requires_the_locked_protocol(launch_host):
    from autobench.campaign import AGENT_PROTOCOL_FILE as protocol_file

    (launch_host["cell_root"].parent / protocol_file).unlink()
    with pytest.raises(CampaignLaunchError, match="cannot verify locked"):
        _preflight(launch_host)


def test_preflight_rejects_user_memory_and_plugins(launch_host):
    user_claude = launch_host["home"] / ".claude"
    user_claude.mkdir()
    (user_claude / "CLAUDE.md").write_text("user memory\n")
    with pytest.raises(CampaignLaunchError, match="user-level memory"):
        _preflight(launch_host)
    (user_claude / "CLAUDE.md").unlink()
    plugins = user_claude / "plugins"
    plugins.mkdir()
    (plugins / "something").mkdir()
    with pytest.raises(CampaignLaunchError, match="user-level plugins"):
        _preflight(launch_host)


def test_preflight_rejects_prior_session_evidence(launch_host):
    bound = launch_host["cell_root"] / "agent_session.json"
    bound.write_text("{}")
    with pytest.raises(CampaignLaunchError, match="already opened its one"):
        _preflight(launch_host)
    bound.unlink()
    journal = launch_host["cell_root"] / "automil" / ".activity.jsonl"
    journal.write_text('{"event": "session_open"}\n')
    with pytest.raises(CampaignLaunchError, match="already records a session"):
        _preflight(launch_host)


def test_preflight_requires_a_running_orchestrator(launch_host):
    pid_file = (
        launch_host["cell_root"] / "automil" / "orchestrator" / "orchestrator.pid"
    )
    pid_file.write_text(json.dumps({"pid": "not-a-pid"}))
    with pytest.raises(CampaignLaunchError, match="orchestrator is not running"):
        _preflight(launch_host)
    pid_file.unlink()
    with pytest.raises(CampaignLaunchError, match="orchestrator is not running"):
        _preflight(launch_host)
    assert _preflight(launch_host, require_orchestrator=False)


def test_preflight_rejects_a_busy_exporter_port(launch_host, monkeypatch):
    probed: list[int] = []

    def fake_in_use(port):
        probed.append(port)
        return True

    monkeypatch.setattr("autobench.campaign_launch._port_in_use", fake_in_use)
    with pytest.raises(CampaignLaunchError, match="already serving"):
        _preflight(launch_host, probe_port=True)
    assert probed == [9464]


def test_preflight_resolves_the_cell_declared_exporter_port(
    launch_host, monkeypatch,
):
    adir = launch_host["cell_root"] / "automil"
    (adir / "config.yaml").write_text(
        "project:\n  name: dataset\nactivity:\n  exporter_port: 9581\n"
    )
    with pytest.raises(CampaignLaunchError, match="declared exporter port 9581"):
        _preflight(launch_host)
    settings = launch_host["cell_root"] / ".claude" / "settings.json"
    settings.write_text(
        json.dumps(claude_activity_settings(9581), indent=2, sort_keys=True)
        + "\n"
    )
    probed: list[int] = []

    def fake_in_use(port):
        probed.append(port)
        return False

    monkeypatch.setattr("autobench.campaign_launch._port_in_use", fake_in_use)
    assert _preflight(launch_host, probe_port=True)
    assert probed == [9581]


def test_preflight_rejects_an_invalid_activity_declaration(launch_host):
    adir = launch_host["cell_root"] / "automil"
    (adir / "config.yaml").write_text(
        "project:\n  name: dataset\nactivity:\n  exporter_port: eighty\n"
    )
    with pytest.raises(CampaignLaunchError, match="exporter_port must be"):
        _preflight(launch_host)


def test_preflight_rejects_a_cell_outside_the_repository(launch_host, tmp_path):
    with pytest.raises(CampaignLaunchError, match="not inside the repository"):
        preflight(
            tmp_path / "elsewhere", launch_host["repo_root"],
            claude_bin=launch_host["claude_bin"], environ={},
            home=launch_host["home"], probe_port=False,
        )


# --- toolset invariants are enforced, not advisory -----------------------


def _protocol_with_toolset(host, mutate):
    data = json.loads(_toolset_for(host["repo_root"]))
    mutate(data)
    return build_agent_protocol(
        proposal_policy=POLICY_SOURCE.read_text(),
        toolset=json.dumps(data) + "\n",
        model="Claude Opus 5",
        model_version=MODEL_VERSION,
        runtime_version=RUNTIME_VERSION,
    )


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.__setitem__("resume_previous_session", True), "resume"),
        (lambda d: d.__setitem__("network_access", False), "network_access"),
        (lambda d: d.__setitem__("mcp_servers", ["github"]), "mcp_servers"),
        (lambda d: d.__setitem__("permission_mode", "default"), "permission_mode"),
        (lambda d: d.pop("tools"), "not exact"),
        (
            lambda d: d.__setitem__(
                "claude_flags", ["--dangerously-skip-permissions", "--resume"],
            ),
            "resume or fallback",
        ),
        (
            lambda d: d.__setitem__(
                "claude_flags",
                ["--dangerously-skip-permissions", "--fallback-model=sonnet"],
            ),
            "resume or fallback",
        ),
        (lambda d: d.__setitem__("surprise", True), "not exact"),
        (
            lambda d: d.__setitem__("claude_flags", ["--effort", "high"]),
            "permission_mode",
        ),
    ],
)
def test_parse_toolset_rejects_drifted_surfaces(launch_host, mutate, message):
    protocol = _protocol_with_toolset(launch_host, mutate)
    with pytest.raises(CampaignLaunchError, match=message):
        parse_toolset(protocol)


# --- the builder script --------------------------------------------------


def test_builder_script_builds_verifies_and_freezes(tmp_path, capsys):
    script = _load_script("campaign_agent_protocol.py", "protocol_script_test")
    policy = tmp_path / "policy.md"
    policy.write_text(POLICY_SOURCE.read_text())
    toolset = tmp_path / "toolset.json"
    toolset.write_text(_toolset_for(REPO_ROOT))
    output = tmp_path / "agent_protocol.json"
    argv_common = [
        "--policy", str(policy), "--toolset", str(toolset),
        "--output", str(output),
    ]
    script.main([
        "build", *argv_common,
        "--model-version", MODEL_VERSION, "--runtime-version", RUNTIME_VERSION,
    ])
    protocol = json.loads(output.read_text())
    assert protocol["proposal_policy_content"] == policy.read_text()
    out = capsys.readouterr().out
    assert f"agent_protocol_sha256 {content_sha256(protocol)}" in out

    # Idempotent rebuild is fine; a different build is refused.
    script.main([
        "build", *argv_common,
        "--model-version", MODEL_VERSION, "--runtime-version", RUNTIME_VERSION,
    ])
    with pytest.raises(SystemExit, match="frozen once"):
        script.main([
            "build", *argv_common,
            "--model-version", "claude-test-5-20260802",
            "--runtime-version", RUNTIME_VERSION,
        ])

    script.main(["verify", *argv_common])
    assert "sources match" in capsys.readouterr().out
    policy.write_text(policy.read_text() + "\ndrifted")
    with pytest.raises(SystemExit):
        script.main(["verify", *argv_common])
    assert "differs from embedded" in capsys.readouterr().err


def test_builder_script_rejects_alias_model_versions(tmp_path):
    script = _load_script("campaign_agent_protocol.py", "protocol_script_alias")
    with pytest.raises(SystemExit):
        script.main([
            "build",
            "--policy", str(POLICY_SOURCE),
            "--toolset", str(TOOLSET_SOURCE),
            "--output", str(tmp_path / "agent_protocol.json"),
            "--model-version", "opus",
            "--runtime-version", RUNTIME_VERSION,
        ])
    with pytest.raises(SystemExit):
        script.main([
            "build",
            "--policy", str(POLICY_SOURCE),
            "--toolset", str(TOOLSET_SOURCE),
            "--output", str(tmp_path / "agent_protocol.json"),
            "--model-version", MODEL_VERSION,
            "--runtime-version", "2.1.220 (Claude Code)",
        ])


def test_builder_script_refreshes_stale_ancestor_hashes(tmp_path):
    script = _load_script("campaign_agent_protocol.py", "protocol_script_refresh")
    policy = tmp_path / "policy.md"
    policy.write_text("frozen instructions\n")
    toolset = tmp_path / "toolset.json"
    data = json.loads(TOOLSET_SOURCE.read_text())
    data["ancestor_memory"] = {"CLAUDE.md": "0" * 64}
    toolset.write_text(json.dumps(data, indent=2) + "\n")
    output = tmp_path / "agent_protocol.json"
    script.main([
        "build",
        "--policy", str(policy), "--toolset", str(toolset),
        "--output", str(output),
        "--model-version", MODEL_VERSION, "--runtime-version", RUNTIME_VERSION,
    ])
    expected = hashlib.sha256((REPO_ROOT / "CLAUDE.md").read_text().encode())
    refreshed = json.loads(toolset.read_text())["ancestor_memory"]["CLAUDE.md"]
    assert refreshed == expected.hexdigest()
    embedded = json.loads(output.read_text())["toolset_content"]
    assert embedded == toolset.read_text()


def test_launch_script_refuses_a_cell_outside_the_repo(tmp_path):
    script = _load_script("campaign_launch.py", "launch_script_test")
    outside = tmp_path / "not-a-cell"
    outside.mkdir()
    with pytest.raises(SystemExit, match="campaign-launch refusal"):
        script.main(["preflight", "--cell-root", str(outside)])


# --- launch-time code-identity gate --------------------------------------


def _commit_all(repo_root: Path, message: str) -> str:
    subprocess.run(
        ["git", "-C", str(repo_root), "add", "-A"], check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo_root), "-c", "user.email=t@test",
         "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", message],
        check=True,
    )
    return subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def test_preflight_refuses_a_moved_head(launch_host):
    _commit_all(launch_host["repo_root"], "drift")
    with pytest.raises(CampaignLaunchError, match="code identity drift"):
        _preflight(launch_host)


def test_preflight_refuses_a_forged_identity_commit(launch_host):
    """Forged violation: editing the recorded commit cannot defeat the gate.

    The check re-derives HEAD from git, so pointing the ledger at another
    commit only passes when HEAD actually IS that commit — at which point
    the launch runs that code, which is exactly the contract.
    """
    state_path = launch_host["cell_root"] / "campaign_state.json"
    state = json.loads(state_path.read_text())
    state["baseline"]["execution_identity"]["commit"] = "a" * 40
    state_path.write_text(json.dumps(state))
    with pytest.raises(CampaignLaunchError, match="code identity drift"):
        _preflight(launch_host)


def test_preflight_refuses_a_dirty_training_tree(launch_host):
    repo_root = launch_host["repo_root"]
    tracked = repo_root / "src" / "module.py"
    tracked.parent.mkdir(parents=True)
    tracked.write_text("VALUE = 1\n")
    head = _commit_all(repo_root, "add training file")
    state_path = launch_host["cell_root"] / "campaign_state.json"
    state = json.loads(state_path.read_text())
    state["baseline"]["execution_identity"]["commit"] = head
    state_path.write_text(json.dumps(state))
    plan = _preflight(launch_host)
    assert plan is not None

    tracked.write_text("VALUE = 2\n")
    with pytest.raises(CampaignLaunchError, match="tracked modifications"):
        _preflight(launch_host)


def test_preflight_untracked_files_are_not_dirt(launch_host):
    (launch_host["repo_root"] / "src").mkdir(exist_ok=True)
    (launch_host["repo_root"] / "src" / "scratch.log").write_text("x\n")
    assert _preflight(launch_host) is not None


def test_preflight_legacy_baseline_anchors_through_reproduction_verdict(
    launch_host,
):
    state_path = launch_host["cell_root"] / "campaign_state.json"
    state_path.write_text(json.dumps({"baseline": {}}))
    with pytest.raises(CampaignLaunchError, match="code-identity anchor"):
        _preflight(launch_host)

    state_path.write_text(json.dumps({
        "baseline": {},
        "baseline_reproduction": {
            "mode": "gate", "verdict": "pass",
            "commit": launch_host["head"],
        },
    }))
    assert _preflight(launch_host) is not None

    state_path.write_text(json.dumps({
        "baseline": {},
        "baseline_reproduction": {
            "mode": "measurement", "verdict": "measured",
            "commit": launch_host["head"],
        },
    }))
    with pytest.raises(CampaignLaunchError, match="code-identity anchor"):
        _preflight(launch_host)


def test_preflight_moved_head_is_rescued_by_a_passing_verdict_at_head(
    launch_host,
):
    """A new HEAD is authorized by re-running the reproduction gate at it —
    identity-bearing baselines must not deadlock on legitimate HEAD moves
    (committing reproduction_policy.json itself moves HEAD)."""
    new_head = _commit_all(launch_host["repo_root"], "policy commit")
    state_path = launch_host["cell_root"] / "campaign_state.json"
    state = json.loads(state_path.read_text())
    assert state["baseline"]["execution_identity"]["commit"] != new_head
    state["baseline_reproduction"] = {
        "mode": "gate", "verdict": "pass", "commit": new_head,
    }
    state_path.write_text(json.dumps(state))
    assert _preflight(launch_host) is not None

    state["baseline_reproduction"]["commit"] = "b" * 40
    state_path.write_text(json.dumps(state))
    with pytest.raises(CampaignLaunchError, match="matches no anchor"):
        _preflight(launch_host)
