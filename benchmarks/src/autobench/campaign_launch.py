"""Per-cell launcher for the formal discovery coding-agent session.

The locked ``runtime/agent_protocol.json`` is the single source of truth for
what a formal session looks like: the launcher renders the agent instruction
surface byte-exact from ``proposal_policy_content``, derives the runtime
invocation from ``toolset_content``, and refuses to start when anything the
protocol declares (CLI version, memory surface, environment, exclusivity)
does not hold on this host.  It never composes instructions or flags of its
own, so a launched session can only ever see the archived policy.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from automil.activity_hooks import (
    claude_activity_settings,
    project_exporter_port,
)

from autobench.campaign import (
    AGENT_PROTOCOL_FILE,
    TRAINING_TREE_PATHS,
    content_sha256,
    validate_agent_protocol,
)

AGENT_INSTRUCTION_FILE = "CLAUDE.md"
TOOLSET_SCHEMA_VERSION = 1
_TOOLSET_KEYS = {
    "schema_version", "permission_mode", "claude_flags", "launcher_env",
    "forbidden_env", "ancestor_memory", "user_memory_absent",
    "user_plugins_absent", "mcp_servers", "plugins",
    "resume_previous_session", "cross_cell_memory", "network_access",
    "session_settings", "tools",
}


class CampaignLaunchError(RuntimeError):
    """A formal session launch precondition does not hold on this host."""


@dataclass(frozen=True)
class LaunchPlan:
    """Everything needed to exec the formal session, fully derived."""

    argv: tuple[str, ...]
    env: dict[str, str]
    cwd: Path
    agent_protocol_sha256: str
    instruction_path: Path
    instruction_content: str


def load_locked_protocol(runtime_root: Path) -> tuple[dict[str, Any], str]:
    """Read and re-validate the campaign-frozen agent protocol."""
    path = runtime_root / AGENT_PROTOCOL_FILE
    try:
        protocol = validate_agent_protocol(json.loads(path.read_text()))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise CampaignLaunchError(
            f"cannot verify locked agent protocol at {path}: {exc}"
        ) from exc
    return protocol, content_sha256(protocol)


def parse_toolset(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Parse and validate the machine-readable locked tool surface."""
    try:
        toolset = json.loads(protocol["toolset_content"])
    except json.JSONDecodeError as exc:
        raise CampaignLaunchError(
            f"toolset_content is not machine-readable JSON: {exc}"
        ) from exc
    if not isinstance(toolset, dict) or set(toolset) != _TOOLSET_KEYS:
        raise CampaignLaunchError("toolset field set is not exact")
    if toolset["schema_version"] != TOOLSET_SCHEMA_VERSION:
        raise CampaignLaunchError("toolset schema_version is not supported")
    if (
        not isinstance(toolset["claude_flags"], list)
        or not all(isinstance(flag, str) for flag in toolset["claude_flags"])
    ):
        raise CampaignLaunchError("toolset claude_flags must be strings")
    if (
        not isinstance(toolset["launcher_env"], dict)
        or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in toolset["launcher_env"].items()
        )
    ):
        raise CampaignLaunchError("toolset launcher_env must map strings")
    if not isinstance(toolset["forbidden_env"], list):
        raise CampaignLaunchError("toolset forbidden_env must be a list")
    if not isinstance(toolset["ancestor_memory"], dict):
        raise CampaignLaunchError("toolset ancestor_memory must be a mapping")
    invariants = (
        ("user_memory_absent", True),
        ("user_plugins_absent", True),
        ("resume_previous_session", False),
        ("cross_cell_memory", False),
        ("network_access", True),
        ("mcp_servers", []),
        ("plugins", []),
    )
    for key, expected in invariants:
        if toolset[key] != expected:
            raise CampaignLaunchError(
                f"toolset {key} must be {expected!r} for this campaign"
            )
    if toolset["permission_mode"] != "bypassPermissions":
        raise CampaignLaunchError(
            "toolset permission_mode must be bypassPermissions"
        )
    if "--dangerously-skip-permissions" not in toolset["claude_flags"]:
        raise CampaignLaunchError(
            "claude_flags do not realize the declared permission_mode"
        )
    forbidden = ("--resume", "--continue", "-c", "-r", "--fallback-model")
    for flag in toolset["claude_flags"]:
        if flag in forbidden or flag.startswith(tuple(f"{name}=" for name in forbidden)):
            raise CampaignLaunchError(
                "claude_flags contain a resume or fallback flag the "
                "protocol forbids"
            )
    return toolset


def _claude_cli_version(claude_bin: str) -> str:
    try:
        output = subprocess.run(
            [claude_bin, "--version"], capture_output=True, text=True,
            timeout=30, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise CampaignLaunchError(
            f"cannot run {claude_bin} --version: {exc}"
        ) from exc
    if not output:
        raise CampaignLaunchError(f"{claude_bin} --version printed nothing")
    return output.split()[0]


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(1.0)
        return probe.connect_ex(("127.0.0.1", port)) == 0


# Public re-exports for operator tooling (campaign_operate.py): the runtime
# version probe and the exporter-port probe must have exactly one
# implementation and one failure mode across the launcher and the operator
# CLI. The underscore names stay the in-module call/monkeypatch surface.
claude_cli_version = _claude_cli_version
port_in_use = _port_in_use


def _sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_text().encode()).hexdigest()


def _check_memory_surface(
    cell_root: Path, repo_root: Path, toolset: Mapping[str, Any], home: Path,
) -> None:
    pinned = {
        (repo_root / relative).resolve(): sha
        for relative, sha in toolset["ancestor_memory"].items()
    }
    for path, sha in pinned.items():
        if not path.is_file():
            raise CampaignLaunchError(
                f"pinned instruction file {path} is missing"
            )
        observed = _sha256_text(path)
        if observed != sha:
            raise CampaignLaunchError(
                f"instruction surface drift: {path} hashes {observed}, the "
                f"frozen protocol pinned {sha} — the repository instruction "
                "files are frozen for the campaign"
            )
    # The runtime reads memory files from the cwd upward to the filesystem
    # root, not stopping at the repository, so the whole path must be clean:
    # nothing unpinned, and no local/scoped variants anywhere on it.
    directory = cell_root
    while True:
        if directory != cell_root:
            candidate = directory / AGENT_INSTRUCTION_FILE
            if candidate.is_file() and candidate.resolve() not in pinned:
                raise CampaignLaunchError(
                    f"unpinned {AGENT_INSTRUCTION_FILE} on the memory path: "
                    f"{candidate}"
                )
        for variant in (
            directory / "CLAUDE.local.md",
            directory / ".claude" / AGENT_INSTRUCTION_FILE,
        ):
            if variant.is_file():
                raise CampaignLaunchError(
                    f"unpinned memory variant on the memory path: {variant}"
                )
        if directory == directory.parent:
            break
        directory = directory.parent
    if toolset["user_memory_absent"]:
        user_memory = home / ".claude" / AGENT_INSTRUCTION_FILE
        if user_memory.exists():
            raise CampaignLaunchError(
                f"user-level memory {user_memory} exists; the formal session "
                "instruction surface must be exactly the frozen protocol"
            )
    if toolset["user_plugins_absent"]:
        plugins = home / ".claude" / "plugins"
        if plugins.exists() and any(plugins.iterdir()):
            raise CampaignLaunchError(
                f"user-level plugins under {plugins} would load into the "
                "formal session; remove them on this host"
            )


def _check_prior_session_evidence(cell_root: Path) -> None:
    bound = cell_root / "agent_session.json"
    if bound.exists():
        raise CampaignLaunchError(
            f"{bound} exists: this cell already opened its one formal "
            "session; a cell never gets a second one"
        )
    journal = cell_root / "automil" / ".activity.jsonl"
    if journal.exists() and journal.read_text().strip():
        raise CampaignLaunchError(
            f"{journal} already records a session for this cell root; if the "
            "runtime died pre-bind, re-materialize the cell (runbook §4e) "
            "instead of relaunching"
        )


def _check_orchestrator_running(cell_root: Path) -> None:
    pid_file = cell_root / "automil" / "orchestrator" / "orchestrator.pid"
    try:
        pid = json.loads(pid_file.read_text())["pid"]
        os.kill(int(pid), 0)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CampaignLaunchError(
            "cell orchestrator is not running (it must scrape the exporter "
            f"before the session binds): {exc}"
        ) from exc


def _is_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(char in "0123456789abcdef" for char in value)
    )


def _check_execution_identity(cell_root: Path, repo_root: Path) -> None:
    """Launch-time code-identity gate.

    The launch HEAD must match one of the two evidence anchors: the commit
    recorded when the baseline trained (execution identity), or the commit a
    PASSING reproduction verdict ran at — re-running the gate is how a new
    HEAD earns the right to launch, since its result equality ties the new
    commit's behavior to the baseline. Requiring the identity commit alone
    would deadlock every legitimate HEAD move (committing
    reproduction_policy.json itself moves HEAD) with no recovery path.
    HEAD and tree cleanliness are re-derived from git at every launch; a
    commit string alone never wins against a moved HEAD.
    """
    try:
        state = json.loads((cell_root / "campaign_state.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignLaunchError(f"cannot read campaign state: {exc}") from exc
    baseline = state.get("baseline") if isinstance(state, dict) else None
    if not isinstance(baseline, dict):
        raise CampaignLaunchError(
            "launch requires a registered native baseline"
        )
    anchors: list[tuple[str, str]] = []
    identity = baseline.get("execution_identity")
    if isinstance(identity, dict) and _is_commit(identity.get("commit")):
        anchors.append(
            ("the baseline's execution identity", identity["commit"])
        )
    reproduction = state.get("baseline_reproduction")
    if (
        isinstance(reproduction, dict)
        and reproduction.get("mode") == "gate"
        and reproduction.get("verdict") == "pass"
        and _is_commit(reproduction.get("commit"))
    ):
        anchors.append(
            ("the passing reproduction verdict", reproduction["commit"])
        )
    if not anchors:
        raise CampaignLaunchError(
            "launch requires a code-identity anchor: no execution identity "
            "was recorded with this baseline and no passing reproduction "
            "verdict exists — run `campaign_stage.py "
            "run-baseline-reproduction` first"
        )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root, check=False, capture_output=True, text=True,
    )
    observed = (head.stdout or "").strip()
    if head.returncode != 0 or len(observed) != 40:
        raise CampaignLaunchError(
            f"cannot resolve the launch HEAD commit: {(head.stderr or '').strip()}"
        )
    if observed not in {commit for _, commit in anchors}:
        described = "; ".join(
            f"{name} commit {commit[:12]}" for name, commit in anchors
        )
        raise CampaignLaunchError(
            f"code identity drift: launch HEAD {observed[:12]} matches no "
            f"anchor ({described}) — re-run the reproduction gate at this "
            "HEAD to authorize it"
        )
    dirty = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *TRAINING_TREE_PATHS],
        cwd=repo_root, check=False, capture_output=True, text=True,
    )
    if dirty.returncode == 1:
        raise CampaignLaunchError(
            "code identity drift: tracked modifications under "
            + ", ".join(TRAINING_TREE_PATHS)
        )
    if dirty.returncode != 0:
        raise CampaignLaunchError(
            "cannot verify training-tree cleanliness at launch: "
            f"{(dirty.stderr or '').strip()}"
        )


def preflight(
    cell_root: Path,
    repo_root: Path,
    *,
    claude_bin: str = "claude",
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
    probe_port: bool = True,
    require_orchestrator: bool = True,
) -> LaunchPlan:
    """Verify every launch precondition and derive the exec plan."""
    cell_root = cell_root.resolve()
    repo_root = repo_root.resolve()
    environ = os.environ if environ is None else environ
    home = Path.home() if home is None else home
    try:
        cell_root.relative_to(repo_root)
    except ValueError as exc:
        raise CampaignLaunchError(
            f"cell root {cell_root} is not inside the repository {repo_root}"
        ) from exc
    adir = cell_root / "automil"
    if not (adir / "config.yaml").is_file():
        raise CampaignLaunchError(f"{cell_root} is not a materialized cell root")
    if not (adir / "campaign_cell.json").is_file():
        raise CampaignLaunchError(f"{cell_root} has no campaign cell record")
    protocol, protocol_sha = load_locked_protocol(cell_root.parent)
    toolset = parse_toolset(protocol)

    try:
        exporter_port = project_exporter_port(adir)
    except ValueError as exc:
        raise CampaignLaunchError(str(exc)) from exc
    settings_path = cell_root / ".claude" / "settings.json"
    try:
        settings = json.loads(settings_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignLaunchError(
            f"cannot read {settings_path}: {exc}"
        ) from exc
    if settings != claude_activity_settings(exporter_port):
        raise CampaignLaunchError(
            f"{settings_path} drifted from the activity observer contract "
            f"(declared exporter port {exporter_port}); re-materialize the "
            "cell instead of editing settings"
        )

    observed_version = _claude_cli_version(claude_bin)
    if observed_version != protocol["runtime_version"]:
        raise CampaignLaunchError(
            f"runtime version drift: {claude_bin} --version reports "
            f"{observed_version}, the frozen protocol requires "
            f"{protocol['runtime_version']}"
        )

    _check_execution_identity(cell_root, repo_root)

    present = [name for name in toolset["forbidden_env"] if name in environ]
    if present:
        raise CampaignLaunchError(
            "environment overrides the frozen runtime identity: unset "
            + ", ".join(sorted(present))
        )

    _check_memory_surface(cell_root, repo_root, toolset, home)
    _check_prior_session_evidence(cell_root)
    if require_orchestrator:
        _check_orchestrator_running(cell_root)
    if probe_port and _port_in_use(exporter_port):
        raise CampaignLaunchError(
            f"this cell's activity exporter port {exporter_port} is already "
            "serving: another session is exporting on it — every concurrent "
            "cell on a host must declare a distinct activity.exporter_port"
        )

    instruction_path = cell_root / AGENT_INSTRUCTION_FILE
    content = protocol["proposal_policy_content"]
    if instruction_path.exists() and instruction_path.read_text() != content:
        raise CampaignLaunchError(
            f"{instruction_path} differs from the locked "
            "proposal_policy_content; refusing to launch over an edited "
            "instruction file"
        )
    argv = (
        claude_bin,
        "--model", protocol["model_version"],
        *toolset["claude_flags"],
    )
    return LaunchPlan(
        argv=argv,
        env={**toolset["launcher_env"], "REPO_ROOT": str(repo_root)},
        cwd=cell_root,
        agent_protocol_sha256=protocol_sha,
        instruction_path=instruction_path,
        instruction_content=content,
    )


def render_instruction(plan: LaunchPlan) -> Path:
    """Write the locked instruction surface into the cell, byte-exact."""
    if not plan.instruction_path.exists():
        plan.instruction_path.write_text(plan.instruction_content)
    return plan.instruction_path


def launch(plan: LaunchPlan) -> None:
    """Replace this process with the formal session (never returns)."""
    render_instruction(plan)
    env = {**os.environ, **plan.env}
    os.chdir(plan.cwd)
    os.execvpe(plan.argv[0], list(plan.argv), env)
