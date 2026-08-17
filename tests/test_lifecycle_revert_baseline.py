"""Coverage for `automil revert-baseline` (CLI-02 / D-42)."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner


def _init_git_repo(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=path, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=path, capture_output=True)
    (path / "README.md").write_text("# Test\n")
    subprocess.run(["git", "add", "."], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, capture_output=True, check=True)


def _setup_with_protected(tmp_path: Path, protected: list[str]) -> tuple[Path, str]:
    """Init project, add a `src/lib.py`, commit; return (automil_dir, base_commit_sha).

    The automil/ directory is committed to the repo so `git stash
    --include-untracked` does NOT sweep it up during revert-baseline tests.
    """
    _init_git_repo(tmp_path)
    from automil.cli import main
    import os
    os.chdir(tmp_path)
    CliRunner().invoke(main, ["init"])
    adir = tmp_path / "automil"

    # Patch config.yaml with the desired protected list.
    cfg = yaml.safe_load((adir / "config.yaml").read_text()) or {}
    cfg.setdefault("registry", {})["protected"] = protected
    (adir / "config.yaml").write_text(yaml.safe_dump(cfg))

    # Commit the automil/ directory so --include-untracked stashes don't sweep it up.
    subprocess.run(["git", "add", "automil/"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "add automil overlay"], cwd=tmp_path,
                   check=True, capture_output=True)

    # Add src/lib.py and commit (this is the base commit revert-baseline will target).
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "lib.py").write_text("# v1\n")
    subprocess.run(["git", "add", "src/lib.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "add lib"], cwd=tmp_path, check=True, capture_output=True)
    base_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True, check=True,
    ).stdout.strip()

    # Write a graph.json with one executed node referencing the base commit.
    # Write directly (the file is already tracked after committing automil/).
    graph = {
        "schema_version": 1,
        "meta": {"best_node_id": "node_0001", "best_primary_value": 0.5,
                 "total_executed": 1, "total_proposed": 0,
                 "next_id": 2, "baseline_primary_value": 0.0,
                 "scoring": {"exploration_weight": 0.005, "novelty_weight": 0.003}},
        "nodes": {
            "node_0001": {
                "id": "node_0001", "type": "executed", "status": "keep",
                "primary_value": 0.5, "base_commit": base_sha,
                "created_at": "2026-05-02T10:00:00Z",
            }
        },
        "technique_stats": {},
    }
    (adir / "graph.json").write_text(json.dumps(graph, indent=2))
    return adir, base_sha


@pytest.fixture
def cli_runner():
    return CliRunner()


# --- happy path ---

def test_protected_file_reverted(tmp_path, cli_runner, monkeypatch):
    adir, base = _setup_with_protected(tmp_path, ["src/lib.py"])
    monkeypatch.chdir(tmp_path)
    # Modify the protected file.
    (tmp_path / "src" / "lib.py").write_text("# v2\n")

    from automil.cli import main
    result = cli_runner.invoke(main, ["revert-baseline"])
    assert result.exit_code == 0, result.output
    # File reverted.
    assert (tmp_path / "src" / "lib.py").read_text() == "# v1\n"


# --- safety: stash ---

def test_mandatory_stash_created(tmp_path, cli_runner, monkeypatch):
    adir, base = _setup_with_protected(tmp_path, ["src/lib.py"])
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src" / "lib.py").write_text("# v2\n")

    from automil.cli import main
    result = cli_runner.invoke(main, ["revert-baseline"])
    assert result.exit_code == 0
    # Stash listed.
    stash_list = subprocess.run(
        ["git", "stash", "list"], cwd=tmp_path,
        capture_output=True, text=True, check=True,
    ).stdout
    assert "automil-revert-" in stash_list


def test_stash_name_format(tmp_path, cli_runner, monkeypatch):
    adir, base = _setup_with_protected(tmp_path, ["src/lib.py"])
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src" / "lib.py").write_text("# v2\n")

    from automil.cli import main
    result = cli_runner.invoke(main, ["revert-baseline"])
    # Stdout includes a name matching automil-revert-YYYYMMDD-HHMMSS.
    assert re.search(r"automil-revert-\d{8}-\d{6}", result.output), result.output


def test_recovery_message_in_output(tmp_path, cli_runner, monkeypatch):
    adir, base = _setup_with_protected(tmp_path, ["src/lib.py"])
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src" / "lib.py").write_text("# v2\n")

    from automil.cli import main
    result = cli_runner.invoke(main, ["revert-baseline"])
    # Operator-friendly: tells them how to recover.
    assert "git stash pop" in result.output or "git stash list" in result.output


def test_uncommitted_non_protected_also_stashed(tmp_path, cli_runner, monkeypatch):
    """D-42 anti-protected: stash captures EVERYTHING uncommitted; checkout
    only touches protected paths.

    Strengthened per WARNING-03: also verifies the stash *contents* include
    the non-protected file via `git stash show --name-only`. Without this
    check, a regression to a path-limited stash (`git stash push <pathspec>`)
    would pass the working-tree-clean assertion but silently lose the
    non-protected work. The stash MUST be full-tree, not pathspec-limited.
    """
    adir, base = _setup_with_protected(tmp_path, ["src/lib.py"])
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src" / "lib.py").write_text("# v2\n")
    # Dirty a non-protected file (must be tracked for git stash show to include it).
    (tmp_path / "src" / "main.py").write_text("# editable code\n")
    subprocess.run(["git", "add", "src/main.py"], cwd=tmp_path, check=True)

    from automil.cli import main
    result = cli_runner.invoke(main, ["revert-baseline"])
    assert result.exit_code == 0, result.output
    # Protected file: reverted.
    assert (tmp_path / "src" / "lib.py").read_text() == "# v1\n"
    # Non-protected file: NOT in working tree (it's in the stash).
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=tmp_path,
        capture_output=True, text=True, check=True,
    ).stdout
    assert "src/main.py" not in status

    # WARNING-03 strengthening: assert the stash CONTENTS include the
    # non-protected file (proves the stash is full-tree, not pathspec-limited).
    stash_files = subprocess.run(
        ["git", "stash", "show", "--name-only"], cwd=tmp_path,
        capture_output=True, text=True, check=True,
    ).stdout
    assert "src/main.py" in stash_files, (
        "Stash is path-limited (regression on D-42 + Leo's never-blind-checkout "
        "memory). The stash must capture all uncommitted work, not just protected "
        "paths. See WARNING-03 in the plan-checker report."
    )
    assert "src/lib.py" in stash_files


def test_untracked_file_included_in_stash(tmp_path, cli_runner, monkeypatch):
    adir, base = _setup_with_protected(tmp_path, ["src/lib.py"])
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src" / "lib.py").write_text("# v2\n")
    # Untracked file.
    (tmp_path / "junk.tmp").write_text("debug noise\n")

    from automil.cli import main
    cli_runner.invoke(main, ["revert-baseline"])
    # Untracked file should be stashed (--include-untracked) -> gone from tree.
    assert not (tmp_path / "junk.tmp").exists()


# --- idempotence ---

def test_clean_tree_no_op(tmp_path, cli_runner, monkeypatch):
    adir, base = _setup_with_protected(tmp_path, ["src/lib.py"])
    monkeypatch.chdir(tmp_path)
    # File untouched. No stash should be created.
    from automil.cli import main
    result = cli_runner.invoke(main, ["revert-baseline"])
    assert result.exit_code == 0
    assert "nothing to do" in result.output.lower() or "already clean" in result.output.lower()
    # No stash created.
    stash_list = subprocess.run(
        ["git", "stash", "list"], cwd=tmp_path,
        capture_output=True, text=True, check=True,
    ).stdout
    assert "automil-revert-" not in stash_list


def test_idempotent_second_run(tmp_path, cli_runner, monkeypatch):
    """Running revert-baseline twice should be safe: second run is a clean no-op
    without creating a second stash."""
    adir, base = _setup_with_protected(tmp_path, ["src/lib.py"])
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src" / "lib.py").write_text("# v2\n")

    from automil.cli import main
    # First run: stash + revert.
    r1 = cli_runner.invoke(main, ["revert-baseline"])
    assert r1.exit_code == 0, r1.output

    # Second run: tree already clean for protected paths -> no-op.
    r2 = cli_runner.invoke(main, ["revert-baseline"])
    assert r2.exit_code == 0, r2.output
    assert "nothing to do" in r2.output.lower() or "already clean" in r2.output.lower()

    # No SECOND automil-revert stash was created (only the one from the first run).
    stash_list = subprocess.run(
        ["git", "stash", "list"], cwd=tmp_path,
        capture_output=True, text=True, check=True,
    ).stdout
    # Count automil-revert entries: should be exactly 1.
    count = stash_list.count("automil-revert-")
    assert count == 1, f"Expected 1 stash but got {count}. stash list:\n{stash_list}"


# --- error paths ---

def test_no_graph_json_hard_fail(tmp_path, cli_runner, monkeypatch):
    adir, base = _setup_with_protected(tmp_path, ["src/lib.py"])
    monkeypatch.chdir(tmp_path)
    # Remove graph.json.
    (adir / "graph.json").unlink()
    (tmp_path / "src" / "lib.py").write_text("# v2\n")

    from automil.cli import main
    result = cli_runner.invoke(main, ["revert-baseline"])
    assert result.exit_code != 0
    assert "graph.json" in result.output or "no executed nodes" in result.output.lower() or "executed" in result.output.lower()


def test_no_executed_nodes_hard_fail(tmp_path, cli_runner, monkeypatch):
    adir, base = _setup_with_protected(tmp_path, ["src/lib.py"])
    monkeypatch.chdir(tmp_path)
    # Replace graph.json with one having only a proposed node.
    graph = json.loads((adir / "graph.json").read_text())
    graph["nodes"] = {"p1": {"id": "p1", "type": "proposed", "status": "pending"}}
    (adir / "graph.json").write_text(json.dumps(graph, indent=2))

    (tmp_path / "src" / "lib.py").write_text("# v2\n")
    from automil.cli import main
    result = cli_runner.invoke(main, ["revert-baseline"])
    assert result.exit_code != 0
    assert "executed" in result.output.lower() or "base_commit" in result.output


def test_empty_protected_hard_fail(tmp_path, cli_runner, monkeypatch):
    adir, base = _setup_with_protected(tmp_path, [])
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src" / "lib.py").write_text("# v2\n")

    from automil.cli import main
    result = cli_runner.invoke(main, ["revert-baseline"])
    assert result.exit_code != 0
    assert "registry.protected is empty" in result.output or "empty" in result.output.lower()


def test_invalid_base_commit_hard_fail(tmp_path, cli_runner, monkeypatch):
    adir, base = _setup_with_protected(tmp_path, ["src/lib.py"])
    monkeypatch.chdir(tmp_path)
    # Patch graph.json with a fake SHA.
    graph = json.loads((adir / "graph.json").read_text())
    graph["nodes"]["node_0001"]["base_commit"] = "deadbeef" * 5
    (adir / "graph.json").write_text(json.dumps(graph, indent=2))

    (tmp_path / "src" / "lib.py").write_text("# v2\n")
    from automil.cli import main
    result = cli_runner.invoke(main, ["revert-baseline"])
    assert result.exit_code != 0
    assert "not a valid git SHA" in result.output or "rev-parse" in result.output.lower()


# --- help text ---

def test_help_mentions_stash(cli_runner):
    from automil.cli import main
    result = cli_runner.invoke(main, ["revert-baseline", "--help"])
    assert "stash" in result.output.lower() or "blind" in result.output.lower()


# --- multiple executed nodes ---

def test_picks_most_recent_executed_node(tmp_path, cli_runner, monkeypatch):
    adir, base = _setup_with_protected(tmp_path, ["src/lib.py"])
    monkeypatch.chdir(tmp_path)
    # Add a SECOND executed node with later created_at + a different base_commit.
    # Make a new commit (lib.py v1.5).
    (tmp_path / "src" / "lib.py").write_text("# v1.5\n")
    subprocess.run(["git", "add", "src/lib.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "v1.5"], cwd=tmp_path, check=True, capture_output=True)
    new_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True, check=True,
    ).stdout.strip()

    graph = json.loads((adir / "graph.json").read_text())
    graph["nodes"]["node_0002"] = {
        "id": "node_0002", "type": "executed", "status": "keep",
        "base_commit": new_sha, "created_at": "2026-05-03T10:00:00Z",
    }
    (adir / "graph.json").write_text(json.dumps(graph, indent=2))

    # Now modify lib.py and revert; should reset to new_sha (v1.5), not base (v1).
    (tmp_path / "src" / "lib.py").write_text("# v9\n")
    from automil.cli import main
    result = cli_runner.invoke(main, ["revert-baseline"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "src" / "lib.py").read_text() == "# v1.5\n"


def test_stash_name_printed_before_checkout(tmp_path, cli_runner, monkeypatch):
    """The stash name must appear in output BEFORE the checkout-confirmation line,
    so even if git checkout crashes, the operator sees the stash name."""
    adir, base = _setup_with_protected(tmp_path, ["src/lib.py"])
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src" / "lib.py").write_text("# v2\n")

    from automil.cli import main
    result = cli_runner.invoke(main, ["revert-baseline"])
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    # Find the line that mentions the stash name and the line that confirms revert.
    stash_line_idx = next(
        (i for i, line in enumerate(lines) if re.search(r"automil-revert-\d{8}-\d{6}", line)),
        None,
    )
    revert_line_idx = next(
        (i for i, line in enumerate(lines) if "reverted" in line.lower() or "protected paths" in line.lower()),
        None,
    )
    assert stash_line_idx is not None, "Stash name not in output"
    assert revert_line_idx is not None, "Revert confirmation not in output"
    assert stash_line_idx < revert_line_idx, (
        "Stash name must appear BEFORE the revert confirmation "
        "(operator needs the name in case checkout crashes)."
    )


# --- untracked protected pattern (pre-protected path for an incoming branch) ---

def test_pattern_unknown_to_git_is_skipped_not_fatal(tmp_path, cli_runner, monkeypatch):
    """A protected path that exists on no commit and in no index must not
    fail the whole revert: `git checkout` is all-or-nothing over its
    pathspecs, so one unmatched pattern used to error out AND leave every
    other protected path dirty. Such a pattern has no baseline state to
    revert to and is skipped with a notice."""
    adir, base = _setup_with_protected(
        tmp_path, ["src/lib.py", "src/not_born_yet.py"],
    )
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src" / "lib.py").write_text("# v2\n")

    from automil.cli import main
    result = cli_runner.invoke(main, ["revert-baseline"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "src" / "lib.py").read_text() == "# v1\n"
    assert "Skipped" in result.output
    assert "src/not_born_yet.py" in result.output


def test_untracked_protected_file_created_by_agent_is_cleared(
    tmp_path, cli_runner, monkeypatch,
):
    """An agent-created file at a pre-protected path is cleared by the
    mandatory --include-untracked stash; the revert still succeeds."""
    adir, base = _setup_with_protected(
        tmp_path, ["src/lib.py", "src/not_born_yet.py"],
    )
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src" / "not_born_yet.py").write_text("# rogue\n")

    from automil.cli import main
    result = cli_runner.invoke(main, ["revert-baseline"])
    assert result.exit_code == 0, result.output
    assert not (tmp_path / "src" / "not_born_yet.py").exists()


def test_pattern_tracked_at_head_but_absent_at_base_is_skipped(
    tmp_path, cli_runner, monkeypatch,
):
    """A protected file committed AFTER base_commit is in the live index but
    not in base_commit's tree. `git checkout <base> -- <path>` matches
    against the TREE, so such a pathspec aborts the whole all-or-nothing
    checkout — the exact state every pre-existing template hits when a new
    protected file lands in the same PR that protects it. The probe must
    therefore query base_commit's tree, not the index (`ls-files
    --with-tree` lists the union and reproduces the abort)."""
    adir, base = _setup_with_protected(tmp_path, ["src/lib.py", "src/newer.py"])
    monkeypatch.chdir(tmp_path)
    # Commit a protected file AFTER the base commit: tracked at HEAD,
    # absent from base's tree.
    (tmp_path / "src" / "newer.py").write_text("# born after base\n")
    subprocess.run(["git", "add", "src/newer.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "newer"], cwd=tmp_path,
                   check=True, capture_output=True)
    # Dirty both protected files.
    (tmp_path / "src" / "lib.py").write_text("# v2\n")
    (tmp_path / "src" / "newer.py").write_text("# dirty\n")

    from automil.cli import main
    result = cli_runner.invoke(main, ["revert-baseline"])
    assert result.exit_code == 0, result.output
    # The revertable pattern actually reverted (the old probe let newer.py
    # into the checkout, which aborted and left lib.py dirty).
    assert (tmp_path / "src" / "lib.py").read_text() == "# v1\n"
    assert "Skipped" in result.output
    assert "src/newer.py" in result.output
    # newer.py has no base state; its dirt went into the stash and the file
    # reverted to its committed (HEAD) content.
    assert (tmp_path / "src" / "newer.py").read_text() == "# born after base\n"


def test_glob_pattern_matching_base_tree_is_reverted(
    tmp_path, cli_runner, monkeypatch,
):
    """Glob patterns (`dir/**`) must survive the probe. `git ls-tree`'s path
    arguments are NOT pathspecs — probing with it silently skips every glob
    pattern and turns the revert into a no-op. The probe must run git's own
    pathspec engine against base_commit's tree."""
    adir, base = _setup_with_protected(tmp_path, ["src/**"])
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src" / "lib.py").write_text("# v2\n")

    from automil.cli import main
    result = cli_runner.invoke(main, ["revert-baseline"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "src" / "lib.py").read_text() == "# v1\n"
    assert "Skipped" not in result.output


def test_all_patterns_skipped_does_not_claim_success(
    tmp_path, cli_runner, monkeypatch,
):
    """When every protected pattern lacks base state, nothing is checked out
    — the output must say so instead of printing the 'Reverted protected
    paths' success line over an empty pattern list."""
    adir, base = _setup_with_protected(tmp_path, ["src/not_born_yet.py"])
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src" / "not_born_yet.py").write_text("# rogue\n")

    from automil.cli import main
    result = cli_runner.invoke(main, ["revert-baseline"])
    assert result.exit_code == 0, result.output
    assert "Reverted protected paths" not in result.output
    assert "nothing checked out" in result.output
    assert "Skipped" in result.output
