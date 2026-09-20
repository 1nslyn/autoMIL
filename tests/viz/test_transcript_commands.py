"""viz.transcript_commands: automil calls inside shell commands."""
from __future__ import annotations

from automil.viz.transcript_commands import AutomilCall, created_node_ids, detect_automil, summarize_calls


def test_detects_the_launch_forms_the_agent_uses():
    command = (
        'cd "$PWD" && export REPO_ROOT=/data/repo && '
        "uv run --project \"$REPO_ROOT\" automil propose --parent node_0001 --desc 'lr down: five times' --kind hp; "
        "automil --project /data/cell submit --node node_0002 --files train.py | tee out.txt && "
        "python -m automil rank -n 5"
    )
    calls = detect_automil(command)
    assert [c.sub for c in calls] == ["propose", "submit", "rank"]
    assert calls[0] == AutomilCall(
        sub="propose",
        argv=("propose", "--parent", "node_0001", "--desc", "lr down: five times", "--kind", "hp"),
        node_ids=("node_0001",),
    )
    assert calls[1].argv == ("submit", "--node", "node_0002", "--files", "train.py")
    assert calls[1].node_ids == ("node_0002",)
    assert calls[2].node_ids == ()


def test_paths_and_plain_words_are_not_commands():
    assert detect_automil("cat automil/plan.md && ls automil") == ()
    assert detect_automil("grep -n automil_dir src/x.py") == ()


def test_unbalanced_quotes_fall_back_to_whitespace_splitting():
    (call,) = detect_automil("automil submit --node node_0003 --desc 'unterminated")
    assert call.argv[:3] == ("submit", "--node", "node_0003")
    assert call.node_ids == ("node_0003",)


def test_created_node_ids_match_the_exact_cli_output_for_every_call():
    output = (
        "Added proposal node_0002 [hp]: lr down\n"
        "Added proposal node_0003 [hp]: lr up\n"
        "warning: x\nSubmitted node_0007: 1 file(s) snapshotted\n"
        "1. [node_0004] leaderboard line\n"
    )
    assert created_node_ids(["propose", "submit", "rank"], output) == (
        ("node_0002", "propose"), ("node_0003", "propose"), ("node_0007", "submit"),
    )
    assert created_node_ids(["resubmit"], "resubmitting\nnode_0012\n") == (("node_0012", "resubmit"),)
    assert created_node_ids(["rank"], "1. [node_0004] ...") == ()
    assert created_node_ids(["propose"], "Error: parent node_0001 is not kept") == ()


def test_summarize_calls_unions_argv_and_created_ids():
    calls = detect_automil("automil propose --parent node_0001 --desc a; automil propose --parent node_0001 --desc b")
    summary = summarize_calls(calls, "Added proposal node_0002 [hp]: a\nAdded proposal node_0003 [hp]: b\n")
    assert [c["sub"] for c in summary["calls"]] == ["propose", "propose"]
    assert summary["node_ids"] == ["node_0001", "node_0002", "node_0003"]
    assert summary["created"] == [
        {"node_id": "node_0002", "sub": "propose"},
        {"node_id": "node_0003", "sub": "propose"},
    ]
    assert summarize_calls(calls)["created"] == []
