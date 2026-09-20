"""Builders for scrubbed Claude Code transcripts and small project trees.

Every fixture is built in code from a handful of line factories so a test can
say exactly which shapes it feeds the parser. The shapes mirror the real
runtime files (line types, block layouts, sidecar naming) with no real content.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SID = "0f0f0f0f-1111-4222-8333-444444444444"
CWD = "/data/project"


def _line(kind: str, ts: str, uuid: str, parent: str | None, **extra: Any) -> dict[str, Any]:
    return {
        "type": kind,
        "uuid": uuid,
        "parentUuid": parent,
        "timestamp": ts,
        "sessionId": SID,
        "cwd": CWD,
        "isSidechain": False,
        "version": "2.1.228",
        **extra,
    }


def assistant(
    uuid: str,
    parent: str | None,
    ts: str,
    blocks: list[dict[str, Any]],
    *,
    request_id: str,
    usage: dict[str, int] | None = None,
    model: str = "claude-opus-5",
) -> dict[str, Any]:
    return _line(
        "assistant",
        ts,
        uuid,
        parent,
        requestId=request_id,
        message={
            "id": f"msg_{request_id}",
            "role": "assistant",
            "model": model,
            "content": blocks,
            "usage": usage
            or {
                "input_tokens": 2,
                "output_tokens": 50,
                "cache_read_input_tokens": 1000,
                "cache_creation_input_tokens": 0,
            },
        },
    )


def tool_use(tool_use_id: str, name: str, **input_: Any) -> dict[str, Any]:
    return {"type": "tool_use", "id": tool_use_id, "name": name, "input": input_}


def text(value: str) -> dict[str, Any]:
    return {"type": "text", "text": value}


def thinking(value: str) -> dict[str, Any]:
    return {"type": "thinking", "thinking": value, "signature": "sig"}


def tool_result(
    uuid: str,
    parent: str,
    ts: str,
    tool_use_id: str,
    content: Any,
    *,
    is_error: bool = False,
    tool_use_result: Any = None,
    source_assistant: str | None = None,
) -> dict[str, Any]:
    block: dict[str, Any] = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
    if is_error:
        block["is_error"] = True
    extra: dict[str, Any] = {}
    if tool_use_result is not None:
        extra["toolUseResult"] = tool_use_result
    if source_assistant is not None:
        extra["sourceToolAssistantUUID"] = source_assistant
    return _line("user", ts, uuid, parent, message={"role": "user", "content": [block]}, **extra)


def bash_result(stdout: str, *, stderr: str = "", interrupted: bool = False) -> dict[str, Any]:
    return {
        "stdout": stdout,
        "stderr": stderr,
        "interrupted": interrupted,
        "isImage": False,
        "noOutputExpected": False,
    }


def human(uuid: str, parent: str | None, ts: str, prompt: str, **extra: Any) -> dict[str, Any]:
    return _line(
        "user",
        ts,
        uuid,
        parent,
        message={"role": "user", "content": prompt},
        origin={"kind": "human"},
        promptSource="typed",
        **extra,
    )


def notification(uuid: str, parent: str, ts: str, task_id: str, tool_use_id: str, summary: str, body: str) -> dict[str, Any]:
    content = (
        "<task-notification>\n"
        f"<task-id>{task_id}</task-id>\n"
        f"<tool-use-id>{tool_use_id}</tool-use-id>\n"
        "<status>completed</status>\n"
        f"<summary>{summary}</summary>\n"
        f"<result>{body}</result>\n"
        "</task-notification>"
    )
    return _line(
        "user",
        ts,
        uuid,
        parent,
        message={"role": "user", "content": content},
        origin={"kind": "task-notification"},
    )


def queue_operation(ts: str, content: str) -> dict[str, Any]:
    return {"type": "queue-operation", "operation": "enqueue", "timestamp": ts, "sessionId": SID, "content": content}


def system(uuid: str, parent: str | None, ts: str, subtype: str, **extra: Any) -> dict[str, Any]:
    return _line("system", ts, uuid, parent, subtype=subtype, **extra)


def write_lines(path: Path, lines: list[Any], *, partial_tail: str | None = None) -> Path:
    """Write dict lines as JSON and bytes lines verbatim; optionally a partial last line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        for item in lines:
            if isinstance(item, bytes):
                fh.write(item)
            else:
                fh.write(json.dumps(item, ensure_ascii=False).encode("utf-8"))
            fh.write(b"\n")
        if partial_tail is not None:
            fh.write(partial_tail.encode("utf-8"))
    return path


T0 = "2026-09-07T00:22:30.000Z"


def stamp(seconds: int) -> str:
    """A UTC stamp ``seconds`` after T0 (keeps fixtures readable)."""
    base = 22 * 60 + 30 + seconds
    return f"2026-09-07T00:{base // 60:02d}:{base % 60:02d}.000Z"


def mini_session_lines() -> list[Any]:
    """The standard scrubbed session: one propose, one submit, one subagent, edge cases."""
    big = "x" * 5000
    return [
        {"type": "mode", "mode": "normal", "sessionId": SID},
        {"type": "permission-mode", "permissionMode": "bypassPermissions", "sessionId": SID},
        human("u1", None, stamp(0), "Session is bound. Begin the discovery loop per your policy."),
        {"type": "attachment", "uuid": "att1", "parentUuid": "u1", "attachment": {"type": "deferred_tools_delta"}},
        # one response, two assistant lines sharing a requestId
        assistant("a1", "u1", stamp(5), [thinking("think first"), text("I will read the state.")], request_id="req1"),
        assistant("a2", "a1", stamp(6), [tool_use("t1", "Bash", command="uv run automil propose --parent node_0001 --desc 'lr down' --kind hp", description="propose")], request_id="req1"),
        tool_result("r1", "a2", stamp(8), "t1", "Added proposal node_0002 [hp]: lr down", tool_use_result=bash_result("Added proposal node_0002 [hp]: lr down"), source_assistant="a2"),
        assistant("a3", "r1", stamp(10), [tool_use("t2", "Bash", command="cd /data/project && uv run automil submit --node node_0002 --files train.py", description="submit")], request_id="req2"),
        tool_result("r2", "a3", stamp(12), "t2", "Submitted node_0002: 1 file(s) snapshotted", tool_use_result=bash_result("Submitted node_0002: 1 file(s) snapshotted")),
        # a subagent
        assistant("a4", "r2", stamp(20), [tool_use("t3", "Agent", prompt="research recipes", description="Research MIL recipes", run_in_background=True)], request_id="req3"),
        tool_result("r3", "a4", stamp(21), "t3", "Async agent launched successfully.", tool_use_result={"isAsync": True, "status": "async_launched", "agentId": "agent1", "description": "Research MIL recipes"}),
        queue_operation(stamp(40), "<task-notification><task-id>agent1</task-id></task-notification>"),
        notification("n1", "r3", stamp(41), "agent1", "t3", 'Agent "Research MIL recipes" finished', "# Report\nUse a lower lr."),
        # a list-content result, an error result, an image-bearing result
        assistant("a5", "n1", stamp(50), [tool_use("t4", "Read", file_path="/data/project/train.py")], request_id="req4"),
        tool_result("r4", "a5", stamp(51), "t4", [{"type": "text", "text": "line 1"}, {"type": "text", "text": "line 2"}]),
        assistant("a6", "r4", stamp(52), [tool_use("t5", "Bash", command="false", description="fail")], request_id="req5"),
        tool_result("r5", "a6", stamp(53), "t5", "Exit code 1", is_error=True, tool_use_result=bash_result("", stderr="boom")),
        assistant("a7", "r5", stamp(54), [tool_use("t6", "Read", file_path="/data/project/fig.png")], request_id="req6"),
        tool_result("r6", "a7", stamp(55), "t6", [{"type": "image", "source": {"type": "base64", "data": "AAAA"}}]),
        # a large result and a secret in it
        assistant("a8", "r6", stamp(60), [tool_use("t7", "Bash", command="cat big.log", description="big")], request_id="req7"),
        tool_result("r7", "a8", stamp(61), "t7", "HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz1234\n" + big + "\nTAIL-MARKER", tool_use_result=bash_result(big)),
        # an interrupted call with no result, followed by a compaction boundary
        assistant("a9", "r7", stamp(70), [tool_use("t8", "Bash", command="sleep 600", description="wait")], request_id="req8"),
        system("s1", "a9", stamp(71), "compact_boundary", compactMetadata={"preTokens": 150000, "postTokens": 20000}),
        {"type": "zzz-unknown", "uuid": "z1"},
        b"\xff\xfe not utf-8",
        b"this is not json",
        assistant("a10", "s1", stamp(80), [text("Done for now.")], request_id="req9"),
    ]


def write_mini_session(root: Path, *, partial_tail: str | None = None) -> Path:
    """Write the standard session at ``root/<SID>.jsonl`` with its sidecar dir."""
    path = write_lines(root / f"{SID}.jsonl", mini_session_lines(), partial_tail=partial_tail)
    sub = root / SID / "subagents"
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "agent-agent1.meta.json").write_text(json.dumps({
        "agentType": "general-purpose",
        "description": "Research MIL recipes",
        "toolUseId": "t3",
        "spawnDepth": 1,
    }))
    write_lines(sub / "agent-agent1.jsonl", [
        dict(human("su1", None, stamp(22), "research recipes"), isSidechain=True, agentId="agent1"),
        dict(assistant("sa1", "su1", stamp(30), [text("# Report\nUse a lower lr.")], request_id="sreq1"), isSidechain=True, agentId="agent1"),
    ])
    return path


# --- project trees ---------------------------------------------------------

BASELINE_FOLDS = [0.60, 0.57, 0.65]
KEPT_FOLDS = [0.66, 0.62, 0.70]
LOST_FOLDS = [0.61, 0.55, 0.66]


def _folds(values: list[float], bacc: float) -> list[dict[str, Any]]:
    return [
        {"fold_index": i, "metrics": {"val_auc": v, "val_bacc": bacc}, "primary_value": v}
        for i, v in enumerate(values)
    ]


def graph_payload(*, plant: dict[str, Any] | None = None) -> dict[str, Any]:
    """A five-node schema-3 graph: baseline, a discarded child, a kept child,
    a crash, and a pending proposal. ``plant`` merges extra keys into every
    node's metrics (the forged-violation tests use it)."""
    extra = plant or {}
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    nodes = {
        "node_0001": {
            "id": "node_0001", "parent_id": None, "type": "executed", "status": "keep",
            "description": "native baseline", "techniques": [], "primary_value": mean(BASELINE_FOLDS),
            "primary_se": 0.02, "global_delta": 0.0, "parent_delta": mean(BASELINE_FOLDS),
            "metrics": {"val_auc": mean(BASELINE_FOLDS), "val_bacc": 0.56, **extra},
            "vram_gb": 0.0, "elapsed_min": 0.0, "gpu": -1, "commit": None, "archive_id": "node_0001",
            "config_hash": "aaaa", "potential": 0.6, "child_count": 3, "created_at": "2026-09-06T17:00:00.000000",
            "bootstrapped": True, "cell_id": "c40d2b3a7a790598",
            "metadata": {"cell_id": "c40d2b3a7a790598", "mil_model": "abmil", "validation_folds": _folds(BASELINE_FOLDS, 0.56)},
        },
        "node_0002": {
            "id": "node_0002", "parent_id": "node_0001", "type": "executed", "status": "discard",
            "description": "lr down", "techniques": ["lr"], "tier": 2, "kind": "hp",
            "primary_value": mean(LOST_FOLDS), "primary_se": 0.03, "global_delta": -0.001, "parent_delta": -0.001,
            "metrics": {"val_auc": mean(LOST_FOLDS), "val_bacc": 0.55, **extra},
            "fold_primary_values": [{"fold_index": i, "primary_value": v} for i, v in enumerate(LOST_FOLDS)],
            "vram_gb": 0.9, "elapsed_min": 95.5, "gpu": 0, "commit": None, "archive_id": "node_0002",
            "config_hash": "bbbb", "potential": 0.61, "child_count": 0, "created_at": "2026-09-06T17:30:28.759919",
            "cell_id": "c40d2b3a7a790598", "metadata": {"candidate_class": "config-only"},
        },
        "node_0003": {
            "id": "node_0003", "parent_id": "node_0001", "type": "executed", "status": "keep",
            "description": "dropout 0.5", "techniques": ["dropout"], "tier": 2, "kind": "regularization",
            "primary_value": mean(KEPT_FOLDS), "primary_se": 0.03, "global_delta": 0.053, "parent_delta": 0.053,
            "metrics": {"val_auc": mean(KEPT_FOLDS), "val_bacc": 0.60, **extra},
            "fold_primary_values": [{"fold_index": i, "primary_value": v} for i, v in enumerate(KEPT_FOLDS)],
            "vram_gb": 0.9, "elapsed_min": 96.0, "gpu": 1, "commit": None, "archive_id": "node_0003",
            "config_hash": "cccc", "potential": 0.66, "child_count": 0, "created_at": "2026-09-06T17:30:29.899969",
            "cell_id": "c40d2b3a7a790598", "metadata": {"candidate_class": "train-only-source"},
        },
        "node_0004": {
            "id": "node_0004", "parent_id": "node_0001", "type": "executed", "status": "crash",
            "description": "bad policy", "techniques": [], "tier": 2, "kind": "hp",
            "primary_value": 0.0, "primary_se": None, "global_delta": -0.6, "parent_delta": -0.6,
            "metrics": {}, "vram_gb": 0.0, "elapsed_min": 0.8, "gpu": 1, "commit": None, "archive_id": "node_0004",
            "config_hash": "dddd", "potential": 0.0, "child_count": 0, "created_at": "2026-09-06T17:30:30.000000",
            "cell_id": "c40d2b3a7a790598", "metadata": {}, "error": "Traceback: test_auc=0.91 leaked\nValueError: boom",
        },
        "node_0005": {
            "id": "node_0005", "parent_id": "node_0003", "type": "proposed", "status": "pending",
            "description": "dropout 0.75", "techniques": ["dropout"], "tier": 2, "kind": "regularization",
            "rationale": "", "reference": None, "expected_gain": "low", "effort": "low", "potential": 0.65,
            "created_at": "2026-09-06T19:00:00.000000", "cell_id": "c40d2b3a7a790598", "metadata": {},
        },
    }
    return {
        "schema_version": 3,
        "meta": {
            "best_primary_value": mean(KEPT_FOLDS), "best_node_id": "node_0003", "total_executed": 4,
            "total_proposed": 1, "next_id": 6, "baseline_primary_value": mean(BASELINE_FOLDS),
            "scoring": {
                "exploration_weight": 0.005, "novelty_weight": 0.003, "accept_margin": 0.015,
                "se_multiplier": 1.0, "formula": "val_auc",
                "guard": {"metric": "val_bacc", "margin": 0.0099},
            },
        },
        "nodes": nodes,
        "technique_stats": {"dropout": {"times_tried": 2, "best_parent_delta": 0.053, "avg_parent_delta": 0.026}},
    }


def write_project(root: Path, *, plant: dict[str, Any] | None = None, running: tuple[str, ...] = ()) -> Path:
    """Write a project tree at ``root`` and return its ``automil/`` dir."""
    automil = root / "automil"
    orch = automil / "orchestrator"
    for sub in ("queue", "completed", "archive", "running"):
        (orch / sub).mkdir(parents=True, exist_ok=True)
    (automil / "cells").mkdir(exist_ok=True)
    (automil / "config.yaml").write_text(
        "project:\n  name: demo_project\n  description: demo\ntask:\n  name: kras\n"
        "encoders:\n  primary: hoptimus1\nrun:\n  mil_model: abmil\nscoring:\n  formula: val_auc\n"
    )
    graph = graph_payload(plant=plant)
    (automil / "graph.json").write_text(json.dumps(graph))
    (automil / "plan.md").write_text("# Plan\n\nTry dropout.\n")
    (automil / "learnings.md").write_text("# Learnings\n\n- lr down does nothing.\n")
    (orch / "gpu_state.json").write_text(json.dumps({
        "counter": 1, "last_updated": "2026-09-06T19:10:00.000000",
        "gpus": {"0": {"running": list(running)}},
        "execution_slots": {"cuda:0": {"accelerator": "cuda", "device_index": 0, "running": list(running), "capacity": 4}},
    }))
    (orch / "orchestrator.log").write_text(
        "2026-09-06 17:36:40,814 [INFO] Launched node_0002 on CUDA GPU 0 (PID 1, est. 0.5GB, timeout 600min)\n"
        "2026-09-06 17:36:41,000 [INFO] Launched node_0003 on CUDA GPU 1 (PID 2, est. 0.5GB, timeout 600min)\n"
        "2026-09-06 17:36:42,000 [INFO] Launched node_0004 on CUDA GPU 1 (PID 3, est. 0.5GB, timeout 600min)\n"
        "2026-09-06 17:37:30,000 [INFO] Completed node_0004: status=crash, primary_value=0.0, elapsed=0.8min, CUDA GPU 1\n"
        "2026-09-06 19:12:58,109 [INFO] Completed node_0002: status=completed, primary_value=0.6067, elapsed=96.2min, CUDA GPU 0\n"
        "2026-09-06 19:13:00,000 [INFO] Completed node_0003: status=completed, primary_value=0.66, elapsed=96.0min, CUDA GPU 1\n"
    )
    extra = plant or {}
    for node_id, node in graph["nodes"].items():
        if node["type"] != "executed" or node.get("bootstrapped"):
            continue
        archive = orch / "archive" / node_id
        archive.mkdir(parents=True, exist_ok=True)
        folds = {"node_0002": LOST_FOLDS, "node_0003": KEPT_FOLDS}.get(node_id, [])
        result = {
            "status": "completed" if node["status"] != "crash" else "crash",
            "metrics": dict(node["metrics"]),
            "primary_value": node["primary_value"], "primary_se": node["primary_se"],
            "elapsed_seconds": node["elapsed_min"] * 60, "peak_vram_mb": int(node["vram_gb"] * 1000),
            "validation_folds": _folds(folds, 0.55) if folds else [],
            "held_out": {"test_auc": 0.93, "test_bacc": 0.88},
            "summary": {"test_auc": 0.93},
        }
        if extra:
            for fold in result["validation_folds"]:
                fold["metrics"].update(extra)
        # the archive root result.json is the sealed-off copy: no held_out block
        (archive / "result.json").write_text(json.dumps({k: v for k, v in result.items() if k not in ("held_out", "summary")}))
        certify = archive / "certify"
        certify.mkdir(exist_ok=True)
        (certify / "result.json").write_text(json.dumps(result))
        (certify / "certify.json").write_text(json.dumps({"held_out": result["held_out"], "summary": result["summary"]}))
        (archive / "spec.json").write_text(json.dumps({
            "id": node_id, "description": node["description"], "base_commit": "1660b0d8f267d58cd2c68644ddb4847ab70c81b3",
            "overlay_dir": f"archive/{node_id}",
            "overlay_manifest": {"automil/variants/_policies/dropout.py": "sha256:" + "ab" * 32} if node_id == "node_0003" else {},
            "deletions": [], "framework_overlay_files": [],
            "submitted_at": "2026-09-07T00:3%d:23.214441+00:00" % (int(node_id[-1]) % 10),
            "metadata": {"backend": "local", "cell_id": "c40d2b3a7a790598"},
        }))
        if node_id == "node_0003":
            policy = archive / "automil" / "variants" / "_policies"
            policy.mkdir(parents=True)
            (policy / "dropout.py").write_text("DROPOUT = 0.5\n")
        (archive / "run.log").write_text(
            "epoch 1 val_auc=0.61\n" + ("test_auc=0.93 leaked line\n" if extra else "") +
            "[selected] epoch=2\nExperiment complete\n"
        )
        (orch / "completed" / f"{node_id}.json").write_text(json.dumps({
            "id": node_id, "status": result["status"], "primary_value": node["primary_value"],
            "primary_se": node["primary_se"], "fold_primary_values": node.get("fold_primary_values"),
            "metrics": {**node["metrics"], **({"test_auc": 0.93} if extra else {})},
            "elapsed_seconds": node["elapsed_min"] * 60, "peak_vram_mb": int(node["vram_gb"] * 1000),
            "accelerator": "cuda", "gpu": node["gpu"],
            "completed_at": {"node_0002": "2026-09-06T19:12:51.003299", "node_0003": "2026-09-06T19:12:59.000000", "node_0004": "2026-09-06T17:37:29.000000"}[node_id],
            "budget_killed": False,
            "graph_metadata": {"parent_id": node["parent_id"], "techniques": node["techniques"], "config_hash": node["config_hash"]},
        }))
    (automil / ".activity.jsonl").write_text(
        '{"cell_id":null,"event":"session_open","observed_at":1788740513.9171042,"session_id":"%s"}\n' % SID
    )
    return automil
