#!/usr/bin/env python
"""Build or verify the frozen coding-agent protocol for the preprint campaign.

``build`` embeds the two committed source payloads (proposal policy text and
toolset description) verbatim, stamps the operator-pinned identity fields, and
writes the publication ``agent_protocol.json`` that ``campaign_manifest.py
materialize`` locks for all 130 cells.  ``verify`` re-validates an existing
protocol file and reports any drift against the committed sources.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from autobench.campaign import (
    AGENT_PROTOCOL_FILE,
    build_agent_protocol,
    content_sha256,
    validate_agent_protocol,
)

CAMPAIGN_DIR = "benchmarks/campaigns/preprint_130"


def _read_source(path: Path, label: str) -> str:
    try:
        text = path.read_text()
    except OSError as exc:
        raise SystemExit(f"cannot read {label} source {path}: {exc}")
    if not text.strip():
        raise SystemExit(f"{label} source {path} is empty")
    return text


def _serialize(protocol: dict) -> str:
    return json.dumps(protocol, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "verify"))
    parser.add_argument(
        "--policy", default=f"{CAMPAIGN_DIR}/proposal_policy.md",
        help="Proposal-policy source text, embedded verbatim.",
    )
    parser.add_argument(
        "--toolset", default=f"{CAMPAIGN_DIR}/toolset.json",
        help="Toolset description source, embedded verbatim.",
    )
    parser.add_argument(
        "--output", default=f"{CAMPAIGN_DIR}/agent_protocol.json",
        help="Protocol file to write (build) or check (verify).",
    )
    parser.add_argument(
        "--model", default="Claude Opus 5",
        help="Human-readable model name recorded in the protocol.",
    )
    parser.add_argument(
        "--model-version",
        help="Immutable model ID as reported by a real runtime session "
        "(e.g. from a throwaway session's /status). Required by build.",
    )
    parser.add_argument(
        "--runtime-version",
        help="Exact CLI version the launcher will enforce "
        "(claude --version). Required by build.",
    )
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]

    def _resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (repo_root / path).resolve()

    policy_path = _resolve(args.policy)
    toolset_path = _resolve(args.toolset)
    output_path = _resolve(args.output)

    if args.action == "build":
        if not args.model_version:
            parser.error("build requires --model-version (from a real session)")
        if not args.runtime_version:
            parser.error("build requires --runtime-version (claude --version)")
        if not any(char.isdigit() for char in args.model_version):
            parser.error(
                "--model-version does not look like an immutable model ID "
                f"({args.model_version!r}); aliases like 'opus' are not "
                "publication evidence"
            )
        toolset_text = _read_source(toolset_path, "toolset")
        try:
            toolset_data = json.loads(toolset_text)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"toolset source {toolset_path} is not JSON: {exc}")
        memory = toolset_data.get("ancestor_memory")
        if isinstance(memory, dict):
            refreshed = {}
            for relative in memory:
                target = repo_root / relative
                if not target.is_file():
                    raise SystemExit(f"ancestor memory file {target} is missing")
                refreshed[relative] = hashlib.sha256(
                    target.read_text().encode()
                ).hexdigest()
            if refreshed != memory:
                toolset_data["ancestor_memory"] = refreshed
                toolset_text = (
                    json.dumps(toolset_data, indent=2, ensure_ascii=False) + "\n"
                )
                toolset_path.write_text(toolset_text)
                print(f"refreshed ancestor_memory hashes in {toolset_path}")
        protocol = build_agent_protocol(
            proposal_policy=_read_source(policy_path, "proposal policy"),
            toolset=toolset_text,
            model=args.model,
            model_version=args.model_version,
            runtime_version=args.runtime_version,
        )
        serialized = _serialize(protocol)
        if output_path.exists() and output_path.read_text() != serialized:
            raise SystemExit(
                f"{output_path} already exists with different content; a "
                "campaign protocol is frozen once — delete it explicitly if "
                "it was never materialized"
            )
        output_path.write_text(serialized)
        print(f"wrote {output_path}")
        print(f"agent_protocol_sha256 {content_sha256(protocol)}")
        return

    try:
        protocol = validate_agent_protocol(json.loads(output_path.read_text()))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"invalid agent protocol {output_path}: {exc}")
    drift: list[str] = []
    for label, path, key in (
        ("proposal policy", policy_path, "proposal_policy_content"),
        ("toolset", toolset_path, "toolset_content"),
    ):
        if not path.exists():
            drift.append(f"{label} source {path} is missing")
        elif path.read_text() != protocol[key]:
            drift.append(f"{label} source {path} differs from embedded content")
    locked_path = output_path.parent / "runtime" / AGENT_PROTOCOL_FILE
    if locked_path.exists() and locked_path.read_text() != _serialize(protocol):
        drift.append(f"materialized lock {locked_path} differs from {output_path}")
    print(f"agent_protocol_sha256 {content_sha256(protocol)}")
    print(f"model {protocol['model']} ({protocol['model_version']})")
    print(f"runtime {protocol['runtime']} {protocol['runtime_version']}")
    if drift:
        for line in drift:
            print(f"drift: {line}", file=sys.stderr)
        raise SystemExit(1)
    print("sources match the embedded payloads")


if __name__ == "__main__":
    main()
