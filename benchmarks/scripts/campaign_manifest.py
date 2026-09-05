#!/usr/bin/env python
"""Generate, verify, or materialize the frozen preprint-130 campaign."""
from __future__ import annotations

import argparse
from pathlib import Path

from automil.backends.pidfile import is_pid_alive_with_starttime, load_pid_file
from autobench.campaign import (
    build_preprint_manifest,
    file_sha256,
    load_manifest,
    materialize_discovery_cells,
    run_materialization_canary,
    write_manifest,
)
from autobench.campaign_stages import certify_campaign, freeze_campaign_selections
from autobench.campaign_analysis import write_publication_report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=(
            "generate", "check", "materialize", "canary",
            "freeze-selections", "certify-all", "report",
        ),
    )
    parser.add_argument(
        "--manifest", default="benchmarks/campaigns/preprint_130/manifest.json",
    )
    parser.add_argument(
        "--output-root", default="benchmarks/campaigns/preprint_130/runtime",
    )
    parser.add_argument(
        "--agent-protocol",
        help="Publication-ready agent protocol JSON; required by materialize.",
    )
    parser.add_argument(
        "--cells",
        help="materialize only these active-roster cell ids (comma-separated, "
             "or @<roster.json> carrying a cell_ids list) — a rehearsal set "
             "built beside the campaign runtime; row indices and ports are unchanged",
    )
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = (repo_root / args.manifest).resolve()
    output_root = (repo_root / args.output_root).resolve()
    try:
        output_root.relative_to(repo_root)
    except ValueError:
        parser.error("--output-root must live inside the git repository")
    if args.action == "generate":
        # Live-daemon guard: daemons anchored at this checkout re-hash
        # manifest.json at every launch revalidation against the sha pinned in
        # their queued specs. Rewriting it under them refuses their own billed
        # retries mid-flight (observed live 2026-08-15 on the canary recovery,
        # 43 seconds after an in-place rewrite). Drain and stop daemons first.
        # Scanned roots: the campaign dir (covers the default runtime/ and
        # runtime-canary/ layouts) AND the declared --output-root, which may
        # live elsewhere in the repo.
        guard_roots = {manifest_path.parent, output_root}
        live = []
        seen_pid_files = set()
        for root in sorted(guard_roots):
            for pid_file in sorted(root.glob("**/orchestrator.pid")):
                if pid_file in seen_pid_files:
                    continue
                seen_pid_files.add(pid_file)
                loaded = load_pid_file(pid_file)
                if loaded and is_pid_alive_with_starttime(
                    loaded["pid"], loaded["starttime_ticks"]
                ):
                    live.append(
                        f"PID {loaded['pid']}: {pid_file.parent.parent.parent}"
                    )
        if live:
            raise SystemExit(
                "refusing to rewrite the manifest while orchestrator daemons "
                "are alive under this checkout's campaign roots — their queued "
                "specs pin the current manifest bytes and every launch "
                "revalidates them:\n  " + "\n  ".join(live)
                + "\nStop each daemon (`uv run automil --project <root> "
                "orchestrator stop`) after its queue drains, then re-run."
            )
        digest = write_manifest(build_preprint_manifest(repo_root), manifest_path)
        print(f"wrote {manifest_path} ({digest})")
    elif args.action == "check":
        manifest = load_manifest(manifest_path)
        rebuilt = build_preprint_manifest(repo_root)
        if manifest != rebuilt:
            raise SystemExit("manifest differs from the current canonical roster")
        print(f"verified 130 cells ({file_sha256(manifest_path)})")
    elif args.action == "materialize":
        if not args.agent_protocol:
            parser.error("materialize requires --agent-protocol")
        import json

        agent_protocol_path = Path(args.agent_protocol)
        if not agent_protocol_path.is_absolute():
            agent_protocol_path = (repo_root / agent_protocol_path).resolve()
        try:
            agent_protocol = json.loads(agent_protocol_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"cannot read --agent-protocol: {exc}")
        only_cells = None
        if args.cells:
            if args.cells.startswith("@"):
                try:
                    only_cells = set(json.loads(Path(args.cells[1:]).read_text())["cell_ids"])
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    parser.error(f"cannot read cell_ids from {args.cells[1:]}: {exc}")
            else:
                only_cells = {c.strip() for c in args.cells.split(",") if c.strip()}
            if not only_cells:
                parser.error("--cells names no cell")
        roots = materialize_discovery_cells(
            manifest_path, output_root, repo_root,
            agent_protocol=agent_protocol, only_cells=only_cells,
        )
        print(f"materialized {len(roots)} isolated discovery roots")
    elif args.action == "canary":
        report = run_materialization_canary(manifest_path, repo_root=repo_root)
        import json

        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.action == "freeze-selections":
        artifact = freeze_campaign_selections(output_root, manifest_path)
        print(
            "froze "
            f"{artifact['cell_count']} selections ({artifact['freeze_sha256']})"
        )
    elif args.action == "certify-all":
        index = certify_campaign(output_root, manifest_path)
        print(
            "certified "
            f"{index['cell_count']} cells ({index['certification_sha256']})"
        )
    else:
        report = write_publication_report(
            runtime_root=output_root,
            manifest_path=manifest_path,
            repo_root=repo_root,
        )
        print(
            "reported "
            f"{report['cell_count']} cells ({report['report_sha256']})"
        )


if __name__ == "__main__":
    main()
