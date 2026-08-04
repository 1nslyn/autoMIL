#!/usr/bin/env python
"""Generate, verify, or materialize the frozen preprint-130 campaign."""
from __future__ import annotations

import argparse
from pathlib import Path

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
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = (repo_root / args.manifest).resolve()
    output_root = (repo_root / args.output_root).resolve()
    try:
        output_root.relative_to(repo_root)
    except ValueError:
        parser.error("--output-root must live inside the git repository")
    if args.action == "generate":
        digest = write_manifest(build_preprint_manifest(repo_root), manifest_path)
        print(f"wrote {manifest_path} ({digest})")
    elif args.action == "check":
        manifest = load_manifest(manifest_path)
        rebuilt = build_preprint_manifest(repo_root)
        if manifest != rebuilt:
            raise SystemExit("manifest differs from the current canonical roster")
        print(f"verified 130 cells ({file_sha256(manifest_path)})")
    elif args.action == "materialize":
        roots = materialize_discovery_cells(
            manifest_path, output_root, repo_root,
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
