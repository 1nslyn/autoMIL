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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("generate", "check", "materialize", "canary"),
    )
    parser.add_argument(
        "--manifest", default="benchmarks/campaigns/preprint_130/manifest.json",
    )
    parser.add_argument(
        "--output-root", default="benchmarks/campaigns/preprint_130/runtime",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = (repo_root / args.manifest).resolve()
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
            manifest_path, (repo_root / args.output_root), repo_root,
        )
        print(f"materialized {len(roots)} isolated discovery roots")
    else:
        report = run_materialization_canary(manifest_path, repo_root=repo_root)
        import json

        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
