"""Contracts for the guard-root benchmark shim builder.

The shim's whole point is that campaign-reachable writes can no longer
touch the frozen source tree; every rule here is exercised by a forged
violation that would defeat a naive implementation.
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "build_guard_root_shims",
        REPO_ROOT / "benchmarks/scripts/build_guard_root_shims.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shims = _load_module()


def _frozen_tree(tmp_path: Path) -> Path:
    """A miniature frozen benchmark tree with every audited entry."""
    source = tmp_path / "frozen" / "benchmark_5fold"
    (source / "dataset_csv").mkdir(parents=True)
    (source / "dataset_csv" / "kras.csv").write_text("slide_id,label\na,1\n")
    (source / "dataset_csv" / "manifest_fingerprint.json").write_text(
        json.dumps({"sha256": "f" * 64})
    )
    (source / "splits" / "standard" / "kras").mkdir(parents=True)
    for fold in range(2):
        (source / "splits" / "standard" / "kras" / f"splits_{fold}.csv").write_text(
            "train,val\n"
        )
    (source / "titan" / "kras").mkdir(parents=True)
    (source / "titan" / "kras" / "manifest.json").write_text("{}")
    (source / "nnmil" / "standard" / "kras_uni_v2").mkdir(parents=True)
    (source / "nnmil" / "standard" / "kras_uni_v2" / "dataset_plan.json").write_text(
        "{}"
    )
    (source / "features" / "uni_v2" / "pt_files").mkdir(parents=True)
    (source / "features" / "uni_v2" / "pt_files" / "slide_a.pt").write_bytes(
        b"tensor-bytes"
    )
    for legacy in ("results", "aggregated", "logs"):
        (source / legacy / "old").mkdir(parents=True)
        (source / legacy / "old" / "artifact.txt").write_text("legacy\n")
    return source


def _legacy_guard_root(tmp_path: Path, source: Path) -> Path:
    cohort_root = tmp_path / "guard_roots" / "tcga_luad"
    cohort_root.mkdir(parents=True)
    (cohort_root / "benchmark").symlink_to(source)
    return cohort_root / "benchmark"


def test_shim_shape_realizes_the_write_map(tmp_path):
    source = _frozen_tree(tmp_path)
    benchmark = _legacy_guard_root(tmp_path, source)
    shims.build_shim(benchmark)

    assert benchmark.is_dir() and not benchmark.is_symlink()
    # Write-prone subtrees are byte-copies of real files.
    split = benchmark / "splits" / "standard" / "kras" / "splits_0.csv"
    assert split.is_file() and not split.is_symlink()
    fingerprint = benchmark / "dataset_csv" / "manifest_fingerprint.json"
    assert fingerprint.is_file() and not fingerprint.is_symlink()
    assert (benchmark / "titan" / "kras" / "manifest.json").read_text() == "{}"
    # Features: real directories, per-file symlinks.
    pt_dir = benchmark / "features" / "uni_v2" / "pt_files"
    assert pt_dir.is_dir() and not pt_dir.is_symlink()
    pt = pt_dir / "slide_a.pt"
    assert pt.is_symlink() and pt.read_bytes() == b"tensor-bytes"
    # Legacy outputs become empty write sinks — no symlink back, no content.
    for legacy in ("results", "aggregated", "logs"):
        sink = benchmark / legacy
        assert sink.is_dir() and not sink.is_symlink()
        assert list(sink.iterdir()) == []
    assert (benchmark / ".shim_source").read_text().strip() == str(source)


def test_writes_into_the_shim_never_reach_the_frozen_tree(tmp_path):
    """The forged violation the shim exists for: the TITAN rewrite and a
    fingerprint refresh land in the shim while the source stays byte-frozen."""
    source = _frozen_tree(tmp_path)
    benchmark = _legacy_guard_root(tmp_path, source)
    before = {
        str(path): path.stat().st_mtime_ns
        for path in source.rglob("*") if path.is_file()
    }
    shims.build_shim(benchmark)

    (benchmark / "titan" / "kras" / "manifest.json").write_text('{"new": 1}')
    (benchmark / "dataset_csv" / "manifest_fingerprint.json").write_text("{}")
    (benchmark / "features" / "uni_v2" / "pt_files" / "slide_b.pt").write_bytes(
        b"regenerated"
    )
    (benchmark / "results" / "stray.json").write_text("{}")

    after = {
        str(path): path.stat().st_mtime_ns
        for path in source.rglob("*") if path.is_file()
    }
    assert after == before
    assert (source / "titan" / "kras" / "manifest.json").read_text() == "{}"
    assert not (source / "features" / "uni_v2" / "pt_files" / "slide_b.pt").exists()
    assert not (source / "results" / "stray.json").exists()


def test_unknown_top_level_entry_fails_closed(tmp_path):
    """A new entry means the write map must be re-audited, never guessed."""
    source = _frozen_tree(tmp_path)
    (source / "mystery_store").mkdir()
    benchmark = _legacy_guard_root(tmp_path, source)
    with pytest.raises(shims.ShimError, match="mystery_store"):
        shims.build_shim(benchmark)
    # Failed builds must not have swapped anything.
    assert benchmark.is_symlink()


def test_rebuild_from_a_prior_shim_reuses_the_recorded_source(tmp_path):
    source = _frozen_tree(tmp_path)
    benchmark = _legacy_guard_root(tmp_path, source)
    shims.build_shim(benchmark)
    # Mutate the shim, then rebuild: content is restored from the source.
    (benchmark / "titan" / "kras" / "manifest.json").write_text('{"dirty": 1}')
    shims.build_shim(benchmark)
    assert (benchmark / "titan" / "kras" / "manifest.json").read_text() == "{}"
    assert (benchmark / ".shim_source").read_text().strip() == str(source)


def test_unrecognized_benchmark_object_is_refused(tmp_path):
    source = _frozen_tree(tmp_path)
    cohort_root = tmp_path / "guard_roots" / "tcga_luad"
    cohort_root.mkdir(parents=True)
    (cohort_root / "benchmark").mkdir()  # real dir, no marker
    with pytest.raises(shims.ShimError, match="refusing to guess"):
        shims.build_shim(cohort_root / "benchmark")


def test_copy_verification_fails_closed_on_divergence(tmp_path, monkeypatch):
    """Forged violation: a copy that does not byte-match must abort the
    build before any swap."""
    source = _frozen_tree(tmp_path)
    benchmark = _legacy_guard_root(tmp_path, source)

    real_copytree = shims.shutil.copytree

    def corrupting(src, dst, **kwargs):
        result = real_copytree(src, dst, **kwargs)
        target = Path(dst) / "kras.csv"
        if target.exists():
            target.write_text("corrupted\n")
        return result

    monkeypatch.setattr(shims.shutil, "copytree", corrupting)
    with pytest.raises(shims.ShimError, match="verification failed"):
        shims.build_shim(benchmark)
    assert benchmark.is_symlink()


def test_cli_requires_the_idle_confirmation(tmp_path, capsys):
    rc = shims.main(["--guard-roots", str(tmp_path)])
    assert rc == 1
    assert "--i-know-idle" in capsys.readouterr().err


def test_cli_builds_every_roster_cohort_from_a_fabricated_repo(tmp_path):
    repo = tmp_path / "repo"
    campaign = repo / "benchmarks/campaigns/preprint_130"
    campaign.mkdir(parents=True)
    (campaign / "active_roster.json").write_text(json.dumps({
        "cohorts": ["tcga_luad"], "cells": 26,
    }))
    source = _frozen_tree(tmp_path)
    guard_roots = tmp_path / "guard_roots"
    _legacy_guard_root(tmp_path, source)

    rc = shims.main([
        "--guard-roots", str(guard_roots),
        "--repo-root", str(repo),
        "--i-know-idle",
    ])
    assert rc == 0
    assert (guard_roots / "tcga_luad" / "benchmark" / ".shim_source").is_file()
