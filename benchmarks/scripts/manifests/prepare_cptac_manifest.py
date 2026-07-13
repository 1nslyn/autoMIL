#!/usr/bin/env python3
"""Prepare normalized_manifest.csv for CPTAC datasets from Patho-Bench splits.

This replaces the GOLDMARK download step used for TCGA. Patho-Bench provides
canonical train/test splits for all CPTAC cohorts on HuggingFace; this script
downloads them and merges multiple tasks into a single autobench-compatible CSV.

Usage:
    python benchmarks/scripts/manifests/prepare_cptac_manifest.py \\
        --source cptac_ccrcc \\
        --tasks BAP1_mutation VHL_mutation \\
        --saveto /path/to/dataset/root

Output:
    {saveto}/normalized_manifest.csv   — consumed by autobench pipeline as mapping_csv
    {saveto}/pb_splits/                — raw Patho-Bench TSVs (kept for reference)

Column conventions in the output CSV:
    case_id    — patient-level ID (from Patho-Bench sample_col)
    slide_id   — slide stem, no extension; matches TRIDENT .h5 filenames
    {GENE}_binary — 0/1 label per task (e.g. BAP1_binary, VHL_binary)

Available CPTAC sources on HuggingFace (MahmoodLab/Patho-Bench):
    cptac_brca, cptac_ccrcc, cptac_coad, cptac_gbm, cptac_hnsc,
    cptac_lscc, cptac_luad, cptac_pdac, cptac_ucec, cptac_ov
"""

from __future__ import annotations

import argparse
import os
import shutil

import pandas as pd
import yaml

PATHO_BENCH_HF = "MahmoodLab/Patho-Bench"


def _pb_split_paths(pb_root: str, source: str, task: str) -> tuple[str, str]:
    split = os.path.join(pb_root, source, task, "k=all.tsv")
    config = os.path.join(pb_root, source, task, "config.yaml")
    return split, config


def _download(pb_root: str, source: str, task: str) -> tuple[str, str]:
    split_path, config_path = _pb_split_paths(pb_root, source, task)
    if os.path.exists(split_path):
        print(f"  [{source}/{task}] already downloaded, skipping.")
        return split_path, config_path

    print(f"  [{source}/{task}] downloading from HuggingFace...")
    import datasets  # noqa: PLC0415

    datasets.load_dataset(
        PATHO_BENCH_HF,
        cache_dir=pb_root,
        dataset_to_download=source,
        task_in_dataset=task,
        trust_remote_code=True,
    )

    # Remove HF cache clutter
    for d in ["MahmoodLab___patho-bench", ".cache"]:
        p = os.path.join(pb_root, d)
        if os.path.exists(p):
            shutil.rmtree(p)
    for root, _, files in os.walk(pb_root):
        for f in files:
            if f.endswith(".lock"):
                os.remove(os.path.join(root, f))

    assert os.path.exists(split_path), (
        f"Download appeared to succeed but {split_path} not found. "
        "Check that the source/task names are valid for MahmoodLab/Patho-Bench."
    )
    return split_path, config_path


def _load_task(split_path: str, config_path: str, task: str) -> pd.DataFrame:
    with open(config_path) as f:
        info = yaml.safe_load(f)

    sample_col: str = info["sample_col"]       # e.g. "case_id"
    task_col: str = info.get("task_col", task)  # e.g. "BAP1_mutation"

    df = pd.read_csv(
        split_path,
        sep="\t",
        dtype={"case_id": str, "slide_id": str},
    )

    # Keep only what we need; drop pre-assigned fold columns
    keep = [c for c in [sample_col, "slide_id", task_col] if c in df.columns]
    df = df[keep].copy()

    # Normalise to autobench column names
    df = df.rename(columns={sample_col: "case_id"})

    # Map task label → {GENE}_binary convention
    gene = (
        task
        .replace("_mutation", "")
        .replace("_status", "")
        .replace("_expression", "")
        .upper()
    )
    label_col = f"{gene}_binary"
    df = df.rename(columns={task_col: label_col})

    # Coerce label to int where numeric
    try:
        df[label_col] = df[label_col].astype(float).astype("Int64")
    except (ValueError, TypeError):
        pass

    return df


def list_tasks(source: str) -> None:
    """Print all available tasks for a given Patho-Bench source."""
    from huggingface_hub import list_repo_files  # noqa: PLC0415

    print(f"Fetching task list for '{source}' from MahmoodLab/Patho-Bench...")
    all_files = list(list_repo_files(PATHO_BENCH_HF, repo_type="dataset"))

    # Files are at: {source}/{task}/k=all.tsv
    prefix = f"{source}/"
    tasks = sorted({
        f.removeprefix(prefix).split("/")[0]
        for f in all_files
        if f.startswith(prefix) and f.endswith(".tsv")
    })

    if not tasks:
        print(f"No tasks found for source '{source}'.")
        sources = sorted({
            f.split("/")[0]
            for f in all_files
            if f.endswith(".tsv") and "/" in f
        })
        print("Available sources:", sources)
        return

    print(f"\nAvailable tasks for {source} ({len(tasks)}):")
    for t in tasks:
        config_path = os.path.join(source, t, "config.yaml")
        if config_path in all_files:
            from huggingface_hub import hf_hub_download  # noqa: PLC0415
            local = hf_hub_download(PATHO_BENCH_HF, config_path, repo_type="dataset")
            with open(local) as f:
                info = yaml.safe_load(f)
            task_type = info.get("task_type", "?")
            label_dict = info.get("label_dict", {})
            labels = ", ".join(f"{k}={v}" for k, v in sorted(label_dict.items()))
            print(f"  {t}  [{task_type}]  labels: {labels}")
        else:
            print(f"  {t}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare CPTAC normalized_manifest.csv from Patho-Bench HuggingFace splits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Patho-Bench source name, e.g. cptac_ccrcc",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Task name(s) to include, e.g. BAP1_mutation VHL_mutation",
    )
    parser.add_argument(
        "--saveto",
        help="Dataset root directory; normalized_manifest.csv is written here",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List all available tasks for --source and exit",
    )
    args = parser.parse_args()

    if args.list_tasks:
        list_tasks(args.source)
        return

    if not args.tasks:
        parser.error("--tasks is required unless --list-tasks is set")
    if not args.saveto:
        parser.error("--saveto is required unless --list-tasks is set")

    saveto = os.path.abspath(args.saveto)
    pb_root = os.path.join(saveto, "pb_splits")
    os.makedirs(pb_root, exist_ok=True)

    merged: pd.DataFrame | None = None

    for task in args.tasks:
        split_path, config_path = _download(pb_root, args.source, task)
        df = _load_task(split_path, config_path, task)

        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=["case_id", "slide_id"], how="outer")

    if merged is None:
        print("No tasks processed.")
        return

    out = os.path.join(saveto, "normalized_manifest.csv")
    merged.to_csv(out, index=False)

    print(f"\nWrote: {out}")
    print(f"  Rows   : {len(merged)}")
    print(f"  Columns: {list(merged.columns)}")
    label_cols = [c for c in merged.columns if c.endswith("_binary")]
    for col in label_cols:
        total = merged[col].notna().sum()
        counts = merged[col].value_counts(dropna=True).sort_index()
        breakdown = " / ".join(f"{int(v)}={c}" for v, c in counts.items())
        n_classes = len(counts)
        print(f"  {col}: {total} labelled, {n_classes} classes ({breakdown})")
    print()
    print("Next steps:")
    print(f"  1. Check slide IDs match your .svs filenames: head -5 {out}")
    print(f"  2. Create benchmarks/datasets/{args.source}.yaml using cptac_template.yaml")
    print(f"  3. Add AUTOBENCH_{args.source.upper()}_ROOT={saveto} to benchmarks/.env")


if __name__ == "__main__":
    main()
