#!/usr/bin/env python
"""Generate metadata CSV and WSI symlinks for the Hancock (HNSCC) dataset.

Reads the raw JSON data files (clinical, pathological, treatment outcome splits)
and produces:
  1. hancock_metadata.csv — one row per patient with WSI, all label columns
  2. wsi/ directory — flat symlinks to SVS files in site-specific subdirectories

Usage:
    uv run --package autobench python benchmarks/scripts/manifests/generate_hancock_metadata.py [--data-root PATH]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Hancock metadata CSV and WSI symlinks")
    parser.add_argument(
        "--data-root",
        type=str,
        default=os.environ.get(
            "AUTOBENCH_HANCOCK_ROOT",
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets", "hancock"),
        ),
        help="Path to hancock dataset root directory",
    )
    return parser.parse_args()


def load_json(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def discover_wsi_files(data_root: str) -> dict[str, str]:
    """Discover all primary WSI files (excluding _a re-scans).

    Returns dict mapping patient_id -> relative path from data_root.
    e.g. {"001": "WSI_PrimaryTumor_Hypopharynx/PrimaryTumor_HE_001.svs"}
    """
    pattern = re.compile(r"^PrimaryTumor_HE_(\d+)\.svs$")
    pid_to_relpath: dict[str, str] = {}

    for entry in sorted(os.listdir(data_root)):
        subdir = os.path.join(data_root, entry)
        if not os.path.isdir(subdir) or not entry.startswith("WSI_PrimaryTumor_"):
            continue
        if "Annotations" in entry:
            continue

        for fname in sorted(os.listdir(subdir)):
            m = pattern.match(fname)
            if m:
                pid = m.group(1)
                pid_to_relpath[pid] = os.path.join(entry, fname)

    return pid_to_relpath


def create_symlinks(data_root: str, pid_to_relpath: dict[str, str]) -> int:
    """Create a flat wsi/ directory with symlinks to all primary SVS files."""
    wsi_dir = os.path.join(data_root, "wsi")
    os.makedirs(wsi_dir, exist_ok=True)

    created = 0
    for pid, relpath in sorted(pid_to_relpath.items()):
        fname = os.path.basename(relpath)
        link_path = os.path.join(wsi_dir, fname)
        target = os.path.join(os.pardir, relpath)

        if os.path.islink(link_path):
            os.unlink(link_path)
        elif os.path.exists(link_path):
            continue  # real file, don't overwrite

        os.symlink(target, link_path)
        created += 1

    return created


def build_metadata(data_root: str, pid_to_relpath: dict[str, str]) -> pd.DataFrame:
    """Build the metadata DataFrame by joining all data sources."""
    struct_dir = os.path.join(data_root, "StructuredData")
    splits_dir = os.path.join(data_root, "DataSplits_DataDictionaries")

    # Load structured data
    clinical = pd.DataFrame(load_json(os.path.join(struct_dir, "clinical_data.json")))
    pathological = pd.DataFrame(load_json(os.path.join(struct_dir, "pathological_data.json")))

    # Load splits
    split_treatment = pd.DataFrame(load_json(os.path.join(splits_dir, "dataset_split_treatment_outcome.json")))
    split_in = pd.DataFrame(load_json(os.path.join(splits_dir, "dataset_split_in.json")))
    split_out = pd.DataFrame(load_json(os.path.join(splits_dir, "dataset_split_out.json")))

    # Merge clinical + pathological on patient_id
    df = pd.merge(clinical, pathological, on="patient_id", how="inner")

    # Merge treatment outcome split (has the outcome label + adjuvant_treatment)
    split_treatment = split_treatment.rename(columns={
        "dataset": "split_treatment_outcome",
        "recurrent event or death": "outcome_event",
        "adjuvant_treatment": "adjuvant_treatment_category",
    })
    df = pd.merge(df, split_treatment, on="patient_id", how="left")

    # Merge in/out splits
    split_in = split_in.rename(columns={"dataset": "split_in"})
    split_out = split_out.rename(columns={"dataset": "split_out"})
    df = pd.merge(df, split_in[["patient_id", "split_in"]], on="patient_id", how="left")
    df = pd.merge(df, split_out[["patient_id", "split_out"]], on="patient_id", how="left")

    # Filter to patients with WSI files
    df["has_wsi"] = df["patient_id"].isin(pid_to_relpath)
    n_total = len(df)
    df = df[df["has_wsi"]].copy()
    n_with_wsi = len(df)
    print(f"Patients: {n_total} total, {n_with_wsi} with WSI ({n_total - n_with_wsi} excluded)")

    # Create slide_id (without .svs extension)
    df["slide_id"] = df["patient_id"].apply(lambda pid: f"PrimaryTumor_HE_{pid}")

    # --- Label columns ---

    # 1. Tumor site: 4-class (excluding CUP which has only 1 slide)
    tumor_site_map = {
        "Hypopharynx": 0,
        "Larynx": 1,
        "Oral_Cavity": 2,
        "Oropharynx": 3,
    }
    df["label_tumor_site"] = df["primary_tumor_site"].map(tumor_site_map)
    # CUP patients get NaN (not mapped)

    # 2. HPV p16 status: binary (only for tested patients)
    hpv_map = {"negative": 0, "positive": 1}
    df["label_hpv_p16"] = df["hpv_association_p16"].map(hpv_map)
    # "not_tested" patients get NaN (not mapped)

    # 3. Survival: binary
    survival_map = {"living": 0, "deceased": 1}
    df["label_survival"] = df["survival_status"].map(survival_map)

    # 4. Treatment outcome: binary (recurrent event or death)
    df["label_treatment_outcome"] = df["outcome_event"].astype("Int64")

    # 5. High grade: binary (G1/G2 = 0, G3 = 1; HPV-graded cases get NaN)
    def map_grade(g):
        if g in ("G1", "G2"):
            return 0
        elif g == "G3":
            return 1
        else:
            return pd.NA  # "hpv_association_p16" or other non-standard values
    df["label_high_grade"] = df["grading"].apply(map_grade).astype("Int64")

    # Select and order output columns
    output_cols = [
        "slide_id",
        "patient_id",
        "primary_tumor_site",
        # Labels
        "label_tumor_site",
        "label_hpv_p16",
        "label_survival",
        "label_treatment_outcome",
        "label_high_grade",
        # Splits (for reference / future use)
        "split_treatment_outcome",
        "split_in",
        "split_out",
        # Key clinical variables
        "age_at_initial_diagnosis",
        "sex",
        "smoking_status",
        "survival_status",
        "survival_status_with_cause",
        "days_to_last_information",
        # Key pathological variables
        "pT_stage",
        "pN_stage",
        "grading",
        "hpv_association_p16",
        "histologic_type",
        "resection_status",
        "adjuvant_treatment_category",
    ]
    df = df[output_cols].copy()
    df = df.sort_values("patient_id").reset_index(drop=True)

    return df


def main():
    args = parse_args()
    data_root = args.data_root
    print(f"Hancock data root: {data_root}")

    if not os.path.isdir(data_root):
        print(f"ERROR: Data root not found: {data_root}")
        sys.exit(1)

    # Discover WSI files
    pid_to_relpath = discover_wsi_files(data_root)
    print(f"Discovered {len(pid_to_relpath)} primary WSI files")

    # Create symlinks
    n_created = create_symlinks(data_root, pid_to_relpath)
    wsi_dir = os.path.join(data_root, "wsi")
    n_links = len([f for f in os.listdir(wsi_dir) if f.endswith(".svs")])
    print(f"Symlinks: {n_created} created, {n_links} total in wsi/")

    # Build metadata
    df = build_metadata(data_root, pid_to_relpath)

    # Write CSV
    csv_path = os.path.join(data_root, "hancock_metadata.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nMetadata CSV written to: {csv_path}")
    print(f"Shape: {df.shape}")

    # Print label distributions
    print("\n--- Label Distributions ---")
    for col in ["label_tumor_site", "label_hpv_p16", "label_survival",
                 "label_treatment_outcome", "label_high_grade"]:
        counts = df[col].value_counts(dropna=False).sort_index()
        n_valid = df[col].notna().sum()
        print(f"\n{col} ({n_valid} valid / {len(df)} total):")
        for val, count in counts.items():
            print(f"  {val}: {count}")

    # Verify symlink targets
    broken = 0
    for fname in os.listdir(wsi_dir):
        link = os.path.join(wsi_dir, fname)
        if os.path.islink(link) and not os.path.exists(link):
            broken += 1
            print(f"  BROKEN: {link}")
    if broken:
        print(f"\nWARNING: {broken} broken symlinks found!")
    else:
        print(f"\nAll {n_links} symlinks verified OK")


if __name__ == "__main__":
    main()
