"""Shared data preparation: task CSV creation, split generation, orchestration.

Framework-specific preparation lives in each adapter:
- ``clam/prepare.py``: H5 -> PT conversion
- ``nnmil/prepare.py``: dataset.json, dataset_plan.json generation
- ``smmile/prepare.py``: H5 -> NIC conversion, superpixel generation
"""

from __future__ import annotations

import os

import pandas as pd

from autobench.config import DatasetConfig
from autobench.data import load_all_slides
from autobench.pipeline.splits import create_strategy_splits


# ---------------------------------------------------------------------------
# Task CSV generation
# ---------------------------------------------------------------------------


def create_task_csv(
    mapping_csv: str,
    output_csv: str,
    label_col: str,
    label_map: dict[int, str],
    ds: DatasetConfig,
) -> pd.DataFrame:
    """Create a CLAM-compatible task CSV from mapping.csv.

    Loads all slides and maps labels using ``label_map``.
    """
    df = load_all_slides(mapping_csv, ds)

    df = df.dropna(subset=[label_col]).reset_index(drop=True)

    # Handle label values that may be strings (multiclass) or numeric
    slide_col = ds.slide_id_column
    case_col = ds.case_id_column

    if _is_numeric_labels(df[label_col]):
        df[label_col] = df[label_col].astype(int)
        task_df = pd.DataFrame({
            "case_id": df[case_col],
            "slide_id": df[slide_col].apply(ds.get_slide_id),
            "label": df[label_col].map(label_map),
        })
    else:
        # Labels are already strings (e.g., CLWD where CSV has "Acinar", "Solid", etc.)
        # label_map is {0: "Acinar", ...} -- we need reverse: {"Acinar": "Acinar", ...}
        # Just use the raw label values directly since they are already class names
        task_df = pd.DataFrame({
            "case_id": df[case_col],
            "slide_id": df[slide_col].apply(ds.get_slide_id),
            "label": df[label_col],
        })

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    task_df.to_csv(output_csv, index=False)
    print(f"  Task CSV: {output_csv}  ({len(task_df)} slides, "
          f"{task_df['label'].value_counts().to_dict()})")
    return task_df


def _is_numeric_labels(series: pd.Series) -> bool:
    """Check if a label column contains numeric values."""
    try:
        series.dropna().astype(float)
        return True
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Full preparation orchestrator
# ---------------------------------------------------------------------------


def prepare_all(
    benchmark_dir: str,
    mapping_csv: str,
    features_base_dir: str,
    encoder_keys: list[str],
    ds: DatasetConfig,
    seed: int = 42,
    n_splits: int = 5,
    holdout_frac: float | None = None,
) -> None:
    """Run the complete data-preparation pipeline (idempotent).

    Uses task definitions from ``DatasetConfig`` instead of hardcoded values.

    ``holdout_frac`` is forwarded to ``create_strategy_splits``: ``None`` keeps
    the conservative 3-way split (default); a float selects GOLDMARK-parity
    2-way splits (val==test). Splits are cached per ``benchmark_dir``, so a
    parity run must use a distinct ``benchmark_dir`` to avoid reusing
    previously-written 3-way splits.
    """
    all_slide_ids: set[str] = set()

    # Get default strategy (first one defined in the config)
    default_strategy = list(ds.split_strategies.keys())[0]

    for task_name, tdef in ds.tasks.items():
        csv_path = os.path.join(benchmark_dir, "dataset_csv", f"{task_name}.csv")
        if not os.path.exists(csv_path):
            print(f"[prep] Creating task CSV: {task_name}")
            task_df = create_task_csv(
                mapping_csv, csv_path,
                label_col=tdef.label_col,
                label_map=tdef.label_map,
                ds=ds,
            )
        else:
            print(f"[prep] Task CSV already exists: {csv_path}")
            task_df = pd.read_csv(csv_path)
        all_slide_ids.update(task_df["slide_id"].tolist())

        splits_dir = os.path.join(benchmark_dir, "splits", default_strategy, task_name)
        first_split = os.path.join(splits_dir, "splits_0.csv")
        if not os.path.exists(first_split):
            print(f"[prep] Creating splits: {task_name}")
            create_strategy_splits(
                csv_path, splits_dir, n_splits=n_splits, seed=seed,
                holdout_frac=holdout_frac,
            )
        else:
            print(f"[prep] Splits already exist: {splits_dir}")

    # H5 -> PT for each encoder (CLAM-specific)
    from autobench.pipeline.clam.prepare import convert_h5_to_pt

    slide_ids_sorted = sorted(all_slide_ids)
    for encoder_key in encoder_keys:
        h5_dir = os.path.join(features_base_dir, f"features_{encoder_key}")
        pt_dir = os.path.join(benchmark_dir, "features", encoder_key)
        print(f"[prep] Converting H5->PT: {encoder_key} ({len(slide_ids_sorted)} slides)")
        n = convert_h5_to_pt(h5_dir, pt_dir, encoder_key, slide_ids_sorted)
        if n > 0:
            print(f"  Converted {n} new files")
        else:
            print(f"  All files already converted")
