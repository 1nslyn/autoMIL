"""Strategy-aware split generation for benchmarking.

Generates split CSVs in the same format as ``prepare.create_stratified_splits``
(columns: train, val, test with slide_ids, padded with NA) so that CLAM's
``return_splits`` and the nnMIL plan builder can consume them identically.

Splits are **patient-level**: all slides from the same ``case_id`` go to the
same fold. This matches CLAM upstream's ``patient_strat=True`` and nnMIL
upstream's patient-keyed stratification, and avoids same-patient leakage
across train/val/test.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from autobench.pipeline.config import StrategyConfig


def create_strategy_splits(
    task_csv: str,
    splits_dir: str,
    strategy_cfg: StrategyConfig | None = None,
    n_splits: int = 5,
    seed: int = 42,
    stratify_col: str = "label",
) -> list[str]:
    """Create split CSVs using patient-stratified k-fold CV.

    Each fold: ~80 % (train+val) / ~20 % test, with val carved as ~12.5 %
    of (train+val) to match nnMIL upstream's ``val_frac=0.125``.

    The task CSV must have columns ``case_id``, ``slide_id``, and ``label``.
    All slides from the same ``case_id`` are forced into the same partition
    (train, val, or test) to prevent patient-level leakage.

    Output CSV format: columns ``[train, val, test]`` with slide_ids,
    padded with ``pd.NA`` — identical to CLAM's expected format.

    Parameters
    ----------
    strategy_cfg:
        Only ``standard`` is implemented today. A non-``standard`` value
        is rejected explicitly so a new strategy added to a dataset YAML
        doesn't silently produce in-distribution splits while the
        operator believes they configured a held-out cohort.

    Returns a list of paths to the generated split CSVs.
    """
    df = pd.read_csv(task_csv)
    if "case_id" not in df.columns:
        raise ValueError(
            f"Task CSV {task_csv} is missing required 'case_id' column. "
            "Patient-level stratification cannot proceed."
        )
    if strategy_cfg is not None:
        strategy_name = getattr(strategy_cfg, "strategy", None)
        if strategy_name and strategy_name != "standard":
            raise NotImplementedError(
                f"strategy {strategy_name!r} is not implemented; only "
                "'standard' (patient-stratified k-fold) is supported. "
                "A held-out cohort split would silently regress to "
                "in-distribution if we let this fall through — implement "
                "the strategy branch explicitly before using it."
            )
    if stratify_col not in df.columns:
        raise ValueError(
            f"Task CSV {task_csv} is missing the stratification column "
            f"{stratify_col!r}. Available columns: {list(df.columns)}."
        )
    os.makedirs(splits_dir, exist_ok=True)

    return _splits_standard_cv(df, splits_dir, n_splits, seed, stratify_col)


def _splits_standard_cv(
    df: pd.DataFrame,
    splits_dir: str,
    n_splits: int,
    seed: int,
    stratify_col: str = "label",
) -> list[str]:
    """Patient-stratified k-fold: dedup to cases, split cases, expand to slides.

    Slides from the same ``case_id`` share a stratification key — the class
    label for classification, or the event indicator (``status``) for
    survival — so we can dedup safely and run standard StratifiedKFold on
    cases. Stratifying survival splits on ``status`` keeps every fold from
    landing zero events (which would make the c-index undefined and collapse
    nllsurv's quantile bins). Avoids StratifiedGroupKFold's minimum-stratum-
    size limitation.
    """
    # One row per case with its stratification key (slides of the same case
    # share it — mutations and survival outcomes are both case-level).
    case_table = df.groupby("case_id", sort=True)[stratify_col].first().reset_index()
    case_ids = np.asarray(case_table["case_id"], dtype=object)
    # Preserve the native dtype: string labels stay object (sklearn reads them
    # as 'binary'/'multiclass'), but integer survival status must NOT be cast
    # to object or sklearn's type_of_target rejects it as 'unknown'. .to_numpy()
    # (not .values) also keeps this pandas-3 safe: on pandas>=3 string columns
    # are Arrow-backed and .values returns an ExtensionArray sklearn can't index.
    case_labels = case_table[stratify_col].to_numpy()

    # Upfront feasibility check: sklearn raises mid-fit with a generic message
    # ("n_splits=N cannot be greater than the number of members in each class")
    # which is hard to interpret on a small minority class. Fail early with the
    # concrete numbers so the operator can drop n_splits or merge classes.
    label_counts = pd.Series(case_labels).value_counts().to_dict()
    min_label_count = int(min(label_counts.values()))
    if min_label_count < n_splits:
        raise ValueError(
            f"Cannot run {n_splits}-fold patient-stratified CV: smallest "
            f"class has only {min_label_count} cases. Per-class case counts "
            f"(case_id-deduplicated): {label_counts}. Reduce n_splits to "
            f"<= {min_label_count} or merge minority classes."
        )
    # The inner train_test_split (test_size=0.125) needs each class to have
    # >= 2 train_val cases after the outer split removes ~1/n_splits cases.
    # With min_label_count >= n_splits guaranteed above, the worst case has
    # min_label_count*(1-1/n_splits) train_val cases per class; we still want
    # at least 2 (1 train + 1 val) so refuse if the projected count is < 2.
    projected_train_val_per_class = min_label_count - (min_label_count // n_splits)
    if projected_train_val_per_class < 2:
        raise ValueError(
            f"Cannot carve an inner val set: after the outer {n_splits}-fold "
            f"split, the smallest class would have only "
            f"{projected_train_val_per_class} cases in train+val, but "
            f"train_test_split(stratify=...) requires >= 2. Reduce n_splits "
            f"or augment the minority class."
        )

    # Map case_id -> list of slide_ids for expansion
    case_to_slides = df.groupby("case_id")["slide_id"].apply(list).to_dict()

    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    split_paths: list[str] = []

    for fold_idx, (train_val_case_idx, test_case_idx) in enumerate(
        outer.split(case_ids, case_labels)
    ):
        train_val_cases = case_ids[train_val_case_idx]
        train_val_labels = case_labels[train_val_case_idx]
        test_cases = case_ids[test_case_idx]

        # Inner val carve: ~12.5% val on cases, matches nnMIL upstream val_frac=0.125
        train_cases, val_cases = train_test_split(
            train_val_cases,
            test_size=0.125,
            stratify=train_val_labels,
            random_state=seed + fold_idx,
        )

        _assert_no_patient_leakage(
            train_cases=np.asarray(train_cases),
            val_cases=np.asarray(val_cases),
            test_cases=np.asarray(test_cases),
            fold_idx=fold_idx,
        )

        train_ids = _expand_cases_to_slides(train_cases, case_to_slides)
        val_ids = _expand_cases_to_slides(val_cases, case_to_slides)
        test_ids = _expand_cases_to_slides(test_cases, case_to_slides)

        path = _write_split_csv(splits_dir, fold_idx, train_ids, val_ids, test_ids)
        split_paths.append(path)

    print(f"  Splits: {splits_dir}  ({n_splits} folds, patient-stratified CV)")
    return split_paths


def _expand_cases_to_slides(
    cases: np.ndarray | list,
    case_to_slides: dict[str, list[str]],
) -> np.ndarray:
    """Expand a list of case_ids to the flat list of their slide_ids."""
    out: list[str] = []
    for c in cases:
        out.extend(case_to_slides.get(c, []))
    return np.asarray(out, dtype=object)


def _assert_no_patient_leakage(
    train_cases: np.ndarray,
    val_cases: np.ndarray,
    test_cases: np.ndarray,
    fold_idx: int,
) -> None:
    """Hard fail if any case_id crosses train/val/test boundaries."""
    train_set = set(train_cases.tolist())
    val_set = set(val_cases.tolist())
    test_set = set(test_cases.tolist())
    train_val = train_set & val_set
    train_test = train_set & test_set
    val_test = val_set & test_set
    if train_val or train_test or val_test:
        raise AssertionError(
            f"Patient leakage in fold {fold_idx}: "
            f"train∩val={train_val}, train∩test={train_test}, val∩test={val_test}"
        )


def _write_split_csv(
    splits_dir: str,
    fold_idx: int,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    test_ids: np.ndarray,
) -> str:
    """Write a single split CSV with columns [train, val, test], padded with NA."""
    max_len = max(len(train_ids), len(val_ids), len(test_ids))

    def _pad(arr: np.ndarray) -> list:
        return list(arr) + [pd.NA] * (max_len - len(arr))

    split_df = pd.DataFrame({
        "train": _pad(train_ids),
        "val": _pad(val_ids),
        "test": _pad(test_ids),
    })
    path = os.path.join(splits_dir, f"splits_{fold_idx}.csv")
    split_df.to_csv(path, index=False)
    return path
