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
from functools import partial

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

from autobench.pipeline.config import StrategyConfig

#: Every development-CV strategy and the seed it partitions standard fold 0's
#: train+val slides with. ``devcv_r1`` and ``devcv_r2`` are fresh partitions of
#: the same slides ``devcv`` cuts; the strategy registry is built from this map.
DEVCV_PARTITION_SEEDS = {"devcv": 1000, "devcv_r1": 1001, "devcv_r2": 1002}


def create_strategy_splits(
    task_csv: str,
    splits_dir: str,
    strategy_cfg: StrategyConfig,
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
        ``standard`` is patient-stratified k-fold CV: every slide is a test
        slide exactly once. ``devcv`` keeps standard fold 0's test slides
        aside as one fixed test set and cross-validates the remaining
        (development) slides: fold k trains on k-1 of ``n_splits``
        patient-grouped, class-stratified parts and validates on the k-th,
        so every development slide is validated exactly once and no test
        slide is ever validated. ``devcv_r1`` and ``devcv_r2`` do the same
        with a different partition seed (``DEVCV_PARTITION_SEEDS``). Any
        other value is rejected explicitly so a new strategy added to a
        dataset YAML doesn't silently produce in-distribution splits while
        the operator believes they configured a held-out cohort.

    Returns a list of paths to the generated split CSVs.
    """
    df = pd.read_csv(task_csv)
    if "case_id" not in df.columns:
        raise ValueError(
            f"Task CSV {task_csv} is missing required 'case_id' column. "
            "Patient-level stratification cannot proceed."
        )
    strategy_name = strategy_cfg.strategy
    if strategy_name not in _STRATEGIES:
        raise NotImplementedError(
            f"strategy {strategy_name!r} is not implemented; supported: "
            f"{sorted(_STRATEGIES)}. A held-out cohort split would silently "
            "regress to in-distribution if we let this fall through — "
            "implement the strategy branch explicitly before using it."
        )
    if stratify_col not in df.columns:
        raise ValueError(
            f"Task CSV {task_csv} is missing the stratification column "
            f"{stratify_col!r}. Available columns: {list(df.columns)}."
        )
    os.makedirs(splits_dir, exist_ok=True)

    return _STRATEGIES[strategy_name](df, splits_dir, n_splits, seed, stratify_col)


def _splits_standard_cv(
    df: pd.DataFrame,
    splits_dir: str,
    n_splits: int,
    seed: int,
    stratify_col: str = "label",
) -> list[str]:
    """Write every standard fold."""
    split_paths = [
        _write_split_csv(splits_dir, fold_idx, train_ids, val_ids, test_ids)
        for fold_idx, (train_ids, val_ids, test_ids)
        in enumerate(_standard_folds(df, n_splits, seed, stratify_col))
    ]
    print(f"  Splits: {splits_dir}  ({n_splits} folds, patient-stratified CV)")
    return split_paths


def _splits_devcv(
    df: pd.DataFrame,
    splits_dir: str,
    n_splits: int,
    seed: int,
    stratify_col: str = "label",
    *,
    partition_seed: int,
) -> list[str]:
    """Cross-validate standard fold 0's development slides; test on its test slides.

    The development slides (fold 0's train then val, in that order) are cut
    into ``n_splits`` parts by a ``StratifiedGroupKFold`` shuffled with
    ``partition_seed``, grouped on ``case_id`` and stratified on
    ``stratify_col``. Fold k validates on part k and trains on the rest;
    every fold keeps fold 0's test slides.
    """
    train0, val0, test0 = next(iter(_standard_folds(df, n_splits, seed, stratify_col)))
    development = np.concatenate([train0, val0])
    slide_rows = df.set_index("slide_id")
    labels = slide_rows.loc[development, stratify_col].to_numpy()
    groups = slide_rows.loc[development, "case_id"].to_numpy()
    parts = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=partition_seed,
    ).split(development, labels, groups)
    split_paths = []
    for fold_idx, (train_idx, val_idx) in enumerate(parts):
        train_ids, val_ids = development[train_idx], development[val_idx]
        _assert_no_patient_leakage(
            train_cases=groups[train_idx], val_cases=groups[val_idx],
            test_cases=slide_rows.loc[test0, "case_id"].to_numpy(), fold_idx=fold_idx,
        )
        split_paths.append(_write_split_csv(splits_dir, fold_idx, train_ids, val_ids, test0))
    print(f"  Splits: {splits_dir}  ({n_splits} development folds inside standard fold 0)")
    return split_paths


def _standard_folds(
    df: pd.DataFrame,
    n_splits: int,
    seed: int,
    stratify_col: str = "label",
):
    """Patient-stratified k-fold: dedup to cases, split cases, expand to slides.

    Yields ``(train_ids, val_ids, test_ids)`` slide arrays per fold.

    Slides from the same ``case_id`` share a stratification key — the class
    label for classification, or the event indicator (``status``) for
    survival — so we can dedup safely and run standard StratifiedKFold on
    cases. Stratifying survival splits on ``status`` keeps every fold from
    landing zero events (which would make the c-index undefined and collapse
    nllsurv's quantile bins). Avoids StratifiedGroupKFold's minimum-stratum-
    size limitation.
    """
    # L-5: a case whose slides DISAGREE on the stratification value would
    # otherwise be resolved by whichever slide `.first()` (below) happens to
    # sort first -- an arbitrary, silent choice the operator has no way to
    # detect from the output (it changes neither the row count nor the
    # schema). Assert one distinct value per case before that dedup runs.
    conflict_counts = df.groupby("case_id")[stratify_col].nunique(dropna=False)
    conflicted = conflict_counts[conflict_counts > 1].index.tolist()
    if conflicted:
        detail = {
            case_id: sorted(
                df.loc[df["case_id"] == case_id, stratify_col].unique().tolist(),
                key=str,
            )
            for case_id in conflicted[:10]
        }
        raise ValueError(
            f"{len(conflicted)} case(s) have conflicting {stratify_col!r} "
            "values across their slides -- case-level stratification "
            f"requires exactly one value per case_id: {detail}"
            + (f" ... +{len(conflicted) - 10} more" if len(conflicted) > 10 else "")
        )

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

        yield (
            _expand_cases_to_slides(train_cases, case_to_slides),
            _expand_cases_to_slides(val_cases, case_to_slides),
            _expand_cases_to_slides(test_cases, case_to_slides),
        )


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


_STRATEGIES = {
    "standard": _splits_standard_cv,
    **{
        name: partial(_splits_devcv, partition_seed=partition_seed)
        for name, partition_seed in DEVCV_PARTITION_SEEDS.items()
    },
}
