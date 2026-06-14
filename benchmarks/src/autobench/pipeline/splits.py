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
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    train_test_split,
)

from autobench.pipeline.config import StrategyConfig


def create_strategy_splits(
    task_csv: str,
    splits_dir: str,
    strategy_cfg: StrategyConfig | None = None,
    n_splits: int = 5,
    seed: int = 42,
    holdout_frac: float | None = None,
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
    holdout_frac:
        When ``None`` (default), produces our conservative 3-way
        train/val/test split (``_splits_standard_cv``). When set (e.g.
        ``0.30``), produces GOLDMARK-parity 2-way splits where the holdout
        fold doubles as val (model selection) AND test (reported metric) —
        see ``_splits_goldmark_parity``. This is an opt-in comparison mode,
        never the default.

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
    os.makedirs(splits_dir, exist_ok=True)

    if holdout_frac is not None:
        return _splits_goldmark_parity(df, splits_dir, n_splits, seed, holdout_frac)

    return _splits_standard_cv(df, splits_dir, n_splits, seed)


def _splits_standard_cv(
    df: pd.DataFrame,
    splits_dir: str,
    n_splits: int,
    seed: int,
) -> list[str]:
    """Patient-stratified k-fold: dedup to cases, split cases, expand to slides.

    Slides from the same ``case_id`` share a label (mutations are case-level)
    so we can dedup safely and run standard StratifiedKFold on cases.
    Avoids StratifiedGroupKFold's minimum-stratum-size limitation.
    """
    # One row per case with its label (slides of the same case share a label)
    case_table = df.groupby("case_id", sort=True)["label"].first().reset_index()
    case_ids = case_table["case_id"].values
    case_labels = case_table["label"].values

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


def _splits_goldmark_parity(
    df: pd.DataFrame,
    splits_dir: str,
    n_splits: int,
    seed: int,
    holdout_frac: float,
) -> list[str]:
    """GOLDMARK-parity 2-way patient splits (holdout doubles as val AND test).

    Reproduces GOLDMARK's *internal TCGA-CV* protocol, in which
    ``val_split_value == test_split_value == 'test'`` (their
    ``scripts/train_task_v2.py``): the single held-out fold of an N×(1-frac)/frac
    patient split is used BOTH for best-epoch checkpoint selection (val) AND as
    the reported metric (test). That makes their number a *select-on-report*
    validation figure (they self-report ~+0.039 AUROC of selection optimism),
    not a held-out test figure like our 3-way default.

    We mirror it by writing the *same* holdout slide_ids into BOTH the ``val``
    and ``test`` columns, so the existing select-on-val / report-on-test
    machinery (CLAM early-stopping checkpoint + ``test`` summary; nnMIL
    ``evaluate('test')``) lands on the identical fold with no downstream change.

    Uses ``StratifiedShuffleSplit`` (matching GOLDMARK's
    ``StratifiedShuffleSplit``, test_frac≈0.33): the N folds are independent
    random (1-frac)/frac draws, NOT a disjoint K-fold partition. Splitting is
    case-level (patient), then expanded to slides, so no patient crosses the
    train/holdout boundary.

    This path is opt-in (``--goldmark_parity``); it is a comparison instrument,
    never our headline. The conservative 3-way held-out test split
    (``_splits_standard_cv``) remains the default.
    """
    case_table = df.groupby("case_id", sort=True)["label"].first().reset_index()
    case_ids = case_table["case_id"].values
    case_labels = case_table["label"].values

    # Feasibility: StratifiedShuffleSplit needs >= 2 cases per class, and the
    # holdout must retain >= 1 case per class (so AUC is computable on it).
    label_counts = pd.Series(case_labels).value_counts().to_dict()
    min_label_count = int(min(label_counts.values()))
    holdout_per_min_class = int(round(min_label_count * holdout_frac))
    if min_label_count < 2 or holdout_per_min_class < 1:
        raise ValueError(
            f"Cannot build GOLDMARK-parity {1 - holdout_frac:.0%}/{holdout_frac:.0%} "
            f"splits: smallest class has {min_label_count} cases, which would put "
            f"{holdout_per_min_class} in the holdout fold (need >= 1 per class on "
            f"both sides). Per-class case counts (case_id-deduplicated): "
            f"{label_counts}. Lower holdout_frac or use a larger cohort."
        )

    case_to_slides = df.groupby("case_id")["slide_id"].apply(list).to_dict()

    splitter = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=holdout_frac, random_state=seed
    )
    split_paths: list[str] = []

    for fold_idx, (train_case_idx, holdout_case_idx) in enumerate(
        splitter.split(case_ids, case_labels)
    ):
        train_cases = case_ids[train_case_idx]
        holdout_cases = case_ids[holdout_case_idx]

        _assert_no_patient_leakage_2way(
            train_cases=np.asarray(train_cases),
            holdout_cases=np.asarray(holdout_cases),
            fold_idx=fold_idx,
        )

        train_ids = _expand_cases_to_slides(train_cases, case_to_slides)
        holdout_ids = _expand_cases_to_slides(holdout_cases, case_to_slides)

        # val == test == holdout: GOLDMARK's select-on-report behavior.
        path = _write_split_csv(
            splits_dir, fold_idx, train_ids, holdout_ids, holdout_ids
        )
        split_paths.append(path)

    print(
        f"  Splits: {splits_dir}  ({n_splits} folds, GOLDMARK-parity "
        f"{int(round((1 - holdout_frac) * 100))}/{int(round(holdout_frac * 100))} "
        f"patient split, val==test [comparison-only])"
    )
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


def _assert_no_patient_leakage_2way(
    train_cases: np.ndarray,
    holdout_cases: np.ndarray,
    fold_idx: int,
) -> None:
    """Hard fail if any case_id crosses the train/holdout boundary.

    The GOLDMARK-parity split is 2-way (val==test==holdout), so the only
    leakage that matters is train vs holdout; val∩test overlap is intentional.
    """
    overlap = set(train_cases.tolist()) & set(holdout_cases.tolist())
    if overlap:
        raise AssertionError(
            f"Patient leakage in GOLDMARK-parity fold {fold_idx}: "
            f"train∩holdout={overlap}"
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
