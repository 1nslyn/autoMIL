"""Shared data preparation: task CSV creation, split generation, orchestration.

Framework-specific preparation lives in each adapter:
- ``clam/prepare.py``: H5 -> PT conversion
- ``nnmil/prepare.py``: dataset.json, dataset_plan.json generation
- ``smmile/prepare.py``: H5 -> NIC conversion, superpixel generation
"""

from __future__ import annotations

import os
import re

import pandas as pd

from autobench.config import DatasetConfig

# A cached cross-validation fold file: `splits_0.csv`, `splits_1.csv`, ...
# Deliberately anchored so CLAM's `splits_<i>_bool.csv` / `splits_<i>_descriptor.csv`
# companions do not match.
_SPLIT_FILE_RE = re.compile(r"splits_\d+\.csv")
from autobench.data import load_all_slides
from autobench.pipeline.manifest_guard import check_manifest_fingerprint
from autobench.pipeline.splits import create_strategy_splits


# ---------------------------------------------------------------------------
# Task CSV generation
# ---------------------------------------------------------------------------


def create_task_csv(
    mapping_csv: str,
    output_csv: str,
    label_col: str | None = None,
    label_map: dict[int, str] | None = None,
    ds: DatasetConfig | None = None,
    *,
    task_type: str = "classification",
    event_col: str | None = None,
    time_col: str | None = None,
) -> pd.DataFrame:
    """Create a task CSV from mapping.csv.

    Classification tasks (default) map labels via ``label_map`` and emit a
    ``label`` column. Survival tasks (``task_type="survival"``) read
    ``event_col`` / ``time_col`` and emit ``status`` (0/1 event) and ``time``
    (continuous) columns — the column names nnMIL's ``UnifiedMILDataset``
    expects for survival mode.
    """
    if task_type == "survival":
        return _create_survival_task_csv(
            mapping_csv, output_csv, event_col, time_col, ds
        )

    df = load_all_slides(mapping_csv, ds)

    n_before = len(df)
    df = df.dropna(subset=[label_col]).reset_index(drop=True)
    if df.empty:
        raise ValueError(
            f"Task CSV for {label_col!r} would be empty: all {n_before} rows have a "
            f"null label. This usually means {label_col!r} was never populated in "
            f"{mapping_csv}, or a manifest join used the wrong case-id column. "
            "(An empty CSV would otherwise fail much later inside split generation.)"
        )

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
        # Just use the raw label values directly since they are already class names.
        #
        # Check membership explicitly: this branch never calls .map(), so an
        # unrecognised class name would otherwise flow straight into the splits
        # as an unannounced extra class. String-labelled datasets are precisely
        # the ones most likely to gain a new class name upstream.
        known = set(label_map.values()) if label_map else set()
        if known:
            unknown = sorted(set(df[label_col].astype(str).unique()) - known)
            if unknown:
                raise ValueError(
                    f"{len(unknown)} class name(s) in {label_col!r} are absent from "
                    f"the task's label_map: {unknown[:10]}. Add them to the dataset "
                    f"YAML's label_map (known: {sorted(known)}), or exclude them "
                    "upstream in the manifest."
                )
        task_df = pd.DataFrame({
            "case_id": df[case_col],
            "slide_id": df[slide_col].apply(ds.get_slide_id),
            "label": df[label_col],
        })

    # A value absent from label_map becomes NaN via .map() and would flow into
    # the split CSV as a silently-corrupt class. Fail instead — this is how a
    # new class (e.g. a 4th tumour grade) would otherwise slip in unnoticed.
    if task_df["label"].isna().any():
        unmapped = sorted(set(df.loc[task_df["label"].isna(), label_col].unique()))
        raise ValueError(
            f"{len(unmapped)} value(s) in {label_col!r} are absent from the task's "
            f"label_map and became NaN: {unmapped[:10]}. Add them to the dataset "
            "YAML's label_map, or exclude them upstream in the manifest."
        )

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    task_df.to_csv(output_csv, index=False)
    print(f"  Task CSV: {output_csv}  ({len(task_df)} slides, "
          f"{task_df['label'].value_counts().to_dict()})")
    return task_df


def _create_survival_task_csv(
    mapping_csv: str,
    output_csv: str,
    event_col: str | None,
    time_col: str | None,
    ds: DatasetConfig,
) -> pd.DataFrame:
    """Build a survival task CSV with ``case_id, slide_id, status, time``.

    ``status`` is the integer event indicator (1=event, 0=censored) read
    from ``event_col``; ``time`` is the continuous survival/follow-up time
    from ``time_col``. Rows missing either are dropped, as are rows with a
    non-positive ``time`` (see below).
    """
    if not event_col or not time_col:
        raise ValueError(
            "Survival task requires both event_col and time_col to be set "
            f"(got event_col={event_col!r}, time_col={time_col!r})."
        )
    df = load_all_slides(mapping_csv, ds)
    n_total = len(df)
    df = df.dropna(subset=[event_col, time_col]).reset_index(drop=True)
    n_missing = n_total - len(df)

    # Non-positive follow-up (a GDC date artifact — e.g. a death recorded at
    # day 0) is undefined for Cox's partial likelihood and breaks most c-index
    # implementations. Drop it here rather than letting it silently corrupt the
    # fit; the count is reported below so the loss is never invisible.
    times = pd.to_numeric(df[time_col], errors="coerce")
    # Count unparseable separately from non-positive: both are dropped, but
    # reporting a non-numeric time as "time <= 0" would misdescribe the cause.
    n_unparseable = int(times.isna().sum())
    keep = times > 0
    n_nonpos = int((~keep).sum()) - n_unparseable
    df = df[keep].reset_index(drop=True)
    if df.empty:
        # Mirror the classification guard. Without this a 0-row CSV is written,
        # and the *next* run's cache check reports it as "empty (header only)"
        # and blames a task-type change — the wrong cause — then tells the
        # operator to purge and re-run, which regenerates the same empty file.
        raise ValueError(
            f"Survival task CSV for {event_col!r}/{time_col!r} would be empty: no "
            f"row has both a non-null event/time and a positive time. Of {n_total} "
            f"slides: {n_missing} missing event/time, {n_nonpos} with time <= 0, "
            f"{n_unparseable} with a non-numeric time. Check that the survival "
            f"columns were joined onto {mapping_csv} correctly."
        )

    slide_col = ds.slide_id_column
    case_col = ds.case_id_column
    task_df = pd.DataFrame({
        "case_id": df[case_col],
        "slide_id": df[slide_col].apply(ds.get_slide_id),
        "status": df[event_col].astype(int),
        "time": df[time_col].astype(float),
    })

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    task_df.to_csv(output_csv, index=False)
    n_events = int(task_df["status"].sum())
    print(f"  Survival task CSV: {output_csv}  ({len(task_df)} slides, "
          f"{n_events} events, {len(task_df) - n_events} censored)")
    if n_missing or n_nonpos or n_unparseable:
        print(f"    dropped {n_missing + n_nonpos + n_unparseable} of {n_total} slides: "
              f"{n_missing} missing event/time, {n_nonpos} with time <= 0, "
              f"{n_unparseable} with a non-numeric time")
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
) -> None:
    """Run the complete data-preparation pipeline (idempotent).

    Uses task definitions from ``DatasetConfig`` instead of hardcoded values.
    """
    # M-10: the per-task schema/coverage checks below (B2, and the splits
    # fold0-vs-csv check) can tell a cached CSV doesn't match ITS OWN task's
    # shape, but neither can tell the MANIFEST it was derived from has since
    # been rebuilt with different values -- same columns, same slide_id set,
    # different content (e.g. a corrected OS date). Check this once, up
    # front, before touching any task CSV: fails loudly (does not purge) if
    # this benchmark_dir's cached artefacts came from a different manifest.
    dataset_csv_dir = os.path.join(benchmark_dir, "dataset_csv")
    check_manifest_fingerprint(dataset_csv_dir, mapping_csv)

    all_slide_ids: set[str] = set()

    # Get default strategy (first one defined in the config)
    default_strategy = list(ds.split_strategies.keys())[0]

    for task_name, tdef in ds.tasks.items():
        csv_path = os.path.join(benchmark_dir, "dataset_csv", f"{task_name}.csv")
        if not os.path.exists(csv_path):
            print(f"[prep] Creating task CSV: {task_name}")
            if tdef.task_type == "survival":
                task_df = create_task_csv(
                    mapping_csv, csv_path, ds=ds,
                    task_type="survival",
                    event_col=tdef.event_col,
                    time_col=tdef.time_col,
                )
            else:
                task_df = create_task_csv(
                    mapping_csv, csv_path,
                    label_col=tdef.label_col,
                    label_map=tdef.label_map,
                    ds=ds,
                )
        else:
            task_df = pd.read_csv(csv_path)
            # The cache key is the task NAME only, so a CSV left over from an
            # earlier roster can be reused under a task whose type has since
            # changed (e.g. `os` was a Patho-Bench classification task before it
            # became survival). Detect that and FAIL LOUDLY — deliberately do
            # NOT self-heal here.
            #
            # prepare_all runs once per EXPERIMENT against the *shared*
            # benchmark_dir (see scripts/run_experiment.py), so under the agentic
            # loop many processes execute this block concurrently. Rewriting the
            # CSV or removing the splits directory would race: a concurrent
            # reader can observe a truncated-but-line-aligned CSV and build
            # splits from a partial cohort, and a purge can delete splits another
            # process is already training from. So the operator purges explicitly.
            #
            # Note this only removes the *destructive* race. Creation itself is
            # still not atomic (`to_csv` truncates then writes), so run
            # `run_benchmark.py --dataset <cohort> --prep_only` once before
            # launching concurrent work, rather than letting 8xN experiment
            # processes race to generate the same CSVs and splits.
            expected = {"status", "time"} if tdef.task_type == "survival" else {"label"}
            stale_splits = os.path.join(
                benchmark_dir, "splits", default_strategy, task_name
            )
            if task_df.empty or not expected.issubset(task_df.columns):
                why = (
                    "it is empty (header only)" if task_df.empty
                    else f"it is missing {sorted(expected - set(task_df.columns))}"
                )
                raise ValueError(
                    f"Cached task CSV does not match task {task_name!r} "
                    f"(task_type={tdef.task_type}): {why}.\n"
                    f"  file:  {csv_path}\n"
                    f"  found: {list(task_df.columns)}\n"
                    "The task's type or labels changed since this cache was written "
                    "(e.g. a roster pivot). Purge the derived artefacts and re-run "
                    "prep — this is not done automatically because prep runs "
                    "concurrently against a shared directory:\n"
                    f"  rm -f {csv_path}\n"
                    f"  rm -rf {stale_splits}"
                )
            print(f"[prep] Task CSV already exists: {csv_path}")
        all_slide_ids.update(task_df["slide_id"].tolist())

        splits_dir = os.path.join(benchmark_dir, "splits", default_strategy, task_name)
        # Look for ANY cached fold file, not just splits_0.csv. Keying on
        # splits_0 alone leaves a hole: delete just that file (a natural way to
        # "force a regenerate") and this branch regenerates folds 0..n-1 while
        # stale higher-numbered folds survive alongside them. The immediate run
        # is fine, but the directory is left holding a mix — and a later run at
        # the *higher* fold count would match on total and silently train on
        # part-new, part-stale splits.
        # Match ONLY `splits_<int>.csv`, and only real files. A `startswith`
        # test also catches CLAM's own `splits_0_bool.csv` /
        # `splits_0_descriptor.csv` companions (lib/CLAM/create_splits_seq.py),
        # which would make a perfectly correct directory look wrong — and the
        # remedy this raises is `rm -rf`, so a false positive destroys good
        # folds. The isfile check additionally rejects a directory or symlink
        # named like a fold file, which would otherwise satisfy the set.
        cached_splits = (
            {f for f in os.listdir(splits_dir)
             if _SPLIT_FILE_RE.fullmatch(f)
             and os.path.isfile(os.path.join(splits_dir, f))
             and not os.path.islink(os.path.join(splits_dir, f))}
            if os.path.isdir(splits_dir) else set()
        )
        if not cached_splits:
            print(f"[prep] Creating splits: {task_name}")
            stratify_col = "status" if tdef.task_type == "survival" else "label"
            create_strategy_splits(
                csv_path, splits_dir,
                n_splits=n_splits,
                seed=seed, stratify_col=stratify_col,
            )
        else:
            # A cached splits directory is reused wholesale, so `n_splits` would
            # otherwise be silently ignored: a 5-fold request trains on whatever
            # fold count happens to be on disk. These directories still hold
            # legacy 10-fold splits from the phase-1 runs, which would halve the
            # validation set and change the train/val/test proportions without
            # any message. Verify the count and fail loudly — and, as with the
            # task CSV above, deliberately do NOT self-heal: this path runs
            # concurrently against a shared benchmark_dir.
            # Require the EXACT set {splits_0..n-1}. A bare count would accept a
            # directory holding the right number of files with the wrong indices
            # — e.g. folds 0-4 regenerated next to stale folds 5-9 totals 10 and
            # would pass a 10-fold request while being half stale.
            expected = {f"splits_{i}.csv" for i in range(n_splits)}
            if cached_splits != expected:
                extra = sorted(cached_splits - expected)
                missing = sorted(expected - cached_splits)
                detail = []
                if extra:
                    detail.append(f"unexpected: {extra[:6]}{'...' if len(extra) > 6 else ''}")
                if missing:
                    detail.append(f"missing: {missing[:6]}{'...' if len(missing) > 6 else ''}")
                raise ValueError(
                    f"Cached splits for task {task_name!r} do not match a "
                    f"{n_splits}-fold run ({len(cached_splits)} file(s) on disk; "
                    f"{'; '.join(detail)}).\n"
                    f"  dir: {splits_dir}\n"
                    "Reusing them would change the train/val/test proportions (and "
                    "the validation-set size that model selection depends on), or "
                    "mix freshly-generated folds with stale ones. Purge and re-run "
                    "prep:\n"
                    f"  rm -rf {splits_dir}"
                )
            # The set check above only proves the fold COUNT is right. It says
            # nothing about WHICH cohort those folds were built from — so splits
            # can be silently stale relative to their own task CSV. That is
            # reachable through the remedy the task-CSV guard prints above: it
            # removes only the CSV, which is then regenerated fresh while these
            # folds survive, leaving part-new/part-stale training with no
            # message. Verify fold 0 covers exactly the CSV's slides.
            fold0 = pd.read_csv(os.path.join(splits_dir, "splits_0.csv"))
            split_ids: set[str] = set()
            for col in ("train", "val", "test"):
                if col in fold0.columns:
                    split_ids |= set(fold0[col].dropna().astype(str))
            csv_ids = set(task_df["slide_id"].astype(str))
            if split_ids != csv_ids:
                only_splits = len(split_ids - csv_ids)
                only_csv = len(csv_ids - split_ids)
                raise ValueError(
                    f"Cached splits for task {task_name!r} were built from a "
                    f"different set of slides than its task CSV.\n"
                    f"  splits dir: {splits_dir}\n"
                    f"  task CSV:   {csv_path}\n"
                    f"  fold 0 covers {len(split_ids)} slides; the CSV has "
                    f"{len(csv_ids)} ({only_splits} only in splits, "
                    f"{only_csv} only in the CSV).\n"
                    "The CSV was regenerated after these splits (labels or the "
                    "manifest changed). Purge the splits and re-run prep:\n"
                    f"  rm -rf {splits_dir}"
                )
            print(f"[prep] Splits already exist: {splits_dir} ({n_splits}-fold)")

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
