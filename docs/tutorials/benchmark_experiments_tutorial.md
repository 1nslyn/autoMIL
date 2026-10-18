# Benchmark Experiments Tutorial

How to run the autobench MIL benchmark on one dataset: prepare labels, run the experiment grid with `run_benchmark.py` or `submit_benchmark.sh`, and read the results. It follows the [TCGA](tcga_feature_extraction_tutorial.md) or [CPTAC](cptac_feature_extraction_tutorial.md) feature extraction tutorial. The agentic campaign has its own guide: [run_agentic_campaign.md](run_agentic_campaign.md).

## What gets trained

One benchmark run trains every combination of task, tile encoder and model for one dataset, with 5-fold patient-stratified cross-validation (all slides of one case stay in the same partition).

- **Encoders:** `virchow2`, `hoptimus1`, `uni_v2` (set in the dataset YAML under `encoders`).
- **Frameworks and models:** CLAM `clam_mb`, nnMIL `simple_mil`, DTFD `dtfd_mil`, ABMIL `abmil`. Each dataset YAML pins one model per framework in `clam_models`, `nnmil_models`, `dtfd_models`, `abmil_models`.
- **Tasks:** one classification task plus the shared `os` (overall survival) task. The three roster datasets are `tcga_luad` (KRAS, binary), `tcga_hnsc` (tumor grade, 3-class) and `cptac_pdac` (immune class, 3-class).
- **TITAN:** a separate slide-level arm. It has no encoder sweep, so it adds one experiment per task.

For a roster dataset this gives 3 encoders x 4 models x 2 tasks = **24 tile experiments**, plus **2 TITAN experiments**, 26 in total. The roster YAMLs list only `nllsurv` under `survival_losses`, so `os` adds no extra loss variants. A dataset YAML that lists several survival losses produces one experiment per loss. Each experiment trains 5 folds.

The roster used by the campaign is 3 cohorts and 78 cells (3 x 26); see [the campaign README](../../benchmarks/campaigns/preprint_130/README.md).

## Prerequisites

1. Feature extraction is done: `.h5` files in `{dataset_root}/trident_output/20x_224px_0px_overlap/features_{encoder}/`.
2. A dataset YAML exists in `benchmarks/datasets/{tcga,cptac,other,templates}/`. The loader finds it by name (`--dataset tcga_luad`), so the subfolder does not matter.
3. `benchmarks/.env` sets the dataset root variable (for example `AUTOBENCH_TCGA_LUAD_ROOT`). Copy `benchmarks/.env.example` and fill in your paths.
4. The packages are installed: `uv sync --all-packages`.
5. The manifest has the label columns the YAML names, plus `OS_event` and `OS_time` for the `os` task. See [Prepare labels](#prepare-labels).

For TCGA datasets the YAML needs `slide_id_transform: "strip_svs"`. The manifest slide names carry the `.svs` extension and the feature files do not, so without it the H5 to PT conversion finds no files.

Check the setup:

```bash
cd ~/scratch/autoMIL
set -a && source benchmarks/.env && set +a

uv run python -c "
from autobench.config import load_dataset_config
ds = load_dataset_config('tcga_luad')
print(ds.name, list(ds.tasks), list(ds.encoder_dims))
"
ls ${AUTOBENCH_TCGA_LUAD_ROOT}/trident_output/20x_224px_0px_overlap/features_virchow2/*.h5 | wc -l
```

## Prepare labels

Each dataset's `normalized_manifest.csv` needs the classification label column the YAML names (`label_col`) and the survival columns `OS_event` and `OS_time`. The three roster datasets already have both. Use the recipe below to rebuild a manifest or add a new cohort.

| Dataset | Label column | Source |
|---------|--------------|--------|
| `tcga_luad` | `KRAS_binary` | GOLDMARK manifest |
| `tcga_hnsc` | `GRADE` (0/1/2) | `add_grade_to_manifest.py` from the GDC `clinical.tsv` |
| `cptac_pdac` | `IMMUNE_CLASS_binary` (0/1/2) | `prepare_cptac_manifest.py` from Patho-Bench |

Survival labels come from the GDC clinical export for every cohort, so all cohorts use one source. For CPTAC cohorts without a `clinical.tsv`, `fetch_gdc_clinical.py` pulls the fields from the public GDC API.

```bash
ROOT=/path/to/dataset/root     # the directory holding normalized_manifest.csv
cp $ROOT/normalized_manifest.csv $ROOT/normalized_manifest.csv.bak.$(date +%Y%m%d)   # the scripts overwrite in place

# TCGA cohort with a GDC clinical.tsv: grade (HNSC only) and survival
uv run --package autobench python benchmarks/scripts/manifests/add_grade_to_manifest.py \
    --manifest $ROOT/normalized_manifest.csv --clinical $ROOT/clinical.tsv --case-col sample_names
uv run --package autobench python benchmarks/scripts/manifests/add_os_to_manifest.py \
    --manifest $ROOT/normalized_manifest.csv --clinical $ROOT/clinical.tsv --case-col sample_names

# CPTAC cohort: classification labels from Patho-Bench, then survival from GDC
uv run --package autobench python benchmarks/scripts/manifests/prepare_cptac_manifest.py \
    --source cptac_pda --tasks Immune_class --saveto $ROOT
uv run --package autobench python benchmarks/scripts/manifests/fetch_gdc_clinical.py \
    --manifest $ROOT/normalized_manifest.csv --case-col case_id --output $ROOT/clinical.tsv
uv run --package autobench python benchmarks/scripts/manifests/add_os_to_manifest.py \
    --manifest $ROOT/normalized_manifest.csv --clinical $ROOT/clinical.tsv --case-col case_id
```

Run each script with `--help` for its options. `--case-col` names the manifest column that holds the GDC case ID: `sample_names` for TCGA manifests and `case_id` for CPTAC manifests. `add_grade_to_manifest.py` maps G1/G2/G3 to 0/1/2 and leaves GX and G4 empty, so those cases drop out of the grade task (TCGA-HNSC keeps 414 of 431 cases).

To get a clinical export by hand, open the cohort's GDC page (for CPTAC, follow the GDC link from its TCIA page), choose **Clinical, TSV**, unzip the download and keep `clinical.tsv` in the dataset folder.

Cached task CSVs and splits do not notice that a manifest changed. After rebuilding a manifest, delete `dataset_csv/` and `splits/` under the benchmark directory (never `features/` or `results/`) so the next run regenerates them.

## Run the benchmark

Data preparation (task CSVs, patient-stratified splits, H5 to PT conversion) runs automatically at the start of every run and skips files that already exist.

### Single experiment, interactive

Use this to test a dataset before a full run:

```bash
uv run python benchmarks/scripts/run_benchmark.py \
    --dataset tcga_luad --gpu 0 \
    --frameworks clam --encoders hoptimus1 --models clam_mb --tasks kras \
    --no_wandb
```

Without `--benchmark_dir`, results go to the `benchmark_dir` in the dataset YAML (`{data_root}/benchmark`). `uv run python benchmarks/scripts/run_benchmark.py --help` lists every option. The ones used most often:

| Option | Purpose |
|--------|---------|
| `--frameworks` | Any of `clam nnmil dtfd abmil titan` (default `clam`) |
| `--encoders`, `--tasks` | Restrict the grid (default: everything in the YAML) |
| `--models`, `--nnmil_models`, `--dtfd_models`, `--abmil_models` | Restrict the models of one framework |
| `--n_folds` | Number of CV folds (default 5) |
| `--all_gpus`, `--gpus 0 1` | Run several experiments at once across GPUs |
| `--prep_only` | Run data preparation and stop |

The training defaults are in `TrainConfig` in `benchmarks/src/autobench/pipeline/config.py`: learning rate 1e-4 (CLAM's own default, and CLAM is the only arm that reads it), 200 maximum epochs, early stopping with patience 20 after epoch 50, class-weighted sampling, seed 42.

### SLURM, full grid

`submit_benchmark.sh` runs all four tile frameworks on 4 H100 GPUs for one dataset. It takes the dataset and an optional fold count as positional arguments. Submit from the repo root, since the script locates the checkout through `SLURM_SUBMIT_DIR`.

```bash
mkdir -p logs
sbatch benchmarks/scripts/slurm/submit_benchmark.sh tcga_luad        # 5-fold
sbatch benchmarks/scripts/slurm/submit_benchmark.sh tcga_luad 10     # 10-fold comparison run

# Run a subset of the four frameworks
FRAMEWORKS="clam nnmil" sbatch benchmarks/scripts/slurm/submit_benchmark.sh tcga_luad

# Use a different SLURM account
sbatch --account=YOUR_ACCOUNT benchmarks/scripts/slurm/submit_benchmark.sh tcga_luad
```

The script always runs the dataset YAML's full encoder, task and model lists. To run a narrower slice, call `run_benchmark.py` directly. Check progress with:

```bash
squeue -u $USER
tail -f logs/bench_mil_bench_*.out
```

The script validates the config and prints the experiment count, runs data preparation, then trains with memory-aware scheduling across the 4 GPUs. Results go to `<Pathology root>/autoMIL/phase2/<dataset>/benchmark_<n>fold/`, where `<Pathology root>` is two levels above the dataset root. The job resubmits itself shortly before the 24 hour wall, and finished experiments are skipped, so you can also rerun the same command at any time.

### TITAN arm

TITAN uses CONCH v1.5 features at 20x and 512 px, so it needs its own extraction. Extract once per dataset (the features do not depend on the fold count), then train:

```bash
EXTRACT=$(sbatch --parsable benchmarks/scripts/slurm/submit_titan_extract.sh tcga_luad)
sbatch --dependency=afterok:$EXTRACT benchmarks/scripts/slurm/submit_titan.sh tcga_luad
```

TITAN is a slide-level model with its own aggregation, so its rows are not a same-aggregator comparison with the tile-encoder rows.

## Survival (`os`) task

The `os` task trains a time-to-event model on overall survival. It runs in the same grid as the classification task, so the commands above already include it. Use `--tasks os` to run it alone.

**Labels.** `OS_event` is 1 for a death and 0 for a censored patient. `OS_time` is days to death or to last follow-up. Patients missing either value are dropped from the task. The task CSV has one row per slide with `case_id`, `slide_id`, `status` and `time`, and splits are stratified on `status`.

**Losses.** `survival_losses` in the YAML lists the loss variants, each of which becomes its own experiment. The roster YAMLs use `nllsurv` only: a discrete-time negative log-likelihood with `nll_bins: 4` time bins whose edges come from event (uncensored) times only. `cox` is the Cox partial likelihood and is also available. The default when a task lists nothing is `["cox"]`.

**Which frameworks run survival.** All five: CLAM, nnMIL, DTFD, ABMIL and TITAN. The grid generator keeps these loss and model combinations:

| Framework | `cox` | `nllsurv` |
|-----------|:-----:|:---------:|
| CLAM | `clam_sb` only | `clam_sb`, `clam_mb` |
| DTFD | no | yes |
| nnMIL, ABMIL, TITAN | yes | yes |

**Metric.** The c-index is computed per patient (slide risk scores are averaged within a case) with `scikit-survival`'s `concordance_index_censored`. 0.5 is chance. A fold with no comparable pairs gives NaN, and the aggregation drops NaN folds instead of counting them as 0. Classification metrics do not apply, and the `auc_roc` and accuracy columns are NaN for `os` rows.

**Checkpoint selection.** Every arm keeps the epoch with the highest in-fold validation c-index. A tie keeps the earlier epoch, and an undefined c-index never selects.

**Caveats.**
- Few events make the per-fold c-index noisy. With roughly two events in a validation fold, a per-fold value is close to random, and fold-to-fold standard deviations of 0.15 to 0.25 are normal on small cohorts. Keep 5 folds: 10 folds halve the events per fold. The summary also carries a `val_pooled` block with the c-index computed once over the pooled validation risks of all folds, which is more stable than any single fold.
- `scikit-survival` must be importable. Check with `uv run --package autobench python -c "import sksurv.metrics"`; `uv sync --all-packages` installs it.

## Results

```
{benchmark_dir}/
├── dataset_csv/<task>.csv                 # task CSVs
├── splits/standard/<task>/splits_0..4.csv
├── features/<encoder>/pt_files/           # .pt tensors converted from .h5
├── results/
│   ├── _completed.json, _failed.json
│   └── <framework>/standard/<task>/<encoder>/<model>[/<loss>]/s<seed>/
│       ├── config.json
│       ├── summary.json                   # mean, std, 95% CI across folds
│       └── fold_<i>/metrics.json, predictions.csv   # predictions.csv: classification tasks
└── aggregated/<framework>/standard.csv    # one row per experiment
```

`<loss>` appears for survival tasks only. `summary.json` holds `test` and `val` blocks, each metric with `mean`, `std`, `ci_low` and `ci_high`, plus the per-fold values (`per_fold_test`, `per_fold_val`). Classification metrics are `auc_roc`, `accuracy`, `balanced_accuracy` and `f1` (binary tasks also get `sensitivity` and `specificity`, and the ordinal HNSC grade task adds `qwk`). Survival experiments report `c_index`. The aggregated CSV has a `survival_loss` column and columns like `test_auc_roc_mean` or `test_c_index_mean`.

Rank experiments from the aggregated tables:

```bash
uv run --package autobench python -c "
import glob, pandas as pd
df = pd.concat(pd.read_csv(p) for p in glob.glob('${AUTOBENCH_TCGA_LUAD_ROOT}/benchmark/aggregated/*/standard.csv'))
col = 'test_auc_roc_mean'
print(df.sort_values(col, ascending=False)[['framework','task','encoder','model_type',col]].head(10))
"
```

Failed experiments are recorded in `results/_failed.json`.

## Resuming and troubleshooting

- **Resume.** Completed experiments are listed in `_completed.json` and skipped. Finished folds inside a partly done experiment are skipped too. Resubmit the same command.
- **Rerun after a code or label change.** Existing results count as finished, so move them aside first. If prep stops with a message that cached splits or a task CSV do not match, delete `dataset_csv/` and `splits/` as the message says and rerun.
- **CUDA out of memory.** The scheduler retries with a larger memory estimate. Any other worker exception aborts the whole run: read `logs/bench_*.err` and `_failed.json`, fix the cause and resubmit.
- **Missing feature files.** The run continues with the slides that have features. Compare `ls features_{encoder}/*.h5 | wc -l` with the slide count and check `trident_output/skipped_slides.txt`; see the feature extraction tutorial's troubleshooting section.
- **Config errors.** `Dataset config not found` means a wrong name. `Environment variable ... is not set` means `benchmarks/.env` is missing or not sourced (`set -a && source benchmarks/.env && set +a`). A `KeyError` on a column name means `slide_id_column` or `case_id_column` in the YAML does not match the manifest.
- **Weights & Biases.** The SLURM script disables it. For interactive runs, pass `--no_wandb` or set `WANDB_API_KEY` in `benchmarks/.env`.
