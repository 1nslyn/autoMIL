"""Dataset configuration system for multi-dataset benchmarking.

Replaces the old hardcoded config.py. All dataset-specific information
(paths, tasks, encoders, etc.) is loaded from YAML files in
the ``benchmarks/datasets/`` directory.

Paths support ``${ENV_VAR:default}`` syntax for environment-specific overrides
(e.g., HPC vs. workstation).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from autobench import BENCHMARKS_ROOT

# Directory containing dataset YAML files
DATASETS_DIR = BENCHMARKS_ROOT / "datasets"

# Regex for ${ENV_VAR} or ${ENV_VAR:default_value}
_ENV_VAR_RE = re.compile(r"\$\{([^}:]+)(?::([^}]*))?\}")


def _resolve_env_vars(value: str) -> str:
    """Resolve ``${VAR}`` and ``${VAR:default}`` references in a string."""

    def _replace(match: re.Match) -> str:
        var_name = match.group(1)
        default = match.group(2)  # None if no default provided
        env_val = os.environ.get(var_name)
        if env_val is not None:
            return env_val
        if default is not None:
            return default
        raise ValueError(
            f"Environment variable ${{{var_name}}} is not set and no default provided"
        )

    return _ENV_VAR_RE.sub(_replace, value)


def _resolve_paths(raw: dict[str, str]) -> dict[str, str]:
    """Resolve env vars and inter-field references in a paths dict.

    Resolution order per field (in declaration order):
    1. Substitute references to already-resolved fields (``${field_name}``)
    2. Resolve remaining ``${ENV_VAR}`` and ``${ENV_VAR:default}`` references
    """
    resolved: dict[str, str] = {}
    for key, val in raw.items():
        # First resolve references to already-resolved fields
        for prev_key, prev_val in resolved.items():
            val = val.replace(f"${{{prev_key}}}", prev_val)
        # Then resolve env vars for any remaining ${...} references
        val = _resolve_env_vars(val)
        resolved[key] = val
    return resolved


# ---------------------------------------------------------------------------
# Sub-config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TaskDef:
    """Definition of a task within a dataset.

    Supports two ``task_type`` values:

    - ``"classification"`` (default): uses ``label_col`` / ``label_map`` /
      ``n_classes``.
    - ``"survival"``: uses ``event_col`` (0/1 event indicator) and
      ``time_col`` (continuous survival/follow-up time). The label fields
      are unused. ``survival_losses`` lists which loss variants form the
      experiment grid (each runs as a separate experiment); ``nll_bins``
      is consulted only by the ``nllsurv`` loss.
    """

    name: str
    label_col: str | None = None
    label_map: dict[int, str] | None = None  # raw int -> class name
    n_classes: int | None = None
    task_type: str = "classification"
    event_col: str | None = None
    time_col: str | None = None
    survival_losses: list[str] = field(default_factory=lambda: ["cox"])
    nll_bins: int = 4


@dataclass
class StrategyDef:
    """Split strategy definition loaded from dataset YAML."""

    name: str
    train_cohorts: list[str]
    test_cohorts: list[str]


# ---------------------------------------------------------------------------
# Main DatasetConfig
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    """Complete configuration for a single dataset."""

    name: str
    description: str

    # Resolved paths
    data_root: str
    wsi_dir: str
    mapping_csv: str
    output_dir: str
    benchmark_dir: str
    features_base_dir: str

    # Tasks
    tasks: dict[str, TaskDef]

    # Split strategies
    split_strategies: dict[str, StrategyDef]
    task_strategy_feasibility: dict[str, list[str]]

    # Column mappings
    slide_id_column: str
    slide_id_transform: str | None  # "strip_svs" or None
    wsi_extension: str | None  # e.g. ".svs" — appended to slide IDs for file lookups
    case_id_column: str
    status_column: str | None
    status_value: str | None

    # Encoders
    encoder_models: dict[str, str]  # HF repo -> key
    encoder_dims: dict[str, int]  # key -> dimension
    nnmil_models: list[str] = field(default_factory=list)

    # Extraction params
    magnification: int = 20
    patch_size: int = 224
    batch_size: int = 64

    # CLAM model types (universal across datasets)
    clam_models: list[str] = field(
        default_factory=lambda: ["clam_sb", "clam_mb", "mil"]
    )

    def get_slide_id(self, raw_value: str) -> str:
        """Apply slide_id_transform to a raw value from the CSV."""
        if self.slide_id_transform == "strip_svs":
            return raw_value.replace(".svs", "")
        return raw_value

    def get_wsi_filename(self, raw_value: str) -> str:
        """Convert a slide ID from the CSV to the actual WSI filename on disk."""
        if self.wsi_extension and not raw_value.endswith(self.wsi_extension):
            return f"{raw_value}{self.wsi_extension}"
        return raw_value


def _parse_tasks(raw: dict[str, Any]) -> dict[str, TaskDef]:
    """Parse task definitions from YAML.

    Classification tasks require ``label_col`` / ``label_map``. Survival
    tasks (``task_type: survival``) require ``event_col`` / ``time_col``
    and may list ``survival_losses`` (default ``["cox"]``) and ``nll_bins``.
    """
    tasks = {}
    for name, tdef in raw.items():
        task_type = tdef.get("task_type", "classification")
        if task_type == "survival":
            tasks[name] = TaskDef(
                name=name,
                task_type="survival",
                event_col=tdef["event_col"],
                time_col=tdef["time_col"],
                survival_losses=tdef.get("survival_losses", ["cox"]),
                nll_bins=tdef.get("nll_bins", 4),
            )
        else:
            label_map = {int(k): v for k, v in tdef["label_map"].items()}
            tasks[name] = TaskDef(
                name=name,
                label_col=tdef["label_col"],
                label_map=label_map,
                n_classes=tdef.get("n_classes", len(label_map)),
            )
    return tasks


def _parse_strategies(raw: dict[str, Any]) -> dict[str, StrategyDef]:
    """Parse split strategy definitions from YAML."""
    strategies = {}
    for name, sdef in raw.items():
        strategies[name] = StrategyDef(
            name=name,
            train_cohorts=sdef.get("train_cohorts", []),
            test_cohorts=sdef.get("test_cohorts", []),
        )
    return strategies


def load_dataset_config(name_or_path: str) -> DatasetConfig:
    """Load a DatasetConfig from a YAML file.

    Parameters
    ----------
    name_or_path:
        Either a dataset name (e.g., ``"ovarian"``) which is looked up in
        ``benchmarks/datasets/{name}.yaml``, or a full path to a YAML file.
    """
    if os.sep in name_or_path or name_or_path.endswith(".yaml"):
        yaml_path = Path(name_or_path)
    else:
        yaml_path = DATASETS_DIR / f"{name_or_path}.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {yaml_path}\n"
            f"Available datasets: {[p.stem for p in DATASETS_DIR.glob('*.yaml')]}"
        )

    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    # Resolve paths with env var interpolation
    paths = _resolve_paths(raw.get("paths", {}))

    # Parse sub-configs
    tasks = _parse_tasks(raw.get("tasks", {}))
    strategies = _parse_strategies(raw.get("split_strategies", {}))

    # Parse feasibility
    task_feasibility = raw.get("task_strategy_feasibility", {})

    # Parse encoders
    encoders = raw.get("encoders", {})
    encoder_models = encoders.get("models", {})
    encoder_dims = encoders.get("dims", {})
    nnmil_models = raw.get("nnmil_models", [])

    # Parse extraction params
    extraction = raw.get("extraction", {})

    return DatasetConfig(
        name=raw["name"],
        description=raw.get("description", ""),
        data_root=paths.get("data_root", ""),
        wsi_dir=paths.get("wsi_dir", ""),
        mapping_csv=paths.get("mapping_csv", ""),
        output_dir=paths.get("output_dir", ""),
        benchmark_dir=paths.get("benchmark_dir", ""),
        features_base_dir=paths.get("features_base_dir", ""),
        tasks=tasks,
        split_strategies=strategies,
        task_strategy_feasibility=task_feasibility,
        slide_id_column=raw.get("slide_id_column", "slide_id"),
        slide_id_transform=raw.get("slide_id_transform"),
        wsi_extension=raw.get("wsi_extension"),
        case_id_column=raw.get("case_id_column", "case_id"),
        status_column=raw.get("status_column"),
        status_value=raw.get("status_value"),
        encoder_models=encoder_models,
        encoder_dims=encoder_dims,
        nnmil_models=nnmil_models,
        magnification=extraction.get("magnification", 20),
        patch_size=extraction.get("patch_size", 224),
        batch_size=extraction.get("batch_size", 64),
        clam_models=raw.get("clam_models", ["clam_sb", "clam_mb", "mil"]),
    )
