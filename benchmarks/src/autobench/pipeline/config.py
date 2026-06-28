"""Benchmark configuration dataclasses and dynamic registries.

Registries (tasks, strategies, models) are built at runtime from a
``DatasetConfig`` rather than being hardcoded. This allows the same
benchmark code to work across different datasets.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from enum import Enum

from autobench.config import DatasetConfig


# ---------------------------------------------------------------------------
# Framework enum (universal -- not dataset-specific)
# ---------------------------------------------------------------------------


class Framework(str, Enum):
    """Model frameworks."""

    CLAM = "clam"
    NNMIL = "nnmil"


# ---------------------------------------------------------------------------
# Strategy configuration
# ---------------------------------------------------------------------------


@dataclass
class StrategyConfig:
    """Defines train/test cohort assignment for a split strategy."""

    strategy: str  # strategy name (e.g., "standard")
    train_cohorts: list[str]
    test_cohorts: list[str]


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TaskConfig:
    name: str
    label_col: str | None
    label_dict: dict[str, int] | None
    n_classes: int = 2
    task_type: str = "classification"
    event_col: str | None = None
    time_col: str | None = None
    survival_losses: list[str] = field(default_factory=lambda: ["cox"])
    nll_bins: int = 4


@dataclass
class ModelConfig:
    model_type: str  # "clam_sb", "clam_mb", "mil"
    model_size: str = "small"
    dropout: float = 0.25
    bag_weight: float = 0.7
    B: int = 8  # patches sampled for instance-level training


@dataclass
class TrainConfig:
    max_epochs: int = 200
    lr: float = 2e-4
    weight_decay: float = 1e-5
    optimizer: str = "adam"
    early_stopping: bool = True
    patience: int = 20
    stop_epoch: int = 50
    weighted_sample: bool = True
    seed: int = 42


@dataclass
class ExperimentConfig:
    task: TaskConfig
    encoder_key: str
    embed_dim: int
    model: ModelConfig
    train: TrainConfig
    n_folds: int = 10
    framework: Framework = Framework.CLAM
    strategy: str = "standard"
    # Survival loss variant (cox/mse/mae/nllsurv). None for classification —
    # kept None so classification experiment_id / results_subdir are byte-
    # identical to pre-survival behaviour (no result-dir migration).
    survival_loss: str | None = None

    @property
    def is_survival(self) -> bool:
        return self.task.task_type == "survival"

    @property
    def experiment_id(self) -> str:
        base = (
            f"{self.framework.value}__{self.strategy}"
            f"__{self.task.name}__{self.encoder_key}"
            f"__{self.model.model_type}__s{self.train.seed}"
        )
        return base if self.survival_loss is None else f"{base}__{self.survival_loss}"

    @property
    def results_subdir(self) -> str:
        """Relative results path: framework/strategy/task/encoder/model[/loss]."""
        parts = [
            self.framework.value,
            self.strategy,
            self.task.name,
            self.encoder_key,
            self.model.model_type,
        ]
        if self.survival_loss is not None:
            parts.append(self.survival_loss)
        return os.path.join(*parts)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["framework"] = self.framework.value
        return d

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class BenchmarkConfig:
    benchmark_dir: str = ""
    mapping_csv: str = ""
    features_base_dir: str = ""
    encoder_keys: list[str] = field(default_factory=list)
    model_types: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    train: TrainConfig = field(default_factory=TrainConfig)
    n_folds: int = 10
    gpu: int = 0
    wandb_project: str | None = None
    experiments_per_gpu: int | None = None
    max_tasks_per_child: int | None = 1
    strategies: list[str] = field(default_factory=list)
    frameworks: list[Framework] = field(default_factory=lambda: [Framework.CLAM])
    nnmil_model_types: list[str] = field(default_factory=list)

    @classmethod
    def from_dataset_config(cls, ds: DatasetConfig, **overrides) -> BenchmarkConfig:
        """Create a BenchmarkConfig pre-populated from a DatasetConfig."""
        defaults = {
            "benchmark_dir": ds.benchmark_dir,
            "mapping_csv": ds.mapping_csv,
            "features_base_dir": ds.features_base_dir,
            "encoder_keys": list(ds.encoder_dims.keys()),
            "model_types": ds.clam_models,
            "tasks": list(ds.tasks.keys()),
            "strategies": list(ds.split_strategies.keys()),
            "nnmil_model_types": ds.nnmil_models,
            "wandb_project": f"{ds.name}-benchmark",
        }
        defaults.update(overrides)
        return cls(**defaults)


# ---------------------------------------------------------------------------
# Dynamic registries -- built from DatasetConfig
# ---------------------------------------------------------------------------


@dataclass
class Registries:
    """All registries built from a DatasetConfig at runtime."""

    task_registry: dict[str, TaskConfig]
    model_registry: dict[str, ModelConfig]
    strategy_registry: dict[str, StrategyConfig]
    task_strategy_feasibility: dict[str, list[str]]
    encoder_dims: dict[str, int]
    nnmil_models: list[str]


def build_registries(ds: DatasetConfig) -> Registries:
    """Build all registries from a DatasetConfig."""
    # Tasks
    task_registry: dict[str, TaskConfig] = {}
    for name, tdef in ds.tasks.items():
        if tdef.task_type == "survival":
            task_registry[name] = TaskConfig(
                name=name,
                label_col=None,
                label_dict=None,
                task_type="survival",
                event_col=tdef.event_col,
                time_col=tdef.time_col,
                survival_losses=tdef.survival_losses,
                nll_bins=tdef.nll_bins,
            )
            continue
        # Invert label_map: {0: "neg", 1: "pos"} -> {"neg": 0, "pos": 1}
        label_dict = {v: k for k, v in tdef.label_map.items()}
        task_registry[name] = TaskConfig(
            name=name,
            label_col=tdef.label_col,
            label_dict=label_dict,
            n_classes=tdef.n_classes,
        )

    # Models (CLAM models are universal)
    model_registry: dict[str, ModelConfig] = {
        m: ModelConfig(model_type=m) for m in ds.clam_models
    }

    # Strategies
    strategy_registry: dict[str, StrategyConfig] = {}
    for name, sdef in ds.split_strategies.items():
        strategy_registry[name] = StrategyConfig(
            strategy=name,
            train_cohorts=sdef.train_cohorts,
            test_cohorts=sdef.test_cohorts,
        )

    return Registries(
        task_registry=task_registry,
        model_registry=model_registry,
        strategy_registry=strategy_registry,
        task_strategy_feasibility=ds.task_strategy_feasibility,
        encoder_dims=ds.encoder_dims,
        nnmil_models=ds.nnmil_models,
    )


# ---------------------------------------------------------------------------
# nnMIL runtime overrides (universal, not dataset-specific)
# ---------------------------------------------------------------------------
#
# ``NNMIL_RUNTIME_DEFAULTS`` are applied to every nnMIL trainer
# regardless of model type. ``NNMIL_MODEL_RUNTIME_OVERRIDES`` is a
# hook-point left empty by default — per-model VRAM mitigations
# (e.g., batch_size=4 for transformer-family heads) go here. Empty dict
# means "no model-specific overrides"; the planner-emitted batch_size
# stands. Restoration of the Phase-pre-Level-D values for
# vision_transformer / rrt / trans_mil / ilra_mil is documented in
# .planning/decisions/N4-overrides-removed.md (placeholder; see audit).

NNMIL_RUNTIME_DEFAULTS: dict[str, int] = {
    # Default 8 DataLoader subprocess workers; env-overridable via
    # NNMIL_NUM_WORKERS. Caveat: on TCGA-LUAD job 43324927, num_workers=2 was
    # observed to cause a ~3.7x per-epoch slowdown vs num_workers=0 (440s vs
    # 120s per epoch on identically-sized features), likely due to CUDA-aware
    # fork overhead in the spawn-context orchestrator pool. Set
    # NNMIL_NUM_WORKERS=0 if that regression reappears.
    "num_workers": int(os.environ.get("NNMIL_NUM_WORKERS", "8")),
}

NNMIL_MODEL_RUNTIME_OVERRIDES: dict[str, dict[str, int]] = {}


def get_nnmil_runtime_overrides(model_type: str) -> dict[str, int]:
    """Return fixed runtime overrides for nnMIL model type.

    Always includes ``NNMIL_RUNTIME_DEFAULTS``; layers any model-specific
    overrides from ``NNMIL_MODEL_RUNTIME_OVERRIDES`` on top. The dict
    being empty by default is intentional — see module-level comment.
    """
    overrides = dict(NNMIL_RUNTIME_DEFAULTS)
    overrides.update(NNMIL_MODEL_RUNTIME_OVERRIDES.get(model_type, {}))
    return overrides


# ---------------------------------------------------------------------------
# Experiment grid generation
# ---------------------------------------------------------------------------


# Survival CV uses fewer folds than classification. With few events, 10-fold
# leaves only ~2 events per test fold (c-index near chance), so survival follows
# nnMIL's native 5-fold convention while classification keeps the configured
# n_folds.
SURVIVAL_N_FOLDS = 5


def resolve_n_folds(task_type: str, n_folds: int) -> int:
    """Fold count for a task: survival is pinned to SURVIVAL_N_FOLDS; others use n_folds."""
    return SURVIVAL_N_FOLDS if task_type == "survival" else n_folds


def generate_all_experiments(
    cfg: BenchmarkConfig,
    registries: Registries,
) -> list[ExperimentConfig]:
    """Generate the full experiment grid using dynamic registries.

    Respects ``task_strategy_feasibility``.
    """
    experiments: list[ExperimentConfig] = []
    seen_ids: set[str] = set()

    for framework in cfg.frameworks:
        if framework == Framework.CLAM:
            model_types = [m for m in cfg.model_types if m in registries.model_registry]
        else:
            model_types = cfg.nnmil_model_types

        for strategy in cfg.strategies:
            for task_name in cfg.tasks:
                task_cfg = registries.task_registry[task_name]

                feasible = registries.task_strategy_feasibility.get(task_name, [])

                # Check feasibility (first strategy in the list is always allowed)
                first_strategy = list(registries.strategy_registry.keys())[0] if registries.strategy_registry else None
                if strategy not in feasible and strategy != first_strategy:
                    continue

                # Survival tasks fan out over their loss variants (each a
                # separate experiment); classification yields a single [None]
                # so the grid is unchanged.
                loss_values = (
                    task_cfg.survival_losses
                    if task_cfg.task_type == "survival"
                    else [None]
                )

                for encoder_key in cfg.encoder_keys:
                    for model_type in model_types:
                        model_cfg = (
                            registries.model_registry[model_type]
                            if framework == Framework.CLAM
                            else ModelConfig(model_type=model_type)
                        )
                        for survival_loss in loss_values:
                            # CLAM survival: attention models only; cox is
                            # clam_sb-only (single risk output), nllsurv works
                            # for clam_sb and clam_mb.
                            if (
                                framework == Framework.CLAM
                                and task_cfg.task_type == "survival"
                            ):
                                if model_type not in ("clam_sb", "clam_mb"):
                                    continue
                                if survival_loss not in ("cox", "nllsurv"):
                                    continue
                                if survival_loss == "cox" and model_type != "clam_sb":
                                    continue
                            exp = ExperimentConfig(
                                task=task_cfg,
                                encoder_key=encoder_key,
                                embed_dim=registries.encoder_dims[encoder_key],
                                model=model_cfg,
                                train=cfg.train,
                                n_folds=resolve_n_folds(task_cfg.task_type, cfg.n_folds),
                                framework=framework,
                                strategy=strategy,
                                survival_loss=survival_loss,
                            )
                            if exp.experiment_id not in seen_ids:
                                experiments.append(exp)
                                seen_ids.add(exp.experiment_id)

    return experiments
