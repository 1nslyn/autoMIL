"""Benchmark configuration dataclasses and dynamic registries.

Registries (tasks, strategies, models) are built at runtime from a
``DatasetConfig`` rather than being hardcoded. This allows the same
benchmark code to work across different datasets.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from enum import Enum

from autobench.config import DatasetConfig


# ---------------------------------------------------------------------------
# Framework enum (universal -- not dataset-specific)
# ---------------------------------------------------------------------------


class Framework(str, Enum):
    """Model frameworks."""

    CLAM = "clam"
    NNMIL = "nnmil"
    DTFD = "dtfd"
    TITAN = "titan"
    ABMIL = "abmil"


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
    #: Ordered classes (see autobench.config.TaskDef.ordinal). Threaded through
    #: to compute_extended_metrics so ``qwk`` is emitted for these tasks only.
    ordinal: bool = False


@dataclass
class ModelConfig:
    model_type: str  # "clam_sb", "clam_mb", "mil"
    model_size: str = "small"
    dropout: float = 0.25
    bag_weight: float = 0.7
    B: int = 8  # patches sampled for instance-level training
    # H-3b: CLAM's instance-clustering branch. These were hardcoded in
    # `_make_clam_args`, so three live upstream knobs (core_utils.py:117 bag_loss,
    # :141 inst_loss, :185 no_inst_cluster) sat outside the search space while the
    # rest of CLAM's surface sat inside it. Defaults reproduce the hardcoded
    # values exactly, so no dispatched experiment changes.
    bag_loss: str = "ce"              # "ce" | "svm"
    inst_loss: str | None = None      # None | "svm" | "ce"
    no_inst_cluster: bool = False     # True disables CLAM's instance-clustering


@dataclass
class TrainConfig:
    max_epochs: int = 200
    # 2026-07-28: returned to CLAM's own upstream default (lib/CLAM/main.py:74,
    # `--lr default=1e-4`). Was 2e-4, a 2x deviation with no recorded rationale
    # (see provenance.py). CLAM reads this field directly (clam/train.py); no
    # other arm consumes TrainConfig.lr, so this is CLAM-only in effect.
    lr: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "adam"
    early_stopping: bool = True
    patience: int = 20
    stop_epoch: int = 50
    weighted_sample: bool = True
    seed: int = 42


def _arm_as_dict(arm_cfg) -> dict | None:
    """Arm config dataclass / nnMIL plan mapping -> plain dict; ``None`` if absent."""
    if arm_cfg is None:
        return None
    if is_dataclass(arm_cfg) and not isinstance(arm_cfg, type):
        return asdict(arm_cfg)
    if isinstance(arm_cfg, dict):
        return dict(arm_cfg)
    raise TypeError(
        f"arm_cfg must be a dataclass instance or a mapping, got {type(arm_cfg).__name__}"
    )


def _train_fields_superseded_by_arm(arm: dict | None) -> list[str]:
    """Which ``train`` fields this arm's OWN config governs instead (H-3).

    ``TrainConfig`` is the shared transport, but only CLAM actually trains off
    it: DTFD, nnMIL and TITAN each carry their own ``lr``/``wd``/epoch count.
    ``config.json`` recorded the shared block regardless, so 102 of the 195
    campaign configs described a recipe that never ran -- a methods table built
    from that artifact would be fiction.

    Recording the arm block alone is not enough, because the stale ``train``
    block sits right next to it and a reader cannot tell which one governed.
    This list names, per run, exactly which ``train`` entries are superseded.
    Computed from the arm's real field names (via ``hparams.FIELD_ALIASES``, so
    DTFD's ``wd`` and nnMIL's ``learning_rate`` resolve correctly), never
    asserted from a hand-maintained table that could drift.
    """
    if not arm:
        return []
    from autobench.pipeline.hparams import FIELD_ALIASES

    return sorted(
        canonical
        for canonical, aliases in FIELD_ALIASES.items()
        if any(alias in arm for alias in aliases)
    )


@dataclass
class ExperimentConfig:
    task: TaskConfig
    encoder_key: str
    embed_dim: int
    model: ModelConfig
    train: TrainConfig
    n_folds: int = 5
    framework: Framework = Framework.CLAM
    strategy: str = "standard"
    # DATA-ID: dataset identity was recorded in NO results artifact — dataset
    # existed only as a filesystem path, so summary.json / aggregated/*.csv
    # carried task/encoder/model/framework/seed but never which cohort
    # produced them. Defaulted to "" so existing constructions (tests, ad-hoc
    # scripts) do not break.
    dataset: str = ""
    # Survival loss variant (cox/mse/mae/nllsurv). None for classification —
    # kept None so classification experiment_id / results_subdir are byte-
    # identical to pre-survival behaviour (no result-dir migration).
    survival_loss: str | None = None
    # H-3b: the opaque per-arm override channel. The shared transport above is
    # CLAM-shaped, so DTFD's `numGroup`, ABMIL's `M`/`L` and nnMIL's
    # `warmup_epochs` have no field to travel in. This dict carries them by name;
    # `hparams.apply_overrides` checks each against the arm's DECLARED search
    # space (search_space.py) and raises on anything undeclared or locked.
    # Empty by default, so a plain grid run is byte-identical to before.
    hparam_overrides: dict = field(default_factory=dict)
    # Optional registered train-only PolicyVariant. The protected trainers own
    # model/forward/loss/measurement code; this selects only the optimizer,
    # scheduler, and stopping adapter exposed by policy_dispatch.py.
    policy_variant: str | None = None
    # Stage controller fold subset. ``n_folds`` remains the immutable split
    # definition (five for the preprint); this selects which of those prepared
    # folds the current stage is allowed to train. None means all folds and is
    # byte-compatible with every static-grid run.
    fold_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.fold_indices is None:
            return
        indices = tuple(self.fold_indices)
        if not indices:
            raise ValueError("fold_indices must not be empty")
        if any(type(index) is not int for index in indices):
            raise TypeError("fold_indices must contain only integers")
        if len(set(indices)) != len(indices):
            raise ValueError("fold_indices must not contain duplicates")
        if any(index < 0 or index >= self.n_folds for index in indices):
            raise ValueError(
                f"fold_indices {indices} must lie in [0, {self.n_folds})"
            )
        self.fold_indices = indices

    @property
    def is_survival(self) -> bool:
        return self.task.task_type == "survival"

    @property
    def selected_folds(self) -> tuple[int, ...]:
        """Prepared fold indices trained by this stage, in declared order."""
        return self.fold_indices or tuple(range(self.n_folds))

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
        """Relative results path: framework/strategy/task/encoder/model[/loss]/s{seed}.

        CR-5b: the seed segment is load-bearing. Every trainer resumes a fold from
        ``<results_dir>/fold_N/metrics.json``, so without it a second seed reads
        the first seed's folds off disk and a multi-seed variance study reports
        zero variance. Seeds are meant to coexist, hence a path segment rather
        than the fingerprint guard used for the other knobs (see
        ``results_cache.py``).
        """
        parts = [
            self.framework.value,
            self.strategy,
            self.task.name,
            self.encoder_key,
            self.model.model_type,
        ]
        if self.survival_loss is not None:
            parts.append(self.survival_loss)
        parts.append(f"s{self.train.seed}")
        return os.path.join(*parts)

    def to_dict(self) -> dict:
        """The shared transport, verbatim.

        Deliberately NOT widened by ``save``'s H-3 fields: ``results_cache.
        fingerprint_payload`` is built on this, so an extra key here would change
        every stored digest and make every existing results directory raise
        ``StaleResultsCacheError`` on resume. The provenance fields belong to the
        human-facing artifact only.
        """
        d = asdict(self)
        d["framework"] = self.framework.value
        # Preserve every existing baseline fingerprint/config.json byte shape.
        # The field appears only when a source policy is actually selected.
        if self.policy_variant is None:
            d.pop("policy_variant", None)
        if self.fold_indices is None:
            d.pop("fold_indices", None)
        return d

    def save(self, path: str, arm_cfg=None) -> None:
        """Write ``config.json``: the configuration that ACTUALLY governed the run.

        H-3: the shared ``train`` block is what every arm's ``config.json``
        recorded, but only CLAM trains off it. Pass the arm's own config
        (``DTFDConfig``, ``ABMILConfig``, ``TitanHeadConfig``, or nnMIL's computed
        plan dict) and it is recorded under ``arm``, with
        ``train_fields_superseded_by_arm`` naming which ``train`` entries it
        overrides. ``arm: null`` with an empty list is CLAM's honest answer -- the
        shared block really did govern -- and is written explicitly so an absent
        arm config reads as a fact rather than as an omission.
        """
        arm = _arm_as_dict(arm_cfg)
        payload = self.to_dict()
        payload["arm"] = arm
        payload["train_fields_superseded_by_arm"] = _train_fields_superseded_by_arm(arm)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            # default=str mirrors the fingerprint sidecar: nnMIL's plan dict is
            # externally produced, and a human-facing record is better written
            # with a stringified value than not written at all.
            json.dump(payload, f, indent=2, default=str)


@dataclass
class BenchmarkConfig:
    # DATA-ID: which cohort this benchmark run belongs to (populated from
    # DatasetConfig.name in from_dataset_config below), threaded into every
    # generated ExperimentConfig so results are attributable across cohorts.
    dataset: str = ""
    benchmark_dir: str = ""
    mapping_csv: str = ""
    features_base_dir: str = ""
    encoder_keys: list[str] = field(default_factory=list)
    model_types: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    train: TrainConfig = field(default_factory=TrainConfig)
    n_folds: int = 5
    gpu: int = 0
    wandb_project: str | None = None
    experiments_per_gpu: int | None = None
    max_tasks_per_child: int | None = 1
    strategies: list[str] = field(default_factory=list)
    frameworks: list[Framework] = field(default_factory=lambda: [Framework.CLAM])
    nnmil_model_types: list[str] = field(default_factory=list)
    dtfd_model_types: list[str] = field(default_factory=list)
    abmil_model_types: list[str] = field(default_factory=list)
    # Placeholder embed_dim for the TITAN pseudo-encoder. Per design spec §7
    # this is only a grid-generation default -- the real dimension is read
    # from the extracted slide-feature file at prepare time (TRIDENT emits
    # 768-d); ``_prepare_titan_plans`` is responsible for validating/updating
    # it against the actual feature file, never hard-coding it downstream.
    titan_embed_dim: int = 768
    # H-5c: the seed axis. EMPTY means "one run at train.seed" — exactly the
    # single-seed grid, so a config that says nothing is unchanged. A non-empty
    # list turns the grid into a repeated-measures design: the SAME folds
    # (splits are cached and not seed-keyed) trained from different
    # initialisations, which isolates training stochasticity from partition
    # variance. That is the component the reported cross-fold std cannot see,
    # and it is the one C3's within-lineage lift analysis has to clear.
    #
    # Safe only because CR-5b put the seed in the results path: before that, a
    # second seed silently resumed the first seed's per-fold metrics.json and a
    # variance study would have reported exactly zero variance.
    seeds: list[int] = field(default_factory=list)

    @classmethod
    def from_dataset_config(cls, ds: DatasetConfig, **overrides) -> BenchmarkConfig:
        """Create a BenchmarkConfig pre-populated from a DatasetConfig."""
        defaults = {
            "dataset": ds.name,
            "benchmark_dir": ds.benchmark_dir,
            "mapping_csv": ds.mapping_csv,
            "features_base_dir": ds.features_base_dir,
            "encoder_keys": list(ds.encoder_dims.keys()),
            "model_types": ds.clam_models,
            "tasks": list(ds.tasks.keys()),
            "strategies": list(ds.split_strategies.keys()),
            "nnmil_model_types": ds.nnmil_models,
            "dtfd_model_types": ds.dtfd_models,
            "abmil_model_types": ds.abmil_models,
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
            ordinal=bool(getattr(tdef, "ordinal", False)),
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
# stands. The pre-Level-D values for vision_transformer / rrt / trans_mil /
# ilra_mil were removed deliberately, not lost; restore them here if a
# transformer-family head starts OOMing again.

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
        elif framework == Framework.DTFD:
            model_types = cfg.dtfd_model_types
        elif framework == Framework.TITAN:
            # TITAN *is* the encoder (a frozen slide embedding) -- there is no
            # tile-encoder or model_type axis to sweep, so both are pinned to
            # the pseudo-encoder key "titan" below (see design spec §7).
            model_types = ["titan"]
        elif framework == Framework.NNMIL:
            model_types = cfg.nnmil_model_types
        elif framework == Framework.ABMIL:
            model_types = cfg.abmil_model_types
        else:
            raise ValueError(
                f"Unknown framework in experiment generation: {framework!r}"
            )

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

                # TITAN *is* the encoder -> pin the pseudo-key "titan"; every
                # other framework sweeps the configured tile encoders.
                encoder_keys = ["titan"] if framework == Framework.TITAN else cfg.encoder_keys
                for encoder_key in encoder_keys:
                    for model_type in model_types:
                        if framework == Framework.CLAM:
                            model_cfg = registries.model_registry[model_type]
                        elif framework == Framework.TITAN:
                            model_cfg = ModelConfig(model_type="titan")
                        else:
                            model_cfg = ModelConfig(model_type=model_type)
                        embed_dim = (
                            cfg.titan_embed_dim
                            if framework == Framework.TITAN
                            else registries.encoder_dims[encoder_key]
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
                            # ABMIL/TITAN survival: only cox/nllsurv have a
                            # trainer (adapter-side, mirrors CLAM); mse/mae
                            # would otherwise silently generate an experiment
                            # that crashes at runtime. Both ABMIL variants and
                            # TITAN's single linear-probe head support either
                            # loss (arbitrary output width), so no per-model
                            # restriction is needed beyond the loss itself.
                            if (
                                framework in (Framework.ABMIL, Framework.TITAN)
                                and task_cfg.task_type == "survival"
                                and survival_loss not in ("cox", "nllsurv")
                            ):
                                continue
                            # DTFD survival: nllsurv only. Its two-tier
                            # pseudo-bag distillation repeats the slide's
                            # target across pseudo-bags -- a discrete
                            # (bin_idx, censor) pair works the same way a
                            # classification label does, but cox's
                            # partial-likelihood loss needs a cross-patient
                            # risk set that doesn't exist within one slide's
                            # own pseudo-bags. See dtfd/survival_train.py.
                            if (
                                framework == Framework.DTFD
                                and task_cfg.task_type == "survival"
                                and survival_loss != "nllsurv"
                            ):
                                continue
                            # H-5c: one experiment per seed. `cfg.seeds` empty
                            # reproduces the single-seed grid exactly.
                            for _seed in (cfg.seeds or [cfg.train.seed]):
                                _train = (
                                    cfg.train if _seed == cfg.train.seed
                                    else replace(cfg.train, seed=_seed)
                                )
                                exp = ExperimentConfig(
                                    task=task_cfg,
                                    encoder_key=encoder_key,
                                    embed_dim=embed_dim,
                                    model=model_cfg,
                                    train=_train,
                                    n_folds=cfg.n_folds,
                                    framework=framework,
                                    strategy=strategy,
                                    survival_loss=survival_loss,
                                    dataset=cfg.dataset,
                                )
                                if exp.experiment_id not in seen_ids:
                                    experiments.append(exp)
                                    seen_ids.add(exp.experiment_id)

    return experiments
