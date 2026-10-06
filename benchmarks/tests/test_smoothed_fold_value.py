"""Protocol v5: every arm records its fold's smoothed selection value.

A fold is scored on the validation curve around its restored epoch: the mean of
the five evaluated epochs centred on it (``smoothed_selection_value``). Each arm
prints that score on a ``[smoothed]`` line straight after its ``[selected]``
line, returns it in the fold's validation metrics (``auc_roc_smooth`` for
classification, ``c_index_smooth`` for survival) and persists it through the
fold's ``metrics.json``, which is also the resume path. Nothing reads the key yet.

Every arm runs for real -- its own training loop, through its runner or its
nnMIL adapter -- with only the per-epoch validation score scripted, so the curve
is known and is not flat. A flat curve would let a wrong epoch or a wrong window
return the right number.
"""
from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import autobench.pipeline.nnmil._imports as nnmil_imports  # noqa: E402  (puts lib/nnMIL on sys.path)
from autobench.pipeline.config import (  # noqa: E402
    ExperimentConfig,
    Framework,
    ModelConfig,
    TaskConfig,
    TrainConfig,
    build_registries,
)
from autobench.pipeline.policy_dispatch import PolicyRuntime  # noqa: E402
from autobench.pipeline.selection import smoothed_selection_value  # noqa: E402
from tests.test_abmil_arm import (  # noqa: E402
    _build_benchmark_fixture as _abmil_fixture,
    _exp_cfg as _abmil_exp_cfg,
    _smoke_cfg as _abmil_smoke_cfg,
    make_test_ds,
)
from tests.test_checkpoint_selection import (  # noqa: E402
    _MeanBagModel,
    _clam_fixture,
    _validate_clam_standin,
)
from tests.test_dtfd_arm import (  # noqa: E402
    _build_benchmark_fixture as _dtfd_fixture,
    _build_survival_fixture,
    _exp_cfg as _dtfd_exp_cfg,
    _smoke_cfg as _dtfd_smoke_cfg,
    _survival_exp_cfg,
)

NAN = float("nan")

#: The scripted validation score at each evaluated epoch. Its maximum is interior
#: (index 4), so the window is indices 2-6 and its mean, 0.728, is neither that
#: maximum nor the mean of the whole curve.
CURVE = [0.55, 0.60, 0.72, 0.70, 0.95, 0.66, 0.61, 0.63]
BEST = 4

_EPOCH = re.compile(r"^\[epoch (\d+)\](?: (.*))?$")
_SELECTED = re.compile(r"^\[selected\] epoch=(-?\d+) source=(best|final|untrained)$")
_SMOOTHED = re.compile(
    r"^\[smoothed\] epoch=(-?\d+) (val_auc_smooth|val_c_index_smooth)=(\S+)$"
)

_Y_TRUE = np.array([0, 1])
_Y_PROBS = np.array([[0.7, 0.3], [0.2, 0.8]])
_FINAL_CLS = {
    "auc_roc": 0.5, "accuracy": 0.5, "balanced_accuracy": 0.5, "f1": 0.5,
    "sensitivity": 0.5, "specificity": 0.5,
}


# ---------------------------------------------------------------------------
# Reading what a fold printed
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Printed:
    curve: dict[int, float]  # the metric at each ``[epoch k]`` line
    selected: tuple[int, ...]  # k of every ``[selected]`` line
    smoothed_at: tuple[int, ...]  # line numbers of every ``[smoothed]`` line
    selected_at: tuple[int, ...]  # line numbers of every ``[selected]`` line
    lines: tuple[str, ...]


def read_log(stdout: str, metric: str) -> Printed:
    lines = tuple(stdout.splitlines())
    curve: dict[int, float] = {}
    for line in lines:
        match = _EPOCH.match(line)
        if match:
            fields = dict(token.partition("=")[::2] for token in (match.group(2) or "").split())
            curve[int(match.group(1))] = float(fields[metric])
    selected_at = tuple(i for i, line in enumerate(lines) if _SELECTED.match(line))
    return Printed(
        curve=curve,
        selected=tuple(int(_SELECTED.match(lines[i]).group(1)) for i in selected_at),
        smoothed_at=tuple(i for i, line in enumerate(lines) if line.startswith("[smoothed]")),
        selected_at=selected_at,
        lines=lines,
    )


def window_mean(values: list[float], best: int) -> float:
    """The mean of five scripted values centred on index ``best``, shifted inward at the ends."""
    start = max(0, min(best - 2, len(values) - 5))
    window = values[start:start + 5]
    return math.fsum(window) / len(window)


# ---------------------------------------------------------------------------
# One fold per arm, run twice over the same results directory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldRun:
    returned: dict  # the validation metrics the fold returned
    stdout: str  # everything the first run printed
    persisted: dict  # the validation metrics in the fold's metrics.json
    resumed: dict  # the validation metrics the second run returned


def _execute(run: Callable[[], dict], metrics_path: str) -> FoldRun:
    """Run a fold, then run it again so the second call takes the resume branch."""
    grad_enabled = torch.is_grad_enabled()  # the nnMIL survival trainers switch it off
    printed = io.StringIO()
    try:
        with contextlib.redirect_stdout(printed):
            returned = run()
        with contextlib.redirect_stdout(io.StringIO()):
            resumed = run()
    finally:
        torch.set_grad_enabled(grad_enabled)
    with open(metrics_path) as f:
        persisted = json.load(f)["val_metrics"]
    return FoldRun(returned, printed.getvalue(), persisted, resumed)


def _scripted(values: list[float]):
    """Per-epoch scores in order, then 0.5 for the final val and test scoring."""
    scores = iter(values)
    return lambda *args, **kwargs: next(scores, 0.5)


def _abmil(values, root, mp):
    from autobench.pipeline.abmil import train as train_mod
    from autobench.pipeline.abmil.runner import run_abmil_experiment

    scores = iter(values)

    def evaluate(*args, **kwargs):
        if not kwargs.get("return_probs"):
            return dict(_FINAL_CLS)
        return {**_FINAL_CLS, "auc_roc": next(scores)}, _Y_TRUE, _Y_PROBS

    mp.setattr(train_mod, "_evaluate", evaluate)
    _abmil_fixture(str(root), n_folds=1)
    exp = _abmil_exp_cfg(build_registries(make_test_ds()), n_folds=1)
    cfg = dataclasses.replace(_abmil_smoke_cfg(), max_epochs=len(values))
    return _execute(
        lambda: run_abmil_experiment(exp, str(root), device="cpu", cfg=cfg)["per_fold_val"][0],
        os.path.join(root, "results", exp.results_subdir, "fold_0", "metrics.json"),
    )


def _dtfd(values, root, mp):
    from autobench.pipeline.dtfd import train as train_mod
    from autobench.pipeline.dtfd.runner import run_dtfd_experiment

    scores = iter(values)
    mp.setattr(train_mod, "val_scores", lambda *args, **kwargs: (next(scores), 0.5))
    _dtfd_fixture(str(root), n_folds=1)
    exp = _dtfd_exp_cfg(build_registries(make_test_ds()), n_folds=1)
    cfg = dataclasses.replace(_dtfd_smoke_cfg(), max_epochs=len(values))
    return _execute(
        lambda: run_dtfd_experiment(exp, str(root), device="cpu", cfg=cfg)["per_fold_val"][0],
        os.path.join(root, "results", exp.results_subdir, "fold_0", "metrics.json"),
    )


def _titan(values, root, mp):
    from autobench.pipeline.titan import train as train_mod

    class Slides(torch.utils.data.Dataset):
        def __init__(self, n):
            self.x = torch.randn(n, 768)
            self.y = torch.tensor([i % 2 for i in range(n)])

        def __len__(self):
            return len(self.x)

        def __getitem__(self, i):
            return self.x[i], self.y[i]

    scores = iter(values)
    real = train_mod._evaluate

    def evaluate(*args, **kwargs):
        if not kwargs.get("return_probs"):
            return real(*args, **kwargs)
        return {"auc_roc": next(scores)}, _Y_TRUE, _Y_PROBS

    mp.setattr(train_mod, "_evaluate", evaluate)
    exp = ExperimentConfig(
        task=TaskConfig(name="t", label_col="y", label_dict={"a": 0, "b": 1}, n_classes=2),
        encoder_key="titan", embed_dim=768, model=ModelConfig(model_type="titan"),
        train=TrainConfig(max_epochs=len(values), patience=5, seed=0, early_stopping=False),
        n_folds=1, framework=Framework.TITAN, strategy="standard",
    )
    results_dir = str(root / "results")
    return _execute(
        lambda: train_mod.train_titan_fold(
            exp, Slides(6), Slides(2), Slides(2), fold=0, results_dir=results_dir, device="cpu",
        )["val_metrics"],
        os.path.join(results_dir, "fold_0", "metrics.json"),
    )


def _clam(values, root, mp):
    import utils.core_utils as core_utils
    from autobench.pipeline.clam.train import train_fold

    mp.setattr(core_utils, "device", torch.device("cpu"))
    mp.setattr(core_utils, "validate_clam", _validate_clam_standin(values))
    exp, (train, val, test) = _clam_fixture(root, max_epochs=len(values))
    results_dir = str(root / "results")
    return _execute(
        lambda: train_fold(
            exp, train, val, test, fold=0, results_dir=results_dir,
            device=torch.device("cpu"), policy_runtime=PolicyRuntime(),
        )["val_metrics"],
        os.path.join(results_dir, "fold_0", "metrics.json"),
    )


def _nnmil_batches(kind):
    if kind == "classification":
        return [(torch.randn(2, 5, 8), torch.zeros(2, 5, 2), torch.tensor([5, 5]),
                 torch.tensor([0, 1]))], 2
    if kind == "cox":
        return [(torch.randn(4, 5, 8), torch.zeros(4, 5, 2), torch.tensor([5] * 4),
                 torch.tensor([1.0, 0.0, 1.0, 0.0]), torch.tensor([100.0, 200.0, 300.0, 400.0]),
                 ["p0", "p1", "p2", "p3"], ["s0", "s1", "s2", "s3"])], 1
    return [(torch.randn(1, 5, 8), torch.zeros(1, 5, 2), torch.tensor([5]),
             torch.tensor([1.0]), torch.tensor([150.0]), ["p0"], ["s0"])], 4


def _nnmil(kind):
    """The vendored nnMIL trainer for ``kind`` behind the real ``train_nnmil_fold`` adapter.

    The subclass skips only what needs the plan's data (construction, loaders, the
    model) and scripts what ``evaluate`` reports; ``train()`` and the adapter are the
    production code. The first ``len(rows)`` evaluations are the training epochs.
    """
    from autobench.pipeline.nnmil.train import train_nnmil_fold

    base_name, plan = {
        "classification": ("ClassificationTrainer", {"task_type": "classification"}),
        "cox": ("SurvivalTrainer", {"task_type": "survival", "survival_loss": "cox"}),
        "nllsurv": ("SurvivalPorpoiseTrainer",
                    {"task_type": "survival", "survival_loss": "nllsurv", "nll_bins": 4}),
    }[kind]

    def drive(values, root, mp):
        rows = [
            {"val/loss": 0.5, "val/bacc": 0.5, "val/weighted_f1": 0.5, "val/auroc": v}
            if kind == "classification" else {"val_c_index": v}
            for v in values
        ]
        # The cox/mse/mae trainer skips validation for its first two epochs.
        epochs = len(values) + (2 if kind == "cox" else 0)
        batches, n_out = _nnmil_batches(kind)
        base = getattr(nnmil_imports, base_name)

        class Scripted(base):
            def __init__(self, plan_path, model_type, fold, save_dir, seed, **kwargs):
                self.plan_path, self.model_type, self.fold = plan_path, model_type, fold
                self.save_dir, self.seed = save_dir, seed
                self.config = {"num_epochs": epochs, "warmup_epochs": 0,
                               "patience": epochs + 1, "learning_rate": 1e-3}
                self.survival_loss = kwargs.get("survival_loss")
                self.nll_bin_edges = torch.tensor([0.0, 100.0, 200.0, 300.0, 1e9])
                self.device = torch.device("cpu")
                self.logger = logging.getLogger("smoothed-fold-value")
                self.writer = None
                self.dataset_info = {}
                self.num_classes = 2
                self.model = self.train_loader = self.val_loader = self.test_loader = None
                self.policy_runtime = None
                self.save_training_config = lambda: None
                self.evaluations = 0

            def create_model(self):
                torch.manual_seed(0)
                self.model = _MeanBagModel(n_out=n_out)

            def create_data_loaders(self):
                self.train_loader = self.val_loader = self.test_loader = batches

            def evaluate(self, split="val"):
                row = rows[min(self.evaluations, len(rows) - 1)]
                self.evaluations += 1
                return dict(row)

            def _compute_val_loss(self, loss_fn):
                return 0.5

        mp.setattr(nnmil_imports, base_name, Scripted)
        plan_path = str(root / "dataset_plan.json")
        with open(plan_path, "w") as f:
            json.dump(plan, f)
        exp = ExperimentConfig(
            task=TaskConfig(name="t", label_col="y", label_dict={"a": 0, "b": 1}),
            encoder_key="e", embed_dim=8, model=ModelConfig(model_type="simple_mil"),
            train=TrainConfig(seed=1), n_folds=1, framework=Framework.NNMIL, strategy="standard",
        )
        results_dir = str(root / "results")
        return _execute(
            lambda: train_nnmil_fold(exp, plan_path, 0, results_dir, device="cpu")["val_metrics"],
            os.path.join(results_dir, "fold_0", "metrics.json"),
        )

    return drive


def _abmil_survival(values, root, mp):
    from autobench.pipeline.abmil import survival_train as train_mod
    from autobench.pipeline.abmil.config import ABMILConfig
    from autobench.pipeline.abmil.runner import run_abmil_experiment

    mp.setattr(train_mod, "survival_c_index", _scripted(values))
    _build_survival_fixture(str(root), n_folds=1, survival_loss="cox")
    exp = dataclasses.replace(
        _survival_exp_cfg(n_folds=1, survival_loss="cox"),
        model=ModelConfig(model_type="abmil"), framework=Framework.ABMIL,
    )
    cfg = ABMILConfig(M=16, L=8, max_epochs=len(values), early_stopping=False)
    return _execute(
        lambda: run_abmil_experiment(exp, str(root), device="cpu", cfg=cfg)["per_fold_val"][0],
        os.path.join(root, "results", exp.results_subdir, "fold_0", "metrics.json"),
    )


def _dtfd_survival(values, root, mp):
    from autobench.pipeline.dtfd import survival_train as train_mod
    from autobench.pipeline.dtfd.runner import run_dtfd_experiment

    mp.setattr(train_mod, "_c_index", _scripted(values))
    _build_survival_fixture(str(root), n_folds=1)
    exp = _survival_exp_cfg(n_folds=1)
    cfg = dataclasses.replace(_dtfd_smoke_cfg(), max_epochs=len(values))
    return _execute(
        lambda: run_dtfd_experiment(exp, str(root), device="cpu", cfg=cfg)["per_fold_val"][0],
        os.path.join(root, "results", exp.results_subdir, "fold_0", "metrics.json"),
    )


def _titan_survival(values, root, mp):
    from autobench.pipeline.titan import survival_train as train_mod

    class Patients(torch.utils.data.Dataset):
        def __init__(self, n):
            self.x = torch.randn(n, 768)
            self.status = torch.tensor([i % 2 for i in range(n)])
            self.time = torch.tensor([100.0 + 50 * i for i in range(n)])

        def __len__(self):
            return len(self.x)

        def __getitem__(self, i):
            return self.x[i], self.status[i], self.time[i], f"P{i}"

    mp.setattr(train_mod, "survival_c_index", _scripted(values))
    exp = ExperimentConfig(
        task=TaskConfig(name="os", label_col="status", label_dict={}, n_classes=2,
                        task_type="survival"),
        encoder_key="titan", embed_dim=768, model=ModelConfig(model_type="titan"),
        train=TrainConfig(max_epochs=len(values), patience=5, seed=0, early_stopping=False),
        n_folds=1, framework=Framework.TITAN, strategy="standard", survival_loss="cox",
    )
    results_dir = str(root / "results")
    return _execute(
        lambda: train_mod.train_titan_survival_fold(
            exp, Patients(8), Patients(4), Patients(4), fold=0, results_dir=results_dir,
            device="cpu",
        )["val_metrics"],
        os.path.join(results_dir, "fold_0", "metrics.json"),
    )


def _clam_survival(values, root, mp):
    import pandas as pd
    from autobench.pipeline.clam import survival_train as train_mod

    rng = np.random.default_rng(5)
    benchmark_dir = str(root / "benchmark")
    pt_dir = os.path.join(benchmark_dir, "features", "e", "pt_files")
    os.makedirs(pt_dir)
    slide_ids = [f"s{i}" for i in range(16)]
    for sid in slide_ids:
        torch.save(torch.from_numpy(rng.standard_normal((15, 32)).astype("float32")),
                   os.path.join(pt_dir, f"{sid}.pt"))
    os.makedirs(os.path.join(benchmark_dir, "dataset_csv"))
    pd.DataFrame({
        "slide_id": slide_ids, "case_id": [f"P{i}" for i in range(16)],
        "status": [i % 2 for i in range(16)], "time": [100.0 + 50 * i for i in range(16)],
    }).to_csv(os.path.join(benchmark_dir, "dataset_csv", "os.csv"), index=False)
    splits_dir = os.path.join(benchmark_dir, "splits", "standard", "os")
    os.makedirs(splits_dir)
    pad = lambda ids: ids + [None] * (8 - len(ids))
    pd.DataFrame({
        "train": slide_ids[:8], "val": pad(slide_ids[8:12]), "test": pad(slide_ids[12:]),
    }).to_csv(os.path.join(splits_dir, "splits_0.csv"), index=False)

    mp.setattr(train_mod, "survival_c_index", _scripted(values))
    exp = ExperimentConfig(
        task=TaskConfig(name="os", label_col="status", label_dict={}, n_classes=2,
                        task_type="survival"),
        encoder_key="e", embed_dim=32, model=ModelConfig(model_type="clam_sb"),
        train=TrainConfig(max_epochs=len(values), early_stopping=False, seed=3),
        n_folds=1, framework=Framework.CLAM, strategy="standard", survival_loss="cox",
    )
    results_dir = str(root / "results")
    return _execute(
        lambda: train_mod.train_survival_fold(
            exp, benchmark_dir, 0, results_dir, torch.device("cpu"),
        )["val_metrics"],
        os.path.join(results_dir, "fold_0", "metrics.json"),
    )


@dataclass(frozen=True)
class Arm:
    name: str
    metric: str  # the key of the ``[epoch k]`` line the arm selects on
    key: str  # the key in the fold's validation metrics
    token: str  # the key on the ``[smoothed]`` line
    drive: Callable[[list[float], Path, pytest.MonkeyPatch], FoldRun]
    unselected: list[float]  # the scripted curve under which no epoch is ever selected


_CLS = ("val_auc", "auc_roc_smooth", "val_auc_smooth")
_SURV = ("val_c_index", "c_index_smooth", "val_c_index_smooth")
_NEVER = [NAN] * len(CURVE)

ARMS = [
    Arm("abmil", *_CLS, _abmil, _NEVER),
    Arm("dtfd", *_CLS, _dtfd, _NEVER),
    Arm("titan", *_CLS, _titan, _NEVER),
    # CLAM refuses a fold whose every epoch is non-finite, so its unselected fold
    # is the one that runs no epoch.
    Arm("clam", *_CLS, _clam, []),
    Arm("nnmil", *_CLS, _nnmil("classification"), _NEVER),
    Arm("abmil-survival", *_SURV, _abmil_survival, _NEVER),
    Arm("dtfd-survival", *_SURV, _dtfd_survival, _NEVER),
    Arm("titan-survival", *_SURV, _titan_survival, _NEVER),
    Arm("clam-survival", *_SURV, _clam_survival, _NEVER),
    Arm("nnmil-cox", *_SURV, _nnmil("cox"), _NEVER),
    Arm("nnmil-nllsurv", *_SURV, _nnmil("nllsurv"), _NEVER),
]


def _drive(request, tmp_path_factory, values) -> tuple[Arm, FoldRun]:
    arm = request.param
    root = tmp_path_factory.mktemp(arm.name)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CUDA_VISIBLE_DEVICES", "")
        mp.delenv("AUTOMIL_RESULTS_DIR", raising=False)
        return arm, arm.drive(values(arm), root, mp)


@pytest.fixture(scope="module", params=ARMS, ids=lambda arm: arm.name)
def selected_fold(request, tmp_path_factory):
    return _drive(request, tmp_path_factory, lambda arm: CURVE)


@pytest.fixture(scope="module", params=ARMS, ids=lambda arm: arm.name)
def unselected_fold(request, tmp_path_factory):
    return _drive(request, tmp_path_factory, lambda arm: arm.unselected)


def _printed_value(token: str):
    return None if token == "None" else float(token)


# ---------------------------------------------------------------------------
# A fold that selected an epoch
# ---------------------------------------------------------------------------


class TestEveryArmRecordsItsSmoothedValue:
    def test_the_value_is_the_mean_of_the_printed_curve_around_the_selected_epoch(
        self, selected_fold,
    ):
        arm, fold = selected_fold
        printed = read_log(fold.stdout, arm.metric)
        (selected,) = printed.selected
        # The oracle: the helper over the curve this fold printed, around the epoch it kept.
        assert fold.returned[arm.key] == smoothed_selection_value(printed.curve, selected)
        # The fold printed the scripted curve and kept its maximum, so the line above
        # checked a real window, and the window's mean is worked out independently here.
        epochs = sorted(printed.curve)
        assert [printed.curve[epoch] for epoch in epochs] == CURVE
        assert selected == epochs[BEST]
        assert fold.returned[arm.key] == window_mean(CURVE, BEST)
        assert fold.returned[arm.key] not in (max(CURVE), math.fsum(CURVE) / len(CURVE))

    def test_exactly_one_smoothed_line_follows_the_selected_line(self, selected_fold):
        arm, fold = selected_fold
        printed = read_log(fold.stdout, arm.metric)
        (selected_at,) = printed.selected_at
        assert printed.smoothed_at == (selected_at + 1,)
        match = _SMOOTHED.match(printed.lines[selected_at + 1])
        assert match, printed.lines[selected_at + 1]
        epoch, token, value = match.groups()
        assert (int(epoch), token) == (printed.selected[0], arm.token)
        assert _printed_value(value) == fold.returned[arm.key]

    def test_the_key_is_the_arms_own_and_only_one(self, selected_fold):
        arm, fold = selected_fold
        assert [k for k in fold.returned if k.endswith("_smooth")] == [arm.key]

    def test_the_per_fold_metrics_json_carries_the_key(self, selected_fold):
        arm, fold = selected_fold
        assert fold.persisted[arm.key] == fold.returned[arm.key]

    def test_a_resumed_fold_returns_the_key(self, selected_fold):
        arm, fold = selected_fold
        assert fold.resumed[arm.key] == fold.returned[arm.key]


# ---------------------------------------------------------------------------
# A fold that selected nothing
# ---------------------------------------------------------------------------


class TestAFoldThatSelectedNothing:
    """No epoch ever has a finite score (CLAM: no epoch runs), so the weights are
    the final or untrained ones, ``[selected] epoch=-1``, and there is no score."""

    def test_the_value_is_none_on_every_surface(self, unselected_fold):
        arm, fold = unselected_fold
        printed = read_log(fold.stdout, arm.metric)
        assert printed.selected == (-1,)
        (selected_at,) = printed.selected_at
        assert printed.smoothed_at == (selected_at + 1,)
        assert printed.lines[selected_at + 1] == f"[smoothed] epoch=-1 {arm.token}=None"
        for metrics in (fold.returned, fold.persisted, fold.resumed):
            assert arm.key in metrics and metrics[arm.key] is None
