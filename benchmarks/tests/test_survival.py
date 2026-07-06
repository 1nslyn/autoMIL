"""Tests for survival-prediction wiring (cox/mse/mae/nllsurv) across both
frameworks — nnMIL and CLAM.

Covers the autobench wrapper layer — config/grid (per-framework survival
combos), task-CSV creation, status-stratified splits, nnMIL plan generation per
loss, trainer selection, metric normalization, the survival composite — plus a
CPU model+loss+c-index smoke for both frameworks' survival models. Full training
runs on real features live elsewhere (gated).
"""

import json
import os

import h5py
import numpy as np
import pandas as pd
import pytest

from autobench.config import TaskDef, _parse_tasks
from autobench.pipeline.config import (
    BenchmarkConfig,
    Framework,
    build_registries,
    generate_all_experiments,
)
from autobench.pipeline.clam.runner import _write_fold_result_json
from autobench.pipeline.nnmil.evaluate import normalize_nnmil_metrics
from autobench.pipeline.nnmil.prepare import (
    _load_splits_as_nnmil_format,
    nnmil_plan_dir,
    prepare_nnmil_experiment,
)
from autobench.pipeline.nnmil.train import select_nnmil_trainer
from autobench.pipeline.prepare import create_task_csv
from autobench.pipeline.splits import create_strategy_splits
from _helpers import make_test_ds


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_survival_ds(**overrides):
    """A DatasetConfig with a single survival task (plain slide/case columns)."""
    base = dict(
        slide_id_column="slide_id",
        slide_id_transform=None,
        case_id_column="case_id",
        status_column=None,
        status_value=None,
        tasks={
            "os_survival": TaskDef(
                name="os_survival",
                task_type="survival",
                event_col="OS_event",
                time_col="OS_time",
                survival_losses=["cox", "mse", "mae", "nllsurv"],
                nll_bins=2,
            ),
        },
        task_strategy_feasibility={"os_survival": ["standard"]},
    )
    base.update(overrides)
    return make_test_ds(**base)


@pytest.fixture
def survival_benchmark(tmp_path):
    """Benchmark dir with synthetic h5 features + a status/time task CSV + splits."""
    rng = np.random.default_rng(0)
    bd = str(tmp_path / "benchmark")
    os.makedirs(os.path.join(bd, "dataset_csv"), exist_ok=True)

    h5_dir = tmp_path / "features_conch_v15"
    h5_dir.mkdir()
    for i in range(30):
        n_patches = int(rng.integers(50, 200))
        with h5py.File(h5_dir / f"slide_{i:05d}.h5", "w") as f:
            f.create_dataset(
                "features", data=rng.standard_normal((n_patches, 768)).astype(np.float32)
            )
            f.create_dataset("coords", data=rng.integers(0, 1000, (n_patches, 2)))

    # status alternates (15 events / 15 censored); varied times for nllsurv bins.
    rows = [
        {
            "case_id": f"P{i:03d}",
            "slide_id": f"slide_{i:05d}",
            "status": i % 2,
            "time": float(10 + (i * 3) % 40),
        }
        for i in range(30)
    ]
    csv_path = os.path.join(bd, "dataset_csv", "os_survival.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    splits_dir = os.path.join(bd, "splits", "standard", "os_survival")
    create_strategy_splits(
        csv_path, splits_dir, n_splits=3, seed=42, stratify_col="status"
    )
    return bd, str(h5_dir)


# ---------------------------------------------------------------------------
# Config / parsing / grid
# ---------------------------------------------------------------------------


class TestSurvivalConfig:
    def test_parse_tasks_survival(self):
        raw = {
            "os": {
                "task_type": "survival",
                "event_col": "OS_event",
                "time_col": "OS_time",
                "survival_losses": ["cox", "nllsurv"],
                "nll_bins": 3,
            }
        }
        t = _parse_tasks(raw)["os"]
        assert t.task_type == "survival"
        assert t.event_col == "OS_event"
        assert t.time_col == "OS_time"
        assert t.survival_losses == ["cox", "nllsurv"]
        assert t.nll_bins == 3
        assert t.label_col is None
        assert t.label_map is None

    def test_parse_tasks_classification_unchanged(self):
        raw = {"brca": {"label_col": "BRCA", "label_map": {0: "neg", 1: "pos"}}}
        t = _parse_tasks(raw)["brca"]
        assert t.task_type == "classification"
        assert t.label_col == "BRCA"
        assert t.n_classes == 2

    def test_build_registries_survival(self):
        reg = build_registries(make_survival_ds())
        t = reg.task_registry["os_survival"]
        assert t.task_type == "survival"
        assert t.event_col == "OS_event"
        assert t.time_col == "OS_time"
        assert t.label_dict is None
        assert t.survival_losses == ["cox", "mse", "mae", "nllsurv"]
        assert t.nll_bins == 2

    def test_grid_fans_out_over_losses(self):
        ds = make_survival_ds()
        cfg = BenchmarkConfig.from_dataset_config(
            ds,
            frameworks=[Framework.NNMIL],
            encoder_keys=["conch_v15"],
            nnmil_model_types=["simple_mil"],
            tasks=["os_survival"],
            strategies=["standard"],
            n_folds=3,
        )
        exps = generate_all_experiments(cfg, build_registries(ds))
        assert len(exps) == 4  # 4 losses x 1 encoder x 1 model
        assert {e.survival_loss for e in exps} == {"cox", "mse", "mae", "nllsurv"}
        # No id / results_subdir collisions across losses.
        assert len({e.experiment_id for e in exps}) == 4
        assert len({e.results_subdir for e in exps}) == 4
        for e in exps:
            assert e.experiment_id.endswith(f"__{e.survival_loss}")
            assert e.is_survival

    def test_clam_survival_generates_valid_combos(self):
        """CLAM survival combos: cox is clam_sb-only (single risk output);
        nllsurv works for clam_sb and clam_mb. mse/mae and mil are excluded."""
        ds = make_survival_ds()
        cfg = BenchmarkConfig.from_dataset_config(
            ds,
            frameworks=[Framework.CLAM],
            encoder_keys=["conch_v15"],
            tasks=["os_survival"],
            strategies=["standard"],
            n_folds=3,
        )
        exps = generate_all_experiments(cfg, build_registries(ds))
        combos = sorted((e.model.model_type, e.survival_loss) for e in exps)
        assert combos == [
            ("clam_mb", "nllsurv"),
            ("clam_sb", "cox"),
            ("clam_sb", "nllsurv"),
        ]
        for e in exps:
            assert e.is_survival
            assert e.framework == Framework.CLAM


# ---------------------------------------------------------------------------
# Task CSV creation
# ---------------------------------------------------------------------------


class TestSurvivalTaskCsv:
    def test_emits_status_time_columns(self, tmp_path):
        ds = make_survival_ds()
        mapping = tmp_path / "mapping.csv"
        pd.DataFrame({
            "case_id": [f"P{i}" for i in range(6)],
            "slide_id": [f"s{i}" for i in range(6)],
            "OS_event": [1, 0, 1, 0, 1, None],   # last row dropped (NaN event)
            "OS_time": [10.0, 20.0, 5.0, 30.0, 15.0, 8.0],
        }).to_csv(mapping, index=False)
        out = tmp_path / "out.csv"

        df = create_task_csv(
            str(mapping), str(out), ds=ds,
            task_type="survival", event_col="OS_event", time_col="OS_time",
        )
        assert list(df.columns) == ["case_id", "slide_id", "status", "time"]
        assert "label" not in df.columns
        assert len(df) == 5                      # NaN-event row dropped
        assert set(df["status"].tolist()) <= {0, 1}
        assert df["time"].dtype == float

    def test_missing_cols_raises(self, tmp_path):
        ds = make_survival_ds()
        mapping = tmp_path / "m.csv"
        pd.DataFrame({"case_id": ["P0"], "slide_id": ["s0"]}).to_csv(mapping, index=False)
        with pytest.raises(ValueError):
            create_task_csv(
                str(mapping), str(tmp_path / "o.csv"), ds=ds,
                task_type="survival", event_col=None, time_col=None,
            )


# ---------------------------------------------------------------------------
# Status-stratified splits
# ---------------------------------------------------------------------------


class TestSurvivalSplits:
    def test_each_fold_has_events(self, survival_benchmark):
        bd, _ = survival_benchmark
        task_df = pd.read_csv(os.path.join(bd, "dataset_csv", "os_survival.csv"))
        splits_dir = os.path.join(bd, "splits", "standard", "os_survival")
        status_by_slide = dict(zip(task_df["slide_id"], task_df["status"]))
        for fold in range(3):
            sdf = pd.read_csv(os.path.join(splits_dir, f"splits_{fold}.csv"))
            for split in ("train", "val", "test"):
                events = sum(
                    status_by_slide[s] for s in sdf[split].dropna()
                )
                assert events > 0, f"fold {fold} {split} has zero events"


# ---------------------------------------------------------------------------
# nnMIL plan generation (per loss)
# ---------------------------------------------------------------------------


class TestSurvivalPlan:
    def test_load_splits_survival_keys(self, survival_benchmark):
        bd, _ = survival_benchmark
        task_df = pd.read_csv(os.path.join(bd, "dataset_csv", "os_survival.csv"))
        splits_dir = os.path.join(bd, "splits", "standard", "os_survival")
        data = _load_splits_as_nnmil_format(
            splits_dir, task_df, None, 3, task_type="survival"
        )
        info = data["fold_0"]["train"]["slide_info"][0]
        assert set(info.keys()) == {"slide_id", "patient_id", "status", "time"}
        assert isinstance(info["status"], int)
        assert isinstance(info["time"], float)

    @pytest.mark.parametrize("loss", ["cox", "mse", "mae", "nllsurv"])
    def test_prepare_plan_per_loss(self, survival_benchmark, loss):
        bd, h5_dir = survival_benchmark
        plan_path = prepare_nnmil_experiment(
            benchmark_dir=bd,
            task_name="os_survival",
            encoder_key="conch_v15",
            strategy="standard",
            embed_dim=768,
            features_base_dir=os.path.dirname(h5_dir),
            seed=42,
            n_splits=3,
            task_type="survival",
            event_col="OS_event",
            time_col="OS_time",
            survival_loss=loss,
            nll_bins=2,
        )
        # Loss-suffixed plan dir (no overwrite across losses).
        assert plan_path.endswith(
            os.path.join(f"os_survival_conch_v15_{loss}", "dataset_plan.json")
        )

        with open(plan_path) as f:
            plan = json.load(f)
        assert plan["task_type"] == "survival"
        assert plan["metric"] == "c_index"
        assert plan["survival_loss"] == loss

        tc = plan["training_configuration"]
        assert tc["survival_loss"] == loss
        if loss == "nllsurv":
            assert tc["num_classes"] == 2
            assert tc["nll_bins"] == 2
        else:
            assert tc["num_classes"] == 1

        info = plan["data_splits"]["fold_0"]["train"]["slide_info"][0]
        assert set(info.keys()) == {"slide_id", "patient_id", "status", "time"}
        assert "label" not in info and "event" not in info

        ds_csv = pd.read_csv(os.path.join(os.path.dirname(plan_path), "dataset.csv"))
        assert set(ds_csv.columns) == {"slide_id", "patient_id", "status", "time"}

    def test_plan_dir_helper_suffix(self):
        cls = nnmil_plan_dir("/b", "standard", "t", "enc", survival_loss=None)
        surv = nnmil_plan_dir("/b", "standard", "t", "enc", survival_loss="cox")
        assert cls.endswith(os.path.join("standard", "t_enc"))
        assert surv.endswith(os.path.join("standard", "t_enc_cox"))


# ---------------------------------------------------------------------------
# Trainer selection
# ---------------------------------------------------------------------------


class TestSelectTrainer:
    def test_classification(self):
        assert select_nnmil_trainer("classification", None) == "classification"

    @pytest.mark.parametrize("loss", ["cox", "mse", "mae"])
    def test_survival_regression_losses(self, loss):
        assert select_nnmil_trainer("survival", loss) == "survival"

    def test_nllsurv_uses_porpoise(self):
        assert select_nnmil_trainer("survival", "nllsurv") == "survival_porpoise"


# ---------------------------------------------------------------------------
# Metric normalization
# ---------------------------------------------------------------------------


class TestSurvivalNormalize:
    def test_maps_survival_keys(self):
        raw = {
            "test_c_index": 0.62,
            "test_events": 10,
            "test_censored": 20,
            "test_event_rate": 0.33,
            "test_mean_time": 5.0,
            "test_median_time": 4.0,
        }
        result = normalize_nnmil_metrics(raw, split="test", task_type="survival")
        assert result["c_index"] == 0.62
        assert result["events"] == 10
        assert result["censored"] == 20
        assert result["event_rate"] == 0.33
        # No classification keys leak in.
        assert "auc_roc" not in result

    def test_missing_cindex_defaults_nan(self):
        result = normalize_nnmil_metrics({}, split="test", task_type="survival")
        assert np.isnan(result["c_index"])

    def test_classification_path_unchanged(self):
        raw = {"test_test/bacc": 0.82, "test_test/auroc": 0.9}
        result = normalize_nnmil_metrics(raw, split="test")
        assert result["balanced_accuracy"] == 0.82
        assert result["auc_roc"] == 0.9
        assert "c_index" not in result


# ---------------------------------------------------------------------------
# Composite (per-fold archive writer)
# ---------------------------------------------------------------------------


class TestSurvivalComposite:
    def test_fold_result_uses_c_index(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUTOMIL_RESULTS_DIR", str(tmp_path))
        monkeypatch.setenv("AUTOMIL_FOLD_COUNT", "3")
        result = {
            "test_metrics": {"c_index": 0.7},
            "val_metrics": {"c_index": 0.6},
        }
        _write_fold_result_json(0, result)
        payload = json.loads((tmp_path / "fold_0_result.json").read_text())
        assert payload["composite"] == 0.7
        assert payload["metrics"]["test_c_index"] == 0.7
        assert payload["metrics"]["val_c_index"] == 0.6

    def test_fold_result_classification_unchanged(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUTOMIL_RESULTS_DIR", str(tmp_path))
        monkeypatch.setenv("AUTOMIL_FOLD_COUNT", "3")
        result = {
            "test_metrics": {"auc_roc": 0.8, "balanced_accuracy": 0.7},
            "val_metrics": {"auc_roc": 0.75, "balanced_accuracy": 0.65},
        }
        _write_fold_result_json(0, result)
        payload = json.loads((tmp_path / "fold_0_result.json").read_text())
        assert payload["composite"] == pytest.approx((0.8 + 0.7) / 2)
        assert payload["metrics"]["test_auc"] == 0.8


# ---------------------------------------------------------------------------
# Model + loss + c-index smoke — both frameworks (CPU, dummy features)
# ---------------------------------------------------------------------------


class TestSurvivalTrainerSmoke:
    """Forward the survival model, compute the loss, and score the c-index for
    every valid (framework, model, loss) combo on tiny dummy bags — no GPU, no
    data. Guards the model/loss/metric contract each survival trainer relies on:
    nnMIL bags are ``(1, N, dim)`` and return ``{"logits": ...}``; CLAM bags are
    ``(N, dim)`` with ``instance_eval=False``.
    """

    COMBOS = [
        # NB: ab_mil was removed from nnMIL (promoted to Framework.ABMIL in the
        # MIL-integration work); survival is implemented for nnMIL + CLAM only,
        # so a survival ABMIL-framework arm is future work, not covered here.
        ("nnmil", "trans_mil", "cox"),
        ("nnmil", "trans_mil", "nllsurv"),
        ("clam", "clam_sb", "cox"),
        ("clam", "clam_sb", "nllsurv"),
        ("clam", "clam_mb", "nllsurv"),
    ]

    def _bag_logits(self, framework, model_type, n_out, embed_dim, feats):
        """One bag's logits as ``(1, n_out)`` for either framework."""
        if framework == "clam":
            from autobench.pipeline.clam._imports import CLAM_SB, CLAM_MB

            cls = CLAM_MB if model_type == "clam_mb" else CLAM_SB
            model = cls(n_classes=n_out, embed_dim=embed_dim)
            logits, *_ = model(feats, instance_eval=False)  # CLAM bag: (N, dim)
            return logits.view(1, -1)

        from nnMIL.network_architecture.model_factory import create_mil_model

        model = create_mil_model(
            model_type=model_type, input_dim=embed_dim, hidden_dim=64,
            num_classes=n_out, dropout=0.0,
        )
        out = model(feats.unsqueeze(0))  # nnMIL bag: (1, N, dim)
        logits = out["logits"] if isinstance(out, dict) else out
        return logits.view(1, -1)

    @pytest.mark.parametrize("framework,model_type,loss", COMBOS)
    def test_model_loss_cindex(self, framework, model_type, loss):
        import torch

        # Importing the CLAM survival trainer puts the vendored nnMIL tree on
        # sys.path so the survival core imports below resolve.
        import autobench.pipeline.clam.survival_train  # noqa: F401
        from nnMIL.training.losses.survival_loss import SurvivalLoss, survival_c_index
        from nnMIL.training.losses.survival_loss_nll import NLLSurvLoss

        torch.manual_seed(0)
        embed_dim, n_bins, batch, n_patches = 64, 4, 4, 20
        n_out = n_bins if loss == "nllsurv" else 1

        logits = torch.cat([
            self._bag_logits(framework, model_type, n_out, embed_dim, torch.randn(n_patches, embed_dim))
            for _ in range(batch)
        ])  # (batch, n_out)
        assert logits.shape == (batch, n_out)

        status = torch.tensor([1.0, 0.0, 1.0, 0.0])
        time = torch.tensor([100.0, 200.0, 300.0, 400.0])
        if loss == "nllsurv":
            y = torch.randint(0, n_bins, (batch,))
            value = NLLSurvLoss()(logits, y, (1 - status).long())
            risk = -torch.cumprod(1 - torch.sigmoid(logits), dim=1).sum(dim=1)
        else:
            value = SurvivalLoss(loss_type="cox")(logits.view(-1), status, time)
            risk = logits.view(-1)
        assert torch.isfinite(value)

        ci = survival_c_index(
            risk.detach(), status, time, [f"p{i}" for i in range(batch)],
        )
        assert ci is None or np.isnan(float(ci)) or 0.0 <= float(ci) <= 1.0
