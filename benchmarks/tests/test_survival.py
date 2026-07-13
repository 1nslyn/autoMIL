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

    def test_abmil_survival_generates_valid_combos(self):
        """ABMIL survival: both abmil/abmil_gated support cox and nllsurv
        (arbitrary output width, unlike CLAM's clam_sb-only cox restriction);
        mse/mae are excluded (no trainer support)."""
        ds = make_survival_ds()
        cfg = BenchmarkConfig.from_dataset_config(
            ds,
            frameworks=[Framework.ABMIL],
            encoder_keys=["conch_v15"],
            abmil_model_types=["abmil", "abmil_gated"],
            tasks=["os_survival"],
            strategies=["standard"],
            n_folds=3,
        )
        exps = generate_all_experiments(cfg, build_registries(ds))
        combos = sorted((e.model.model_type, e.survival_loss) for e in exps)
        assert combos == [
            ("abmil", "cox"),
            ("abmil", "nllsurv"),
            ("abmil_gated", "cox"),
            ("abmil_gated", "nllsurv"),
        ]
        for e in exps:
            assert e.is_survival
            assert e.framework == Framework.ABMIL

    def test_dtfd_survival_generates_valid_combos(self):
        """DTFD survival: nllsurv only. Its two-tier pseudo-bag distillation
        can repeat a discrete (bin_idx, censor) target across pseudo-bags the
        same way it repeats a classification label, but cox's partial-
        likelihood loss needs a cross-patient risk set that doesn't exist
        within one slide's own pseudo-bags -- cox must be excluded."""
        ds = make_survival_ds()
        cfg = BenchmarkConfig.from_dataset_config(
            ds,
            frameworks=[Framework.DTFD],
            encoder_keys=["conch_v15"],
            dtfd_model_types=["dtfd_mil"],
            tasks=["os_survival"],
            strategies=["standard"],
            n_folds=3,
        )
        exps = generate_all_experiments(cfg, build_registries(ds))
        combos = sorted((e.model.model_type, e.survival_loss) for e in exps)
        assert combos == [("dtfd_mil", "nllsurv")]
        for e in exps:
            assert e.is_survival
            assert e.framework == Framework.DTFD

    def test_titan_survival_generates_valid_combos(self):
        """TITAN survival: the single linear-probe head supports cox and
        nllsurv; mse/mae are excluded (no trainer support)."""
        ds = make_survival_ds()
        cfg = BenchmarkConfig.from_dataset_config(
            ds,
            frameworks=[Framework.TITAN],
            tasks=["os_survival"],
            strategies=["standard"],
            n_folds=3,
        )
        exps = generate_all_experiments(cfg, build_registries(ds))
        combos = sorted((e.model.model_type, e.survival_loss) for e in exps)
        assert combos == [("titan", "cox"), ("titan", "nllsurv")]
        for e in exps:
            assert e.is_survival
            assert e.framework == Framework.TITAN


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
    nnMIL/ABMIL bags are ``(1, N, dim)`` and return ``{"logits": ...}``; CLAM
    bags are ``(N, dim)`` with ``instance_eval=False``; TITAN has no bag at
    all (one frozen embedding per slide).
    """

    COMBOS = [
        # DTFD: nllsurv only -- cox has no per-slide analog in its two-tier
        # pseudo-bag distillation (see dtfd/survival_train.py module docstring).
        ("nnmil", "trans_mil", "cox"),
        ("nnmil", "trans_mil", "nllsurv"),
        ("clam", "clam_sb", "cox"),
        ("clam", "clam_sb", "nllsurv"),
        ("clam", "clam_mb", "nllsurv"),
        ("abmil", "abmil", "cox"),
        ("abmil", "abmil_gated", "nllsurv"),
        ("titan", "titan", "cox"),
        ("titan", "titan", "nllsurv"),
        ("dtfd", "dtfd_mil", "nllsurv"),
    ]

    def _bag_logits(self, framework, model_type, n_out, embed_dim, feats):
        """One bag's logits as ``(1, n_out)`` for any of the four frameworks."""
        if framework == "clam":
            from autobench.pipeline.clam._imports import CLAM_SB, CLAM_MB

            cls = CLAM_MB if model_type == "clam_mb" else CLAM_SB
            model = cls(n_classes=n_out, embed_dim=embed_dim)
            logits, *_ = model(feats, instance_eval=False)  # CLAM bag: (N, dim)
            return logits.view(1, -1)

        if framework == "abmil":
            from autobench.pipeline.abmil.model import build_abmil_model

            model = build_abmil_model(model_type, in_dim=embed_dim, num_classes=n_out)
            out = model(feats.unsqueeze(0))  # ABMIL bag: (1, N, dim) -> {"logits": ...}
            return out["logits"].view(1, -1)

        if framework == "titan":
            from autobench.pipeline.titan.model import TitanLinearProbe

            model = TitanLinearProbe(embed_dim, n_out)
            # TITAN has no bag -- one frozen embedding per slide; collapse the
            # dummy patch bag to a single vector as a synthetic stand-in.
            embedding = feats.mean(dim=0, keepdim=True)
            return model(embedding).view(1, -1)

        if framework == "dtfd":
            import torch

            from autobench.pipeline.dtfd.config import DTFDConfig
            from autobench.pipeline.dtfd.model import build_dtfd_bundle
            from autobench.pipeline.dtfd.survival_train import _slide_survival_logits

            dtfd_cfg = DTFDConfig()
            bundle = build_dtfd_bundle(embed_dim, n_out, dtfd_cfg)
            rng = np.random.default_rng(0)
            logits = _slide_survival_logits(bundle, feats, dtfd_cfg, torch.device("cpu"), rng)
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


# ---------------------------------------------------------------------------
# Risk-score orientation — regression guard for the mse/mae inverted-c-index bug
# ---------------------------------------------------------------------------


class TestSurvivalRiskOrientation:
    """Every arm must orient its scalar model output as a RISK score before the
    c-index (higher risk = earlier event). cox/nllsurv are already risk-oriented,
    but mse/mae regress (log) survival time -- a higher logit means a LONGER
    survival -- so their risk must be the NEGATED logit. Getting this wrong
    silently inverts the c-index (a perfect model scores ~0 instead of ~1)."""

    # (label, module) for the arms that score cox/mse/mae via _risk_from_logits.
    ARMS = [
        ("abmil", "autobench.pipeline.abmil.survival_train"),
        ("clam", "autobench.pipeline.clam.survival_train"),
        ("titan", "autobench.pipeline.titan.survival_train"),
    ]

    @pytest.mark.parametrize("label,module", ARMS)
    def test_orientation_per_loss(self, label, module):
        import importlib

        import torch

        m = importlib.import_module(module)
        logits = torch.tensor([-2.0, 0.5, 3.0])  # single-output survival head
        # cox: the logit already IS the risk score.
        assert torch.allclose(m._risk_from_logits(logits, "cox"), logits.view(-1))
        # mse/mae: head predicts (log) time -> risk is the negated logit.
        assert torch.allclose(m._risk_from_logits(logits, "mse"), -logits.view(-1))
        assert torch.allclose(m._risk_from_logits(logits, "mae"), -logits.view(-1))
        # nllsurv: matches the dedicated hazard->risk helper.
        hazards = torch.randn(4, 3)
        assert torch.allclose(
            m._risk_from_logits(hazards, "nllsurv"), m._nllsurv_risk(hazards)
        )

    def test_all_arms_orient_identically(self):
        """A drift between arms would silently invert one arm's c-index."""
        import importlib

        import torch

        mods = [importlib.import_module(mod) for _, mod in self.ARMS]
        logits = torch.tensor([-1.5, 0.0, 2.5])
        for loss in ("cox", "mse", "mae", "nllsurv"):
            arg = logits if loss != "nllsurv" else torch.randn(3, 3)
            refs = [mo._risk_from_logits(arg, loss) for mo in mods]
            for r in refs[1:]:
                assert torch.allclose(refs[0], r)

    def test_mse_optimal_model_is_concordant(self):
        """End-to-end: a model that perfectly learned the mse objective
        (logit == log time) must score c-index ~1.0 after orientation. The raw
        un-negated logit -- the bug -- would score the inverted ~0.0."""
        import importlib

        import torch

        m = importlib.import_module("autobench.pipeline.abmil.survival_train")
        from nnMIL.training.losses.survival_loss import survival_c_index

        time = torch.tensor([1.0, 2.0, 4.0, 8.0])
        status = torch.tensor([1.0, 1.0, 1.0, 1.0])
        pids = [f"p{i}" for i in range(4)]
        opt_logit = torch.log(time + 1e-8)  # the mse loss's optimum

        oriented = m._risk_from_logits(opt_logit, "mse")
        assert float(survival_c_index(oriented, status, time, pids)) > 0.99
        # The raw logit (pre-fix behavior) inverts the metric -- documents the bug.
        assert float(survival_c_index(opt_logit.view(-1), status, time, pids)) < 0.01

    def test_unknown_loss_raises(self):
        """A typo'd/unknown loss must fail loudly, not be silently scored as cox."""
        import importlib

        import torch

        logits = torch.tensor([-1.0, 0.0, 1.0])
        for _, module in self.ARMS:
            m = importlib.import_module(module)
            with pytest.raises(ValueError, match="unknown survival loss"):
                m._risk_from_logits(logits, "coxx")


class TestNnmilTrainerRiskOrientation:
    """The nnMIL SurvivalTrainer scores cox/mse/mae from a single-output head
    (nllsurv routes to SurvivalPorpoiseTrainer). It must orient mse/mae as the
    NEGATED prediction before the c-index, like the adapter arms -- otherwise
    the nnmil arm's mse/mae c-index is silently inverted (the same bug, in the
    one framework whose grid actually generates mse/mae experiments)."""

    def test_risk_from_preds_orientation(self):
        import numpy as np

        import autobench.pipeline.clam.survival_train  # noqa: F401  (puts nnMIL on path)
        from nnMIL.training.trainers.survival_trainer import _risk_from_preds

        preds = np.array([-2.0, 0.5, 3.0], dtype=np.float32)
        np.testing.assert_allclose(_risk_from_preds(preds, "cox"), preds)
        np.testing.assert_allclose(_risk_from_preds(preds, "mse"), -preds)
        np.testing.assert_allclose(_risk_from_preds(preds, "mae"), -preds)

    def test_mse_optimal_model_is_concordant(self):
        """A trainer output that perfectly learned the mse objective
        (pred == log time) scores c-index ~1.0 after orientation; the raw
        (pre-fix) prediction inverts it to ~0.0."""
        import numpy as np
        import torch

        import autobench.pipeline.clam.survival_train  # noqa: F401
        from nnMIL.training.losses.survival_loss import survival_c_index
        from nnMIL.training.trainers.survival_trainer import _risk_from_preds

        time = np.array([1.0, 2.0, 4.0, 8.0], dtype=np.float32)
        status = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        pids = [f"p{i}" for i in range(4)]
        opt_pred = np.log(time + 1e-8)  # the mse loss's optimum

        def _ci(risk_np):
            return float(survival_c_index(
                torch.tensor(risk_np, dtype=torch.float32),
                torch.tensor(status, dtype=torch.float32),
                torch.tensor(time, dtype=torch.float32),
                pids,
            ))

        assert _ci(_risk_from_preds(opt_pred, "mse")) > 0.99
        # Raw prediction (pre-fix) inverts the metric -- documents the bug.
        assert _ci(opt_pred) < 0.01

    def test_unknown_loss_raises(self):
        """Mirror the adapters: an unknown/typo'd loss must fail loudly here too,
        not silently fall through as cox-identity."""
        import numpy as np

        import autobench.pipeline.clam.survival_train  # noqa: F401  (nnMIL on path)
        from nnMIL.training.trainers.survival_trainer import _risk_from_preds

        with pytest.raises(ValueError, match="unknown survival loss"):
            _risk_from_preds(np.array([1.0, 2.0], dtype=np.float32), "coxx")


class TestNnmilPredictorRiskOrientation:
    """The standalone nnMIL SurvivalPredictor (inference tooling) must orient a
    single-output mse/mae head as NEGATED risk too -- the same bug class as the
    trainer, on the inference path. A multi-bin output stays nllsurv (survival
    curve); a single output uses the loss type to tell cox from mse/mae."""

    def test_output_to_risk_orientation(self):
        import torch

        import autobench.pipeline.clam.survival_train  # noqa: F401  (nnMIL on path)
        from nnMIL.inference.predictors.survival_predictor import _output_to_risk

        single = torch.tensor([[-2.0], [0.5], [3.0]])  # [N, 1] single-output head
        assert torch.allclose(_output_to_risk(single, "cox"), single)
        assert torch.allclose(_output_to_risk(single, "mse"), -single)
        assert torch.allclose(_output_to_risk(single, "mae"), -single)
        # Multi-bin output is unambiguously nllsurv (survival-curve risk),
        # regardless of the survival_loss argument.
        hz = torch.randn(4, 3)
        surv = torch.cumprod(1 - torch.sigmoid(hz), dim=1)
        assert torch.allclose(_output_to_risk(hz, "cox"), -surv.sum(dim=1, keepdim=True))

    def test_mse_optimal_is_concordant(self):
        import torch

        import autobench.pipeline.clam.survival_train  # noqa: F401
        from nnMIL.inference.predictors.survival_predictor import _output_to_risk
        from nnMIL.training.losses.survival_loss import survival_c_index

        time = torch.tensor([1.0, 2.0, 4.0, 8.0])
        status = torch.tensor([1.0, 1.0, 1.0, 1.0])
        pids = [f"p{i}" for i in range(4)]
        opt = torch.log(time + 1e-8).view(-1, 1)  # mse optimum, single-output head
        risk = _output_to_risk(opt, "mse").view(-1)
        assert float(survival_c_index(risk, status, time, pids)) > 0.99
        # Treating it as cox (pre-fix behavior) inverts the metric.
        raw = _output_to_risk(opt, "cox").view(-1)
        assert float(survival_c_index(raw, status, time, pids)) < 0.01

    def test_unknown_loss_raises(self):
        import torch

        import autobench.pipeline.clam.survival_train  # noqa: F401
        from nnMIL.inference.predictors.survival_predictor import _output_to_risk

        with pytest.raises(ValueError, match="unknown survival loss"):
            _output_to_risk(torch.tensor([[1.0], [2.0]]), "coxx")


class TestSurvivalCIndexDirection:
    """survival_c_index must rank higher risk as shorter survival (a sign flip
    here silently inverts every arm at once)."""

    def test_direction(self):
        import autobench.pipeline.clam.survival_train  # noqa: F401  (puts nnMIL on path)
        import torch
        from nnMIL.training.losses.survival_loss import survival_c_index

        risk = torch.tensor([3.0, 2.0, 1.0])  # sample 0 highest risk...
        time = torch.tensor([1.0, 2.0, 3.0])  # ...and shortest survival (perfect)
        status = torch.tensor([1.0, 1.0, 1.0])
        pids = ["a", "b", "c"]
        assert float(survival_c_index(risk, status, time, pids)) > 0.99
        assert float(survival_c_index(-risk, status, time, pids)) < 0.01


class TestSurvivalLossGoldenValues:
    """Pin the survival losses to known-correct outputs so a future sign flip or
    formula change is caught (there is no CI on this repo)."""

    def test_cox_matches_independent_breslow_reference(self):
        import autobench.pipeline.clam.survival_train  # noqa: F401
        import torch
        from nnMIL.training.losses.survival_loss import SurvivalLoss

        logits = torch.tensor([0.5, -0.5, 1.0, 0.0], requires_grad=True)  # cox guards on grad
        status = torch.tensor([1.0, 1.0, 0.0, 1.0])  # sample 2 censored
        time = torch.tensor([1.0, 2.0, 3.0, 4.0])  # distinct times

        def ref_cox_nll(lg, st, tm):
            # -mean over events of [x_i - log sum_{t_j >= t_i} exp x_j] (Breslow).
            terms = []
            for i in range(len(tm)):
                if st[i] != 1:  # outer sum ranges over events only
                    continue
                risk_set = tm >= tm[i]  # everyone still at risk (event or censored)
                terms.append(lg[i] - torch.log(torch.exp(lg[risk_set]).sum()))
            return -torch.stack(terms).mean()

        lib = SurvivalLoss(loss_type="cox")(logits, status, time)
        ref = ref_cox_nll(logits, status, time)
        assert torch.allclose(lib, ref, atol=1e-6)

    def test_nllsurv_matches_golden_value(self):
        import autobench.pipeline.clam.survival_train  # noqa: F401
        import torch
        from nnMIL.training.losses.survival_loss_nll import NLLSurvLoss

        logits = torch.tensor(
            [
                [0.10, -0.20, 0.30],
                [0.50, 0.00, -0.50],
                [-0.30, 0.20, 0.10],
                [0.40, -0.10, 0.20],
            ],
            requires_grad=True,
        )
        y = torch.tensor([0, 1, 2, 1])
        c = torch.tensor([0, 1, 0, 0])  # 1=censored
        # Golden captured from the current (reference-verified) implementation;
        # regenerate only if the loss definition intentionally changes.
        value = float(NLLSurvLoss()(logits, y, c).detach())
        assert value == pytest.approx(1.49148083, abs=1e-5)


class TestDtfdCoxGuard:
    """DTFD survival must reject cox at runtime (nllsurv-only), not just in the
    grid -- its within-slide pseudo-bags have no cross-patient risk set."""

    def test_run_dtfd_experiment_rejects_cox(self, tmp_path):
        import dataclasses

        from autobench.pipeline.config import (
            BenchmarkConfig,
            Framework,
            build_registries,
            generate_all_experiments,
        )
        from autobench.pipeline.dtfd.runner import run_dtfd_experiment

        ds = make_survival_ds()
        cfg = BenchmarkConfig.from_dataset_config(
            ds,
            frameworks=[Framework.DTFD],
            encoder_keys=["conch_v15"],
            dtfd_model_types=["dtfd_mil"],
            tasks=["os_survival"],
            strategies=["standard"],
            n_folds=1,
        )
        # The grid only emits dtfd+nllsurv; force cox to exercise the runtime guard.
        exp = generate_all_experiments(cfg, build_registries(ds))[0]
        exp_cox = dataclasses.replace(exp, survival_loss="cox")
        with pytest.raises(ValueError, match="does not support cox"):
            run_dtfd_experiment(exp_cox, str(tmp_path), device="cpu")
