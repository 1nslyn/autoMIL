"""Adapter-side survival training for the CLAM attention backbone.

Reuses CLAM's ``CLAM_SB``/``CLAM_MB`` model but trains it with a survival loss
in a custom loop, leaving CLAM's vendored classification loop untouched. Mirrors
the nnMIL survival trainer: cox/nllsurv loss, val-loss model selection,
event-time nllsurv bins, patient-level sksurv c-index. CLAM forwards one slide
per call, so cox's risk set is formed over a mini-batch of slides.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from autobench import LIB_ROOT
from autobench.pipeline.clam._imports import CLAM_SB, CLAM_MB, get_optim
from autobench.pipeline.clam.dataset import load_survival_fold_splits
from autobench.pipeline.config import ExperimentConfig, TrainConfig
from autobench.pipeline.evaluate import file_sha256_or_none, write_survival_predictions_csv
from autobench.pipeline.policy_dispatch import PolicyRuntime

# The framework-agnostic survival core lives under the vendored nnMIL tree;
# import it adapter -> lib (the normal autobench direction).
if str(LIB_ROOT) not in sys.path:
    sys.path.insert(0, str(LIB_ROOT))
from nnMIL.training.losses.survival_loss import SurvivalLoss, survival_c_index  # noqa: E402
from nnMIL.training.losses.survival_loss_nll import NLLSurvLoss  # noqa: E402
from nnMIL.training.callbacks.early_stopping import EarlyStoppingSurvival  # noqa: E402

# Slides per optimizer step. For cox this is the risk set; each slide is a
# full-patch CLAM forward held in the graph until backward, so keep it modest.
_SURV_BATCH = 16


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_model(model_type: str, n_out: int, embed_dim: int, dropout: float, size_arg: str):
    # clam_mb gives one attention branch per output (per time-bin for nllsurv);
    # clam_sb shares a single attention branch.
    cls = CLAM_MB if model_type == "clam_mb" else CLAM_SB
    return cls(
        gate=True, size_arg=size_arg, dropout=dropout,
        n_classes=n_out, embed_dim=embed_dim,
    )


def _build_optimizer(model: torch.nn.Module, train_cfg: TrainConfig):
    """Build the live CLAM optimizer from the declared training recipe.

    Classification reaches CLAM's upstream ``get_optim`` through
    ``core_utils.train``. Survival is an adapter-owned loop, so it must call the
    same factory explicitly; otherwise ``train.optimizer`` is only recorded in
    the experiment config and silently ignored by the process that actually
    trains the model.
    """
    args = SimpleNamespace(
        opt=train_cfg.optimizer,
        lr=train_cfg.lr,
        reg=train_cfg.weight_decay,
    )
    return get_optim(model, args)


def _should_stop(train_cfg: TrainConfig, stopping_state) -> bool:
    """Return whether the declared early-stopping switch permits termination."""
    return bool(train_cfg.early_stopping and stopping_state.early_stop)


def _load_feats(pt_path: str, device: torch.device) -> torch.Tensor:
    return torch.load(pt_path, map_location="cpu").float().to(device)


def _bag_logits(model, feats: torch.Tensor) -> torch.Tensor:
    """Bag-level logits ``(n_out,)`` from one slide's patch features.

    ``instance_eval=False`` skips CLAM's instance-clustering branch — it is
    class-supervised and has no meaning for survival.
    """
    logits, *_ = model(feats, instance_eval=False)
    return logits.view(-1)


def _event_time_bin_edges(times, statuses, n_bins: int) -> np.ndarray:
    """NLL bin edges from event (uncensored) train times — the PORPOISE/MCAT
    convention that stops censored outliers skewing the bins. Falls back to all
    train times when a fold has too few distinct event times."""
    times = np.asarray(times, dtype=float)
    statuses = np.asarray(statuses)
    event_times = times[statuses == 1]
    src = event_times if len(np.unique(event_times)) >= n_bins else times
    edges = np.quantile(src, np.linspace(0, 1, n_bins + 1))
    eps = 1e-6
    for i in range(1, len(edges)):  # force strictly increasing
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + eps
    return edges


def _time_to_bin(time_val: float, edges: np.ndarray) -> int:
    b = int(np.digitize([time_val], edges[1:-1])[0])
    return int(np.clip(b, 0, len(edges) - 2))


def _nllsurv_risk(logits: torch.Tensor) -> torch.Tensor:
    """Scalar risk from nllsurv hazard logits ``(B, n_bins)`` — lower predicted
    survival = higher risk, oriented like the cox score (higher = earlier event)."""
    survival = torch.cumprod(1 - torch.sigmoid(logits), dim=1)
    return -survival.sum(dim=1)


def _risk_from_logits(logits: torch.Tensor, loss_type: str) -> torch.Tensor:
    """Orient a model's scalar survival output as a risk score (higher = earlier
    event) for the c-index, per loss type:
      - cox:      the logit already IS the risk score.
      - nllsurv:  ``-sum`` of the predicted survival curve (see ``_nllsurv_risk``).
      - mse/mae:  the head regresses (log) survival time, so a HIGHER logit means
                  a LONGER survival — negate it, else the c-index is inverted.
    """
    if loss_type == "nllsurv":
        return _nllsurv_risk(logits)
    if loss_type in ("mse", "mae"):
        return -logits.view(-1)
    if loss_type == "cox":
        return logits.view(-1)
    raise ValueError(
        f"unknown survival loss {loss_type!r}; expected cox/mse/mae/nllsurv"
    )


def train_survival_fold(
    exp_cfg: ExperimentConfig,
    benchmark_dir: str,
    fold: int,
    results_dir: str,
    device: torch.device,
    policy_runtime: PolicyRuntime | None = None,
) -> dict:
    """Train one CLAM survival fold; return ``{test_metrics, val_metrics, fold}``."""
    fold_dir = os.path.join(results_dir, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)
    metrics_path = os.path.join(fold_dir, "metrics.json")
    if os.path.exists(metrics_path):  # fold-level resume
        with open(metrics_path) as f:
            return json.load(f)

    # FOLD-TIMING CONTRACT: covers the whole fold, checkpoint restore and final
    # scoring included, so the number is comparable across arms and task types.
    # This arm reported no elapsed_seconds at all, so every CLAM survival fold
    # (and therefore every CLAM survival cell's elapsed_seconds_total) read 0.
    start = time.time()
    _seed(exp_cfg.train.seed)
    loss_type = exp_cfg.survival_loss or "cox"
    is_nll = loss_type == "nllsurv"
    n_bins = exp_cfg.task.nll_bins
    # Survival head width: a single risk score (cox), or n_bins hazard logits (nllsurv).
    n_out = n_bins if is_nll else 1
    model_type = exp_cfg.model.model_type

    train, val, test = load_survival_fold_splits(exp_cfg, benchmark_dir, fold)
    print(
        f"    [CLAM-surv fold {fold}] {model_type}/{loss_type} — "
        f"train={len(train)} val={len(val)} test={len(test)} slides"
    )

    model = _build_model(
        model_type, n_out, exp_cfg.embed_dim, exp_cfg.model.dropout, exp_cfg.model.model_size,
    ).to(device)

    if is_nll:
        loss_fn = NLLSurvLoss()
        edges = _event_time_bin_edges(
            [s["time"] for s in train], [s["status"] for s in train], n_bins,
        )
    else:
        loss_fn = SurvivalLoss(loss_type=loss_type)
        edges = None

    optimizer = _build_optimizer(model, exp_cfg.train)
    policy_runtime = policy_runtime or PolicyRuntime()
    optimizer = policy_runtime.wrap_optimizer(optimizer)
    # mode="min": select the checkpoint on val LOSS. With ~2 events per val fold
    # the val c-index is near-random, so maximizing it would overfit to noise.
    early_stopping = EarlyStoppingSurvival(
        patience=exp_cfg.train.patience, verbose=True, metric="c_index",
        save_dir=fold_dir, model_type=model_type, mode="min",
    )

    def _batch_loss(batch: list[dict]) -> torch.Tensor:
        # Forward each slide separately (CLAM = one bag per call) and stack the
        # bag logits so the loss sees the whole mini-batch at once.
        logits_list, status_list, time_list = [], [], []
        for s in batch:
            logits_list.append(_bag_logits(model, _load_feats(s["pt_path"], device)))
            status_list.append(s["status"])
            time_list.append(s["time"])
        logits = torch.stack(logits_list)  # (B, n_out)
        status = torch.tensor(status_list, dtype=torch.float32, device=device)
        time = torch.tensor(time_list, dtype=torch.float32, device=device)
        if is_nll:
            y = torch.tensor(
                [_time_to_bin(t, edges) for t in time_list], dtype=torch.long, device=device,
            )
            return loss_fn(logits, y, (1 - status).long())  # c: 1=censored, 0=event
        return loss_fn(logits.view(-1), status, time)  # cox partial likelihood over the batch

    @torch.no_grad()
    def _val_loss() -> float:
        # Whole val set as one batch (cox: a single risk set; nll: mean NLL).
        model.eval()
        return float(_batch_loss(val).item())

    @torch.no_grad()
    def _risk_records(samples: list[dict]) -> dict:
        """Per-sample risk scores for this split (CR-3: pooled across folds)."""
        model.eval()
        risks, statuses, times, pids = [], [], [], []
        for s in samples:
            lg = _bag_logits(model, _load_feats(s["pt_path"], device)).unsqueeze(0)
            r = _risk_from_logits(lg, loss_type)
            risks.append(float(r.item()))
            statuses.append(float(s["status"]))
            times.append(float(s["time"]))
            pids.append(s["patient_id"])
        return {"risks": risks, "statuses": statuses, "times": times,
                "patient_ids": pids}

    def _c_index_from(records: dict) -> float:
        # survival_c_index aggregates to patient level and is NaN-safe.
        ci = survival_c_index(
            torch.tensor(records["risks"], dtype=torch.float32),
            torch.tensor(records["statuses"], dtype=torch.float32),
            torch.tensor(records["times"], dtype=torch.float32),
            records["patient_ids"],
        )
        return float(ci) if ci is not None else float("nan")

    def _c_index(samples: list[dict]) -> float:
        return _c_index_from(_risk_records(samples))

    rng = random.Random(exp_cfg.train.seed)
    for epoch in range(exp_cfg.train.max_epochs):
        model.train()
        order = train[:]
        rng.shuffle(order)
        for i in range(0, len(order), _SURV_BATCH):  # each chunk = one risk set
            optimizer.zero_grad()
            loss = _batch_loss(order[i:i + _SURV_BATCH])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        v_loss, v_cidx = _val_loss(), _c_index(val)
        print(
            f"    [CLAM-surv fold {fold}] epoch {epoch + 1}: "
            f"val_loss={v_loss:.4f} val_c_index={v_cidx:.4f}"
        )
        early_stopping(v_loss, v_cidx, model, epoch=epoch)
        default_stop = _should_stop(exp_cfg.train, early_stopping)
        if policy_runtime.should_stop(
            default_stop,
            epoch=epoch,
            metrics={"val_loss": v_loss, "val_c_index": v_cidx},
        ):
            break

    # Restore the val-loss-selected best checkpoint before final scoring.
    restored = False
    best_path = os.path.join(fold_dir, f"best_{model_type}.pth")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        restored = True
    elif getattr(early_stopping, "best_model_state", None) is not None:
        model.load_state_dict(early_stopping.best_model_state)
        restored = True
    # A3: source=best when a val-selected checkpoint was restored above,
    # source=final when the final weights were kept (no restore).
    print(f"[selected] epoch={early_stopping.best_epoch} "
          f"source={'best' if restored else 'final'}", flush=True)

    # CR-3: export the val risk records so the runner can score concordance over
    # the POOLED cross-fold validation set. The per-fold c-index below stays for
    # reporting; the pooled value is what the selection primary_value uses.
    _val_records = _risk_records(val)
    # A4': persist the selected model's val risk scores (the arrays are already
    # in hand) so the fold carries a hashable no-op detector.
    val_predictions_path = os.path.join(fold_dir, "predictions_val.csv")
    write_survival_predictions_csv(val_predictions_path, _val_records)
    test_metrics = {"c_index": _c_index(test)}
    val_metrics = {"c_index": _c_index_from(_val_records)}
    fold_result = {
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
        "val_records": _val_records,
        "val_predictions_sha256": file_sha256_or_none(val_predictions_path),
        "fold": fold,
        "elapsed_seconds": time.time() - start,
    }
    with open(metrics_path, "w") as f:
        json.dump(fold_result, f, indent=2)
    return fold_result
