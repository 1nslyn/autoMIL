"""Adapter-side survival training for the TITAN linear probe.

Reuses ``TitanLinearProbe`` but trains it with a survival loss, mirroring
``clam/survival_train.py``: cox/nllsurv loss, val-loss model selection,
event-time nllsurv bins, patient-level sksurv c-index. TITAN already batches
via ``DataLoader`` (no bag/attention -- one frozen embedding per slide), so
each training minibatch IS the risk set; no manual per-slide loop is needed
(unlike CLAM/ABMIL).
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from autobench import LIB_ROOT
from autobench.pipeline.config import ExperimentConfig
from autobench.pipeline.determinism import seed_everything as _seed_everything
from autobench.pipeline.evaluate import file_sha256, write_survival_predictions_csv
from autobench.pipeline.titan.config import TitanHeadConfig, resolve_head_config
from autobench.pipeline.titan.dataset import TitanSurvivalDataset
from autobench.pipeline.titan.model import TitanLinearProbe
from autobench.pipeline.policy_dispatch import PolicyRuntime

# The framework-agnostic survival core lives under the vendored nnMIL tree;
# import it adapter -> lib (the normal autobench direction).
if str(LIB_ROOT) not in sys.path:
    sys.path.insert(0, str(LIB_ROOT))
from nnMIL.training.losses.survival_loss import SurvivalLoss, survival_c_index  # noqa: E402
from nnMIL.training.losses.survival_loss_nll import NLLSurvLoss  # noqa: E402
from nnMIL.training.callbacks.early_stopping import EarlyStoppingSurvival  # noqa: E402


def _event_time_bin_edges(times, statuses, n_bins: int) -> np.ndarray:
    """NLL bin edges from event (uncensored) train times — same PORPOISE/MCAT
    convention as the CLAM/ABMIL survival trainers."""
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


@torch.no_grad()
def _predict_risks(model, loader: DataLoader, device: torch.device, loss_type: str):
    model.eval()
    risks, statuses, times, pids = [], [], [], []
    for embeddings, status, time_, pid in loader:
        embeddings = embeddings.to(device)
        logits = model(embeddings)
        r = _risk_from_logits(logits, loss_type)
        risks.extend(float(v) for v in r.cpu().tolist())
        statuses.extend(int(s) for s in status.tolist())
        times.extend(float(t) for t in time_.tolist())
        pids.extend(list(pid))
    return risks, statuses, times, pids


def train_titan_survival_fold(
    exp_cfg: ExperimentConfig,
    train_ds: TitanSurvivalDataset,
    val_ds: TitanSurvivalDataset,
    test_ds: TitanSurvivalDataset,
    fold: int,
    results_dir: str,
    device: str = "cuda:0",
    head_cfg: TitanHeadConfig | None = None,
    policy_runtime: PolicyRuntime | None = None,
) -> dict:
    """Train and evaluate one TITAN survival fold.

    Mirrors ``train_titan_fold``'s resume/``metrics.json`` contract and
    ``clam/survival_train.py``'s cox/nllsurv loss + val-loss early stopping +
    patient-level c-index, adapted to TITAN's ``DataLoader``-batched linear
    probe.
    """
    # H-3: mixed provenance, resolved through one seam — head knobs land on
    # TitanHeadConfig, the opaque channel's max_epochs/early_stopping on
    # exp_cfg.train (see resolve_head_config).
    head_cfg = resolve_head_config(exp_cfg, head_cfg)

    fold_dir = os.path.join(results_dir, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)
    metrics_path = os.path.join(fold_dir, "metrics.json")

    if os.path.exists(metrics_path):
        print(f"\n    [TITAN-surv fold {fold}] already completed, loading from disk")
        with open(metrics_path) as f:
            return json.load(f)

    seed = exp_cfg.train.seed + fold
    _seed_everything(seed)

    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    survival_loss = exp_cfg.survival_loss or "cox"
    is_nll = survival_loss == "nllsurv"
    nll_bins = exp_cfg.task.nll_bins
    n_out = nll_bins if is_nll else 1

    model = TitanLinearProbe(exp_cfg.embed_dim, n_out).to(torch_device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=head_cfg.lr, weight_decay=head_cfg.weight_decay,
    )
    policy_runtime = policy_runtime or PolicyRuntime()
    optimizer = policy_runtime.wrap_optimizer(optimizer)

    if is_nll:
        loss_fn = NLLSurvLoss()
        edges = _event_time_bin_edges(train_ds.times, train_ds.statuses, nll_bins)
    else:
        loss_fn = SurvivalLoss(loss_type=survival_loss)
        edges = None

    batch_size = min(32, len(train_ds)) or 1
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=max(1, len(val_ds)), shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=max(1, len(test_ds)), shuffle=False)

    early_stopping = EarlyStoppingSurvival(
        patience=head_cfg.patience, verbose=True, metric="c_index",
        save_dir=fold_dir, model_type="titan", mode="min",
    )

    def _batch_loss(embeddings: torch.Tensor, status: torch.Tensor, time_: torch.Tensor) -> torch.Tensor:
        embeddings = embeddings.to(torch_device)
        status_t = status.to(torch_device).float()
        time_t = time_.to(torch_device).float()
        logits = model(embeddings)
        if is_nll:
            y = torch.tensor(
                [_time_to_bin(float(t), edges) for t in time_.tolist()],
                dtype=torch.long, device=torch_device,
            )
            return loss_fn(logits, y, (1 - status_t).long())  # c: 1=censored, 0=event
        return loss_fn(logits.view(-1), status_t, time_t)  # cox partial likelihood over the batch

    @torch.no_grad()
    def _val_loss() -> float:
        # Whole val set as one batch (cox: a single risk set; nll: mean NLL).
        model.eval()
        for embeddings, status, time_, _pid in val_loader:
            return float(_batch_loss(embeddings, status, time_).item())
        return float("nan")

    def _risk_records(loader: DataLoader) -> dict:
        """Per-sample risk scores (CR-3: pooled across folds by the runner)."""
        risks, statuses, times, pids = _predict_risks(model, loader, torch_device, survival_loss)
        return {
            "risks": [float(r) for r in risks],
            "statuses": [float(s) for s in statuses],
            "times": [float(t) for t in times],
            "patient_ids": list(pids),
        }

    def _c_index(loader: DataLoader) -> float:
        risks, statuses, times, pids = _predict_risks(model, loader, torch_device, survival_loss)
        if not risks:
            return float("nan")
        # survival_c_index aggregates to patient level and is NaN-safe.
        ci = survival_c_index(
            torch.tensor(risks, dtype=torch.float32),
            torch.tensor(statuses, dtype=torch.float32),
            torch.tensor(times, dtype=torch.float32),
            pids,
        )
        return float(ci) if ci is not None else float("nan")

    start = time.time()
    for epoch in range(exp_cfg.train.max_epochs):
        model.train()
        for embeddings, status, time_, _pid in train_loader:
            optimizer.zero_grad()
            loss = _batch_loss(embeddings, status, time_)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        v_loss, v_cidx = _val_loss(), _c_index(val_loader)
        print(
            f"    [TITAN-surv fold {fold}] epoch {epoch + 1}: "
            f"val_loss={v_loss:.4f} val_c_index={v_cidx:.4f}"
        )
        # Always save the best (val-loss) checkpoint; early_stopping only gates
        # stopping early (matches classification/DTFD).
        early_stopping(v_loss, v_cidx, model, epoch=epoch)
        default_stop = exp_cfg.train.early_stopping and early_stopping.early_stop
        if policy_runtime.should_stop(
            default_stop,
            epoch=epoch,
            metrics={"val_loss": v_loss, "val_c_index": v_cidx},
        ):
            break

    # Restore the best (val-loss) checkpoint from disk before scoring: the
    # in-memory best_model_state is a shallow copy aliasing the live params,
    # so it decays to the last epoch's weights. Mirrors CLAM.
    best_path = os.path.join(fold_dir, "best_titan.pth")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=torch_device))
    elif getattr(early_stopping, "best_model_state", None) is not None:
        model.load_state_dict(early_stopping.best_model_state)
    print(f"[selected] epoch={early_stopping.best_epoch}", flush=True)

    test_metrics = {"c_index": _c_index(test_loader)}
    val_metrics = {"c_index": _c_index(val_loader)}
    # CR-3: pooled cross-fold val concordance is computed by the runner.
    val_records = _risk_records(val_loader)
    # A4': persist the selected model's val risk scores (the arrays are already
    # in hand) so the fold carries a hashable no-op detector.
    val_predictions_path = os.path.join(fold_dir, "predictions_val.csv")
    write_survival_predictions_csv(val_predictions_path, val_records)
    fold_result = {
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
        "val_records": val_records,
        # A4': no-op detector — hash of the persisted val risk scores above.
        "val_predictions_sha256": file_sha256(val_predictions_path),
        "fold": fold,
        # FOLD-TIMING CONTRACT: covers the whole fold, checkpoint restore and
        # final scoring included, so the number is comparable across arms and
        # task types.
        "elapsed_seconds": time.time() - start,
    }

    with open(metrics_path, "w") as f:
        json.dump(fold_result, f, indent=2)

    return fold_result
