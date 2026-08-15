"""Adapter-side survival training for the ABMIL backbone.

Reuses ABMIL's model classes but trains them with a survival loss in a custom
loop.
"""

from __future__ import annotations

import os
import random
import sys
import time

import numpy as np
import torch

from autobench import LIB_ROOT
from autobench.pipeline.abmil.config import ABMILConfig
from autobench.pipeline.abmil.dataset import ABMILSurvivalSlide, _read_bag
from autobench.pipeline.abmil.model import build_abmil_model
from autobench.pipeline.evaluate import write_survival_predictions_csv
from autobench.pipeline.policy_dispatch import PolicyRuntime

# The framework-agnostic survival core lives under the vendored nnMIL tree;
# import it adapter -> lib (the normal autobench direction).
if str(LIB_ROOT) not in sys.path:
    sys.path.insert(0, str(LIB_ROOT))
from nnMIL.training.losses.survival_loss import SurvivalLoss, survival_c_index  # noqa: E402
from nnMIL.training.losses.survival_loss_nll import NLLSurvLoss  # noqa: E402
from nnMIL.training.callbacks.early_stopping import EarlyStoppingSurvival  # noqa: E402

# Slides per optimizer step. For cox this is the risk set; each slide is a
# full bag held in the graph until backward, so keep it modest (same as CLAM).
_SURV_BATCH = 16


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _bag_logits(model, features: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Bag-level logits ``(n_out,)`` from one slide's full H5 bag.

    ABMIL's ``forward`` takes a batched ``[B, N, in_dim]`` tensor and returns
    a dict with key ``'logits'`` — unwrap both the batch dim (B=1, one slide
    per call) and the dict.
    """
    x = features.to(device).unsqueeze(0)  # [1, N, in_dim]
    return model(x)["logits"].view(-1)


def train_abmil_survival_fold(
    model_type: str,
    train_samples: list[ABMILSurvivalSlide],
    val_samples: list[ABMILSurvivalSlide],
    test_samples: list[ABMILSurvivalSlide],
    embed_dim: int,
    survival_loss: str,
    nll_bins: int,
    cfg: ABMILConfig,
    device: torch.device,
    seed: int,
    fold_dir: str,
    policy_runtime: PolicyRuntime | None = None,
) -> dict:
    """Train one ABMIL survival fold; return shared-schema c-index metrics.

    Mirrors ``clam/survival_train.py::train_survival_fold`` (same cox/nllsurv
    loss, val-loss early stopping, patient-level c-index), adapted to ABMIL's
    dict-returning ``forward`` and the pre-loaded-slide-list calling
    convention already used by ``train_abmil_fold`` for classification.
    """
    grad_was_enabled = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    try:
        _seed(seed)
        if not train_samples:
            raise ValueError("ABMIL survival train split is empty; cannot train a fold.")

        is_nll = survival_loss == "nllsurv"
        n_out = nll_bins if is_nll else 1

        print(
            f"    [ABMIL-surv] {model_type}/{survival_loss} — "
            f"train={len(train_samples)} val={len(val_samples)} test={len(test_samples)} slides"
        )

        model = build_abmil_model(
            model_type, in_dim=embed_dim, num_classes=n_out,
            M=cfg.M, L=cfg.L, dropout=cfg.dropout,
        ).to(device)

        if is_nll:
            loss_fn = NLLSurvLoss()
            edges = _event_time_bin_edges(
                [s.time for s in train_samples], [s.status for s in train_samples], nll_bins,
            )
        else:
            loss_fn = SurvivalLoss(loss_type=survival_loss)
            edges = None

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        policy_runtime = policy_runtime or PolicyRuntime()
        optimizer = policy_runtime.wrap_optimizer(optimizer)
        early_stopping = EarlyStoppingSurvival(
            patience=cfg.patience, verbose=True, metric="c_index",
            save_dir=fold_dir, model_type=model_type, mode="min",
        )

        def _batch_loss(batch: list[ABMILSurvivalSlide]) -> torch.Tensor:
            logits_list = [_bag_logits(model, _read_bag(s.h5_path), device) for s in batch]
            logits = torch.stack(logits_list)  # (B, n_out)
            status = torch.tensor([s.status for s in batch], dtype=torch.float32, device=device)
            time_t = torch.tensor([s.time for s in batch], dtype=torch.float32, device=device)
            if is_nll:
                y = torch.tensor(
                    [_time_to_bin(s.time, edges) for s in batch], dtype=torch.long, device=device,
                )
                return loss_fn(logits, y, (1 - status).long())  # c: 1=censored, 0=event
            return loss_fn(logits.view(-1), status, time_t)  # cox partial likelihood over the batch

        @torch.no_grad()
        def _val_loss() -> float:
            model.eval()
            return float(_batch_loss(val_samples).item())

        @torch.no_grad()
        def _risk_records(samples: list[ABMILSurvivalSlide]) -> dict:
            """Per-sample risk scores (CR-3: pooled across folds by the runner)."""
            model.eval()
            risks, statuses, times, pids = [], [], [], []
            for s in samples:
                lg = _bag_logits(model, _read_bag(s.h5_path), device).unsqueeze(0)
                r = _risk_from_logits(lg, survival_loss)
                risks.append(float(r.item()))
                statuses.append(float(s.status))
                times.append(float(s.time))
                pids.append(s.patient_id)
            return {"risks": risks, "statuses": statuses, "times": times,
                    "patient_ids": pids}

        def _c_index_from(records: dict) -> float:
            if not records["risks"]:
                return float("nan")
            ci = survival_c_index(
                torch.tensor(records["risks"], dtype=torch.float32),
                torch.tensor(records["statuses"], dtype=torch.float32),
                torch.tensor(records["times"], dtype=torch.float32),
                records["patient_ids"],
            )
            return float(ci) if ci is not None else float("nan")

        def _c_index(samples: list[ABMILSurvivalSlide]) -> float:
            return _c_index_from(_risk_records(samples))

        rng = random.Random(seed)
        best_epoch = -1  # -1: no val-selected checkpoint; final weights kept
        start = time.time()
        for epoch in range(cfg.max_epochs):
            model.train()
            order = train_samples[:]
            rng.shuffle(order)
            for i in range(0, len(order), _SURV_BATCH):  # each chunk = one risk set
                optimizer.zero_grad()
                loss = _batch_loss(order[i:i + _SURV_BATCH])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            if val_samples:
                v_loss, v_cidx = _val_loss(), _c_index(val_samples)
                print(
                    f"    [ABMIL-surv fold] epoch {epoch + 1}: "
                    f"val_loss={v_loss:.4f} val_c_index={v_cidx:.4f}"
                )
                # Always save the best (val-loss) checkpoint; cfg.early_stopping
                # only gates stopping early (matches classification/DTFD).
                early_stopping(v_loss, v_cidx, model)
                if early_stopping.counter == 0:  # saved a new best this epoch
                    best_epoch = epoch
                default_stop = cfg.early_stopping and early_stopping.early_stop
                if policy_runtime.should_stop(
                    default_stop,
                    epoch=epoch,
                    metrics={"val_loss": v_loss, "val_c_index": v_cidx},
                ):
                    break

        # Restore the best (val-loss) checkpoint from disk before scoring: the
        # in-memory best_model_state is a shallow copy aliasing the live params,
        # so it decays to the last epoch's weights. Mirrors CLAM.
        best_path = os.path.join(fold_dir, f"best_{model_type}.pth")
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, map_location=device))
        elif getattr(early_stopping, "best_model_state", None) is not None:
            model.load_state_dict(early_stopping.best_model_state)
        print(f"[selected] epoch={best_epoch}", flush=True)

        # CR-3: export val risk records so the runner can pool concordance
        # across folds instead of averaging five ~2-event c-indices.
        _val_records = _risk_records(val_samples) if val_samples else {
            "risks": [], "statuses": [], "times": [], "patient_ids": []
        }
        # A4': persist the selected model's val risk scores (already in hand)
        # so the fold carries a hashable no-op detector.
        write_survival_predictions_csv(
            os.path.join(fold_dir, "predictions_val.csv"), _val_records,
        )
        test_metrics = {
            "c_index": _c_index(test_samples) if test_samples else float("nan")
        }
        val_metrics = {"c_index": _c_index_from(_val_records)}
        return {
            "test_metrics": test_metrics,
            "val_metrics": val_metrics,
            "val_records": _val_records,
            # FOLD-TIMING CONTRACT: covers the whole fold, checkpoint restore
            # and final scoring included, so the number is comparable across
            # arms and task types.
            "elapsed_seconds": time.time() - start,
        }
    finally:
        torch.set_grad_enabled(grad_was_enabled)
