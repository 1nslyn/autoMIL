"""Single-fold training for the TITAN linear probe.

Standard CE + Adam on frozen slide embeddings, early-stopping on val AUC,
final evaluation via the SAME ``compute_extended_metrics`` every other arm
uses -- so ``test_auc``/``test_bacc`` are computed by identical code
across all four models (design spec §4, §7).
"""

from __future__ import annotations

import copy
import json
import os
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from autobench.pipeline.config import ExperimentConfig
from autobench.pipeline.determinism import seed_everything as _seed_everything
from autobench.pipeline.evaluate import (
    compute_extended_metrics,
    file_sha256,
    write_predictions_csv,
)
from autobench.pipeline.policy_dispatch import PolicyRuntime
from autobench.pipeline.titan.config import TitanHeadConfig, resolve_head_config
from autobench.pipeline.titan.dataset import TitanSlideDataset
from autobench.pipeline.titan.model import TitanLinearProbe


@torch.no_grad()
def _evaluate(
    model: TitanLinearProbe,
    loader: DataLoader,
    device: torch.device,
    n_classes: int,
    ordinal: bool = False,
    predictions_path: str | None = None,
) -> dict[str, float]:
    """Run the probe over a split and compute the shared extended metrics."""
    model.eval()
    all_probs: list[np.ndarray] = []
    all_labels: list[int] = []

    for embeddings, labels in loader:
        embeddings = embeddings.to(device)
        logits = model(embeddings)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.extend(labels.tolist())

    y_probs = np.concatenate(all_probs, axis=0)
    y_true = np.array(all_labels, dtype=int)
    y_pred = y_probs.argmax(axis=1)

    if predictions_path:
        # TITAN's loader yields (embeddings, labels) only -- no slide ids to
        # carry through -- so rows fall back to positional sample_<i>. Still
        # enough for any confusion-matrix metric; just not joinable by slide.
        write_predictions_csv(predictions_path, None, y_true, y_probs, y_pred)
    return compute_extended_metrics(y_true, y_probs, y_pred, n_classes, ordinal=ordinal)


def train_titan_fold(
    exp_cfg: ExperimentConfig,
    train_ds: TitanSlideDataset,
    val_ds: TitanSlideDataset,
    test_ds: TitanSlideDataset,
    fold: int,
    results_dir: str,
    device: str = "cuda:0",
    head_cfg: TitanHeadConfig | None = None,
    policy_runtime: PolicyRuntime | None = None,
    ordinal: bool = False,
) -> dict:
    """Train and evaluate one fold of a TITAN linear probe.

    Writes ``<results_dir>/fold_<i>/metrics.json`` in the SAME schema as
    every other arm's per-fold result (``test_metrics``, ``val_metrics``,
    ``fold``) and returns that dict.

    Seed follows the shared convention: ``train.seed + fold`` (matches
    nnMIL/DTFD).
    """
    # H-3: head-side filtering only — the opaque channel's max_epochs/
    # early_stopping were already applied onto exp_cfg.train at the RUNNER
    # level (apply_train_overrides, before config.json was saved), so
    # exp_cfg.train is read here as already-effective.
    head_cfg = resolve_head_config(exp_cfg, head_cfg)

    fold_dir = os.path.join(results_dir, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)
    metrics_path = os.path.join(fold_dir, "metrics.json")

    # Resume: skip if already completed
    if os.path.exists(metrics_path):
        print(f"\n    [fold {fold}] Already completed, loading from disk")
        with open(metrics_path) as f:
            return json.load(f)

    seed = exp_cfg.train.seed + fold
    _seed_everything(seed)

    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    n_classes = exp_cfg.task.n_classes
    ordinal = exp_cfg.task.ordinal

    model = TitanLinearProbe(exp_cfg.embed_dim, n_classes).to(torch_device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=head_cfg.lr, weight_decay=head_cfg.weight_decay,
    )
    policy_runtime = policy_runtime or PolicyRuntime()
    optimizer = policy_runtime.wrap_optimizer(optimizer)
    criterion = nn.CrossEntropyLoss()

    batch_size = min(32, len(train_ds)) or 1
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=max(1, len(val_ds)), shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=max(1, len(test_ds)), shuffle=False)

    best_val_auc = -float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = -1  # -1: never improved; the pre-training snapshot is kept
    epochs_without_improvement = 0

    start = time.time()
    for _epoch in range(exp_cfg.train.max_epochs):
        model.train()
        for embeddings, labels in train_loader:
            embeddings = embeddings.to(torch_device)
            labels = labels.to(torch_device)

            optimizer.zero_grad()
            logits = model(embeddings)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        val_metrics = _evaluate(model, val_loader, torch_device, n_classes, ordinal=ordinal,
                            predictions_path=os.path.join(fold_dir, "predictions_val.csv"))
        val_auc = val_metrics["auc_roc"]
        # NaN AUC (e.g. a val split missing a class) can't drive early
        # stopping -- treat it as "no improvement" rather than crashing.
        improved = not np.isnan(val_auc) and val_auc > best_val_auc

        if improved:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = _epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        default_stop = (
            exp_cfg.train.early_stopping
            and epochs_without_improvement >= head_cfg.patience
        )
        if policy_runtime.should_stop(
            default_stop, epoch=_epoch, metrics={"val_auc": val_auc},
        ):
            break

    model.load_state_dict(best_state)
    # A3: this arm ALWAYS restores best_state. source=best when some epoch
    # improved on it; source=untrained when the restored snapshot is the
    # pre-loop deepcopy that predates any training step (best_epoch == -1,
    # e.g. an all-NaN val split) — NOT the "final weights kept" of the arms
    # that print source=final.
    print(f"[selected] epoch={best_epoch} "
          f"source={'best' if best_epoch >= 0 else 'untrained'}", flush=True)
    test_metrics = _evaluate(model, test_loader, torch_device, n_classes, ordinal=ordinal,
                             predictions_path=os.path.join(fold_dir, "predictions.csv"))
    val_metrics = _evaluate(model, val_loader, torch_device, n_classes, ordinal=ordinal,
                            predictions_path=os.path.join(fold_dir, "predictions_val.csv"))

    fold_result = {
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
        # A4': no-op detector — hash of the persisted val predictions above.
        "val_predictions_sha256": file_sha256(
            os.path.join(fold_dir, "predictions_val.csv")
        ),
        "fold": fold,
        # FOLD-TIMING CONTRACT: covers the whole fold, final evaluation
        # included, so the number is comparable across arms and task types.
        "elapsed_seconds": time.time() - start,
    }

    with open(metrics_path, "w") as f:
        json.dump(fold_result, f, indent=2)

    return fold_result
