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
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from autobench.pipeline.hparams import apply_overrides, overrides_from_exp_cfg
from autobench.pipeline.config import ExperimentConfig
from autobench.pipeline.evaluate import compute_extended_metrics
from autobench.pipeline.titan.config import TitanHeadConfig
from autobench.pipeline.titan.dataset import TitanSlideDataset
from autobench.pipeline.titan.model import TitanLinearProbe


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _evaluate(
    model: TitanLinearProbe,
    loader: DataLoader,
    device: torch.device,
    n_classes: int,
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

    return compute_extended_metrics(y_true, y_probs, y_pred, n_classes)


def train_titan_fold(
    exp_cfg: ExperimentConfig,
    train_ds: TitanSlideDataset,
    val_ds: TitanSlideDataset,
    test_ds: TitanSlideDataset,
    fold: int,
    results_dir: str,
    device: str = "cuda:0",
    head_cfg: TitanHeadConfig | None = None,
) -> dict:
    """Train and evaluate one fold of a TITAN linear probe.

    Writes ``<results_dir>/fold_<i>/metrics.json`` in the SAME schema as
    every other arm's per-fold result (``test_metrics``, ``val_metrics``,
    ``fold``) and returns that dict.

    Seed follows the shared convention: ``train.seed + fold`` (matches
    nnMIL/DTFD).
    """
    if head_cfg is None:
        head_cfg = TitanHeadConfig()
    # H-3: TitanHeadConfig stays the source of truth for lr/weight_decay/
    # patience; layer on only the explicitly-set overrides. max_epochs and
    # early_stopping are deliberately excluded — this arm reads those straight
    # off exp_cfg.train (its documented mixed provenance), so routing them here
    # would double-apply and trip the fail-loud guard.
    _titan_ov = {k: v for k, v in overrides_from_exp_cfg(exp_cfg).items()
                 if k not in ("max_epochs", "early_stopping")}
    head_cfg = apply_overrides(head_cfg, _titan_ov, arm="titan")

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

    model = TitanLinearProbe(exp_cfg.embed_dim, n_classes).to(torch_device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=head_cfg.lr, weight_decay=head_cfg.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    batch_size = min(32, len(train_ds)) or 1
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=max(1, len(val_ds)), shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=max(1, len(test_ds)), shuffle=False)

    best_val_auc = -float("inf")
    best_state = copy.deepcopy(model.state_dict())
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

        val_metrics = _evaluate(model, val_loader, torch_device, n_classes)
        val_auc = val_metrics["auc_roc"]
        # NaN AUC (e.g. a val split missing a class) can't drive early
        # stopping -- treat it as "no improvement" rather than crashing.
        improved = not np.isnan(val_auc) and val_auc > best_val_auc

        if improved:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if exp_cfg.train.early_stopping and epochs_without_improvement >= head_cfg.patience:
            break

    elapsed = time.time() - start

    model.load_state_dict(best_state)
    test_metrics = _evaluate(model, test_loader, torch_device, n_classes)
    val_metrics = _evaluate(model, val_loader, torch_device, n_classes)

    fold_result = {
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
        "fold": fold,
        "elapsed_seconds": elapsed,
    }

    with open(metrics_path, "w") as f:
        json.dump(fold_result, f, indent=2)

    return fold_result
