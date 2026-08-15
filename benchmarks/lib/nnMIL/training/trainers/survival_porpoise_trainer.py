#!/usr/bin/env python3
"""
Survival Porpoise Trainer for nnMIL
Handles survival tasks with NLLSurv loss, batch_size=1
"""

import os
import sys
import time as time_module
import json
import functools
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nnMIL.training.trainers.base_trainer import BaseTrainer
from nnMIL.network_architecture.model_factory import create_mil_model
from nnMIL.utilities.utils import cosine_lr
from nnMIL.utilities.plan_loader import create_dataset_from_plan
from nnMIL.training.losses.survival_loss import SurvivalLoss, survival_c_index
from nnMIL.training.losses.survival_loss_nll import NLLSurvLoss
from nnMIL.training.callbacks.early_stopping import EarlyStoppingSurvival


def map_time_to_bins_with_edges(time_tensor: torch.Tensor, edges: np.ndarray) -> torch.Tensor:
    """Map times to bin indices using precomputed global edges"""
    device = time_tensor.device
    t_np = np.atleast_1d(time_tensor.detach().float().cpu().numpy())
    bins = np.digitize(t_np, edges[1:-1], right=False)
    bins = np.clip(bins, 0, len(edges) - 2).astype(np.int64)
    bins_t = torch.from_numpy(bins).to(device=device, dtype=torch.long)
    return bins_t.view(time_tensor.shape)


def bs1_collate(batch):
    """Collate function for batch_size=1"""
    return batch[0]


def _surv_worker_init(worker_id, seed):
    """Picklable DataLoader worker init (spawn-safe): seed numpy per worker."""
    np.random.seed(seed + worker_id)


class SurvivalPorpoiseTrainer(BaseTrainer):
    """Trainer for survival tasks with NLLSurv loss (batch_size=1)"""
    
    def __init__(self, plan_path, model_type, fold=None, **kwargs):
        super().__init__(plan_path, model_type, fold, **kwargs)
        
        # Survival Porpoise-specific initialization
        self.survival_loss = kwargs.get('survival_loss', 'nllsurv')
        self.nll_bins = kwargs.get('nll_bins', 4)
        self.nll_bin_edges = None
        
        self.logger.info(f"Survival Porpoise task initialized")
        self.logger.info(f"Survival loss: {self.survival_loss}, NLL bins: {self.nll_bins}")
    
    def create_model(self):
        """Create survival model (NLLSurv: num_classes = nll_bins)"""
        # NLLSurv model outputs discrete hazard probabilities for each time bin
        self.model = create_mil_model(
            model_type=self.model_type,
            input_dim=self.config['feature_dimension'],
            hidden_dim=self.config['hidden_dim'],
            num_classes=self.nll_bins,  # NLLSurv: output per time bin
            dropout=self.config['dropout']
        )
        self.model = self.model.to(self.device)
        self.logger.info(f"Model created: {self.model_type} (NLLSurv with {self.nll_bins} bins)")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        return self.model
    
    def create_data_loaders(self):
        """Create data loaders for survival porpoise task (batch_size=1)"""
        self.logger.info("Loading datasets from plan file...")
        
        train_dataset = create_dataset_from_plan(self.plan_path, split='train', fold=self.fold)
        val_dataset = create_dataset_from_plan(self.plan_path, split='val', fold=self.fold)
        test_dataset = create_dataset_from_plan(self.plan_path, split='test', fold=self.fold)
        
        self.logger.info(f"Datasets loaded - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        # Compute NLL bin edges from EVENT (uncensored) training times only.
        # This follows the original PORPOISE/MCAT convention: discretizing on
        # event times keeps censored outliers (including any corrupted/negative
        # follow-up times) from distorting the quantile boundaries. Falls back
        # to all training times when a fold has too few events to define bins.
        all_status, all_times = [], []
        for i in range(len(train_dataset)):
            sample = train_dataset[i]
            s, t = sample[3], sample[4]
            all_status.append(s.item() if hasattr(s, "item") else float(s))
            all_times.append(t.item() if hasattr(t, "item") else float(t))
        all_times = np.array(all_times)
        event_times = all_times[np.array(all_status) == 1]
        if len(np.unique(event_times)) >= self.nll_bins:
            bin_src = event_times
            self.logger.info(
                f"NLL bins from {len(event_times)} event (uncensored) train times"
            )
        else:
            bin_src = all_times
            self.logger.warning(
                f"Only {len(np.unique(event_times))} unique event times "
                f"(< nll_bins={self.nll_bins}); falling back to all "
                f"{len(all_times)} train times for NLL binning"
            )
        qs = np.linspace(0, 1, self.nll_bins + 1)
        edges = np.quantile(bin_src, qs)
        # Ensure strictly increasing
        eps = 1e-6
        for i in range(1, len(edges)):
            if edges[i] <= edges[i-1]:
                edges[i] = edges[i-1] + eps
        self.nll_bin_edges = edges
        self.logger.info(f"NLL bin edges computed: {self.nll_bin_edges}")
        
        # Create data loaders with batch_size=1
        self.train_loader = DataLoader(
            train_dataset, batch_size=1, shuffle=True,
            num_workers=4, worker_init_fn=functools.partial(_surv_worker_init, seed=self.seed),
            collate_fn=bs1_collate
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False,
            num_workers=4, worker_init_fn=functools.partial(_surv_worker_init, seed=self.seed),
            collate_fn=bs1_collate
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False,
            num_workers=4, worker_init_fn=functools.partial(_surv_worker_init, seed=self.seed),
            collate_fn=bs1_collate
        )
        
        self.logger.info(f"Data loaders created - Train: {len(self.train_loader)}, Val: {len(self.val_loader)}, Test: {len(self.test_loader)}")
        
        return self.train_loader, self.val_loader, self.test_loader
    
    def train(self):
        """Run training loop (batch_size=1, NLLSurv loss)"""
        # CRITICAL: Reset PyTorch global state before training
        # This is especially important when running multiple folds sequentially
        torch.set_grad_enabled(True)
        
        # Clear CUDA cache to avoid memory issues between folds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if self.model is None:
            self.create_model()
        
        # Ensure model is in training mode and all parameters require grad
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True
        
        if self.train_loader is None:
            self.create_data_loaders()
        
        # Save training configuration
        self.save_training_config()
        
        # Setup optimizer
        named_parameters = list(self.model.named_parameters())
        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or "logit_scale" in n
        include = lambda n, p: not exclude(n, p)
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
        
        optimizer = torch.optim.AdamW([
            {"params": gain_or_bias_params, "weight_decay": 0.0},
            {"params": rest_params, "weight_decay": self.config.get('weight_decay', 0.01)}
        ], lr=self.config.get('learning_rate', 1e-4))
        if self.policy_runtime is not None:
            optimizer = self.policy_runtime.wrap_optimizer(optimizer)
        
        # Setup loss function (NLLSurv)
        loss_fn = NLLSurvLoss()
        
        # Setup LR scheduler
        num_epochs = self.config.get('num_epochs', 100)
        total_steps = len(self.train_loader) * num_epochs
        warmup_steps = len(self.train_loader) * self.config.get('warmup_epochs', 5)
        lr_scheduler = cosine_lr(optimizer, self.config.get('learning_rate', 1e-4), warmup_steps, total_steps)
        
        # Setup early stopping
        metric = self.dataset_info.get('metric', 'c_index')
        early_stopping = EarlyStoppingSurvival(
            patience=self.config.get('patience', 10),
            verbose=True,
            metric=metric,
            save_dir=self.save_dir,
            model_type=self.model_type,
            logger=self.logger,
            mode='min',  # select on val NLL loss: val c-index is near-random with few events
        )
        
        # Setup mixed precision
        device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
        # For bfloat16, GradScaler is typically not needed due to sufficient dynamic range
        # Disable GradScaler to avoid the "No inf checks were recorded" error
        fp16_scaler = None
        
        # Training loop
        self.model.train()
        global_step = 0
        best_epoch = -1  # epoch of the best checkpoint restored at fold end

        self.logger.info(f"Starting training for {num_epochs} epochs (batch_size=1, NLLSurv)")

        for epoch in tqdm(range(num_epochs), desc="Training"):
            epoch_start_time = time_module.time()
            self.model.train()
            epoch_loss = 0.0
            epoch_steps = 0
            
            current_lr = optimizer.param_groups[0]['lr']
            self.logger.info(f"Epoch {epoch+1}/{num_epochs} - Learning Rate: {current_lr:.2e}")
            if self.writer:
                self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
            for batch_idx, batch in enumerate(pbar):
                # Survival dataset now returns 7 items: features, coords, bag_sizes, status, time, patient_ids, slide_ids
                if len(batch) == 7:
                    features, coords, bag_sizes, status, time, patient_ids, slide_ids = batch
                else:
                    # Old format with 6 items
                    features, coords, bag_sizes, status, time, patient_ids = batch

                features = features.to(self.device)
                if features.dim() == 2:
                    features = features.unsqueeze(0)
                status = status.to(self.device)
                time = time.to(self.device)
                
                optimizer.zero_grad()
                
                with torch.amp.autocast(device_type, dtype=torch.bfloat16):
                    if hasattr(self.model, 'forward') and 'is_cox' in self.model.forward.__code__.co_varnames:
                        output = self.model(features, is_cox=True)
                    else:
                        output = self.model(features)
                    
                    if isinstance(output, dict):
                        logits = output['logits']
                    else:
                        logits = output
                    
                    logits = logits.float()
                    
                    # NLLSurv: map time to bins and compute loss
                    y = map_time_to_bins_with_edges(time, self.nll_bin_edges)
                    c = (1 - status).long()
                    loss = loss_fn(logits, y, c)
                
                if fp16_scaler is not None:
                    fp16_scaler.scale(loss).backward()
                    fp16_scaler.step(optimizer)
                    fp16_scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                lr_scheduler(global_step)
                global_step += 1
                epoch_loss += loss.item()
                epoch_steps += 1
                
                pbar.set_postfix({'loss': loss.item()})
            
            avg_loss = epoch_loss / max(epoch_steps, 1)
            epoch_time = time_module.time() - epoch_start_time
            self.logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
            
            if self.writer:
                self.writer.add_scalar('Train/Loss', avg_loss, epoch)
            
            # Validation
            self.model.eval()
            torch.set_grad_enabled(False)
            val_metrics = self.evaluate('val')
            # Select on val NLL loss, not the tiny-event val c-index (see
            # SurvivalTrainer._compute_val_loss for rationale).
            val_loss = self._compute_val_loss(loss_fn)
            torch.set_grad_enabled(True)
            self.model.train()

            # c-index kept for logging/reference only; selection is on val loss
            val_cidx = val_metrics.get('val_c_index', 0.0)
            val_cidx = float(val_cidx.item()) if isinstance(val_cidx, torch.Tensor) else float(val_cidx)
            self.logger.info(f"Val loss: {val_loss:.4f}, Val c-index: {val_cidx:.4f}")
            early_stopping(val_loss, val_cidx, self.model)
            if early_stopping.counter == 0:  # saved a new best this epoch
                best_epoch = epoch
            default_stop = early_stopping.early_stop
            if self.policy_runtime is not None:
                default_stop = self.policy_runtime.should_stop(
                    default_stop,
                    epoch=epoch,
                    metrics={'val_loss': val_loss, 'val_c_index': val_cidx},
                )
            if default_stop:
                self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Load best model (EarlyStopping saves as best_{model_type}.pth)
        best_model_path = os.path.join(self.save_dir, f'best_{self.model_type}.pth')
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            self.logger.info(f"Loaded best model from {best_model_path}")
        elif hasattr(early_stopping, 'best_model_state') and early_stopping.best_model_state is not None:
            # Fallback: load from early_stopping's saved state
            self.model.load_state_dict(early_stopping.best_model_state)
            self.logger.info("Loaded best model from early stopping state")
        print(f"[selected] epoch={best_epoch}", flush=True)

        self.model.eval()
        torch.set_grad_enabled(False)
        
        return self.model
    
    def _compute_val_loss(self, loss_fn):
        """Mean NLLSurv loss over the val set — the model-selection signal,
        robust to the tiny-event val c-index. Returns a float. Assumes the
        model is already in eval mode."""
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in self.val_loader:
                if len(batch) == 7:
                    features, coords, bag_sizes, status, time, patient_ids, slide_ids = batch
                else:
                    features, coords, bag_sizes, status, time, patient_ids = batch
                features = features.to(self.device)
                if features.dim() == 2:
                    features = features.unsqueeze(0)
                if hasattr(self.model, 'forward') and 'is_cox' in self.model.forward.__code__.co_varnames:
                    output = self.model(features, is_cox=True)
                else:
                    output = self.model(features)
                logits = (output['logits'] if isinstance(output, dict) else output).float()
                y = map_time_to_bins_with_edges(time.to(self.device), self.nll_bin_edges)
                c = (1 - status.to(self.device)).long()
                total += float(loss_fn(logits, y, c).item())
                n += 1
        return total / max(n, 1)

    def evaluate(self, split='test'):
        """Evaluate on a split"""
        if self.model is None:
            raise RuntimeError("Model not created. Call create_model() first.")
        
        loader = self.val_loader if split == 'val' else self.test_loader
        prefix = split.capitalize()
        
        self.model.eval()
        torch.set_grad_enabled(False)
        
        all_preds = []
        all_status = []
        all_time = []
        all_patient_ids = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"{prefix} Eval"):
                # Survival dataset now returns 7 items: features, coords, bag_sizes, status, time, patient_ids, slide_ids
                if len(batch) == 7:
                    features, coords, bag_sizes, status, time, patient_ids, slide_ids = batch
                else:
                    # Old format with 6 items
                    features, coords, bag_sizes, status, time, patient_ids = batch

                features = features.to(self.device)
                if features.dim() == 2:
                    features = features.unsqueeze(0)
                status = status.to(self.device)
                time = time.to(self.device)
                
                # Ensure status and time are at least 1D for batch_size=1 case (like train_surv_porpoise.py)
                if status.dim() == 0:
                    status = status.unsqueeze(0)
                if time.dim() == 0:
                    time = time.unsqueeze(0)
                
                if hasattr(self.model, 'forward') and 'is_cox' in self.model.forward.__code__.co_varnames:
                    output = self.model(features, is_cox=True)
                else:
                    output = self.model(features)
                
                if isinstance(output, dict):
                    logits = output['logits']
                else:
                    logits = output
                
                # Save logits first (like train_surv_porpoise.py: line 368)
                # Convert to risk later after concatenating all logits
                all_preds.append(logits.float().cpu().numpy())
                all_status.append(np.atleast_1d(status.cpu().numpy()).astype(np.float32))
                all_time.append(np.atleast_1d(time.cpu().numpy()).astype(np.float32))
                # Handle patient_ids: could be string or list
                if isinstance(patient_ids, (list, tuple)):
                    all_patient_ids.extend(patient_ids)
                else:
                    all_patient_ids.append(patient_ids)
        
        # Concatenate all predictions (like train_surv_porpoise.py: lines 375-377)
        all_preds = np.concatenate(all_preds) if len(all_preds) else np.zeros((0, 1), dtype=np.float32)
        all_status = np.concatenate(all_status) if len(all_status) else np.zeros((0,), dtype=np.float32)
        all_time = np.concatenate(all_time) if len(all_time) else np.zeros((0,), dtype=np.float32)
        
        # Convert logits to risk score for NLLSurv (like train_surv_porpoise.py: lines 379-385)
        if all_preds.shape[0] > 0:
            logits_t = torch.from_numpy(all_preds)
            # NLLSurv: convert discrete hazards to risk score
            hazards = torch.sigmoid(logits_t)
            survival = torch.cumprod(1 - hazards, dim=1)
            # Risk score = negative expected survival time (area under survival curve)
            risk_tensor = -survival.sum(dim=1, keepdim=True)  # [N, 1]
        else:
            risk_tensor = torch.zeros((0, 1), dtype=torch.float32)
        
        status_tensor = torch.from_numpy(all_status)
        time_tensor = torch.from_numpy(all_time)
        
        # Calculate C-index (like train_surv_porpoise.py: lines 386-391)
        if all_preds.shape[0] > 0:
            c_index = survival_c_index(risk_tensor, status_tensor, time_tensor, all_patient_ids)
        else:
            c_index = float("nan")
        
        # Calculate event statistics
        num_events = int(all_status.sum())
        num_censored = int((1 - all_status).sum())
        event_rate = float(all_status.mean())
        mean_time = float(all_time.mean())
        median_time = float(np.median(all_time))
        
        metrics = {
            f"{split}_c_index": c_index if c_index is not None else float("nan"),
            f"{split}_events": num_events,
            f"{split}_censored": num_censored,
            f"{split}_event_rate": event_rate,
            f"{split}_mean_time": mean_time,
            f"{split}_median_time": median_time,
        }
        
        # Log metrics (val-firewall: seal test metrics from run.log during search)
        if split != 'test' or os.environ.get("AUTOMIL_CERTIFY"):
            for key, value in metrics.items():
                self.logger.info(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
        
        # A4': persist val risk scores under the benchmark's shared name so the
        # fold carries a hashable no-op detector (mirrors the test CSV below).
        if split == 'val':
            save_csv_path = os.path.join(self.save_dir, "predictions_val.csv")
            risk_for_csv = risk_tensor.squeeze(1) if risk_tensor.dim() > 1 else risk_tensor
            risk_for_csv_np = risk_for_csv.detach().cpu().numpy() if isinstance(risk_for_csv, torch.Tensor) else risk_for_csv
            results_df = pd.DataFrame({
                'patient_id': all_patient_ids,
                'status': all_status.astype(int),
                'time': all_time,
                'risk_score': risk_for_csv_np.flatten()
            })
            results_df.to_csv(save_csv_path, index=False)
            self.logger.info(f"Val predictions saved to {save_csv_path}")

        # Save results to CSV if test split (like train_surv_porpoise.py: lines 591-592)
        if split == 'test':
            save_csv_path = os.path.join(self.save_dir, f"results_{self.model_type}.csv")
            # For CSV, use squeezed risk (like train_surv_porpoise.py: line 517)
            risk_for_csv = risk_tensor.squeeze(1) if risk_tensor.dim() > 1 else risk_tensor
            risk_for_csv_np = risk_for_csv.detach().cpu().numpy() if isinstance(risk_for_csv, torch.Tensor) else risk_for_csv
            results_df = pd.DataFrame({
                'sample_id': [f"sample_{i}" for i in range(all_preds.shape[0])],
                'patient_id': all_patient_ids,
                'status': all_status.astype(int),
                'time': all_time,
                'risk_score': risk_for_csv_np.flatten()
            })
            results_df.to_csv(save_csv_path, index=False)
            self.logger.info(f"Results saved to {save_csv_path}")
        
        return metrics
    
    def save_training_config(self):
        """Save training configuration to JSON file"""
        actual_config = {
            "batch_size": 1,  # Fixed for Porpoise
            "learning_rate": self.config.get('learning_rate', 1e-4),
            "batch_sampler": "none",  # No batch sampler for bs=1
            "model_type": self.model_type,
            "num_epochs": self.config.get('num_epochs', 100),
            "warmup_epochs": self.config.get('warmup_epochs', 5),
            "weight_decay": self.config.get('weight_decay', 0.01),
            "dropout": self.config.get('dropout', 0.25),
            "patience": self.config.get('patience', 10),
            "hidden_dim": self.config.get('hidden_dim', 256),
            "feature_dimension": self.config.get('feature_dimension'),
            "max_seq_length": self.config.get('max_seq_length'),
            "metric": self.dataset_info.get('metric', 'c_index'),
            "survival_loss": self.survival_loss,
            "nll_bins": self.nll_bins,
        }
        
        dataset_info_dict = {
            "task_type": self.dataset_info.get('task_type'),
            "dataset_name": os.path.basename(os.path.dirname(self.plan_path)),
            "plan_path": self.plan_path,
            "fold": self.fold,
            "evaluation_setting": self.evaluation_setting,
        }
        
        config_path = os.path.join(self.save_dir, 'training_config.json')
        config_dict = {
            "dataset_info": dataset_info_dict,
            "actual_configuration": actual_config,
            "random_seed": self.seed,
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        self.logger.info(f"Training configuration saved to {config_path}")
