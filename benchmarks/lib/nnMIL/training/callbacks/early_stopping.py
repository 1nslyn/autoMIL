"""
Early stopping callbacks for different task types.
"""
import os
import torch
import numpy as np


def _discard_stale_best_checkpoint(save_dir, model_type, logger=None):
    """Delete a ``best_<model>.pth`` left behind by a PRIOR attempt.

    The end-of-training restore checks file existence, not authorship; under
    the orchestrator a same-node relaunch reuses the results dir, so without
    this a run whose every epoch was degenerate ("no checkpoint saved") would
    restore — and certify — the previous attempt's weights as source=best.
    """
    if not save_dir or not model_type:
        return
    stale = os.path.join(save_dir, f"best_{model_type}.pth")
    if os.path.exists(stale):
        os.remove(stale)
        msg = f"EarlyStopping: removed stale checkpoint from a prior attempt: {stale}"
        if logger:
            logger.info(msg)
        else:
            print(msg)


class EarlyStopping:
    """Checkpoint tracker on validation AUC (protocol v4), with early stopping.

    The checkpoint saved is the epoch with the highest validation AUC: a
    strictly higher AUC saves, a tie keeps the earlier epoch, a non-finite AUC
    never saves and counts toward patience -- the same contract as autobench's
    SelectionTracker and CLAM's callback. The trainer still computes the
    validation loss for its epoch line; it does not vote.
    """
    def __init__(self, patience=7, verbose=False, save_dir=None, model_type=None, logger=None):
        """
        Args:
            patience: Early stopping patience (consecutive epochs without a higher val AUC)
            verbose: Print progress when no logger is given
            save_dir: Directory to save the best model
            model_type: Model type name for saving
            logger: Optional logger for logging messages
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.save_dir = save_dir
        self.model_type = model_type
        self.logger = logger
        # Epoch of the checkpoint currently saved (-1: none yet). Owned HERE,
        # where the checkpoint is saved, so callers read it instead of
        # inferring "saved this epoch" from counter == 0. Callers pass the
        # true epoch to __call__; the internal per-call counter stands in for
        # callers that do not (one __call__ per epoch).
        self.best_epoch = -1
        self._epochs_seen = 0
        _discard_stale_best_checkpoint(save_dir, model_type, logger)
        self._log("EarlyStopping: selecting the checkpoint on VAL_AUC (protocol v4)")

    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)
        elif self.verbose:
            print(msg)

    @property
    def early_stop(self) -> bool:
        """Patience exhausted right now. Derived from the counter, never
        latched: a stop a policy suppressed must not stick once the metric
        improves again (the counter resets, and so does this)."""
        return self.counter >= self.patience

    def __call__(self, val_auc, model, epoch=None):
        current_epoch = self._epochs_seen if epoch is None else epoch
        self._epochs_seen += 1

        score = val_auc
        # A non-finite AUC must never become (or defend) the checkpoint.
        if np.isnan(score) or np.isinf(score):
            score = float("-inf")

        if self.best_score is None and score == float("-inf"):
            # Non-finite val AUC with no checkpoint yet: nothing worth saving.
            # Count toward patience; an all-non-finite run ends with no
            # checkpoint at all rather than certifying epoch-0 garbage.
            self.counter += 1
            self._log(f'EarlyStopping: non-finite VAL_AUC at epoch {current_epoch}; '
                      f'no checkpoint saved ({self.counter}/{self.patience})')
            return
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = current_epoch
            self.save_checkpoint(model)
            # Degenerate epochs may have accumulated patience before the first
            # valid checkpoint; a real save starts the count fresh.
            self.counter = 0
            self._log(f'EarlyStopping: Initial VAL_AUC = {score:.4f}')
        elif score <= self.best_score:
            # Not strictly better (a tie keeps the earlier epoch).
            self.counter += 1
            self._log(f'EarlyStopping counter: {self.counter}/{self.patience} '
                      f'(VAL_AUC: {score:.4f} <= best {self.best_score:.4f})')
            if self.counter >= self.patience:
                self._log(f'Early stopping triggered! No improvement for {self.patience} epochs.')
        else:
            old_score = self.best_score
            self.best_score = score
            self.best_epoch = current_epoch
            self.save_checkpoint(model)
            self.counter = 0
            self._log(f'EarlyStopping: VAL_AUC improved from {old_score:.4f} to {score:.4f}. Reset counter.')

    def save_checkpoint(self, model):
        self._log('Checkpoint selection score improved. Saving model...')
        # A true copy: state_dict() tensors alias the live parameters.
        self.best_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if self.save_dir and self.model_type:
            best_model_path = os.path.join(self.save_dir, f"best_{self.model_type}.pth")
            torch.save(model.state_dict(), best_model_path)
            self._log(f'Saved best model to {best_model_path}')


class RegressionEarlyStopping:
    """Early stopping for regression tasks using metric from plan file"""
    def __init__(self, patience=10, verbose=False, delta=0, metric='pearson', save_dir=None, model_type=None, logger=None):
        """
        Args:
            patience: Early stopping patience
            verbose: Verbose output
            delta: Minimum change to qualify as improvement
            metric: Primary metric from plan file ('pearson', 'r2', 'mse', etc.)
            save_dir: Directory to save best model
            model_type: Model type name for saving
            logger: Optional logger for logging messages
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.save_dir = save_dir
        self.model_type = model_type
        self.logger = logger
        _discard_stale_best_checkpoint(save_dir, model_type, logger)

        # Use metric from plan file
        metric_lower = metric.lower()
        if 'pearson' in metric_lower or 'corr' in metric_lower:
            self.primary_metric = "PEARSON"
        elif 'r2' in metric_lower or 'r_squared' in metric_lower:
            self.primary_metric = "R2"
        elif 'mse' in metric_lower:
            self.primary_metric = "MSE"
            # MSE is lower-is-better, so we'll handle it differently
            self.higher_is_better = False
        else:
            # Default to Pearson
            self.primary_metric = "PEARSON"
            self.higher_is_better = True
        
        # Most regression metrics are higher-is-better, except MSE
        if not hasattr(self, 'higher_is_better'):
            self.higher_is_better = True
        
        msg = f"RegressionEarlyStopping: Using {self.primary_metric} as primary metric (from plan: {metric})"
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def __call__(self, val_mse, val_pearson, val_r2, model):
        # Use metric from plan file
        if self.primary_metric == "PEARSON":
            score = val_pearson
        elif self.primary_metric == "R2":
            score = val_r2
        elif self.primary_metric == "MSE":
            score = -val_mse  # Convert to higher-is-better for comparison
        else:
            score = val_pearson  # Default
        
        # Handle NaN/inf scores
        if np.isnan(score) or np.isinf(score):
            score = 0.0 if self.higher_is_better else -1e6
            
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_mse, val_pearson, val_r2, model)
            msg = f'RegressionEarlyStopping: Initial {self.primary_metric} = {score:.4f}'
            if self.logger:
                self.logger.info(msg)
            elif self.verbose:
                print(msg)
        elif score < self.best_score + self.delta:
            self.counter += 1
            msg = f'EarlyStopping counter: {self.counter}/{self.patience} ({self.primary_metric}: {score:.4f} < {self.best_score:.4f} + {self.delta:.4f})'
            if self.logger:
                self.logger.info(msg)
            elif self.verbose:
                print(msg)
            if self.counter >= self.patience:
                self.early_stop = True
                msg = f'Early stopping triggered! No improvement for {self.patience} epochs.'
                if self.logger:
                    self.logger.info(msg)
                elif self.verbose:
                    print(msg)
        else:
            improvement = score - self.best_score
            old_score = self.best_score
            self.best_score = score
            self.save_checkpoint(val_mse, val_pearson, val_r2, model)
            self.counter = 0
            msg = f'RegressionEarlyStopping: {self.primary_metric} improved from {old_score:.4f} to {self.best_score:.4f} (+{improvement:.4f}). Reset counter.'
            if self.logger:
                self.logger.info(msg)
            elif self.verbose:
                print(msg)

    def save_checkpoint(self, val_mse, val_pearson, val_r2, model):
        msg = 'Checkpoint selection score improved. Saving model...'
        if self.logger:
            self.logger.info(msg)
        elif self.verbose:
            print(msg)
        self.best_model_state = model.state_dict().copy()
        
        # Save best model to file
        if self.save_dir and self.model_type:
            best_model_path = os.path.join(self.save_dir, f"best_{self.model_type}.pth")
            torch.save(model.state_dict(), best_model_path)
            msg = f'Saved best model to {best_model_path}'
            if self.logger:
                self.logger.info(msg)
            elif self.verbose:
                print(msg)


class EarlyStoppingSurvival:
    """Checkpoint tracker on the validation C-index (protocol v4), with early stopping.

    Same contract as EarlyStopping above: a strictly higher C-index saves, a
    tie keeps the earlier epoch, a non-finite C-index never saves and counts
    toward patience. A finite C-index of exactly 0.0 is a (terrible) real score.
    The trainer still computes the validation loss for its epoch line.
    """
    def __init__(self, patience=10, verbose=False, save_dir=None, model_type=None, logger=None):
        """
        Args:
            patience: Early stopping patience (consecutive epochs without a higher val C-index)
            verbose: Print progress when no logger is given
            save_dir: Directory to save the best model
            model_type: Model type name for saving
            logger: Optional logger
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.save_dir = save_dir
        self.model_type = model_type
        self.logger = logger
        # Epoch of the checkpoint currently saved (-1: none yet); same
        # contract as EarlyStopping.best_epoch above -- owned where the
        # checkpoint is saved, never inferred from counter == 0.
        self.best_epoch = -1
        self._epochs_seen = 0
        self.best_model_state = None
        _discard_stale_best_checkpoint(save_dir, model_type, logger)
        self._log("EarlyStopping: selecting the checkpoint on the validation C-index (protocol v4)")

    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)
        elif self.verbose:
            print(msg)

    @property
    def early_stop(self) -> bool:
        """Patience exhausted right now. Derived from the counter, never
        latched: a stop a policy suppressed must not stick once the metric
        improves again (the counter resets, and so does this)."""
        return self.counter >= self.patience

    def __call__(self, val_c_index, model, epoch=None):
        current_epoch = self._epochs_seen if epoch is None else epoch
        self._epochs_seen += 1

        score = val_c_index
        # Degeneracy is a property of the RAW observation; a legitimate finite
        # C-index of exactly 0.0 is a (terrible) real score, not a NaN.
        if np.isnan(score) or np.isinf(score):
            score = float("-inf")

        if self.best_score is None and score == float("-inf"):
            # Non-finite first observation: nothing worth saving; count toward
            # patience so an all-degenerate run ends with no checkpoint.
            self.counter += 1
            self._log(f'EarlyStopping: degenerate C-index at epoch {current_epoch}; '
                      f'no checkpoint saved ({self.counter}/{self.patience})')
            return
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = current_epoch
            self.save_checkpoint(model)
            self.counter = 0  # degenerate epochs before the first save don't linger
            self._log(f'EarlyStopping: Initial C-index = {score:.4f}')
        elif score <= self.best_score:
            # Not strictly better (a tie keeps the earlier epoch).
            self.counter += 1
            self._log(f'EarlyStopping counter: {self.counter}/{self.patience} '
                      f'(C-index: {score:.4f} <= best {self.best_score:.4f})')
            if self.counter >= self.patience:
                self._log(f'Early stopping triggered! No improvement for {self.patience} epochs.')
        else:
            old_score = self.best_score
            self.best_score = score
            self.best_epoch = current_epoch
            self.save_checkpoint(model)
            self.counter = 0
            self._log(f'EarlyStopping: C-index improved from {old_score:.4f} to {score:.4f}. Reset counter.')

    def save_checkpoint(self, model):
        self._log(f'Validation C-index improved ({self.best_score:.4f}). Saving model...')
        # A true copy: state_dict() tensors alias the live parameters.
        self.best_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if self.save_dir and self.model_type:
            best_model_path = os.path.join(self.save_dir, f"best_{self.model_type}.pth")
            torch.save(model.state_dict(), best_model_path)
            self._log(f'Saved best model to {best_model_path}')
