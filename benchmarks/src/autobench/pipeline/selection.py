"""Checkpoint selection on the primary validation metric (protocol v4).

Every classification arm feeds this tracker its per-epoch validation AUC and
every survival arm its in-fold validation C-index; the epoch it names is the
one whose weights are restored and reported. The rule, shared with the two
vendored callbacks (CLAM's ``EarlyStopping``, nnMIL's ``EarlyStopping`` and
``EarlyStoppingSurvival``) and pinned by ``TestSelectorContract``:

- higher is better and only a strictly greater value improves, so a tie keeps
  the earlier epoch;
- a non-finite value never becomes (or defends) the selection and counts
  toward patience like any non-improving epoch, so an all-non-finite run ends
  with nothing selected (``best_epoch == -1``) rather than epoch-0 garbage;
- patience counts consecutive non-improving observations since the last
  selection; the caller decides whether ``early_stop`` may end training.

The tracker owns no weights: the caller snapshots on ``True`` with whatever
copy its model needs. Measurement code; belongs on ``registry.protected``.
"""
from __future__ import annotations

import math


class SelectionTracker:
    def __init__(self, patience: int) -> None:
        if int(patience) < 1:
            raise ValueError(f"patience must be a positive integer, got {patience!r}")
        self.patience = int(patience)
        self.best_value: float | None = None
        self.best_epoch = -1
        self.counter = 0

    def observe(self, epoch: int, value: float) -> bool:
        """Record one epoch's primary validation metric.

        Returns True iff this epoch is now the selection, in which case the
        caller snapshots the weights it will restore at the end of the fold.
        """
        v = float(value)
        if math.isfinite(v) and (self.best_value is None or v > self.best_value):
            self.best_value = v
            self.best_epoch = int(epoch)
            self.counter = 0
            return True
        self.counter += 1
        return False

    @property
    def early_stop(self) -> bool:
        return self.counter >= self.patience
