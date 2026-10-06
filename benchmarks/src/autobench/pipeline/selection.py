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
copy its model needs.

Protocol v5 scores a fold on the curve around that epoch rather than on the
epoch alone (``smoothed_selection_value``): the checkpoint is still the argmax,
but on a ~47-slide validation split the best of many epochs is partly a lucky
draw, and a recipe that evaluates more epochs collects more such draws.
Measurement code; belongs on ``registry.protected``.
"""
from __future__ import annotations

import math
from typing import Mapping

SMOOTHING_WINDOW = 5


def smoothed_selection_value(
    history: Mapping[int, float],
    selected_epoch: int,
    window: int = SMOOTHING_WINDOW,
) -> float | None:
    """Mean primary metric over ``window`` evaluated epochs around the selection.

    Non-finite epochs are dropped first. The window is the ``window``
    consecutive remaining epochs centred on the selected one, shifted inward at
    either end, or every remaining epoch when the run evaluated fewer. Epochs
    are matched by number, never by position, because some arms skip
    evaluations. Returns None when the selected epoch has no finite value.
    """
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError(f"window must be a positive integer, got {window!r}")
    finite = sorted(
        (int(epoch), float(value))
        for epoch, value in history.items()
        if math.isfinite(float(value))
    )
    epochs = [epoch for epoch, _ in finite]
    if int(selected_epoch) not in epochs:
        return None
    centre = epochs.index(int(selected_epoch))
    start = max(0, min(centre - window // 2, len(finite) - window))
    span = finite[start:start + window]
    return math.fsum(value for _, value in span) / len(span)


class SelectionTracker:
    def __init__(self, patience: int) -> None:
        # Zero is legal (a degenerate one-epoch run, as the vendored stoppers
        # already allow): a `--hparams` override is applied after the attempt
        # is charged, so refusing it here would only turn a bad run into a
        # crash.
        if isinstance(patience, bool) or int(patience) != patience or int(patience) < 0:
            raise ValueError(f"patience must be a non-negative integer, got {patience!r}")
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
