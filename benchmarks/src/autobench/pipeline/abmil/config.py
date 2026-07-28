"""ABMIL hyperparameter configuration.

Defaults reproduce Ilse et al., 2018 (``lib/AttentionDeepMIL/main.py``) in
full: both the paper-exact architecture AND the paper's own optimizer/training
schedule. Before 2026-07-28 this dataclass instead copied the shared benchmark
schedule (``TrainConfig`` defaults in ``autobench.pipeline.config``, which is
CLAM's schedule) — a hand-me-down with no rationale of its own. See
``provenance.py`` for the audit that found it and the resolution.

Disclosure, not silent deviation: upstream's ``--epochs default=20``
(``main.py:16``) comes from a toy MNIST-bags experiment, not a WSI benchmark,
so 20 epochs is arguably not transferable to slide-level training. The
decision is to reproduce it faithfully anyway and say so here, rather than
quietly pick a "better" number. One consequence worth naming explicitly:
``patience`` (20) now equals ``max_epochs`` (20), which effectively disables
early stopping — every fold runs the full 20 epochs. That is a byproduct of
matching upstream's epoch count, not a new patience value chosen for this
benchmark; ``patience=20`` is unchanged from before.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ABMILConfig:
    """Immutable ABMIL hyperparameters."""

    # Architecture (paper-exact: Ilse et al., 2018) — unaffected by the
    # 2026-07-28 optimizer change below.
    M: int = 500       # instance embedding dim
    L: int = 128       # attention hidden dim
    dropout: float = 0.0

    # Optimization (upstream lib/AttentionDeepMIL/main.py:18-21; was
    # lr=2e-4/weight_decay=1e-5, copied from the shared CLAM schedule)
    lr: float = 5e-4          # main.py:18 `--lr default=0.0005`
    weight_decay: float = 1e-4  # main.py:20 `--reg default=10e-5` (== 1e-4)

    # Training schedule (upstream main.py:16 `--epochs default=20`; was 200).
    # See module docstring: 20 epochs is a toy-MNIST value, disclosed not
    # silently changed, and it collapses patience below to a no-op.
    max_epochs: int = 20
    early_stopping: bool = True
    patience: int = 20  # unchanged value; now == max_epochs, so early
                         # stopping never triggers (every fold runs to term)
