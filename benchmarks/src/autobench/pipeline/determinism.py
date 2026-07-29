"""Single seeding routine shared by every trainer (L-2).

Before this module existed, each of the seven per-arm trainers (clam,
smmile, dtfd classification + survival, abmil, titan classification +
survival) defined its own ``seed_everything``/``_seed_everything``, and the
copies had silently drifted: only CLAM's and SMMILe's set the cuDNN
determinism flags --

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

Without them, cuDNN's autotuner is free to time several convolution kernels
and pick whichever is fastest *this run* -- a choice that depends on
current GPU load, not just the seed -- so DTFD's classification AND
survival trainers, ABMIL's, and TITAN's classification AND survival
trainers were not actually reproducible under a fixed seed. That undercuts
exactly the cross-arm comparison the shared training schedule (H-3/H-3b)
exists to make meaningful: "same seed" is supposed to mean the same thing
everywhere.

One function, imported everywhere it was previously reimplemented, so the
seven copies cannot drift again.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch

__all__ = ["seed_everything"]


def seed_everything(seed: int) -> None:
    """Seed python/numpy/torch RNGs and force deterministic cuDNN kernels.

    ``cudnn.benchmark = False`` stops cuDNN's autotuner from timing several
    convolution algorithms and picking whichever is fastest on this
    invocation (itself a run-to-run-variable choice); ``cudnn.deterministic
    = True`` additionally restricts it to bit-reproducible kernels. Both
    must be set together -- ``deterministic`` alone still leaves
    ``benchmark`` mode's kernel selection non-deterministic across runs.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
