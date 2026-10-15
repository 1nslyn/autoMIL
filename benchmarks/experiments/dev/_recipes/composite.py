"""Composite training recipe: one policy whose components are class constants.

Every constant is off at None, 0 or False; with all of them off every seam falls
through to the trainer's own behaviour, so the run is the native run exactly.
The module touches only the sealed ``PolicyVariant`` seams a campaign agent has,
so a recipe found with it is one an agent could have written. Classification
cells only: the survival trainers call none of the data seams and the policy
smoke refuses the file there.

Constant         Seam that carries it                                  Arms
EMA_DECAY        wrap_optimizer(_for): the optimizer wrapper's step()   all; DTFD keeps one
                 updates a zero-initialised shadow AFTER inner.step()   average per tier
                 (a step GradScaler skips never reaches it);
                 before_validation swaps in shadow / (1 - decay^t);
                 should_stop restores. Parameters only, no buffers.
LS_EPS           train_step -> backward_on(target=): (1 - eps) onehot    all (CLAM's instance
                 + eps / classes, through the arm's own cross-entropy    branch stays native)
POSITIVE_SHARE   train_step -> backward_on(weight=): per-bag weight      all; binary tasks only
                 0.5 / share for label 1, 0.5 / (1 - share) for label 0
CLASS_WEIGHTS    the same weight, one number per class; multiplies      all
                 with POSITIVE_SHARE when both are set
MAX_TILES,       transform_bag: a random subset of the bag, original     ABMIL, CLAM, DTFD; refused on
KEEP_FRAC,       order kept, drawn from the fold-private generator;      simple_mil (nnMIL re-pads every
MIN_TILES        size = the smaller of the two caps, never below         bag to its original length and
                 MIN_TILES (CLAM's k_sample) and never above the bag     pools the zero rows unmasked);
                                                                         nothing to cut on TITAN's one
                                                                         vector
WARMUP_STEPS,    the wrapper's step(): per-group lr x factor, then       all
WARMUP_START,    restored; the factor is linear warmup from WARMUP_START,
COSINE_EPOCHS,   then a per-step cosine to 0 over COSINE_EPOCHS x the
STEP_ANNEAL      steps epoch 0 took (1.0 until epoch 0 ends), times
                 STEP_ANNEAL ratio from its epoch on
MIN_EPOCHS       should_stop: bool(default and epoch >= MIN_EPOCHS)      all
UNFREEZE         wrap_optimizer(_for) moves the feature projection to    all; DTFD tier 1 only
(e1, e2, f)      its own param group; step() applies f up to epoch e1,
                 a linear ramp to 1 by e2. Projection = the 2-D weights
                 taking the most inputs, which must be the features'
                 width (raised otherwise), each with the 1-D parameter
                 listed right after it (its bias: DTFD has none, and
                 nnMIL lists its biases first, so there it is the
                 weights alone); simple_mil leaves out its
                 classification head. Regrouped only when switched on.
BODY_LR_SCALE    the same regrouping: the last 2-D and the last 1-D      simple_mil only (nnMIL's
                 parameter of each group are the head, every other       ema_attn10); refused on the
                 parameter trains at lr x BODY_LR_SCALE                  other arms, where it would slow
                                                                         the feature projection

The learning-rate factors live in step() because wrap_scheduler is live on DTFD
only and nnMIL's cosine_lr rewrites every group's lr after each step: a rate set
when the optimizer is wrapped is a silent no-op. They multiply whatever rate the
group holds, so the arms' own schedules stay in force underneath.

Known no-op traps: CLAM has no scheduler, so its scheduler seam is inert. nnMIL
warms up and anneals natively (--hparams warmup_epochs / num_epochs); WARMUP_STEPS
and COSINE_EPOCHS multiply with that. DTFD decays natively (--hparams lr_decay_step
/ lr_decay_ratio); STEP_ANNEAL multiplies with it. A slowed recipe needs MIN_EPOCHS:
early stopping armed during warmup restores an untrained checkpoint.
"""
from automil.registry import PolicyVariant, VariantSpec, register


class _Optimizer:
    """The arm's optimizer with the recipe's work around ``inner.step()``."""

    def __init__(self, recipe, inner, role):
        import torch

        self._recipe = recipe
        self._inner = inner
        self._role = role
        self._tags = {}
        self._width = None
        self._steps = 0
        self._epoch = 0
        self._steps_per_epoch = 0
        self._ema_steps = 0
        self._raw = None
        self._regroup()
        self._params = [p for group in inner.param_groups for p in group["params"]]
        self._shadow = [torch.zeros_like(p) for p in self._params] if recipe.EMA_DECAY else []

    def __getattr__(self, name):
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)

    def zero_grad(self, *args, **kwargs):
        return self._inner.zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs):
        factors = self._lr_factors()
        if factors is None:
            result = self._inner.step(*args, **kwargs)
        else:
            groups = self._inner.param_groups
            saved = [group["lr"] for group in groups]
            for group, factor in zip(groups, factors):
                group["lr"] = group["lr"] * factor
            try:
                result = self._inner.step(*args, **kwargs)
            finally:
                for group, lr in zip(groups, saved):
                    group["lr"] = lr
        self._steps += 1
        if self._shadow:
            self._update_average()
        return result

    # -- what train_step and should_stop tell the wrapper -----------------------------------
    def observe_batch(self, epoch, width):
        if self._width is not None and self._width != width:
            raise ValueError(
                "UNFREEZE took the layer with " + str(self._width) + " inputs for the feature "
                "projection, but the features are " + str(width) + " wide; the widest layer is "
                "not the projection in this model"
            )
        self._epoch = epoch

    def end_epoch(self, epoch):
        if epoch == 0 and not self._steps_per_epoch:
            self._steps_per_epoch = self._steps

    # -- the weight average -----------------------------------------------------------------
    def _update_average(self):
        import torch

        decay = self._recipe.EMA_DECAY
        with torch.no_grad():
            for shadow, param in zip(self._shadow, self._params):
                shadow.mul_(decay).add_(param.detach(), alpha=1.0 - decay)
        self._ema_steps += 1

    def swap_in(self):
        import torch

        if not self._shadow or not self._ema_steps:
            return
        if self._raw is not None:
            raise RuntimeError("the averaged weights are already in place")
        correction = 1.0 - self._recipe.EMA_DECAY ** self._ema_steps
        with torch.no_grad():
            self._raw = [param.detach().clone() for param in self._params]
            for param, shadow in zip(self._params, self._shadow):
                param.copy_(shadow / correction)

    def restore(self):
        import torch

        if self._raw is None:
            return
        with torch.no_grad():
            for param, raw in zip(self._params, self._raw):
                param.copy_(raw)
        self._raw = None

    # -- the learning-rate factors ----------------------------------------------------------
    def _lr_factors(self):
        """One multiplier per param group for this step; None when every one is 1."""
        recipe = self._recipe
        base = self._schedule_factor()
        ramp = self._unfreeze_factor() if recipe.UNFREEZE else 1.0
        body = recipe.BODY_LR_SCALE or 1.0
        if base == 1.0 and ramp == 1.0 and body == 1.0:
            return None
        factors = []
        for group in self._inner.param_groups:
            projection, in_body = self._tags.get(id(group), (False, False))
            factors.append(base * (ramp if projection else 1.0) * (body if in_body else 1.0))
        return factors

    def _schedule_factor(self):
        import math

        recipe = self._recipe
        step, epoch = self._steps, self._epoch
        warmup = recipe.WARMUP_STEPS or 0
        factor = 1.0
        if step < warmup:
            factor = recipe.WARMUP_START + (1.0 - recipe.WARMUP_START) * step / warmup
        elif recipe.COSINE_EPOCHS and self._steps_per_epoch:
            span = max(1, recipe.COSINE_EPOCHS * self._steps_per_epoch - warmup)
            progress = min(1.0, (step - warmup) / span)
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        if recipe.STEP_ANNEAL and epoch >= recipe.STEP_ANNEAL[0]:
            factor *= recipe.STEP_ANNEAL[1]
        return factor

    def _unfreeze_factor(self):
        first, second, start = self._recipe.UNFREEZE
        if self._epoch < first:
            return start
        if self._epoch >= second:
            return 1.0
        return start + (1.0 - start) * (self._epoch - first) / (second - first)

    # -- param groups, split only when UNFREEZE or BODY_LR_SCALE asks ----------------------
    def _regroup(self):
        """Split each group by (projection, body) so one factor reaches one part."""
        recipe, inner = self._recipe, self._inner
        # DTFD's second tier sits on the first tier's output: it has no feature projection.
        unfreeze = bool(recipe.UNFREEZE) and self._role != "tier2"
        scale_body = bool(recipe.BODY_LR_SCALE)
        if not (unfreeze or scale_body):
            return
        projection = set()
        if unfreeze:
            self._width, projection = self._projection()
        for group in list(inner.param_groups):
            heads = self._head_ids(group) if scale_body else set()
            parts = {}
            for param in group["params"]:
                key = (id(param) in projection, scale_body and id(param) not in heads)
                parts.setdefault(key, []).append(param)
            hyper = {name: value for name, value in group.items() if name != "params"}
            for index, (key, params) in enumerate(parts.items()):
                if index == 0:
                    group["params"] = params
                    part = group
                else:
                    inner.add_param_group({**hyper, "params": params})
                    part = inner.param_groups[-1]
                self._tags[id(part)] = key
        if scale_body and not any(in_body for _, in_body in self._tags.values()):
            raise ValueError(
                "BODY_LR_SCALE found no body: every parameter of this optimizer is the head"
            )

    def _projection(self):
        """The input width of the widest 2-D weights, and the ids of those weights and their biases."""
        flat = [p for group in self._inner.param_groups for p in group["params"]]
        weights = [p for p in flat if p.ndim == 2]
        if not weights:
            raise ValueError("UNFREEZE found no 2-D weight to take for the feature projection")
        width = max(p.shape[1] for p in weights)
        widest = [p for p in weights if p.shape[1] == width]
        if self._recipe.ARM == "simple_mil":
            # V and U (hidden x D) are the projection; cls (classes x D) is the head.
            tallest = max(p.shape[0] for p in widest)
            widest = [p for p in widest if p.shape[0] == tallest]
        ids = {id(p) for p in widest}
        # nn.Linear registers its weight, then its bias: the 1-D parameter right after a
        # projection weight is that weight's bias. DTFD's projection has none (what follows
        # it is a matrix, or nothing).
        biases = {id(after) for before, after in zip(flat, flat[1:]) if id(before) in ids and after.ndim == 1}
        return width, ids | biases

    @staticmethod
    def _head_ids(group):
        """The last 2-D and the last 1-D parameter of the group."""
        ids = set()
        for ndim in (1, 2):
            matching = [p for p in group["params"] if p.ndim == ndim]
            if matching:
                ids.add(id(matching[-1]))
        return ids


@register(VariantSpec(
    name="__NAME__", kind="policy", parent=None, base_commit="__BASE_COMMIT__",
    primary_value=0.0, node_id="__NODE_ID__", created_at="__CREATED_AT__",
))
class Recipe(PolicyVariant):
    ARM = "abmil"
    EMA_DECAY = None
    LS_EPS = None
    POSITIVE_SHARE = None
    CLASS_WEIGHTS = None
    MAX_TILES = None
    KEEP_FRAC = None
    MIN_TILES = 8
    WARMUP_STEPS = None
    WARMUP_START = 0.1
    COSINE_EPOCHS = None
    STEP_ANNEAL = None
    MIN_EPOCHS = None
    UNFREEZE = None
    BODY_LR_SCALE = None

    def __init__(self):
        self._validate()
        self._wrappers = {}

    # -- optimizer, validation and stopping seams --------------------------------------------
    def wrap_optimizer(self, opt):
        return self.wrap_optimizer_for(opt, role="main")

    def wrap_optimizer_for(self, opt, *, role):
        if role in self._wrappers:
            raise ValueError("optimizer role " + repr(role) + " is already wrapped")
        self._wrappers[role] = _Optimizer(self, opt, role)
        return self._wrappers[role]

    def before_validation(self, *, epoch):
        for wrapper in self._wrappers.values():
            wrapper.swap_in()
        return None

    def should_stop(self, *, default, epoch, metrics):
        for wrapper in self._wrappers.values():
            wrapper.restore()
            wrapper.end_epoch(epoch)
        return bool(default and epoch >= (self.MIN_EPOCHS or 0))

    # -- training bag and training step ------------------------------------------------------
    def transform_bag(self, features, *, label, epoch, generator):
        import torch

        rows = int(features.shape[0])
        keep = rows
        if self.MAX_TILES:
            keep = min(keep, self.MAX_TILES)
        if self.KEEP_FRAC:
            keep = min(keep, int(round(rows * self.KEEP_FRAC)))
        keep = min(rows, max(self.MIN_TILES, keep))
        if keep >= rows:
            return features
        chosen = torch.sort(torch.randperm(rows, generator=generator)[:keep]).values
        return features.index_select(0, chosen.to(features.device))

    def train_step(self, step):
        for wrapper in self._wrappers.values():
            wrapper.observe_batch(step.epoch, int(step.features.shape[-1]))
        target = self._soft_target(step)
        weight = self._bag_weight(step)
        step.zero_grad()
        step.backward_on(target=target, weight=weight)
        step.step()
        return None

    def _soft_target(self, step):
        if not self.LS_EPS:
            return None
        import torch

        hard = torch.nn.functional.one_hot(step.labels.long(), step.n_classes).float()
        return (1.0 - self.LS_EPS) * hard + self.LS_EPS / step.n_classes

    def _bag_weight(self, step):
        if not (self.POSITIVE_SHARE or self.CLASS_WEIGHTS):
            return None
        import torch

        per_class = [1.0] * step.n_classes
        if self.POSITIVE_SHARE:
            if step.n_classes != 2:
                raise ValueError(
                    "POSITIVE_SHARE weights two classes; this task has " + str(step.n_classes)
                )
            per_class = [0.5 / (1.0 - self.POSITIVE_SHARE), 0.5 / self.POSITIVE_SHARE]
        if self.CLASS_WEIGHTS:
            if len(self.CLASS_WEIGHTS) != step.n_classes:
                raise ValueError(
                    "CLASS_WEIGHTS holds " + str(len(self.CLASS_WEIGHTS)) + " weights; this task "
                    "has " + str(step.n_classes) + " classes"
                )
            per_class = [w * c for w, c in zip(per_class, self.CLASS_WEIGHTS)]
        table = torch.tensor(per_class, dtype=torch.float32, device=step.labels.device)
        return table[step.labels.long()]

    # -- the constants -----------------------------------------------------------------------
    def _validate(self):
        def real(value):
            return isinstance(value, (int, float)) and not isinstance(value, bool)

        def whole(value, low):
            return isinstance(value, int) and not isinstance(value, bool) and value >= low

        def switched_on(name):
            return getattr(self, name) not in (None, False, 0)

        if self.ARM not in ("abmil", "clam_mb", "dtfd_mil", "simple_mil"):
            raise ValueError(
                "Recipe.ARM must be abmil, clam_mb, dtfd_mil or simple_mil; got " + repr(self.ARM)
            )
        unit = ("strictly between 0 and 1", lambda v: real(v) and 0.0 < v < 1.0)
        tiles = ("a whole number of at least 1", lambda v: whole(v, 1))
        checks = (
            ("EMA_DECAY", True, *unit),
            ("LS_EPS", True, *unit),
            ("POSITIVE_SHARE", True, *unit),
            ("CLASS_WEIGHTS", True, "a tuple of two or more positive numbers",
             lambda v: isinstance(v, tuple) and len(v) > 1 and all(real(w) and w > 0 for w in v)),
            ("MAX_TILES", True, *tiles),
            ("KEEP_FRAC", True, "a number in (0, 1]", lambda v: real(v) and 0.0 < v <= 1.0),
            ("MIN_TILES", False, *tiles),
            ("WARMUP_STEPS", True, *tiles),
            ("WARMUP_START", False, "a number in [0, 1]", lambda v: real(v) and 0.0 <= v <= 1.0),
            ("COSINE_EPOCHS", True, *tiles),
            ("STEP_ANNEAL", True, "(epoch, ratio): a whole epoch and a positive ratio",
             lambda v: isinstance(v, tuple) and len(v) == 2 and whole(v[0], 0)
             and real(v[1]) and v[1] > 0),
            ("MIN_EPOCHS", True, "a whole number of at least 0", lambda v: whole(v, 0)),
            ("UNFREEZE", True, "(first, second, start): whole epochs first <= second and a "
             "start factor in [0, 1]",
             lambda v: isinstance(v, tuple) and len(v) == 3 and whole(v[0], 0)
             and whole(v[1], v[0]) and real(v[2]) and 0.0 <= v[2] <= 1.0),
            ("BODY_LR_SCALE", True, "a positive number", lambda v: real(v) and v > 0),
        )
        for name, may_be_off, rule, holds in checks:
            value = getattr(self, name)
            if may_be_off and not switched_on(name):
                continue
            if not holds(value):
                raise ValueError(
                    "Recipe." + name + " must be " + ("off (None) or " if may_be_off else "")
                    + rule + "; got " + repr(value)
                )
        if self.ARM == "simple_mil":
            for name in ("MAX_TILES", "KEEP_FRAC"):
                if switched_on(name):
                    raise ValueError(
                        "Recipe." + name + " is refused on simple_mil: nnMIL pads every transformed "
                        "bag back to its original length and pools the zero rows without a mask, so "
                        "a tile cap there is a subset plus heavy zero padding"
                    )
        elif switched_on("BODY_LR_SCALE"):
            raise ValueError(
                "Recipe.BODY_LR_SCALE is for simple_mil only: on the other arms it would slow "
                "the feature projection, which is known to hurt"
            )
