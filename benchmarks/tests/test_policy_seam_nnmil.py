"""Protocol v5: the training-bag transform and the pre-validation hook reach nnMIL.

nnMIL trains on zero-padded ``[B, Nmax, D]`` batches with a ``bag_sizes`` vector,
so the transform is applied bag by bag on the unpadded slices and the padded
batch is rebuilt only when some bag changed. The trainer is driven for real on
tiny CPU batches (its own ``evaluate`` included); the policy reaches it as the
``policy_runtime`` attribute ``train_nnmil_fold`` sets after construction.
"""
from __future__ import annotations

import logging

import pytest

torch = pytest.importorskip("torch")

# Installs benchmarks/lib/nnMIL on sys.path (same mechanism the trainer uses).
import autobench.pipeline.nnmil._imports  # noqa: F401, E402
from _seam_probe import (  # noqa: E402
    expected_calls,
    first_half,
    fold_runtime,
    recording_policy,
    seam_calls,
    weight_shifting_policy,
)
from training.trainers.classification_trainer import (  # noqa: E402
    ClassificationTrainer,
    create_mask_from_bag_sizes,
    transform_training_bags,
)

WIDTH = 8
NMAX = 6
EPOCHS = 4
TRAIN_SIZES = [[6, 4, 5, 6], [3, 6, 6, 4]]
TRAIN_LABELS = [[0, 1, 0, 1], [1, 0, 1, 0]]
VAL_BAGS = [(6, 0), (4, 1), (5, 0), (6, 1)]


@pytest.fixture(autouse=True)
def _restore_torch_grad_state():
    """The trainer toggles torch's global grad switch; leave it as found."""
    was_enabled = torch.is_grad_enabled()
    yield
    torch.set_grad_enabled(was_enabled)


def _batch(sizes, labels, seed):
    """A padded batch: class-separable bags, zeros past each bag's size."""
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(len(sizes), NMAX, WIDTH, generator=generator)
    for row, (size, label) in enumerate(zip(sizes, labels)):
        features[row, :size] += 2.0 * label
        features[row, size:] = 0.0
    coords = torch.zeros(len(sizes), NMAX, 2)
    return features, coords, torch.tensor(sizes), torch.tensor(labels)


class _MeanBag(torch.nn.Module):
    """``simple_mil`` stand-in: mean over the padded bag, then a linear head.
    Keeps every training batch (and attention mask) it is handed."""

    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(WIDTH, 2)
        self.batches: list = []
        self.masks: list = []

    def forward(self, features, **kwargs):
        if self.training:
            self.batches.append(features.detach().clone())
            if "mask" in kwargs:
                self.masks.append(kwargs["mask"].clone())
        return self.head(features.float().mean(dim=1))


def _train(tmp_path, runtime, *, model_type="simple_mil"):
    """One real nnMIL classification fold; returns the trainer and a comparable record."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    trainer = ClassificationTrainer.__new__(ClassificationTrainer)
    trainer.model = _MeanBag()
    trainer.train_loader = [
        _batch(sizes, labels, seed)
        for seed, (sizes, labels) in enumerate(zip(TRAIN_SIZES, TRAIN_LABELS))
    ]
    trainer.val_loader = [_batch([s], [y], 10 + i) for i, (s, y) in enumerate(VAL_BAGS)]
    trainer.test_loader = trainer.val_loader
    trainer.config = {
        "num_epochs": EPOCHS, "warmup_epochs": 0, "patience": 10, "learning_rate": 1e-2,
    }
    trainer.device = torch.device("cpu")
    trainer.save_dir = str(tmp_path)
    trainer.model_type = model_type
    trainer.logger = logging.getLogger("nnmil-seam-test")
    trainer.writer = None
    trainer.dataset_info = {}
    trainer.num_classes = 2
    trainer.policy_runtime = runtime
    trainer.save_training_config = lambda: None
    trainer.train()
    record = {
        "val": {key: float(value) for key, value in trainer.evaluate("val").items()},
        "weights": [p.tolist() for p in trainer.model.state_dict().values()],
        "batches": [batch.tolist() for batch in trainer.model.batches],
        "masks": [mask.tolist() for mask in trainer.model.masks],
    }
    return trainer, record


def _epoch_lines(capsys) -> list[str]:
    return [
        line for line in capsys.readouterr().out.splitlines() if line.startswith("[epoch ")
    ]


def test_a_policy_that_changes_nothing_changes_nothing(tmp_path, capsys):
    _, native = _train(tmp_path / "native", fold_runtime())
    native_trajectory = _epoch_lines(capsys)
    assert len(native_trajectory) == EPOCHS
    assert len(native["batches"]) == EPOCHS * len(TRAIN_SIZES)

    log: list = []
    _, inert = _train(tmp_path / "inert", fold_runtime(recording_policy(log, torch.clone)))
    assert inert == native
    assert _epoch_lines(capsys) == native_trajectory
    assert seam_calls(log) == expected_calls(EPOCHS, sum(map(len, TRAIN_SIZES)))
    bags = [entry for entry in log if entry[0] == "bag"]
    assert {shape for _, _, _, shape in bags} == {(size, WIDTH) for size in (3, 4, 5, 6)}
    assert {label for _, _, label, _ in bags} == {0, 1}


def test_the_hook_acts_on_the_weights_the_validation_then_scores(tmp_path, capsys):
    _train(tmp_path / "native", fold_runtime())
    native = _epoch_lines(capsys)
    _train(tmp_path / "shifted", fold_runtime(weight_shifting_policy()))
    shifted = _epoch_lines(capsys)
    # Epoch 0 trains the same either way, so only a hook that runs before that
    # epoch's validation can make its score differ.
    assert shifted[0] != native[0]


def test_a_trainer_without_a_runtime_trains_like_one_with_a_native_runtime(tmp_path):
    _, bare = _train(tmp_path / "bare", None)
    _, native = _train(tmp_path / "native", fold_runtime())
    assert bare == native


def test_a_policy_that_changes_the_bag_changes_the_run(tmp_path):
    _, native = _train(tmp_path / "native", fold_runtime())
    log: list = []
    _, live = _train(tmp_path / "live", fold_runtime(recording_policy(log, first_half)))
    assert seam_calls(log) == expected_calls(EPOCHS, sum(map(len, TRAIN_SIZES)))
    assert live["weights"] != native["weights"]

    # The model trained on the rebuilt batches: the first half of each bag, the rest zeros.
    for original, rebuilt in zip(native["batches"], live["batches"]):
        for bag, new in zip(original, rebuilt):
            kept = sum(any(row) for row in bag) // 2
            assert new[:kept] == bag[:kept]
            assert not any(any(row) for row in new[kept:])


@pytest.mark.parametrize("transform", [first_half, lambda bag: torch.cat([bag, bag])])
def test_vision_transformer_refuses_a_transform_that_changes_a_bag_size(tmp_path, transform):
    """It attends over per-instance coordinates the policy never sees."""
    policy = recording_policy([], transform)
    with pytest.raises(TypeError, match="keep this arm's bag shape"):
        _train(tmp_path, fold_runtime(policy), model_type="vision_transformer")


def test_vision_transformer_still_takes_a_transform_that_keeps_the_bag_size(tmp_path):
    _, native = _train(tmp_path / "native", fold_runtime(), model_type="vision_transformer")
    noisy = recording_policy([], lambda bag: bag + 1.0)
    _, shifted = _train(tmp_path / "noisy", fold_runtime(noisy), model_type="vision_transformer")
    assert shifted["weights"] != native["weights"]
    # The attention mask comes from the bag sizes, which did not move.
    assert shifted["masks"] == native["masks"] != []


def _padded_batch():
    features = torch.arange(1.0, 31.0).reshape(3, 5, 2)
    sizes = torch.tensor([5, 3, 4])
    for row, size in enumerate(sizes.tolist()):
        features[row, size:] = 0.0
    return features, sizes, torch.tensor([0, 1, 1])


def test_the_native_runtime_hands_the_batch_back_untouched():
    features, sizes, labels = _padded_batch()
    out_features, out_sizes = transform_training_bags(
        fold_runtime(), features, sizes, labels, epoch=0,
    )
    assert out_features is features and out_sizes is sizes


def test_the_policy_sees_each_unpadded_bag_with_its_label():
    features, sizes, labels = _padded_batch()
    log: list = []
    transform_training_bags(
        fold_runtime(recording_policy(log, torch.clone)), features, sizes, labels, epoch=3,
    )
    assert log == [("bag", 3, 0, (5, 2)), ("bag", 3, 1, (3, 2)), ("bag", 3, 1, (4, 2))]


def test_a_changed_bag_rebuilds_the_padded_batch_and_its_sizes():
    features, sizes, labels = _padded_batch()
    before = features.clone()
    out_features, out_sizes = transform_training_bags(
        fold_runtime(recording_policy([], first_half)), features, sizes, labels, epoch=0,
    )
    assert out_sizes.tolist() == [2, 1, 2] and out_sizes.dtype == sizes.dtype
    assert out_features.shape == features.shape
    for row, size in enumerate(out_sizes.tolist()):
        assert torch.equal(out_features[row, :size], features[row, :size])
        assert not out_features[row, size:].any()
    assert torch.equal(features, before), "the input batch was modified"
    # What ViT would mask follows the rebuilt sizes.
    assert create_mask_from_bag_sizes(out_features, out_sizes).sum(dim=1).tolist() == [3, 4, 3]


def test_a_bag_may_grow_up_to_the_batch_width():
    features, sizes, labels = _padded_batch()
    grow = recording_policy([], lambda bag: torch.cat([bag, bag])[:5])
    out_features, out_sizes = transform_training_bags(
        fold_runtime(grow), features, sizes, labels, epoch=0,
    )
    assert out_sizes.tolist() == [5, 5, 5] and out_features.shape == features.shape


def test_a_bag_longer_than_the_batch_width_is_refused():
    features, sizes, labels = _padded_batch()
    too_long = recording_policy([], lambda bag: torch.cat([bag, bag]))
    with pytest.raises(ValueError, match="at most 5"):
        transform_training_bags(fold_runtime(too_long), features, sizes, labels, epoch=0)
