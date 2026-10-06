"""Protocol v5: the training-bag transform and the pre-validation hook reach the arms.

``test_policy_seams.py`` pins the runtime. These tests drive the real fold
trainers of ABMIL, DTFD, TITAN and CLAM (nnMIL, whose batches are padded, is in
``test_policy_seam_nnmil.py``) on tiny CPU fixtures, and hold each arm to the
same three facts:

* a policy that overrides both seams but changes nothing leaves the fold exactly
  as a native run left it (validation metrics, prediction hash, the per-epoch
  trajectory printed at full precision), sees every training bag once per epoch
  and no validation or test bag, and has its hook called right before each
  epoch's stopping decision;
* a policy that does change the bag changes the run, and one whose hook shifts
  the weights changes the first epoch's validation, so each seam is wired and
  the hook sits before the validation it is meant to act on;
* the arm's own constraint on a transformed bag is enforced (TITAN keeps its
  vector's shape, CLAM keeps enough instances for its instance-level loss).
"""
from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from _seam_probe import (  # noqa: E402
    expected_calls,
    first_half,
    fold_runtime,
    recording_policy,
    seam_calls,
    weight_shifting_policy,
)
from autobench.pipeline.abmil.train import train_abmil_fold  # noqa: E402
from autobench.pipeline.clam.train import train_fold as train_clam_fold  # noqa: E402
from autobench.pipeline.config import (  # noqa: E402
    ExperimentConfig,
    Framework,
    ModelConfig,
    TaskConfig,
    TrainConfig,
)
from autobench.pipeline.dtfd.train import train_dtfd_fold  # noqa: E402
from autobench.pipeline.titan.dataset import TitanSlideDataset  # noqa: E402
from autobench.pipeline.titan.train import train_titan_fold  # noqa: E402
from tests import test_abmil_arm, test_dtfd_arm  # noqa: E402
from tests.test_checkpoint_selection import _clam_fixture  # noqa: E402

CPU = torch.device("cpu")
TITAN_SLIDES, TITAN_WIDTH, TITAN_EPOCHS = 24, 16, 5
CLAM_EPOCHS = 2


def _comparable(result: dict) -> dict:
    """Everything a fold reports except the wall clock."""
    return {key: value for key, value in result.items() if key != "elapsed_seconds"}


def _splits(module, seed: int = 0):
    rng = np.random.default_rng(seed)
    return (
        module._make_split(rng, "t", 12),
        module._make_split(rng, "v", 4),
        module._make_split(rng, "e", 4),
    )


def _abmil(workdir: Path, runtime) -> dict:
    train, val, test = _splits(test_abmil_arm)
    workdir.mkdir()
    return _comparable(train_abmil_fold(
        "abmil", train, val, test, embed_dim=test_abmil_arm.IN_DIM, num_classes=2,
        cfg=test_abmil_arm._smoke_cfg(), device=CPU, seed=42,
        policy_runtime=runtime, fold_dir=str(workdir),
    ))


def _dtfd(workdir: Path, runtime) -> dict:
    train, val, test = _splits(test_dtfd_arm)
    workdir.mkdir()
    return _comparable(train_dtfd_fold(
        train, val, test, embed_dim=test_dtfd_arm.EMB, num_classes=2,
        cfg=test_dtfd_arm._smoke_cfg(), device=CPU, seed=42, return_history=True,
        policy_runtime=runtime, fold_dir=str(workdir),
    ))


def _titan(workdir: Path, runtime) -> dict:
    features_dir = workdir / "features_titan"
    features_dir.mkdir(parents=True)
    rng = np.random.default_rng(3)
    slide_ids = [f"s{i}" for i in range(TITAN_SLIDES)]
    labels = [i % 2 for i in range(TITAN_SLIDES)]
    for slide_id, label in zip(slide_ids, labels):
        with h5py.File(features_dir / f"{slide_id}.h5", "w") as handle:
            vector = (rng.standard_normal(TITAN_WIDTH) + 3.0 * label).astype("float32")
            handle.create_dataset("features", data=vector)
    dataset = TitanSlideDataset(slide_ids, labels, str(features_dir))
    exp_cfg = ExperimentConfig(
        task=TaskConfig(name="brca", label_col="label", label_dict={"neg": 0, "pos": 1}),
        encoder_key="titan", embed_dim=TITAN_WIDTH, model=ModelConfig(model_type="titan"),
        train=TrainConfig(seed=1, max_epochs=TITAN_EPOCHS, early_stopping=False),
        n_folds=1, framework=Framework.TITAN, strategy="standard",
    )
    return _comparable(train_titan_fold(
        exp_cfg, dataset, dataset, dataset, fold=0, results_dir=str(workdir / "results"),
        device="cpu", policy_runtime=runtime,
    ))


def _clam(workdir: Path, runtime, *, no_inst_cluster: bool = False) -> dict:
    exp_cfg, (train, val, test) = _clam_fixture(workdir, max_epochs=CLAM_EPOCHS)
    model = dataclasses.replace(exp_cfg.model, no_inst_cluster=no_inst_cluster)
    exp_cfg = dataclasses.replace(exp_cfg, model=model)
    return _comparable(train_clam_fold(
        exp_cfg, train, val, test, fold=0, results_dir=str(workdir / "results"),
        device=CPU, policy_runtime=runtime,
    ))


def _clam_without_instance_clustering(workdir: Path, runtime) -> dict:
    return _clam(workdir, runtime, no_inst_cluster=True)


@dataclass(frozen=True)
class Arm:
    name: str
    run: Callable[[Path, object], dict]
    epochs: int
    train_bags: int
    bag_shapes: frozenset
    #: A transform that changes this arm's bag and is legal under its constraints.
    change: Callable = first_half


ARMS = (
    Arm(
        "abmil", _abmil, test_abmil_arm._smoke_cfg().max_epochs, 12,
        frozenset({(test_abmil_arm.N_INSTANCES, test_abmil_arm.IN_DIM)}),
    ),
    Arm(
        "dtfd", _dtfd, test_dtfd_arm._smoke_cfg().max_epochs, 12,
        frozenset({(test_dtfd_arm.N_PATCHES, test_dtfd_arm.EMB)}),
    ),
    # A slide is one vector, so the change cannot drop instances: it flips them.
    Arm("titan", _titan, TITAN_EPOCHS, TITAN_SLIDES, frozenset({(1, TITAN_WIDTH)}), torch.neg),
    # The two CLAM training loops: with and without the instance-level loss.
    Arm("clam", _clam, CLAM_EPOCHS, 8, frozenset({(24, 64)})),
    Arm(
        "clam-without-instance-clustering", _clam_without_instance_clustering,
        CLAM_EPOCHS, 8, frozenset({(24, 64)}),
    ),
)


def _epoch_lines(capsys) -> list[str]:
    """The ``[epoch k] val_auc=... val_loss=...`` trajectory the dispatch printed."""
    return [
        line for line in capsys.readouterr().out.splitlines() if line.startswith("[epoch ")
    ]


@pytest.mark.parametrize("arm", ARMS, ids=lambda arm: arm.name)
def test_a_policy_that_changes_nothing_changes_nothing(arm, tmp_path, capsys):
    native = arm.run(tmp_path / "native", fold_runtime())
    native_trajectory = _epoch_lines(capsys)
    assert len(native_trajectory) == arm.epochs

    # The fixture is deterministic, so any difference below is the policy's doing.
    assert arm.run(tmp_path / "again", fold_runtime()) == native
    assert _epoch_lines(capsys) == native_trajectory

    log: list = []
    policy = recording_policy(log, torch.clone)
    assert arm.run(tmp_path / "inert", fold_runtime(policy)) == native
    assert _epoch_lines(capsys) == native_trajectory

    assert seam_calls(log) == expected_calls(arm.epochs, arm.train_bags)
    bags = [entry for entry in log if entry[0] == "bag"]
    assert {shape for _, _, _, shape in bags} == arm.bag_shapes
    assert {label for _, _, label, _ in bags} == {0, 1}
    assert all(type(label) is int for _, _, label, _ in bags)


@pytest.mark.parametrize("arm", ARMS, ids=lambda arm: arm.name)
def test_a_policy_that_changes_the_bag_changes_the_run(arm, tmp_path):
    native = arm.run(tmp_path / "native", fold_runtime())
    log: list = []
    live = arm.run(tmp_path / "live", fold_runtime(recording_policy(log, arm.change)))
    assert seam_calls(log) == expected_calls(arm.epochs, arm.train_bags)
    assert live["val_predictions_sha256"] != native["val_predictions_sha256"]


@pytest.mark.parametrize("arm", ARMS, ids=lambda arm: arm.name)
def test_the_hook_acts_on_the_weights_the_validation_then_scores(arm, tmp_path, capsys):
    arm.run(tmp_path / "native", fold_runtime())
    native = _epoch_lines(capsys)
    arm.run(tmp_path / "shifted", fold_runtime(weight_shifting_policy()))
    shifted = _epoch_lines(capsys)
    # Epoch 0 trains the same either way, so only a hook that runs before that
    # epoch's validation can make its score differ.
    assert shifted[0] != native[0]


def test_titan_keeps_the_shape_of_a_slide_vector(tmp_path):
    policy = recording_policy([], lambda row: torch.cat([row, row]))
    with pytest.raises(TypeError, match="keep this arm's bag shape"):
        _titan(tmp_path, fold_runtime(policy))


def test_clam_needs_enough_instances_left_for_its_instance_loss(tmp_path):
    policy = recording_policy([], lambda bag: bag[:3])
    with pytest.raises(ValueError, match="at least 8"):
        _clam(tmp_path, fold_runtime(policy))


def test_clam_without_instance_clustering_needs_no_more_than_one_instance(tmp_path):
    log: list = []
    policy = recording_policy(log, lambda bag: bag[:3])
    _clam_without_instance_clustering(tmp_path, fold_runtime(policy))
    assert seam_calls(log) == expected_calls(CLAM_EPOCHS, 8)
