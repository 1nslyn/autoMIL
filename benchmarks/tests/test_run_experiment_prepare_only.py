"""``run_experiment.py --prepare-only`` runs the data preparation of a normal run and stops.

No data, no GPU: the preparation functions and the runners are spies, so the
tests compare which preparation calls a normal run and a prepare-only run make,
and with which arguments, without preparing or training anything.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from autobench.config import TaskDef
from _helpers import make_test_ds

PREPARE_ALL = "autobench.pipeline.prepare.prepare_all"
PREPARE_TITAN = "autobench.pipeline.titan.prepare.prepare_titan_experiment"
PREPARE_NNMIL = "autobench.pipeline.nnmil.prepare.prepare_nnmil_experiment"
RUNNERS = [
    "autobench.pipeline.clam.runner.run_experiment",
    "autobench.pipeline.titan.runner.run_titan_experiment",
    "autobench.pipeline.abmil.runner.run_abmil_experiment",
    "autobench.pipeline.dtfd.run_dtfd_experiment",
    "autobench.pipeline.nnmil.runner.run_nnmil_experiment",
]


class ReachedTraining(Exception):
    """Raised by the runner spies: the run got past data preparation."""


@pytest.fixture(scope="module")
def run_experiment_mod() -> ModuleType:
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_experiment.py"
    name = "run_experiment_prepare_only"
    spec = importlib.util.spec_from_file_location(name, script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def prep_calls(run_experiment_mod, monkeypatch, tmp_path):
    """Spy on every preparation function; any runner reached raises ReachedTraining."""
    ds = make_test_ds(
        benchmark_dir=str(tmp_path / "default_benchmark"),
        tasks={
            "brca": TaskDef(name="brca", label_col="BRCA_predict_label",
                            label_map={0: "neg", 1: "pos"}, n_classes=2),
            "os": TaskDef(name="os", task_type="survival", event_col="OS_event",
                          time_col="OS_time", survival_losses=["nllsurv"]),
        },
    )
    monkeypatch.setattr(run_experiment_mod, "load_dataset_config", lambda name: ds)
    monkeypatch.setattr(run_experiment_mod, "register_sigterm_flush", lambda: None)
    monkeypatch.chdir(tmp_path)

    calls: list[tuple[str, dict]] = []

    def spy(name):
        def _spy(*args, **kwargs):
            assert not args, f"{name} must be called with keyword arguments"
            calls.append((name, kwargs))
        return _spy

    def reached_training(*args, **kwargs):
        raise ReachedTraining

    for name, target in (("prepare_all", PREPARE_ALL), ("prepare_titan", PREPARE_TITAN),
                         ("prepare_nnmil", PREPARE_NNMIL)):
        monkeypatch.setattr(target, spy(name))
    for target in RUNNERS:
        monkeypatch.setattr(target, reached_training)
    return calls


def _argv(tmp_path, framework, task="brca", *extra):
    encoder = "titan" if framework == "titan" else "uni_v2"
    return [
        "run_experiment.py", "--dataset", "x", "--task", task, "--encoder", encoder,
        "--model", framework, "--framework", framework,
        "--strategy", "devcv_r1", "--benchmark-dir", str(tmp_path / "bench"),
        "--n_folds", "3", "--seed", "7", *extra,
    ]


def _run(run_experiment_mod, monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", argv)
    run_experiment_mod.main()


CASES = [
    ("clam", "brca"),
    ("titan", "brca"),
    ("nnmil", "brca"),
    ("abmil", "brca"),
    ("dtfd", "brca"),
    ("nnmil", "os"),
]


@pytest.mark.parametrize(("framework", "task"), CASES)
def test_prepare_only_makes_the_preparation_calls_of_a_normal_run(
    run_experiment_mod, prep_calls, monkeypatch, tmp_path, framework, task,
):
    with pytest.raises(ReachedTraining):
        _run(run_experiment_mod, monkeypatch, _argv(tmp_path, framework, task))
    normal_run = list(prep_calls)
    prep_calls.clear()

    _run(run_experiment_mod, monkeypatch, _argv(tmp_path, framework, task, "--prepare-only"))

    assert prep_calls == normal_run
    expected = {"clam": ["prepare_all"], "titan": ["prepare_all", "prepare_titan"]}
    assert [name for name, _ in prep_calls] == expected.get(
        framework, ["prepare_all", "prepare_nnmil"])


def test_the_preparation_arguments_come_from_the_resolved_experiment(
    run_experiment_mod, prep_calls, monkeypatch, tmp_path,
):
    _run(run_experiment_mod, monkeypatch, _argv(tmp_path, "nnmil", "os", "--prepare-only"))
    (_, prepare_all_kwargs), (_, nnmil_kwargs) = prep_calls
    assert prepare_all_kwargs["strategies"] == ("devcv_r1",)
    assert prepare_all_kwargs["encoder_keys"] == ["uni_v2"]
    assert prepare_all_kwargs["benchmark_dir"] == str(tmp_path / "bench")
    assert (prepare_all_kwargs["seed"], prepare_all_kwargs["n_splits"]) == (7, 3)
    assert nnmil_kwargs["strategy"] == "devcv_r1"
    assert (nnmil_kwargs["seed"], nnmil_kwargs["n_splits"]) == (7, 3)
    assert nnmil_kwargs["survival_loss"] == "nllsurv"
    assert (nnmil_kwargs["event_col"], nnmil_kwargs["time_col"]) == ("OS_event", "OS_time")


def test_prepare_only_does_not_train_or_write_results(
    run_experiment_mod, prep_calls, monkeypatch, tmp_path, capsys,
):
    _run(run_experiment_mod, monkeypatch, _argv(tmp_path, "clam", "brca", "--prepare-only"))
    assert not (tmp_path / "result.json").exists()
    assert "--prepare-only" in capsys.readouterr().out


def test_prepare_only_refuses_skip_prep(
    run_experiment_mod, prep_calls, monkeypatch, tmp_path, capsys,
):
    argv = _argv(tmp_path, "clam", "brca", "--prepare-only", "--skip-prep")
    with pytest.raises(SystemExit) as exit_info:
        _run(run_experiment_mod, monkeypatch, argv)
    assert exit_info.value.code != 0
    assert "not allowed with" in capsys.readouterr().err
    assert prep_calls == []


def test_skip_prep_still_skips_every_preparation_call(
    run_experiment_mod, prep_calls, monkeypatch, tmp_path,
):
    for framework in ("clam", "titan", "nnmil"):
        with pytest.raises(ReachedTraining):
            _run(run_experiment_mod, monkeypatch, _argv(tmp_path, framework, "brca", "--skip-prep"))
    assert prep_calls == []


def test_two_runs_prepare_one_benchmark_root_one_at_a_time(
    run_experiment_mod, monkeypatch, tmp_path,
):
    """Split and plan files are not written atomically, so two runs preparing a
    strategy for the first time at once (parallel re-scores of two cells) would
    let one read a file the other is still writing."""
    import threading
    import time
    from types import SimpleNamespace

    active: list[int] = []
    overlaps: list[int] = []
    guard = threading.Lock()

    def slow_prepare_all(**kwargs):
        with guard:
            active.append(1)
            overlaps.append(len(active))
        time.sleep(0.2)
        with guard:
            active.pop()

    monkeypatch.setattr(PREPARE_ALL, slow_prepare_all)
    monkeypatch.setattr(PREPARE_TITAN, lambda **kwargs: None)
    exp_cfg = SimpleNamespace(
        framework=run_experiment_mod.Framework.TITAN, encoder_key="titan",
        train=SimpleNamespace(seed=42), n_folds=5, strategy="devcv_r1",
        task=SimpleNamespace(name="kras"),
    )
    ds = SimpleNamespace(mapping_csv="m.csv", features_base_dir="f")
    benchmark_dir = str(tmp_path / "benchmark")
    runs = [
        threading.Thread(
            target=run_experiment_mod._prepare_data, args=(exp_cfg, ds, benchmark_dir),
        )
        for _ in range(2)
    ]
    for run in runs:
        run.start()
    for run in runs:
        run.join()

    assert overlaps == [1, 1]
