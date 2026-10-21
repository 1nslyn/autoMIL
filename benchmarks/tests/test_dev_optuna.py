"""The menu-AutoML arm: the TPE menu and the driver that runs it through a dev cell."""
from __future__ import annotations

import json
from pathlib import Path

import optuna
import pytest
import yaml
from _dev_cell_helpers import (
    REPO_ROOT,
    build_git_dev_cell,
    fake_instantiate_recipe,
    finish_node,
    fold_entries,
    load_script,
    seed_executed_root,
)
from optuna.trial import FixedTrial, TrialState

from automil.graph import locked_update

ARMS = ("abmil", "clam_mb", "dtfd_mil", "simple_mil")
ROOT_ID = "node_0001"


def noop(line: str) -> None:
    pass


@pytest.fixture(scope="module")
def space():
    return load_script("dev_optuna_space")


@pytest.fixture(scope="module")
def driver():
    return load_script("dev_optuna")


@pytest.fixture(scope="module")
def dev_cell():
    return load_script("dev_cell")


@pytest.fixture(scope="module")
def dev_loop():
    return load_script("dev_loop")


@pytest.fixture
def study():
    return optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=0))


def off_draw(space, arm: str, **overrides) -> dict:
    """A draw with every switch off and every hyperparameter native (the learning rate at its default)."""
    params = {name: None for name in {**space.COMMON_CONSTANTS, **space.ARM_CONSTANTS[arm]}}
    field, default = space.NATIVE_LR[arm]
    return {**params, "UNFREEZE": "off", field: default,
            **{name: space.NATIVE for name in space.ARM_HPARAMS[arm]}, **overrides}


# ---------------------------------------------------------------------------
# The menu
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("arm", ARMS)
def test_the_native_run_is_a_point_of_every_arms_menu(space, arm):
    for choices in {**space.COMMON_CONSTANTS, **space.ARM_CONSTANTS[arm]}.values():
        assert None in choices
    assert space.UNFREEZE["off"] is None
    assert all(space.NATIVE in choices for choices in space.ARM_HPARAMS[arm].values())

    sample = space.sample(FixedTrial(off_draw(space, arm)), arm)
    field, default = space.NATIVE_LR[arm]
    assert sample.constants == {} and sample.hparams == {field: default}
    assert space.describe(sample) == f"{field}={default:g}"


def test_switches_map_to_recipe_constants_and_hparams_by_the_arms_own_names(space):
    draw = off_draw(space, "dtfd_mil", EMA_DECAY=0.998, UNFREEZE="8-16-0.1", MAX_TILES=512, droprate_2=0.7, wd=1e-3)
    sample = space.sample(FixedTrial(draw), "dtfd_mil")
    assert sample.constants == {"EMA_DECAY": 0.998, "MAX_TILES": 512, "UNFREEZE": (8, 16, 0.1)}
    assert sample.hparams == {"lr": 1e-4, "wd": 1e-3, "droprate_2": 0.7}
    assert sample.params["UNFREEZE"] == "8-16-0.1"  # the study records the key, the recipe gets the tuple
    assert space.describe(sample) == "EMA_DECAY=0.998, MAX_TILES=512, UNFREEZE=(8, 16, 0.1), lr=0.0001, wd=0.001, droprate_2=0.7"
    fine = space.sample(FixedTrial(off_draw(space, "dtfd_mil", EMA_DECAY=0.9995)), "dtfd_mil")
    assert space.describe(fine).startswith("EMA_DECAY=0.9995,")  # never rounded to 1


def test_tile_caps_only_where_the_bag_seam_is_live_and_the_body_scale_only_on_nnmil(space):
    assert set(space.ARM_CONSTANTS["simple_mil"]) == {"BODY_LR_SCALE"}
    for arm in ("abmil", "clam_mb", "dtfd_mil"):
        assert set(space.ARM_CONSTANTS[arm]) == {"MAX_TILES"}
    sample = space.sample(FixedTrial(off_draw(space, "simple_mil", BODY_LR_SCALE=0.1, batch_size=4)), "simple_mil")
    assert sample.constants == {"BODY_LR_SCALE": 0.1}
    assert sample.hparams == {"learning_rate": 3e-4, "batch_size": 4}


def test_class_balancing_is_offered_only_with_a_positive_share(space):
    plain = space.sample(FixedTrial(off_draw(space, "abmil")), "abmil")
    assert "BALANCED" not in plain.params
    on = space.sample(FixedTrial(off_draw(space, "abmil", BALANCED=True)), "abmil", positive_share=0.438)
    assert on.constants == {"POSITIVE_SHARE": 0.438} and on.params["BALANCED"] is True
    off = space.sample(FixedTrial(off_draw(space, "abmil", BALANCED=False)), "abmil", positive_share=0.438)
    assert off.constants == {}


@pytest.mark.parametrize("arm, share", [("titan", None), ("abmil", 1.0), ("abmil", 0.0), ("abmil", "0.4")])
def test_sample_refuses_an_arm_without_a_menu_or_a_share_outside_the_unit_interval(space, arm, share):
    with pytest.raises(ValueError):
        space.sample(FixedTrial(off_draw(space, "abmil")), arm, positive_share=share)


def test_the_learning_rate_spans_a_quarter_to_four_times_the_arms_default(space):
    assert space.LR_SPAN == 4.0
    edge = space.sample(FixedTrial(off_draw(space, "abmil", lr=5e-4 * 4)), "abmil")
    assert edge.hparams == {"lr": 2e-3}
    with pytest.warns(UserWarning, match=r"out of the range.*high=0\.002.*low=0\.000125"):
        space.sample(FixedTrial(off_draw(space, "abmil", lr=5e-4 * 8)), "abmil")


@pytest.mark.parametrize("arm", ARMS)
def test_a_drawn_attempt_is_a_policy_the_sealed_seam_validators_accept(space, arm):
    """Twenty random draws name only the template's constants; one of them instantiates as a policy."""
    recipe = load_script("dev_recipe")
    template = REPO_ROOT / "benchmarks" / "experiments" / "dev" / "_recipes" / "composite.py"
    known = set(recipe._constant_nodes(recipe._recipe_class(recipe.ast.parse(template.read_text()))))
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=1))
    samples = [space.sample(study.ask(), arm, positive_share=0.4) for _ in range(20)]
    assert all(set(sample.constants) <= known for sample in samples)
    richest = max(samples, key=lambda sample: len(sample.constants))
    text = recipe.instantiate_recipe(template, name="t000", constants={**richest.constants, "ARM": arm},
                                     node_id="node_0002", base_commit="abc1234", created_at="2026-10-21T00:00:00")
    assert "class Recipe" in text


# ---------------------------------------------------------------------------
# The driver, against a cell whose orchestrator the test drives
# ---------------------------------------------------------------------------
def score(entry) -> tuple[float, bool]:
    """EMA helps, label smoothing 0.2 hurts and fails the guard."""
    constants = entry.constants or {}
    value = 0.70 + (0.02 if constants.get("EMA_DECAY") else 0.0) - (0.05 if constants.get("LS_EPS") == 0.2 else 0.0)
    return value, constants.get("LS_EPS") == 0.2


class World:
    """A dev cell whose attempts finish when the test says so; nodes are replaced, never edited."""

    def __init__(self, arm: str = "abmil"):
        self.arm = arm
        self.nodes: dict = {}
        self.entries: dict = {}
        self.max_running = 0

    def submit(self, entry) -> str:
        node_id = f"node_{len(self.nodes) + 1:04d}"
        parent = None if entry.parent is None else ROOT_ID
        self.nodes = {**self.nodes, node_id: {"id": node_id, "parent_id": parent, "type": "proposed",
                                              "status": "running", "techniques": [f"recipe:{entry.name}"]}}
        self.entries = {**self.entries, node_id: entry}
        self.max_running = max(self.max_running, self.running_attempts())
        return node_id

    def running_attempts(self) -> int:
        return sum(1 for node in self.nodes.values() if node["status"] == "running" and node["parent_id"])

    def finish(self, k: int = 1, status=lambda node_id: "keep") -> None:
        """The k oldest running nodes end, the root first; ``status`` says how each ends."""
        oldest = [node_id for node_id, node in sorted(self.nodes.items()) if node["status"] == "running"][:k]
        for node_id in oldest:
            value, guard_fail = score(self.entries[node_id])
            self.nodes = {**self.nodes, node_id: {**self.nodes[node_id], "type": "executed", "status": status(node_id),
                                                  "primary_value": value, "guard_fail": guard_fail}}

    def judge(self, node) -> tuple[float, float]:
        return node["primary_value"], (1.0 if node["guard_fail"] else -1.0)

    def cell(self, driver):
        return driver.Cell(self.arm, self.submit, lambda: self.nodes, self.judge)


def test_run_submits_the_root_first_then_keeps_in_flight_attempts_up_to_the_budget(driver, study):
    world = World()
    ticks: list[float] = []

    def sleep(seconds: float) -> None:
        ticks.append(seconds)
        world.finish(k=2)

    driver.run(study, world.cell(driver), driver.Settings(n_trials=7, in_flight=3, poll_seconds=5), sleep=sleep, log=noop)

    assert world.entries[ROOT_ID].parent is None and world.nodes[ROOT_ID]["parent_id"] is None
    attempts = {node_id: entry for node_id, entry in world.entries.items() if node_id != ROOT_ID}
    assert len(attempts) == 7 and world.max_running == 3 and set(ticks) == {5}
    assert all(entry.parent == "baseline" and entry.axis == "optuna" and entry.kind == "hp" for entry in attempts.values())
    assert sorted(entry.name for entry in attempts.values()) == [f"t{n:03d}" for n in range(7)]
    trials = study.get_trials()
    assert [trial.state for trial in trials] == [TrialState.COMPLETE] * 7
    assert {trial.user_attrs["node_id"] for trial in trials} == set(attempts)
    for trial in trials:
        node = world.nodes[trial.user_attrs["node_id"]]
        assert trial.values == [node["primary_value"]]
        assert trial.constraints == {"guard": driver.GUARD_FAIL if node["guard_fail"] else driver.GUARD_PASS}
        assert trial.user_attrs["guard"] == ("fail" if node["guard_fail"] else "pass")
        assert trial.user_attrs["name"] == attempts[trial.user_attrs["node_id"]].name
    assert driver.submitted_nodes(study) == frozenset(attempts)


def test_the_driver_waits_for_a_root_that_is_already_running_and_refuses_a_root_that_crashed(driver, study):
    world = World()
    world.submit(driver.root_entry())
    polls: list[int] = []

    def sleep(_: float) -> None:
        polls.append(1)
        world.finish(k=1, status=lambda node_id: "crash")

    with pytest.raises(driver.DevOptunaError, match="root"):
        driver.run(study, world.cell(driver), driver.Settings(n_trials=1, in_flight=1), sleep=sleep, log=noop)
    assert polls == [1] and len(world.entries) == 1 and study.get_trials() == []


def test_a_crashed_attempt_is_a_failed_trial_that_still_counts_against_the_budget(driver, study):
    world = World()

    def sleep(_: float) -> None:
        world.finish(k=1, status=lambda node_id: "keep" if node_id == ROOT_ID else "timeout")

    driver.run(study, world.cell(driver), driver.Settings(n_trials=3, in_flight=1), sleep=sleep, log=noop)
    assert len(world.entries) == 4  # root + 3 attempts
    assert [trial.state for trial in study.get_trials()] == [TrialState.FAIL] * 3
    assert len(driver.submitted_nodes(study)) == 3


def test_a_restarted_driver_resumes_the_running_attempts_and_fails_the_trial_it_never_submitted(driver, tmp_path):
    world = World()
    cell = world.cell(driver)
    settings = driver.Settings(n_trials=4, in_flight=2)
    study = driver.open_study(tmp_path, "tpe", seed=0)
    driver.ensure_root(cell, sleep=lambda: world.finish(), log=noop)
    running = driver._ask(study, cell, settings, {}, log=noop)
    assert len(running) == 2 and world.running_attempts() == 2
    orphan = study.ask()  # asked, then the driver died before its submit
    orphan.set_user_attr("name", "t999")

    reopened = driver.open_study(tmp_path, "tpe", seed=0)
    resumed = driver.resume(reopened, cell, settings, log=noop)
    assert {node for _, node in resumed.values()} == {node for _, node in running.values()}
    states = [trial.state for trial in reopened.get_trials()]
    assert states == [TrialState.FAIL] * 3 + [TrialState.RUNNING] * 2
    assert {trial.user_attrs["name"] for _, (trial, _) in resumed.items()} == {"t000", "t001"}

    world.finish(k=2)
    assert driver._settle(reopened, cell, resumed, log=noop) == {}
    complete = reopened.get_trials(states=(TrialState.COMPLETE,))
    assert len(complete) == 2 and len(driver.submitted_nodes(reopened)) == 2
    for trial in complete:
        entry = world.entries[trial.user_attrs["node_id"]]
        assert trial.params["EMA_DECAY"] == entry.constants.get("EMA_DECAY")  # the same draw, re-issued
        assert trial.values == [world.nodes[trial.user_attrs["node_id"]]["primary_value"]]
    assert driver._ask(reopened, cell, settings, {}, log=noop).keys() and len(driver.submitted_nodes(reopened)) == 4


def test_settings_refuse_a_budget_or_a_slot_count_below_one(driver):
    for kwargs in ({"n_trials": 0, "in_flight": 1}, {"n_trials": 1, "in_flight": 0}):
        with pytest.raises(driver.DevOptunaError):
            driver.Settings(**kwargs)


def test_status_lists_trials_in_submission_order_with_the_best_within_each_budget(driver, study):
    world = World()
    driver.run(study, world.cell(driver), driver.Settings(n_trials=6, in_flight=2), sleep=lambda _: world.finish(k=2), log=noop)
    rows = driver.trial_rows(study)
    assert [row.node_id for row in rows] == sorted(row.node_id for row in rows)
    assert all(row.state == "COMPLETE" and row.value is not None for row in rows)
    eligible = [row for row in rows[:3] if row.guard == "pass"]
    best = driver.best_within(rows, 3)
    assert best is None if not eligible else best.value == max(row.value for row in eligible)
    text = driver.format_status(study, budgets=(3, 6))
    assert "best of the first 3" in text and "best of the first 6" in text and "t000" in text


# ---------------------------------------------------------------------------
# A real dev cell: the attempt goes through the batch driver's path
# ---------------------------------------------------------------------------
def test_bind_cell_submits_a_drawn_attempt_through_the_batch_path_and_judges_it_under_the_guard(
    driver, space, dev_cell, dev_loop, tmp_path, monkeypatch,
):
    cell_dir = build_git_dev_cell(dev_cell, tmp_path)
    declare_guard(cell_dir)
    calls: list[dict] = []
    monkeypatch.setattr(dev_loop, "_instantiate_recipe", fake_instantiate_recipe(calls))
    cell, adir = driver.bind_cell(cell_dir)
    assert cell.arm == "abmil" and cell.nodes() == {}  # no graph yet

    assert cell.submit(driver.root_entry()) == ROOT_ID
    with pytest.raises(driver.DevOptunaError):
        cell.submit(driver.root_entry())  # a second root is refused, not skipped
    seed_root_result(cell_dir)

    sample = space.sample(FixedTrial(off_draw(space, "abmil", EMA_DECAY=0.998)), "abmil")
    node_id = cell.submit(driver.attempt_entry("t000", sample))
    node = cell.nodes()[node_id]
    assert node["parent_id"] == ROOT_ID and "recipe:t000" in node["techniques"]
    assert node["description"].startswith("t000: EMA_DECAY=0.998")
    assert calls[-1]["name"] == "t000" and calls[-1]["constants"] == {"EMA_DECAY": 0.998, "ARM": "abmil"}
    assert json.loads((adir / "dev_batch.json").read_text())["t000"] == node_id
    assert driver.node_state(node) == "waiting"

    finish_node(cell_dir, node_id, [0.75] * 5)  # companion 0.6 on every fold, like the root
    assert cell.judge(cell.nodes()[node_id]) == (0.75, driver.GUARD_PASS)
    worse = cell.submit(driver.attempt_entry("t001", sample))
    finish_node(cell_dir, worse, [0.75] * 5)
    with locked_update(adir / "graph.json") as graph:
        graph.nodes[worse]["fold_primary_values"] = fold_entries([0.75] * 5, companion=0.5)
    assert cell.judge(cell.nodes()[worse]) == (0.75, driver.GUARD_FAIL)


def seed_root_result(cell_dir: Path) -> None:
    """The submitted root finishes as the daemon leaves it: kept, with its per-fold evidence."""
    finish_node(cell_dir, ROOT_ID, [0.70, 0.72, 0.68, 0.71, 0.69])


def declare_guard(cell_dir: Path) -> None:
    """A companion guard on the test cell (no test host mounts the splits a derived one needs)."""
    path = cell_dir / "automil" / "config.yaml"
    config = yaml.safe_load(path.read_text())
    scoring = {**config["scoring"], "guard": {"metric": "val_bacc", "margin": 0.01}}
    path.write_text(yaml.safe_dump({**config, "scoring": scoring}, sort_keys=False))
