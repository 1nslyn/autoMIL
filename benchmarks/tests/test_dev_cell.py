"""Development cells: dataset YAMLs, the pure config builder and its writer."""
from __future__ import annotations

import csv
import json
import math
import shlex
import statistics
from pathlib import Path

import pytest
import yaml
from _dev_cell_helpers import (
    DEV_ROOT,
    EMA_FOLDS,
    LS_FOLDS,
    LUAD_TEMPLATE,
    LUAD_YAML,
    ROOT_FOLDS,
    SPIKY_CURVE,
    build_luad_cell,
    copy_luad_inputs,
    load_script,
    write_synthetic_cell,
)
from test_preprint_campaign import AGENT_PROTOCOL, _copy_campaign_sources

from automil.scoring import paired_delta_se
from autobench.campaign import (
    PROTOCOL,
    PROTOCOL_VERSION,
    WINNER_FLOOR,
    build_preprint_manifest,
    materialize_discovery_cells,
    max_lift_winner,
    write_manifest,
)
from autobench.config import load_dataset_config
from autobench.guard_margin import derive_guard

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS = REPO_ROOT / "benchmarks" / "datasets"
DEV_DATASETS = {
    "ccrcc": (DATASETS / "cptac" / "ccrcc.yaml", "AUTOBENCH_CCRCC_ROOT"),
    "clwd_dev": (DATASETS / "other" / "clwd_dev.yaml", "AUTOBENCH_CLWD_ROOT"),
}
ARM_FRAMEWORKS = [("clam_mb", "clam"), ("abmil", "abmil"), ("dtfd_mil", "dtfd"), ("simple_mil", "nnmil")]


@pytest.fixture(scope="module")
def dev_cell():
    return load_script("dev_cell")


@pytest.fixture(scope="module")
def dev_loop():
    return load_script("dev_loop")


@pytest.fixture(autouse=True)
def _no_dataset_root(monkeypatch):
    """No test inherits the operator's data roots: the guard needs mounted splits."""
    monkeypatch.delenv("AUTOBENCH_TCGA_LUAD_ROOT", raising=False)


@pytest.mark.parametrize("name", sorted(DEV_DATASETS))
def test_dev_datasets_declare_devcv_and_the_256px_features(name, monkeypatch):
    path, root_var = DEV_DATASETS[name]
    raw = yaml.safe_load(path.read_text())
    assert raw["split_strategies"]["devcv"] == {"train_cohorts": [], "test_cohorts": []}
    assert set(raw["task_strategy_feasibility"]) == set(raw["tasks"])
    assert all("devcv" in strategies for strategies in raw["task_strategy_feasibility"].values())
    assert raw["paths"]["features_base_dir"] == "${output_dir}/20x_256px_0px_overlap"

    # The same facts through the loader every run goes through.
    monkeypatch.setenv(root_var, "/data/root")
    dataset = load_dataset_config(str(path))
    assert "devcv" in dataset.split_strategies
    assert all("devcv" in strategies for strategies in dataset.task_strategy_feasibility.values())
    assert dataset.features_base_dir == "/data/root/trident_output/20x_256px_0px_overlap"


# --- the pure config builder -------------------------------------------------------


@pytest.fixture(scope="module")
def campaign_repo(tmp_path_factory):
    """A fake repo with the checked-in roster sources and its frozen manifest."""
    repo = tmp_path_factory.mktemp("campaign") / "repo"
    _copy_campaign_sources(repo)
    manifest = repo / "benchmarks/campaigns/preprint_130/manifest.json"
    write_manifest(build_preprint_manifest(repo), manifest)
    return repo, manifest


def _campaign_config(campaign_repo, framework: str) -> dict:
    repo, manifest = campaign_repo
    cell_id = f"tcga_luad__kras__hoptimus1__{framework}__s42__{PROTOCOL_VERSION}"
    roots = materialize_discovery_cells(
        manifest, repo / "benchmarks/campaigns/preprint_130/runtime", repo,
        agent_protocol=AGENT_PROTOCOL, only_cells={cell_id},
    )
    return yaml.safe_load((roots[0] / "config.yaml").read_text())


def _command_tokens(config: dict) -> list[str]:
    """The run command's tokens, with the ``--dataset`` spelling masked."""
    tokens = shlex.split(config["run"]["command"])
    index = tokens.index("--dataset")
    return [*tokens[: index + 1], "<dataset>", *tokens[index + 2:]]


@pytest.mark.parametrize(("arm", "framework"), ARM_FRAMEWORKS)
def test_dev_config_equals_the_campaign_materializer_on_the_campaign_owned_blocks(
    dev_cell, campaign_repo, arm, framework,
):
    repo, _ = campaign_repo
    campaign = _campaign_config(campaign_repo, framework)
    dev = dev_cell.build_dev_config(
        repo / LUAD_YAML, "kras", "hoptimus1", arm,
        template_path=repo / LUAD_TEMPLATE, eval_budget=200,
        max_concurrent_per_gpu=4, repo_root=repo, no_guard=True,
    )
    for block in ("metrics", "registry", "training", "task"):
        assert dev[block] == campaign[block], block
    for key in ("formula", "se_multiplier", "accept_margin", "technique_map"):
        assert dev["scoring"][key] == campaign["scoring"][key], key
    assert dev["data"]["num_folds"] == campaign["data"]["num_folds"] == 5
    assert dev["data"]["seed"] == campaign["data"]["seed"] == 42
    assert dev["encoders"]["primary"] == campaign["encoders"]["primary"] == "hoptimus1"
    assert dev["orchestrator"]["default_timeout_min"] == campaign["orchestrator"]["default_timeout_min"]
    assert dev["run"]["mil_model"] == campaign["run"]["mil_model"]
    assert dev["run"]["script"] is None
    assert _command_tokens(dev) == _command_tokens(campaign)
    assert "--strategy devcv" in dev["run"]["command"] and "--folds 0,1,2,3,4" in dev["run"]["command"]


def test_dev_config_is_a_plain_project_with_its_own_budget_and_policy_folder(dev_cell, tmp_path):
    repo = copy_luad_inputs(tmp_path / "repo")
    config, spec, cell_dir = build_luad_cell(dev_cell, repo, arm="dtfd_mil", eval_budget=150)
    assert not {"campaign", "activity", "baseline", "cohort"} & set(config)
    assert config["cap"] == {
        "mode": "wall_clock", "budget": "90d", "safety_buffer": "1h", "eval_budget": 150,
    }
    assert config["files"]["editable"] == [
        f"{(DEV_ROOT / spec.cell_name).as_posix()}/automil/variants/_policies/*.py"
    ]
    assert config["project"]["name"] == spec.cell_name == "tcga_luad__kras__hoptimus1__dtfd_mil"
    assert config["orchestrator"]["max_concurrent_per_gpu"] == 4
    assert config["env"] == {
        "passthrough": ["AUTOBENCH_ROOT", "AUTOBENCH_TCGA_LUAD_ROOT", "AUTOBENCH_BAG_CACHE", "NNMIL_NUM_WORKERS"],
        "dotenv_files": ["benchmarks/.env"],
    }
    assert 8500 <= config["viz"]["port"] < 8900
    assert "scoring" in config and "guard" not in config["scoring"]
    assert (spec.framework, spec.model) == ("dtfd", "dtfd_mil")
    assert cell_dir == repo / DEV_ROOT / spec.cell_name


def test_dev_config_points_the_data_block_at_its_own_dataset_not_the_template_cohort(dev_cell, tmp_path):
    repo = copy_luad_inputs(tmp_path / "repo")
    config, _, _ = build_luad_cell(dev_cell, repo)
    data = config["data"]
    assert data["features_dir"] == (
        "${AUTOBENCH_TCGA_LUAD_ROOT}/trident_output/20x_224px_0px_overlap/features_hoptimus1"
    )
    assert data["splits_dir"] == "${AUTOBENCH_TCGA_LUAD_ROOT}/benchmark/splits/devcv/kras"
    assert data["mapping_csv"] == "${AUTOBENCH_TCGA_LUAD_ROOT}/normalized_manifest.csv"
    assert config["encoders"]["available"]["hoptimus1"] == 1536


@pytest.mark.parametrize(("task", "encoder", "arm", "message"), [
    ("nope", "hoptimus1", "abmil", "task 'nope'"),
    ("kras", "nope_encoder", "abmil", "encoder 'nope_encoder'"),
    ("kras", "hoptimus1", "nope_arm", "arm 'nope_arm'"),
    ("os", "hoptimus1", "abmil", "survival"),
])
def test_dev_config_refuses_what_the_dataset_or_the_arm_table_does_not_declare(
    dev_cell, tmp_path, task, encoder, arm, message,
):
    repo = copy_luad_inputs(tmp_path / "repo")
    with pytest.raises(dev_cell.DevCellError, match=message):
        dev_cell.build_dev_config(
            repo / LUAD_YAML, task, encoder, arm, template_path=repo / LUAD_TEMPLATE,
            eval_budget=200, max_concurrent_per_gpu=4, repo_root=repo,
        )


def test_dev_config_refuses_a_cell_outside_the_repo(dev_cell, tmp_path):
    repo = copy_luad_inputs(tmp_path / "repo")
    with pytest.raises(dev_cell.DevCellError, match="inside the repo"):
        dev_cell.build_dev_config(
            repo / LUAD_YAML, "kras", "hoptimus1", "abmil", template_path=repo / LUAD_TEMPLATE,
            eval_budget=200, max_concurrent_per_gpu=4, repo_root=repo, out_root=tmp_path / "elsewhere",
        )


# --- seed replicates ---------------------------------------------------------------------


def _flag_value(config: dict, flag: str) -> str:
    """The token the run command passes after ``flag``."""
    tokens = shlex.split(config["run"]["command"])
    return tokens[tokens.index(flag) + 1]


def _tokens_without_seed(config: dict) -> list[str]:
    """The run command's tokens with ``--seed`` and its value taken out."""
    tokens = shlex.split(config["run"]["command"])
    index = tokens.index("--seed")
    return [*tokens[:index], *tokens[index + 2:]]


def test_a_seed_replicate_has_its_own_cell_folder_and_trains_under_its_seed(dev_cell, tmp_path):
    repo = copy_luad_inputs(tmp_path / "repo")
    base, _, base_dir = build_luad_cell(dev_cell, repo)
    config, spec, cell_dir = build_luad_cell(dev_cell, repo, seed=43)

    assert _flag_value(config, "--seed") == "43"
    assert cell_dir.name == f"{base_dir.name}__seed43" == "tcga_luad__kras__hoptimus1__abmil__seed43"
    assert config["project"]["name"] == cell_dir.name
    assert config["files"]["editable"] == [
        f"{(DEV_ROOT / cell_dir.name).as_posix()}/automil/variants/_policies/*.py"
    ]
    # The seed changes the training only: the replicate reads the protocol cell's folds
    # and runs the protocol cell's command otherwise.
    assert config["data"] == base["data"]
    assert _tokens_without_seed(config) == _tokens_without_seed(base)
    assert {key for key in config if config[key] != base[key]} == {"project", "files", "run", "viz"}

    dev_cell.write_dev_cell(cell_dir, config, spec.record())
    assert json.loads((cell_dir / "automil" / "dev_cell.json").read_text())["seed"] == 43


def test_replicates_get_their_own_viz_ports_so_they_can_run_side_by_side(dev_cell, tmp_path):
    repo = copy_luad_inputs(tmp_path / "repo")
    ports = {seed: build_luad_cell(dev_cell, repo, seed=seed)[0]["viz"]["port"] for seed in (42, 43, 44)}
    assert len(set(ports.values())) == 3 and all(8500 <= port < 8900 for port in ports.values())


def test_the_default_seed_is_the_protocol_seed_and_is_recorded_like_any_other(dev_cell, tmp_path):
    repo = copy_luad_inputs(tmp_path / "repo")
    config, spec, cell_dir = build_luad_cell(dev_cell, repo)  # no seed given
    assert _flag_value(config, "--seed") == str(PROTOCOL["seed"]) == "42"
    assert cell_dir.name == "tcga_luad__kras__hoptimus1__abmil"  # no suffix
    assert build_luad_cell(dev_cell, repo, seed=PROTOCOL["seed"]) == (config, spec, cell_dir)

    dev_cell.write_dev_cell(cell_dir, config, spec.record())
    assert json.loads((cell_dir / "automil" / "dev_cell.json").read_text())["seed"] == 42


@pytest.mark.parametrize("seed", [0, -43, True])
def test_dev_config_refuses_a_seed_that_is_not_a_positive_integer(dev_cell, tmp_path, seed):
    repo = copy_luad_inputs(tmp_path / "repo")
    with pytest.raises(dev_cell.DevCellError, match="seed must be a positive integer"):
        build_luad_cell(dev_cell, repo, seed=seed)


# --- the companion guard needs mounted splits -----------------------------------------


def _mount_devcv_splits(root: Path, task: str = "kras") -> Path:
    """Five folds of 10 validation slides (5 per class) under ``<root>/benchmark``."""
    benchmark = root / "benchmark"
    (benchmark / "dataset_csv").mkdir(parents=True)
    labels = {f"s{i:02d}": ("mutant" if i % 2 else "wildtype") for i in range(50)}
    with (benchmark / "dataset_csv" / f"{task}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["slide_id", "label"])
        writer.writerows(labels.items())
    splits = benchmark / "splits" / "devcv" / task
    splits.mkdir(parents=True)
    for fold in range(5):
        validation = [f"s{i:02d}" for i in range(fold * 10, fold * 10 + 10)]
        with (splits / f"splits_{fold}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["train", "val", "test"])
            writer.writerows((f"s{i:02d}" if i < 10 else "", validation[i] if i < 10 else "", "") for i in range(10))
    return benchmark


def test_guard_is_derived_from_the_devcv_validation_splits_when_they_are_mounted(
    dev_cell, tmp_path, monkeypatch,
):
    repo = copy_luad_inputs(tmp_path / "repo")
    benchmark = _mount_devcv_splits(tmp_path / "data")
    monkeypatch.setenv("AUTOBENCH_TCGA_LUAD_ROOT", str(tmp_path / "data"))
    config, _, _ = build_luad_cell(dev_cell, repo, no_guard=False)
    guard = config["scoring"]["guard"]
    assert guard == derive_guard(benchmark, "devcv", "kras", range(5))
    assert guard["metric"] == "val_bacc" and guard["strategy"] == "devcv"
    assert sorted(guard["validation_class_counts"]) == ["0", "1", "2", "3", "4"]


@pytest.mark.parametrize("mounted", [False, True])
def test_no_guard_omits_the_guard_with_a_note_whether_or_not_the_splits_are_there(
    dev_cell, tmp_path, monkeypatch, capsys, mounted,
):
    repo = copy_luad_inputs(tmp_path / "repo")
    if mounted:  # `--no-guard` means no guard, even where one could be derived
        _mount_devcv_splits(tmp_path / "data")
        monkeypatch.setenv("AUTOBENCH_TCGA_LUAD_ROOT", str(tmp_path / "data"))
    config, _, _ = build_luad_cell(dev_cell, repo, no_guard=True)
    assert "guard" not in config["scoring"]
    assert "scoring.guard omitted" in capsys.readouterr().err


@pytest.mark.parametrize(("mounted", "reason"), [
    (False, "AUTOBENCH_TCGA_LUAD_ROOT is not set"),
    (True, "does not exist"),  # the data root resolves but no devcv splits were prepared under it
])
def test_a_cell_that_cannot_be_guarded_is_refused_with_the_reason_and_the_flag(
    dev_cell, tmp_path, monkeypatch, mounted, reason,
):
    repo = copy_luad_inputs(tmp_path / "repo")
    if mounted:
        (tmp_path / "data" / "benchmark").mkdir(parents=True)
        monkeypatch.setenv("AUTOBENCH_TCGA_LUAD_ROOT", str(tmp_path / "data"))
    with pytest.raises(dev_cell.DevCellError, match=reason) as raised:
        build_luad_cell(dev_cell, repo, no_guard=False)
    assert "--no-guard" in str(raised.value) and "no companion guard" in str(raised.value)


def test_a_broken_split_set_is_an_error_not_a_silently_missing_guard(dev_cell, tmp_path, monkeypatch):
    repo = copy_luad_inputs(tmp_path / "repo")
    benchmark = _mount_devcv_splits(tmp_path / "data")
    (benchmark / "splits" / "devcv" / "kras" / "splits_3.csv").unlink()
    monkeypatch.setenv("AUTOBENCH_TCGA_LUAD_ROOT", str(tmp_path / "data"))
    with pytest.raises(dev_cell.DevCellError, match="companion guard"):
        build_luad_cell(dev_cell, repo, no_guard=False)


# --- publishing a cell -----------------------------------------------------------------


def test_write_dev_cell_publishes_a_complete_cell_in_one_step_and_never_a_graph(dev_cell, tmp_path):
    repo = copy_luad_inputs(tmp_path / "repo")
    config, spec, cell_dir = build_luad_cell(dev_cell, repo)
    assert dev_cell.write_dev_cell(cell_dir, config, spec.record()) == "created"

    adir = cell_dir / "automil"
    assert yaml.safe_load((adir / "config.yaml").read_text()) == config
    assert json.loads((adir / "dev_cell.json").read_text()) == {
        "cell_name": "tcga_luad__kras__hoptimus1__abmil", "dataset": "tcga_luad", "task": "kras",
        "task_family": "binary", "encoder": "hoptimus1", "arm": "abmil",
        "framework": "abmil", "model": "abmil", "seed": 42,
    }
    for name in (".gitignore", "plan.md", "learnings.md", "variants/_policies/.gitkeep"):
        assert (adir / name).is_file(), name
    assert "graph.json" in (adir / ".gitignore").read_text().splitlines()
    assert not (adir / "graph.json").exists()
    assert [p.name for p in cell_dir.parent.iterdir()] == [spec.cell_name]  # no staging leftovers


def test_rewriting_an_identical_cell_is_a_no_op_and_a_drifted_one_is_refused(dev_cell, tmp_path):
    repo = copy_luad_inputs(tmp_path / "repo")
    config, spec, cell_dir = build_luad_cell(dev_cell, repo)
    dev_cell.write_dev_cell(cell_dir, config, spec.record())
    config_path = cell_dir / "automil" / "config.yaml"
    before = config_path.stat().st_mtime_ns

    assert dev_cell.write_dev_cell(cell_dir, config, spec.record()) == "unchanged"
    assert config_path.stat().st_mtime_ns == before

    drifted, _, _ = build_luad_cell(dev_cell, repo, eval_budget=100)
    with pytest.raises(dev_cell.DevCellError, match=r"cap\.eval_budget"):
        dev_cell.write_dev_cell(cell_dir, drifted, spec.record())
    assert yaml.safe_load(config_path.read_text()) == config  # nothing was overwritten

    config_path.unlink()
    with pytest.raises(dev_cell.DevCellError, match="incomplete"):
        dev_cell.write_dev_cell(cell_dir, config, spec.record())


def test_materialize_cli_creates_the_cell_then_reports_it_unchanged(dev_cell, tmp_path, monkeypatch, capsys):
    repo = copy_luad_inputs(tmp_path / "repo")
    monkeypatch.setattr(dev_cell, "REPO_ROOT", repo)
    argv = ["materialize", "--dataset", str(repo / LUAD_YAML), "--task", "kras", "--arm", "clam_mb",
            "--encoder", "hoptimus1", "--out", str(repo / DEV_ROOT), "--no-guard"]
    dev_cell.main(argv)
    cell = repo / DEV_ROOT / "tcga_luad__kras__hoptimus1__clam_mb"
    assert (cell / "automil" / "config.yaml").is_file()
    assert f"created: {cell}" in capsys.readouterr().out

    dev_cell.main(argv)
    assert f"unchanged: {cell}" in capsys.readouterr().out


def test_materialize_cli_refuses_a_cell_it_cannot_guard_unless_no_guard_is_passed(
    dev_cell, tmp_path, monkeypatch, capsys,
):
    repo = copy_luad_inputs(tmp_path / "repo")
    monkeypatch.setattr(dev_cell, "REPO_ROOT", repo)
    argv = ["materialize", "--dataset", str(repo / LUAD_YAML), "--task", "kras", "--arm", "abmil",
            "--out", str(repo / DEV_ROOT)]
    cell = repo / DEV_ROOT / "tcga_luad__kras__hoptimus1__abmil"
    with pytest.raises(SystemExit) as raised:  # no data root, so no devcv splits to derive the guard from
        dev_cell.main(argv)
    err = capsys.readouterr().err
    assert raised.value.code == 2 and "AUTOBENCH_TCGA_LUAD_ROOT is not set" in err and "--no-guard" in err
    assert not cell.exists()

    _mount_devcv_splits(tmp_path / "data")
    monkeypatch.setenv("AUTOBENCH_TCGA_LUAD_ROOT", str(tmp_path / "data"))
    dev_cell.main(argv)  # the splits are there now: the guard is derived without the flag
    assert "guard" in yaml.safe_load((cell / "automil" / "config.yaml").read_text())["scoring"]
    assert f"created: {cell}" in capsys.readouterr().out


def test_materialize_cli_refuses_with_a_message_and_exit_status_2(dev_cell, tmp_path, monkeypatch, capsys):
    repo = copy_luad_inputs(tmp_path / "repo")
    monkeypatch.setattr(dev_cell, "REPO_ROOT", repo)
    with pytest.raises(SystemExit) as raised:
        dev_cell.main(["materialize", "--dataset", str(repo / LUAD_YAML), "--task", "nope",
                       "--arm", "abmil", "--out", str(repo / DEV_ROOT)])
    assert raised.value.code == 2
    assert "task 'nope'" in capsys.readouterr().err
    assert not (repo / DEV_ROOT).exists()


def test_materialize_cli_writes_a_seed_replicate_beside_the_protocol_cell(
    dev_cell, tmp_path, monkeypatch, capsys,
):
    repo = copy_luad_inputs(tmp_path / "repo")
    monkeypatch.setattr(dev_cell, "REPO_ROOT", repo)
    argv = ["materialize", "--dataset", str(repo / LUAD_YAML), "--task", "kras", "--arm", "abmil",
            "--out", str(repo / DEV_ROOT), "--no-guard"]
    dev_cell.main(argv)
    base = repo / DEV_ROOT / "tcga_luad__kras__hoptimus1__abmil"
    protocol_config = (base / "automil" / "config.yaml").read_bytes()
    capsys.readouterr()

    dev_cell.main([*argv, "--seed", "43"])
    replicate = repo / DEV_ROOT / "tcga_luad__kras__hoptimus1__abmil__seed43"
    assert f"created: {replicate}" in capsys.readouterr().out
    config = yaml.safe_load((replicate / "automil" / "config.yaml").read_text())
    assert _flag_value(config, "--seed") == "43"
    assert json.loads((replicate / "automil" / "dev_cell.json").read_text())["seed"] == 43
    assert (base / "automil" / "config.yaml").read_bytes() == protocol_config  # protocol cell untouched

    dev_cell.main([*argv, "--seed", "43"])
    assert f"unchanged: {replicate}" in capsys.readouterr().out
    dev_cell.main([*argv, "--seed", str(PROTOCOL["seed"])])  # the protocol seed names the protocol cell
    assert f"unchanged: {base}" in capsys.readouterr().out


@pytest.mark.parametrize("seed", ["0", "-1"])
def test_materialize_cli_refuses_a_seed_that_is_not_a_positive_integer(
    dev_cell, tmp_path, monkeypatch, capsys, seed,
):
    repo = copy_luad_inputs(tmp_path / "repo")
    monkeypatch.setattr(dev_cell, "REPO_ROOT", repo)
    with pytest.raises(SystemExit) as raised:
        dev_cell.main(["materialize", "--dataset", str(repo / LUAD_YAML), "--task", "kras", "--arm", "abmil",
                       "--out", str(repo / DEV_ROOT), "--no-guard", "--seed", seed])
    err = capsys.readouterr().err
    assert raised.value.code == 2 and "--seed" in err and "positive integer" in err
    assert not (repo / DEV_ROOT).exists()


# --- readout ----------------------------------------------------------------------------


def _nodes(data: dict) -> dict[str, dict]:
    return {node["id"]: node for node in data["nodes"]}


def _paired(child: list[float], parent: list[float]) -> tuple[float, float]:
    """Hand formula: mean per-fold delta and its standard error."""
    deltas = [c - p for c, p in zip(child, parent)]
    return statistics.fmean(deltas), statistics.stdev(deltas) / math.sqrt(len(deltas))


@pytest.mark.parametrize("root_in_metadata", [False, True])
def test_readout_reproduces_paired_deltas_against_the_parent_and_the_root(dev_loop, tmp_path, root_in_metadata):
    cell = write_synthetic_cell(tmp_path / "cell", root_in_metadata=root_in_metadata)
    data = dev_loop.build_readout(cell)
    nodes = _nodes(data)
    assert [node["id"] for node in data["nodes"]] == ["node_0001", "node_0002", "node_0003", "node_0004"]
    assert data["counts"] == {"executed": 4, "running": 1, "pending": 1} and data["root"] == "node_0001"

    delta, se = _paired(EMA_FOLDS, ROOT_FOLDS)
    assert se == pytest.approx(paired_delta_se(dict(enumerate(EMA_FOLDS)), dict(enumerate(ROOT_FOLDS))), abs=1e-12)
    ema = nodes["node_0002"]
    assert ema["fold_values"] == EMA_FOLDS and ema["fold_indices"] == [0, 1, 2, 3, 4]
    assert ema["vs_parent"] == ema["vs_root"] == {"delta": pytest.approx(delta), "se": pytest.approx(se)}

    ls = nodes["node_0003"]
    assert ls["vs_parent"] == {"delta": pytest.approx(_paired(LS_FOLDS, EMA_FOLDS)[0]),
                               "se": pytest.approx(_paired(LS_FOLDS, EMA_FOLDS)[1])}
    assert ls["vs_root"] == {"delta": pytest.approx(_paired(LS_FOLDS, ROOT_FOLDS)[0]),
                             "se": pytest.approx(_paired(LS_FOLDS, ROOT_FOLDS)[1])}
    for bare in (nodes["node_0001"], nodes["node_0004"]):  # the root, and a crash with no folds
        assert bare["vs_parent"] is None and bare["vs_root"] is None
    assert nodes["node_0004"]["fold_values"] is None and nodes["node_0004"]["status"] == "crash"


def test_readout_parses_selected_epochs_and_the_five_epoch_level_jitter_and_hindsight_gap(dev_loop, tmp_path):
    log = _nodes(dev_loop.build_readout(write_synthetic_cell(tmp_path / "cell")))["node_0002"]["log"]
    assert [row["fold"] for row in log] == [0, 1, 2, 3, 4]
    spiky = log[0]  # argmax at epoch 3; the five epochs around it are epochs 1-5
    assert (spiky["selected_epoch"], spiky["n_epochs"], spiky["peak"]) == (3, 10, 0.74)
    assert spiky["level"] == pytest.approx(statistics.fmean([0.61, 0.62, 0.74, 0.60, 0.70]))
    assert spiky["jitter"] == pytest.approx(0.74 - spiky["level"])
    assert spiky["hindsight_level"] == pytest.approx(statistics.fmean(SPIKY_CURVE[5:]))  # best window: epochs 5-9
    assert spiky["hindsight_gap"] == pytest.approx(spiky["hindsight_level"] - spiky["level"])
    rising = log[1]  # a short run: the window is every epoch
    assert (rising["selected_epoch"], rising["n_epochs"]) == (4, 5)
    assert rising["level"] == pytest.approx(0.60) and rising["jitter"] == pytest.approx(0.10)
    assert rising["hindsight_gap"] == pytest.approx(0.0)


@pytest.mark.parametrize(("framework", "level"), [("nnmil", 0.65), ("abmil", (0.0 + 0.6 + 0.7) / 3)])
def test_a_zero_auc_epoch_is_nnmils_undefined_sentinel_and_only_there(dev_loop, tmp_path, framework, level):
    cell = write_synthetic_cell(tmp_path / "cell", framework=framework, first_curves=[[0.0, 0.6, 0.7]] * 5)
    row = _nodes(dev_loop.build_readout(cell))["node_0002"]["log"][0]
    assert row["selected_epoch"] == 2 and row["level"] == pytest.approx(level)


def test_readout_verdict_is_the_winner_rule_over_the_complete_attempts_and_the_n_attempt_bar(dev_loop, tmp_path):
    verdict = dev_loop.build_readout(write_synthetic_cell(tmp_path / "cell"))["verdict"]
    pool = [{"candidate_id": "node_0002", "candidate_sha256": "c2" * 8, "fold_values": EMA_FOLDS},
            {"candidate_id": "node_0003", "candidate_sha256": "c3" * 8, "fold_values": LS_FOLDS}]
    record = max_lift_winner(ROOT_FOLDS, pool)  # the crash and the queued nodes are not complete attempts
    assert verdict["available"] is True and verdict["n_pool"] == 2
    assert verdict["frozen_rule"] == record and record["leader"] == "node_0002"
    z = statistics.NormalDist().inv_cdf(0.95 ** (1 / 2))
    bar = max(WINNER_FLOOR, z * record["standard_error"])
    assert verdict["sidak"] == {"n": 2, "z": pytest.approx(z), "bar": pytest.approx(bar),
                                "accepted": record["leader_lift"] > bar}


def test_readout_without_a_root_baseline_has_a_stated_reason_and_no_verdict(dev_loop, tmp_path):
    cell = write_synthetic_cell(tmp_path / "cell")
    graph_path = cell / "automil" / "graph.json"
    graph = json.loads(graph_path.read_text())
    del graph["nodes"]["node_0001"]["fold_primary_values"]
    graph_path.write_text(json.dumps(graph))
    verdict = dev_loop.build_readout(cell)["verdict"]
    assert verdict["available"] is False and "per-fold" in verdict["reason"]


def test_an_unreadable_run_log_is_reported_on_its_node_and_does_not_stop_the_readout(dev_loop, tmp_path):
    cell = write_synthetic_cell(tmp_path / "cell")
    (cell / "automil/orchestrator/archive/node_0003/run.log").write_text("[epoch 0] val_auc=0.5\n")
    nodes = _nodes(dev_loop.build_readout(cell))
    assert nodes["node_0003"]["log"] is None and "epoch lines" in nodes["node_0003"]["log_error"]
    assert nodes["node_0002"]["log"] is not None and nodes["node_0004"]["log_error"] is None


def test_readout_needs_a_dev_cell_with_a_graph(dev_loop, tmp_path):
    with pytest.raises(dev_loop.DevLoopError, match="dev_cell.json"):
        dev_loop.build_readout(tmp_path / "missing")
    cell = write_synthetic_cell(tmp_path / "cell")
    (cell / "automil" / "graph.json").unlink()
    with pytest.raises(dev_loop.DevLoopError, match="graph.json"):
        dev_loop.build_readout(cell)


def test_readout_cli_prints_the_executed_nodes_as_text_and_the_same_data_as_json(
    dev_cell, dev_loop, tmp_path, capsys,
):
    cell = write_synthetic_cell(tmp_path / "cell")
    dev_cell.main(["readout", str(cell)])
    text = capsys.readouterr().out
    assert all(node_id in text for node_id in ("node_0001", "node_0002", "node_0003", "node_0004"))
    assert "node_0005" not in text and "node_0006" not in text  # queued, not executed

    dev_cell.main(["readout", str(cell), "--json"])
    assert json.loads(capsys.readouterr().out) == json.loads(json.dumps(dev_loop.build_readout(cell)))

    with pytest.raises(SystemExit) as raised:
        dev_cell.main(["readout", str(tmp_path / "missing")])
    assert raised.value.code == 2 and "dev_cell.json" in capsys.readouterr().err
