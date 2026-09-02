"""Contract tests for the standalone SLURM job-shape predictor
(campaign_shape.py).

campaign_shape.py is stdlib-only and delivered standalone to the cluster, so
it is loaded here by file path rather than imported as a package module (same
pattern as test_campaign_export.py).
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "campaign_shape", REPO_ROOT / "benchmarks/scripts/campaign_shape.py"
    )
    module = importlib.util.module_from_spec(spec)
    # dataclasses looks up sys.modules[cls.__module__] while processing the
    # class body, so the module must be registered before exec_module runs.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cs = _load_module()


# ---------------------------------------------------------------------------
# predict_hours


def test_predict_hours_known_value_one_gpu():
    # e5 = 3600s = 1h -> attempt=0.6, promo=0.4, capacity=4*1*0.8=3.2
    # predicted = 0.6 + 30*0.6/3.2 + 10*0.4/3.2 + 2.0 = 9.475
    predicted = cs.predict_hours(3600.0, 1)
    assert predicted == pytest.approx(9.475, abs=1e-9)


def test_predict_hours_more_gpus_predicts_less_time():
    e5_seconds = 6.2 * 3600
    one_gpu = cs.predict_hours(e5_seconds, 1)
    four_gpu = cs.predict_hours(e5_seconds, 4)
    assert four_gpu < one_gpu


# ---------------------------------------------------------------------------
# choose_shape


def test_choose_shape_small_baseline_gets_smallest_shape():
    e5_seconds = 0.05 * 3600
    shape = cs.choose_shape(e5_seconds)
    assert shape is not None
    assert (shape.gpus, shape.wall_hours) == (1, 12)
    assert shape.cpus == 12
    assert shape.mem_gb == 128
    assert shape.whole_node is False
    assert shape.predicted_hours == pytest.approx(cs.predict_hours(e5_seconds, 1))


def test_choose_shape_medium_baseline_needs_four_gpus_and_24h():
    e5_seconds = 6.2 * 3600

    # 1 GPU cannot fit even the 24h wall.
    assert cs.predict_hours(e5_seconds, 1) > cs.FIT_FRACTION * 24
    # 4 GPUs fit the 24h wall but not the 12h wall -> (4, 24) is the first
    # candidate that fits under either preference.
    assert cs.predict_hours(e5_seconds, 4) > cs.FIT_FRACTION * 12
    assert cs.predict_hours(e5_seconds, 4) <= cs.FIT_FRACTION * 24

    shape = cs.choose_shape(e5_seconds)
    assert shape is not None
    assert (shape.gpus, shape.wall_hours) == (4, 24)
    assert shape.cpus == 48
    assert shape.mem_gb == 512
    assert shape.whole_node is True
    assert shape.predicted_hours == pytest.approx(cs.predict_hours(e5_seconds, 4))


def test_choose_shape_returns_none_when_nothing_fits():
    e5_seconds = 1000 * 3600  # absurdly slow baseline: no candidate fits
    assert cs.choose_shape(e5_seconds) is None


def test_candidate_order_cheap_minimizes_gpus_then_wall():
    assert cs.candidate_shapes("cheap") == ((1, 12), (1, 24), (2, 12), (2, 24), (4, 12), (4, 24))


def test_candidate_order_fast_minimizes_wall_then_gpus():
    assert cs.candidate_shapes("fast") == ((1, 12), (2, 12), (4, 12), (1, 24), (2, 24), (4, 24))


def test_unknown_preference_is_refused():
    with pytest.raises(ValueError, match="unknown preference"):
        cs.candidate_shapes("greedy")


def test_preference_changes_the_shape_for_a_typical_abmil_cell():
    e5_seconds = 2.17 * 3600  # median abmil 5-fold baseline on fir
    cheap = cs.choose_shape(e5_seconds)              # default
    fast = cs.choose_shape(e5_seconds, prefer="fast")
    assert (cheap.gpus, cheap.wall_hours) == (1, 24)
    assert (fast.gpus, fast.wall_hours) == (4, 12)
    assert cheap.gpus * cheap.predicted_hours < fast.gpus * fast.predicted_hours  # fewer GPU-hours


# ---------------------------------------------------------------------------
# shape_cells


def _write_state(runtime: Path, cell_id: str, state: dict) -> None:
    cell_dir = runtime / cell_id
    cell_dir.mkdir(parents=True)
    (cell_dir / "campaign_state.json").write_text(json.dumps(state))


def _baseline_state(total_seconds: float) -> dict:
    return {
        "baseline": {
            "resources": {
                "elapsed_seconds": {"total": total_seconds},
            },
        },
    }


@pytest.fixture()
def fabricated_runtime(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    _write_state(runtime, "cell_a", _baseline_state(0.05 * 3600))
    _write_state(runtime, "cell_b", _baseline_state(6.2 * 3600))
    _write_state(runtime, "cell_c", {"baseline": None})
    return runtime


def test_shape_cells_reports_two_shapes_and_one_unshaped(fabricated_runtime):
    reports = cs.shape_cells(fabricated_runtime, ["cell_a", "cell_b", "cell_c"])
    assert set(reports) == {"cell_a", "cell_b", "cell_c"}

    shaped = [r for r in reports.values() if r.shape is not None]
    unshaped = [r for r in reports.values() if r.shape is None]
    assert len(shaped) == 2
    assert len(unshaped) == 1
    assert unshaped[0].cell_id == "cell_c"
    assert unshaped[0].reason  # non-empty explanation
    assert reports["cell_a"].shape.gpus == 1
    assert reports["cell_b"].shape.gpus == 4


def test_shape_cells_missing_campaign_state_is_unshaped_not_a_crash(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "cell_missing").mkdir()  # no campaign_state.json inside
    reports = cs.shape_cells(runtime, ["cell_missing"])
    assert reports["cell_missing"].shape is None
    assert reports["cell_missing"].reason


def test_shape_cells_null_elapsed_total_is_unshaped_not_a_crash(tmp_path):
    runtime = tmp_path / "runtime"
    _write_state(
        runtime, "cell_null",
        {"baseline": {"resources": {"elapsed_seconds": {"total": None}}}},
    )
    reports = cs.shape_cells(runtime, ["cell_null"])
    assert reports["cell_null"].shape is None
    assert reports["cell_null"].reason


# ---------------------------------------------------------------------------
# CLI


def test_cli_json_round_trips(fabricated_runtime, capsys):
    rc = cs.main(["--runtime", str(fabricated_runtime), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)

    assert set(payload) == {"cell_a", "cell_b", "cell_c"}
    assert payload["cell_a"]["gpus"] == 1
    assert payload["cell_a"]["wall_hours"] == 12
    assert payload["cell_b"]["gpus"] == 4
    assert payload["cell_b"]["wall_hours"] == 24
    assert isinstance(payload["cell_c"]["unshaped"], str)
    assert payload["cell_c"]["unshaped"]


def test_cli_cell_field_prints_a_bare_int(fabricated_runtime, capsys):
    rc = cs.main([
        "--runtime", str(fabricated_runtime),
        "--cell", "cell_a", "--field", "gpus",
    ])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    assert int(out) == 1


def test_cli_unknown_cell_via_dash_dash_cell_exits_2(fabricated_runtime, capsys):
    rc = cs.main([
        "--runtime", str(fabricated_runtime),
        "--cell", "does_not_exist", "--field", "gpus",
    ])
    assert rc == 2
    assert capsys.readouterr().err


def test_cli_unknown_cell_via_dash_dash_cells_exits_2(fabricated_runtime, capsys):
    rc = cs.main([
        "--runtime", str(fabricated_runtime),
        "--cells", "cell_a,does_not_exist",
        "--json",
    ])
    assert rc == 2
    assert capsys.readouterr().err


def test_cli_invalid_runtime_exits_2(tmp_path, capsys):
    rc = cs.main(["--runtime", str(tmp_path / "nope"), "--json"])
    assert rc == 2
    assert capsys.readouterr().err


def test_cli_default_table_lists_all_cells(fabricated_runtime, capsys):
    rc = cs.main(["--runtime", str(fabricated_runtime)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "cell_a" in out
    assert "cell_b" in out
    assert "cell_c" in out


def test_finish_lane_shape_is_one_gpu_short_wall():
    shape = cs.finish_shape()
    assert (shape.gpus, shape.wall_hours, shape.cpus, shape.mem_gb) == (1, 12, 12, 128)


def test_cli_finish_prints_the_finish_shape(capsys):
    assert cs.main(["--finish"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["gpus"] == 1 and payload["wall_hours"] == 12


def test_json_report_carries_the_prediction_input(fabricated_runtime, capsys):
    assert cs.main(["--runtime", str(fabricated_runtime), "--cells", "cell_a", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["cell_a"]["baseline_elapsed_seconds"] == pytest.approx(0.05 * 3600)
