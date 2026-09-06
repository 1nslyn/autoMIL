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
    # e5 = 3600s = 1h -> gate=0.6 (undilated), attempt=2*0.6=1.2,
    # promo=2*0.4=0.8, capacity=4*1*0.8=3.2
    # predicted = 0.6 + 30*1.2/3.2 + 10*0.8/3.2 + 2.0 = 16.35
    predicted = cs.predict_hours(3600.0, 1)
    assert predicted == pytest.approx(16.35, abs=1e-9)


def test_packed_attempts_cost_the_dilated_per_fold_time():
    # The 2026-09-06 CLAM rehearsal cell: e5 = 0.71 h, candidates averaged
    # 52 min, not the 25 min per-fold time.
    assert cs.attempt_hours(0.71 * 3600, cs.DISCOVERY_FOLDS) == pytest.approx(2 * 0.71 * 0.6)
    assert cs.fold_hours(0.71 * 3600, cs.DISCOVERY_FOLDS) == pytest.approx(0.71 * 0.6)


def test_tiny_baselines_pay_the_per_attempt_floor():
    # The TITAN rehearsal cell: e5 = 194 s, attempts still took ~16 min.
    assert cs.attempt_hours(194.0, cs.DISCOVERY_FOLDS) == cs.ATTEMPT_FLOOR_H
    assert cs.fold_hours(194.0, cs.DISCOVERY_FOLDS) == cs.ATTEMPT_FLOOR_H
    # e5 = 180 s: every term sits on the floor -> 0.25 + 30*0.25/3.2 + 10*0.25/3.2 + 2
    assert cs.predict_hours(180.0, 1) == pytest.approx(5.375, abs=1e-9)


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
    e5_seconds = 3.0 * 3600  # the LUAD DTFD / nnMIL class on fir

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


def test_preference_changes_the_shape_for_a_one_hour_cell():
    e5_seconds = 1.0 * 3600  # the LUAD nnMIL survival class on fir
    cheap = cs.choose_shape(e5_seconds)              # default
    fast = cs.choose_shape(e5_seconds, prefer="fast")
    assert (cheap.gpus, cheap.wall_hours) == (1, 24)
    assert (fast.gpus, fast.wall_hours) == (2, 12)
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
    _write_state(runtime, "cell_b", _baseline_state(3.0 * 3600))
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


def _write_baseline_log(runtime, cell_id, cached_folds):
    log = runtime / cell_id / "baseline-execution" / "archive" / "run.log"
    log.parent.mkdir(parents=True)
    # Two frameworks' wordings: CLAM-style and DTFD-style (lower case, prefixed).
    lines = ["[automil] cwd = x"] + [
        (f"    [fold {k}] Already completed, loading from disk" if k % 2 == 0
         else f"    [DTFD fold {k}] already completed, loading from disk")
        for k in range(cached_folds)
    ]
    log.write_text("\n".join(lines) + "\nExperiment complete in 2634s\n")


def test_cached_folds_scale_the_baseline_elapsed_time(tmp_path):
    """A re-run baseline loads finished folds from the cache, so the ledger's
    elapsed total covers only the fresh folds: scale it back to five folds
    (seen on tcga_luad kras hoptimus1 clam: 0.73 h ledger, 3.52 h true)."""
    runtime = tmp_path / "runtime"; runtime.mkdir()
    state = _baseline_state(2634.0)
    state["baseline"]["validation_folds"] = [{"fold_index": k} for k in range(5)]
    _write_state(runtime, "cell_k", state)
    _write_baseline_log(runtime, "cell_k", cached_folds=4)
    report = cs.shape_cells(runtime, ["cell_k"])["cell_k"]
    assert report.cached_folds == 4
    assert report.baseline_elapsed_seconds == pytest.approx(2634.0 * 5)
    assert report.shape is not None
    assert report.shape.predicted_hours == pytest.approx(cs.predict_hours(2634.0 * 5, report.shape.gpus))
    # the --cell --field path shapes from the same corrected input
    assert cs.main(["--runtime", str(runtime), "--cell", "cell_k", "--field", "predicted_hours"]) == 0


def test_baseline_with_no_fresh_fold_is_unshaped(tmp_path):
    runtime = tmp_path / "runtime"; runtime.mkdir()
    state = _baseline_state(10.0)
    state["baseline"]["validation_folds"] = [{"fold_index": k} for k in range(5)]
    _write_state(runtime, "cell_all", state)
    _write_baseline_log(runtime, "cell_all", cached_folds=5)
    report = cs.shape_cells(runtime, ["cell_all"])["cell_all"]
    assert report.shape is None
    assert "cached" in report.reason


def test_operator_supplied_time_shapes_a_baseline_with_no_fresh_fold(tmp_path, capsys):
    """A retry that loaded every fold from cache registers a valid baseline
    with no timing; the operator supplies the five-fold time and the shape
    follows it, recorded as such."""
    runtime = tmp_path / "runtime"; runtime.mkdir()
    state = _baseline_state(10.0)
    state["baseline"]["validation_folds"] = [{"fold_index": k} for k in range(5)]
    _write_state(runtime, "cell_all", state)
    _write_baseline_log(runtime, "cell_all", cached_folds=5)
    report = cs.shape_cells(runtime, ["cell_all"], e5_override=3.5 * 3600)["cell_all"]
    assert report.shape is not None
    assert report.baseline_elapsed_seconds == pytest.approx(3.5 * 3600)
    assert report.baseline_elapsed_source == "operator"
    assert report.cached_folds == 5
    assert cs.main(["--runtime", str(runtime), "--cells", "cell_all", "--json", "--e5-seconds", str(3.5 * 3600)]) == 0
    payload = json.loads(capsys.readouterr().out)["cell_all"]
    assert payload["baseline_elapsed_source"] == "operator" and payload["cached_folds"] == 5
    ledger = cs.shape_cells(tmp_path / "runtime", ["cell_all"])["cell_all"]
    assert ledger.shape is None                     # without the override: refused
    plain = cs.shape_cells(runtime, ["cell_all"], e5_override=None)["cell_all"]
    assert plain.baseline_elapsed_source is None or plain.shape is None


def test_missing_baseline_log_means_no_correction(fabricated_runtime):
    report = cs.shape_cells(fabricated_runtime, ["cell_b"])["cell_b"]
    assert report.cached_folds == 0
    assert report.baseline_elapsed_seconds == pytest.approx(3.0 * 3600)


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
