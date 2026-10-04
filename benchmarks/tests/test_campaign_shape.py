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
    # e5 = 1 h -> gate 0.6 (undilated), slowest attempt 2*0.6 = 1.2, four
    # batches 4.8, one batch at the 10 h timeout adds 8.8, promotion two
    # rounds of 2*0.4 = 1.6, overhead 2.0
    assert cs.predict_hours(3600.0, 1) == pytest.approx(17.8, abs=1e-9)


def test_predict_hours_known_value_three_gpus():
    # The workstation's three GPUs run promotion in one round: 1.6 -> 0.8.
    assert cs.predict_hours(3600.0, 3) == pytest.approx(17.0, abs=1e-9)


def test_more_gpus_shorten_only_promotion():
    """A batch of 8 fits one GPU, so a second GPU saves one promotion round
    and a third saves nothing."""
    e5_seconds = 2.5 * 3600
    one, two, three = (cs.predict_hours(e5_seconds, gpus) for gpus in (1, 2, 3))
    assert one - two == pytest.approx(cs.attempt_hours(e5_seconds, cs.PROMOTION_FOLDS))
    assert two == pytest.approx(three)
    assert cs.discovery_hours(e5_seconds, 1) == pytest.approx(cs.discovery_hours(e5_seconds, 3))


def test_the_slowest_attempt_costs_the_dilated_per_fold_time():
    assert cs.attempt_hours(0.71 * 3600, cs.DISCOVERY_FOLDS) == pytest.approx(2 * 0.71 * 0.6)
    assert cs.fold_hours(0.71 * 3600, cs.DISCOVERY_FOLDS) == pytest.approx(0.71 * 0.6)


def test_tiny_baselines_pay_the_per_attempt_floor():
    # The TITAN rehearsal cell: e5 = 194 s, attempts still took ~16 min.
    assert cs.attempt_hours(194.0, cs.DISCOVERY_FOLDS) == cs.ATTEMPT_FLOOR_H
    assert cs.fold_hours(194.0, cs.DISCOVERY_FOLDS) == cs.ATTEMPT_FLOOR_H
    # e5 = 180 s: 0.25 gate + 4*0.25 + (10 - 0.25) + 2*0.25 + 2
    assert cs.predict_hours(180.0, 1) == pytest.approx(13.5, abs=1e-9)


def test_an_attempt_as_long_as_the_timeout_adds_no_timeout_batch():
    e5_seconds = 9.0 * 3600  # slowest attempt 10.8 h, past the 10 h timeout
    attempt = cs.attempt_hours(e5_seconds, cs.DISCOVERY_FOLDS)
    assert attempt > cs.ATTEMPT_TIMEOUT_H
    assert cs.discovery_hours(e5_seconds, 1) == pytest.approx(
        cs.fold_hours(e5_seconds, cs.DISCOVERY_FOLDS) + len(cs.DISCOVERY_BATCHES) * attempt
    )


def test_the_copied_constants_match_the_frozen_protocol():
    from autobench import campaign

    assert list(cs.DISCOVERY_BATCHES) == campaign.DISCOVERY_PHASING["batches"]
    assert cs.PROMOTION_CANDIDATES == campaign.PROMOTION_CANDIDATES
    assert cs.DISCOVERY_FOLDS == len(campaign.STAGE_FOLDS["discovery"])
    assert cs.PROMOTION_FOLDS == len(campaign.STAGE_FOLDS["promotion"])
    assert cs.TOTAL_FOLDS == len(campaign.BASELINE_FOLDS)
    assert cs.ATTEMPT_TIMEOUT_H * 60 == campaign.ATTEMPT_TIMEOUT_MIN


# Measured on the 2026-10 trial cells (aihub: three RTX 6000 Ada; fir: two
# H100): each cell's time from the job's start to the discovery freeze, with
# its registered baseline time. A session that has not finished discovery by the wall less the
# finish reserve is cut and the cell is stranded, so this is the time the
# prediction must hold. DTFD's third batch ran into the 10 h attempt
# timeout; its anchor fails for any dilation below 1.72.
TRIAL_DISCOVERY = (
    # (cell, e5 seconds, GPUs, hours from job start to discovery frozen)
    ("aihub kras hoptimus1 dtfd", 10496.3, 3, 22.795),
    ("aihub kras hoptimus1 abmil", 5500.9, 3, 8.918),
    ("aihub kras hoptimus1 clam", 16573.9, 3, 16.135),
    ("fir kras hoptimus1 abmil", 10672.6, 2, 11.049),
)


@pytest.mark.parametrize("cell, e5_seconds, gpus, measured", TRIAL_DISCOVERY,
                         ids=[row[0] for row in TRIAL_DISCOVERY])
def test_the_prediction_holds_each_trial_cells_discovery(cell, e5_seconds, gpus, measured):
    assert cs.discovery_hours(e5_seconds, gpus) + cs.OVERHEAD_H >= measured


# ---------------------------------------------------------------------------
# choose_shape


def test_every_shape_is_one_gpu_with_a_quarter_node():
    shape = cs.choose_shape(0.05 * 3600)
    assert (shape.gpus, shape.wall_hours, shape.cpus, shape.mem_gb) == (1, 24, 12, 128)
    assert shape.predicted_hours == pytest.approx(cs.predict_hours(0.05 * 3600, 1))


def test_the_wall_moves_to_72_hours_past_the_24_hour_fit():
    # 1.4 h predicts 20.12 h (fits 0.85 * 24 = 20.4); 1.5 h predicts 20.7 h.
    assert cs.choose_shape(1.4 * 3600).wall_hours == 24
    slow = cs.choose_shape(1.5 * 3600)
    assert (slow.gpus, slow.wall_hours) == (1, 72)
    assert slow.predicted_hours == pytest.approx(20.7)


def test_choose_shape_returns_none_when_nothing_fits():
    e5_seconds = 1000 * 3600  # absurdly slow baseline: no wall holds it
    assert cs.choose_shape(e5_seconds) is None


def test_the_finish_lane_holds_every_cell_that_fits_a_discovery_shape():
    """The finish-only lane needs no baseline time: it holds the promotion of
    the slowest baseline that still fits a discovery shape."""
    lo, hi = 0.0, 1000 * 3600.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cs.choose_shape(mid) else (lo, mid)
    lane = cs.finish_shape()
    assert (lane.gpus, lane.wall_hours) == (1, 24)
    assert cs.promotion_hours(lo, lane.gpus) + cs.OVERHEAD_H <= cs.FIT_FRACTION * lane.wall_hours


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
    assert reports["cell_a"].shape.wall_hours == 24
    assert reports["cell_b"].shape.wall_hours == 72


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
    assert payload["cell_a"]["wall_hours"] == 24
    assert payload["cell_b"]["gpus"] == 1
    assert payload["cell_b"]["wall_hours"] == 72
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


def test_cli_field_prints_every_shape_field_and_nothing_else(fabricated_runtime, capsys):
    from dataclasses import fields

    for field in fields(cs.Shape):
        assert cs.main(["--runtime", str(fabricated_runtime), "--cell", "cell_a", "--field", field.name]) == 0
    with pytest.raises(SystemExit):
        cs.main(["--runtime", str(fabricated_runtime), "--cell", "cell_a", "--field", "whole_node"])


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


def test_cli_finish_prints_the_finish_shape(capsys):
    assert cs.main(["--finish"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert (payload["gpus"], payload["wall_hours"], payload["cpus"], payload["mem_gb"]) == (1, 24, 12, 128)
    assert "whole_node" not in payload


def test_json_report_carries_the_prediction_input(fabricated_runtime, capsys):
    assert cs.main(["--runtime", str(fabricated_runtime), "--cells", "cell_a", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["cell_a"]["baseline_elapsed_seconds"] == pytest.approx(0.05 * 3600)
