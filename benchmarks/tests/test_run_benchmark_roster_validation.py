"""L-1: an empty per-framework model roster must fail loudly, not launch nothing.

``generate_all_experiments`` (``pipeline/config.py``) iterates
``for model_type in model_types:`` per requested framework. If that
framework's roster resolves to an empty list -- e.g. a dataset YAML that
never set ``nnmil_models``, which defaults to ``[]`` in
``autobench/config.py`` -- the loop simply doesn't iterate for that
framework. Nothing raises: the run prints its normal banner, "completes",
and exits 0 having launched zero experiments. That is indistinguishable
from "there was nothing to do" and would pass a shell script's `$?` check.

Fix: validate every requested framework's resolved roster is non-empty
before any experiment generation, exactly like the existing encoder/strategy
checks in ``run_benchmark.py::main``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from autobench.pipeline.config import Framework  # noqa: E402


def _empty_rosters():
    import run_benchmark
    return run_benchmark._empty_rosters


class TestEmptyRosterDetection:
    """Unit-level: the pure function that decides which frameworks are broken."""

    def test_empty_nnmil_roster_is_flagged(self):
        result = _empty_rosters()(
            [Framework.NNMIL],
            models=["clam_sb"], dtfd_models=["dtfd_mil"],
            abmil_models=["abmil"], nnmil_models=[],
        )
        assert [fw for fw, _ in result] == [Framework.NNMIL]

    def test_populated_rosters_are_not_flagged(self):
        result = _empty_rosters()(
            [Framework.CLAM, Framework.DTFD, Framework.ABMIL, Framework.NNMIL],
            models=["clam_sb"], dtfd_models=["dtfd_mil"],
            abmil_models=["abmil"], nnmil_models=["trans_mil"],
        )
        assert result == []

    def test_titan_has_no_roster_axis(self):
        """TITAN has no model-type axis (generate_all_experiments pins the
        "titan" pseudo-model unconditionally) -- never flagged even though
        every roster list here is empty."""
        result = _empty_rosters()(
            [Framework.TITAN],
            models=[], dtfd_models=[], abmil_models=[], nnmil_models=[],
        )
        assert result == []

    def test_unrequested_frameworks_are_ignored(self):
        """An empty DTFD roster is not a problem if DTFD was never requested."""
        result = _empty_rosters()(
            [Framework.CLAM],
            models=["clam_sb"], dtfd_models=[], abmil_models=[], nnmil_models=[],
        )
        assert result == []

    def test_multiple_empty_rosters_all_reported(self):
        result = _empty_rosters()(
            [Framework.DTFD, Framework.ABMIL],
            models=["clam_sb"], dtfd_models=[], abmil_models=[], nnmil_models=[],
        )
        assert {fw for fw, _ in result} == {Framework.DTFD, Framework.ABMIL}


class TestMainExitsOnEmptyRoster:
    """End-to-end: main() must stop before generating (and silently running) an
    empty grid."""

    def _fake_dataset_config(self, **overrides):
        from autobench.config import DatasetConfig, StrategyDef

        kwargs = dict(
            name="fake", description="fake dataset for the roster-validation test",
            data_root="/x", wsi_dir="/x", mapping_csv="/x/map.csv",
            output_dir="/x", benchmark_dir="/x", features_base_dir="/x",
            tasks={},
            split_strategies={
                "standard": StrategyDef(name="standard", train_cohorts=[], test_cohorts=[]),
            },
            task_strategy_feasibility={},
            slide_id_column="slide_id", slide_id_transform=None, wsi_extension=None,
            case_id_column="case_id", status_column=None, status_value=None,
            encoder_models={}, encoder_dims={"conch_v15": 768},
        )
        kwargs.update(overrides)
        return DatasetConfig(**kwargs)

    def test_main_exits_nonzero_when_requested_roster_is_empty(self, monkeypatch, capsys):
        import run_benchmark

        ds = self._fake_dataset_config(nnmil_models=[])
        monkeypatch.setattr(run_benchmark, "load_dataset_config", lambda name: ds)
        monkeypatch.setattr(
            sys, "argv",
            ["run_benchmark.py", "--dataset", "fake", "--frameworks", "nnmil"],
        )

        with pytest.raises(SystemExit) as exc_info:
            run_benchmark.main()

        assert exc_info.value.code == 1
        assert "nnmil" in capsys.readouterr().out

    def test_main_proceeds_past_roster_check_when_populated(self, monkeypatch):
        """A populated roster must not be rejected by the new check (it should
        fail later, deeper in main(), for the usual "no real GPU/data" reasons
        -- never on the roster-emptiness check itself)."""
        import run_benchmark

        ds = self._fake_dataset_config(nnmil_models=["trans_mil"])
        monkeypatch.setattr(run_benchmark, "load_dataset_config", lambda name: ds)
        monkeypatch.setattr(
            sys, "argv",
            ["run_benchmark.py", "--dataset", "fake", "--frameworks", "nnmil",
             "--prep_only"],
        )

        # prep_only drives straight into prepare_all, which will fail on the
        # fake paths -- proof the roster check let it through rather than
        # exiting(1) itself.
        with pytest.raises(Exception) as exc_info:
            run_benchmark.main()
        assert not (
            isinstance(exc_info.value, SystemExit) and exc_info.value.code == 1
        )
