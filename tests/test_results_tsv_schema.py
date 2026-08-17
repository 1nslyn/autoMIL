"""TSV-1: results.tsv locked its metric columns from the first row and dropped the rest.

``_append_results_tsv`` derived the header from the first result's ``metrics``
keys — deliberately, to avoid hardcoding MIL vocabulary — and then aligned every
later row to that header, filling missing keys with blanks. A key the header did
not already have was **silently dropped**.

The preprint campaign is exactly the case that breaks: 65 classification
experiments emit ``val_auc`` / ``val_bacc`` and 100 survival experiments emit
``val_c_index``. Whichever finishes first defines the header, so one of those
two groups loses its only metric — every survival row reading blank, with no
error anywhere. ``primary_value`` still lands, so the file looks populated.

The fix keeps the no-hardcoded-vocabulary property: on meeting a genuinely new
metric key, widen the header and rewrite the file, backfilling earlier rows with
blanks (they really had no value for that column).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def orch(tmp_path, monkeypatch):
    """A minimally-constructed orchestrator with results_tsv pointed at tmp."""
    from automil.backends._orchestrator_daemon import ExperimentOrchestrator

    o = object.__new__(ExperimentOrchestrator)
    o.results_tsv = tmp_path / "results.tsv"
    return o


def _rows(path: Path) -> list[list[str]]:
    return [ln.split("\t") for ln in path.read_text().strip().splitlines()]


def _cls(primary_value=0.8):
    return {"metrics": {"val_auc": 0.81, "val_bacc": 0.79}, "primary_value": primary_value,
            "status": "completed", "elapsed_seconds": 60, "peak_vram_mb": 1024}


def _surv(primary_value=0.66):
    return {"metrics": {"val_c_index": 0.66}, "primary_value": primary_value,
            "status": "completed", "elapsed_seconds": 60, "peak_vram_mb": 1024}


class TestMixedCampaignKeepsEveryMetric:
    def test_survival_after_classification_is_not_dropped(self, orch):
        orch._append_results_tsv("0001", _cls(), "clam kras")
        orch._append_results_tsv("0002", _surv(), "clam os")

        header, *rows = _rows(orch.results_tsv)
        assert "val_c_index" in header, (
            "the survival metric was dropped because the header was locked by "
            "the first (classification) row"
        )
        col = header.index("val_c_index")
        assert rows[1][col] == "0.6600"

    def test_classification_after_survival_is_not_dropped(self, orch):
        orch._append_results_tsv("0001", _surv(), "clam os")
        orch._append_results_tsv("0002", _cls(), "clam kras")

        header, *rows = _rows(orch.results_tsv)
        assert {"val_auc", "val_bacc"} <= set(header)
        assert rows[1][header.index("val_auc")] == "0.8100"

    def test_earlier_rows_are_backfilled_not_corrupted(self, orch):
        orch._append_results_tsv("0001", _cls(), "a")
        orch._append_results_tsv("0002", _surv(), "b")

        header, r1, r2 = _rows(orch.results_tsv)
        assert len(r1) == len(header) == len(r2)
        # Row 1 genuinely had no c-index: blank, not zero.
        assert r1[header.index("val_c_index")] == ""
        assert r2[header.index("val_auc")] == ""

    def test_every_row_keeps_its_own_primary_value(self, orch):
        orch._append_results_tsv("0001", _cls(primary_value=0.80), "a")
        orch._append_results_tsv("0002", _surv(primary_value=0.66), "b")

        header, r1, r2 = _rows(orch.results_tsv)
        c = header.index("primary_value")
        assert (r1[c], r2[c]) == ("0.800000", "0.660000")

    def test_the_trailing_columns_stay_at_the_end(self, orch):
        """Widening must not shuffle primary_value/status/description into the
        middle — anything parsing by position would silently misread."""
        orch._append_results_tsv("0001", _cls(), "a")
        orch._append_results_tsv("0002", _surv(), "b")
        header = _rows(orch.results_tsv)[0]
        assert header[0] == "node_id"
        assert header[-6:] == ["primary_value", "primary_se", "vram_gb",
                               "elapsed_min", "status", "description"]

    def test_node_ids_and_descriptions_survive_the_rewrite(self, orch):
        orch._append_results_tsv("0001", _cls(), "first idea")
        orch._append_results_tsv("0002", _surv(), "second idea")
        header, r1, r2 = _rows(orch.results_tsv)
        assert (r1[0], r2[0]) == ("0001", "0002")
        d = header.index("description")
        assert (r1[d], r2[d]) == ("first idea", "second idea")


class TestNoNeedlessRewrites:
    def test_a_matching_schema_appends_without_rewriting(self, orch, monkeypatch):
        orch._append_results_tsv("0001", _cls(), "a")
        before = orch.results_tsv.read_text()
        orch._append_results_tsv("0002", _cls(), "b")
        after = orch.results_tsv.read_text()
        assert after.startswith(before), "an append must not rewrite existing bytes"

    def test_a_subset_of_known_metrics_does_not_widen(self, orch):
        orch._append_results_tsv("0001", _cls(), "a")
        orch._append_results_tsv("0002", {"metrics": {"val_auc": 0.5}, "primary_value": 0.5,
                                          "status": "completed"}, "b")
        header = _rows(orch.results_tsv)[0]
        assert header.count("val_auc") == 1
        assert "val_bacc" in header


class TestPrimary_valueSeColumn:
    def test_primary_se_lands_in_its_own_column(self, orch):
        orch._append_results_tsv(
            "0001", dict(_cls(), primary_se=0.0116), "a",
        )
        header, row = _rows(orch.results_tsv)
        assert row[header.index("primary_se")] == "0.011600"

    def test_missing_primary_value_se_is_blank_not_zero(self, orch):
        orch._append_results_tsv("0001", _cls(), "a")
        header, row = _rows(orch.results_tsv)
        assert row[header.index("primary_se")] == ""


class TestTrailingSchemaWidening:
    """A results.tsv written before ``primary_se`` joined the trailing block
    must be rewritten — with every old cell mapped by the header actually on
    disk. Zipping old rows against a reconstruction from the CURRENT trailing
    tuple shifted every backfilled trailing cell one column left (the exact
    silent corruption this class pins)."""

    LEGACY_HEADER = "node_id\tval_auc\tval_bacc\tprimary_value\tvram_gb\telapsed_min\tstatus\tdescription"
    LEGACY_ROW = "node_0007\t0.6046\t0.6317\t0.618150\t1.4\t286.3\tcompleted\twarmup 2 + patience 40"

    def _seed_legacy(self, orch):
        orch.results_tsv.write_text(self.LEGACY_HEADER + "\n" + self.LEGACY_ROW + "\n")

    def test_legacy_file_widens_on_next_append(self, orch):
        self._seed_legacy(orch)
        orch._append_results_tsv("0002", dict(_cls(), primary_se=0.02), "b")
        header, r1, r2 = _rows(orch.results_tsv)
        assert "primary_se" in header
        assert len(r1) == len(header) == len(r2)

    def test_legacy_row_cells_keep_their_columns(self, orch):
        self._seed_legacy(orch)
        orch._append_results_tsv("0002", dict(_cls(), primary_se=0.02), "b")
        header, r1, _ = _rows(orch.results_tsv)
        by = dict(zip(header, r1))
        assert by["primary_value"] == "0.618150"
        assert by["primary_se"] == ""          # legacy row genuinely had none
        assert by["vram_gb"] == "1.4"
        assert by["elapsed_min"] == "286.3"
        assert by["status"] == "completed"
        assert by["description"] == "warmup 2 + patience 40"

    def test_legacy_rewrite_happens_even_without_new_metric_keys(self, orch):
        """The old trigger fired only on new METRIC keys; a trailing-schema
        change alone must also rewrite, or the append writes rows wider than
        the on-disk header."""
        self._seed_legacy(orch)
        orch._append_results_tsv("0002", _cls(), "b")     # same metric keys
        header, r1, r2 = _rows(orch.results_tsv)
        assert "primary_se" in header
        assert len(r1) == len(header) == len(r2)


class TestDegenerateInputs:
    def test_a_result_with_no_metrics_still_writes_a_row(self, orch):
        orch._append_results_tsv("0001", {"primary_value": 0.0, "status": "crash"}, "boom")
        header, row = _rows(orch.results_tsv)
        assert row[0] == "0001"
        assert row[header.index("status")] == "crash"

    def test_a_crash_row_before_any_metric_row_does_not_lock_an_empty_header(self, orch):
        """A crash arriving first used to fix the header at zero metric columns,
        so the whole campaign's metrics vanished."""
        orch._append_results_tsv("0001", {"primary_value": 0.0, "status": "crash"}, "boom")
        orch._append_results_tsv("0002", _cls(), "a")
        header = _rows(orch.results_tsv)[0]
        assert {"val_auc", "val_bacc"} <= set(header)

    def test_a_metric_name_containing_a_tab_is_rejected(self, orch):
        """A tab in a key would silently shift every column right of it."""
        with pytest.raises(ValueError, match="tab"):
            orch._append_results_tsv(
                "0001", {"metrics": {"bad\tkey": 1.0}, "primary_value": 0.5}, "a",
            )

    def test_a_description_with_a_newline_is_flattened(self, orch):
        orch._append_results_tsv(
            "0001", _cls(), "line one\nline two",
        )
        assert len(orch.results_tsv.read_text().strip().splitlines()) == 2


class TestLegacyCompositeTrailingColumns:
    """A results.tsv written before the composite retirement carries
    `composite`/`composite_se` in its trailing block. Those names are no
    longer in _TSV_TRAILING, so without the legacy map they re-enter the
    header as METRIC columns — the agent-facing file keeps a phantom metric
    named after the retired concept forever, and its cells never migrate."""

    def _legacy_file(self, orch):
        orch.results_tsv.write_text(
            "node_id\tval_auc\tval_bacc\tcomposite\tcomposite_se\tvram_gb"
            "\telapsed_min\tstatus\tdescription\n"
            "0001\t0.85\t0.80\t0.825000\t0.020000\t4.5\t1.0\tcompleted\told row\n"
        )

    def test_legacy_columns_do_not_become_metrics(self, orch):
        self._legacy_file(orch)
        orch._append_results_tsv("0002", _cls(), "new row")
        header, *_ = _rows(orch.results_tsv)
        assert "composite" not in header
        assert "composite_se" not in header
        assert header.index("primary_value") > header.index("val_bacc"), (
            "primary_value must stay in the trailing block"
        )

    def test_legacy_cells_migrate_into_the_renamed_columns(self, orch):
        self._legacy_file(orch)
        orch._append_results_tsv("0002", _cls(primary_value=0.9), "new row")
        header, *rows = _rows(orch.results_tsv)
        by_id = {r[0]: dict(zip(header, r)) for r in rows}
        assert by_id["0001"]["primary_value"] == "0.825000"
        assert by_id["0001"]["primary_se"] == "0.020000"
        assert by_id["0002"]["primary_value"] == "0.900000"
