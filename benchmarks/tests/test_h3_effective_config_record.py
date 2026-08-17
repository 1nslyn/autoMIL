"""H-3: ``config.json`` must record the recipe that ACTUALLY ran.

The defect, measured on the finished phase-2 campaign: all 195 ``config.json``
files carried the shared ``TrainConfig`` block, but only CLAM and ABMIL trained
off it. **102 of the 195 described a recipe that never ran** -- a methods table
built from that artifact would state DTFD's lr as 2e-4 when it was 1e-4, nnMIL's
epochs as 200 when they were 100, TITAN's lr as 2e-4 when it was 1e-3.

The fingerprint sidecar was already correct (every non-CLAM runner passes
``arm_cfg`` to ``resolve_results_dir``); the *human-facing* record was not.
"""

from __future__ import annotations

import json

from autobench.pipeline.abmil.config import ABMILConfig
from autobench.pipeline.config import (
    ExperimentConfig,
    Framework,
    ModelConfig,
    TaskConfig,
    TrainConfig,
)
from autobench.pipeline.dtfd.config import DTFDConfig
from autobench.pipeline.results_cache import fingerprint_payload
from autobench.pipeline.titan.config import TitanHeadConfig


def _exp(framework: Framework = Framework.CLAM) -> ExperimentConfig:
    return ExperimentConfig(
        task=TaskConfig(name="kras", label_col="KRAS", label_dict={"WT": 0, "MUT": 1}),
        encoder_key="uni_v2",
        embed_dim=1536,
        model=ModelConfig(model_type="clam_mb"),
        train=TrainConfig(),
        framework=framework,
    )


def _nnmil_plan() -> dict:
    """nnMIL's self-configured plan -- a mapping, not a dataclass."""
    return {
        "feature_dimension": 1536, "hidden_dim": 384, "max_seq_length": 2048,
        "batch_size": 32, "batch_sampler": "auc", "learning_rate": 3e-4,
        "weight_decay": 1e-4, "num_epochs": 100, "warmup_epochs": 10,
        "dropout": 0.25, "patience": 10, "num_classes": 2,
    }


def _saved(tmp_path, arm_cfg=None) -> dict:
    path = tmp_path / "results" / "config.json"
    _exp().save(str(path), arm_cfg=arm_cfg)
    return json.loads(path.read_text())


class TestArmBlockIsRecorded:
    def test_clam_records_null_arm_explicitly(self, tmp_path):
        """Absence must read as a FACT, not as an omission -- CLAM is the one arm
        whose `train` block genuinely governed."""
        saved = _saved(tmp_path)
        assert "arm" in saved and saved["arm"] is None
        assert saved["train_fields_superseded_by_arm"] == []

    def test_dtfd_records_its_own_optimizer(self, tmp_path):
        saved = _saved(tmp_path, DTFDConfig())
        assert saved["arm"]["lr"] == 1e-4
        assert saved["arm"]["wd"] == 1e-4
        assert saved["arm"]["numGroup"] == 4  # a knob with no `train` counterpart

    def test_abmil_records_its_own_schedule(self, tmp_path):
        saved = _saved(tmp_path, ABMILConfig())
        assert saved["arm"]["lr"] == 5e-4
        assert saved["arm"]["max_epochs"] == 20

    def test_nnmil_plan_mapping_is_accepted(self, tmp_path):
        saved = _saved(tmp_path, _nnmil_plan())
        assert saved["arm"]["learning_rate"] == 3e-4
        assert saved["arm"]["num_epochs"] == 100


class TestSupersededFieldsResolveAliases:
    """The list is what disambiguates: `arm` and the stale `train` sit side by
    side, and a reader cannot otherwise tell which one won."""

    def test_dtfd_wd_alias_resolves_to_weight_decay(self, tmp_path):
        superseded = _saved(tmp_path, DTFDConfig())["train_fields_superseded_by_arm"]
        assert superseded == [
            "early_stopping", "lr", "max_epochs", "patience", "weight_decay",
        ]

    def test_nnmil_learning_rate_and_num_epochs_aliases_resolve(self, tmp_path):
        superseded = _saved(tmp_path, _nnmil_plan())["train_fields_superseded_by_arm"]
        assert superseded == ["lr", "max_epochs", "patience", "weight_decay"]
        # nnMIL's plan has no early_stopping key -> TrainConfig's still governs.
        assert "early_stopping" not in superseded

    def test_titan_is_recorded_as_genuinely_mixed(self, tmp_path):
        """TITAN reads lr/weight_decay/patience off its head config but
        max_epochs off the SHARED block -- the one arm where neither block is
        wholly authoritative, so the artifact has to say so per field."""
        superseded = _saved(tmp_path, TitanHeadConfig())["train_fields_superseded_by_arm"]
        assert superseded == ["lr", "patience", "weight_decay"]
        assert "max_epochs" not in superseded


class TestTheDefectItself:
    """Reproduces what made 102 configs fiction, and pins the repair: the stale
    value is still present (it is a real field of the transport), but the
    artifact now names it as superseded and carries the true one alongside."""

    def test_reading_train_weight_decay_for_dtfd_gives_the_wrong_number(self, tmp_path):
        saved = _saved(tmp_path, DTFDConfig())
        assert saved["train"]["weight_decay"] == 1e-5   # what the old record said
        assert saved["arm"]["wd"] == 1e-4               # what actually trained
        assert "weight_decay" in saved["train_fields_superseded_by_arm"]

    def test_reading_train_max_epochs_for_abmil_gives_the_wrong_number(self, tmp_path):
        """The largest gap in the roster: 200 recorded, 20 trained."""
        saved = _saved(tmp_path, ABMILConfig())
        assert saved["train"]["max_epochs"] == 200
        assert saved["arm"]["max_epochs"] == 20
        assert "max_epochs" in saved["train_fields_superseded_by_arm"]

    def test_dtfd_lr_agreeing_with_the_shared_block_is_a_coincidence(self, tmp_path):
        """A field-by-field eyeball is NOT a substitute for the recorded list.

        Since CLAM's lr was returned to its upstream 1e-4 (provenance.py,
        2026-07-28) it happens to equal DTFD's 1e-4 -- so a reader spot-checking
        `train.lr` against `arm.lr` sees agreement and concludes the whole
        `train` block is trustworthy for this run. It is not: weight_decay is off
        by 10x. Pinned so that a future change to either default surfaces here
        rather than silently restoring the trap.
        """
        saved = _saved(tmp_path, DTFDConfig())
        assert saved["train"]["lr"] == saved["arm"]["lr"]
        assert "lr" in saved["train_fields_superseded_by_arm"]
        assert saved["train"]["weight_decay"] != saved["arm"]["wd"]


class TestTitanArchivedProvenance:
    """The archived config.json must carry the EFFECTIVE train values.

    The defect (F-K11): the opaque channel's train-side slice used to be
    applied inside the TITAN trainers (via resolve_head_config) — AFTER the
    runner had resolved the results dir and saved config.json — so a run
    overridden to max_epochs=7 archived the default 200. The seam is now
    ``apply_train_overrides`` at the runner level, contract-matched to
    ``hparams.apply_overrides_to_exp_cfg``: run before results-dir
    resolution and ``exp_cfg.save``.
    """

    def test_archived_config_carries_the_overridden_max_epochs(self, tmp_path):
        from autobench.pipeline.titan.config import (
            TitanHeadConfig,
            apply_train_overrides,
        )

        exp = _exp(framework=Framework.TITAN)
        exp.hparam_overrides = {"max_epochs": 7}

        # The runner's real order: apply, then save.
        apply_train_overrides(exp)
        path = tmp_path / "results" / "config.json"
        exp.save(str(path), arm_cfg=TitanHeadConfig())

        saved = json.loads(path.read_text())
        assert saved["train"]["max_epochs"] == 7
        # Still the arm's documented mixed provenance: the SHARED block
        # governs max_epochs, so it must not be listed as superseded.
        assert "max_epochs" not in saved["train_fields_superseded_by_arm"]


class TestFingerprintIsNotDisturbed:
    def test_to_dict_does_not_gain_the_provenance_fields(self):
        """Load-bearing. ``fingerprint_payload`` is built on ``to_dict``; a new
        key there would change every stored digest, so every results directory
        written before this change would raise ``StaleResultsCacheError`` on
        resume and the whole finished campaign would refuse to be re-read."""
        d = _exp().to_dict()
        assert "arm" not in d
        assert "train_fields_superseded_by_arm" not in d

    def test_digest_payload_is_unchanged_for_a_clam_run(self):
        payload = fingerprint_payload(_exp())
        assert "train_fields_superseded_by_arm" not in payload
        assert "arm" not in payload  # only added when an arm_cfg is supplied
