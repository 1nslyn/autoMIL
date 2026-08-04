"""Fail-closed candidate classification for architecture-preserving search."""
from __future__ import annotations

import pytest
import yaml

from automil.admissibility import (
    AdmissibilityError,
    CandidateClass,
    CandidatePolicy,
    CandidateVerdict,
    load_candidate_policy,
    revalidate_candidate_spec,
)


@pytest.fixture
def recipe_policy() -> CandidatePolicy:
    return CandidatePolicy(
        mode="architecture-preserving",
        editable=("recipes/**",),
        protected=("models/**", "evaluate.py"),
        allowed_override_options=("--hparams",),
        allowed_variant_kinds=("policy",),
        identity_locked_hparams=("no_inst_cluster", "bag_weight"),
    )


class TestCandidateClassification:
    def test_train_only_source_is_accepted(self, recipe_policy):
        verdict = recipe_policy.classify(["recipes/cosine.py"])
        assert verdict.accepted is True
        assert verdict.candidate_class is CandidateClass.TRAIN_ONLY_SOURCE
        assert verdict.files == ("recipes/cosine.py",)

    def test_explicit_file_outside_editable_is_rejected(self, recipe_policy):
        verdict = recipe_policy.classify(["train.py"])
        assert verdict.accepted is False
        assert verdict.candidate_class is CandidateClass.PROTECTED_SURFACE_VIOLATION
        assert "files.editable" in verdict.reason

    def test_protected_path_names_the_matching_rule(self, recipe_policy):
        verdict = recipe_policy.classify(["models/clam.py"])
        assert verdict.accepted is False
        assert verdict.matched_patterns == ("models/**",)
        assert "registry.protected" in verdict.reason

    @pytest.mark.parametrize("path", ["/etc/passwd", "../model.py", "x/../../model.py", ""])
    def test_invalid_paths_are_not_surface_violations(self, recipe_policy, path):
        verdict = recipe_policy.classify([path])
        assert verdict.accepted is False
        assert verdict.candidate_class is CandidateClass.INVALID

    def test_hparam_only_override_is_config_only(self, recipe_policy):
        verdict = recipe_policy.classify(
            [], override="--hparams '{\"lr\":0.0001}'",
        )
        assert verdict.accepted is True
        assert verdict.candidate_class is CandidateClass.CONFIG_ONLY

    def test_hparam_hash_canonicalizes_json_whitespace_and_key_order(
        self, recipe_policy,
    ):
        first = recipe_policy.classify(
            [], override='--hparams \'{"lr":0.001,"wd":0.01}\'',
        )
        second = recipe_policy.classify(
            [], override='--hparams \'{ "wd": 0.01, "lr": 0.001 }\'',
        )
        changed = recipe_policy.classify(
            [], override='--hparams \'{"lr":0.002,"wd":0.01}\'',
        )

        assert first.override_hash == second.override_hash
        assert first.override_hash != changed.override_hash

    def test_override_hash_canonicalizes_option_spelling_and_order(self):
        policy = CandidatePolicy(
            mode="architecture-preserving",
            allowed_override_options=("--hparams", "--policy-variant"),
        )
        first = policy.classify(
            [],
            override='--hparams \'{"lr":0.001,"wd":0.01}\' '
            '--policy-variant cosine',
        )
        second = policy.classify(
            [],
            override='--policy-variant=cosine '
            '--hparams=\'{ "wd": 0.01, "lr": 0.001 }\'',
        )
        assert first.accepted and second.accepted
        assert first.override_hash == second.override_hash

    @pytest.mark.parametrize("key", ["no_inst_cluster", "bag_weight"])
    def test_identity_erasing_hparam_is_rejected(self, recipe_policy, key):
        verdict = recipe_policy.classify(
            [], override=f"--hparams '{{\"{key}\":true}}'",
        )
        assert verdict.accepted is False
        assert verdict.candidate_class is CandidateClass.PROTECTED_SURFACE_VIOLATION
        assert "identity_locked_hparams" in verdict.reason

    @pytest.mark.parametrize(
        "override",
        ["--hparams {lr:0.001}", "--hparams '[1,2]'", "--hparams"],
    )
    def test_hparams_must_be_one_quoted_flat_json_object(
        self, recipe_policy, override,
    ):
        verdict = recipe_policy.classify([], override=override)
        assert verdict.accepted is False
        assert verdict.candidate_class is CandidateClass.INVALID

    @pytest.mark.parametrize(
        "override",
        ["--dataset other", "--model abmil", "--seed 7", "--hparams {} --task os"],
    )
    def test_undeclared_override_option_is_rejected(self, recipe_policy, override):
        verdict = recipe_policy.classify([], override=override)
        assert verdict.accepted is False
        assert verdict.candidate_class is CandidateClass.PROTECTED_SURFACE_VIOLATION
        assert "allowed_override_options" in verdict.reason

    def test_malformed_override_is_invalid(self, recipe_policy):
        verdict = recipe_policy.classify([], override='--hparams "unterminated')
        assert verdict.accepted is False
        assert verdict.candidate_class is CandidateClass.INVALID

    @pytest.mark.parametrize(
        "override",
        ["extra --hparams '{}'", "--hparams '{}' trailing", "--hparams '{}' --hparams '{}'"],
    )
    def test_override_rejects_positional_tokens_and_duplicate_options(
        self, recipe_policy, override,
    ):
        verdict = recipe_policy.classify([], override=override)
        assert verdict.accepted is False
        assert verdict.candidate_class is CandidateClass.INVALID

    def test_no_change_is_invalid(self, recipe_policy):
        verdict = recipe_policy.classify([])
        assert verdict.accepted is False
        assert verdict.candidate_class is CandidateClass.INVALID

    def test_model_variant_is_a_protected_surface_violation(self, recipe_policy):
        verdict = recipe_policy.classify(
            [],
            variant_selection={"model": {"variant": "clam_as_abmil"}},
        )
        assert verdict.accepted is False
        assert verdict.candidate_class is CandidateClass.PROTECTED_SURFACE_VIOLATION
        assert "allowed_variant_kinds" in verdict.reason

    def test_policy_variant_is_part_of_the_train_only_candidate(self, recipe_policy):
        verdict = recipe_policy.classify(
            [],
            variant_selection={"policy": {"variant": "cosine_restart"}},
        )
        assert verdict.accepted is True
        assert verdict.candidate_class is CandidateClass.TRAIN_ONLY_SOURCE
        assert verdict.variant_kinds == ("policy",)
        assert verdict.variant_selection_hash

    def test_free_mode_keeps_generic_source_search(self):
        policy = CandidatePolicy(mode="free", protected=("evaluate.py",))
        verdict = policy.classify(["models/new_pool.py"])
        assert verdict.accepted is True
        assert verdict.candidate_class is CandidateClass.FREE_SOURCE


class TestRecordedVerdictRevalidation:
    def test_verdict_round_trips_as_json_data(self, recipe_policy):
        original = recipe_policy.classify(["recipes/cosine.py"])
        restored = CandidateVerdict.from_dict(original.to_dict())
        assert restored == original

    def test_same_policy_and_candidate_revalidate(self, recipe_policy):
        original = recipe_policy.classify(["recipes/cosine.py"])
        assert recipe_policy.revalidate(
            ["recipes/cosine.py"], recorded=original.to_dict(),
        ) == original

    def test_policy_change_between_submit_and_launch_fails_closed(self, recipe_policy):
        original = recipe_policy.classify(["recipes/cosine.py"])
        changed = CandidatePolicy(
            mode="architecture-preserving",
            editable=("recipes/safe/**",),
            protected=("models/**",),
            allowed_override_options=("--hparams",),
            allowed_variant_kinds=("policy",),
        )
        with pytest.raises(AdmissibilityError, match="policy changed"):
            changed.revalidate(
                ["recipes/cosine.py"], recorded=original.to_dict(),
            )

    def test_missing_record_fails_closed_in_architecture_preserving_mode(self, recipe_policy):
        with pytest.raises(AdmissibilityError, match="missing"):
            recipe_policy.revalidate(["recipes/cosine.py"], recorded=None)

    def test_record_tampering_fails_closed(self, recipe_policy):
        original = recipe_policy.classify(["recipes/cosine.py"]).to_dict()
        original["candidate_class"] = CandidateClass.CONFIG_ONLY.value
        with pytest.raises(AdmissibilityError, match="does not match"):
            recipe_policy.revalidate(["recipes/cosine.py"], recorded=original)

    def test_recorded_accepted_must_be_json_boolean(self, recipe_policy):
        original = recipe_policy.classify(["recipes/cosine.py"]).to_dict()
        original["accepted"] = "true"
        with pytest.raises(AdmissibilityError, match="JSON boolean"):
            recipe_policy.revalidate(["recipes/cosine.py"], recorded=original)

    def test_variant_selection_tampering_fails_closed(self, recipe_policy):
        original = recipe_policy.classify(
            [], variant_selection={"policy": {"variant": "cosine"}},
        )
        with pytest.raises(AdmissibilityError, match="does not match"):
            recipe_policy.revalidate(
                [],
                variant_selection={"policy": {"variant": "lookahead"}},
                recorded=original.to_dict(),
            )

    def test_command_override_value_tampering_fails_closed(self, recipe_policy):
        original = recipe_policy.classify(
            [], override="--hparams '{\"lr\":0.0001}'",
        )
        with pytest.raises(AdmissibilityError, match="does not match"):
            recipe_policy.revalidate(
                [],
                override="--hparams '{\"lr\":0.01}'",
                recorded=original.to_dict(),
            )

    def test_free_mode_can_revalidate_legacy_unrecorded_spec(self):
        policy = CandidatePolicy(mode="free", protected=("evaluate.py",))
        verdict = policy.revalidate(["train.py"], recorded=None)
        assert verdict.candidate_class is CandidateClass.FREE_SOURCE

    def test_free_variant_only_candidate_is_free_source(self):
        policy = CandidatePolicy(mode="free")
        verdict = policy.classify(
            [], variant_selection={"policy": {"variant": "lookahead"}},
        )
        assert verdict.accepted
        assert verdict.candidate_class is CandidateClass.FREE_SOURCE


def test_policy_loads_all_enforcement_fields_from_one_config(tmp_path):
    adir = tmp_path / "automil"
    adir.mkdir()
    (adir / "config.yaml").write_text(yaml.safe_dump({
        "registry": {
            "mode": "architecture-preserving",
            "protected": ["models/**"],
            "allowed_override_options": ["--hparams"],
            "allowed_variant_kinds": ["policy"],
            "identity_locked_hparams": ["no_inst_cluster", "bag_weight"],
        },
        "files": {"editable": ["recipes/**"]},
    }))
    policy = load_candidate_policy(adir)
    assert policy == CandidatePolicy(
        mode="architecture-preserving",
        editable=("recipes/**",),
        protected=("models/**",),
        allowed_override_options=("--hparams",),
        allowed_variant_kinds=("policy",),
        identity_locked_hparams=("no_inst_cluster", "bag_weight"),
    )


class TestArchivedSpecRevalidation:
    def _spec_and_archive(self, tmp_path, policy):
        archive = tmp_path / "archive"
        (archive / "recipes").mkdir(parents=True)
        (archive / "recipes" / "cosine.py").write_text("NAME = 'cosine'\n")
        verdict = policy.classify(["recipes/cosine.py"])
        spec = {
            "overlay_manifest": {"recipes/cosine.py": "sha256:placeholder"},
            "deletions": [],
            "admissibility": verdict.to_dict(),
        }
        return spec, archive

    def test_exact_archived_overlay_revalidates(self, tmp_path, recipe_policy):
        spec, archive = self._spec_and_archive(tmp_path, recipe_policy)
        verdict = revalidate_candidate_spec(recipe_policy, spec, archive)
        assert verdict.candidate_class is CandidateClass.TRAIN_ONLY_SOURCE

    def test_unmanifested_archive_file_fails_closed(self, tmp_path, recipe_policy):
        spec, archive = self._spec_and_archive(tmp_path, recipe_policy)
        (archive / "train.py").write_text("# bypass\n")
        with pytest.raises(AdmissibilityError, match="unclaimed"):
            revalidate_candidate_spec(recipe_policy, spec, archive)

    def test_archived_variant_change_is_detected(self, tmp_path, recipe_policy):
        archive = tmp_path / "archive"
        archive.mkdir()
        original_selection = {"policy": {"variant": "cosine"}}
        verdict = recipe_policy.classify([], variant_selection=original_selection)
        (archive / "applied_variant.json").write_text(
            __import__("json").dumps({"policy": {"variant": "lookahead"}})
        )
        spec = {
            "overlay_manifest": {"applied_variant.json": "sha256:placeholder"},
            "deletions": [],
            "admissibility": verdict.to_dict(),
        }
        with pytest.raises(AdmissibilityError, match="does not match"):
            revalidate_candidate_spec(recipe_policy, spec, archive)

    def test_nested_framework_overlay_is_not_misclassified_as_agent_source(
        self, tmp_path, recipe_policy,
    ):
        archive = tmp_path / "archive"
        rel = "benchmarks/experiments/ccrcc/automil/applied_variant.json"
        path = archive / rel
        path.parent.mkdir(parents=True)
        selection = {"policy": {"variant": "cosine"}}
        path.write_text(__import__("json").dumps(selection))
        verdict = recipe_policy.classify([], variant_selection=selection)
        spec = {
            "overlay_manifest": {rel: "sha256:placeholder"},
            "framework_overlay_files": [rel],
            "deletions": [],
            "admissibility": verdict.to_dict(),
        }
        assert revalidate_candidate_spec(recipe_policy, spec, archive) == verdict

    def test_legacy_empty_free_mode_baseline_remains_launchable(self, tmp_path):
        policy = CandidatePolicy(mode="free")
        archive = tmp_path / "archive"
        archive.mkdir()
        verdict = revalidate_candidate_spec(
            policy, {"overlay_manifest": {}, "deletions": []}, archive,
        )
        assert verdict.accepted
        assert verdict.candidate_class is CandidateClass.FREE_SOURCE
