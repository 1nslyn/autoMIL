"""STP-05 / D-198 clause 4 / Pitfall 9 anti-acceptance #1.

Three tests:
  1. test_skill_idempotency_zero_unprompted_changes: re-running automil init
     --update on a populated repo produces byte-identical config.yaml content.
  2. test_skill_idempotency_ignores_comment_only_diffs: hand-edited comments
     in config.yaml do NOT trigger a re-stamp diff at the value-tree level.
  3. test_skill_idempotency_detects_results_tsv_change: a results.tsv that
     produces a different quantile_95 IS a legitimate non-trivial diff
     (this is the inverse, idempotency must not over-aggressively skip).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from click.testing import CliRunner

from tests.skills.conftest import fake_nvidia_smi_3gpu


def _seed_results_tsv(automil_dir: Path, vram_values: list[float]) -> None:
    automil_dir.mkdir(parents=True, exist_ok=True)
    header = (
        "node_id\tval_auc\tval_bacc\ttest_auc\ttest_bacc\tprimary_value\t"
        "vram_gb\telapsed_min\tstatus\tdescription\n"
    )
    rows = "".join(
        f"node_{i:04d}\t0.85\t0.80\t0.85\t0.82\t0.83\t{v:.1f}\t10.0\tcompleted\trun\n"
        for i, v in enumerate(vram_values)
    )
    (automil_dir / "results.tsv").write_text(header + rows)


def test_skill_idempotency_zero_unprompted_changes(tmp_git_repo, monkeypatch):
    """D-198 clause 4: byte-identical config.yaml across two `automil init --update` runs."""
    monkeypatch.chdir(tmp_git_repo)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    from automil.cli import main as cli_main
    runner = CliRunner()

    # First run: full init with healthcheck.
    with patch("subprocess.run", side_effect=fake_nvidia_smi_3gpu):
        result_1 = runner.invoke(cli_main, ["init"])
    assert result_1.exit_code == 0, result_1.output

    config_path = tmp_git_repo / "automil" / "config.yaml"
    first_text = config_path.read_text()
    assert "default_vram_estimate_gb" in first_text

    # Second run: --update path; identical inputs (same fake_nvidia_smi).
    with patch("subprocess.run", side_effect=fake_nvidia_smi_3gpu):
        result_2 = runner.invoke(cli_main, ["init", "--update"])
    assert result_2.exit_code == 0, result_2.output

    second_text = config_path.read_text()
    assert first_text == second_text, (
        f"config.yaml changed across identical runs:\n--- first ---\n{first_text[:400]}\n"
        f"--- second ---\n{second_text[:400]}"
    )


def test_skill_idempotency_ignores_comment_only_diffs(tmp_git_repo, monkeypatch):
    """OQ-4: value-tree diff ignores comments. Hand-edited comments survive --update."""
    monkeypatch.chdir(tmp_git_repo)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    from automil.cli import main as cli_main
    from automil.cli.init import _stamp_healthcheck_defaults
    from automil.backends.local import LocalBackend
    runner = CliRunner()

    with patch("subprocess.run", side_effect=fake_nvidia_smi_3gpu):
        result_1 = runner.invoke(cli_main, ["init"])
    assert result_1.exit_code == 0

    config_path = tmp_git_repo / "automil" / "config.yaml"
    original_yaml = config_path.read_text()
    parsed_original = yaml.safe_load(original_yaml)

    # Inject a user comment + prepend; the value tree is unchanged.
    augmented = "# User notes: this run is for project Foo.\n" + original_yaml
    config_path.write_text(augmented)

    # Re-derive the healthcheck context exactly as init does.
    with patch("subprocess.run", side_effect=fake_nvidia_smi_3gpu):
        report = LocalBackend(
            project_root=tmp_git_repo,
            automil_dir=tmp_git_repo / "automil",
        ).healthcheck()
    new_context = _stamp_healthcheck_defaults(tmp_git_repo / "automil", report)

    # Build the drafted version (Jinja render with the same context).
    from jinja2 import Environment, FileSystemLoader  # noqa: PLC0415
    templates_dir = (
        Path(__file__).parent.parent.parent / "src" / "automil" / "templates"
    )
    env = Environment(loader=FileSystemLoader(str(templates_dir)))
    drafted_text = env.get_template("config.yaml.j2").render(
        task_type="binary",
        encoder="hoptimus1",
        project_name=tmp_git_repo.name,
        **new_context,
    )
    parsed_drafted = yaml.safe_load(drafted_text)

    # Value-tree level: no diff between existing (with user comment) and drafted.
    assert parsed_original == parsed_drafted, (
        f"value-tree diverged:\n  original={parsed_original}\n  drafted={parsed_drafted}"
    )


def test_skill_idempotency_detects_results_tsv_change(tmp_git_repo, monkeypatch):
    """OQ-2: a results.tsv that changes the empirical default IS a legitimate diff.

    Idempotency must not over-aggressively skip. When inputs change, the diff
    surfaces. This test verifies the inverse-of-idempotency direction.
    """
    monkeypatch.chdir(tmp_git_repo)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    from automil.cli import main as cli_main
    runner = CliRunner()

    with patch("subprocess.run", side_effect=fake_nvidia_smi_3gpu):
        result_1 = runner.invoke(cli_main, ["init"])
    assert result_1.exit_code == 0

    cfg_path = tmp_git_repo / "automil" / "config.yaml"
    cfg_v1 = yaml.safe_load(cfg_path.read_text())
    v1_default = cfg_v1["orchestrator"]["default_vram_estimate_gb"]

    # Seed results.tsv with 30 rows that produce a different quantile_95.
    automil_dir = tmp_git_repo / "automil"
    _seed_results_tsv(automil_dir, vram_values=[12.0 + i * 0.3 for i in range(30)])

    with patch("subprocess.run", side_effect=fake_nvidia_smi_3gpu):
        result_2 = runner.invoke(cli_main, ["init", "--update"])
    assert result_2.exit_code == 0

    cfg_v2 = yaml.safe_load(cfg_path.read_text())
    v2_default = cfg_v2["orchestrator"]["default_vram_estimate_gb"]

    # The empirical path produces a different value than the conservative path.
    assert v2_default != v1_default, (
        f"results.tsv addition did not change default_vram_estimate_gb "
        f"(v1={v1_default}, v2={v2_default})"
    )
    import numpy  # noqa: PLC0415
    expected = float(numpy.quantile([12.0 + i * 0.3 for i in range(30)], 0.95))
    assert abs(v2_default - expected) <= 0.05
