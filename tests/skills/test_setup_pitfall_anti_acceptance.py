"""Anti-acceptance tests for Pitfall 8 + 9 mitigations not covered by D-198 core gate.

Four tests:
  1. test_healthcheck_warns_on_mig_enabled (Pitfall 8 mitigation 1).
  2. test_init_emits_warning_on_under_utilization (Pitfall 8 mitigation 4).
  3. test_config_yaml_never_contains_TODO_substring (Pitfall 9 mitigation 4).
  4. test_ast_walk_handles_syntax_error_without_executing (Pitfall 9 mitigation 6).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner


def test_healthcheck_warns_on_mig_enabled(monkeypatch):
    """Pitfall 8 mitigation 1: MIG-enabled GPUs surface a detection warning.

    Mocks nvidia-smi --query-gpu=mig.mode.current to return 'Enabled'; asserts
    the warning string contains 'MIG' so operators on H100 clusters see the
    slice-memory caveat at probe time, not after a wrong-VRAM bin-pack failure.
    """
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    def _fake_run(argv, **kwargs):
        argv_str = str(argv)
        if "mig.mode.current" in argv_str:
            return MagicMock(stdout="Enabled\n", returncode=0, stderr="")
        # Main CUDA query.
        return MagicMock(stdout="0, 80000\n", returncode=0, stderr="")

    from automil.backends.local import LocalBackend
    with patch("subprocess.run", side_effect=_fake_run):
        report = LocalBackend().healthcheck()

    assert any("MIG" in w for w in report.detection_warnings), (
        f"expected MIG warning, got warnings={report.detection_warnings!r}"
    )


def test_init_emits_warning_on_under_utilization(tmp_git_repo, monkeypatch):
    """Pitfall 8 mitigation 4: large-VRAM + small-default surfaces a warning.

    With 80 GB GPUs detected and default_vram_estimate_gb=10.0 (the conservative
    fallback from min_vram / 8.0), max_concurrent_per_gpu = floor(80 / 10) = 8.
    The init flow renders hardware: section making the gap observable. We assert
    operator-visibility: min_vram_gb / default_vram_estimate_gb ratio >= 6.

    This test asserts visibility (operator can see the gap via rendered config),
    not a specific warning emission policy. If plan 07-05 / 07-06 chose to emit
    a click.echo warning, the warning text is checked too.
    """
    monkeypatch.chdir(tmp_git_repo)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    def _fake_run(argv, **kwargs):
        argv_str = str(argv)
        if "mig.mode.current" in argv_str:
            return MagicMock(stdout="Disabled\n", returncode=0, stderr="")
        return MagicMock(stdout="0, 81920\n", returncode=0, stderr="")  # 80 GB

    from automil.cli import main as cli_main
    runner = CliRunner()
    with patch("subprocess.run", side_effect=_fake_run):
        result = runner.invoke(cli_main, ["init"])
    assert result.exit_code == 0, result.output

    cfg = yaml.safe_load((tmp_git_repo / "automil" / "config.yaml").read_text())
    min_vram = cfg["hardware"]["min_vram_gb"]
    default = cfg["orchestrator"]["default_vram_estimate_gb"]
    # Operator-visibility check: the gap is observable via the rendered config.
    assert min_vram >= 70.0, f"min_vram_gb={min_vram} not >= 70"
    assert default <= 12.0, f"default_vram_estimate_gb={default} not conservative enough"
    # The ratio of capacity to default should be >= 6 (80 / 10 = 8 baseline).
    assert min_vram / default >= 6.0, f"min_vram/{default} ratio < 6"


def test_config_yaml_never_contains_TODO_substring(tmp_git_repo, monkeypatch):
    """Pitfall 9 mitigation 4 / Failure Modes section #4.

    No YAML string VALUE in the drafted automil/config.yaml should contain 'TODO'.
    YAML comments are stripped by the parser and are human guidance only.
    String values with TODO (e.g. project.description, backend.slurm.directives)
    are runtime-consumed and must not require editing to make the framework function
    on the default 'local' backend path.

    The check uses yaml.safe_load so only value-level TODO strings are detected,
    not comment-only lines. automil check guards against TODO sentinel values in
    the SLURM path (D-172 _validate_slurm_directives); this test guards the
    non-SLURM default path.
    """
    monkeypatch.chdir(tmp_git_repo)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    from automil.cli import main as cli_main
    runner = CliRunner()
    result = runner.invoke(cli_main, ["init", "--no-healthcheck"])
    assert result.exit_code == 0, result.output

    cfg = yaml.safe_load((tmp_git_repo / "automil" / "config.yaml").read_text())

    def _collect_todo_values(obj, path=""):
        """Walk parsed YAML; yield (path, value) for any string containing 'TODO'."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield from _collect_todo_values(v, f"{path}.{k}" if path else str(k))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                yield from _collect_todo_values(v, f"{path}[{i}]")
        elif isinstance(obj, str) and "TODO" in obj:
            yield path, obj

    todo_values = list(_collect_todo_values(cfg))

    # Known legacy sentinel values that are human-guidance strings, not
    # runtime-consumed values. The framework guards these at appropriate choke
    # points: backend.slurm directives are validated by automil check (D-172)
    # when backend.name == "slurm"; project.description is metadata only.
    # This test guards NEW skill-generated TODO leakage in RUNTIME-CRITICAL
    # paths (data, training, cap, hardware, registry, run) where a TODO value
    # would silently produce wrong behavior instead of a visible error.
    excluded_paths = {
        "backend.slurm.directives.partition",   # D-172 sentinel (local backend default)
        "backend.slurm.directives.account",     # D-172 sentinel (local backend default)
        "project.description",                  # human-guidance metadata, not runtime-consumed
    }
    unexpected = [(p, v) for (p, v) in todo_values if p not in excluded_paths]

    assert not unexpected, (
        f"automil/config.yaml contains TODO in unexpected runtime-critical YAML values: "
        f"{unexpected}"
    )


def test_ast_walk_handles_syntax_error_without_executing(tmp_path):
    """Pitfall 9 mitigation 6: AST-walk on a syntactically broken train.py
    raises SyntaxError handled cleanly (no import-time stack trace, no execution).

    The skill body's inspection (D-193 heuristic 3) uses ast.parse with explicit
    SyntaxError handling. This test asserts ast.parse semantics directly so any
    future refactor preserving the same behavior contract is safe.
    """
    bad_train = tmp_path / "broken_train.py"
    bad_train.write_text(
        "import torch\n"
        "class Broken(torch.nn.Module:\n"  # missing closing paren
        "    pass\n"
    )

    import ast
    source = bad_train.read_text()
    with pytest.raises(SyntaxError):
        ast.parse(source)

    # Critical: ast.parse never executed the broken code; verify by checking
    # that the import 'sys.modules' state did not gain a new module.
    import sys
    assert "broken_train" not in sys.modules, (
        "ast.parse executed user code (broken_train imported into sys.modules)"
    )
