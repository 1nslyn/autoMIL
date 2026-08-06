"""Coverage for orchestrator._load_dotenv quoted values, export prefix, comments (CLN-03).

Six corner cases the legacy `partition('=')` parser silently mishandled:

1. Double- and single-quoted values keep embedded spaces and have quotes stripped.
2. `export KEY=value` produces key `KEY`, not `export KEY`.
3. Inline `# comment` after an unquoted value is dropped from the value.
4. Pre-existing `os.environ` entries are NOT overwritten (setdefault semantic).
5. `<root>/.env` is always loaded; additional files come from
   `env.dotenv_files` in automil/config.yaml. First writer wins.
6. Missing files are silently ignored — constructor never raises.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from automil.orchestrator import ExperimentOrchestrator


def _write_env(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)


@pytest.fixture
def orch_factory(tmp_path, monkeypatch):
    """Build an orchestrator pointed at tmp_path with a minimal automil/ skeleton.

    Returns (factory_callable, project_root_path). Calling the factory
    with an optional ``dotenv_files`` list constructs a fresh
    ExperimentOrchestrator whose ``__init__`` invokes ``_load_dotenv``
    against ``tmp_path``.
    """
    automil_dir = tmp_path / "automil"
    automil_dir.mkdir()
    config_path = automil_dir / "config.yaml"
    config_path.write_text("orchestrator:\n  poll_interval_sec: 5\n")
    (tmp_path / ".git").mkdir()  # so _find_git_root succeeds if walked

    def _factory(dotenv_files: list[str] | None = None):
        if dotenv_files:
            entries = "\n".join(f"    - {p}" for p in dotenv_files)
            config_path.write_text(
                "orchestrator:\n  poll_interval_sec: 5\n"
                "env:\n  dotenv_files:\n" + entries + "\n"
            )
        return ExperimentOrchestrator(project_root=tmp_path, automil_dir=automil_dir)

    return _factory, tmp_path


def test_quoted_values(orch_factory, monkeypatch):
    """Double- and single-quoted values keep embedded spaces; quotes are stripped."""
    factory, root = orch_factory
    _write_env(
        root / ".env",
        'QUOTED_DOUBLE="hello world"\nQUOTED_SINGLE=\'single\'\n',
    )
    monkeypatch.delenv("QUOTED_DOUBLE", raising=False)
    monkeypatch.delenv("QUOTED_SINGLE", raising=False)
    factory()  # Constructor calls _load_dotenv
    assert os.environ.get("QUOTED_DOUBLE") == "hello world"
    assert os.environ.get("QUOTED_SINGLE") == "single"


def test_export_prefix(orch_factory, monkeypatch):
    """A line `export KEY=value` produces key `KEY`, not `export KEY`."""
    factory, root = orch_factory
    _write_env(root / ".env", "export EXPORTED_KEY=exported_value\n")
    monkeypatch.delenv("EXPORTED_KEY", raising=False)
    monkeypatch.delenv("export EXPORTED_KEY", raising=False)
    factory()
    assert os.environ.get("EXPORTED_KEY") == "exported_value"
    assert "export EXPORTED_KEY" not in os.environ


def test_no_override_existing_env(orch_factory, monkeypatch):
    """Pre-existing env vars are NOT overwritten — setdefault semantic."""
    factory, root = orch_factory
    _write_env(root / ".env", "PREEXISTING=from_dotenv\n")
    monkeypatch.setenv("PREEXISTING", "from_shell")
    factory()
    assert os.environ.get("PREEXISTING") == "from_shell"


def test_search_order_root_then_configured(orch_factory, monkeypatch):
    """`<root>/.env` is loaded BEFORE configured extra dotenv files; first writer wins.

    Extra files are listed in ``env.dotenv_files`` of automil/config.yaml,
    not hardcoded to any consumer-specific directory.
    """
    factory, root = orch_factory
    _write_env(root / ".env", "OVERLAP=root_wins\n")
    _write_env(
        root / "benchmarks" / ".env",
        "OVERLAP=benchmarks_loses\nONLY_BENCH=ok\n",
    )
    monkeypatch.delenv("OVERLAP", raising=False)
    monkeypatch.delenv("ONLY_BENCH", raising=False)
    factory(dotenv_files=["benchmarks/.env"])
    assert os.environ.get("OVERLAP") == "root_wins"
    assert os.environ.get("ONLY_BENCH") == "ok"


def test_benchmarks_env_not_loaded_without_config(orch_factory, monkeypatch):
    """`benchmarks/.env` is NOT loaded by default — framework purity (memory rule:
    autoMIL is generic; autobench is one consumer). Consumers opt in via
    ``env.dotenv_files``.
    """
    factory, root = orch_factory
    _write_env(root / "benchmarks" / ".env", "BENCH_ONLY=should_not_load\n")
    monkeypatch.delenv("BENCH_ONLY", raising=False)
    factory()  # no dotenv_files configured
    assert "BENCH_ONLY" not in os.environ


def test_missing_files_is_noop(orch_factory):
    """Missing `.env` is a silent no-op — no exception."""
    factory, _ = orch_factory
    factory()  # Must not raise


def test_comment_after_value(orch_factory, monkeypatch):
    """Inline `# comment` after an unquoted value is stripped from the value.

    The legacy parser kept `actual_value  # trailing` as the entire value.
    python-dotenv strips the inline comment, leaving only `actual_value`.
    """
    factory, root = orch_factory
    _write_env(root / ".env", "COMMENTED=actual_value  # trailing\n")
    monkeypatch.delenv("COMMENTED", raising=False)
    factory()
    assert os.environ.get("COMMENTED") == "actual_value"
