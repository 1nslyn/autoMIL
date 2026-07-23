"""M-11 + L-4 (audit 2026-07-23) config-resolution robustness:

- M-11: a bare dataset name matching multiple grouped YAMLs must raise, not
  silently resolve to the alphabetically-first (mis-resolving everything
  downstream).
- L-4: a set-but-blank env var or an explicit empty ``${VAR:}`` default must
  fail fast instead of resolving to "" (a silently-broken relative path).
"""
from __future__ import annotations

import pytest

from autobench.config import _resolve_env_vars, load_dataset_config


def test_env_var_resolves(monkeypatch):
    monkeypatch.setenv("AUTOBENCH_TEST_ROOT_XYZ", "/data/root")
    assert _resolve_env_vars("${AUTOBENCH_TEST_ROOT_XYZ}/wsi") == "/data/root/wsi"


def test_default_used_when_unset(monkeypatch):
    monkeypatch.delenv("AUTOBENCH_UNSET_DEF_XYZ", raising=False)
    assert _resolve_env_vars("${AUTOBENCH_UNSET_DEF_XYZ:/fallback}") == "/fallback"


def test_blank_env_var_fails_fast(monkeypatch):
    monkeypatch.setenv("AUTOBENCH_TEST_BLANK_XYZ", "")
    with pytest.raises(ValueError):
        _resolve_env_vars("${AUTOBENCH_TEST_BLANK_XYZ}/wsi")


def test_empty_default_fails_fast(monkeypatch):
    monkeypatch.delenv("AUTOBENCH_UNSET_ABC_XYZ", raising=False)
    with pytest.raises(ValueError):
        _resolve_env_vars("${AUTOBENCH_UNSET_ABC_XYZ:}/wsi")


def test_ambiguous_dataset_name_raises(tmp_path, monkeypatch):
    (tmp_path / "tcga").mkdir()
    (tmp_path / "cptac").mkdir()
    (tmp_path / "tcga" / "foo.yaml").write_text("name: foo\n")
    (tmp_path / "cptac" / "foo.yaml").write_text("name: foo\n")
    monkeypatch.setattr("autobench.config.DATASETS_DIR", tmp_path)
    with pytest.raises(ValueError, match="[Aa]mbiguous"):
        load_dataset_config("foo")
