# SPDX-License-Identifier: MIT
"""Tests for configuration loading."""

import os
import sys
from pathlib import Path

import pytest

from runtime.settings import load_settings

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_load_settings_reads_env(monkeypatch) -> None:
    """Environment variables should populate the settings model."""
    monkeypatch.setenv("SA_OPENAI_API_KEY", "token")
    monkeypatch.setenv("SA_USE_LOCAL_CACHE", "1")
    monkeypatch.setenv("SA_CACHE_MODE", "write")
    monkeypatch.setenv("SA_CACHE_DIR", "/tmp/cache")
    settings = load_settings()
    assert settings.openai_api_key == "token"
    assert settings.model == "openai:gpt-5-mini"
    assert settings.models is not None
    assert settings.models.descriptions == "openai:gpt-5"
    assert settings.log_level == "INFO"
    assert settings.request_timeout == 60
    assert settings.use_local_cache is True
    assert settings.cache_mode == "write"
    assert settings.cache_dir == Path("/tmp/cache")
    assert settings.mapping_data_dir == Path("data")
    assert settings.diagnostics is False
    assert settings.strict is False
    assert settings.strict_mapping is False
    assert settings.reasoning is not None
    assert settings.reasoning.effort == "medium"


def test_load_settings_requires_key(monkeypatch) -> None:
    """Missing API key should raise ``RuntimeError``."""
    monkeypatch.delenv("SA_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SA_USE_LOCAL_CACHE", raising=False)
    monkeypatch.delenv("SA_CACHE_MODE", raising=False)
    monkeypatch.delenv("SA_CACHE_DIR", raising=False)
    with pytest.raises(RuntimeError):
        load_settings()


def test_load_settings_uses_xdg_cache_home(monkeypatch, tmp_path) -> None:
    """Default cache directory should honour ``XDG_CACHE_HOME``."""
    monkeypatch.setenv("SA_OPENAI_API_KEY", "token")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv("SA_CACHE_DIR", "$XDG_CACHE_HOME/service-ambitions")
    settings = load_settings()
    expected = tmp_path / "service-ambitions"
    assert settings.cache_dir == expected
    assert expected.is_dir()


def test_load_settings_falls_back_without_xdg(monkeypatch, tmp_path) -> None:
    """Cache directory should default when ``XDG_CACHE_HOME`` is unset."""
    monkeypatch.setenv("SA_OPENAI_API_KEY", "token")
    monkeypatch.delenv("SA_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    # Ensure fallback default is writable in CI/sandboxed environments.
    writable_default = tmp_path / "service-ambitions"
    monkeypatch.setattr(
        "runtime.settings.DEFAULT_CACHE_DIR", writable_default, raising=False
    )
    settings = load_settings()
    assert settings.cache_dir == writable_default


def test_load_settings_rejects_unwritable_cache(monkeypatch, tmp_path) -> None:
    """Cache directory should raise ``RuntimeError`` when unwritable."""
    read_only = tmp_path / "no_write"
    read_only.mkdir()
    read_only.chmod(0o500)
    monkeypatch.setenv("SA_OPENAI_API_KEY", "token")
    monkeypatch.setenv("SA_CACHE_DIR", str(read_only))
    if os.geteuid() == 0:
        pytest.skip("running as root allows writes")
    with pytest.raises(RuntimeError):
        load_settings()
    read_only.chmod(0o700)
