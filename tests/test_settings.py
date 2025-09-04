# SPDX-License-Identifier: MIT
"""Tests for configuration loading."""

import sys
from pathlib import Path

import pytest

from runtime.settings import load_settings

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_load_settings_reads_env(monkeypatch) -> None:
    """Environment variables should populate the settings model."""

    monkeypatch.setenv("OPENAI_API_KEY", "token")
    monkeypatch.setenv("USE_LOCAL_CACHE", "1")
    monkeypatch.setenv("CACHE_MODE", "write")
    monkeypatch.setenv("CACHE_DIR", "/tmp/cache")
    settings = load_settings()
    assert settings.openai_api_key == "token"
    assert settings.model == "openai:gpt-5-mini"
    assert settings.models is not None
    assert settings.models.descriptions == "openai:o4-mini"
    assert settings.log_level == "INFO"
    assert settings.request_timeout == 60
    assert settings.retries == 5
    assert settings.retry_base_delay == 0.5
    assert settings.features_per_role == 5
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

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("USE_LOCAL_CACHE", raising=False)
    monkeypatch.delenv("CACHE_MODE", raising=False)
    monkeypatch.delenv("CACHE_DIR", raising=False)
    with pytest.raises(RuntimeError):
        load_settings()


def test_load_settings_uses_xdg_cache_home(monkeypatch, tmp_path) -> None:
    """Default cache directory should honour ``XDG_CACHE_HOME``."""

    monkeypatch.setenv("OPENAI_API_KEY", "token")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    settings = load_settings()
    expected = tmp_path / "service-ambitions"
    assert settings.cache_dir == expected
    assert expected.is_dir()


def test_load_settings_rejects_unwritable_cache(monkeypatch, tmp_path) -> None:
    """Cache directory should raise ``RuntimeError`` when unwritable."""

    read_only = tmp_path / "no_write"
    read_only.mkdir()
    read_only.chmod(0o500)
    monkeypatch.setenv("OPENAI_API_KEY", "token")
    monkeypatch.setenv("CACHE_DIR", str(read_only))
    with pytest.raises(RuntimeError):
        load_settings()
    read_only.chmod(0o700)
