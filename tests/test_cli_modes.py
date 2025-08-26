# SPDX-License-Identifier: MIT
"""Integration tests for CLI subcommands."""

import sys
from types import SimpleNamespace

import cli


def _prepare_settings():
    """Return minimal settings namespace for CLI tests."""

    return SimpleNamespace(
        log_level="INFO",
        logfire_token=None,
        diagnostics=False,
        strict_mapping=False,
    )


def test_run_invokes_generator(monkeypatch):
    """The run subcommand should call the generator without enabling diagnostics."""

    called = {}

    async def fake_generate(args, settings, transcripts_dir):
        called["args"] = args
        called["settings"] = settings

    monkeypatch.setattr(cli, "_cmd_generate_evolution", fake_generate)
    monkeypatch.setattr(cli, "load_settings", _prepare_settings)
    monkeypatch.setattr(cli, "_configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["main", "run", "--dry-run", "--no-logs"])

    cli.main()

    assert called["args"].dry_run is True
    assert called["settings"].diagnostics is False


def test_diagnose_enables_diagnostics(monkeypatch):
    """Diagnose subcommand forces diagnostics and transcripts."""

    called = {}

    async def fake_generate(args, settings, transcripts_dir):
        called["args"] = args
        called["settings"] = settings

    monkeypatch.setattr(cli, "_cmd_generate_evolution", fake_generate)
    monkeypatch.setattr(cli, "load_settings", _prepare_settings)
    monkeypatch.setattr(cli, "_configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["main", "diagnose", "--dry-run", "--no-logs"])

    cli.main()

    assert called["args"].dry_run is True
    assert called["settings"].diagnostics is True
    assert called["args"].no_logs is False


def test_validate_sets_dry_run(monkeypatch):
    """Validate subcommand performs a dry run by default."""

    called = {}

    async def fake_generate(args, settings, transcripts_dir):
        called["args"] = args
        called["settings"] = settings

    monkeypatch.setattr(cli, "_cmd_generate_evolution", fake_generate)
    monkeypatch.setattr(cli, "load_settings", _prepare_settings)
    monkeypatch.setattr(cli, "_configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["main", "validate", "--no-logs"])

    cli.main()

    assert called["args"].dry_run is True
    assert hasattr(called["args"], "transcripts_dir")
    assert called["args"].transcripts_dir is None


def test_cache_args_defaults(monkeypatch):
    """Cache CLI options default to read-only caching."""

    called = {}

    async def fake_generate(args, settings, transcripts_dir):
        called["args"] = args

    monkeypatch.setattr(cli, "_cmd_generate_evolution", fake_generate)
    monkeypatch.setattr(cli, "load_settings", _prepare_settings)
    monkeypatch.setattr(cli, "_configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["main", "run", "--dry-run", "--no-logs"])

    cli.main()

    args = called["args"]
    assert args.use_local_cache is True
    assert args.cache_mode == "read"
    assert args.cache_dir == ".cache"


def test_cache_args_custom(monkeypatch):
    """Custom cache options propagate to runtime."""

    called = {}

    async def fake_generate(args, settings, transcripts_dir):
        called["args"] = args

    monkeypatch.setattr(cli, "_cmd_generate_evolution", fake_generate)
    monkeypatch.setattr(cli, "load_settings", _prepare_settings)
    monkeypatch.setattr(cli, "_configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main",
            "run",
            "--dry-run",
            "--no-logs",
            "--use-local-cache",
            "--cache-mode",
            "write",
            "--cache-dir",
            "/tmp/cache",
        ],
    )

    cli.main()

    args = called["args"]
    assert args.use_local_cache is True
    assert args.cache_mode == "write"
    assert args.cache_dir == "/tmp/cache"
