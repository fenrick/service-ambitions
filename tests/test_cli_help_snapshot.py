# SPDX-License-Identifier: MIT
"""Snapshot-like checks for CLI help text to pin one-shot messaging."""

from __future__ import annotations

from cli.main import _build_parser


def test_run_help_mentions_one_shot_and_flags(capsys) -> None:
    parser = _build_parser()
    run_parser = parser._subparsers._group_actions[0].choices["run"]  # type: ignore[attr-defined]
    run_parser.print_help()
    out = capsys.readouterr().out
    assert "one-shot per plateau" in out
    assert "--cache-mode" in out
    assert "--concurrency" in out
    assert "--trace" in out


def test_validate_help_mentions_one_shot(capsys) -> None:
    parser = _build_parser()
    val_parser = parser._subparsers._group_actions[0].choices["validate"]  # type: ignore[attr-defined]
    val_parser.print_help()
    out = capsys.readouterr().out
    assert "one-shot per plateau" in out
