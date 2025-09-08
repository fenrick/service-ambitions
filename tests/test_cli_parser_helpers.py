# SPDX-License-Identifier: MIT
"""Unit tests for CLI parser helper functions."""

import argparse
import importlib

cli = importlib.import_module("cli.main")


def test_add_common_args_parses_model():
    """Common parser should accept shared options."""
    parser = cli._add_common_args(argparse.ArgumentParser())
    args = parser.parse_args(["--model", "foo", "--verbose", "--config", "c.yaml"])
    assert args.model == "foo"
    assert args.verbose == 1
    assert args.config == "c.yaml"


def _setup_parser():
    parser = argparse.ArgumentParser()
    common = cli._add_common_args(argparse.ArgumentParser(add_help=False))
    subparsers = parser.add_subparsers(dest="command", required=True)
    return parser, subparsers, common


def test_map_subparser_parses_files():
    """Map subparser should expose input and output arguments."""
    parser, subparsers, common = _setup_parser()
    cli._add_map_subparser(subparsers, common)
    args = parser.parse_args(["map", "--input-file", "in", "--output-file", "out"])
    assert args.func is cli._cmd_map
    assert args.input_file == "in"
    assert args.output_file == "out"


def test_run_subparser_parses_transcripts_dir():
    """Run subparser should capture transcripts directory."""
    parser, subparsers, common = _setup_parser()
    cli._add_run_subparser(subparsers, common)
    args = parser.parse_args(["run", "--transcripts-dir", "t"])
    assert args.func is cli._cmd_run
    assert args.transcripts_dir == "t"


def test_validate_subparser_sets_func():
    """Validate subparser should set the correct handler."""
    parser, subparsers, common = _setup_parser()
    cli._add_validate_subparser(subparsers, common)
    args = parser.parse_args(["validate"])
    assert args.func is cli._cmd_validate


def test_reverse_subparser_parses_files():
    """Reverse subparser should accept input and output paths."""
    parser, subparsers, common = _setup_parser()
    cli._add_reverse_subparser(subparsers, common)
    args = parser.parse_args(["reverse", "--input-file", "in", "--output-file", "out"])
    assert args.func is cli._cmd_reverse
    assert args.input_file == "in"
    assert args.output_file == "out"
