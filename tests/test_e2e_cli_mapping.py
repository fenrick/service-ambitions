# SPDX-License-Identifier: MIT
"""End-to-end tests for the CLI mapping subcommand."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from core import mapping
from models import Contribution, MappingSet, Role

cli = importlib.import_module("cli.main")
cli_mapping = importlib.import_module("cli.mapping")


def _settings() -> SimpleNamespace:
    """Return minimal settings for the mapping CLI."""
    return SimpleNamespace(
        log_level="INFO",
        logfire_token=None,
        diagnostics=False,
        strict_mapping=False,
        strict=False,
        model="openai:gpt-5",
        models=None,
        use_local_cache=True,
        cache_mode="off",
        cache_dir=Path(".cache"),
        prompt_dir=Path("prompts"),
        mapping_data_dir=Path("tests/fixtures/catalogue"),
        roles_file=Path("tests/fixtures/roles.json"),
        mapping_sets=[
            MappingSet(
                name="Applications",
                file="applications.json",
                field="applications",
            ),
            MappingSet(
                name="Technologies",
                file="technologies.json",
                field="technologies",
            ),
        ],
    )


def _stub_map_set(*args, **kwargs):
    """Return deterministic mappings for test features."""

    session, set_name, _, features, *rest = args
    mapped = []
    app_lookup = {
        "F1": "app1",
        "F2": "app2",
        "ABCDEF": "app1",
        "ABCDEG": "app2",
    }
    tech_lookup = {
        "F1": "tech1",
        "F2": "tech2",
        "ABCDEF": "tech1",
        "ABCDEG": "tech2",
    }
    for feat in features:
        mappings = dict(feat.mappings)
        lookup = app_lookup if set_name == "applications" else tech_lookup
        mapping_id = lookup.get(feat.feature_id)
        if mapping_id:
            mappings.setdefault(set_name, []).append(Contribution(item=mapping_id))
        mapped.append(feat.model_copy(update={"mappings": mappings}))
    return mapped


async def _stub_map_set_async(*args, **kwargs):
    return _stub_map_set(*args, **kwargs)


def _stub_roles() -> list[Role]:
    """Return a deterministic role list for tests."""

    return [Role(role_id="learners", name="Learner", description="Learner role")]


def test_cli_map_matches_golden(monkeypatch, tmp_path) -> None:
    """The map subcommand produces the locked golden output."""
    monkeypatch.setattr(mapping, "map_set", _stub_map_set_async)
    monkeypatch.setattr(cli, "load_settings", lambda _path=None: _settings())
    monkeypatch.setattr(cli, "_configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(cli_mapping, "load_roles", lambda *a, **k: _stub_roles())

    input_file = Path("tests/fixtures/mapping_services.jsonl")
    output_file = tmp_path / "out.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main",
            "map",
            "--input-file",
            str(input_file),
            "--output-file",
            str(output_file),
        ],
    )

    cli.main()

    expected = Path("tests/golden/mapping_run.jsonl").read_text(encoding="utf-8")
    assert output_file.read_text(encoding="utf-8") == expected


def test_cli_map_single_service_filters_output(monkeypatch, tmp_path) -> None:
    """Mapping with --service-id limits the output to the requested service."""

    monkeypatch.setattr(mapping, "map_set", _stub_map_set_async)
    monkeypatch.setattr(cli, "load_settings", lambda _path=None: _settings())
    monkeypatch.setattr(cli, "_configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(cli_mapping, "load_roles", lambda *a, **k: _stub_roles())

    input_file = Path("tests/fixtures/mapping_services.jsonl")
    output_file = tmp_path / "svc.jsonl"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main",
            "map",
            "--input-file",
            str(input_file),
            "--output-file",
            str(output_file),
            "--service-id",
            "svc1",
        ],
    )

    cli.main()

    lines = [
        line for line in output_file.read_text(encoding="utf-8").splitlines() if line
    ]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["service"]["service_id"] == "svc1"
    apps = record["plateaus"][0]["features"][0]["mappings"]["applications"]
    assert apps[0]["item"] == "app1"


def test_cli_map_single_service_with_service_file(monkeypatch, tmp_path) -> None:
    """Mapping a single service can combine separate service and feature sources."""

    monkeypatch.setattr(mapping, "map_set", _stub_map_set_async)
    monkeypatch.setattr(cli, "load_settings", lambda _path=None: _settings())
    monkeypatch.setattr(cli, "_configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(cli_mapping, "load_roles", lambda *a, **k: _stub_roles())

    features_payload = {
        "service_id": "svc1",
        "plateau": 1,
        "plateau_name": "alpha",
        "service_description": "desc",
        "features": [
            {
                "feature_id": "ABCDEF",
                "name": "Feature1",
                "description": "Desc1",
                "score": {"level": 1, "label": "Initial", "justification": "j"},
                "customer_type": "learners",
                "mappings": {},
            }
        ],
    }

    features_file = tmp_path / "features.jsonl"
    features_file.write_text(json.dumps(features_payload) + "\n", encoding="utf-8")

    service_file = Path("tests/fixtures/services-valid.jsonl")
    output_file = tmp_path / "svc-from-service-file.jsonl"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main",
            "map",
            "--input-file",
            str(features_file),
            "--features-file",
            str(features_file),
            "--service-file",
            str(service_file),
            "--service-id",
            "svc1",
            "--output-file",
            str(output_file),
        ],
    )

    cli.main()

    lines = [
        line for line in output_file.read_text(encoding="utf-8").splitlines() if line
    ]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["service"]["name"] == "alpha"
    apps = record["plateaus"][0]["features"][0]["mappings"]["applications"]
    assert apps[0]["item"] == "app1"
