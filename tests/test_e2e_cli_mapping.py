# SPDX-License-Identifier: MIT
"""End-to-end tests for the CLI mapping subcommand."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from models import MappingSet, Role

cli = importlib.import_module("cli.main")
cli_mapping = importlib.import_module("cli.mapping")

_MAPPING_LOOKUP = {
    "applications": {
        "F1": "app1",
        "F2": "app2",
        "ABCDEF": "app1",
        "ABCDEG": "app2",
    },
    "technologies": {
        "F1": "tech1",
        "F2": "tech2",
        "ABCDEF": "tech1",
        "ABCDEG": "tech2",
    },
}


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


def _stub_roles() -> list[Role]:
    """Return a deterministic role list for tests."""
    return [Role(role_id="learners", name="Learner", description="Learner role")]


class _FakeSession:
    """Minimal stand-in for ``ConversationSession`` used in mapping tests."""

    def __init__(
        self,
        mapping_lookup: dict[str, dict[str, str]],
        *,
        stage: str = "mapping",
        history: list[str] | None = None,
    ) -> None:
        self._mapping_lookup = mapping_lookup
        self.stage = stage
        self.diagnostics = False
        self.log_prompts = False
        self.transcripts_dir = None
        self.use_local_cache = True
        self.cache_mode: str = "off"
        self.client = SimpleNamespace(model=SimpleNamespace(model_name="stub-model"))
        self._history = history or []
        self._service_context: str | None = None

    def add_parent_materials(
        self, service_input
    ) -> None:  # pragma: no cover - simple setter
        context = service_input.model_dump_json()
        self._service_context = context
        self._history = [context]

    def history_context_text(self) -> str:
        return self._service_context or ""

    def derive(self) -> "_FakeSession":
        clone = _FakeSession(
            self._mapping_lookup,
            stage=self.stage,
            history=list(self._history),
        )
        clone.log_prompts = self.log_prompts
        clone.use_local_cache = self.use_local_cache
        clone.cache_mode = self.cache_mode
        clone._service_context = self._service_context
        return clone

    async def ask_async(
        self, prompt: str, *, feature_id: str | None = None
    ) -> dict[str, object]:
        field = self.stage.rsplit("_", 1)[-1]
        lookup = self._mapping_lookup.get(field, {})
        features = getattr(self, "_pending_features", [])
        payload: list[dict[str, object]] = []
        for feat in features:
            mapping_id = lookup.get(feat.feature_id)
            entries = []
            if mapping_id:
                entries.append({"item": mapping_id, "facets": None})
            payload.append({"feature_id": feat.feature_id, field: entries})
        return {"features": payload}


def _install_fake_sessions(
    monkeypatch, mapping_lookup: dict[str, dict[str, str]]
) -> None:
    """Patch the mapping session factory to return fake sessions."""

    def _factory(
        settings, *, allow_prompt_logging: bool, transcripts_dir, cache_mode: str
    ):
        def _session_for(evo):
            session = _FakeSession(mapping_lookup)
            session.log_prompts = allow_prompt_logging
            session.use_local_cache = getattr(settings, "use_local_cache", True)
            session.cache_mode = cache_mode
            session.add_parent_materials(evo.service)
            return session

        return _session_for

    monkeypatch.setattr(cli_mapping, "_build_mapping_session_factory", _factory)


def test_cli_map_matches_golden(monkeypatch, tmp_path) -> None:
    """The map subcommand produces the locked golden output."""
    monkeypatch.setattr(cli, "load_settings", lambda _path=None: _settings())
    monkeypatch.setattr(cli, "_configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(cli_mapping, "load_roles", lambda *a, **k: _stub_roles())
    _install_fake_sessions(monkeypatch, _MAPPING_LOOKUP)

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
    monkeypatch.setattr(cli, "load_settings", lambda _path=None: _settings())
    monkeypatch.setattr(cli, "_configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(cli_mapping, "load_roles", lambda *a, **k: _stub_roles())
    _install_fake_sessions(monkeypatch, _MAPPING_LOOKUP)

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
    monkeypatch.setattr(cli, "load_settings", lambda _path=None: _settings())
    monkeypatch.setattr(cli, "_configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(cli_mapping, "load_roles", lambda *a, **k: _stub_roles())
    _install_fake_sessions(monkeypatch, _MAPPING_LOOKUP)

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

    record = json.loads(output_file.read_text(encoding="utf-8").strip())
    mappings = record["plateaus"][0]["features"][0]["mappings"]
    assert mappings["applications"][0]["item"] == "app1"
    assert mappings["technologies"][0]["item"] == "tech1"
