# SPDX-License-Identifier: MIT
"""End-to-end test for CLI generate subcommand."""

from __future__ import annotations

import sys
import types
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

# Provide a lightweight stub for the logfire dependency required by ``cli``.
dummy_logfire = types.SimpleNamespace(
    metric_counter=lambda name: types.SimpleNamespace(add=lambda *a, **k: None),
    span=lambda name, attributes=None: nullcontext(),
    info=lambda *a, **k: None,
    warning=lambda *a, **k: None,
    error=lambda *a, **k: None,
    debug=lambda *a, **k: None,
    exception=lambda *a, **k: None,
    force_flush=lambda: None,
)
sys.modules.setdefault("logfire", cast(types.ModuleType, dummy_logfire))

# Minimal stub for the ``pydantic_ai`` package required by :mod:`cli` and
# :mod:`conversation`.
dummy_pydantic = types.SimpleNamespace(
    Agent=object,
    messages=types.SimpleNamespace(ModelMessage=object),
)
sys.modules.setdefault("pydantic_ai", cast(types.ModuleType, dummy_pydantic))
sys.modules.setdefault(
    "pydantic_ai.models",
    cast(types.ModuleType, types.SimpleNamespace(Model=object)),
)
sys.modules.setdefault(
    "generator",
    cast(types.ModuleType, types.SimpleNamespace(build_model=lambda *a, **k: None)),
)


class _DummyTqdm:  # pragma: no cover - simple progress bar stub
    def __init__(self, *args, **kwargs) -> None:
        pass

    def update(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        return None


sys.modules.setdefault(
    "tqdm", cast(types.ModuleType, types.SimpleNamespace(tqdm=_DummyTqdm))
)

# Basic YAML stub used by the loader module during import.
sys.modules.setdefault(
    "yaml", cast(types.ModuleType, types.SimpleNamespace(safe_load=lambda *a, **k: {}))
)

# Minimal loader stub to satisfy imports in :mod:`cli`.
dummy_loader = types.SimpleNamespace(
    configure_mapping_data_dir=lambda *a, **k: None,
    configure_prompt_dir=lambda *a, **k: None,
    load_evolution_prompt=lambda *a, **k: "prompt",
    load_mapping_items=lambda *a, **k: ([], "hash"),
    load_role_ids=lambda *a, **k: ["role"],
    load_plateau_definitions=lambda: [],
    load_prompt_text=lambda *a, **k: "",
    MAPPING_DATA_DIR=Path("data"),
)
sys.modules.setdefault("loader", cast(types.ModuleType, dummy_loader))

# Stub mapping and telemetry modules to satisfy CLI imports.
sys.modules.setdefault(
    "mapping",
    cast(
        types.ModuleType,
        types.SimpleNamespace(
            cache_write_json_atomic=lambda *a, **k: None,
            group_features_by_mapping=lambda *a, **k: {},
            map_set=lambda *a, **k: [],
        ),
    ),
)
sys.modules.setdefault(
    "telemetry",
    cast(
        types.ModuleType,
        types.SimpleNamespace(
            reset=lambda: None,
            print_summary=lambda: None,
            has_quarantines=lambda: False,
            record_quarantine=lambda *a, **k: None,
        ),
    ),
)
sys.modules.setdefault(
    "settings",
    cast(types.ModuleType, SimpleNamespace(load_settings=lambda: SimpleNamespace())),
)
sys.modules.setdefault(
    "service_loader",
    cast(types.ModuleType, SimpleNamespace(load_services=lambda *a, **k: [])),
)


# Stub implementations of the ``models`` module required by ``cli``.
@dataclass
class ServiceInput:
    service_id: str
    name: str
    description: str
    jobs_to_be_done: list[dict[str, Any]]
    features: list[dict[str, Any]] = field(default_factory=list)
    parent_id: str | None = None
    customer_type: str | None = None

    def model_dump_json(self) -> str:
        import json

        return json.dumps(
            {
                "service_id": self.service_id,
                "name": self.name,
                "parent_id": self.parent_id,
                "customer_type": self.customer_type,
                "description": self.description,
                "jobs_to_be_done": self.jobs_to_be_done,
                "features": self.features,
            },
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def model_dump(self, mode: str | None = None) -> dict[str, Any]:
        return {
            "service_id": self.service_id,
            "name": self.name,
            "parent_id": self.parent_id,
            "customer_type": self.customer_type,
            "description": self.description,
            "jobs_to_be_done": self.jobs_to_be_done,
            "features": self.features,
        }


class ServiceMeta:  # pragma: no cover - placeholder container
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class ServiceEvolution:  # pragma: no cover - placeholder container
    pass


class FeatureMappingRef:  # pragma: no cover - placeholder container
    pass


class MappingFeatureGroup:  # pragma: no cover - placeholder container
    pass


sys.modules.setdefault(
    "models",
    cast(
        types.ModuleType,
        types.SimpleNamespace(
            ServiceInput=ServiceInput,
            ServiceMeta=ServiceMeta,
            ServiceEvolution=ServiceEvolution,
            FeatureMappingRef=FeatureMappingRef,
            MappingFeatureGroup=MappingFeatureGroup,
            ReasoningConfig=object,
            StageModels=object,
        ),
    ),
)


class DummySession:
    """Minimal conversation session stub."""

    def __init__(self, client, **_: object) -> None:
        self.client = client
        self.stage = "stage"

    def add_parent_materials(
        self, service_input: ServiceInput
    ) -> None:  # pragma: no cover
        return None


class DummyModelFactory:
    """Factory providing placeholder models."""

    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover
        pass

    def model_name(self, *args, **kwargs) -> str:
        return "model"

    def get(self, *args, **kwargs):  # pragma: no cover
        return None


# Expose a placeholder ``PlateauGenerator`` before importing the CLI.
sys.modules.setdefault(
    "plateau_generator",
    cast(types.ModuleType, types.SimpleNamespace(PlateauGenerator=object)),
)

import cli  # noqa: E402


def _settings() -> SimpleNamespace:
    """Return minimal settings for the CLI."""

    return SimpleNamespace(
        model="model",
        openai_api_key="key",
        log_level="INFO",
        prompt_dir=Path("prompts"),
        context_id="ctx",
        inspiration="general",
        concurrency=1,
        reasoning=None,
        logfire_token=None,
        diagnostics=True,
        strict_mapping=False,
        use_local_cache=True,
        cache_mode="read",
        cache_dir=Path(".cache"),
        mapping_data_dir=Path("data"),
        mapping_sets=[],
        features_per_role=5,
        web_search=False,
        mapping_mode="per_set",
    )


def _load_services_stub(*_args, **_kwargs):
    service = ServiceInput(
        service_id="svc",
        name="alpha",
        description="desc",
        jobs_to_be_done=[{"name": "job"}],
    )
    return [service]


def test_cli_generate_matches_golden(monkeypatch, tmp_path, dummy_agent) -> None:
    """The run subcommand produces the locked golden output."""

    class DummyPlateauGenerator:
        """Generate an evolution using the ``dummy_agent`` fixture."""

        def __init__(self, *args, **kwargs) -> None:  # pragma: no cover
            self.agent = dummy_agent()

        async def _request_descriptions_async(self, names, session=None):
            return {name: "desc" for name in names}

        async def generate_service_evolution_async(
            self, service_input: ServiceInput, runtimes, *_, **__
        ):
            resp = await self.agent.run(service_input.model_dump_json(), dict)
            return SimpleNamespace(
                model_dump=lambda mode=None: resp.output.model_dump()
            )

    monkeypatch.setattr(cli, "load_settings", _settings)
    monkeypatch.setattr(cli, "_configure_logging", lambda *a, **k: None)
    import engine.processing_engine as pe
    import engine.service_execution as se

    monkeypatch.setattr(pe, "ModelFactory", DummyModelFactory)
    monkeypatch.setattr(pe, "_load_services_list", _load_services_stub)
    monkeypatch.setattr(pe, "configure_prompt_dir", lambda *a, **k: None)
    monkeypatch.setattr(pe, "configure_mapping_data_dir", lambda *a, **k: None)
    monkeypatch.setattr(pe, "load_evolution_prompt", lambda *a, **k: "prompt")
    monkeypatch.setattr(pe, "load_role_ids", lambda *a, **k: ["role"])
    monkeypatch.setattr(se, "load_mapping_items", lambda *a, **k: ([], "hash"))
    monkeypatch.setattr(se, "Agent", dummy_agent)
    monkeypatch.setattr(se, "ConversationSession", DummySession)
    monkeypatch.setattr(se, "PlateauGenerator", DummyPlateauGenerator)
    monkeypatch.setattr(se, "canonicalise_record", lambda d: d)
    monkeypatch.setattr(cli, "canonicalise_record", lambda d: d)

    output_file = tmp_path / "out.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main",
            "run",
            "--input-file",
            "sample-services.json",
            "--output-file",
            str(output_file),
            "--no-logs",
        ],
    )

    cli.main()

    expected = Path("tests/golden/sample_run.jsonl").read_text(encoding="utf-8")
    assert output_file.read_text(encoding="utf-8") == expected
