"""End-to-end tests for the CLI reverse command."""

import importlib
import json
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from pydantic_core import to_json

from core import mapping
from io_utils import loader
from models import (
    FeatureMappingRef,
    MappingFeatureGroup,
    MappingSet,
    MaturityScore,
    PlateauFeature,
    PlateauResult,
    ServiceEvolution,
    ServiceInput,
    ServiceMeta,
)
from observability import telemetry

cli = importlib.import_module("cli.main")

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


def test_cli_reverse_generates_caches(monkeypatch, tmp_path) -> None:
    """The reverse subcommand writes feature and mapping caches."""
    cache_dir = tmp_path / ".cache"
    settings = SimpleNamespace(
        log_level="INFO",
        logfire_token=None,
        diagnostics=False,
        strict_mapping=False,
        strict=False,
        models=None,
        use_local_cache=True,
        cache_mode="read",
        cache_dir=cache_dir,
        mapping_data_dir=tmp_path,
        prompt_dir=Path("prompts"),
        mapping_sets=[
            MappingSet(
                name="Applications",
                file="applications.json",
                field="applications",
            )
        ],
        context_id="unknown",
        model="gpt-5",
    )
    monkeypatch.setattr(cli, "load_settings", lambda _path=None: settings)
    monkeypatch.setattr(cli, "_configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(telemetry, "reset", lambda: None)
    monkeypatch.setattr(telemetry, "print_summary", lambda: None)
    monkeypatch.setattr(telemetry, "has_quarantines", lambda: False)
    monkeypatch.setattr(cli, "configure_mapping_data_dir", lambda *a, **k: None)
    monkeypatch.setattr(cli, "load_mapping_items", lambda *a, **k: ({}, "0" * 64))
    monkeypatch.setattr(loader, "load_prompt_text", lambda *a, **k: "")
    monkeypatch.setattr(cli, "canonicalise_record", lambda d: d)

    meta = ServiceMeta(
        run_id="r1",
        mapping_types=["applications"],
        seed=0,
        models={},
        web_search=False,
        catalogue_hash="0" * 64,
    )
    service = ServiceInput(
        service_id="svc",
        name="svc",
        description="d",
        jobs_to_be_done=[{"name": "job"}],
        features=[],
    )
    score = MaturityScore(level=1, label="Initial", justification="j")
    feat = PlateauFeature(
        feature_id="F1",
        name="Feature1",
        description="Desc1",
        score=score,
        customer_type="learners",
        mappings={},
    )
    group = MappingFeatureGroup(
        id="app1",
        name="App1",
        mappings=[FeatureMappingRef(feature_id="F1", description="Desc1")],
    )
    plateau = PlateauResult(
        plateau=1,
        plateau_name="alpha",
        service_description="desc",
        features=[feat],
        mappings={"applications": [group]},
    )
    evo = ServiceEvolution(meta=meta, service=service, plateaus=[plateau])
    input_file = tmp_path / "evo.jsonl"
    input_file.write_text(
        to_json(evo.model_dump(mode="json")).decode() + "\n",
        encoding="utf-8",
    )

    output_file = tmp_path / "features.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main",
            "reverse",
            "--input-file",
            str(input_file),
            "--output-file",
            str(output_file),
        ],
    )
    cli.main()

    expected_plateau = PlateauResult(
        plateau=1,
        plateau_name="alpha",
        service_description="desc",
        features=[feat],
        mappings={},
    )
    expected = ServiceEvolution(
        meta=meta.model_copy(update={"mapping_types": []}),
        service=service,
        plateaus=[expected_plateau],
    )
    assert (
        output_file.read_text(encoding="utf-8")
        == to_json(expected.model_dump(mode="json")).decode() + "\n"
    )

    feat_cache = cache_dir / "unknown" / "svc" / "1" / "features.json"
    assert feat_cache.exists()
    assert json.loads(feat_cache.read_text(encoding="utf-8")) == {
        "features": {
            "learners": [
                {
                    "name": "Feature1",
                    "description": "Desc1",
                    "score": {
                        "level": 1,
                        "label": "Initial",
                        "justification": "j",
                    },
                }
            ]
        }
    }

    key = mapping.build_cache_key(
        settings.model, "applications", "0" * 64, [feat], settings.diagnostics
    )
    map_cache = (
        cache_dir
        / "unknown"
        / "svc"
        / "1"
        / "mappings"
        / "applications"
        / f"{key}.json"
    )
    assert map_cache.exists()
    assert json.loads(map_cache.read_text(encoding="utf-8")) == {
        "features": [
            {"feature_id": "F1", "mappings": {"applications": [{"item": "app1"}]}}
        ]
    }
