import argparse
import asyncio
import json
from types import SimpleNamespace
from typing import Any

import cli
from cli import _cmd_generate_mapping
from models import (
    MaturityScore,
    PlateauFeature,
    PlateauResult,
    ServiceEvolution,
    ServiceInput,
    ServiceMeta,
)


def test_generate_mapping_maps_features(tmp_path, monkeypatch) -> None:
    feature = PlateauFeature(
        feature_id="f1",
        name="feat",
        description="desc",
        score=MaturityScore(level=1, label="Initial", justification="j"),
        customer_type="learner",
    )
    plateau = PlateauResult(
        plateau=1,
        plateau_name="alpha",
        service_description="desc",
        features=[feature],
    )
    evo = ServiceEvolution(
        meta=ServiceMeta(run_id="r1"),
        service=ServiceInput(
            service_id="s1",
            name="svc",
            description="d",
            jobs_to_be_done=[{"name": "job"}],
        ),
        plateaus=[plateau],
    )
    input_path = tmp_path / "evo.jsonl"
    output_path = tmp_path / "out.jsonl"
    input_path.write_text(f"{evo.model_dump_json()}\n", encoding="utf-8")

    called: dict[str, Any] = {}

    async def fake_map_features(self, session, feats):
        called["count"] = len(feats)
        return [f.model_copy(update={"mappings": {"applications": []}}) for f in feats]

    monkeypatch.setattr(cli, "Agent", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(cli.PlateauGenerator, "_map_features", fake_map_features)
    monkeypatch.setattr(
        cli, "configure_mapping_data_dir", lambda p: called.setdefault("path", p)
    )
    monkeypatch.setattr(
        cli,
        "canonicalise_record",
        lambda r: json.loads(json.dumps(r, default=str)),
    )

    settings = SimpleNamespace(
        model="m",
        openai_api_key="k",
        diagnostics=False,
        strict_mapping=False,
        mapping_data_dir="data",
        web_search=False,
        reasoning=None,
    )
    args = argparse.Namespace(
        input=str(input_path),
        output=str(output_path),
        model=None,
        mapping_model=None,
        diagnostics=None,
        strict_mapping=None,
        seed=0,
        no_logs=False,
        allow_prompt_logging=False,
        mapping_data_dir="maps",
        web_search=None,
        use_local_cache=False,
    )

    asyncio.run(_cmd_generate_mapping(args, settings))

    out = output_path.read_text(encoding="utf-8").strip()
    parsed = ServiceEvolution.model_validate_json(out)
    assert parsed.plateaus[0].features[0].mappings["applications"] == []
    assert called["count"] == 1
    assert called["path"] == "maps"
