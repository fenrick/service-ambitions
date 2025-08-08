import sys
from pathlib import Path

import pytest

from loader import (
    load_app_config,
    load_mapping_type_config,
    load_plateau_definitions,
    load_prompt,
    load_prompt_text,
    load_services,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_load_prompt_assembles_components(tmp_path):
    base = tmp_path / "prompts"
    (base / "situational_context").mkdir(parents=True)
    (base / "inspirations").mkdir(parents=True)
    (base / "situational_context" / "ctx.md").write_text("ctx", encoding="utf-8")
    (base / "service_feature_plateaus.md").write_text("plat", encoding="utf-8")
    (base / "definitions.md").write_text("defs", encoding="utf-8")
    (base / "inspirations" / "insp.md").write_text("insp", encoding="utf-8")
    (base / "task_definition.md").write_text("task", encoding="utf-8")
    (base / "response_structure.md").write_text("resp", encoding="utf-8")
    prompt = load_prompt("ctx", "insp", str(base))
    assert prompt == "ctx\n\nplat\n\ndefs\n\ninsp\n\ntask\n\nresp"


def test_load_prompt_missing_component(tmp_path):
    base = tmp_path / "prompts"
    base.mkdir()
    with pytest.raises(FileNotFoundError):
        load_prompt("ctx", "insp", str(base))


def test_load_prompt_text_plateau(tmp_path):
    base = tmp_path / "prompts"
    base.mkdir()
    (base / "plateau_prompt.md").write_text("content", encoding="utf-8")
    assert load_prompt_text("plateau_prompt", str(base)) == "content"


def test_load_prompt_text_mapping(tmp_path):
    base = tmp_path / "prompts"
    base.mkdir()
    (base / "mapping_prompt.md").write_text("map", encoding="utf-8")
    assert load_prompt_text("mapping_prompt", str(base)) == "map"


def test_load_prompt_text_description(tmp_path):
    base = tmp_path / "prompts"
    base.mkdir()
    (base / "description_prompt.md").write_text("desc", encoding="utf-8")
    assert load_prompt_text("description_prompt", str(base)) == "desc"


def test_load_plateau_definitions(tmp_path):
    base = tmp_path / "data"
    base.mkdir()
    (base / "service_feature_plateaus.json").write_text(
        '[{"id": "P1", "name": "Alpha", "description": "d"}]',
        encoding="utf-8",
    )
    plateaus = load_plateau_definitions(str(base))
    assert plateaus[0].name == "Alpha"


def test_load_services_reads_jsonl(tmp_path):
    data = tmp_path / "services.jsonl"
    data.write_text(
        '{"service_id": "a1", "name": "alpha", "description": "d", "jobs_to_be_done":'
        ' [], "features": [{"feature_id": "F1", "name": "Feat", "description":'
        ' "Desc"}]}\n\n{"service_id": "b2", "name": "beta", "description": "d",'
        ' "jobs_to_be_done": []}\n',
        encoding="utf-8",
    )
    services = list(load_services(str(data)))
    assert services[0].service_id == "a1"
    assert services[1].name == "beta"
    assert services[0].features[0].name == "Feat"


def test_load_services_missing(tmp_path):
    missing = tmp_path / "missing.jsonl"
    with pytest.raises(FileNotFoundError):
        list(load_services(str(missing)))


def test_load_services_invalid_json(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        '{"service_id": "a1", "name": "alpha", "jobs_to_be_done": []}\n{invalid}\n',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError):
        list(load_services(str(bad)))


def test_valid_fixture_parses():
    path = Path(__file__).parent / "fixtures" / "services-valid.jsonl"
    services = list(load_services(str(path)))
    assert services[0].service_id == "svc1"
    assert services[0].jobs_to_be_done == ["job1"]
    assert services[1].description == "Test"
    assert services[0].features[0].feature_id == "F1"


def test_invalid_fixture_raises():
    path = Path(__file__).parent / "fixtures" / "services-invalid.jsonl"
    with pytest.raises(RuntimeError):
        list(load_services(str(path)))


def test_load_mapping_type_config(tmp_path):
    base = tmp_path / "config"
    base.mkdir()
    (base / "app.json").write_text(
        '{"mapping_types": {"alpha": {"dataset": "ds", "label": "Alpha"}}}',
        encoding="utf-8",
    )
    load_app_config.cache_clear()
    load_mapping_type_config.cache_clear()
    config = load_mapping_type_config(str(base))
    assert config["alpha"].dataset == "ds"


def test_load_app_config(tmp_path):
    base = tmp_path / "config"
    base.mkdir()
    (base / "app.json").write_text(
        '{"plateau_map": {"Foundational": 1}, "mapping_types": {"beta": {"dataset":'
        ' "ds2", "label": "Beta"}}}',
        encoding="utf-8",
    )
    load_app_config.cache_clear()
    config = load_app_config(str(base))
    assert "beta" in config.mapping_types
    assert config.plateau_map["Foundational"] == 1
