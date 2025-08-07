import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from loader import load_prompt, load_services


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
    prompt = load_prompt(str(base), "ctx", "insp")
    assert prompt == "ctx\n\nplat\n\ndefs\n\ninsp\n\ntask\n\nresp"


def test_load_prompt_missing_component(tmp_path):
    base = tmp_path / "prompts"
    base.mkdir()
    with pytest.raises(FileNotFoundError):
        load_prompt(str(base), "ctx", "insp")


def test_load_services_reads_jsonl(tmp_path):
    data = tmp_path / "services.jsonl"
    data.write_text('{"name": "alpha"}\n\n{"name": "beta"}\n', encoding="utf-8")
    services = list(load_services(str(data)))
    assert services == [{"name": "alpha"}, {"name": "beta"}]


def test_load_services_missing(tmp_path):
    missing = tmp_path / "missing.jsonl"
    with pytest.raises(FileNotFoundError):
        list(load_services(str(missing)))


def test_load_services_invalid_json(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"name": "alpha"}\n{invalid}\n', encoding="utf-8")
    with pytest.raises(RuntimeError):
        list(load_services(str(bad)))


def test_valid_fixture_parses():
    path = Path(__file__).parent / "fixtures" / "services-valid.jsonl"
    services = list(load_services(str(path)))
    assert services[0]["name"] == "alpha"
    assert services[1]["description"] == "Test"


def test_invalid_fixture_raises():
    path = Path(__file__).parent / "fixtures" / "services-invalid.jsonl"
    with pytest.raises(RuntimeError):
        list(load_services(str(path)))
