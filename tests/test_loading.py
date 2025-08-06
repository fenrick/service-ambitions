import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from service_ambitions.loader import load_prompt, load_services


def test_load_prompt_reads_file(tmp_path):
    prompt = tmp_path / "prompt.md"
    prompt.write_text("Sample prompt", encoding="utf-8")
    assert load_prompt(str(prompt)) == "Sample prompt"


def test_load_prompt_missing(tmp_path):
    missing = tmp_path / "missing.md"
    with pytest.raises(FileNotFoundError):
        load_prompt(str(missing))


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
