# SPDX-License-Identifier: MIT
import sys
from pathlib import Path

import pytest

from loader import (
    NORTH_STAR,
    load_ambition_prompt,
    load_app_config,
    load_mapping_type_config,
    load_plateau_definitions,
    load_prompt,
    load_prompt_text,
    load_role_ids,
    load_roles,
)
from models import JobToBeDone
from service_loader import load_services

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_load_prompt_assembles_components(tmp_path):
    prompts_dir = tmp_path / "prompts"
    data_dir = tmp_path / "data"
    (prompts_dir / "situational_context").mkdir(parents=True)
    (prompts_dir / "inspirations").mkdir(parents=True)
    (prompts_dir / "situational_context" / "ctx.md").write_text("ctx", encoding="utf-8")
    (prompts_dir / "inspirations" / "insp.md").write_text("insp", encoding="utf-8")
    (prompts_dir / "task_definition.md").write_text("task", encoding="utf-8")
    (prompts_dir / "response_structure.md").write_text("resp", encoding="utf-8")
    data_dir.mkdir()
    (data_dir / "definitions.json").write_text(
        '{"title": "Defs", "bullets": [{"id": "d1", "name": "d1", "definition":'
        ' "defs"}, {"id": "d2", "name": "d2", "definition": "extra"}]}',
        encoding="utf-8",
    )
    (data_dir / "service_feature_plateaus.json").write_text(
        '[{"id": "P1", "name": "Alpha", "description": {"core_idea": "core",'
        ' "key_characteristics": ["kc1"], "what_it_feels_like": "feel"}}]',
        encoding="utf-8",
    )
    prompt = load_prompt(
        "ctx",
        "insp",
        str(prompts_dir),
        definitions_dir=str(data_dir),
        plateaus_dir=str(data_dir),
    )
    expected = (
        "You are the world's leading service designer and enterprise architect; your"
        " job is to produce strictly-valid JSON structured outputs aligned to the"
        " schema.\n\nctx\n\n## Service feature plateaus\n\n1. **Alpha**\n   - Core"
        " idea: core\n   - Key characteristics:\n     - kc1\n   - What it feels like:"
        " feel\n\n## Defs\n\n1. **d1**\n   - Definition: defs\n2. **d2**\n   -"
        " Definition: extra\n\ninsp\n\ntask\n\nresp"
    )
    assert prompt == expected


def test_load_prompt_missing_component(tmp_path):
    prompts_dir = tmp_path / "prompts"
    data_dir = tmp_path / "data"
    prompts_dir.mkdir()
    data_dir.mkdir()
    (data_dir / "service_feature_plateaus.json").write_text(
        '[{"id": "P1", "name": "Alpha", "description": {"core_idea": "core",'
        ' "key_characteristics": ["kc1"], "what_it_feels_like": "feel"}}]',
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError):
        load_prompt(
            "ctx",
            "insp",
            str(prompts_dir),
            definitions_dir=str(data_dir),
            plateaus_dir=str(data_dir),
        )


def test_load_prompt_with_definition_keys(tmp_path):
    prompts_dir = tmp_path / "prompts"
    data_dir = tmp_path / "data"
    (prompts_dir / "situational_context").mkdir(parents=True)
    (prompts_dir / "inspirations").mkdir(parents=True)
    (prompts_dir / "situational_context" / "ctx.md").write_text("ctx", encoding="utf-8")
    (prompts_dir / "inspirations" / "insp.md").write_text("insp", encoding="utf-8")
    (prompts_dir / "task_definition.md").write_text("task", encoding="utf-8")
    (prompts_dir / "response_structure.md").write_text("resp", encoding="utf-8")
    data_dir.mkdir()
    (data_dir / "definitions.json").write_text(
        '{"title": "Defs", "bullets": [{"id": "d1", "name": "d1", "definition":'
        ' "defs1"}, {"id": "d2", "name": "d2", "definition": "defs2"}]}',
        encoding="utf-8",
    )
    (data_dir / "service_feature_plateaus.json").write_text(
        '[{"id": "P1", "name": "Alpha", "description": {"core_idea": "core",'
        ' "key_characteristics": ["kc1"], "what_it_feels_like": "feel"}}]',
        encoding="utf-8",
    )
    prompt = load_prompt(
        "ctx",
        "insp",
        str(prompts_dir),
        definitions_dir=str(data_dir),
        definition_keys=("d2",),
        plateaus_dir=str(data_dir),
    )
    expected = (
        "You are the world's leading service designer and enterprise architect; your"
        " job is to produce strictly-valid JSON structured outputs aligned to the"
        " schema.\n\nctx\n\n## Service feature plateaus\n\n1. **Alpha**\n   - Core"
        " idea: core\n   - Key characteristics:\n     - kc1\n   - What it feels like:"
        " feel\n\n## Defs\n\n1. **d2**\n   - Definition: defs2\n\ninsp\n\ntask\n\nresp"
    )
    assert prompt == expected


def test_load_ambition_prompt_includes_north_star(tmp_path):
    prompts_dir = tmp_path / "prompts"
    data_dir = tmp_path / "data"
    (prompts_dir / "situational_context").mkdir(parents=True)
    (prompts_dir / "inspirations").mkdir(parents=True)
    (prompts_dir / "situational_context" / "ctx.md").write_text("ctx", encoding="utf-8")
    (prompts_dir / "inspirations" / "insp.md").write_text("insp", encoding="utf-8")
    (prompts_dir / "task_definition.md").write_text("task", encoding="utf-8")
    (prompts_dir / "response_structure.md").write_text("resp", encoding="utf-8")
    data_dir.mkdir()
    (data_dir / "definitions.json").write_text(
        '{"title": "Defs", "bullets": [{"id": "d1", "name": "d1", "definition":'
        ' "defs"}]}',
        encoding="utf-8",
    )
    (data_dir / "service_feature_plateaus.json").write_text(
        '[{"id": "P1", "name": "Alpha", "description": {"core_idea": "core",'
        ' "key_characteristics": ["kc1"], "what_it_feels_like": "feel"}}]',
        encoding="utf-8",
    )
    prompt = load_ambition_prompt(
        "ctx",
        "insp",
        base_dir=str(prompts_dir),
        definitions_dir=str(data_dir),
        plateaus_dir=str(data_dir),
    )
    assert prompt.startswith(NORTH_STAR)


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
        '[{"id": "P1", "name": "Alpha", "description": {"core_idea": "d",'
        ' "key_characteristics": ["k"], "what_it_feels_like": "f"}}]',
        encoding="utf-8",
    )
    plateaus = load_plateau_definitions(str(base))
    assert plateaus[0].name == "Alpha"
    assert plateaus[0].description.core_idea == "d"


def test_load_roles_accepts_file_path(tmp_path):
    """``load_roles`` accepts a direct file path or a data directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    roles_file = data_dir / "roles.json"
    roles_file.write_text(
        '[{"role_id": "r1", "name": "Role", "description": "Desc"}]',
        encoding="utf-8",
    )

    from_dir = load_roles(str(data_dir))
    from_file = load_roles(str(roles_file))
    assert from_dir == from_file
    assert from_file[0].role_id == "r1"


def test_load_role_ids_extracts_identifiers(tmp_path):
    """``load_role_ids`` returns role identifiers for any path type."""

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    roles_file = data_dir / "roles.json"
    roles_file.write_text(
        '[{"role_id": "r1", "name": "Role", "description": "Desc"}]',
        encoding="utf-8",
    )

    ids_from_dir = load_role_ids(str(data_dir))
    ids_from_file = load_role_ids(str(roles_file))
    assert ids_from_dir == ["r1"]
    assert ids_from_file == ["r1"]


def test_load_services_reads_jsonl(tmp_path):
    """Services are parsed while dropping any provided features."""

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
    assert services[0].features == []


def test_load_services_missing(tmp_path):
    missing = tmp_path / "missing.jsonl"
    with pytest.raises(FileNotFoundError):
        list(load_services(str(missing)))


def test_load_services_quarantines_and_continues(tmp_path):
    """Invalid entries are quarantined while valid ones are yielded."""

    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        '{"service_id": "a1", "name": "alpha", "jobs_to_be_done":'
        ' []}\n{invalid}\n{"service_id": "a2", "name": "beta", "description": "d",'
        ' "jobs_to_be_done": []}\n',
        encoding="utf-8",
    )

    services = list(load_services(str(bad)))
    assert [s.service_id for s in services] == ["a2"]

    quarantine_dir = tmp_path / "quarantine"
    assert (quarantine_dir / "a1.json").exists()
    assert (quarantine_dir / "2.json").exists()


def test_load_services_with_job_objects(tmp_path):
    """Service loader preserves full job objects."""
    data = tmp_path / "services.jsonl"
    data.write_text(
        '{"service_id": "s1", "name": "alpha", "description": "d", '
        '"jobs_to_be_done": [{"name": "JobA", "note": "x"}, {"name": "JobB"}], '
        '"features": []}\n',
        encoding="utf-8",
    )
    services = list(load_services(str(data)))
    first_job = services[0].jobs_to_be_done[0]
    assert isinstance(first_job, JobToBeDone)
    assert first_job.name == "JobA"
    # Extra fields should be retained on the model
    assert first_job.note == "x"


def test_valid_fixture_parses():
    """Fixtures load successfully with features removed."""

    path = Path(__file__).parent / "fixtures" / "services-valid.jsonl"
    services = list(load_services(str(path)))
    assert services[0].service_id == "svc1"
    assert [j.name for j in services[0].jobs_to_be_done] == ["job1"]
    assert services[1].description == "Test"
    assert services[0].features == []


def test_invalid_fixture_quarantines(tmp_path):
    """Invalid fixture entries are quarantined without raising errors."""

    src = Path(__file__).parent / "fixtures" / "services-invalid.jsonl"
    data = tmp_path / "services-invalid.jsonl"
    data.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    services = list(load_services(str(data)))
    assert services == []

    quarantine_dir = tmp_path / "quarantine"
    assert quarantine_dir.exists()
    assert any(quarantine_dir.iterdir())


def test_load_mapping_type_config(tmp_path):
    base = tmp_path / "config"
    base.mkdir()
    (base / "app.yaml").write_text(
        """
mapping_types:
  alpha:
    dataset: ds
    label: Alpha
""",
        encoding="utf-8",
    )
    load_app_config.cache_clear()
    load_mapping_type_config.cache_clear()
    config = load_mapping_type_config(str(base))
    assert config["alpha"].dataset == "ds"


def test_load_app_config(tmp_path):
    base = tmp_path / "config"
    base.mkdir()
    (base / "app.yaml").write_text(
        """
mapping_types:
  beta:
    dataset: ds2
    label: Beta
""",
        encoding="utf-8",
    )
    load_app_config.cache_clear()
    config = load_app_config(str(base))
    assert "beta" in config.mapping_types


def test_load_app_config_models(tmp_path):
    base = tmp_path / "config"
    base.mkdir()
    (base / "app.yaml").write_text(
        """
models:
  features: m1
  mapping: m2
""",
        encoding="utf-8",
    )
    load_app_config.cache_clear()
    config = load_app_config(str(base))
    assert config.models is not None
    assert config.models.features == "m1"
    assert config.models.mapping == "m2"


def test_load_app_config_reasoning(tmp_path):
    base = tmp_path / "config"
    base.mkdir()
    (base / "app.yaml").write_text(
        """
reasoning:
  effort: high
  summary: detailed
""",
        encoding="utf-8",
    )
    load_app_config.cache_clear()
    config = load_app_config(str(base))
    assert config.reasoning is not None
    assert config.reasoning.effort == "high"
    assert config.reasoning.summary == "detailed"
