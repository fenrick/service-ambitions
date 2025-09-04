from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import generation.plateau_generator as plateau_generator
from runtime.environment import RuntimeEnv


def test_plateau_metadata_cached(monkeypatch) -> None:
    RuntimeEnv.reset()
    RuntimeEnv.initialize(
        cast(
            Any,
            SimpleNamespace(
                mapping_data_dir=Path("data"),
                prompt_dir=Path("prompts"),
                roles_file=Path("data/roles.json"),
            ),
        )
    )

    defs_calls = 0
    roles_calls = 0

    def fake_load_defs(base_dir, filename=...):
        nonlocal defs_calls
        defs_calls += 1
        return [SimpleNamespace(name="one"), SimpleNamespace(name="two")]

    def fake_load_roles(base_dir, filename=...):
        nonlocal roles_calls
        roles_calls += 1
        return ["alpha", "beta"]

    monkeypatch.setattr(plateau_generator, "load_plateau_definitions", fake_load_defs)
    monkeypatch.setattr(plateau_generator, "load_role_ids", fake_load_roles)

    plateau_generator.plateau_definitions.cache_clear()
    plateau_generator.default_plateau_map.cache_clear()
    plateau_generator.default_plateau_names.cache_clear()
    plateau_generator.default_role_ids.cache_clear()

    plateau_generator.default_plateau_map()
    plateau_generator.default_plateau_names()
    plateau_generator.default_plateau_map()
    plateau_generator.default_role_ids()
    plateau_generator.default_role_ids()

    assert defs_calls == 1
    assert roles_calls == 1
