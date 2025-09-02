# SPDX-License-Identifier: MIT
"""Verify cache path helper consistency."""

from __future__ import annotations

from engine.plateau_runtime import PlateauRuntime
from generation.plateau_generator import _feature_cache_path
from runtime.environment import RuntimeEnv
from utils.cache_paths import feature_cache


def test_feature_cache_paths_consistent(tmp_path) -> None:
    env = RuntimeEnv.instance()
    env.settings.cache_dir = tmp_path
    env.settings.context_id = "ctx"

    service_id = "svc"
    plateau = 1

    path_helper = feature_cache(service_id, plateau)
    path_runtime = PlateauRuntime(
        plateau=plateau, plateau_name="p", description="d"
    )._feature_cache_path(service_id)
    path_generator = _feature_cache_path(service_id, plateau)

    assert path_helper == path_runtime == path_generator
    assert path_helper.parent.is_dir()
