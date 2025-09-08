# SPDX-License-Identifier: MIT
"""Command-line interface for generating service evolutions."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import logging
import os
import platform
import random
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Coroutine, cast

import logfire
from pydantic_core import to_json

from constants import DEFAULT_CACHE_DIR
from core.canonical import canonicalise_record
from core.conversation import _prompt_cache_key, _prompt_cache_path
from core.mapping import build_cache_key, cache_path, cache_write_json_atomic
from engine.processing_engine import ProcessingEngine
from generation.plateau_generator import _feature_cache_path, default_plateau_names
from io_utils.loader import (
    configure_mapping_data_dir,
    load_mapping_items,
    load_prompt_text,
)
from models import (
    Contribution,
    FeatureItem,
    MappingFeature,
    MappingResponse,
    PlateauFeaturesResponse,
    PlateauResult,
    ServiceEvolution,
    StageModels,
)
from observability import telemetry
from observability.monitoring import init_logfire
from runtime.environment import RuntimeEnv
from runtime.settings import Settings, load_settings

from .data_validation import validate_data_dir
from .mapping import load_catalogue, remap_features, write_output

SERVICES_FILE_HELP = "Path to the services JSONL file"
OUTPUT_FILE_HELP = "File to write the results"
TRANSCRIPTS_HELP = (
    "Directory to store per-service request/response transcripts. "
    "Defaults to a '_transcripts' folder beside the output file."
)


LOG_LEVELS = ["fatal", "error", "warn", "notice", "info", "debug", "trace"]

# Module logger for CLI diagnostics mirroring
logger = logging.getLogger(__name__)


def _print_version() -> None:
    """Print the installed package version."""
    try:
        pkg_version = version("service-ambitions")
    except PackageNotFoundError:  # pragma: no cover - fallback for editable installs
        pkg_version = "unknown"
    line = f"service-ambitions {pkg_version}"
    print(line)
    logger.info(line)


def _print_diagnostics() -> None:
    """Output basic environment information for health checks."""
    _print_version()
    py = f"Python {platform.python_version()}"
    plat = f"Platform {platform.platform()}"
    print(py)
    print(plat)
    logger.info(py)
    logger.info(plat)

    missing = [var for var in ["SA_OPENAI_API_KEY"] if not os.getenv(var)]
    if missing:
        line = "Missing env vars: " + ", ".join(missing)
        print(line)
        logger.info(line)
    else:
        line = "Required env vars present"
        print(line)
        logger.info(line)


def _configure_logging(args: argparse.Namespace, settings: Settings) -> None:
    """Configure Logfire based on verbosity flags."""
    index = 2 + args.verbose - args.quiet
    index = max(0, min(len(LOG_LEVELS) - 1, index))
    min_log_level = LOG_LEVELS[index]
    init_logfire(settings.logfire_token, min_log_level, json_logs=args.json_logs)


async def _cmd_run(args: argparse.Namespace, transcripts_dir: Path | None) -> None:
    """Execute the default evolution generation workflow."""
    await _cmd_generate_evolution(args, transcripts_dir)


async def _cmd_validate(args: argparse.Namespace, transcripts_dir: Path | None) -> None:
    """Validate inputs without invoking the language model."""
    args.dry_run = True
    await _cmd_generate_evolution(args, transcripts_dir)


async def _cmd_map(args: argparse.Namespace, transcripts_dir: Path | None) -> None:
    """Populate feature mappings for an existing features file."""
    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    settings = RuntimeEnv.instance().settings
    items, catalogue_hash = load_catalogue(args.mapping_data_dir, settings)

    text = input_path.read_text(encoding="utf-8")
    evolutions = [
        ServiceEvolution.model_validate_json(line)
        for line in text.splitlines()
        if line.strip()
    ]

    await remap_features(
        evolutions,
        items,
        settings,
        args.cache_mode,
        catalogue_hash,
    )
    write_output(evolutions, Path(args.output_file))


def _should_write_cache(mode: str, exists_before: bool) -> bool:
    """Return True when cache policy permits writing.

    - off: never write
    - refresh: always write (overwrite)
    - read/write: write only when the file is absent
    """
    if mode == "off":
        return False
    if mode == "refresh":
        return True
    return not exists_before


def _reconstruct_feature_cache(
    svc_id: str, plateau: PlateauResult, service_name: str | None = None
) -> None:
    """Write grouped features for ``plateau`` to the cache respecting flags."""
    with logfire.span(
        "cli.reconstruct_feature_cache",
        attributes={"service_id": svc_id, "plateau": plateau.plateau},
    ):
        settings = RuntimeEnv.instance().settings
        if not getattr(settings, "use_local_cache", True):
            return
        block: dict[str, list[FeatureItem]] = {}
        for feat in plateau.features:
            item = FeatureItem(
                name=feat.name, description=feat.description, score=feat.score
            )
            block.setdefault(feat.customer_type, []).append(item)
        payload = PlateauFeaturesResponse(features=block)
        # Write canonical human-friendly file for quick inspection
        feat_path = _feature_cache_path(svc_id, plateau.plateau)
        if _should_write_cache(
            getattr(settings, "cache_mode", "read"), feat_path.exists()
        ):
            cache_write_json_atomic(feat_path, payload.model_dump())

        # Also write hashed prompt-cache entry so different contexts co-exist
        try:
            from generation.plateau_generator import (
                default_role_ids as _role_ids_fn,
            )
            roles = list(_role_ids_fn())
        except ImportError:  # pragma: no cover - defensive import
            # Fallback when defaults are unavailable during import time
            roles = []
        roles_str = ", ".join(f'"{r}"' for r in roles)
        template = load_prompt_text("plateau_prompt")
        prompt = template.format(
            service_name=(service_name or svc_id),
            service_description=plateau.service_description,
            plateau=str(plateau.plateau),
            roles=str(roles_str),
        )
        feature_model = (
            getattr(getattr(settings, "models", None), "features", None)
            or settings.model
        )
        stage = f"features_{plateau.plateau}"
        f_key = _prompt_cache_key(prompt, feature_model, stage)
        f_cache = _prompt_cache_path(svc_id, stage, f_key)
        if _should_write_cache(
            getattr(settings, "cache_mode", "read"), f_cache.exists()
        ):
            cache_write_json_atomic(f_cache, payload.model_dump())


def _rebuild_mapping_cache(
    svc_id: str,
    plateau: PlateauResult,
    settings: Settings,
    catalogue_hash: str,
) -> None:
    """Write mapping cache entries for all configured mapping sets.

    Respects ``settings.use_local_cache`` and ``settings.cache_mode``.
    """
    with logfire.span(
        "cli.rebuild_mapping_cache",
        attributes={"service_id": svc_id, "plateau": plateau.plateau},
    ):
        if not getattr(settings, "use_local_cache", True):
            plateau.mappings = {}
            return
        for cfg in settings.mapping_sets:
            features_by_id: dict[str, MappingFeature] = {
                f.feature_id: MappingFeature(feature_id=f.feature_id)
                for f in plateau.features
            }
            for group in plateau.mappings.get(cfg.field, []):
                for ref in group.mappings:
                    contrib = Contribution(item=group.id)
                    features_by_id[ref.feature_id].mappings.setdefault(
                        cfg.field, []
                    ).append(contrib)
            key = build_cache_key(
                settings.model,
                cfg.field,
                catalogue_hash,
                plateau.features,
                settings.diagnostics,
                plateau=plateau.plateau,
                service_description=plateau.service_description,
                service_name=None,
            )
            cache_file = cache_path(svc_id, plateau.plateau, cfg.field, key)
            if _should_write_cache(
                getattr(settings, "cache_mode", "read"), cache_file.exists()
            ):
                payload = MappingResponse(features=list(features_by_id.values()))
                cache_write_json_atomic(cache_file, payload.model_dump())
        plateau.mappings = {}


def _rebuild_description_cache(
    svc_id: str, evo: ServiceEvolution, settings: Settings
) -> None:
    """Write the descriptions prompt-cache entry for ``svc_id``.

    Reconstructs the same prompt used by the descriptions stage so future runs
    can hit the prompt cache and avoid re-calling the model.
    Respects ``settings.use_local_cache`` and ``settings.cache_mode``.
    """
    with logfire.span(
        "cli.rebuild_description_cache",
        attributes={"service_id": svc_id},
    ):
        if not getattr(settings, "use_local_cache", True):
            return
        # Build the exact prompt used for plateau descriptions.
        names = default_plateau_names()
        # Build "1. <name>" lines using configured plateau definitions
        lines: list[str] = []
        # Map plateau name -> level from configuration
        # Reuse plateau_generator.default_plateau_names ordering and enumerate for level
        for idx, name in enumerate(names, start=1):
            lines.append(f"{idx}. {name}")
        plateaus_str = "\n".join(lines)
        template = load_prompt_text("plateau_descriptions_prompt")
        prompt = template.format(plateaus=plateaus_str)
        # Use the descriptions model override when set, otherwise fallback to base model
        model_name = (
            getattr(getattr(settings, "models", None), "descriptions", None)
            or settings.model
        )
        key = _prompt_cache_key(prompt, model_name, "descriptions")
        cache_file = _prompt_cache_path(svc_id, "descriptions", key)
        if not _should_write_cache(
            getattr(settings, "cache_mode", "read"), cache_file.exists()
        ):
            return
        # Build payload with all plateau descriptions in the expected schema
        desc_lookup = {p.plateau_name: p.service_description for p in evo.plateaus}
        items = [
            {
                "plateau": idx + 1,
                "plateau_name": name,
                "description": desc_lookup.get(name, ""),
            }
            for idx, name in enumerate(names)
        ]
        payload = {"descriptions": items}
        cache_write_json_atomic(cache_file, payload)


def _cmd_reverse(args: argparse.Namespace, transcripts_dir: Path | None) -> None:
    """Backfill feature and mapping caches from ``evolutions.jsonl``."""
    settings = RuntimeEnv.instance().settings
    configure_mapping_data_dir(args.mapping_data_dir or settings.mapping_data_dir)
    _, catalogue_hash = load_mapping_items(settings.mapping_sets)

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    text = input_path.read_text(encoding="utf-8")
    evolutions = [
        ServiceEvolution.model_validate_json(line)
        for line in text.splitlines()
        if line.strip()
    ]

    with output_path.open("w", encoding="utf-8") as out_fh:
        for evo in evolutions:
            svc_id = evo.service.service_id
            _rebuild_description_cache(svc_id, evo, settings)
            for plateau in evo.plateaus:
                _reconstruct_feature_cache(svc_id, plateau, evo.service.name)
                _rebuild_mapping_cache(svc_id, plateau, settings, catalogue_hash)
            evo.meta.mapping_types = []
            record = canonicalise_record(evo.model_dump(mode="json"))
            out_fh.write(to_json(record).decode() + "\n")


async def _cmd_generate_evolution(
    args: argparse.Namespace, transcripts_dir: Path | None
) -> None:
    """Generate service evolution summaries via ``ProcessingEngine``."""
    engine = ProcessingEngine(args, transcripts_dir)
    success = await engine.run()
    await engine.finalise()
    if not success:
        logfire.warning("One or more services failed during processing")


def _add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add CLI options shared across subcommands.

    Parameters
    ----------
    parser:
        Parser to augment with common arguments.

    Returns:
    -------
    argparse.ArgumentParser
        The parser instance with added arguments.
    """
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Global chat model name (default openai:gpt-5). "
            "Can also be set via the SA_MODEL env variable."
        ),
    )
    parser.add_argument(
        "--descriptions-model",
        type=str,
        default=None,
        help="Model for plateau descriptions (default openai:o4-mini)",
    )
    parser.add_argument(
        "--features-model",
        type=str,
        default=None,
        help=(
            "Model for feature generation (default openai:gpt-5; "
            "use openai:o4-mini for lower cost)"
        ),
    )
    parser.add_argument(
        "--mapping-model",
        type=str,
        default=None,
        help="Model for feature mapping (default openai:o4-mini)",
    )
    parser.add_argument(
        "--search-model",
        type=str,
        default=None,
        help="Model for web search (default openai:gpt-4o-search-preview)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help=(
            "Increase logging verbosity (-v notice, -vv info, -vvv debug, -vvvv trace)"
        ),
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="count",
        default=0,
        help="Decrease logging verbosity (-q error, -qq fatal)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Number of services to process concurrently",
    )
    parser.add_argument(
        "--max-services",
        type=int,
        default=None,
        help="Process at most this many services",
    )
    parser.add_argument(
        "--strict-mapping",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Fail when feature mappings are missing",
    )
    parser.add_argument(
        "--mapping-data-dir",
        type=str,
        default=None,
        help="Directory containing mapping reference data",
    )
    parser.add_argument(
        "--roles-file",
        type=str,
        default="data/roles.json",
        help="Path to JSON file containing role identifiers",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed random number generation for reproducible output",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate inputs without calling the API",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        default=False,
        help="Display a progress bar during execution",
    )
    parser.add_argument(
        "--continue",
        dest="resume",
        action="store_true",
        default=False,
        help="Resume processing using processed_ids.txt",
    )
    parser.add_argument(
        "--web-search",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable web search when prompts need external lookups. "
            "Adds latency and cost"
        ),
    )
    parser.add_argument(
        "--allow-prompt-logging",
        action="store_true",
        default=False,
        help="Include raw prompt text in debug logs",
    )
    parser.add_argument(
        "--json-logs",
        action="store_true",
        help="Emit logs as structured JSON for container log scraping",
    )
    parser.add_argument(
        "--trace",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable per-request diagnostics/spans (one-shot generation; no content"
            " retries)"
        ),
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Fail on missing roles or mappings when enabled",
    )
    parser.add_argument(
        "--use-local-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable reading/writing the cache directory for mapping results. "
            "When disabled, cache options are ignored"
        ),
    )
    parser.add_argument(
        "--cache-mode",
        choices=("off", "read", "refresh", "write"),
        default=None,
        help=(
            "Caching behaviour (default 'read'): 'off' disables caching; 'read' uses"
            " existing entries and writes only on miss; 'refresh' refetches and"
            " overwrites; 'write' does not read and only writes on miss"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help=f"Directory to store cache files; defaults to '{DEFAULT_CACHE_DIR}'",
    )
    parser.add_argument(
        "--temp-output-dir",
        type=str,
        default=None,
        help="Directory for intermediate JSON records",
    )

    return parser


def _add_map_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    common: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Create the ``map`` subcommand parser."""
    parser = subparsers.add_parser(
        "map",
        parents=[common],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Populate feature mappings",
        description="Populate feature mappings for an existing features file",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="features.jsonl",
        help="Path to the features JSONL file",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="mapped.jsonl",
        help=OUTPUT_FILE_HELP,
    )
    parser.set_defaults(func=_cmd_map)
    return parser


def _add_run_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    common: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Create the ``run`` subcommand parser."""
    parser = subparsers.add_parser(
        "run",
        parents=[common],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Generate service evolutions (one model call per plateau)",
        description=(
            "Generate service evolutions using the ProcessingEngine (one-shot per"
            " plateau; no orchestration retries of model content). Use --cache-mode"
            " refresh to force fresh calls, --concurrency to adjust parallelism, and"
            " --trace for per-request diagnostics."
        ),
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="services.jsonl",
        help=SERVICES_FILE_HELP,
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="evolutions.jsonl",
        help=OUTPUT_FILE_HELP,
    )
    parser.add_argument(
        "--transcripts-dir", type=str, default=None, help=TRANSCRIPTS_HELP
    )
    parser.set_defaults(func=_cmd_run)
    return parser


def _add_validate_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    common: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Create the ``validate`` subcommand parser."""
    parser = subparsers.add_parser(
        "validate",
        parents=[common],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Validate inputs (one model call per plateau when enabled)",
        description=(
            "Validate inputs without calling the API by default. When used to generate,"
            " the flow is one-shot per plateau with no orchestration content retries."
            " Use --cache-mode refresh to force fresh calls, --concurrency to adjust"
            " parallelism, and --trace for diagnostics."
        ),
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="services.jsonl",
        help=SERVICES_FILE_HELP,
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="evolutions.jsonl",
        help=OUTPUT_FILE_HELP,
    )
    parser.add_argument(
        "--transcripts-dir", type=str, default=None, help=TRANSCRIPTS_HELP
    )
    parser.add_argument(
        "--data",
        help=(
            "Directory containing services.jsonl and an optional catalogue "
            "subdirectory for standalone validation"
        ),
    )
    parser.set_defaults(func=_cmd_validate)
    return parser


def _add_reverse_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    common: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Create the ``reverse`` subcommand parser."""
    parser = subparsers.add_parser(
        "reverse",
        parents=[common],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Rebuild caches from an evolutions file",
        description=(
            "Reconstruct feature and mapping caches from a previously "
            "generated evolutions.jsonl"
        ),
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="evolutions.jsonl",
        help="Path to the evolutions JSONL file",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="features.jsonl",
        help="Path to write extracted features JSONL",
    )
    parser.set_defaults(func=_cmd_reverse)
    return parser


def _build_parser() -> argparse.ArgumentParser:
    """Return an argument parser configured with subcommands."""
    parser = argparse.ArgumentParser(
        description=(
            "Service evolution utilities backed by a layered runtime "
            "architecture. A ProcessingEngine coordinates ServiceExecution "
            "and PlateauRuntime instances and relies on a global RuntimeEnv "
            "for configuration."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the service-ambitions version and exit.",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Print environment diagnostics and exit.",
    )
    common = _add_common_args(
        argparse.ArgumentParser(
            add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    )
    subparsers = parser.add_subparsers(dest="command")
    _add_map_subparser(subparsers, common)
    _add_reverse_subparser(subparsers, common)
    _add_run_subparser(subparsers, common)
    _add_validate_subparser(subparsers, common)
    return parser


def _update_stage_models(args: argparse.Namespace, settings: Settings) -> None:
    """Apply per-stage model overrides from CLI arguments."""
    stage_models = settings.models or StageModels(
        descriptions=None,
        features=None,
        mapping=None,
        search=None,
    )
    stage_mapping = {
        "descriptions_model": "descriptions",
        "features_model": "features",
        "mapping_model": "mapping",
        "search_model": "search",
    }
    for arg_name, field in stage_mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:  # branch: update the requested stage model
            setattr(stage_models, field, value)
    settings.models = stage_models


def _apply_args_to_settings(args: argparse.Namespace, settings: Settings) -> None:
    """Override settings fields based on CLI arguments."""
    _update_stage_models(args, settings)
    arg_mapping: dict[str, tuple[str, Callable[[Any], Any] | None]] = {
        "model": ("model", None),
        "concurrency": ("concurrency", None),
        "strict_mapping": ("strict_mapping", None),
        "mapping_data_dir": ("mapping_data_dir", Path),
        "roles_file": ("roles_file", Path),
        "web_search": ("web_search", None),
        "use_local_cache": ("use_local_cache", None),
        "cache_mode": ("cache_mode", None),
        "cache_dir": ("cache_dir", Path),
        "strict": ("strict", None),
        "trace": ("diagnostics", None),
        "dry_run": ("dry_run", None),
    }
    for arg_name, (attr, converter) in arg_mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:  # branch: override settings when flag provided
            setattr(settings, attr, converter(value) if converter else value)
    if not hasattr(settings, "mapping_mode"):
        settings.mapping_mode = "per_set"


def _execute_subcommand(args: argparse.Namespace, settings: Settings) -> None:
    """Initialise runtime and dispatch to the chosen subcommand."""
    RuntimeEnv.initialize(settings)
    if args.seed is not None:
        random.seed(args.seed)
    _configure_logging(args, settings)
    telemetry.reset()
    result = args.func(args, None)
    if inspect.isawaitable(result):
        asyncio.run(cast(Coroutine[Any, Any, Any], result))
    telemetry.print_summary()
    logfire.force_flush()
    if settings.strict_mapping and telemetry.has_quarantines():
        raise SystemExit(1)


def main() -> None:
    """Parse arguments and dispatch to the requested subcommand."""
    parser = _build_parser()
    args = parser.parse_args()
    if args.version:
        _print_version()
        return
    if args.diagnostics:
        _print_diagnostics()
        return
    if args.command is None:
        parser.print_help()
        raise SystemExit(1)
    if args.command == "validate" and getattr(args, "data", None):
        validate_data_dir(Path(args.data))
        return
    settings = load_settings(args.config)
    _apply_args_to_settings(args, settings)
    _execute_subcommand(args, settings)


if __name__ == "__main__":
    # Allow module to be executed as a standalone script
    main()
