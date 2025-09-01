# SPDX-License-Identifier: MIT
"""Command-line interface for generating service evolutions."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import random
from pathlib import Path
from typing import Any, Coroutine, cast

import logfire
from pydantic_core import to_json

import mapping
import telemetry
from canonical import canonicalise_record
from cli_mapping import load_catalogue, remap_features, write_output
from engine.processing_engine import ProcessingEngine
from loader import configure_mapping_data_dir, load_mapping_items
from models import (
    Contribution,
    FeatureItem,
    MappingFeature,
    MappingResponse,
    PlateauFeaturesResponse,
    ServiceEvolution,
    StageModels,
)
from monitoring import init_logfire
from plateau_generator import _feature_cache_path
from runtime.environment import RuntimeEnv
from settings import load_settings

SERVICES_FILE_HELP = "Path to the services JSONL file"
OUTPUT_FILE_HELP = "File to write the results"
TRANSCRIPTS_HELP = (
    "Directory to store per-service request/response transcripts. "
    "Defaults to a '_transcripts' folder beside the output file."
)


LOG_LEVELS = ["fatal", "error", "warn", "notice", "info", "debug", "trace"]


def _configure_logging(args: argparse.Namespace, settings) -> None:
    """Configure Logfire based on verbosity flags."""

    index = 2 + args.verbose - args.quiet
    index = max(0, min(len(LOG_LEVELS) - 1, index))
    min_log_level = LOG_LEVELS[index]
    init_logfire(settings.logfire_token, min_log_level)


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


def _cmd_reverse(args: argparse.Namespace, transcripts_dir: Path | None) -> None:
    """Backfill feature and mapping caches from ``evolutions.jsonl``."""

    settings = RuntimeEnv.instance().settings
    configure_mapping_data_dir(args.mapping_data_dir or settings.mapping_data_dir)
    items, catalogue_hash = load_mapping_items(settings.mapping_sets)

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
            for plateau in evo.plateaus:
                # Reconstruct feature cache grouped by role
                block: dict[str, list[FeatureItem]] = {}
                for feat in plateau.features:
                    item = FeatureItem(
                        name=feat.name,
                        description=feat.description,
                        score=feat.score,
                    )
                    block.setdefault(feat.customer_type, []).append(item)
                feat_payload = PlateauFeaturesResponse(features=block)
                feat_path = _feature_cache_path(svc_id, plateau.plateau)
                mapping.cache_write_json_atomic(feat_path, feat_payload.model_dump())

                # Rebuild mapping cache per mapping set
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
                    key = mapping._build_cache_key(
                        settings.model,
                        cfg.field,
                        catalogue_hash,
                        plateau.features,
                        settings.diagnostics,
                    )
                    cache_file = mapping._cache_path(
                        svc_id, plateau.plateau, cfg.field, key
                    )
                    map_payload = MappingResponse(
                        features=list(features_by_id.values())
                    )
                    mapping.cache_write_json_atomic(
                        cache_file, map_payload.model_dump()
                    )

                plateau.mappings = {}

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
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--model",
        help=(
            "Global chat model name (default openai:gpt-5). "
            "Can also be set via the MODEL env variable."
        ),
    )
    common.add_argument(
        "--descriptions-model",
        help="Model for plateau descriptions (default openai:o4-mini)",
    )
    common.add_argument(
        "--features-model",
        help=(
            "Model for feature generation (default openai:gpt-5; "
            "use openai:o4-mini for lower cost)"
        ),
    )
    common.add_argument(
        "--mapping-model",
        help="Model for feature mapping (default openai:o4-mini)",
    )
    common.add_argument(
        "--search-model",
        help="Model for web search (default openai:gpt-4o-search-preview)",
    )
    common.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help=(
            "Increase logging verbosity (-v notice, -vv info, -vvv debug, -vvvv trace)"
        ),
    )
    common.add_argument(
        "-q",
        "--quiet",
        action="count",
        default=0,
        help="Decrease logging verbosity (-q error, -qq fatal)",
    )
    common.add_argument(
        "--concurrency",
        type=int,
        help="Number of services to process concurrently",
    )
    common.add_argument(
        "--max-services",
        type=int,
        help="Process at most this many services",
    )
    common.add_argument(
        "--strict-mapping",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Fail when feature mappings are missing",
    )
    common.add_argument(
        "--mapping-data-dir",
        default=None,
        help="Directory containing mapping reference data",
    )
    common.add_argument(
        "--roles-file",
        default="data/roles.json",
        help="Path to JSON file containing role identifiers",
    )
    common.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed random number generation for reproducible output",
    )
    common.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs without calling the API",
    )
    common.add_argument(
        "--progress",
        action="store_true",
        help="Display a progress bar during execution",
    )
    common.add_argument(
        "--continue",
        dest="resume",
        action="store_true",
        help="Resume processing using processed_ids.txt",
    )
    common.add_argument(
        "--web-search",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable web search when prompts need external lookups. "
            "Adds latency and cost"
        ),
    )
    common.add_argument(
        "--allow-prompt-logging",
        action="store_true",
        help="Include raw prompt text in debug logs",
    )
    common.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Fail on missing roles or mappings when enabled",
    )
    common.add_argument(
        "--use-local-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable reading/writing the cache directory for mapping results. "
            "When disabled, cache options are ignored"
        ),
    )
    common.add_argument(
        "--cache-mode",
        choices=("off", "read", "refresh", "write"),
        default=None,
        help=(
            "Caching behaviour (default 'read'): 'off' disables caching, "
            "'read' uses existing entries without writing, 'refresh' "
            "refetches and overwrites cache entries, and 'write' reads and "
            "writes to the cache"
        ),
    )
    common.add_argument(
        "--cache-dir",
        default=None,
        help=(
            "Directory to store cache files; defaults to '.cache' in the "
            "current working directory"
        ),
    )
    common.add_argument(
        "--temp-output-dir",
        help="Directory for intermediate JSON records",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    map_p = subparsers.add_parser(
        "map",
        parents=[common],
        help="Populate feature mappings",
        description="Populate feature mappings for an existing features file",
    )
    map_p.add_argument(
        "--input-file",
        default="features.jsonl",
        help="Path to the features JSONL file",
    )
    map_p.add_argument(
        "--output-file",
        default="mapped.jsonl",
        help=OUTPUT_FILE_HELP,
    )
    map_p.set_defaults(func=_cmd_map)

    rev_p = subparsers.add_parser(
        "reverse",
        parents=[common],
        help="Rebuild caches from an evolutions file",
        description=(
            "Reconstruct feature and mapping caches from a previously "
            "generated evolutions.jsonl"
        ),
    )
    rev_p.add_argument(
        "--input-file",
        default="evolutions.jsonl",
        help="Path to the evolutions JSONL file",
    )
    rev_p.add_argument(
        "--output-file",
        default="features.jsonl",
        help="Path to write extracted features JSONL",
    )
    rev_p.set_defaults(func=_cmd_reverse)

    run_p = subparsers.add_parser(
        "run",
        parents=[common],
        help="Generate service evolutions via ProcessingEngine",
        description=(
            "Generate service evolutions using the ProcessingEngine and "
            "RuntimeEnv architecture"
        ),
    )
    run_p.add_argument(
        "--input-file",
        default="services.jsonl",
        help=SERVICES_FILE_HELP,
    )
    run_p.add_argument(
        "--output-file",
        default="evolutions.jsonl",
        help=OUTPUT_FILE_HELP,
    )
    run_p.add_argument("--transcripts-dir", help=TRANSCRIPTS_HELP)
    run_p.set_defaults(func=_cmd_run)

    val_p = subparsers.add_parser(
        "validate",
        parents=[common],
        help="Generate service evolutions via ProcessingEngine",
        description=(
            "Validate inputs without calling the API and generate service "
            "evolutions using the ProcessingEngine runtime"
        ),
    )
    val_p.add_argument(
        "--input-file",
        default="services.jsonl",
        help=SERVICES_FILE_HELP,
    )
    val_p.add_argument(
        "--output-file",
        default="evolutions.jsonl",
        help=OUTPUT_FILE_HELP,
    )
    val_p.add_argument("--transcripts-dir", help=TRANSCRIPTS_HELP)
    val_p.set_defaults(func=_cmd_validate)

    return parser


def _apply_args_to_settings(args: argparse.Namespace, settings) -> None:
    """Override settings fields based on CLI arguments."""

    if args.model is not None:
        settings.model = args.model
    stage_models = settings.models or StageModels(
        descriptions=None,
        features=None,
        mapping=None,
        search=None,
    )
    if args.descriptions_model is not None:
        stage_models.descriptions = args.descriptions_model
    if args.features_model is not None:
        stage_models.features = args.features_model
    if args.mapping_model is not None:
        stage_models.mapping = args.mapping_model
    if args.search_model is not None:
        stage_models.search = args.search_model
    settings.models = stage_models
    if args.concurrency is not None:
        settings.concurrency = args.concurrency
    if args.strict_mapping is not None:
        settings.strict_mapping = args.strict_mapping
    if args.mapping_data_dir is not None:
        settings.mapping_data_dir = Path(args.mapping_data_dir)
    if args.web_search is not None:
        settings.web_search = args.web_search
    if args.use_local_cache is not None:
        settings.use_local_cache = args.use_local_cache
    if args.cache_mode is not None:
        settings.cache_mode = args.cache_mode
    if args.cache_dir is not None:
        settings.cache_dir = Path(args.cache_dir)
    if args.strict is not None:
        settings.strict = args.strict
    if not hasattr(settings, "mapping_mode"):
        settings.mapping_mode = "per_set"


def _execute_subcommand(args: argparse.Namespace, settings) -> None:
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
    settings = load_settings()
    _apply_args_to_settings(args, settings)
    _execute_subcommand(args, settings)


if __name__ == "__main__":
    # Allow module to be executed as a standalone script
    main()
