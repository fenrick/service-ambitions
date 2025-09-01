# SPDX-License-Identifier: MIT
"""Command-line interface for generating service evolutions."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import logging
import random
from pathlib import Path
from typing import Any, Coroutine, cast

import logfire
from pydantic_core import to_json

import loader
import mapping
import telemetry
from canonical import canonicalise_record
from conversation import ConversationSession
from engine.processing_engine import ProcessingEngine
from loader import configure_mapping_data_dir, load_mapping_items
from models import FeatureMappingRef, MappingFeatureGroup, ServiceEvolution
from monitoring import LOG_FILE_NAME, init_logfire
from runtime.environment import RuntimeEnv
from settings import load_settings

SERVICES_FILE_HELP = "Path to the services JSONL file"
OUTPUT_FILE_HELP = "File to write the results"
TRANSCRIPTS_HELP = (
    "Directory to store per-service request/response transcripts. "
    "Defaults to a '_transcripts' folder beside the output file."
)


def _configure_logging(args: argparse.Namespace, settings) -> None:
    """Configure the logging subsystem."""

    # CLI-specified level takes precedence over configured default
    level_name = settings.log_level

    if args.verbose == 1:
        # Single -v flag bumps log level to INFO for clearer output
        level_name = "INFO"
    elif args.verbose >= 2:
        # Two or more -v flags enable DEBUG for deep troubleshooting
        level_name = "DEBUG"

    if args.no_logs:
        # Disable file logging and telemetry when requested
        logging.basicConfig(
            level=getattr(logging, level_name.upper(), logging.INFO),
            force=True,
        )
        return

    logging.basicConfig(
        filename=LOG_FILE_NAME,
        level=getattr(logging, level_name.upper(), logging.INFO),
        force=True,
    )
    # Initialize logfire regardless of token availability; a missing token
    # keeps logging local without sending telemetry to the cloud.
    init_logfire(settings.logfire_token, diagnostics=settings.diagnostics)


async def _cmd_run(
    args: argparse.Namespace, settings, transcripts_dir: Path | None
) -> None:
    """Execute the default evolution generation workflow."""

    await _cmd_generate_evolution(args, settings, transcripts_dir)


async def _cmd_diagnose(
    args: argparse.Namespace, settings, transcripts_dir: Path | None
) -> None:
    """Run the generator with diagnostics and transcripts enabled."""

    settings.diagnostics = True
    args.no_logs = False
    await _cmd_generate_evolution(args, settings, transcripts_dir)


async def _cmd_validate(
    args: argparse.Namespace, settings, transcripts_dir: Path | None
) -> None:
    """Validate inputs without invoking the language model."""

    args.dry_run = True
    await _cmd_generate_evolution(args, settings, transcripts_dir)


async def _cmd_map(
    args: argparse.Namespace, settings, transcripts_dir: Path | None
) -> None:
    """Populate feature mappings for an existing features file."""

    configure_mapping_data_dir(args.mapping_data_dir or settings.mapping_data_dir)
    items, catalogue_hash = load_mapping_items(
        loader.MAPPING_DATA_DIR, settings.mapping_sets
    )

    text = Path(args.input_file).read_text(encoding="utf-8")
    evolutions = [
        ServiceEvolution.model_validate_json(line)
        for line in text.splitlines()
        if line.strip()
    ]

    features = [f for evo in evolutions for p in evo.plateaus for f in p.features]
    mapped = features
    for cfg in settings.mapping_sets:
        mapped = await mapping.map_set(
            cast(ConversationSession, object()),
            cfg.field,
            items[cfg.field],
            mapped,
            service_name="svc",
            service_description="desc",
            plateau=1,
            strict=settings.strict_mapping,
            diagnostics=settings.diagnostics,
            cache_mode=args.cache_mode,
            catalogue_hash=catalogue_hash,
        )

    catalogue_lookup = {
        cfg.field: {item.id: item.name for item in items[cfg.field]}
        for cfg in settings.mapping_sets
    }
    by_id = {f.feature_id: f for f in mapped}
    for evo in evolutions:
        for plateau in evo.plateaus:
            mapped_feats = [by_id[f.feature_id] for f in plateau.features]
            plateau.mappings = {}
            for cfg in settings.mapping_sets:
                groups: dict[str, list[FeatureMappingRef]] = {}
                for feat in mapped_feats:
                    for contrib in feat.mappings.get(cfg.field, []):
                        groups.setdefault(contrib.item, []).append(
                            FeatureMappingRef(
                                feature_id=feat.feature_id,
                                description=feat.description,
                            )
                        )
                catalogue = catalogue_lookup[cfg.field]
                plateau.mappings[cfg.field] = [
                    MappingFeatureGroup(
                        id=item_id,
                        name=catalogue.get(item_id, item_id),
                        mappings=sorted(refs, key=lambda r: r.feature_id),
                    )
                    for item_id, refs in sorted(groups.items())
                ]

    output_path = Path(args.output_file)
    with output_path.open("w", encoding="utf-8") as fh:
        for evo in evolutions:
            record = canonicalise_record(evo.model_dump(mode="json"))
            fh.write(to_json(record).decode() + "\n")


async def _cmd_generate_evolution(
    args: argparse.Namespace, settings, transcripts_dir: Path | None
) -> None:
    """Generate service evolution summaries via ``ProcessingEngine``."""

    engine = ProcessingEngine(args, settings, transcripts_dir)
    success = await engine.run()
    await engine.finalise()
    if not success:
        logfire.warning("One or more services failed during processing")


def main() -> None:
    """Parse arguments and dispatch to the requested subcommand."""

    parser = argparse.ArgumentParser(
        description="Service evolution utilities",
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
        help="Increase logging verbosity (-v for info, -vv for debug)",
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
        "--diagnostics",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable verbose diagnostics output",
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
        help="Include raw prompt text in debug logs when diagnostics are enabled",
    )
    common.add_argument(
        "--no-logs",
        action="store_true",
        help="Disable file logging and Logfire telemetry",
    )
    common.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail on missing roles or mappings when enabled",
    )
    common.add_argument(
        "--use-local-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable reading/writing the cache directory for mapping results. "
            "When disabled, cache options are ignored"
        ),
    )
    common.add_argument(
        "--cache-mode",
        choices=("off", "read", "refresh", "write"),
        default="read",
        help=(
            "Caching behaviour (default 'read'): 'off' disables caching, "
            "'read' uses existing entries without writing, 'refresh' "
            "refetches and overwrites cache entries, and 'write' reads and "
            "writes to the cache"
        ),
    )
    common.add_argument(
        "--cache-dir",
        default=".cache",
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

    run_p = subparsers.add_parser(
        "run",
        parents=[common],
        help="Generate service evolutions",
        description="Generate service evolutions",
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

    diag_p = subparsers.add_parser(
        "diagnose",
        parents=[common],
        help="Generate service evolutions",
        description="Generate service evolutions with diagnostics enabled",
    )
    diag_p.add_argument(
        "--input-file",
        default="services.jsonl",
        help=SERVICES_FILE_HELP,
    )
    diag_p.add_argument(
        "--output-file",
        default="evolutions.jsonl",
        help=OUTPUT_FILE_HELP,
    )
    diag_p.add_argument("--transcripts-dir", help=TRANSCRIPTS_HELP)
    diag_p.set_defaults(func=_cmd_diagnose)

    val_p = subparsers.add_parser(
        "validate",
        parents=[common],
        help="Generate service evolutions",
        description=(
            "Validate inputs without calling the API and generate service evolutions"
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

    args = parser.parse_args()

    settings = load_settings()
    RuntimeEnv.initialize(settings)

    if args.seed is not None:
        random.seed(args.seed)

    if args.diagnostics is not None:
        settings.diagnostics = args.diagnostics
    if args.strict_mapping is not None:
        settings.strict_mapping = args.strict_mapping
    if args.mapping_data_dir is not None:
        settings.mapping_data_dir = Path(args.mapping_data_dir)
    if not hasattr(settings, "mapping_mode"):
        settings.mapping_mode = "per_set"

    _configure_logging(args, settings)

    telemetry.reset()
    result = args.func(args, settings, None)
    if inspect.isawaitable(result):
        # Cast ensures that asyncio.run receives a proper Coroutine
        asyncio.run(cast(Coroutine[Any, Any, Any], result))

    telemetry.print_summary()
    logfire.force_flush()
    if settings.strict_mapping and telemetry.has_quarantines():
        raise SystemExit(1)


if __name__ == "__main__":
    # Allow module to be executed as a standalone script
    main()
