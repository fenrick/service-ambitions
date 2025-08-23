"""Command-line interface for generating service ambitions and evolutions."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import logging
import os
import random
import sys
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any, Coroutine, Sequence, cast
from uuid import uuid4

import logfire
from pydantic_ai import Agent
from tqdm import tqdm

import telemetry
from canonical import canonicalise_record
from conversation import ConversationSession
from diagnostics import validate_jsonl
from generator import AmbitionModel, ServiceAmbitionGenerator
from loader import (
    MAPPING_DATA_DIR,
    configure_mapping_data_dir,
    configure_prompt_dir,
    load_ambition_prompt,
    load_evolution_prompt,
    load_mapping_items,
    load_role_ids,
)
from mapping import set_quarantine_logger
from model_factory import ModelFactory
from models import ServiceEvolution, ServiceInput, ServiceMeta
from monitoring import LOG_FILE_NAME, init_logfire
from persistence import atomic_write, read_lines
from plateau_generator import PlateauGenerator
from schema_migration import migrate_record
from service_loader import load_services
from settings import load_settings

SERVICES_PROCESSED = logfire.metric_counter("services_processed")
EVOLUTIONS_GENERATED = logfire.metric_counter("evolutions_generated")
LINES_WRITTEN = logfire.metric_counter("lines_written")

_RUN_META: ServiceMeta | None = None

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


def _prepare_paths(output: Path, resume: bool) -> tuple[Path, Path]:
    """Return paths used for output and resume tracking."""

    part_path = output.with_suffix(
        output.suffix + ".tmp" if not resume else output.suffix + ".tmp.part"
    )
    processed_path = output.with_name("processed_ids.txt")
    return part_path, processed_path


def _load_resume_state(
    processed_path: Path, output_path: Path, resume: bool
) -> tuple[set[str], list[str]]:
    """Return previously processed IDs and existing output lines."""

    processed_ids = set(read_lines(processed_path)) if resume else set()
    existing_lines = read_lines(output_path) if resume else []
    return processed_ids, existing_lines


def _ensure_transcripts_dir(path: str | None, output: Path) -> Path:
    """Create and return the directory used to store transcripts."""

    transcripts_dir = Path(path) if path is not None else output.parent / "_transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    return transcripts_dir


def _load_services_list(
    input_file: str, max_services: int | None, processed_ids: set[str]
) -> list[ServiceInput]:
    """Return services filtered for ``processed_ids`` and ``max_services``."""

    with load_services(Path(input_file)) as svc_iter:
        if max_services is not None:
            svc_iter = islice(svc_iter, max_services)
        return [s for s in svc_iter if s.service_id not in processed_ids]


def _save_results(
    *,
    resume: bool,
    part_path: Path,
    output_path: Path,
    existing_lines: list[str],
    processed_ids: set[str],
    new_ids: set[str],
    processed_path: Path,
) -> set[str]:
    """Persist generated lines and update processed IDs."""

    if resume:
        new_lines = read_lines(part_path)
        atomic_write(output_path, [*existing_lines, *new_lines])
        part_path.unlink(missing_ok=True)
        processed_ids.update(new_ids)
    else:
        os.replace(part_path, output_path)
        processed_ids = new_ids
    atomic_write(processed_path, sorted(processed_ids))
    return processed_ids


async def _generate_evolution_for_service(
    service: ServiceInput,
    *,
    factory: ModelFactory,
    settings,
    args: argparse.Namespace,
    system_prompt: str,
    transcripts_dir: Path | None,
    role_ids: Sequence[str],
    lock: asyncio.Lock,
    output,
    new_ids: set[str],
) -> None:
    """Generate evolution for ``service`` and record results."""

    desc_name = factory.model_name(
        "descriptions", args.descriptions_model or args.model
    )
    feat_name = factory.model_name("features", args.features_model or args.model)
    map_name = factory.model_name("mapping", args.mapping_model or args.model)
    attrs = {
        "service_id": service.service_id,
        "service_name": service.name,
        "descriptions_model": desc_name,
        "features_model": feat_name,
        "mapping_model": map_name,
        "output_path": getattr(output, "name", ""),
    }
    with logfire.span("generate_evolution_for_service", attributes=attrs):
        try:
            SERVICES_PROCESSED.add(1)
            desc_model = factory.get(
                "descriptions", args.descriptions_model or args.model
            )
            feat_model = factory.get("features", args.features_model or args.model)
            map_model = factory.get("mapping", args.mapping_model or args.model)

            desc_agent = Agent(desc_model, instructions=system_prompt)
            feat_agent = Agent(feat_model, instructions=system_prompt)
            map_agent = Agent(map_model, instructions=system_prompt)

            desc_session = ConversationSession(
                desc_agent,
                stage="descriptions",
                diagnostics=settings.diagnostics,
                log_prompts=args.allow_prompt_logging,
                redact_prompts=True,
                transcripts_dir=transcripts_dir,
            )
            feat_session = ConversationSession(
                feat_agent,
                stage="features",
                diagnostics=settings.diagnostics,
                log_prompts=args.allow_prompt_logging,
                redact_prompts=True,
                transcripts_dir=transcripts_dir,
            )
            map_session = ConversationSession(
                map_agent,
                stage="mapping",
                diagnostics=settings.diagnostics,
                log_prompts=args.allow_prompt_logging,
                redact_prompts=True,
                transcripts_dir=transcripts_dir,
            )
            generator = PlateauGenerator(
                feat_session,
                required_count=settings.features_per_role,
                roles=role_ids,
                description_session=desc_session,
                mapping_session=map_session,
                strict=args.strict,
            )
            global _RUN_META
            if _RUN_META is None:
                models_map = {
                    "descriptions": desc_name,
                    "features": feat_name,
                    "mapping": map_name,
                    "search": factory.model_name(
                        "search", args.search_model or args.model
                    ),
                }
                items = load_mapping_items(MAPPING_DATA_DIR)
                serialised = json.dumps(
                    {
                        k: [i.model_dump(mode="json") for i in v]
                        for k, v in items.items()
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                )
                catalogue_hash = hashlib.sha256(serialised.encode("utf-8")).hexdigest()
                context_window = getattr(feat_model, "max_input_tokens", 0)
                _RUN_META = ServiceMeta(
                    run_id=str(uuid4()),
                    seed=args.seed,
                    models=models_map,
                    web_search=getattr(factory, "_web_search", False),
                    mapping_types=sorted(getattr(settings, "mapping_types", {}).keys()),
                    context_window=context_window,
                    diagnostics=settings.diagnostics,
                    catalogue_hash=catalogue_hash,
                    created=datetime.now(timezone.utc),
                )
            evolution = await generator.generate_service_evolution_async(
                service, transcripts_dir=transcripts_dir, meta=_RUN_META
            )
            record = canonicalise_record(evolution.model_dump(mode="python"))
            line = json.dumps(
                record, separators=(",", ":"), ensure_ascii=False, sort_keys=True
            )
            async with lock:
                await asyncio.to_thread(output.write, f"{line}\n")
                new_ids.add(service.service_id)
                EVOLUTIONS_GENERATED.add(1)
                LINES_WRITTEN.add(1)
            logfire.info(
                "Generated evolution",
                service_id=service.service_id,
                output_path=getattr(output, "name", ""),
            )
        except Exception as exc:  # noqa: BLE001
            quarantine_dir = Path("quarantine")
            await asyncio.to_thread(quarantine_dir.mkdir, parents=True, exist_ok=True)
            quarantine_file = quarantine_dir / f"{service.service_id}.json"
            await asyncio.to_thread(
                quarantine_file.write_text,
                service.model_dump_json(indent=2),
                encoding="utf-8",
            )
            logfire.exception(
                "Failed to generate evolution",
                service_id=service.service_id,
                error=str(exc),
                quarantine_file=str(quarantine_file),
            )


async def _cmd_generate_ambitions(
    args: argparse.Namespace, settings, transcripts_dir: Path | None
) -> None:
    """Generate service ambitions and write them to disk."""
    output_path = Path(args.output_file)
    attrs = {"output_path": str(output_path), "resume": args.resume}
    with logfire.span("cmd_generate_ambitions", attributes=attrs) as span:
        try:
            if args.validate_only:
                count = validate_jsonl(output_path, AmbitionModel)
                logfire.info(
                    "Validated output",
                    lines=count,
                    output_path=str(output_path),
                )
                return

            # Load prompt components from the configured directory
            configure_prompt_dir(settings.prompt_dir)
            system_prompt = load_ambition_prompt(
                settings.context_id, settings.inspiration
            )

            use_web_search = (
                args.web_search if args.web_search is not None else settings.web_search
            )
            factory = ModelFactory(
                settings.model,
                settings.openai_api_key,
                stage_models=getattr(settings, "models", None),
                reasoning=settings.reasoning,
                seed=args.seed,
                web_search=use_web_search,
            )

            model_name = factory.model_name(
                "features", args.features_model or args.model
            )
            span.set_attribute("model_name", model_name)
            logfire.info("Generating ambitions", model=model_name)
            model = factory.get("features", args.features_model or args.model)
            concurrency = args.concurrency or settings.concurrency
            generator = ServiceAmbitionGenerator(
                model,
                concurrency=concurrency,
                request_timeout=settings.request_timeout,
                retries=settings.retries,
                retry_base_delay=settings.retry_base_delay,
            )

            part_path, processed_path = _prepare_paths(output_path, args.resume)
            processed_ids, existing_lines = _load_resume_state(
                processed_path, output_path, args.resume
            )
            if transcripts_dir is None and not args.no_logs:
                transcripts_dir = _ensure_transcripts_dir(
                    args.transcripts_dir, output_path
                )
            services = _load_services_list(
                args.input_file,
                args.max_services,
                processed_ids if args.resume else set(),
            )

            if args.dry_run:
                logfire.info(
                    "Validated services",
                    count=len(services),
                    resume=args.resume,
                )
                return

            show_progress = args.progress and sys.stdout.isatty()
            progress = tqdm(total=len(services)) if show_progress else None
            new_ids = await generator.generate_async(
                services,
                system_prompt,
                str(part_path),
                progress=progress,
                transcripts_dir=transcripts_dir,
            )
            if progress:
                progress.close()

            SERVICES_PROCESSED.add(len(new_ids))
            LINES_WRITTEN.add(len(new_ids))

            processed_ids = _save_results(
                resume=args.resume,
                part_path=part_path,
                output_path=output_path,
                existing_lines=existing_lines,
                processed_ids=processed_ids,
                new_ids=new_ids,
                processed_path=processed_path,
            )
            logfire.info(
                "Results written",
                output_path=str(output_path),
                lines_written=len(new_ids),
                resume=args.resume,
            )
        except Exception as exc:  # noqa: BLE001
            logfire.exception(
                "Ambition generation failed",
                output_path=str(output_path),
                resume=args.resume,
                error=str(exc),
            )
            raise


async def _cmd_generate_evolution(
    args: argparse.Namespace, settings, transcripts_dir: Path | None
) -> None:
    """Generate service evolution summaries."""

    use_web_search = (
        args.web_search if args.web_search is not None else settings.web_search
    )
    factory = ModelFactory(
        settings.model,
        settings.openai_api_key,
        stage_models=getattr(settings, "models", None),
        reasoning=settings.reasoning,
        seed=args.seed,
        web_search=use_web_search,
    )

    configure_prompt_dir(settings.prompt_dir)
    if args.mapping_data_dir is None and not settings.diagnostics:
        raise RuntimeError("--mapping-data-dir is required in production mode")
    configure_mapping_data_dir(args.mapping_data_dir or settings.mapping_data_dir)
    system_prompt = load_evolution_prompt(settings.context_id, settings.inspiration)

    role_ids = load_role_ids(Path(args.roles_file))

    output_path = Path(args.output_file)
    part_path, processed_path = _prepare_paths(output_path, args.resume)
    processed_ids, existing_lines = _load_resume_state(
        processed_path, output_path, args.resume
    )
    if transcripts_dir is None and not args.no_logs:
        transcripts_dir = _ensure_transcripts_dir(args.transcripts_dir, output_path)
    services = _load_services_list(args.input_file, args.max_services, processed_ids)

    if args.dry_run:
        logfire.info(f"Validated {len(services)} services")
        return

    concurrency = (
        args.concurrency if args.concurrency is not None else settings.concurrency
    )
    if concurrency < 1:
        raise ValueError("concurrency must be a positive integer")
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    new_ids: set[str] = set()
    show_progress = args.progress and sys.stdout.isatty()
    progress = tqdm(total=len(services)) if show_progress else None

    async def run_one(service: ServiceInput) -> None:
        async with sem:
            await _generate_evolution_for_service(
                service,
                factory=factory,
                settings=settings,
                args=args,
                system_prompt=system_prompt,
                transcripts_dir=transcripts_dir,
                role_ids=role_ids,
                lock=lock,
                output=output,
                new_ids=new_ids,
            )
            if progress:
                progress.update(1)

    output = await asyncio.to_thread(part_path.open, "w", encoding="utf-8")
    try:
        async with asyncio.TaskGroup() as tg:
            for service in services:
                tg.create_task(run_one(service))
    finally:
        await asyncio.to_thread(output.close)
    if progress:
        progress.close()

    processed_ids = _save_results(
        resume=args.resume,
        part_path=part_path,
        output_path=output_path,
        existing_lines=existing_lines,
        processed_ids=processed_ids,
        new_ids=new_ids,
        processed_path=processed_path,
    )


async def _cmd_generate_mapping(args: argparse.Namespace, settings) -> None:
    """Augment evolution features with mapping results."""

    use_web_search = (
        args.web_search if args.web_search is not None else settings.web_search
    )
    factory = ModelFactory(
        settings.model,
        settings.openai_api_key,
        stage_models=getattr(settings, "models", None),
        reasoning=settings.reasoning,
        seed=args.seed,
        web_search=use_web_search,
    )

    map_model = factory.get("mapping", args.mapping_model or args.model)
    map_agent = Agent(map_model, instructions="")
    session = ConversationSession(
        map_agent,
        stage="mapping",
        diagnostics=settings.diagnostics,
        log_prompts=args.allow_prompt_logging,
        redact_prompts=True,
    )

    if args.mapping_data_dir is None and not settings.diagnostics:
        raise RuntimeError("--mapping-data-dir is required in production mode")
    configure_mapping_data_dir(args.mapping_data_dir or settings.mapping_data_dir)

    input_path = Path(args.input)
    lines = await asyncio.to_thread(input_path.read_text, encoding="utf-8")
    evolutions = [
        ServiceEvolution.model_validate_json(line)
        for line in lines.splitlines()
        if line.strip()
    ]

    features = [
        feat
        for evo in evolutions
        for plateau in evo.plateaus
        for feat in plateau.features
    ]
    strict_mapping = (
        args.strict_mapping
        if getattr(args, "strict_mapping", None) is not None
        else settings.strict_mapping
    )
    generator = PlateauGenerator(session, strict=strict_mapping)
    mapped = await generator._map_features(session, features)
    mapped_by_id = {f.feature_id: f for f in mapped}
    for evo in evolutions:
        for plateau in evo.plateaus:
            plateau.features = [mapped_by_id[f.feature_id] for f in plateau.features]

    output_path = Path(args.output)
    out = await asyncio.to_thread(output_path.open, "w", encoding="utf-8")
    try:
        for evo in evolutions:
            record = canonicalise_record(evo.model_dump(mode="python"))
            line = json.dumps(
                record, separators=(",", ":"), ensure_ascii=False, sort_keys=True
            )
            await asyncio.to_thread(out.write, f"{line}\n")
    finally:
        await asyncio.to_thread(out.close)


def _cmd_migrate_jsonl(args: argparse.Namespace, _settings) -> None:
    """Migrate records in a JSONL file between schema versions."""

    input_path = Path(args.input)
    output_path = Path(args.output)

    with (
        input_path.open("r", encoding="utf-8") as src,
        output_path.open("w", encoding="utf-8") as dst,
    ):
        for line in src:
            if not line.strip():
                # Skip empty lines to avoid json parser errors
                continue
            record = json.loads(line)
            migrated = migrate_record(args.from_version, args.to_version, record)
            dst.write(f"{json.dumps(migrated)}\n")


def main() -> None:
    """Parse arguments and dispatch to the requested subcommand."""

    settings = load_settings()

    parser = argparse.ArgumentParser(
        description="Service ambitions utilities",
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

    subparsers = parser.add_subparsers(dest="command", required=True)

    amb = subparsers.add_parser(
        "generate-ambitions",
        parents=[common],
        help="Generate service ambitions",
        description="Generate service ambitions",
    )
    amb.add_argument(
        "--input-file",
        default="sample-services.jsonl",
        help=SERVICES_FILE_HELP,
    )
    amb.add_argument(
        "--output-file",
        default="ambitions.jsonl",
        help=OUTPUT_FILE_HELP,
    )
    amb.add_argument("--transcripts-dir", help=TRANSCRIPTS_HELP)
    amb.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate an existing output file and exit",
    )
    amb.set_defaults(func=_cmd_generate_ambitions)

    evo = subparsers.add_parser(
        "generate-evolution",
        parents=[common],
        help="Generate service evolution",
    )
    evo.add_argument(
        "--input-file",
        default="sample-services.jsonl",
        help=SERVICES_FILE_HELP,
    )
    evo.add_argument(
        "--output-file",
        default="evolution.jsonl",
        help=OUTPUT_FILE_HELP,
    )
    evo.add_argument("--transcripts-dir", help=TRANSCRIPTS_HELP)
    evo.add_argument(
        "--roles-file",
        default="data/roles.json",
        help="Path to the roles definition JSON file",
    )
    evo.set_defaults(func=_cmd_generate_evolution)

    map_p = subparsers.add_parser(
        "generate-mapping",
        parents=[common],
        help="Generate feature mappings",
    )
    map_p.add_argument(
        "--input",
        default="evolution.jsonl",
        help="Path to the evolution JSONL file",
    )
    map_p.add_argument(
        "--output",
        default="mapped.jsonl",
        help=OUTPUT_FILE_HELP,
    )
    map_p.set_defaults(func=_cmd_generate_mapping)

    mig = subparsers.add_parser(
        "migrate-jsonl",
        help="Migrate JSONL records between schema versions",
    )
    mig.add_argument("--input", required=True, help="Path to the input JSONL file")
    mig.add_argument(
        "--output", required=True, help="File to write the migrated records"
    )
    mig.add_argument(
        "--from", dest="from_version", required=True, help="Source schema version"
    )
    mig.add_argument(
        "--to", dest="to_version", required=True, help="Target schema version"
    )
    mig.set_defaults(func=_cmd_migrate_jsonl)

    args = parser.parse_args()

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
    set_quarantine_logger(telemetry.record_quarantine)
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
