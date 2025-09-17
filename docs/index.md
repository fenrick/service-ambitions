# Service Ambitions

Welcome to the Service Ambitions documentation. This site contains standards,
architecture notes, and guides for using the CLI tools in this repository.

- Start with Standards to understand coding, linting, typing, and CI expectations.
- See Architecture for an overview of runtime design and data flows.
- Use the Generation guides to run evolutions and mappings.

## Quickstart

Run preflight checks to validate your environment and data paths:

```bash
poetry run service-ambitions preflight --config config/app.yaml
```

Generate evolutions:

```bash
poetry run service-ambitions run \
  --input-file sample-services.jsonl \
  --output-file evolutions.jsonl
```

Remap features using the mapping service:

```bash
poetry run service-ambitions map \
  --input-file evolutions.jsonl \
  --output-file remapped.jsonl
```

## Facets

Facets let you capture and validate the qualities of a mapping relationship
(applications, technologies, data). They are configured per dataset and
enforced dynamically at runtime. See `docs/facets.md` for defaults and usage.

## Mapping CLI (input/output formats)

Input (JSONL; one record per service). Two accepted shapes:

- Full `ServiceEvolution` record (with or without `plateaus`).
- Features+service record with:
  - `service`: minimally `service_id`, `name`, `description` (include
    `jobs_to_be_done` for better context; strings or objects allowed).
  - `features`: list of features with `feature_id`, `name`, `description`,
    `score` (level, label, justification), and `customer_type`.

Example (features+service):

```json
{"service": {"service_id": "svc-001", "name": "Admissions", "description": "Handles student applications."},
 "features": [
   {"feature_id": "ABCDEF", "name": "Online application form", "description": "...", "score": {"level": 2, "label": "Managed", "justification": "..."}, "customer_type": "learner"}
 ]}
```

Output (JSONL): Canonical `ServiceEvolution` per service with `plateaus[]` and
`mappings{}` populated. Each contribution may include a `facets` object,
depending on the dataset’s `facets` schema. See `docs/facets.md`.

### Credentials and caching for mapping-only runs

The mapping CLI reuses the same Pydantic AI agent stack as the full pipeline.
Set `SA_OPENAI_API_KEY` (or configure an equivalent provider key) before running
`service-ambitions map`; the command will fail fast if credentials are missing.

The `--use-local-cache` and `--cache-mode` flags work exactly as they do for the
`run` workflow. Cache entries are keyed by prompt+history, so mapping-only runs
benefit from the same deterministic retrieval when `cache_mode` remains `read`
or `write`.

## Enable the LLM queue

Enable the global queue to overlap feature generation with mapping:

```yaml
llm_queue_enabled: true
llm_queue_concurrency: 3
```

These settings live in `config/app.yaml` and can be overridden with
`SA_LLM_QUEUE_ENABLED` and `SA_LLM_QUEUE_CONCURRENCY` environment
variables. Monitor the `sa_llm_queue_inflight`, `sa_llm_queue_submitted`
and `sa_llm_queue_completed` metrics to verify operation. See
[LLM Queue](llm-queue.md) for tuning and troubleshooting guidance.

For the full project overview, see the README on GitHub:

https://github.com/FromHereOnAU/service-ambitions#readme

## Reference material for agents and maintainers

To stay aligned with upstream behaviour, review the canonical framework
exports before changing retries, usage tracking, or telemetry wiring:

- Pydantic AI LLM guide (plain-text export): https://ai.pydantic.dev/llms-full.txt
- Core Pydantic documentation (plain-text export): https://docs.pydantic.dev/latest/llms-full.txt
- Logfire instrumentation reference (plain-text export): https://logfire.pydantic.dev/docs/llms-full.txt

For offline work and deterministic builds, cached snapshots live in
`docs/external/` and are named `<library>-llms-2025-09-17.txt` (e.g.
`docs/external/pydantic-ai-llms-2025-09-17.txt`). Refresh these files whenever
you need a newer snapshot and update the filename date to match.
