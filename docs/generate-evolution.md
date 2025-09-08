# Generate evolution

The evolution workflow spans the plateaus defined in
`data/service_feature_plateaus.json`. A `ProcessingEngine` coordinates the
process: it creates a `ServiceRuntime` for each service, invokes a
`ServiceExecution` to populate it and spawns a `PlateauRuntime` per plateau.
These engines lazily load data, cache intermediate results and only flush output
once all stages succeed. The CLI evaluates all plateaus in this file alongside
all roles defined in `data/roles.json`.
Plateau name to level mappings are derived from the order of the JSON entries.

Policy: one‑shot per plateau

- Each plateau is generated with a single model call. There are no orchestration
  "top‑ups" or content retries. Only transport‑level retries (timeouts,
  connection errors, rate limits) may occur with bounded backoff.

Runtime configuration and shared state live in the thread‑safe `RuntimeEnv`
singleton initialised by the CLI. Modules access settings via
`RuntimeEnv.instance().settings`, avoiding repeated file reads.

To re-run mapping on an existing evolution output, see
[generate-mapping](generate-mapping.md).

## Running

Example command:

```bash
poetry run service-ambitions generate-evolution \
  --input-file sample-services.jsonl \
  --output-file evolution.jsonl \
  --strict-mapping
```

`--mapping-data-dir` points to a directory of mapping reference data.
`--strict-mapping/--no-strict-mapping` fails when feature mappings are missing or
contain unknown identifiers.
Telemetry via Logfire is always enabled. Prompt text is hidden from logs unless
`--allow-prompt-logging` is passed.
Use `--roles-file` to supply an alternative roles definition file when needed.

Force fresh calls / control caching and diagnostics

- `--cache-mode refresh` forces fresh calls (ignoring existing cache entries).
- `--cache-mode off` disables cache reads/writes.
- `--concurrency` tunes parallelism across services.
- `--trace` enables per‑request diagnostics and spans.

Logfire is required by the CLI but the `SA_LOGFIRE_TOKEN` environment variable is
optional for local runs. Set the token to stream traces to Logfire; without it,
telemetry remains local.

Pass `--strict` to abort if any role lacks features or if generated features
contain empty mapping lists. This turns on a fail-fast mode instead of the
default best‑effort behaviour.

Services run concurrently using a bounded worker pool configured via the
`--concurrency` flag or the `concurrency` setting in `config/app.yaml`.

Include `--seed <value>` to make backoff jitter and model sampling
deterministic when supported by the provider.

### Model selection

Defaults come from `config/app.yaml`:

| Stage        | Default model                  | Fast/cheap alternative |
| ------------ | ------------------------------ | ---------------------- |
| Descriptions | `openai:o4-mini`               | Already cost optimised |
| Features     | `openai:gpt-5`                 | `openai:o4-mini`       |
| Mapping      | `openai:o4-mini`               | Already cost optimised |
| Search       | `openai:gpt-4o-search-preview` | n/a                    |

OpenAI advises using lower‑capacity models like `o4-mini` for budget‑sensitive
workloads and reserving larger models such as `gpt-5` for highest quality
results. See the [model selection guide](https://platform.openai.com/docs/guides/model-selection)
for more guidance.

### When to enable web search

Pass `--web-search` when prompts require external lookups, such as verifying
recent facts or sourcing current statistics. The preview tool adds latency and
cost, so leave it disabled for self‑contained tasks.

## Output schema

Each line in the output file is a JSON object with:

```json
{
  "meta": {
    "schema_version": "1.0",
    "run_id": "20240101-000000",
    "seed": 1234,
    "models": {
      "descriptions": "openai:o4-mini",
      "features": "openai:gpt-5",
      "mapping": "openai:o4-mini",
      "search": "openai:gpt-4o-search-preview"
    },
    "web_search": false,
    "created": "2024-01-01T00:00:00Z"
  },
  "service": {
    "service_id": "string",
    "name": "string",
    "description": "string",
    "customer_type": "string",
    "jobs_to_be_done": [{ "name": "string" }]
  },
  "plateaus": [
    {
      "plateau": 1,
      "plateau_name": "string",
      "service_description": "string",
      "features": [
        {
          "feature_id": "ABC123",
          "name": "string",
          "description": "string",
          "score": {
            "level": 3,
            "label": "Defined",
            "justification": "string"
          },
          "customer_type": "string",
          "mappings": {
            "information": [{ "item": "string" }],
            "applications": [{ "item": "string" }],
            "technologies": [{ "item": "string" }]
          }
        }
      ]
    }
  ]
}
```

`run_id` remains constant for every record produced in a single invocation. The
`created` value uses the ISO-8601 timestamp format.

The conversation seed for each service includes the `service_id` and
`jobs_to_be_done` list so that all plateau calls share the same context.

## Testing

Run project checks before committing:

```bash
black .
ruff .
mypy .
bandit -r src -ll
pip-audit
```

---

This documentation is licensed under the [MIT License](../LICENSE).
