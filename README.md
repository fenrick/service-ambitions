# service-ambitions

## Configuration

The CLI requires an OpenAI API key available in the `OPENAI_API_KEY` environment
variable. Settings are loaded via Pydantic, which reads from a `.env` file if
present. The application will exit if the key is missing. LLM interactions are
handled via [Pydantic AI](https://pydantic.dev/pydantic-ai/).

Create a `.env` file in the project root with:

```
OPENAI_API_KEY=your_api_key_here
```

For production deployments, inject the variable using your platform's secret
manager instead of committing keys to source control.

The chat model can be set with the `--model` flag or the `MODEL` environment
variable. Model identifiers must include a provider prefix, in the form
`<provider>:<model>`. The default is `openai:gpt-5` with medium reasoning effort.

Stage-specific model overrides live under the `models` block in
`config/app.json`. Defaults are:

```json
{
  "descriptions": "openai:o4-mini",
  "features": "openai:gpt-5",
  "mapping": "openai:o4-mini",
  "search": "openai:gpt-4o-search-preview"
}
```

| Stage        | Default model                  | Fast/cheap alternative |
|--------------|--------------------------------|------------------------|
| Descriptions | `openai:o4-mini`               | Already cost optimised |
| Features     | `openai:gpt-5`                 | `openai:o4-mini`       |
| Mapping      | `openai:o4-mini`               | Already cost optimised |
| Search       | `openai:gpt-4o-search-preview` | n/a                    |

OpenAI recommends using cost‑optimised models like `o4-mini` when latency or
price is a concern and higher‑capacity models such as `gpt-5` for best quality
results. See the [model selection guide](https://platform.openai.com/docs/guides/model-selection)
for details.

OpenAI's experimental web search tool can be enabled via the `web_search`
setting or the `--web-search/--no-web-search` CLI flags. The sample
configuration leaves this disabled.

### When to enable web search

Enable web search only when prompts require external lookups, such as verifying
recent facts or gathering up‑to‑date statistics. The preview tool adds latency
and cost, so keep it disabled for self‑contained tasks.

System prompts are assembled from modular markdown components located in the
`prompts/` directory. Use `--prompt-dir` to point at an alternate component
directory, `--context-id` to select a situational context, and
`--inspirations-id` to choose a list of future inspirations. This structure
supports swapping sections to suit different industries.

This project depends on the [Pydantic Logfire](https://logfire.pydantic.dev/)
libraries even when telemetry is disabled. Set the `LOGFIRE_TOKEN` environment
variable and optionally supply a service name via `--logfire-service` to enable
tracing. When configured, the CLI instruments Pydantic, Pydantic AI, OpenAI and
system metrics automatically.

## Installation

Dependencies are managed with [Poetry](https://python-poetry.org/). Install the
tool and then install the project's dependencies with:

```bash
poetry install
```

For more accurate token estimation, include the optional `tiktoken` extra:

```bash
poetry install -E tiktoken
```

After installation the `service-ambitions` console script is available.
Run the CLI through Poetry to ensure it uses the managed environment. Use
subcommands to select the desired operation:

```bash
poetry run service-ambitions generate-ambitions --input-file sample-services.jsonl --output-file ambitions.jsonl
poetry run service-ambitions generate-evolution --input-file sample-services.jsonl --output-file evolution.jsonl
poetry run service-ambitions migrate-jsonl --from 1.0 --to 2.0 --input-file evolution.jsonl --output-file evolution_v2.jsonl
```

Alternatively, use the provided shell script which forwards all arguments to the CLI:

```bash
./run.sh generate-ambitions --input-file sample-services.jsonl --output-file ambitions.jsonl
./run.sh generate-evolution --input-file sample-services.jsonl --output-file evolution.jsonl
./run.sh migrate-jsonl --from 1.0 --to 2.0 --input-file evolution.jsonl --output-file evolution_v2.jsonl
```

## Usage

`sample-services.jsonl` contains example services in
[JSON Lines](https://jsonlines.org/) format, with one JSON object per line. The
output file will also be in JSON Lines format. Use `--concurrency` to control
parallel workers, `--max-services` to limit how many entries are processed, and
`--dry-run` to validate inputs without calling the API. Evolution processing
runs concurrently with a worker pool bounded by this setting. Pass `--progress`
to display a progress bar during long runs; it is suppressed automatically in
CI environments or when stdout is not a TTY. Provide `--seed` to make
stochastic behaviour such as backoff jitter deterministic during tests and
demos.

## Adaptive backpressure

The generator coordinates concurrent requests with an `AdaptiveSemaphore` so
token-heavy completions do not monopolise throughput. Each request estimates its
response size and reserves a weighted permit based on the
`expected_output_tokens` configuration. Larger outputs therefore reduce
available concurrency proportionally.

Configure the baseline token size by passing `expected_output_tokens` to
`Generator` or via the CLI flag `--expected-output-tokens`:

```python
generator = Generator(model, expected_output_tokens=512)
```

When an upstream service signals `Retry-After`, the semaphore halves the current
limit and then gradually restores capacity. The `ramp_interval` parameter
controls this ramp strategy. Consecutive throttling doubles the interval between
permit releases, implementing slow-start recovery until a grace period elapses.

```python
from service_ambitions.backpressure import AdaptiveSemaphore
import math

expected_output_tokens = 256
limiter = AdaptiveSemaphore(permits=5, ramp_interval=1.0)

token_estimate = 800
weight = math.ceil(token_estimate / expected_output_tokens)

async with limiter(weight):
    ...
```

## Plateau-first workflow

Each service is evaluated across the plateaus defined in
`data/service_feature_plateaus.json`. Generation occurs in two phases:

1. **Descriptions** – a single request returns narratives for all plateaus.
2. **Features and Mapping** – for each plateau, generate learner, academic and
   professional staff features and link them to reference Data, Applications
   and Technologies using the previously collected description.

This workflow issues one call to fetch all descriptions, then queries each
plateau separately for feature generation. The result is a complete
`ServiceEvolution` record for every service.

Plateau names and descriptions are sourced entirely from
`data/service_feature_plateaus.json`, allowing the progression to be
reconfigured without code changes. The CLI processes all entries from this
file.

### Generating service evolutions

Use the `generate-evolution` subcommand to score each service against all
configured plateau features and customer segments. It reads services from an
input JSON Lines file and writes a `ServiceEvolution` record for each line in
the output file. Logs are written to `service.log` in the current working
directory. Enable verbose logs with `-v` or `-vv`.

Basic invocation:

```bash
./run.sh generate-evolution --input-file sample-services.jsonl --output-file evolution.jsonl
```

Processing happens concurrently; control parallel workers with `--concurrency`
which defaults to the `concurrency` value in your settings.

Mapping requests are batched. Adjust `--mapping-batch-size` to control how many
features are sent per mapping request; the default is 30. Smaller batches create
shorter prompts and quicker responses but increase API calls. Larger batches
reduce round trips at the cost of bigger prompts, higher latency and a greater
risk of hitting model context limits.

For each batch the CLI maps Data, Applications and Technologies in parallel.
This behaviour is enabled by default via `--mapping-parallel-types`. Disable it
with `--no-mapping-parallel-types` to process mapping types sequentially when
rate limits are tight.

Example invocation tuning mapping behaviour:

```bash
./run.sh generate-evolution --input-file sample-services.jsonl \
  --output-file evolution.jsonl --mapping-batch-size 20 --no-mapping-parallel-types
```

### Conversation seed

The model conversation is seeded with service metadata so each request retains
context. The seed includes the service ID and jobs to be done:

```text
Service ID: S01
Service name: Learning & Teaching
Customer type: retail
Description: Delivers a holistic educational framework ...
Jobs to be done: Access an engaging curriculum, Continually update skills
```

`ConversationSession.add_parent_materials` assembles this seed before any
plateau requests are made.

## ServiceEvolution JSON schema

Each JSON line in the output file follows the `ServiceEvolution` schema:

```json
{
  "schema_version": "1.0",
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
          "feature_id": "string",
          "name": "string",
          "description": "string",
          "score": {
            "level": 3,
            "label": "Defined",
            "justification": "string"
          },
          "customer_type": "string",
          "data": [{ "item": "string", "contribution": 0.5 }],
          "applications": [{ "item": "string", "contribution": 0.5 }],
          "technology": [{ "item": "string", "contribution": 0.5 }]
        }
      ]
    }
  ]
}
```

Fields in the schema:

- `schema_version`: string identifying the output schema revision.
- `service`: `ServiceInput` with `service_id`, `name`, `description`, optional
  `customer_type`, `jobs_to_be_done`, and existing `features`.
- `plateaus`: list of `PlateauResult` entries, each containing:
  - `plateau`: integer plateau level.
  - `plateau_name`: descriptive plateau label.
  - `service_description`: narrative for the service at that plateau.
  - `features`: list of `PlateauFeature` entries with:
    - `feature_id`, `name`, and `description`.
    - `score`: object with CMMI maturity `level`, `label` and `justification`.
    - `customer_type`: audience benefiting from the feature.
    - `data`, `applications`, `technology`: lists of `Contribution` objects
      describing why a mapped item supports the feature.

## Reference Data

Feature mapping uses cached reference lists stored in the `data/` directory.
Each of `information.json`, `applications.json`, and `technologies.json`
contains items with identifiers, names, and descriptions. These lists are
injected into mapping prompts so that features can be matched against consistent
options. Mapping prompts run separately for information, applications and
technologies to keep each decision focused. All application configuration is
stored in `config/app.json`; the chat model and any reasoning parameters live
at the top level, while mapping types and their associated datasets live under
the `mapping_types` section, allowing new categories to be added without code
changes. Plateau definitions and their level mappings come from
`data/service_feature_plateaus.json`. Roles are defined in `data/roles.json`,
and the required number of features per role is controlled by
`features_per_role` in `config/app.json`.

## Prompt examples

### Plateau prompt

```markdown
# Plateau feature generation

Generate service features for the {service_name} service at plateau {plateau}.

## Instructions

- Use the service description: {service_description}.
- Return a single JSON object with keys for each role: {roles}.
- Each key must map to an array containing at least {required_count} feature
  objects.
- Every feature must provide:
  - "feature_id": unique string identifier.
  - "name": short feature title.
  - "description": explanation of the feature.
  - "score": object describing CMMI maturity with:
    - "level": integer 1–5.
    - "label": matching CMMI maturity name.
    - "justification": brief rationale for the level.
- Do not include any text outside the JSON object.
```

### Mapping prompt

```markdown
# Feature mapping

Map each feature to relevant Data, Applications and Technologies from the lists
below.

## Instructions

- Return a JSON object with a top-level "features" array.
- Each element must include "feature_id", "data", "applications" and
  "technology" arrays.
- For each mapping list, return at most 5 items.
- Items in these arrays must provide "item" and "contribution" fields. The
  "contribution" value is a number in [0.1, 1.0] where 1.0 = critical, 0.5 =
  helpful and 0.1 = weak.
- Do not invent IDs; only use those provided.
- Do not include any text outside the JSON object.
```

## Testing

Run the following checks before committing:

```bash
black .
ruff .
mypy .
bandit -r src -ll
pip-audit
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines and mandatory
quality checks.
