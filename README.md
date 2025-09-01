# service-ambitions

## Architecture

The generator runs on a layered engine design.  `ProcessingEngine`
coordinates work across services, `ServiceExecution` manages per‑service
state and spawns `PlateauRuntime` instances for each plateau.  A
thread‑safe `RuntimeEnv` singleton holds configuration and shared state
such as caches.  See [runtime-architecture](docs/runtime-architecture.md)
for a detailed walkthrough.

## Configuration

Copy the sample configuration and customise it for your environment:

```
cp config/app.example.yaml config/app.yaml
```

Then edit `config/app.yaml` to set models, reasoning presets and other options.

The CLI requires an OpenAI API key available in the `OPENAI_API_KEY` environment
variable. Settings are loaded via Pydantic, which reads from a `.env` file if
present. The application will exit if the key is missing. LLM interactions are
handled via [Pydantic AI](https://pydantic.dev/pydantic-ai/).

Create a `.env` file in the project root with:

```
# Required for API access
OPENAI_API_KEY=your_api_key_here
# Optional: provide to publish telemetry to Logfire
# LOGFIRE_TOKEN=your_logfire_token
```

For production deployments, inject the variable using your platform's secret
manager instead of committing keys to source control.

Caching of mapping responses is enabled by default to speed up repeated runs.
Disable or change cache behaviour in `config/app.yaml` or via environment
variables:

```yaml
use_local_cache: true # Enable reading/writing the cache directory.
cache_mode: "read" # Cache behaviour: off, read, refresh or write.
cache_dir: .cache # Directory to store cache files.
```

Set `use_local_cache: false` or `cache_mode: "off"` to bypass the cache, or use
`write` to record new entries and `refresh` to rewrite existing ones.

The chat model can be set with the `--model` flag or the `MODEL` environment
variable. Model identifiers must include a provider prefix, in the form
`<provider>:<model>`. The default is `openai:gpt-5` with medium reasoning effort.

Stage-specific model overrides live under the `models` block in
`config/app.yaml`. Defaults are:

```yaml
descriptions: openai:o4-mini
features: openai:gpt-5
mapping: openai:o4-mini
search: openai:gpt-4o-search-preview
```

| Stage        | Default model                  | Fast/cheap alternative |
| ------------ | ------------------------------ | ---------------------- |
| Descriptions | `openai:o4-mini`               | Already cost optimised |
| Features     | `openai:gpt-5`                 | `openai:o4-mini`       |
| Mapping      | `openai:o4-mini`               | Already cost optimised |
| Search       | `openai:gpt-4o-search-preview` | n/a                    |

OpenAI recommends using cost‑optimised models like `o4-mini` when latency or
price is a concern and higher‑capacity models such as `gpt-5` for best quality
results. See the [model selection guide](https://platform.openai.com/docs/guides/model-selection)
for details.

### Model matrix

| Preset       | Model            | Reasoning effort |
| ------------ | ---------------- | ---------------- |
| cheap & fast | `openai:o4-mini` | low              |
| balanced     | `openai:gpt-5`   | medium           |
| max quality  | `openai:gpt-5`   | high             |

Select a preset by setting `reasoning.effort` in `config/app.yaml` to
`low`, `medium` or `high`.

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
libraries for telemetry. The `LOGFIRE_TOKEN` environment variable is optional:
without it, Logfire still records logs and metrics locally but nothing is sent
to the cloud. Provide a token to stream traces to Logfire. The CLI instruments
Pydantic, Pydantic AI, OpenAI and system metrics by default. Prompts are
excluded from logs unless `--allow-prompt-logging` is specified.

See [Logging levels](docs/logging-levels.md) for guidance on TRACE through
EXCEPTION and when to use each level.

## Installation

Dependencies are managed with [Poetry](https://python-poetry.org/). Install the
tool and then install the project's dependencies with:

```bash
poetry install
```

Enable OpenTelemetry instrumentation with the `observability` extra:

```bash
poetry install -E observability
```

After installation the `service-ambitions` console script is available.
Run the CLI through Poetry to ensure it uses the managed environment. Use
subcommands to select the desired operation:

```bash
poetry run service-ambitions run --input-file sample-services.jsonl --output-file evolutions.jsonl
poetry run service-ambitions validate --input-file sample-services.jsonl
poetry run service-ambitions reverse --input-file evolutions.jsonl --output-file features.jsonl
```

Alternatively, use the provided shell script which forwards all arguments to the CLI:

```bash
./run.sh run --input-file sample-services.jsonl --output-file evolutions.jsonl
./run.sh validate --input-file sample-services.jsonl
./run.sh reverse --input-file evolutions.jsonl --output-file features.jsonl
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

## Concurrency control

Requests run in parallel using a standard semaphore. Each service consumes a
single permit, so throughput is capped only by the `--concurrency` flag (or the
`concurrency` setting in `config/app.yaml`).

## Plateau-first workflow

Each service is evaluated across the plateaus defined in
`data/service_feature_plateaus.json`. Generation occurs in two phases:

1. **Descriptions** – a single request returns narratives for all plateaus.
2. **Features and Mapping** – for each plateau, generate learner, academic and
   professional staff features and link them to reference Information,
   Applications and Technologies using the previously collected description.

This workflow issues one call to fetch all descriptions, then queries each
plateau separately for feature generation. The result is a complete
`ServiceEvolution` record for every service.

Plateau names and descriptions are sourced entirely from
`data/service_feature_plateaus.json`, allowing the progression to be
reconfigured without code changes. The CLI processes all entries from this
file.

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
          "feature_id": "string",
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

Fields in the schema:

- `meta`: run metadata for the invocation:
  - `schema_version`: output schema revision.
  - `run_id`: identifier constant across all records in a single run.
  - `seed`: integer used for deterministic sampling when supported.
  - `models`: mapping of generation stages to model identifiers.
  - `web_search`: whether web search was enabled.
  - `created`: ISO-8601 timestamp when the record was produced.
- `service`: `ServiceInput` with `service_id`, `name`, `description`, optional
  `customer_type`, `jobs_to_be_done`, and existing `features`.
  - `plateaus`: list of `PlateauResult` entries, each containing:
    - `plateau`: integer plateau level.
    - `plateau_name`: descriptive plateau label.
    - `service_description`: narrative for the service at that plateau.
    - `features`: list of `PlateauFeature` entries with:
      - `feature_id` – deterministic six-character code, plus `name` and
        `description`.
      - `score`: object with CMMI maturity `level`, `label` and `justification`.
      - `customer_type`: audience benefiting from the feature.
      - `mappings`: object with `information`, `applications` and
        `technologies` lists of mapping items referencing supporting IDs.

## Reference Data

Feature mapping uses cached reference lists stored in the `data/` directory.
Each of `information.json`, `applications.json`, and `technologies.json`
contains items with identifiers, names, and descriptions. These lists are
injected into mapping prompts so that features can be matched against consistent
options. Mapping prompts run separately for information, applications and
technologies to keep each decision focused. All application configuration is
stored in `config/app.yaml`; the chat model and any reasoning parameters live
at the top level, while mapping types and their associated datasets live under
the `mapping_types` section, allowing new categories to be added without code
changes. Plateau definitions and their level mappings come from
`data/service_feature_plateaus.json`. Roles are defined in `data/roles.json`,
and the required number of features per role is controlled by
`features_per_role` in `config/app.yaml`.

## Prompt examples

### Plateau prompt

```markdown
# Plateau feature generation

Generate service features for the {service_name} service at plateau {plateau}.

## Instructions

- Use the service description: {service_description}.
- Provide keys for each role: {roles}.
- Each key must map to an array containing at least {required_count} feature
  objects.
- Every feature must provide:
  - "name": short feature title.
  - "description": explanation of the feature.
  - "score": object describing CMMI maturity with:
    - "level": integer 1–5.
    - "label": matching CMMI maturity name.
    - "justification": brief rationale for the level.
```

### Mapping prompts

Each mapping set runs with its own prompt. Example for applications:

```markdown
# Applications mapping

Map each feature to relevant Applications from the list below.

## Instructions

- Include a top-level "features" array.
- Each element must include "feature_id" and an "applications" array of objects with an "item" field only.
- Do not invent IDs; only use those provided.
```

Repeat this structure for the `technologies` and `information` datasets.

## IDE support

The repository includes an `.editorconfig` file and default VS Code settings.
Opening the workspace in VS Code enables format-on-save with Black, import
sorting and linting via Ruff, plus Mypy and Bandit checks. Editors that support
EditorConfig will automatically apply the same indentation and newline rules.

Recommended extensions are declared in `.vscode/extensions.json`. Install them
to enable Python language features, Black formatting, Ruff linting, Mypy type
checking and GitHub integration. Debug configurations in `.vscode/launch.json`
run `main.py` or invoke `pytest` directly from the Run and Debug panel.

PyCharm users can rely on the checked-in `.idea` directory. It supplies run configurations for the application and tests. Activate the Poetry interpreter and install the Black and Ruff plugins to match the project's formatting and linting setup.

## Testing

Run the following checks before committing:

```bash
black .
ruff .
mypy .
bandit -r src -ll
pip-audit
```

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines and mandatory
quality checks. All participants are expected to uphold the
[Code of Conduct](CODE_OF_CONDUCT.md).
