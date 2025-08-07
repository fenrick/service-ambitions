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
`<provider>:<model>`. The default is `openai:gpt-4o-mini`.

System prompts are assembled from modular markdown components located in the
`prompts/` directory. Use `--prompt-dir` to point at an alternate component
directory, `--context-id` to select a situational context, and
`--inspirations-id` to choose a list of future inspirations. This structure
supports swapping sections to suit different industries.

To collect detailed traces with [Pydantic Logfire](https://logfire.pydantic.dev/),
set the `LOGFIRE_TOKEN` environment variable and optionally supply a service
 name via `--logfire-service`. The CLI automatically installs Logfire auto
 tracing and instruments Pydantic, Pydantic AI, OpenAI, and system metrics when
 Logfire is enabled.

## Installation

Dependencies are managed with [Poetry](https://python-poetry.org/). Install the
tool and then install the project's dependencies with:

```bash
poetry install
```

Run the CLI through Poetry to ensure it uses the managed environment. Use
subcommands to select the desired operation:

```bash
poetry run python src/cli.py generate-ambitions --input-file sample-services.jsonl --output-file ambitions.jsonl
poetry run python src/cli.py generate-evolution --input-file sample-services.jsonl --output-file evolution.jsonl
```

Alternatively, use the provided shell script which forwards all arguments to the CLI:

```bash
./run.sh generate-ambitions --input-file sample-services.jsonl --output-file ambitions.jsonl
./run.sh generate-evolution --input-file sample-services.jsonl --output-file evolution.jsonl
```

## Usage

`sample-services.jsonl` contains example services in
[JSON Lines](https://jsonlines.org/) format, with one JSON object per line. The
output file will also be in JSON Lines format. Use the `--concurrency` option to
control how many services are processed in parallel when running
`generate-ambitions`.

### Generating service evolutions

Use the `generate-evolution` subcommand to score each service against plateau
features. It reads services from an input JSON Lines file and writes a
`ServiceEvolution` record for each line in the output file. Enable verbose logs
with `-v` or `-vv`.

Basic invocation:

```bash
./run.sh generate-evolution --input-file sample-services.jsonl --output-file evolution.jsonl
```

Restrict evaluation to specific plateaus or customer types as needed:

```bash
./run.sh generate-evolution --plateaus Foundational Enhanced --customers retail enterprise \
  --input-file sample-services.jsonl --output-file evolution.jsonl
```

## ServiceEvolution schema

Each JSON line in the output file follows the `ServiceEvolution` schema:

```json
{
  "service": {
    "name": "string",
    "description": "string",
    "customer_type": "string"
  },
  "results": [
    {
      "feature": {
        "feature_id": "string",
        "name": "string",
        "description": "string"
      },
      "score": 0.0,
      "conceptual_data_types": [{"item": "string", "contribution": "string"}],
      "logical_application_types": [{"item": "string", "contribution": "string"}],
      "logical_technology_types": [{"item": "string", "contribution": "string"}]
    }
  ]
}
```

Fields in the schema:

- `service`: `ServiceInput` with `name`, optional `customer_type`, and
  `description`.
- `results`: list of `PlateauResult` entries, each containing:
  - `feature`: `PlateauFeature` with `feature_id`, `name`, and `description`.
  - `score`: float between `0.0` and `1.0`.
  - `conceptual_data_types`, `logical_application_types`,
    `logical_technology_types`: lists of `Contribution` objects describing why a
    mapped item supports the feature.

## Reference Data

Feature mapping uses cached reference lists stored in the `data/` directory.
Each of `information.json`, `applications.json`, and `technologies.json`
contains items with identifiers, names, and descriptions. These lists are
injected into mapping prompts so that features can be matched against consistent
options. Mapping prompts run separately for information, applications and
technologies to keep each decision focused.

## Testing

Run the following checks before committing:

```bash
black .
ruff .
mypy .
bandit -r src -ll
pip-audit
```
