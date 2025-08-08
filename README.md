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

## Plateau-first workflow

Each service is evaluated across **four** plateaus – **Foundational**,
**Enhanced**, **Experimental** and **Disruptive**. Every plateau requires three
sequential calls:

1. **Description** – request a plateau-specific service narrative.
2. **Features** – generate learner, staff and community features.
3. **Mapping** – link each feature to reference Data, Applications and
   Technologies.

The 4 × 3 workflow totals 12 calls and produces a complete `ServiceEvolution`
record for every service.

Plateau names and descriptions are sourced from
`data/service_feature_plateaus.json`, allowing the progression to be
reconfigured without code changes. By default the CLI uses the first four
entries from this file.

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
./run.sh generate-evolution --plateaus Foundational Enhanced Experimental Disruptive --customers retail enterprise \
  --input-file sample-services.jsonl --output-file evolution.jsonl
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
  "service": {
    "service_id": "string",
    "name": "string",
    "description": "string",
    "customer_type": "string",
    "jobs_to_be_done": ["string"]
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
          "score": 0.0,
          "customer_type": "string",
          "data": [{ "item": "string", "contribution": "string" }],
          "applications": [{ "item": "string", "contribution": "string" }],
          "technology": [{ "item": "string", "contribution": "string" }]
        }
      ]
    }
  ]
}
```

Fields in the schema:

- `service`: `ServiceInput` with `service_id`, `name`, `description`, optional
  `customer_type`, `jobs_to_be_done`, and existing `features`.
- `plateaus`: list of `PlateauResult` entries, each containing:
    - `plateau`: integer plateau level.
    - `plateau_name`: descriptive plateau label.
    - `service_description`: narrative for the service at that plateau.
    - `features`: list of `PlateauFeature` entries with:
        - `feature_id`, `name`, and `description`.
        - `score`: float between `0.0` and `1.0`.
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
stored in `config/app.json`; mapping types and their associated datasets live
under the `mapping_types` section, allowing new categories to be added without
code changes. Plateau name to level associations are defined in the
`plateau_map` section of the same file.

## Prompt examples

### Plateau prompt

```markdown
# Plateau feature generation

Generate service features for the {service_name} service at plateau {plateau}.

## Instructions

- Use the service description: {service_description}.
- Return a single JSON object with three keys: "learners", "staff" and
  "community".
- Each key must map to an array containing at least {required_count} feature
  objects.
- Every feature must provide:
  - "feature_id": unique string identifier.
  - "name": short feature title.
  - "description": explanation of the feature.
  - "score": floating-point maturity between 0 and 1.
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
- Items in these arrays must provide "item" and "contribution" fields.
- Use only identifiers from the provided lists.
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
