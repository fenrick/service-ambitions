# service-ambitions

## Configuration

The CLI requires an OpenAI API key available in the `OPENAI_API_KEY` environment
variable. The key is loaded after `.env` files are processed and the application
will exit if the variable is missing.

An example configuration is provided in `.env.example`. Copy it to `.env` and
fill in your credentials:

```bash
cp .env.example .env
```

Update the `OPENAI_API_KEY` value and any optional settings. For production
deployments, inject the variable using your platform's secret manager instead of
committing keys to source control.

The chat model can be set with the `--model` flag or the `MODEL` environment
variable. Additional parameters such as the desired `response_format` may be
provided with the `--response-format` flag or the `RESPONSE_FORMAT` environment
variable.

To collect detailed traces with [LangSmith](https://docs.smith.langchain.com/),
set the `LANGSMITH_API_KEY` environment variable and optionally supply a
project name via `--langsmith-project`.

## Installation

Dependencies are managed with [Poetry](https://python-poetry.org/). Install the
tool and then either run the project directly through Poetry or use the provided
`run.sh` helper script, which installs dependencies, loads environment variables
from `.env` and launches the CLI:

```bash
./run.sh --input-file sample-services.jsonl --output-file ambitions.jsonl
```

## Usage

`sample-services.jsonl` contains example services in
[JSON Lines](https://jsonlines.org/) format, with one JSON object per line. The
output file will also be in JSON Lines format. Use the `--concurrency` option to
control how many services are processed in parallel.
