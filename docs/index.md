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

See the LLM Queue section to enable centralised concurrency and pipelined
execution when appropriate.

For the full project overview, see the README on GitHub:

https://github.com/FromHereOnAU/service-ambitions#readme
