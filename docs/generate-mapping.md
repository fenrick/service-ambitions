# Generate mapping

Remap feature mappings for existing service evolution results.

## Running

Example command:

```bash
poetry run service-ambitions generate-mapping \
  --input evolution.jsonl \
  --output remapped.jsonl \
  --strict-mapping --diagnostics --no-logs
```

`--mapping-data-dir` points to the directory containing mapping reference data
files.

`--strict-mapping` raises an error when any requested mapping type is missing,
produces an empty list or contains unknown identifiers. Disable with
`--no-strict-mapping` to drop unknown mappings and continue.

`--diagnostics` enables verbose logging and telemetry instrumentation useful for
troubleshooting. Instrumentation runs only when this verbose mode is enabled.
Prompt text is omitted unless `--allow-prompt-logging` is supplied.

`--use-local-cache` reads and writes mapping responses under `.cache/mapping` to
avoid repeated network requests during development.

Mapping runs once per configured set. Each prompt receives the relevant
reference list—`applications`, `technologies` and `information` by default—and
returns items matched to the feature.

## Input format

Provide a JSON Lines file produced by `generate-evolution`. Each line should be a
`ServiceEvolution` object containing plateau features. Existing mapping data, if
present, is replaced.

## Output format

The output mirrors the input structure with refreshed mappings under each
feature. Mapping lists include an `item` identifier only. Pass `--diagnostics`
to also receive a `rationale` explaining each match.

```json
"mappings": {
  "applications": [{"item": "APP001"}],
  "technologies": [{"item": "TECH002"}],
  "information": [{"item": "INFO003"}]
}
```

## Troubleshooting

- Ensure `OPENAI_API_KEY` is set and valid.
- Use `--dry-run` to validate input files without making API calls.
- Check network access and API quotas if mapping requests fail repeatedly.

---

This documentation is licensed under the [MIT License](../LICENSE).
