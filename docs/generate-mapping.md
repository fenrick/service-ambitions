# Generate mapping

Remap feature mappings for existing service evolution results.

## Running

Example command:

```bash
poetry run service-ambitions generate-mapping \
  --input evolution.jsonl \
  --output remapped.jsonl \
  --strict-mapping --diagnostics
```

`--mapping-data-dir` points to the directory containing mapping reference data
files.

`--strict-mapping` raises an error when any requested mapping type is missing or
produces an empty list. Disable with `--no-strict-mapping` to keep existing
mappings and continue.

`--diagnostics` enables verbose logging and spans useful for troubleshooting.

## Input format

Provide a JSON Lines file produced by `generate-evolution`. Each line should be a
`ServiceEvolution` object containing plateau features. Existing mapping data, if
present, is replaced.

## Output format

The output mirrors the input structure with refreshed mapping contributions under
each feature. Mapping lists include an `item` identifier and a numeric
`contribution` score in the range `[0.1, 1.0]`.

## Troubleshooting

- Ensure `OPENAI_API_KEY` is set and valid.
- Use `--dry-run` to validate input files without making API calls.
- Check network access and API quotas if mapping requests fail repeatedly.
