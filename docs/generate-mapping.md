# Generate mapping

Remap feature mappings for existing service evolution results.

## Running

Example command:

```bash
poetry run service-ambitions generate-mapping \
  --input-file evolution.jsonl \
  --output-file remapped.jsonl \
  --mapping-batch-size 20 --mapping-parallel-types
```

`--mapping-batch-size` controls how many features are sent in each mapping
request. Smaller batches shorten prompts but require more API calls; larger
batches reduce round trips at the cost of bigger prompts and potential context
limit pressure.

`--mapping-parallel-types` dispatches mapping requests for each mapping type
concurrently. Disable it with `--no-mapping-parallel-types` to process types
sequentially when working under tight rate limits.

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
