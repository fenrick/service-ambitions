# generate-mapping

Regenerate mapping contributions for existing evolution results.

```bash
poetry run service-ambitions generate-mapping --input evolution.jsonl --output evolution-remapped.jsonl
```

Use `--mapping-batch-size` to control the number of features processed per request
and `--mapping-parallel-types/--no-mapping-parallel-types` to toggle concurrent
mapping type requests. Provide `--mapping-model` to override the default model
for mapping operations.
