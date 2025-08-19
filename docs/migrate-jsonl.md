# migrate-jsonl

The `migrate-jsonl` command upgrades service definition files from schema version
1.0 to the current 1.x format. It reads a legacy JSON Lines file and writes a new
file with updated keys and structure.

```bash
poetry run service-ambitions migrate-jsonl --input-file services_v1.jsonl --output-file services_v1x.jsonl
```

The conversion performs a best-effort rename of fields such as `id` →
`service_id`, `jobs` → `jobs_to_be_done`, and feature `id` → `feature_id`. String
jobs are wrapped in objects to match the new schema. Unknown fields are retained
where possible, but the tool does not perform full validation. Always review the
results before replacing the original file.
