# Generate mapping

Remap feature mappings for existing service evolution results. The
command reuses the same runtime environment and caching strategy as the
evolution workflow, ensuring that relocated cache files and lazy
loading behave consistently across both commands.

## Running

Example command:

```bash
poetry run service-ambitions generate-mapping \
  --input evolution.jsonl \
  --output remapped.jsonl \
  --strict-mapping
```

`--mapping-data-dir` points to the directory containing mapping reference data
files.

`--strict-mapping` raises an error when any requested mapping type is missing,
produces an empty list or contains unknown identifiers. Disable with
`--no-strict-mapping` to drop unknown mappings and continue.

Telemetry via Logfire is always enabled. Prompt text is omitted unless
`--allow-prompt-logging` is supplied.

`--use-local-cache` reads mapping responses under
`<cache_dir>/<service>/mappings` (default cache dir:
`${XDG_CACHE_HOME}/service-ambitions`, falling back to `/tmp/service-ambitions` when `XDG_CACHE_HOME` is unset) and optionally writes new entries to
avoid repeated network requests during development. `--cache-mode` controls how
the cache is used (`off`, `read`, `refresh`, `write`) with `read` as the default,
and `--cache-dir` sets the cache storage location.

Mapping runs once per configured set. Each prompt receives the relevant
reference list—`applications`, `technologies` and `information` by default—and
returns items matched to the feature.

## Dataset format (self‑contained)

Mapping datasets can be provided either as a plain list of items or as a
self‑contained object that embeds configuration alongside the catalogue. The
loader accepts both formats:

- List form (existing behaviour):

  ```json
  [
    {
      "id": "APP001",
      "name": "CRM",
      "description": "Customer relationship mgmt."
    }
  ]
  ```

- Object form (flexible, self‑describing):

  ```json
  {
    "field": "applications",
    "label": "Business Applications",
    "items": [
      {
        "id": "APP001",
        "name": "CRM",
        "description": "Customer relationship mgmt."
      }
    ]
  }
  ```

When the object form is used, the embedded `field` and `label` are included in
digest calculations (cache keys) and validated against external configuration
if provided. External `mapping_sets` still determine which files are loaded and
the target feature field; if the file’s `field` disagrees, a warning is logged
to help you reconcile configuration.

## Input format

Provide a JSON Lines file produced by `generate-evolution`. Each line should be a
`ServiceEvolution` object containing plateau features. Existing mapping data, if
present, is replaced.

## Output format

The output lists mapping catalogue entries and the features associated with
each. Every record contains an `id` and `name` for the mapping item and a
`mappings` array of feature references. Features without matching entries are
simply omitted.

```json
{
  "id": "AC007",
  "name": "Analytics",
  "mappings": [{ "feature_id": "FEAT001", "description": "Student portal" }]
}
```

## Troubleshooting

- Ensure `SA_OPENAI_API_KEY` is set and valid.
- Use `--dry-run` to validate input files without making API calls.
- Check network access and API quotas if mapping requests fail repeatedly.

---

This documentation is licensed under the [MIT License](../LICENSE).
