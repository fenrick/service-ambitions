# Generate evolution

The evolution workflow spans four plateaus—**Foundational**, **Enhanced**,
**Experimental** and **Disruptive**—with three calls per plateau (description,
features, mapping).

## Running

Example command:

```bash
poetry run python src/cli.py generate-evolution \
  --input-file sample-services.jsonl \
  --output-file evolution.jsonl \
  --plateaus Foundational Enhanced Experimental Disruptive \
  --customers retail enterprise
```

## Output schema

Each line in the output file is a JSON object with:

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
      "service_description": "string",
      "features": [
        {
          "feature_id": "string",
          "name": "string",
          "description": "string",
          "score": 0.0,
          "customer_type": "string",
          "data": [{"item": "string", "contribution": "string"}],
          "applications": [{"item": "string", "contribution": "string"}],
          "technology": [{"item": "string", "contribution": "string"}]
        }
      ]
    }
  ]
}
```

The conversation seed for each service includes the `service_id` and
`jobs_to_be_done` list so that all plateau calls share the same context.

## Testing

Run project checks before committing:

```bash
black .
ruff .
mypy .
bandit -r src -ll
pip-audit
```
