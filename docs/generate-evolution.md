# Generate evolution

## Running

Example command:

```bash
poetry run python src/cli.py generate-evolution \
  --input-file sample-services.jsonl \
  --output-file evolution.jsonl \
  --plateaus Foundational Enhanced \
  --customers retail enterprise
```

## Output schema

Each line in the output file is a JSON object with:

```json
{
  "service": {
    "name": "string",
    "description": "string",
    "customer_type": "string"
  },
  "results": [
    {
      "feature": {
        "feature_id": "string",
        "name": "string",
        "description": "string"
      },
      "score": 0.0,
      "conceptual_data_types": [{"item": "string", "contribution": "string"}],
      "logical_application_types": [{"item": "string", "contribution": "string"}],
      "logical_technology_types": [{"item": "string", "contribution": "string"}]
    }
  ]
}
```

## Testing

Run project checks before committing:

```bash
black .
ruff .
mypy .
bandit -r src -ll
pip-audit
```
