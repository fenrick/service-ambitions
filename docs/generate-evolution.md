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

## Testing

Run project checks before committing:

```bash
black .
ruff .
mypy .
bandit -r src -ll
pip-audit
```
