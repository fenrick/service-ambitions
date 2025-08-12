# Generate evolution

The evolution workflow spans the plateaus defined in
`data/service_feature_plateaus.json`, issuing three calls per plateau
(description, features, mapping). The CLI evaluates all plateaus in this file
alongside all customer segments. Plateau name to level mappings are derived
from the order of the JSON entries.

## Running

Example command:

```bash
poetry run service-ambitions generate-evolution \
  --input-file sample-services.jsonl \
  --output-file evolution.jsonl
```

Services are processed in batches controlled by the `batch_size` setting in
`config/app.json` to avoid scheduling all tasks at once.

Include `--seed <value>` to make backoff jitter and model sampling
deterministic when supported by the provider.

## Output schema

Each line in the output file is a JSON object with:

```json
{
  "service": {
    "service_id": "string",
    "name": "string",
    "description": "string",
    "customer_type": "string",
    "jobs_to_be_done": [{"name": "string"}]
  },
  "plateaus": [
    {
      "plateau": 1,
      "plateau_name": "string",
      "service_description": "string",
      "features": [
        {
          "feature_id": "string",
          "name": "string",
          "description": "string",
          "score": {
            "level": 3,
            "label": "Defined",
            "justification": "string"
          },
          "customer_type": "string",
          "data": [{ "item": "string", "contribution": 0.5 }],
          "applications": [{ "item": "string", "contribution": 0.5 }],
          "technology": [{ "item": "string", "contribution": 0.5 }]
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
