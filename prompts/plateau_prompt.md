# Plateau feature generation

Generate service features for the {service_name} service at plateau {plateau} for {customer_type} customers.

## Instructions
- Use the service description: {service_description}.
- Return a valid JSON object.
- Include a top-level "features" array with at least {required_count} objects.
- Each feature must provide:
  - "feature_id": unique string identifier.
  - "name": short feature title.
  - "description": explanation of the feature.
  - "score": floating-point maturity between 0 and 1.
- Do not include any text outside the JSON object.

## Expected Output
```
{
  "features": [
    {
      "feature_id": "feat-001",
      "name": "Accessible content",
      "description": "Learners can access materials from any device.",
      "score": 0.5
    }
  ]
}
```
