# Plateau feature generation

Generate service features for the {service_name} service at plateau {plateau}.

## Instructions
- Use the service description: {service_description}.
- Return a single JSON object with three keys: "learners", "staff" and "community".
- Each key must map to an array containing at least {required_count} feature objects.
- Every feature must provide:
  - "feature_id": unique string identifier.
  - "name": short feature title.
  - "description": explanation of the feature.
  - "score": floating-point maturity between 0 and 1.
- Do not include any text outside the JSON object.

## Expected Output
```
{
  "learners": [
    {
      "feature_id": "learn-001",
      "name": "Accessible content",
      "description": "Learners can access materials from any device.",
      "score": 0.5
    }
  ],
  "staff": [
    {
      "feature_id": "staff-001",
      "name": "Training dashboard",
      "description": "Staff can track learner progress.",
      "score": 0.5
    }
  ],
  "community": [
    {
      "feature_id": "community-001",
      "name": "Public resources",
      "description": "Open materials for the wider community.",
      "score": 0.5
    }
  ]
}
```
