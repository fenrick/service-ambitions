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
- The response must adhere to the JSON schema provided below.
