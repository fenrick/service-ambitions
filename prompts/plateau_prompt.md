# Plateau feature generation

Generate service features for the {service_name} service at plateau {plateau}.

## Instructions

- Reference the situational context, definitions and inspirations to maintain consistent terminology.
- Do not refer to the service by name.
- Do not refer to the plateau by name or number.
- Write clearly and concisely in Australian English, following plain language principles.
- Use short, simple sentences and active voice.
- Avoid unnecessary jargon or “consultant speak” – explain concepts in layperson’s terms unless technical detail is needed.
- If you must use technical terms or acronyms, briefly describe them for clarity.
- Return a single JSON object with three keys: "learners", "academics" and "professional_staff".
- Each key must map to an array containing at least {required_count} feature objects.
- Every feature must provide:
    - "feature_id": unique string identifier.
    - "name": short feature title.
    - "description": explanation of the feature.
    - "score": floating-point maturity between 0 and 1.
- Do not include any text outside the JSON object.
- The response must adhere to the JSON schema provided below.

## Response structure

{schema}
