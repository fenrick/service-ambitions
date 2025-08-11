# Service description generation

Refer to the situational context, definitions and inspirations provided earlier to maintain consistent terminology across stages.

Provide a description of the service at plateau {plateau}.

## Instructions

- Do not refer to the service by name.
- Do not refer to the plateau by name or number.
- Write clearly and concisely in Australian English, following plain language principles.
- Use short, simple sentences and active voice.
- Avoid unnecessary jargon or “consultant speak” – explain concepts in layperson’s terms unless technical detail is needed.
- If you must use technical terms or acronyms, briefly describe them for clarity.
- Base wording on the situational context, definitions and inspirations.
- Return a JSON object containing only a `description` field.
- `description` must be a non-empty string explaining the service at plateau {plateau}.
- Do not include any text outside the JSON object.
- The response must adhere to the JSON schema provided below.

## Response structure

{schema}
