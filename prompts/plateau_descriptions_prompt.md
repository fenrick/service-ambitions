# Service plateau descriptions generation

Provide a standalone description for each service maturity plateau below. Each description must be independent and avoid referencing other plateaus or later stages.

## Plateaus
{plateaus}

## Instructions
- Do not refer to the service by name or as "this service".
- Do not mention the plateau name or number inside the description text.
- Write clearly and concisely in Australian English using plain language.
- Avoid unnecessary jargon; briefly explain any essential technical terms.
- Return a JSON object containing a `descriptions` array with `plateau`, `plateau_name` and `description` fields.
- Descriptions must be non-empty strings beginning directly with the service details (no preamble).
- Do not include any text outside the JSON object.
- Return ONLY valid JSON. No Markdown, commentary or trailing commas.
- If you are about to include text outside JSON, stop and return JSON only.
- The response must adhere to the JSON schema provided below.

## Response structure
{schema}
