# Service plateau descriptions generation

Provide a standalone description for each service maturity plateau below. Each description must be independent and avoid referencing other plateaus or later stages.

## Plateaus

{plateaus}

## Instructions

- **Synthesise the Description**: Base your description on a thorough review of all provided details for the plateau, including its existing description, jobs to be done, customer types, and sample features. Focus on creating a foundational draft that clarifies the service's capabilities and value at this stage without making assumptions.
- **Clarity and Tone**: Write clearly and concisely in Australian English using plain language.
- **Anonymity**: Do not refer to the service by name or as "this service". Do not mention the plateau name or number within the description text.
- **Jargon**: Avoid unnecessary jargon. If a technical term is essential, briefly explain it in simple terms.
- **JSON Output Only**: Return a JSON object containing a `descriptions` array with `plateau`, `plateau_name` and `description` fields.
- **No Preamble**: Descriptions must be non-empty strings that begin directly with the service details.
- **Strict Formatting**: Do not include any text, markdown, commentary, or trailing commas outside the valid JSON object. Your entire response must be the JSON object itself.
- **Schema Adherence**: The response must strictly adhere to the JSON schema provided below.

## Response structure

{schema}
