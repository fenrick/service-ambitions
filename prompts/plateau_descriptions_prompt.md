# Service plateau descriptions generation

Provide a standalone description for each service maturity plateau below. Each description must be independent and avoid referencing other plateaus or later stages.

## Plateaus

{plateaus}

## How to work with the inputs

- Review all provided details for each plateau (existing description, jobs to be done, customer types, sample features, and any supplementary context).
- Use these inputs to synthesise each description, but do not mention or cite them in the output.

## Instructions

- Synthesise the Description: Base each description on a thorough review of the supplied details. Create a foundational draft that clarifies capabilities and value at this stage without making assumptions. There is no strict one-to-one mapping across inputs; elaborate where needed to clarify intent.
- Independence: Do not reference other plateaus or later stages.
- Clarity and Tone: Write clearly and concisely in Australian English using plain language.
- Anonymity: Do not refer to the service by name or as "this service". Do not mention the plateau name or number within the description text.
- Jargon: Avoid unnecessary jargon. If a technical term is essential, briefly explain it in simple terms.
- Technology naming: Avoid company names; describe technologies by their underlying nature (e.g., “distributed ledger” rather than brand or product names).
- No Preamble: Begin each description directly with concrete service details relevant to users and operations.
- Tailoring: Align details to the relevant jobs to be done, customer types, and sample features.
- Quality check: Re-read and refine for clarity, concision, and consistency with the supplied materials before finalising.

## Output rules

- JSON Output Only: Return a JSON object containing a `descriptions` array with `plateau`, `plateau_name`, and `description` fields.
- Strict Formatting: Do not include any text, markdown, commentary, or trailing commas outside the valid JSON object. Your entire response must be the JSON object itself.
- If you are about to include text outside JSON, stop and return JSON only.
- Schema Adherence: The response must strictly adhere to the JSON schema provided below.

## Response structure

{schema}
