# Plateau feature generation

Generate service features for the {service_name} service at plateau {plateau} using the supplied plateau service description.

## Plateau Service description (for review only; do not quote or cite in output)

{service_description}

## How to work with the inputs

- Review all supplied materials to inform the features (situational context, definitions, inspirations, overall service description and plateau description, jobs to be done, and customer types).
- Use these inputs to synthesise accurate, useful features. Do not reference or cite the inputs in the output.
- If information conflicts, prefer the most recent or most specific details. Keep assumptions to a minimum.

## Writing rules

- Do not refer to the service by name or as "this service".
- Do not refer to the plateau by name or number.
- Write clearly and concisely in Australian English using plain language.
- Use short sentences and active voice.
- Avoid unnecessary jargon. If a technical term or acronym is essential, add a brief explanation in simple words.
- Avoid company names; describe technologies by their underlying nature (e.g., “distributed ledger” rather than brand names).
- Describe current capabilities appropriate to the plateau; avoid future-tense roadmaps.

## Feature content requirements

- Align with the reviewed description and maintain consistent terminology from the context and definitions.
- Tailor features to the relevant jobs to be done, customer types, and sample features.
- Each feature’s description should:
  - Explain the user value and operational outcome.
  - Note any critical safeguards where relevant (privacy, security, safety).
  - Be testable and specific enough to validate.
- Differentiate features within the plateau; use the full CMMI range where appropriate.

## CMMI scoring guidance

- Levels and labels must match exactly:
  - 1 Initial
  - 2 Managed
  - 3 Defined
  - 4 Quantitatively Managed
  - 5 Optimizing
- Choose a level based on evidence implied by the feature:
  - 1: ad hoc, inconsistent, individual effort.
  - 2: repeatable and tracked; basic planning and oversight.
  - 3: documented, standardised across teams.
  - 4: measured with quantitative targets and controls.
  - 5: continuously improved with feedback loops and experimentation.
- The justification must briefly state why the level fits (one short sentence).

## Output rules

- Return a single JSON object with a top-level "features" key.
- "features" must include keys for each role: {roles}.
- Each role key must map to an array containing at least {required_count} feature objects.
- Every feature object must include:
  - "name": short feature title.
  - "description": concise explanation of the feature.
  - "score": object with:
    - "level": integer 1–5.
    - "label": one of "Initial", "Managed", "Defined", "Quantitatively Managed", "Optimizing".
    - "justification": brief rationale for the level.
- Do not include any text outside the JSON object.
- Return ONLY valid JSON. No Markdown. No backticks. No commentary. No trailing commas.
- If you are about to include any text outside JSON, stop and return JSON only.
- The response must adhere to the JSON schema provided below.

## Example output

```json
{{
  "features": {{
    "learners": [
      {{
        "name": "Smart enrolment",
        "description": "People complete enrolment online with real-time checks for eligibility and data accuracy.",
        "score": {{
          "level": 3,
          "label": "Defined",
          "justification": "Standardised workflow and documented rules are applied consistently."
        }}
      }}
    ],
    "academics": [],
    "professional_staff": []
  }}
}}

## Response structure

{schema}
