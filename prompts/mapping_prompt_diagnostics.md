# Feature mapping (diagnostics)

Map each feature for the {service_name} service at plateau {plateau} to relevant {mapping_labels} from the lists below.

Lists are provided as JSON arrays in code blocks. Each object contains `id`, `name`, and `description` fields. Use these lists only; do not invent or transform IDs.

{mapping_sections}

## Features (for review only; do not quote or cite in output)

{features}

## Matching approach

- Review all supplied materials (situational context, definitions, inspirations, service description, jobs to be done, customer types, and sample features) to understand how each feature supports the {service_name} service at this plateau.
- Use these inputs to select relevant IDs, but do not mention or cite any inputs in the output.
- Match on meaning, not just keywords. Prefer exact or closely aligned concepts over loose associations.
- If unsure, select fewer but accurate IDs rather than speculative matches.

## Instructions

- Return a single JSON object with a top-level "features" array.
- Each element must include:
  - "feature_id": the ID from the supplied features (use exact value and type).
  - Arrays for each of: {mapping_fields}.
- Selection rules for each array:
  - Include all relevant IDs from the corresponding list(s).
  - Each array element must be an object with exactly:
    - "item": the selected ID (preserve original type and formatting, e.g., string vs number, UUID hyphens).
    - "rationale": a single-line explanation of the match (concise; no line breaks).
  - Do not include weights, scores, or extra fields.
  - Do not invent IDs or rationales.
  - Deduplicate within each array (no repeated IDs per feature).
  - If a feature has no relevant IDs for a field, include an empty array for that field.
- Maintain terminology consistent with the situational context, definitions, and inspirations.
- Do not include any text outside the JSON object.
- Return ONLY valid JSON. No Markdown, no backticks, no commentary, no trailing commas.
- If you are about to include any text outside JSON, stop and return JSON only.
- The response must adhere strictly to the JSON schema provided below.

## Example output (illustrative structure; IDs here are placeholders)

```json
{
  "features": [
    {
      "feature_id": "feat-001",
      "technology": [
        {
          "item": "tech-001",
          "rationale": "Enables automated identity checks aligned to capability scope."
        },
        {
          "item": "tech-004",
          "rationale": "Delivers self-service updates that match profile management."
        }
      ]
    },
    {
      "feature_id": "feat-002",
      "technology": [
        {
          "item": "tech-010",
          "rationale": "Supports on-the-go tasks through the mobile app."
        }
      ]
    }
  ]
}
```

## Response structure

{schema}
