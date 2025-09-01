# Feature mapping

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

- Each element must include:
  - "feature_id": the ID from the supplied features (use exact value and type).
  - Arrays for each of: {mapping_fields}.
- Selection rules for each array:
  - Include all relevant IDs found in the corresponding list(s).
  - Each array element must be an object with only one field: { "item": <ID> }.
  - Do not include weights, explanations, scores, or extra fields.
  - Do not invent IDs. Only use IDs present in the provided lists.
  - No limit on the number of returned items.
  - Deduplicate within each array (no repeated IDs per feature).
  - Preserve the original ID type (string vs number) and formatting (e.g., UUID hyphens).
- If a feature has no relevant IDs for a field, include an empty array for that field.
- Maintain terminology consistent with the situational context, definitions, and inspirations.
- The response must adhere strictly to the JSON schema provided below.
## Response structure

{schema}
