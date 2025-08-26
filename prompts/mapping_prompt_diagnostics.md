# Feature mapping (diagnostics)

Map each feature to relevant {mapping_labels} from the lists below.

Lists are provided as JSON arrays in code blocks.
Each object contains `id`, `name`, and `description` fields.

{mapping_sections}

## Features

{features}

## Instructions

- Return a JSON object with a top-level "features" array.
- Each element must include "feature_id" and the following arrays: {mapping_fields}.
- Select all relevant IDs from the lists provided.
- Each entry in these arrays must be an object containing:
  - "item" for the selected ID.
  - "rationale" with a single-line explanation of the match.
- Do not invent IDs or rationales.
- No limit on the number of items you can return.
- Maintain terminology consistent with the situational context, definitions and inspirations.
- Do not include any text outside the JSON object.
- The response must adhere to the JSON schema provided below.

## Response structure

{schema}
