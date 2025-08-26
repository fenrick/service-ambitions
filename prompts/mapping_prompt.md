# Feature mapping

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
 - Each entry in these arrays must be an object containing only an "item" field for the selected ID.
 - Do not include weights or explanations.
 - Do not invent IDs.
 - No limit on the number of items you can return.
- Maintain terminology consistent with the situational context, definitions and inspirations.
- Do not include any text outside the JSON object.
- The response must adhere to the JSON schema provided below.

## Response structure

{schema}
