# Feature mapping

Map each feature to relevant {mapping_labels} from the lists below.

{mapping_sections}

## Features

{features}

## Instructions

- Return a JSON object with a top-level "features" array.
- Each element must include "feature_id" and the following arrays: {mapping_fields}.
- Items in these arrays must provide "item" and "contribution" fields.
- Use only identifiers from the provided lists.
- Maintain terminology consistent with the situational context, definitions and inspirations.
- Do not include any text outside the JSON object.
- The response must adhere to the JSON schema provided below.

## Response structure

{schema}
