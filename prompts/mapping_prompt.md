# Feature mapping

Map each feature to relevant {mapping_labels} from the lists below.

{mapping_sections}

## Features

{features}

## Instructions

- Return a JSON object with a top-level "features" array.
- Each element must include "feature_id" and the following arrays: {mapping_fields}.
- For each mapping list, return at most 5 items.
- Items in these arrays must provide "item" and "contribution" fields. The
  "contribution" value describes how strongly the mapped item supports the
  feature; higher numbers indicate greater importance.
- contribution is a number in [0.1, 1.0] where 1.0 = critical, 0.5 = helpful, 0.1 = weak.
- Do not invent IDs; only use those provided.
- Maintain terminology consistent with the situational context, definitions and inspirations.
- Do not include any text outside the JSON object.
- The response must adhere to the JSON schema provided below.

## Response structure

{schema}
