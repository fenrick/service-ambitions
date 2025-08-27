# Feature mapping

Map each feature to relevant {mapping_labels} from the lists below.

Lists are provided as JSON arrays in code blocks.
Each object contains `id`, `name`, and `description` fields.

{mapping_sections}

## Features

{features}

## Instructions

- **Analyse and Compare**: For each feature, first thoroughly analyse its `name` and `description` to understand its core purpose. Then, compare this purpose against the `name` and `description` of every item in the provided lists ({mapping_labels}).
- **Establish Mappings**: Create a mapping if the feature **directly solves, enables, or significantly contributes** to the outcome, job, or persona described by an item. Think about the underlying user need the feature serves.
- **Be Comprehensive**: Select all relevant IDs from the lists. A single feature can map to multiple items across different lists. Do not be conservative; if a meaningful relationship exists, create the mapping.
- **Maintain Consistency**: Use terminology consistent with the provided situational context, definitions, and inspirations.
- **Strict JSON Formatting**:
  - Return a JSON object with a top-level "features" array.
  - Each element in the array must include a "feature_id" and the following mapping arrays: {mapping_fields}.
  - Select all relevant IDs from the lists provided.
  - Each entry in the mapping arrays must be an object containing only an "item" field for the selected ID (e.g., `{"item": "jtbd-1"}`).
  - Do not include weights, confidence scores, or explanations.
  - Do not invent new IDs that are not present in the provided lists.
- **Output Integrity**:
  - Do not include any text, markdown, or commentary outside the main JSON object.
  - The entire response must be only the valid JSON object.
  - The response must adhere to the JSON schema provided below.

## Response structure

{schema}
