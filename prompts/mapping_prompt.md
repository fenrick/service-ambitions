# Feature mapping

Map each feature to relevant Data, Applications and Technologies from the lists below.

## Available Data

{data_items}

## Available Applications

{application_items}

## Available Technologies

{technology_items}

## Features

{features}

## Instructions

- Return a JSON object with a top-level "features" array.
- Each element must include "feature_id", "data", "applications" and "technology" arrays.
- Items in these arrays must provide "item" and "contribution" fields.
- Use only identifiers from the provided lists.
- Do not include any text outside the JSON object.
- The response must adhere to the JSON schema provided below.
