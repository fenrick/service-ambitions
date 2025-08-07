# Feature mapping

Associate the feature with relevant {category_label} from the list below.

## Available {category_label}
{category_items}

## Instructions
- Return a JSON object with a "{category_key}" array.
- Each item in the array must include:
  - "item": identifier from the list.
  - "contribution": brief explanation of how it supports the feature.
- Use only items provided above.
- Do not include any text outside the JSON object.

Feature name: {feature_name}
Feature description: {feature_description}

## Expected Output
```
{
  "{category_key}": [
    {
      "item": "INF-1",
      "contribution": "Explanation"
    }
  ]
}
```
