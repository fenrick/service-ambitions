# Facets on Mapping Relationships

Facets let you capture the qualities of a relationship between a feature and a
catalogue item (application, technology, or data asset). They are configured in
the dataset JSON files under `data/` and enforced dynamically at runtime.

- Trigger: A dataset defines a `facets` array. When present, the mapping prompt
  includes the schema and the LLM response is validated against it.
- Enforcement: If any facet is marked `required: true` for a dataset, the
  `facets` object is mandatory on every contribution for that dataset. Values
  are validated by type and enum membership.
- Flexibility: You can add, remove, or change facets in the dataset files
  without modifying code. Around 10 facets per dataset is a practical default.

## Where to configure facets

Add a `facets` array to the object‑form dataset file:

```json
{
  "field": "applications",
  "label": "Applications",
  "facets": [
    { "id": "service_type", "label": "ServiceType", "type": "enum", "required": true,
      "options": [ { "id": "ui", "label": "UI" }, { "id": "api", "label": "API" } ]
    }
  ],
  "items": [ { "id": "APP001", "name": "Learning Platform", "description": "..." } ]
}
```

Types supported: `string`, `integer`, `number`, `boolean`, `enum` (with
`options` id/label pairs). For `enum`, the value must be one of the declared
`options[].id`.

## Defaults provided in this repo

The repository includes facet schemas for three datasets. You can adapt them as
needed.

### Applications (`data/applications.json`)
- Required: `ServiceType`, `ExecutionModel`, `Placement`, `Plane (primary)`,
  `Exposure`, `SecurityPosture` (single string).
- Optional: `SLOs`, `Scaling`, `Tenancy`, `DependsOn`.

### Technologies (`data/technologies.json`)
- Required: `ExecutionModel`, `Placement`, `Plane (primary)`, `TrustModel`,
  `TelemetryBaseline`, `PrivacyBaseline (in-flight)`.
- Optional: `Acceleration`, `DataShape`, `Connectivity (box-level)`,
  `AutomationLevel`.

### Data Assets (`data/information.json` → `field: "data"`)
- Required: `AssetType`, `Domain/Context`, `Shape`, `Sensitivity/Class`,
  `Mastership`, `OwningApp`, `Freshness/Mode`, `FreshnessSLA`,
  `Residency/Region`, `Retention`, `Quality/DQ SLOs Present`, `Lineage Captured`.
- Optional: `PIITypes`, `IdentifierStrategy`, `ContractualSchema`,
  `VersioningPolicy`.

## Example contribution with facets

```json
{
  "item": "APP001",
  "facets": {
    "service_type": "ui",
    "execution_model": "container",
    "placement": "cloud",
    "plane": "control",
    "exposure": "internal",
    "security_posture": "AuthN=OIDC; AuthZ=RBAC; DataSensitivity=internal",
    "slos": "99.9; 250ms; RPO=1h; RTO=4h",
    "scaling": "auto"
  }
}
```

## How validation works

At runtime, the app builds a Pydantic model from the dataset’s `facets` schema
and validates each contribution. When `--strict-mapping` (or
`--fail-on-quarantine`) is enabled, missing required facets or invalid values
cause the run to fail; otherwise they are quarantined and logged.

