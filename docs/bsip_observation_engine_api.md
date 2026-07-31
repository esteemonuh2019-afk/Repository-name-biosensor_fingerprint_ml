# BSIP 2.0 Scientific Observation Engine API

## Purpose

The BSIP 2.0 Scientific Observation Engine contract defines how completed, validated
analysis outputs will be converted into structured factual observations.

This stage is design-only. It defines schemas, enums, interfaces, validation contracts,
registry behavior, and serialization requirements. It does not implement source loading,
analysis reruns, model training, interpretation, or report generation.

## Scientific Boundary

The Observation Engine may state factual items when evidence is present:

- selected model identity
- metric values
- row counts
- QC status
- file presence
- selected feature family
- strain-ablation result values

The Observation Engine may not state:

- whether performance is good or poor
- biological meaning
- causal explanation
- publication readiness
- recommendations
- literature comparisons
- hypotheses

Interpretation, hypotheses, recommendations, supervisor narrative, reviewer response, and
manuscript text belong to later reasoning modules.

## Public API

The public contract lives in `src/scientific_reasoning/observation/`.

Primary imports:

```python
from src.scientific_reasoning.observation import (
    ConfidenceLevel,
    Observation,
    ObservationCategory,
    ObservationEngine,
    ObservationRegistry,
    ObservationStatus,
    ProvenanceRecord,
    SupportingMetric,
    ValidationIssue,
    validate_observation,
    validate_observations,
)
```

### ObservationEngine Interface

Concrete engines must implement:

- `load_sources() -> Mapping[str, Any]`
- `build_observations(sources: Mapping[str, Any]) -> tuple[Observation, ...]`
- `validate_observations(observations: tuple[Observation, ...]) -> tuple[ValidationIssue, ...]`
- `write_outputs(observations: tuple[Observation, ...]) -> tuple[Path, ...]`
- `run() -> ObservationRunResult`

This interface defines lifecycle shape only. The contract layer does not read files.

### ObservationRegistry

`ObservationRegistry` provides minimal deterministic behavior:

- registers observations
- rejects duplicate IDs
- rejects invalid observation IDs
- returns observations by category
- returns deterministic `observation_id` ordering
- exports JSON-serializable records
- exposes structured validation issues

## Object Models

### Observation

Immutable dataclass fields:

| Field | Type | Required | Description |
|---|---|---:|---|
| `observation_id` | `str` | yes | Stable identifier matching `OBS-{CATEGORY}-{NUMBER}`. |
| `category` | `ObservationCategory` | yes | Stable observation category enum. |
| `title` | `str` | yes | Short factual title. |
| `statement` | `str` | yes | Factual observation sentence. |
| `status` | `ObservationStatus` | yes | Completeness or assessability status. |
| `analysis_stage` | `str` | yes | Pipeline stage associated with the observation. |
| `supporting_metrics` | `tuple[SupportingMetric, ...]` | no | Structured metric evidence. |
| `supporting_files` | `tuple[str, ...]` | no | Evidence source paths. |
| `provenance_records` | `tuple[ProvenanceRecord, ...]` | no | Claim-level provenance. |
| `confidence` | `ConfidenceLevel` | yes | Evidence confidence. |
| `limitations` | `tuple[str, ...]` | no | Factual limitations affecting the observation. |
| `created_at` | `str` | yes | ISO 8601 timestamp. |
| `software_version` | `str` | yes | Producer software version. |
| `source_run` | `str | None` | no | Source run identifier where applicable. |
| `tags` | `tuple[str, ...]` | no | Stable machine-readable tags. |
| `metadata` | `Mapping[str, Any]` | no | Additional JSON-serializable metadata. |

### SupportingMetric

Fields:

- `metric_name`
- `metric_value`
- `units`
- `model_name`
- `fold_count`
- `sample_count`
- `source_file`
- `source_run`
- `provenance_id`

`metric_value` may be numeric, string, boolean, or `null`.

### ProvenanceRecord

Fields:

- `provenance_id`
- `source_file`
- `source_run`
- `section`
- `claim_text`
- `metric_name`
- `metric_value`
- `units`
- `model_name`
- `table_or_figure_reference`
- `support_status`

### ValidationIssue

Fields:

- `code`
- `severity`
- `message`
- `observation_id`
- `field`
- `source_file`

Validators return structured `ValidationIssue` records rather than only booleans.

## Enumerations

`ObservationCategory` values:

- `DATASET`
- `QUALITY_CONTROL`
- `FINGERPRINT`
- `EXPLORATORY_ANALYSIS`
- `CLASSIFICATION`
- `REGRESSION`
- `FEATURE_ENGINEERING`
- `FEATURE_SELECTION`
- `STRAIN_CONTRIBUTION`
- `BLIND_PREDICTION`
- `VALIDATION`

`ObservationStatus` values:

- `COMPLETE`
- `INCOMPLETE`
- `ACTIVE`
- `FAILED`
- `NOT_ASSESSABLE`

`ConfidenceLevel` values:

- `HIGH`
- `MODERATE`
- `LOW`
- `NOT_ASSESSABLE`

## ID Convention

Observation IDs must match:

```text
OBS-{CATEGORY}-{NUMBER}
```

The numeric section must be four digits.

Examples:

- `OBS-DATASET-0001`
- `OBS-QC-0001`
- `OBS-CLASSIFICATION-0001`
- `OBS-REGRESSION-0001`

Category token mapping:

| Category | ID token |
|---|---|
| `DATASET` | `DATASET` |
| `QUALITY_CONTROL` | `QC` |
| `FINGERPRINT` | `FINGERPRINT` |
| `EXPLORATORY_ANALYSIS` | `EXPLORATORY_ANALYSIS` |
| `CLASSIFICATION` | `CLASSIFICATION` |
| `REGRESSION` | `REGRESSION` |
| `FEATURE_ENGINEERING` | `FEATURE_ENGINEERING` |
| `FEATURE_SELECTION` | `FEATURE_SELECTION` |
| `STRAIN_CONTRIBUTION` | `STRAIN_CONTRIBUTION` |
| `BLIND_PREDICTION` | `BLIND_PREDICTION` |
| `VALIDATION` | `VALIDATION` |

## Lifecycle

Future implementations should follow:

1. Load validated source payloads.
2. Build factual observations.
3. Validate observation contracts.
4. Register observations.
5. Export deterministic JSON, CSV, and/or Markdown records.
6. Pass observations to later reasoning modules.

The contract intentionally does not prescribe concrete file paths or parsing logic.

## Validation Rules

Contract validators cover:

- required fields
- observation ID format and category-token match
- unique observation IDs
- quantitative metric provenance
- model-metric coherence between metrics and provenance
- blind-validation wording boundaries
- allowed status and confidence values
- JSON serializability
- deterministic ordering by observation ID

Missing evidence should produce `INCOMPLETE` observations or structured validation issues,
not crashes.

## Serialization Format

Canonical JSON serialization:

- UTF-8
- sorted keys where practical
- enum values serialized as strings
- deterministic list ordering
- ISO 8601 timestamps
- missing values represented as `null`
- no non-serializable Python objects

## Example

```python
metric = SupportingMetric(
    metric_name="accuracy_mean",
    metric_value=0.740959,
    model_name="Extra Trees",
    fold_count=10,
    sample_count=9485,
    source_file="classification/stage_8a/best_model_metrics.json",
    provenance_id="P-0001",
)

provenance = ProvenanceRecord(
    provenance_id="P-0001",
    source_file="classification/stage_8a/best_model_metrics.json",
    section="Classification",
    claim_text="Classification metadata lists Extra Trees as rank 1.",
    metric_name="accuracy_mean",
    metric_value=0.740959,
    model_name="Extra Trees",
)

observation = Observation(
    observation_id="OBS-CLASSIFICATION-0001",
    category=ObservationCategory.CLASSIFICATION,
    title="Selected classification model",
    statement="Classification metadata lists Extra Trees as rank 1.",
    status=ObservationStatus.COMPLETE,
    analysis_stage="Stage 8A",
    supporting_metrics=(metric,),
    supporting_files=("classification/stage_8a/best_model_metrics.json",),
    provenance_records=(provenance,),
    confidence=ConfidenceLevel.HIGH,
)
```

## Non-Goals

This contract does not:

- parse real output files
- regenerate outputs
- rerun analyses
- train models
- decide scientific meaning
- recommend actions
- write manuscript text

## Downstream Compatibility

Later implementation stages must be able to consume validated supervisor outputs such as:

```text
outputs/supervisor_results_2/
    supervisor_results_summary.json
    provenance_index.csv
    report_validation.json
    selected_tables.csv
    selected_figures.csv
```

These paths are compatibility targets for future source loaders. They are intentionally not
hard-coded into the model layer.
