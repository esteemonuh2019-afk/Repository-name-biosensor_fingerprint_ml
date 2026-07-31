# BSIP Scientific Interpretation Engine API

Version: BSIP 2.1.0

## Purpose

The Scientific Interpretation Engine is the Level 3 reasoning contract for BSIP. It consumes validated `Observation` objects and produces conservative, evidence-grounded scientific interpretations.

The engine preserves traceability by making every interpretation structurally dependent on Observation IDs. Interpretation records must not replace Observation evidence values as authoritative sources.

## Scientific Boundary

The Interpretation Engine may state evidence-grounded meaning, including:

- observed results suggest discriminative information is present,
- temporal features are associated with improved model performance,
- concentration-related signal is present but incomplete,
- selected strain-ablation results indicate differential contribution,
- QC issues limit confidence,
- absence of blind labels prevents external validation claims.

The Interpretation Engine must not:

- claim causation,
- invent biological mechanisms,
- compare against literature,
- claim publication readiness,
- recommend experiments or actions,
- generate hypotheses,
- make clinical, regulatory, or field-deployment claims,
- state statistical significance unless significance evidence is present,
- use words such as `proves` or `confirms` unless directly supported.

## Reasoning Levels

Level 1: Data  
Raw measurements and completed source outputs.

Level 2: Observation  
Validated factual statements extracted from completed outputs.

Level 3: Interpretation  
Evidence-grounded scientific meaning derived from validated observations.

Level 4: Hypothesis  
Testable explanations requiring future evidence.

Level 5: Recommendation  
Proposed actions, experiments, or deployment choices.

The Interpretation Engine operates only at Level 3.

## Public API

The abstract `InterpretationEngine` interface defines:

```python
load_observations() -> tuple[Observation, ...]

validate_input_observations(
    observations: tuple[Observation, ...]
) -> tuple[InterpretationValidationIssue, ...]

build_interpretations(
    observations: tuple[Observation, ...]
) -> tuple[Interpretation, ...]

validate_interpretations(
    interpretations: tuple[Interpretation, ...],
    observations: tuple[Observation, ...]
) -> tuple[InterpretationValidationIssue, ...]

write_outputs(
    interpretations: tuple[Interpretation, ...]
) -> tuple[Path, ...]

run() -> InterpretationRunResult
```

No file-reading, raw-output parsing, or production interpretation generation is implemented in the model layer.

## Model Definitions

### Interpretation

Immutable dataclass:

```python
@dataclass(frozen=True)
class Interpretation:
    interpretation_id: str
    category: InterpretationCategory
    title: str
    claim: str
    status: InterpretationStatus
    confidence: InterpretationConfidence
    supporting_observation_ids: tuple[str, ...]
    contradicting_observation_ids: tuple[str, ...]
    assumptions: tuple[str, ...]
    limitations: tuple[str, ...]
    evidence_summary: tuple[InterpretationEvidenceLink, ...]
    reasoning_rule_ids: tuple[str, ...]
    created_at: str
    software_version: str
    source_observation_schema_version: str | None
    tags: tuple[str, ...]
    metadata: Mapping[str, Any]
```

### InterpretationEvidenceLink

Immutable dataclass:

```python
@dataclass(frozen=True)
class InterpretationEvidenceLink:
    observation_id: str
    direction: EvidenceDirection
    rationale: str
    metric_names: tuple[str, ...]
    provenance_ids: tuple[str, ...]
    source_files: tuple[str, ...]
```

Evidence links summarize why an Observation supports, contradicts, or contextualizes an interpretation. They must not duplicate underlying metric values as authoritative replacements.

### ReasoningRule

Immutable dataclass:

```python
@dataclass(frozen=True)
class ReasoningRule:
    rule_id: str
    name: str
    description: str
    required_categories: tuple[InterpretationCategory, ...]
    optional_categories: tuple[InterpretationCategory, ...]
    minimum_supporting_observations: int
    allowed_claim_template: str | None
    forbidden_terms: tuple[str, ...]
    confidence_policy: str
    limitation_policy: str
    enabled: bool
```

Rules support deterministic interpretation before any future AI-assisted narrative layer.

### InterpretationValidationIssue

Immutable dataclass:

```python
@dataclass(frozen=True)
class InterpretationValidationIssue:
    code: str
    severity: ReasoningSeverity
    message: str
    interpretation_id: str | None
    field: str | None
    observation_id: str | None
    rule_id: str | None
```

### InterpretationRunResult

Immutable dataclass:

```python
@dataclass(frozen=True)
class InterpretationRunResult:
    interpretations: tuple[Interpretation, ...]
    validation_issues: tuple[InterpretationValidationIssue, ...]
    output_paths: tuple[Path, ...]
    metadata: Mapping[str, Any]
```

## Enumerations

`InterpretationCategory`:

- `DATASET_SCOPE`
- `DATA_QUALITY`
- `FINGERPRINT_STRUCTURE`
- `EXPLORATORY_STRUCTURE`
- `CHEMICAL_CLASSIFICATION`
- `CONCENTRATION_REGRESSION`
- `FEATURE_ENGINEERING`
- `FEATURE_SELECTION`
- `STRAIN_CONTRIBUTION`
- `BLIND_VALIDATION`
- `OVERALL_EVIDENCE`

`InterpretationStatus`:

- `SUPPORTED`
- `PARTIALLY_SUPPORTED`
- `CONFLICTED`
- `INSUFFICIENT_EVIDENCE`
- `NOT_ASSESSABLE`

`InterpretationConfidence`:

- `HIGH`
- `MODERATE`
- `LOW`
- `NOT_ASSESSABLE`

`EvidenceDirection`:

- `SUPPORTING`
- `CONTRADICTING`
- `CONTEXTUAL`

`ReasoningSeverity`:

- `INFO`
- `WARNING`
- `CRITICAL`

## ID Convention

Interpretation IDs use:

```text
INT-{CATEGORY}-{NUMBER}
```

The number must contain four digits.

Examples:

- `INT-CHEMICAL_CLASSIFICATION-0001`
- `INT-CONCENTRATION_REGRESSION-0001`
- `INT-DATA_QUALITY-0001`

Category-token mapping is one-to-one with `InterpretationCategory.value`.

## Lifecycle

1. Load validated Observation objects.
2. Validate source observations for structural usability.
3. Build interpretations from Observation IDs and reasoning rules.
4. Validate interpretation structure, dependencies, language, status, confidence, and serialization.
5. Write downstream outputs.

The current stage defines the contract only. It does not implement production interpretation generation.

## Confidence Rules

`HIGH`:

- at least two coherent supporting observations,
- no contradictory observations,
- supporting observations are `COMPLETE` and `HIGH` confidence,
- no active critical QC limitation directly affects the claim.

`MODERATE`:

- one strong observation, or multiple observations with contextual limitations,
- no unresolved contradiction,
- evidence is complete but external validation may be absent.

`LOW`:

- evidence is incomplete,
- supporting observations have `LOW` or `MODERATE` confidence,
- active QC issues affect the interpretation,
- evidence is indirect.

`NOT_ASSESSABLE`:

- required observations are absent,
- observation validation failed critically,
- supporting evidence cannot be resolved.

Confidence must not be assigned from model-performance magnitude alone.

## Status Rules

`SUPPORTED` means required evidence is present and coherent.

`PARTIALLY_SUPPORTED` means some evidence supports the claim but important context is missing.

`CONFLICTED` means supporting and contradicting observations are both present.

`INSUFFICIENT_EVIDENCE` means the minimum evidence threshold is not met.

`NOT_ASSESSABLE` means dependencies are unavailable or invalid.

## Validation Rules

Validators return structured `InterpretationValidationIssue` objects. They detect:

1. invalid interpretation ID,
2. duplicate interpretation ID,
3. missing supporting observation,
4. nonexistent observation dependency,
5. interpretation without evidence,
6. unsupported confidence assignment,
7. claim containing forbidden causal language,
8. claim containing recommendation language,
9. hypothesis wording,
10. literature-comparison wording,
11. blind-validation overclaim,
12. contradiction not recorded,
13. non-serializable metadata,
14. deterministic ordering violations.

## Serialization

Canonical JSON serialization rules:

- UTF-8,
- ISO 8601 timestamps,
- enum values serialized as strings,
- deterministic ordering,
- supporting observation IDs sorted,
- contradicting observation IDs sorted,
- missing values represented as `null`,
- no unserializable Python objects.

## Example Interpretations

Classification example:

```json
{
  "interpretation_id": "INT-CHEMICAL_CLASSIFICATION-0001",
  "category": "CHEMICAL_CLASSIFICATION",
  "claim": "The available classification observations suggest that the biosensor fingerprints contain information associated with chemical-class discrimination.",
  "supporting_observation_ids": ["OBS-CLASSIFICATION-0001"],
  "limitations": ["No external blind validation labels were available."]
}
```

Regression example:

```json
{
  "interpretation_id": "INT-CONCENTRATION_REGRESSION-0001",
  "category": "CONCENTRATION_REGRESSION",
  "claim": "The regression observations indicate that concentration-related information is present, while a substantial proportion of target variance remains unaccounted for.",
  "supporting_observation_ids": ["OBS-REGRESSION-0001"]
}
```

Feature-engineering example:

```json
{
  "interpretation_id": "INT-FEATURE_ENGINEERING-0001",
  "category": "FEATURE_ENGINEERING",
  "claim": "The feature-engineering observations indicate that the selected temporal feature family was associated with higher reported classification and regression benchmark values than the reference feature configuration.",
  "supporting_observation_ids": ["OBS-FEATURE_ENGINEERING-0001"]
}
```

Blind-validation example:

```json
{
  "interpretation_id": "INT-BLIND_VALIDATION-0001",
  "category": "BLIND_VALIDATION",
  "claim": "The available blind-prediction observations do not establish external validation performance because true labels were absent.",
  "supporting_observation_ids": ["OBS-BLIND_PREDICTION-0001"]
}
```

## Input Compatibility

Future implementations must consume the Observation API. They may load Observation objects produced from:

- `outputs/scientific_observations/observations.json`
- `outputs/scientific_observations/observation_validation.json`
- `outputs/scientific_observations/observation_summary.json`
- `outputs/scientific_observations/observation_provenance.csv`

These paths must not be hard-coded into the model layer.

## Downstream Compatibility

The contract supports later modules:

- Hypothesis Engine,
- Supervisor Engine,
- Reviewer Engine,
- Manuscript Engine,
- optional AI-assisted narrative generation.

Downstream modules must treat Interpretation records as Level 3 reasoning artifacts, not as hypotheses, recommendations, manuscript text, or raw evidence.

## Non-Goals

This contract does not:

- implement production interpretation logic,
- call external AI APIs,
- read raw experimental data,
- parse classification, regression, or QC outputs directly,
- retrain models,
- regenerate analyses,
- generate manuscript prose,
- recommend experiments.
