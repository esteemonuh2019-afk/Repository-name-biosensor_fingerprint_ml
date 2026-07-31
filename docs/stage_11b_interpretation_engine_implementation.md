# Stage 11B Interpretation Engine Implementation

## Purpose

Stage 11B implements the BSIP Scientific Interpretation Engine. The engine consumes validated Observation Engine outputs and emits conservative scientific interpretations linked back to Observation IDs.

The implementation operates at reasoning Level 3 only. It does not read raw experimental data, parse classification or regression result files, retrain models, call external AI services, generate hypotheses, recommend experiments, or generate manuscript prose.

## Source Boundary

The engine reads only a user-specified Observation Engine output directory. The default compatibility target is:

```text
outputs/scientific_observations/
```

Required files:

- `observations.json`
- `observation_validation.json`
- `observation_summary.json`
- `observation_provenance.csv`

The loader reconstructs public `Observation` objects from `observations.json` and checks the package validation report. Missing required files, unreadable packages, `critical_issue_count > 0`, or `validation_passed = false` produce critical structured issues and prevent output generation.

## Implemented Modules

- `src/scientific_reasoning/interpretation/source_loader.py`
- `src/scientific_reasoning/interpretation/confidence.py`
- `src/scientific_reasoning/interpretation/rules.py`
- `src/scientific_reasoning/interpretation/engine.py`
- `src/scientific_reasoning/interpretation/writers.py`
- `scripts/build_scientific_interpretations.py`

The implementation uses the existing interpretation models, enums, interfaces, registry, validators, and policy helpers.

## Deterministic Rules

The engine implements the following stable rule IDs and interpretation IDs:

| Rule ID | Interpretation ID | Category |
|---|---|---|
| `RULE-DATA-QUALITY-001` | `INT-DATA_QUALITY-0001` | `DATA_QUALITY` |
| `RULE-FINGERPRINT-STRUCTURE-001` | `INT-FINGERPRINT_STRUCTURE-0001` | `FINGERPRINT_STRUCTURE` |
| `RULE-CHEMICAL-CLASSIFICATION-001` | `INT-CHEMICAL_CLASSIFICATION-0001` | `CHEMICAL_CLASSIFICATION` |
| `RULE-CONCENTRATION-REGRESSION-001` | `INT-CONCENTRATION_REGRESSION-0001` | `CONCENTRATION_REGRESSION` |
| `RULE-FEATURE-ENGINEERING-001` | `INT-FEATURE_ENGINEERING-0001` | `FEATURE_ENGINEERING` |
| `RULE-FEATURE-SELECTION-001` | `INT-FEATURE_SELECTION-0001` | `FEATURE_SELECTION` |
| `RULE-STRAIN-CONTRIBUTION-001` | `INT-STRAIN_CONTRIBUTION-0001` | `STRAIN_CONTRIBUTION` |
| `RULE-BLIND-VALIDATION-001` | `INT-BLIND_VALIDATION-0001` | `BLIND_VALIDATION` |
| `RULE-OVERALL-EVIDENCE-001` | `INT-OVERALL_EVIDENCE-0001` | `OVERALL_EVIDENCE` |

Rules generate interpretations only when supporting observations exist. The data-quality rule additionally requires active QC limitations such as errors, warnings, excluded rows, active limitations, or package-validation limitations.

## Confidence and Status

Confidence is assigned by rule-based observation coherence:

- multiple coherent `COMPLETE` / `HIGH` observations can support `HIGH`,
- one complete high-confidence observation supports `MODERATE`,
- incomplete, indirect, contradictory, or critically limited evidence lowers confidence,
- absent or invalid required evidence is `NOT_ASSESSABLE`.

Metric magnitude alone never determines confidence.

Blind-prediction observations with absent true labels produce a `PARTIALLY_SUPPORTED` blind-validation boundary claim rather than an external-validation claim.

## Outputs

The engine writes exactly six files:

- `interpretations.json`
- `interpretations.csv`
- `interpretations.md`
- `interpretation_validation.json`
- `interpretation_summary.json`
- `interpretation_dependencies.csv`

Without `--overwrite`, the writer refuses to replace a non-empty output directory. With `--overwrite`, it replaces only the specified output directory after verifying that it is inside the project root and is not the project root itself.

## CLI

```powershell
python scripts\build_scientific_interpretations.py `
  --project-root "." `
  --observations-dir "outputs\scientific_observations" `
  --output-dir "outputs\scientific_interpretations"
```

Optional flags:

- `--overwrite`
- `--software-version`

The CLI prints a concise JSON execution summary. It returns exit code `0` when generation succeeds and critical validation passes, and nonzero when required files are missing, the source package is critically invalid, the output directory is protected, or critical interpretation validation fails.

## Validation Outputs

`interpretation_validation.json` includes:

- `validation_passed`
- `critical_issue_count`
- `warning_count`
- `missing_dependency_count`
- `unsupported_claim_count`
- `causal_language_issue_count`
- `recommendation_language_issue_count`
- `hypothesis_language_issue_count`
- `blind_validation_overclaim_count`
- `confidence_policy_issue_count`
- structured validation issues
- output readability checks

## Scientific Boundaries

Generated claims avoid causal, mechanistic, recommendation, hypothesis, literature-comparison, publication-readiness, clinical, regulatory, and field-deployment language.

The engine does not describe model performance as excellent, strong, weak, poor, successful, or failed. It uses conservative language such as `suggests`, `indicates`, `associated with`, and `remains limited by`.

## Test Coverage

Integration tests use temporary synthetic Observation packages and do not depend on production output folders. They cover:

- successful generation,
- deterministic ordering and serialization,
- missing required files,
- critically invalid packages,
- missing observation dependency validation,
- classification wording,
- regression wording when `0 < R-squared < 0.5`,
- blind-label absence wording,
- causal, recommendation, and hypothesis rejection,
- confidence-policy behavior,
- overwrite protection,
- output readability,
- summary consistency,
- dependency-table consistency.

## Non-Goals

This stage does not implement:

- hypothesis generation,
- experiment recommendations,
- manuscript prose,
- external AI narrative generation,
- raw analysis source loading,
- model training or result regeneration.
