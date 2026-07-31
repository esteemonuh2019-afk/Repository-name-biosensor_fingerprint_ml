# BSIP v2.2.0 Hypothesis Engine

## Purpose

The BSIP Hypothesis Engine consumes validated Scientific Interpretation Engine outputs and generates explicit, testable, evidence-linked scientific hypotheses.

The engine operates downstream of interpretations only. It does not read raw experimental data, parse classification or regression outputs, retrain models, call external AI services, generate manuscript prose, recommend experiments, or present hypotheses as established facts.

## Source Boundary

Default input directory:

```text
outputs/scientific_interpretations/
```

Required files:

- `interpretations.json`
- `interpretation_validation.json`
- `interpretation_summary.json`
- `interpretation_dependencies.csv`

The package is rejected when required files are missing, unreadable, or when the interpretation validation package reports critical validation failure.

## Public Interface

`HypothesisEngine` implements:

```python
load_interpretations() -> tuple[Interpretation, ...]
validate_input_interpretations(
    interpretations: tuple[Interpretation, ...]
) -> tuple[HypothesisValidationIssue, ...]
build_hypotheses(
    interpretations: tuple[Interpretation, ...]
) -> tuple[Hypothesis, ...]
validate_hypotheses(
    hypotheses: tuple[Hypothesis, ...],
    interpretations: tuple[Interpretation, ...]
) -> tuple[HypothesisValidationIssue, ...]
write_outputs(
    hypotheses: tuple[Hypothesis, ...]
) -> tuple[Path, ...]
run() -> HypothesisRunResult
```

## Hypothesis Model

`Hypothesis` is an immutable dataclass with:

- `hypothesis_id`
- `category`
- `title`
- `statement`
- `status`
- `confidence`
- `supporting_interpretation_ids`
- `contradicting_interpretation_ids`
- `supporting_observation_ids`
- `assumptions`
- `alternative_hypothesis_ids`
- `evidence_gaps`
- `falsifiability_statement`
- `rationale`
- `reasoning_rule_ids`
- `priority_score`
- `priority`
- `created_at`
- `software_version`
- `source_interpretation_schema_version`
- `tags`
- `metadata`

The explicit `priority` field records the score-derived `HypothesisPriority` enum required by BSIP v2.2.0.

## Enumerations

`HypothesisCategory`:

- `TEMPORAL_INFORMATION`
- `CHEMICAL_DISCRIMINATION`
- `CONCENTRATION_ENCODING`
- `FEATURE_REPRESENTATION`
- `STRAIN_CONTRIBUTION`
- `DATA_QUALITY_EFFECT`
- `GENERALIZATION`
- `OVERALL_SYSTEM_BEHAVIOR`

`HypothesisStatus`:

- `PLAUSIBLE`
- `COMPETING`
- `WEAKLY_SUPPORTED`
- `CONFLICTED`
- `INSUFFICIENT_EVIDENCE`
- `NOT_ASSESSABLE`

`HypothesisConfidence`:

- `HIGH`
- `MODERATE`
- `LOW`
- `NOT_ASSESSABLE`

`HypothesisPriority`:

- `HIGH`
- `MEDIUM`
- `LOW`
- `NOT_ASSESSABLE`

## Deterministic IDs

Primary deterministic IDs:

- `HYP-TEMPORAL_INFORMATION-0001`
- `HYP-CHEMICAL_DISCRIMINATION-0001`
- `HYP-CONCENTRATION_ENCODING-0001`
- `HYP-FEATURE_REPRESENTATION-0001`
- `HYP-STRAIN_CONTRIBUTION-0001`
- `HYP-DATA_QUALITY_EFFECT-0001`
- `HYP-GENERALIZATION-0001`
- `HYP-OVERALL_SYSTEM_BEHAVIOR-0001`

Competing hypotheses use deterministic second records within the same category, such as `HYP-CHEMICAL_DISCRIMINATION-0002`.

## Implemented Rules

Implemented stable rule IDs:

- `RULE-TEMPORAL-INFORMATION-001`
- `RULE-CHEMICAL-DISCRIMINATION-001`
- `RULE-CONCENTRATION-ENCODING-001`
- `RULE-FEATURE-REPRESENTATION-001`
- `RULE-STRAIN-CONTRIBUTION-001`
- `RULE-DATA-QUALITY-EFFECT-001`
- `RULE-GENERALIZATION-001`
- `RULE-OVERALL-SYSTEM-BEHAVIOR-001`

Rules generate hypotheses only when supporting Interpretation IDs are present. Competing hypotheses are linked reciprocally through `alternative_hypothesis_ids`.

## Scientific Boundary

Allowed hypothesis language includes:

- `may`
- `might`
- `could`
- `is consistent with the possibility that`
- `remains plausible`
- `cannot yet distinguish between`

Forbidden language includes:

- `proves`
- `confirms`
- `demonstrates conclusively`
- `is caused by`
- `mechanism is`
- `definitely`
- `certainly`
- `establishes`
- `should test`
- `recommend`
- `future experiment`
- `ought to perform`
- `publication-ready`

The engine does not invent pathways, mechanisms, literature comparisons, deployment claims, recommendations, or detailed experimental protocols.

## Falsifiability

Every `PLAUSIBLE`, `COMPETING`, or `WEAKLY_SUPPORTED` hypothesis must include a non-empty falsifiability statement. The statement must describe evidence that would weaken or contradict the hypothesis without prescribing a detailed protocol.

## Confidence Policy

`HIGH` requires multiple coherent interpretations, no important contradiction, direct evidence linkage, and no major unresolved external-validation gap.

`MODERATE` applies when coherent support exists but alternatives remain plausible, evidence is indirect, or evidence is internally validated only.

`LOW` applies when one interpretation supports the hypothesis, substantial evidence gaps exist, or QC/generalization limitations affect the claim.

`NOT_ASSESSABLE` applies when required interpretations are absent or invalid.

High confidence is not used for speculative mechanistic explanations.

## Priority Score

Priority score is deterministic from 0 to 100 and uses only:

- number of supporting interpretations,
- confidence of supporting interpretations,
- degree of contradiction,
- evidence-gap count,
- relevance to the primary biosensor research questions.

Priority category:

- `HIGH`: 70-100
- `MEDIUM`: 40-69
- `LOW`: 1-39
- `NOT_ASSESSABLE`: 0

Priority does not use novelty claims or literature comparisons.

## Outputs

The engine writes:

- `hypotheses.json`
- `hypotheses.csv`
- `hypotheses.md`
- `hypothesis_validation.json`
- `hypothesis_summary.json`
- `hypothesis_dependencies.csv`
- `hypothesis_competition_map.csv`

Without `--overwrite`, non-empty output directories are refused. With `--overwrite`, only the specified hypothesis output directory is replaced.

## CLI

```powershell
python scripts\build_scientific_hypotheses.py `
  --project-root "." `
  --interpretations-dir "outputs\scientific_interpretations" `
  --output-dir "outputs\scientific_hypotheses"
```

Optional flags:

- `--overwrite`
- `--software-version`

## Non-Goals

This stage does not implement:

- recommendation generation,
- experimental method prescription,
- manuscript text,
- literature review,
- external AI narrative generation,
- raw analysis loading,
- model training or result regeneration.
