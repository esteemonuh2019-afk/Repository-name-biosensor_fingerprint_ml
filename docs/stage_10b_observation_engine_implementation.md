# Stage 10B Scientific Observation Engine Implementation

## Purpose

Stage 10B implements the production Scientific Observation Engine on top of the frozen BSIP
2.0 observation contract in `src/scientific_reasoning/observation/`.

The engine converts a validated supervisor-results package into factual, provenance-backed
`Observation` objects. It does not interpret biological meaning, judge model performance,
recommend experiments, compare with literature, generate hypotheses, or claim publication
readiness.

## Architecture

Implementation modules:

- `source_loader.py` strictly loads the supplied supervisor-results directory.
- `rules.py` creates deterministic factual observations from loaded source payloads.
- `engine.py` orchestrates loading, rule execution, validation, and writing.
- `writers.py` writes deterministic JSON, CSV, Markdown, provenance, validation, and summary outputs.
- `scripts/build_scientific_observations.py` exposes the command-line interface.

Contract modules reused directly:

- `models.py`
- `enums.py`
- `interfaces.py`
- `registry.py`
- `validators.py`

The implementation does not duplicate the frozen contract models.

## Source Authority Rules

The engine accepts a user-specified supervisor-results directory. The default compatibility
target is:

```text
outputs/supervisor_results_2/
```

Required source files:

- `supervisor_results_summary.json`
- `provenance_index.csv`
- `report_validation.json`

Optional contextual source files:

- `selected_tables.csv`
- `selected_figures.csv`

Missing required files produce critical validation issues. Missing optional files produce
warnings. The loader never searches unrelated folders, never substitutes a different run, and
never mutates supervisor-results source files.

## Observation Generation Rules

The engine generates deterministic observations for:

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

Observation identifiers follow the frozen contract:

```text
OBS-{CATEGORY_TOKEN}-0001
```

Example IDs:

- `OBS-DATASET-0001`
- `OBS-QC-0001`
- `OBS-CLASSIFICATION-0001`
- `OBS-REGRESSION-0001`

## Provenance Requirements

`provenance_index.csv` is authoritative for quantitative supporting metrics.

Rules enforced:

- Every quantitative `SupportingMetric` with a value must have a matching provenance ID.
- Supporting metric values and provenance metric values must agree.
- Classification metrics must belong to the selected classifier.
- Regression metrics must belong to the selected regressor.
- Ranking-table metrics may appear as comparison context but cannot replace selected-model metrics.
- Units are preserved exactly from provenance where available.
- Missing provenance marks the observation incomplete and emits validation issues.
- The engine never invents provenance IDs.

Contextual values without provenance, such as selected figure lists or optional row counts, are
stored in observation metadata rather than as quantitative supporting metrics.

## Confidence Rules

- `HIGH`: validated package, required sources present, and complete matching provenance.
- `MODERATE`: valid source and factual value present, but optional contextual indexes are absent.
- `LOW`: evidence incomplete or ambiguous.
- `NOT_ASSESSABLE`: required evidence missing.

## CLI Usage

```powershell
python scripts\build_scientific_observations.py `
  --project-root "." `
  --supervisor-results "outputs\supervisor_results_2" `
  --output-dir "outputs\scientific_observations" `
  --overwrite
```

Options:

- `--overwrite`: replace only the specified Observation Engine output directory.
- `--software-version`: set the emitted software-version string.

Exit code behavior:

- `0`: generation completed and critical/error validation passed.
- non-zero: required source missing, critical validation issue, model coherence issue, missing provenance, unreadable output, or non-overwrite protection failure.

## Outputs

The engine writes:

- `observations.json`
- `observations.csv`
- `observations.md`
- `observation_validation.json`
- `observation_provenance.csv`
- `observation_summary.json`

`observations.json` contains schema version, software version, source supervisor directory,
generation timestamp, deterministic observations, and validation summary.

`observations.csv` uses one row per observation. Nested values are serialized as deterministic
JSON strings.

`observations.md` groups observations by category and shows ID, title, statement, status,
confidence, analysis stage, supporting metrics, supporting files, provenance IDs, and
limitations.

`observation_validation.json` records validation counts, readability checks, and structured
validation issues.

`observation_summary.json` records observation counts, source-file status, selected model names,
blind-label availability, and validation result.

## Failure Behavior

The engine fails gracefully:

- Missing required source files produce critical validation issues and non-zero CLI exit.
- Missing optional files produce warnings.
- Non-empty output directories are refused unless `--overwrite` is supplied.
- With `--overwrite`, only the specified output directory is removed, after verifying it is
  inside the project root and is not the project root itself.
- Source supervisor-results files are never modified.

## Scientific Non-Goals

The Observation Engine does not:

- interpret biological meaning
- assess whether results are good or poor
- recommend experiments
- compare with literature
- generate hypotheses
- claim publication readiness
- rerun analyses
- retrain models

## Assumptions

- The supervisor-results package has already passed its own validation.
- The provenance index is authoritative for quantitative observation claims.
- Missing contextual selected-table or selected-figure indexes do not invalidate quantitative
  observations.
- External source files referenced by the supervisor package are preserved as paths; the
  Observation Engine does not search for substitutes.
