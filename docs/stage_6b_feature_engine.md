# Stage 6B Core Feature Extraction Engine

## Purpose

Stage 6B implements the first canonical feature extraction engine for biosensor luminescence time series. The engine consumes an already-built canonical dataset and produces one feature vector per canonical measurement series.

This stage does not implement PCA, clustering, machine learning, plotting, Recommended features, Optional features, or Experimental features.

## Input Contract

The engine entry point is:

```python
extract_features(canonical_dataframe)
```

The input must already follow the canonical schema. The feature engine does not read raw CSV files, read Excel workbooks, or call raw-data importers. Series are grouped by:

```text
Experiment_ID, Source_File, Measurement_Unit_ID
```

This matches the Stage 5 measurement-identity design.

## Output Contract

`extract_features` returns a `FeatureDataset` object with:

- `dataframe`
- `metadata`
- `summary`
- `qc`

The feature dataframe retains:

- `Experiment_ID`
- `Measurement_Unit_ID`
- `Source_File`
- `Strain`
- `Chemical`
- `Concentration`
- `Replicate_ID`
- `Duration`
- `QC_Status`

The package also supports writing:

- `outputs/features/feature_dataset.csv`
- `outputs/features/feature_summary.json`
- `outputs/features/feature_qc_report.md`

## Implemented Core Features

| Feature column | Definition | Units |
|---|---|---|
| `baseline` | First valid raw luminescence value after sorting by `Time_Minutes` and `Source_Row_ID`. | RLU |
| `peak` | Maximum valid `Luminescence_Raw` in the series. | RLU |
| `minimum` | Minimum valid `Luminescence_Raw` in the series. | RLU |
| `endpoint` | Raw luminescence at the largest valid `Time_Minutes`; null if the endpoint timestamp has conflicting values. | RLU |
| `dynamic_range` | `peak - minimum`. | RLU |
| `time_to_peak` | Earliest `Time_Minutes` at which `peak` occurs. | minutes |
| `auc` | Trapezoidal area under raw luminescence over sorted time. Null when duplicate timestamps make time intervals unresolved. | RLU * minutes |
| `initial_slope` | `(y_2 - y_1) / (t_2 - t_1)` for the first two valid ordered observations. Null when duplicate timestamps or insufficient points prevent calculation. | RLU per minute |
| `maximum_slope` | Maximum adjacent slope across valid ordered observations. Null when duplicate timestamps or insufficient points prevent calculation. | RLU per minute |
| `fold_change` | Signed peak fold change, `(peak - baseline) / baseline`, when baseline is positive. | unitless |
| `log2_fold_change` | Endpoint log2 fold change, `log2(endpoint / baseline)`, when baseline and endpoint are positive. | log2 unitless ratio |

## QC Behavior

The engine flags problems instead of silently correcting them.

Feature QC validates:

- missing feature values
- infinite feature values
- impossible time-to-peak values
- zero baseline before fold-change calculation
- negative time values
- empty or unusable series
- duplicated `Measurement_Unit_ID` rows in feature output
- duplicate timestamps within a series
- conflicting duplicate timestamp values
- source QC warning or failure rows

Duplicate timestamps are not averaged. If duplicate timestamps occur, `auc`, `initial_slope`, and `maximum_slope` are set to null and the feature row is flagged. If duplicated timestamps have conflicting luminescence values, the feature row fails QC.

Invalid canonical rows are excluded from numerical feature calculations, but they remain represented through input row counts, missing observation counts, source QC fields, and feature QC flags.

## Validation Strategy

Unit tests cover synthetic curves with known:

- baseline
- peak
- minimum
- endpoint
- dynamic range
- time to peak
- AUC
- initial slope
- maximum slope
- fold change
- log2 fold change

Additional tests cover:

- metadata retention
- one feature row per canonical series
- zero baseline denominator safety
- conflicting duplicate timestamps
- identical duplicate timestamps
- missing and infinite observations
- negative time values
- empty canonical datasets
- impossible time-to-peak QC
- duplicate `Measurement_Unit_ID` QC

The integration test proves:

```text
Canonical builder output
  -> feature engine
  -> FeatureDataset
```

## Limitations

- The engine implements only the 11 Stage 6B core features requested in the task, not all Core-priority features named in the broader Stage 6A specification.
- Baseline is currently the first valid observation, not a windowed robust baseline.
- Duplicate timestamps are flagged and block affected time-interval features; they are not resolved.
- No smoothing is applied.
- No control normalization is applied.
- No Recommended, Optional, or Experimental features are implemented.
- `Measurement_Unit_ID` is evaluated for duplicate output rows, but canonical identity remains the full series key: `Experiment_ID`, `Source_File`, `Measurement_Unit_ID`.
- Real-data outputs remain affected by Stage 5C measurement-identity conflicts until Stage 5D corrects the canonical builder.

## Future Extensions

Future stages should add:

- Stage 5D measurement-unit correction before interpreting conflict-affected real-data features.
- Robust baseline-window estimation.
- Explicit duplicate-timestamp resolution policy.
- Recommended features from `docs/feature_specification.md`.
- Feature-level confidence scoring.
- Normalized features after normalization policy is approved.
- Real-data validation summaries by strain, chemical, concentration, source file, and QC status.

