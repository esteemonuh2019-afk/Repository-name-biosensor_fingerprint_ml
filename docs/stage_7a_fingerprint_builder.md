# Stage 7A Fingerprint Dataset Builder

## Purpose

Stage 7A creates the first canonical fingerprint dataset for downstream biosensor analysis. A fingerprint is the validated numeric feature vector for one measurement unit, paired with enough metadata to preserve scientific traceability.

The fingerprint dataset is intended to become the shared input representation for later PCA, clustering, heatmaps, chemical similarity analysis, machine learning, and blind prediction. Those analyses are not implemented in this stage.

## Architecture

The fingerprint package lives in `src/fingerprint/`.

- `fingerprint_builder.py` builds fingerprints from a `FeatureValidationResult`.
- `fingerprint_dataset.py` stores the original, normalised, and consensus fingerprint tables and writes output artifacts.
- `fingerprint_qc.py` audits fingerprint structure and exclusions.
- `fingerprint_similarity.py` exposes reusable distance functions and matrix writers.

The builder requires Stage 6C validation. It does not accept raw CSV files, Excel workbooks, canonical rows, or unvalidated feature tables directly. Command-line orchestration can build canonical and feature datasets, but the fingerprint builder itself receives only the validation result.

## Fingerprint Definition

One fingerprint represents one validated `Measurement_Unit_ID` in its experimental context. Because `Measurement_Unit_ID` can be duplicated across source files or experiments, the output also includes a stable `Fingerprint_ID`:

```text
Experiment_ID::Source_File::Measurement_Unit_ID
```

The fingerprint table contains metadata and Stage 6B core features only. It does not include raw luminescence time series, time-point rows, source row identifiers, or raw reader fields.

Metadata columns:

- `Fingerprint_ID`
- `Experiment_ID`
- `Measurement_Unit_ID`
- `Source_File`
- `Strain`
- `Chemical`
- `Concentration`
- `Replicate_ID`
- `Duration`
- `QC_Status`
- `Feature_QC_Flags`

Feature columns, in fixed order:

- `baseline`
- `peak`
- `minimum`
- `endpoint`
- `dynamic_range`
- `time_to_peak`
- `auc`
- `initial_slope`
- `maximum_slope`
- `fold_change`
- `log2_fold_change`

Rows with failed feature QC or non-finite core feature values are excluded from the fingerprint matrix and recorded in the exclusion/QC summary. Their upstream validation errors remain visible in the output report.

## Fingerprint QC

Fingerprint QC reports:

- duplicate fingerprint vectors
- missing metadata cells
- missing feature cells
- unexpected feature names
- unexpected feature order
- duplicated `Measurement_Unit_ID` values
- non-finite fingerprint values
- rows excluded from the fingerprint matrix

The QC layer reports warnings rather than silently changing data. Missing or non-finite source feature values are exclusion reasons, not imputation triggers.

## Normalisation

The builder writes both original-scale and normalised fingerprint datasets. Original values are never overwritten.

Supported normalisation methods:

- `none`: preserve original feature values.
- `zscore`: subtract feature mean and divide by population standard deviation.
- `minmax`: subtract feature minimum and divide by feature range.
- `robust`: subtract feature median and divide by interquartile range.

If a feature has zero scale under the selected method, its normalised values are set to `0` and the feature is listed in the normalisation warning summary. This protects later matrix operations while preserving the original values in `fingerprint_dataset.csv`.

## Distance Metrics

Stage 7A exposes reusable distance functions only. It does not cluster fingerprints or calculate PCA.

Implemented metrics:

- Euclidean distance: square root of summed squared differences.
- Manhattan distance: summed absolute differences.
- Cosine distance: `1 - cosine similarity`.
- Correlation distance: `1 - Pearson correlation` across feature dimensions.

Distance matrices are written from normalised fingerprints. Rows excluded from fingerprint construction are not included in distance matrices because distance calculations require finite numeric values.

Full pairwise distance matrices scale quadratically: `N` fingerprints require `N x N` distances per metric. For thousands of individual fingerprints this can produce gigabyte-scale CSV files. Therefore, the default distance mode is now `consensus`, not individual.

Distance output modes:

- `none`: write no distance matrices.
- `consensus`: default; group fingerprints by `Strain`, `Chemical`, and `Concentration`, then calculate distances among consensus fingerprints.
- `individual`: explicit opt-in; calculate distances among individual fingerprints only when the row count is within the configured safety threshold, or when `--allow-large-distance-matrix` is supplied.

The default individual safety threshold is `2,000` rows.

## Consensus Fingerprints

Consensus fingerprints preserve the biological grouping dimensions:

- `Strain`
- `Chemical`
- `Concentration`

The default policy never averages across strains and never averages across concentrations. Within each group, consensus feature values are medians of individual fingerprint features. A separate long-form consensus summary reports mean, standard deviation, coefficient of variation, finite count, replicate count, and QC status for each feature.

## Output Artifacts

The command:

```bash
python scripts/build_fingerprint_dataset.py "C:\Users\USER\Desktop\biosensor_phase2_source_files"
```

or:

```bash
python scripts/build_fingerprint_dataset.py --feature-file outputs/features/feature_dataset.csv
```

writes:

- `fingerprint_dataset.csv`
- `fingerprint_dataset_normalized.csv`
- `consensus_fingerprint_dataset.csv`
- `consensus_fingerprint_summary.csv`
- `fingerprint_summary.json`
- `fingerprint_qc_report.md`
- `consensus_distance_matrix_euclidean.csv`
- `consensus_distance_matrix_cosine.csv`
- `consensus_distance_matrix_manhattan.csv`
- `consensus_distance_matrix_correlation.csv`

Individual distance matrices keep the historical names `distance_matrix_*.csv`, but they are written only when `--distance-mode individual` is supplied.

Examples:

```bash
python scripts/build_fingerprint_dataset.py "C:\Users\USER\Desktop\biosensor_phase2_source_files" --distance-mode consensus
python scripts/build_fingerprint_dataset.py --feature-file outputs/features/feature_dataset.csv --distance-mode none
python scripts/build_fingerprint_dataset.py --feature-file outputs/features/feature_dataset.csv --distance-mode individual --max-individual-distance-rows 2000
```

Existing non-empty output directories are not overwritten unless `--overwrite` is supplied.

## Future Compatibility

The fingerprint dataset is designed for later PCA, heatmaps, clustering, supervised learning, chemical-similarity analysis, and blind prediction. Future stages should consume the fingerprint table and its QC summary rather than recomputing features or reading raw source files.

Before supervised ML, downstream code must explicitly decide how to handle:

- rows excluded from the fingerprint matrix
- duplicated `Measurement_Unit_ID` values
- duplicate fingerprint vectors
- high feature correlations reported in Stage 6C
- known upstream measurement-identity issues from Stage 5C

Stage 7A intentionally avoids target labels, feature importance, PCA loadings, model fitting, and blind-prediction logic.
