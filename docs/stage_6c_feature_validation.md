# Stage 6C Feature Validation and Quality Assessment

## Purpose

Stage 6C adds a validation layer for the Stage 6B core feature dataset. The validator evaluates whether extracted features are numerically valid, scientifically informative, traceable, and suitable for later fingerprint construction and machine-learning design.

This stage is deliberately unsupervised. It does not perform PCA, clustering, classification, regression, plotting, blind prediction, or feature importance analysis. It does not use chemical labels as targets. It reports feature quality; it does not remove, impute, transform, or average values.

## Validation Architecture

The validation layer lives in `src/feature_validation/` and is separated from feature calculation.

- `feature_validator.py` orchestrates validation and returns a structured `FeatureValidationResult`.
- `feature_statistics.py` computes missingness, non-finite values, finite descriptive statistics, variance checks, range checks, and pairwise correlations.
- `replicate_reproducibility.py` computes replicate-consistency summaries without claiming biological reproducibility.
- `feature_selection_report.py` creates deterministic unsupervised retention recommendations and writes report artifacts.

The input contract is a `FeatureDataset` from `src/feature_engine/feature_extractor.py` or an already materialized feature dataframe with the Stage 6B feature columns. Metadata columns are preserved for traceability and grouping but are excluded from numeric feature statistics and correlation calculations.

## Assessed Feature Columns

The validator assesses only the Stage 6B core feature columns:

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

Identifiers and labels such as `Experiment_ID`, `Measurement_Unit_ID`, `Source_File`, `Strain`, `Chemical`, `Concentration`, `Replicate_ID`, and `QC_Status` are not treated as numeric predictors.

## Thresholds

Default thresholds are versioned in code and reported in every validation summary.

| Rule | Default | Interpretation |
|---|---:|---|
| Low variance | variance `<= 1e-12` | Flags non-constant features with effectively no numeric spread. |
| Dominant value proportion | `>= 0.95` | Flags features where at least 95% of finite values are identical. |
| High correlation | absolute Pearson or Spearman correlation `>= 0.95` | Flags redundant feature pairs for later review. |
| Stable replicate consistency | CV `<= 0.10` | Experimental group is stable for that feature. |
| Acceptable replicate consistency | CV `<= 0.25` | Experimental group is usable with caution. |
| Unstable replicate consistency | CV `> 0.25` | Experimental group shows high replicate spread. |

These thresholds are screening rules, not automatic exclusion rules except for constant features and features with no finite numeric values.

## Missingness Rules

Missing values are reported for every assessed feature with:

- missing count
- missing percentage
- affected strains
- affected chemicals
- affected concentrations
- affected source files

Missing values are never imputed in Stage 6C. A feature can remain eligible when missingness is low, but missingness contributes to caution or review recommendations. High missingness, currently at or above 20%, is recommended for review.

## Non-Finite Rules

The validator distinguishes:

- positive infinity
- negative infinity
- `NaN` or missing values
- non-numeric values

Infinite and non-numeric values are validation errors because they must not enter downstream machine-learning matrices silently. Missing values are reported as warnings unless they combine with range or QC errors.

## Variance Rules

Constant features are features with exactly one unique finite value across valid feature rows. They are recommended for exclusion because they cannot discriminate biological responses in downstream fingerprint construction.

Near-constant features are non-constant features with variance at or below the low-variance threshold or dominant-value proportion at or above the dominant-value threshold. These are recommended for review because they may be mathematically valid but scientifically uninformative.

## Range Rules

Range checks use the Stage 6A and 6B definitions and flag values without changing them.

The validator reports:

- negative `time_to_peak`
- `time_to_peak` before `Start_Time`, after `End_Time`, or beyond `Duration`
- negative `dynamic_range`
- negative `auc` when all observed signal values are non-negative
- undefined fold change from zero baseline
- `peak` less than `minimum`
- `dynamic_range` inconsistent with `peak - minimum`
- `endpoint` outside the observed `[minimum, peak]` interval

Some values can be suspicious rather than impossible in every biological context. Stage 6C therefore reports them explicitly rather than deleting feature rows.

## Correlation Rules

Pairwise Pearson and Spearman correlations are calculated among finite numeric core features. Correlations are pairwise complete: rows with missing or non-finite values for either feature in a pair are excluded from that pair only.

Highly correlated pairs are reported at absolute correlation `>= 0.95`. The validator does not automatically remove either feature. Correlated features are retained with caution so later fingerprint construction can choose whether redundancy is useful biological structure or unnecessary dimensionality.

## Replicate-Consistency Rules

Replicate consistency is grouped by the best available metadata:

- `Experiment_ID`
- `Strain`
- `Chemical`
- `Concentration`

For each group and feature, the validator reports:

- mean
- standard deviation using population `ddof=0`
- coefficient of variation, `SD / abs(mean)`
- replicate count
- unique replicate identifier count
- stability flag

The stability flags are:

- Stable: at least two finite values and CV `<= 0.10`
- Acceptable: at least two finite values and CV `> 0.10` and `<= 0.25`
- Unstable: at least two finite values and CV `> 0.25`
- Insufficient Data: fewer than two finite values or undefined CV

The report uses the term replicate consistency. It does not claim biological reproducibility because the Stage 6B feature table does not establish whether replicate identifiers are biological, technical, temporal, or importer-derived.

## Recommendation Logic

Each core feature receives one deterministic recommendation:

- Retain
- Retain with caution
- Review
- Exclude

Recommendations are based only on missingness, non-finite values, variance, range validity, correlation, and replicate consistency. They do not use target labels, chemical class labels, model performance, or feature importance.

The current priority order is:

1. Exclude features with no finite numeric values.
2. Exclude constant features.
3. Review low-variance or near-constant features.
4. Review features with infinite or non-numeric values.
5. Review features with range-validation violations.
6. Review features with high missingness.
7. Retain with caution features with low missingness, high correlation, unstable replicate consistency, or insufficient replicate consistency.
8. Retain features without unsupervised validation concerns.

## Output Artifacts

The CLI script `scripts/validate_feature_dataset.py` can validate a freshly generated feature dataset from a source folder or an existing `feature_dataset.csv`.

It writes:

- `feature_validation_summary.json`
- `feature_statistics.csv`
- `feature_missingness.csv`
- `feature_nonfinite_values.csv`
- `constant_features.csv`
- `low_variance_features.csv`
- `pearson_correlations.csv`
- `spearman_correlations.csv`
- `highly_correlated_pairs.csv`
- `replicate_consistency.csv`
- `feature_recommendations.csv`
- `feature_validation_report.md`

Existing output directories are not overwritten unless `--overwrite` is supplied.

## Limitations

Stage 6C does not repair known upstream Stage 5C measurement-identity issues. It preserves failed and warning feature rows in the validation report.

Stage 6C does not impute missing values, normalize features, smooth curves, resolve duplicate timestamps, average replicates, or remove correlated features. It is a quality assessment layer, not a feature-engineering or model-selection layer.

Replicate consistency is descriptive. It is not an intraclass-correlation analysis and it is not a claim of biological reproducibility. ICC should only be added if future metadata explicitly identifies replicate design and if group sizes are statistically adequate.

## Implications for Downstream Fingerprint Construction and ML

Downstream fingerprint construction should start from retained features and explicitly decide how to handle features marked as caution, review, or exclusion. Constant and non-finite features should not enter numerical machine-learning matrices. Highly correlated features should be evaluated in the context of model class and biological interpretation.

Rows carrying failed feature QC remain scientifically important because they identify source-data ambiguity, duplicate timestamps, zero-baseline limitations, or other upstream constraints. Later ML stages must either exclude those rows under a documented policy or use models and preprocessing steps that can represent missingness and QC flags without leakage.

Stage 6C establishes the audit trail required before feature normalization, fingerprint construction, dimensionality reduction, clustering, or supervised learning begins.
