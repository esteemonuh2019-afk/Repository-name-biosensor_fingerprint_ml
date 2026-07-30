# Stage 8A Chemical Classification Benchmark Framework

## Purpose

Stage 8A implements a supervised benchmark for identifying chemical identity from validated whole-cell biosensor fingerprints. The goal is comparative model assessment, not optimisation of one classifier.

The benchmark consumes Stage 7A fingerprint datasets only. It does not read raw luminescence time series, does not recompute feature values, does not bypass feature validation, and does not implement blind prediction or concentration regression.

## Architecture

The benchmark package lives in `src/classification_benchmark/`.

- `models.py` defines the supported classifier registry and optional-library handling.
- `benchmark_runner.py` validates fingerprint input, creates validation splits, builds sklearn pipelines, evaluates models, ranks models, and calculates explainability tables.
- `classification_dataset.py` stores benchmark outputs and writes the Stage 8A artifacts.

The command-line entry point is:

```bash
python scripts/run_classification_benchmark.py "C:\Users\USER\Desktop\biosensor_phase2_source_files"
```

or, when a Stage 7A fingerprint CSV already exists:

```bash
python scripts/run_classification_benchmark.py --fingerprint-file outputs/fingerprints/fingerprint_dataset.csv
```

The CLI rejects `fingerprint_dataset_normalized.csv` because global pre-normalisation would scale before validation splitting and could leak information across folds.

## Input Definition

Each benchmark sample is one validated fingerprint row. The default target label is `Chemical`.

The feature matrix contains the Stage 6B core fingerprint features in fixed order:

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

Metadata columns are preserved for traceability but excluded from the feature matrix. Rows with missing target labels or non-finite feature values are excluded from benchmark fitting and counted in the output metadata. Duplicate fingerprints and duplicated `Measurement_Unit_ID` rows are retained and reported as warnings because they are upstream QC context, not Stage 8A correction targets.

## Supported Models

Required classifiers:

- Random Forest
- Extra Trees
- Gradient Boosting
- Logistic Regression
- Support Vector Machine
- k-Nearest Neighbours

Optional classifiers:

- XGBoost, when `xgboost` imports successfully.
- LightGBM, when `lightgbm` imports successfully.

Unavailable optional libraries are skipped automatically and reported in the benchmark warnings. Required classifiers must be available.

## Validation Strategy

Supported validation strategies:

- Stratified train/test split.
- Stratified K-fold cross-validation.
- Repeated stratified K-fold cross-validation.
- Leave-one-strain-out validation.
- Leave-one-chemical-out validation in research mode.

The default strategy is repeated stratified 5-fold cross-validation with two repeats. If the smallest class cannot support the requested fold count, the fold count is reduced deterministically and reported as a warning.

Leave-one-chemical-out is research mode because the held-out chemical label is absent from the corresponding training fold. Its metrics estimate out-of-label generalisation pressure, not ordinary closed-set chemical identification.

## Leakage Prevention

All preprocessing is implemented through sklearn `Pipeline` objects:

```text
preprocess -> classifier
```

Supported preprocessing modes:

- `none`
- `zscore`
- `robust`
- `minmax`

Scaling is fitted only on each training fold and then applied to the matching held-out fold. The full dataset is never scaled before splitting. This rule also applies to the command-line runner, which refuses pre-normalised fingerprint CSV input.

## Metrics

For every model and validation fold, the benchmark calculates:

- Accuracy
- Balanced accuracy
- Macro precision
- Macro recall
- Macro F1
- Weighted F1
- ROC-AUC, when aligned class probabilities are available
- Log loss, when aligned class probabilities are available
- Training time
- Prediction time

The summary table reports cross-validation mean, population standard deviation, and approximate 95% confidence interval bounds for each fold-level metric.

The selected best model is the highest mean Macro F1 model. Ties are broken by mean balanced accuracy and then mean accuracy. The ranking is deterministic.

## Explainability

For tree models, the benchmark writes model-native feature importance values where the estimator exposes `feature_importances_`.

For the selected best model, the benchmark calculates permutation importance with Macro F1 scoring. This is an explanatory analysis of the fitted benchmark model and is not treated as an independent validation metric.

For tree models, the benchmark can also train leave-one-strain-out importance models. These tables help identify whether feature importance is stable when one strain is withheld from training.

SHAP is intentionally not implemented in Stage 8A.

## Output Artifacts

The benchmark writes:

- `classification_summary.csv`
- `best_model_metrics.json`
- `confusion_matrix.csv`
- `per_class_metrics.csv`
- `feature_importance.csv`
- `permutation_importance.csv`
- `model_rankings.csv`
- `classification_report.md`
- `leave_one_strain_importance.csv`
- `fold_metrics.csv`

The required Stage 8A outputs are the first eight files. The last two files preserve useful audit detail for reproducibility and strain-specific interpretation.

## Limitations

Stage 8A inherits upstream QC limitations from canonical ingestion, feature extraction, feature validation, and fingerprint construction. It does not correct duplicate measurement identities, source-data ambiguity, duplicate fingerprints, or excluded feature rows.

The benchmark compares standard model families under a shared protocol. It does not tune hyperparameters, select a deployment model, build fingerprints for blind unknowns, estimate concentration, perform clustering, generate PCA, or produce plots.

## Recommended Next Steps

Stage 8B should use the Stage 8A results to decide whether additional validation designs or hyperparameter tuning are scientifically justified. Blind prediction should remain separate and should consume only a model-selection protocol that has already been frozen.
