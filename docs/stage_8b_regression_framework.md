# Stage 8B Concentration Regression Benchmark Framework

## Purpose

Stage 8B implements a supervised benchmark for predicting numeric contaminant concentration from validated whole-cell biosensor fingerprints. The objective is model comparison under identical validation conditions, not optimisation of a single regressor.

The benchmark consumes Stage 7A fingerprint datasets only. It does not read raw luminescence time series, does not recompute feature values, does not bypass feature validation, does not alter the Stage 8A classification framework, and does not implement blind prediction.

## Architecture

The regression package lives in `src/regression_benchmark/`.

- `models.py` defines the supported regressor registry and optional-library handling.
- `benchmark_runner.py` validates fingerprint input, resolves numeric concentration targets, creates validation splits, builds sklearn pipelines, evaluates models, ranks models, and calculates feature-analysis tables.
- `regression_dataset.py` stores benchmark outputs and writes tables, JSON, reports, and figures.
- `regression_plots.py` produces publication-oriented prediction, residual, feature-importance, and fold-performance figures.

The command-line entry point is:

```bash
python scripts/run_regression_benchmark.py "C:\Users\USER\Desktop\biosensor_phase2_source_files"
```

or, for an existing Stage 7A fingerprint file:

```bash
python scripts/run_regression_benchmark.py --fingerprint-file outputs/fingerprints/fingerprint_dataset.csv
```

The CLI refuses `fingerprint_dataset_normalized.csv` because global pre-normalisation would scale before validation splitting and could leak information across folds.

## Input and Target Definition

Each benchmark sample is one validated fingerprint row. The feature matrix contains the Stage 6B core fingerprint features in fixed order:

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

Metadata columns are preserved for traceability but excluded from the feature matrix.

The default regression target is the fingerprint metadata column `Concentration`. Numeric values are parsed to `Concentration_Target_ug_mL`. If a future fingerprint dataset contains `Concentration_ug_mL`, the benchmark can use it directly by setting `--target-column Concentration_ug_mL`.

Supported target parsing follows the canonical concentration convention:

- unitless numeric labels are interpreted using the canonical ug/mL convention inherited from the canonical builder;
- `ug/mL` and equivalent microgram-per-milliliter labels are retained as-is;
- `mg/L` is numerically equivalent to `ug/mL`;
- `ng/mL`, `mg/mL`, and `g/L` are converted to ug/mL;
- controls, missing labels, and unsupported non-mass units are excluded and counted.

No concentration values are imputed. Negative concentration targets are excluded and reported.

## Supported Models

Required regressors:

- Random Forest Regressor
- Extra Trees Regressor
- Gradient Boosting Regressor
- Elastic Net
- Ridge Regression
- Lasso Regression
- Support Vector Regression
- kNN Regressor

Optional regressors:

- XGBoost Regressor, when `xgboost` imports successfully.
- LightGBM Regressor, when `lightgbm` imports successfully.

Unavailable optional libraries are skipped automatically and reported in benchmark warnings. Required regressors must run independently.

## Validation Strategy

Supported validation strategies:

- Repeated K-fold cross-validation.
- Leave-one-strain-out regression.
- Leave-one-chemical-out regression in research mode.

The default strategy is repeated 5-fold cross-validation with two repeats. If the usable sample count cannot support the requested fold count, the effective fold count is reduced deterministically and reported.

Leave-one-chemical-out regression is research mode because all rows for one chemical are withheld together. It is useful for stress-testing chemical generalisation, but it is not the same task as closed-design concentration interpolation.

## Leakage Prevention

All preprocessing is implemented through sklearn `Pipeline` objects:

```text
preprocess -> regressor
```

Supported preprocessing modes:

- `none`
- `zscore`
- `robust`
- `minmax`

Scaling is fitted only on each training fold and then applied to the matching held-out fold. The full dataset is never scaled before splitting.

## Metrics

For every model and validation fold, the benchmark calculates:

- R²
- RMSE
- MAE
- Median absolute error
- Explained variance
- MAPE where actual concentration is non-zero
- Residual mean and standard deviation
- Maximum absolute error
- Training time
- Prediction time

The summary table reports cross-validation mean, population standard deviation, and approximate 95% confidence interval bounds for each fold-level metric.

## Model Ranking

Models are ranked by:

1. Highest mean cross-validated R².
2. Lowest mean RMSE.
3. Lowest mean MAE.
4. Model name for deterministic ordering.

The selected best model is written to `best_regression_model.json`.

## Feature Analysis

For tree models, the benchmark writes model-native feature importance values where the estimator exposes `feature_importances_`.

For the selected best model, the benchmark calculates permutation importance with R² scoring. This is an explanatory analysis of the fitted benchmark model and is not treated as an independent validation metric.

For tree models, the benchmark can also train leave-one-strain-out importance models to evaluate feature-importance stability when a strain is withheld from training.

SHAP is intentionally not implemented in Stage 8B.

## Figures

The benchmark writes PNG and PDF versions of:

- Prediction vs actual concentration.
- Residuals by predicted concentration.
- Residual histogram.
- Feature importance.
- Fold-performance comparison.

The figures are report artifacts, not additional modelling steps.

## Output Artifacts

The benchmark writes:

- `regression_summary.csv`
- `best_regression_model.json`
- `per_model_metrics.csv`
- `fold_metrics.csv`
- `prediction_vs_actual.csv`
- `residuals.csv`
- `model_rankings.csv`
- `regression_report.md`
- `feature_importance.csv`
- `permutation_importance.csv`
- `leave_one_strain_importance.csv`
- `prediction_vs_actual.png` and `.pdf`
- `residual_plot.png` and `.pdf`
- `residual_histogram.png` and `.pdf`
- `feature_importance.png` and `.pdf`
- `fold_performance.png` and `.pdf`

## Limitations

Stage 8B inherits upstream QC limitations from canonical ingestion, feature extraction, feature validation, and fingerprint construction. It does not correct duplicate measurement identities, source-data ambiguity, duplicate fingerprints, excluded feature rows, or unverified source concentration units.

The benchmark uses raw numeric concentration values rather than log-transformed concentration. Because the real dataset spans several orders of magnitude, later stages may need a separately specified log-concentration benchmark, but that is not implemented here.

## Transition to Blind Prediction

Blind prediction should begin only after the benchmark protocol, target transformation policy, selected model family, preprocessing strategy, and acceptable validation performance have been frozen. Stage 8B produces the evidence required for that decision, but it does not train or package a blind-prediction model.
