# Stage 8C Advanced Temporal Feature Engineering Benchmark

## Purpose

Stage 8C introduces Feature Engine V2, an isolated advanced feature-engineering layer for whole-cell biosensor luminescence time series. The objective is not to replace the existing Stage 6B core feature engine. The objective is to quantify whether additional temporal, shape, frequency, strain-interaction, response-dynamic, baseline, and normalized feature families improve downstream supervised benchmarks.

The benchmark evaluates improvement for:

- chemical classification;
- concentration regression.

No blind prediction is implemented in Stage 8C.

## Architecture

The new package is `src/feature_engine_v2/`.

- `feature_definitions.py` defines feature-family metadata, mathematical definitions, units, and dependencies.
- `feature_extractor_v2.py` extracts advanced features from canonical time series.
- `feature_dataset_v2.py` stores advanced feature outputs and writes feature dictionaries.
- `feature_ablation.py` merges current Stage 6B core features with V2 features and reruns Stage 8A/8B benchmarks.
- `ablation_dataset.py` writes comparison tables and reports.
- `ablation_plots.py` writes PNG and PDF comparison figures.

The command-line runner is:

```bash
python scripts/run_feature_engineering_v2_benchmark.py "C:\Users\USER\Desktop\biosensor_phase2_source_files"
```

Feature Engine V2 consumes canonical rows. The runner can build canonical rows from source files for real-data verification, but the V2 extractor itself does not read raw CSV or Excel files.

## QC and Input Rules

Feature Engine V2 requires the canonical schema. It validates the canonical schema and carries schema errors forward as QC context rather than silently ignoring them. Rows with `Record_Valid = False` or canonical `QC_Status = fail` are excluded from V2 feature extraction.

The ablation benchmark uses the existing Stage 6B feature engine to build the current core feature baseline. Failed Stage 6B feature rows are excluded from benchmark feature matrices. This preserves the validated-feature contract and prevents Stage 8C from bypassing upstream QC.

## Feature Families

### Temporal Kinetics

Features:

- `temporal_time_to_peak`
- `temporal_time_to_half_peak`
- `temporal_rise_time`
- `temporal_decay_time`
- `temporal_recovery_time`
- `temporal_peak_width`
- `temporal_peak_prominence`
- `temporal_maximum_derivative`
- `temporal_minimum_derivative`
- `temporal_derivative_variance`
- `temporal_derivative_entropy`

These features describe response timing, peak geometry, and first-derivative behavior.

### Window Features

Windows:

- 0-2 h
- 2-6 h
- 6-12 h
- 12-24 h

Per-window statistics:

- mean
- median
- maximum
- minimum
- variance
- slope
- AUC
- standard deviation

These features quantify early, middle, and late response regimes without averaging across measurement units.

### Shape Descriptors

Features:

- `shape_skewness`
- `shape_kurtosis`
- `shape_entropy`
- `shape_signal_energy`
- `shape_roughness`
- `shape_symmetry`
- `shape_peak_count`
- `shape_zero_crossings`
- `shape_coefficient_of_variation`

These features describe distributional shape, waveform complexity, and baseline-centered crossing behavior.

### Frequency Features

Features:

- `frequency_dominant_frequency`
- `frequency_spectral_entropy`
- `frequency_spectral_energy`
- `frequency_fft_coefficient_1` through `frequency_fft_coefficient_5`

FFT features are computed on baseline-centered series. Wavelets remain optional and are not implemented in this stage.

### Strain-Interaction Features

Features:

- `strain_interaction_difference`
- `strain_interaction_ratio`
- `strain_interaction_mean`
- `strain_interaction_variance`

These describe the response magnitude of one strain relative to other strains measured under matched experiment, source file, chemical, concentration, replicate, and duration context.

### Response Dynamics

Features:

- `response_induction_delay`
- `response_inhibition_delay`
- `response_duration`
- `response_recovery_fraction`
- `response_sustained_response_score`

These features estimate response onset, inhibitory deviation, recovery, and sustained late response.

### Baseline Features

Features:

- `baseline_stability`
- `baseline_noise`
- `baseline_drift`

These estimate early-series stability and pre-response noise.

### Normalized Features

Features:

- `normalized_peak_over_baseline`
- `normalized_endpoint_over_baseline`
- `normalized_auc_over_baseline_duration`
- `normalized_dynamic_range_over_baseline`
- `normalized_positive_area_over_total_area`
- `normalized_signal_zscore_auc`

Original values are never overwritten. Normalized features are written as additional columns only.

## Ablation Methodology

The ablation benchmark evaluates:

- current Stage 6B core feature set;
- current features plus each V2 feature family independently;
- current features plus all V2 feature families combined.

For each feature set, Stage 8C reruns:

- the Stage 8A classification benchmark;
- the Stage 8B regression benchmark.

The default Stage 8C runner uses fixed Extra Trees classifier/regressor models to isolate feature-family contribution while keeping runtime manageable. The CLI supports `--classification-models all` and `--regression-models all` for wider benchmark reruns.

The default Stage 8C screening split is 3-fold, 1-repeat cross-validation. This is a contribution-screening protocol. Stage 8A and Stage 8B remain the definitive full model-family benchmarks.

## Output Tables

The Stage 8C output directory contains:

- `advanced_feature_dataset.csv`
- `advanced_feature_dictionary.csv`
- `advanced_feature_summary.json`
- `feature_family_ablation_summary.csv`
- `feature_family_vs_macro_f1.csv`
- `feature_family_vs_r2.csv`
- `feature_family_vs_rmse.csv`
- `feature_family_vs_mae.csv`
- `feature_family_runtime.csv`
- `feature_family_importance.csv`
- `feature_family_redundancy.csv`
- `stage_8c_summary.json`
- `stage_8c_feature_engineering_report.md`

## Figures

PNG and PDF figures are generated for:

- feature-family comparison;
- feature-family ablation;
- classification improvement;
- regression improvement;
- runtime comparison.

## Runtime

Runtime scales with:

- number of measurement units;
- number of feature families included;
- selected classification and regression model lists;
- number of cross-validation splits and repeats;
- permutation-importance repeats.

The default Stage 8C configuration intentionally uses a compact, deterministic ablation benchmark to make real-data iteration practical. Full all-model reruns are supported but can be substantially slower.

## Limitations

Window features for windows with no observations are missing, not imputed. Downstream benchmarks exclude rows with non-finite selected features, and those exclusions affect feature-family comparisons.

Feature Engine V2 does not smooth curves, perform wavelet decomposition, tune models, implement SHAP, or perform blind prediction.

The real dataset includes known upstream issues: duplicated measurement identifiers, duplicate fingerprint vectors, missing identifiers, excluded feature rows, and unverified concentration units. Stage 8C reports and inherits that context but does not correct it.

## Recommendations

Feature families that improve both Macro F1 and R2 with acceptable runtime should be candidates for a future Feature Engine V3 or publication-oriented feature set. Families with high redundancy and no benchmark gain should remain exploratory unless a biological rationale justifies retaining them.
