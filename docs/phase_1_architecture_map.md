# Phase 1 Architecture Map

## 1. Concise Module Map

```text
Raw CSV data
  |
  v
src/data_ingestion/
  loader.py
    - load_csv
    - load_multiple_csv
    - validate_required_columns
  |
  v
src/preprocessing/
  schema_harmonizer.py
    - harmonize_schema
    - validate_harmonized_schema
  cleaner.py
    - standardize_strain_names
    - standardize_chemical_names
    - remove_excluded_chemicals
    - filter_target_chemicals
    - parse_concentration
  |
  v
src/pipeline/
  run_pipeline.py
    - run_analysis_pipeline
    - _prepare_feature_input
  |
  v
src/feature_engineering/
  features.py
    - base kinetic features
  advanced_features.py
    - baseline ratio, fold change, derivatives, segmented AUC
  normalized_features.py
    - experiment and strain-experiment z-score features
  |
  v
src/model_training/
  models.py
    - Random Forest classifier/regressor helpers
  |
  v
src/model_evaluation/
  evaluate.py
  loeo_validation.py
  per_chemical_analysis.py
  panel_optimization.py
  strain_ablation.py
  feature_importance.py
  advanced_loeo_comparison.py
  advanced_per_chemical_analysis.py
  chemical_specific_strains.py
  specialist_ensemble.py
  confidence_intervals.py
  repeated_runs.py
  |
  v
src/visualization/
  plots.py
    - heatmap, PCA, dose response, time course
  |
  v
src/reporting/
  report.py
    - Markdown report helpers

Validation support:
  src/data_validation/validator.py
  src/data_validation/inspect_dataset.py
  src/validation/ml_validation.py
  src/validation/scientific_validation.py
  src/validation/traceability.py
  src/validation/evidence.py
  src/validation/requirements.py
```

## 2. Text-Based End-to-End Data-Flow Diagram

```text
data/raw/*.csv
  |
  | scripts/run_real_analysis.py discovers files with RAW_DATA_DIR.glob("*.csv")
  v
src.data_ingestion.loader.load_multiple_csv
  |
  | pd.read_csv per file, concat, add source_file
  v
src.preprocessing.schema_harmonizer.harmonize_schema
  |
  | rename bacteria_id -> strain, antibiotic -> chemical,
  | Experiment -> experiment, time_min -> time;
  | drop Unnamed columns
  v
src.pipeline.run_pipeline._fill_missing_strain_from_source_file
  |
  | fill missing strain from source file stem
  v
src.data_validation.validator.validate_schema
  |
  | require strain, chemical, concentration, experiment,
  | replicate, time, luminescence
  v
src.preprocessing.cleaner + run_real_analysis aliases
  |
  | strain/chemical standardization, Monensin exclusion,
  | target chemical filter, concentration numeric parse,
  | required-column drop
  v
src.pipeline.run_pipeline._prepare_feature_input
  |
  | mean luminescence per strain/chemical/concentration/
  | experiment/replicate/time; keep groups with >= 2 time points
  v
src.feature_engineering.features.extract_features
  |
  | per replicate kinetic feature vectors
  v
outputs/tables/features.csv
  |
  +--> scripts/run_real_analysis.py writes cleaned_data.csv and processed_data.csv
  |
  +--> src.model_training.models train/predict on full feature table
  |       |
  |       v
  |     outputs/tables/model_metrics.json
  |
  +--> src.visualization.plots
  |       |
  |       v
  |     outputs/figures/heatmap.png, pca.png,
  |     dose_response.png, time_course.png
  |
  +--> src.reporting.report.generate_markdown_report
          |
          v
        outputs/reports/scientific_performance_report.md
        outputs/reports/analysis_report.md
```

## 3. GUI-to-Pipeline Interaction Diagram

No GUI implementation was found.

```text
[GUI layer]
  |
  | Not present in repository:
  | - no GUI framework imports
  | - no main window class
  | - no buttons/commands
  | - no raw-data folder picker
  | - no subprocess or direct GUI pipeline launcher
  v
[script layer]
  |
  | Current user-facing execution is through scripts/*.py
  v
scripts/run_real_analysis.py and post-processing scripts
  |
  v
src pipeline, feature, model, visualization, reporting modules
```

Likely future GUI connection point:

```text
Future GUI folder picker
  |
  v
Future data-import facade
  |
  v
Canonical long-form table
  |
  v
Existing schema/cleaning/feature pipeline
```

## 4. Data-Import Flow Diagram

```text
CSV file paths
  |
  v
load_csv(file_path)
  |
  | pd.read_csv(path)
  | raises FileNotFoundError or empty CSV ValueError
  v
DataFrame per file
  |
  v
load_multiple_csv(file_paths)
  |
  | add source_file = path
  | concat rows
  v
Combined raw DataFrame
  |
  v
harmonize_schema(dataframe)
  |
  | strip column names
  | drop Unnamed columns
  | rename known raw CSV columns
  v
Canonical-ish DataFrame
  |
  v
_fill_missing_strain_from_source_file(dataframe)
  |
  | if strain exists and is missing, use Path(source_file).stem
  v
validate_schema(dataframe, REQUIRED_COLUMNS)
  |
  v
Cleaner and parser functions
  |
  v
Long-form analysis table
```

Current import limitations:

- CSV only.
- Long-form only.
- Schema aliases hard-coded.
- Excel dependency exists but no Excel reader is wired.
- Controls are detected in tests/raw data but not used for normalization.

## 5. Feature-to-Model Flow Diagram

```text
processed_data.csv style table
  columns: strain, chemical, concentration, experiment, replicate, time, luminescence
  |
  v
extract_features
  |
  | group by strain, chemical, concentration, experiment, replicate
  v
features.csv
  columns:
    strain, chemical, concentration, experiment, replicate,
    auc, max_signal, min_signal, time_to_peak, initial_slope, final_signal
  |
  +--> train_classifier / train_regressor
  |       model features:
  |       auc, max_signal, min_signal, time_to_peak, initial_slope, final_signal
  |
  +--> run_loeo_classification / run_loeo_regression
  |       split:
  |       hold out one experiment at a time
  |
  +--> add_experiment_zscore_features
  |       |
  |       v
  |     add_strain_experiment_zscore_features
  |       |
  |       v
  |     normalized LOEO using *_zexp and *_zstrain_exp features
  |
  +--> extract_advanced_features from processed_data.csv
          |
          v
        features_advanced.csv
          columns:
            peak_to_baseline_ratio, fold_change,
            max_derivative, min_derivative, signal_decay_rate,
            auc_early, auc_mid, auc_late
          |
          v
        combined original + advanced feature table
          |
          v
        advanced panel, per-chemical, chemical-specific, specialist workflows
```

## 6. Report-Generation Flow Diagram

```text
run_analysis_pipeline
  |
  | sections built in memory
  v
src.reporting.report.generate_markdown_report
  |
  v
outputs/reports/analysis_report.md

run_real_analysis
  |
  | metrics + figure paths built in memory
  | generate_validation_summary(metrics)
  v
generate_markdown_report
  |
  v
outputs/reports/scientific_performance_report.md

run_confidence_intervals
  |
  | reads outputs/tables/per_chemical_loeo.csv
  | optional reads outputs/tables/model_metrics.json
  v
script-local _build_report
  |
  v
outputs/reports/confidence_interval_report.md

run_repeated_runs
  |
  | builds run metrics and summary
  v
script-local _build_report
  |
  v
outputs/reports/repeated_run_analysis.md
```

## 7. Pipeline Stage Mapping

| Pipeline stage | Entry script | Core module | Input | Output | Tests |
| --- | --- | --- | --- | --- | --- |
| Raw CSV loading | `run_real_analysis.py`; callable pipeline | `src.data_ingestion.loader` | CSV paths | Combined DataFrame with `source_file` | `tests/unit/test_loader.py`, real-data smoke tests |
| Raw dataset inspection | `src/data_validation/inspect_dataset.py` | same file | raw CSV folder | printed summaries/list of dicts | Not directly unit-tested by name |
| Schema harmonization | `run_real_analysis.py`; callable pipeline | `src.preprocessing.schema_harmonizer` | raw DataFrame | renamed/drop-column DataFrame | `tests/unit/test_schema_harmonizer.py` |
| Schema validation | callable pipeline | `src.data_validation.validator` | harmonized DataFrame | validation result | `tests/unit/test_validator.py` |
| Cleaning/standardization | `run_real_analysis.py`; callable pipeline | `src.preprocessing.cleaner` | harmonized DataFrame | cleaned DataFrame | `tests/unit/test_cleaner.py` |
| Feature input aggregation | `run_real_analysis.py`; callable pipeline | `src.pipeline.run_pipeline._prepare_feature_input` | cleaned long-form rows | processed long-form rows | `tests/unit/test_run_pipeline.py`, e2e real pipeline |
| Base feature extraction | `run_real_analysis.py`; callable pipeline | `src.feature_engineering.features` | processed long-form rows | base feature table | `tests/unit/test_features.py` |
| Advanced feature extraction | `run_advanced_feature_generation.py` | `src.feature_engineering.advanced_features` | `processed_data.csv` | `features_advanced.csv` | `tests/unit/test_advanced_features.py` |
| Normalized features | `run_normalized_loeo.py` | `src.feature_engineering.normalized_features` | `features.csv` | `features_normalized.csv` | `tests/unit/test_normalized_features.py` |
| Baseline model train/eval | `run_real_analysis.py` | `src.model_training.models`, `src.model_evaluation.evaluate` | `features.csv` style table | `model_metrics.json` | `tests/unit/test_models.py`, `tests/unit/test_evaluate.py` |
| LOEO validation | `run_loeo_validation.py` | `src.model_evaluation.loeo_validation` | `features.csv` | `loeo_metrics.json` | `tests/unit/test_loeo_validation.py` |
| Panel optimization | `run_panel_optimization.py` | `src.model_evaluation.panel_optimization` | `features.csv` | panel CSV and figures | `tests/unit/test_panel_optimization.py` |
| Per-chemical LOEO | `run_per_chemical_analysis.py` | `src.model_evaluation.per_chemical_analysis` | `features.csv` | per-chemical CSV and confusion matrix | `tests/unit/test_per_chemical_analysis.py` |
| Strain ablation | `run_strain_ablation.py` | `src.model_evaluation.strain_ablation` | `features.csv` | ablation CSVs and figures | `tests/unit/test_strain_ablation.py` |
| Feature importance/PCA | `analyze_feature_importance.py` | `src.model_evaluation.feature_importance` | `features.csv` | importance CSV, PCA figures | `tests/unit/test_feature_importance.py` |
| Advanced LOEO comparison | `run_advanced_loeo_comparison.py` | `src.model_evaluation.advanced_loeo_comparison` | original + advanced feature tables | advanced panel CSV and figure | `tests/unit/test_advanced_loeo_comparison.py` |
| Advanced per-chemical LOEO | `run_advanced_per_chemical_analysis.py` | `src.model_evaluation.advanced_per_chemical_analysis` | original + advanced feature tables | advanced per-chemical CSV and confusion matrix | `tests/unit/test_advanced_per_chemical_analysis.py` |
| Chemical-specific strain ranking | `run_chemical_specific_strains.py` | `src.model_evaluation.chemical_specific_strains` | advanced feature table | ranking CSV and heatmap | `tests/unit/test_chemical_specific_strains.py` |
| Specialist ensemble | `run_specialist_ensemble.py` | `src.model_evaluation.specialist_ensemble` | advanced feature table | metrics JSON and confusion matrix | `tests/unit/test_specialist_ensemble.py` |
| Confidence intervals | `run_confidence_intervals.py` | `src.model_evaluation.confidence_intervals` | `per_chemical_loeo.csv` | CI CSV and report | `tests/unit/test_confidence_intervals.py` |
| Repeated runs | `run_repeated_runs.py` | `src.model_evaluation.repeated_runs` | `features.csv` | metrics/summary CSV, report, boxplot | `tests/unit/test_repeated_runs.py` |
| Visualization | `run_real_analysis.py` and multiple scripts | `src.visualization.plots` plus module-local plotters | feature/raw/metric tables | PNG figures | `tests/unit/test_plots.py` and module-specific tests |
| Reporting | `run_real_analysis.py`, callable pipeline, CI/repeated scripts | `src.reporting.report` and script-local builders | metrics/sections/tables | Markdown reports | `tests/unit/test_report.py` |
| Scientific validation utilities | none wired into main script | `src.validation.scientific_validation` | supplied arrays/matrices | result objects | `tests/unit/test_scientific_validation.py` |
| ML validation utilities | none wired into main script | `src.validation.ml_validation` | predictions or experiment IDs | result objects | `tests/unit/test_ml_validation.py` |
| Traceability/evidence | direct utility calls | `src.validation.traceability`, `src.validation.evidence` | requirement metadata/evidence args | Markdown/evidence log | `tests/unit/test_traceability.py`, `tests/unit/test_evidence.py` |

## 8. Recommended Future Integration Points

| Future integration | Recommended location | Current dependency to protect | Notes |
| --- | --- | --- | --- |
| 12-hour Excel import | New module under `src/data_ingestion/` | Existing `load_csv` behavior | Add adapter returning canonical long-form rows |
| Mixed CSV/Excel input discovery | New orchestration layer called by `run_real_analysis.py` or a future script | `data/raw/*.csv` current behavior | Avoid changing ML modules for file discovery |
| Canonical long schema | `src/preprocessing/schema_harmonizer.py` or new schema module | Required columns expected downstream | Keep `strain`, `chemical`, `concentration`, `experiment`, `replicate`, `time`, `luminescence` stable |
| Configurable chemical/strain lists | New config layer consumed by cleaner/validator/model-evaluation | Current six-target behavior | Needed for ten-chemical analysis |
| 0-12h common window | Pre-feature transformation near `_prepare_feature_input` | Feature extraction input contract | Ensures base features use same window |
| 12-24h late window | `src.feature_engineering.advanced_features` plus explicit window metadata | Existing `auc_late` behavior | Avoid silent zeros for 12h-only files |
| 0-24h full-duration workflow | Separate script or parameterized workflow layer | Existing `run_real_analysis.py` fixed outputs | Preserve old outputs while adding duration-specific outputs |
| Duration-normalized features | New feature functions in `src.feature_engineering` | `NUMERIC_FEATURE_COLUMNS` consumers | Add new columns without replacing old columns immediately |
| Output versioning | Script orchestration layer | Existing output file paths | Prevent overwriting old evidence |
| Combined reporting | Extend `src.reporting.report` | Existing report title/sections | Reduce ad hoc report duplication |
| GUI folder selection | Future GUI layer calling data-import facade | No GUI currently | UI should select folders, not parse Excel internally |
| Test coverage for dual-source data | New fixtures under `tests/fixtures/` | Current CSV fixture tests | Include 12h Excel fixture or generated fixture once allowed |
