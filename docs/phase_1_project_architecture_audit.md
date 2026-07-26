# Phase 1 Project Architecture Audit

## 1. Audit Scope

- Audit timestamp: 2026-07-25 20:50:57 +03:00.
- Project root: `C:\Users\USER\Desktop\biosensor_fingerprint_ml`.
- Phase: Phase 1 only.
- Goal: inspect the existing Python project and map current architecture.
- Explicitly not performed: no 12-hour Excel support, no pipeline changes, no refactoring, no model training, no output regeneration, no package changes, no Git operations that change state.
- Initial Phase 1 `git status --short`:

```text
 M .gitignore
?? docs/phase_0_safety_audit.md
?? scripts/run_confidence_intervals.py
?? scripts/run_repeated_runs.py
?? src/model_evaluation/confidence_intervals.py
?? src/model_evaluation/repeated_runs.py
?? tests/unit/test_confidence_intervals.py
?? tests/unit/test_repeated_runs.py
```

Those files are treated as pre-existing work for this audit.

## 2. Project Structure

```text
biosensor_fingerprint_ml/
|-- .agents/
|-- .github/
|-- .pytest_cache/
|-- .venv/
|-- data/
|   `-- raw/
|       |-- BL011.csv
|       |-- BL027ab.csv
|       |-- BL029.csv
|       |-- BL030.csv
|       |-- BL031.csv
|       `-- BL032.csv
|-- docs/
|   |-- DATASET_CARD.md
|   |-- LIMITATIONS_AND_RISKS.md
|   |-- ML_VALIDITY_AUDIT.md
|   |-- MODEL_CARD.md
|   |-- REPRODUCIBILITY_GUIDE.md
|   |-- RESEARCH_QUESTION_AND_GAP.md
|   |-- REVIEW_SUMMARY.md
|   |-- SAMPLE_SIZE_AND_REPLICATION_JUSTIFICATION.md
|   |-- STATISTICAL_ANALYSIS_PLAN.md
|   |-- TRACEABILITY_MATRIX.md
|   |-- UNCERTAINTY_ANALYSIS.md
|   |-- phase_0_safety_audit.md
|   |-- SSDD.docx
|   `-- VV_Plan_v1.0.docx
|-- outputs/
|   |-- figures/
|   |-- reports/
|   `-- tables/
|-- scripts/
|   |-- analyze_feature_importance.py
|   |-- run_advanced_feature_generation.py
|   |-- run_advanced_loeo_comparison.py
|   |-- run_advanced_per_chemical_analysis.py
|   |-- run_chemical_specific_strains.py
|   |-- run_confidence_intervals.py
|   |-- run_loeo_validation.py
|   |-- run_normalized_loeo.py
|   |-- run_panel_optimization.py
|   |-- run_per_chemical_analysis.py
|   |-- run_real_analysis.py
|   |-- run_repeated_runs.py
|   |-- run_specialist_ensemble.py
|   `-- run_strain_ablation.py
|-- src/
|   |-- data_ingestion/
|   |-- data_validation/
|   |-- feature_engineering/
|   |-- model_evaluation/
|   |-- model_training/
|   |-- pipeline/
|   |-- preprocessing/
|   |-- reporting/
|   |-- validation/
|   `-- visualization/
|-- tests/
|   |-- blackbox/
|   |-- contract/
|   |-- e2e/
|   |-- fixtures/
|   |-- integration/
|   |-- regression/
|   `-- unit/
|-- .gitignore
|-- README.md
`-- requirements.txt
```

Important observations:

- Python implementation is under `src/`, divided by workflow layer.
- Workflow execution is script-driven under `scripts/`.
- Tests are present across unit, integration, regression, black-box, contract, and e2e categories.
- Raw data is under `data/raw/`, ignored by Git and untracked.
- Generated outputs are under `outputs/figures`, `outputs/reports`, and `outputs/tables`; 40 output files are tracked by Git.
- No GUI source file, notebook, shell launcher, batch launcher, desktop launcher, `main.py`, `app.py`, `biosensor_gui.py`, or `__main__.py` was found.
- No `outputs/models/` directory or model artifact file was found, although `.gitignore` ignores `outputs/models/`, `*.pkl`, and `*.joblib`.
- Generated `__pycache__` folders exist under `src/` and `tests/`.

## 3. Entry Points

| Path | Purpose | Expected command | Downstream modules | Outputs created | Status |
| --- | --- | --- | --- | --- | --- |
| `scripts/run_real_analysis.py` | Real raw-data workflow from `data/raw/*.csv` to tables, figures, metrics, and report | `python scripts/run_real_analysis.py` | `src.pipeline.run_pipeline`, `src.preprocessing`, `src.feature_engineering.features`, `src.model_training.models`, `src.model_evaluation.evaluate`, `src.visualization.plots`, `src.reporting.report` | `outputs/tables/cleaned_data.csv`, `processed_data.csv`, `features.csv`, `model_metrics.json`; figures; `outputs/reports/scientific_performance_report.md`; also pipeline report | Current and most complete workflow, but README does not explicitly name it primary |
| `src/pipeline/run_pipeline.py` | Callable raw-data-to-basic-report pipeline | Imported/called, not direct CLI | loader, schema harmonizer, validator, cleaner, feature extraction, Markdown report | `analysis_report.md` in supplied output dir | Current callable pipeline |
| `src/data_validation/inspect_dataset.py` | Inspect raw CSV structure | `python -m src.data_validation.inspect_dataset [folder]` or `python src/data_validation/inspect_dataset.py [folder]` | pandas CSV reader | Printed summaries only | Current inspection utility |
| `scripts/run_advanced_feature_generation.py` | Derive advanced features from `processed_data.csv` | `python scripts/run_advanced_feature_generation.py` | `src.feature_engineering.advanced_features` | `outputs/tables/features_advanced.csv` | Current post-processing script |
| `scripts/run_loeo_validation.py` | LOEO classification and regression on `features.csv` | `python scripts/run_loeo_validation.py` | `src.model_evaluation.loeo_validation` | `outputs/tables/loeo_metrics.json` | Current post-processing script |
| `scripts/run_normalized_loeo.py` | Z-score normalized LOEO | `python scripts/run_normalized_loeo.py` | `src.feature_engineering.normalized_features`, `loeo_validation` | `features_normalized.csv`, `loeo_metrics_normalized.json` | Current post-processing script |
| `scripts/analyze_feature_importance.py` | Random Forest feature importance and PCA by chemical/experiment | `python scripts/analyze_feature_importance.py` | `src.model_evaluation.feature_importance` | `feature_importance.csv`, `feature_importance.png`, `pca_by_chemical.png`, `pca_by_experiment.png` | Current post-processing script |
| `scripts/run_panel_optimization.py` | Candidate strain-panel LOEO comparison | `python scripts/run_panel_optimization.py` | `src.model_evaluation.panel_optimization` | `panel_optimization.csv`, panel figures | Current post-processing script |
| `scripts/run_per_chemical_analysis.py` | Per-chemical LOEO on selected panel | `python scripts/run_per_chemical_analysis.py` | `src.model_evaluation.per_chemical_analysis` | `per_chemical_loeo.csv`, `normalized_confusion_matrix.png` | Current post-processing script |
| `scripts/run_strain_ablation.py` | Single-strain and leave-one-strain-out LOEO | `python scripts/run_strain_ablation.py` | `src.model_evaluation.strain_ablation` | `single_strain_loeo.csv`, `leave_one_strain_out_loeo.csv`, figures | Current post-processing script |
| `scripts/run_advanced_loeo_comparison.py` | Advanced-feature panel comparison | `python scripts/run_advanced_loeo_comparison.py` | `src.model_evaluation.advanced_loeo_comparison` | `advanced_panel_optimization.csv`, `advanced_panel_macro_f1.png` | Current post-processing script |
| `scripts/run_advanced_per_chemical_analysis.py` | Advanced-feature per-chemical LOEO for BL027 | `python scripts/run_advanced_per_chemical_analysis.py` | `advanced_loeo_comparison`, `advanced_per_chemical_analysis` | `advanced_per_chemical_loeo_BL027.csv`, advanced confusion matrix | Current post-processing script |
| `scripts/run_chemical_specific_strains.py` | Chemical-specific strain ranking | `python scripts/run_chemical_specific_strains.py` | `src.model_evaluation.chemical_specific_strains` | `chemical_specific_strain_rankings.csv`, chemical-specific heatmap | Current post-processing script |
| `scripts/run_specialist_ensemble.py` | Specialist-strain ensemble LOEO | `python scripts/run_specialist_ensemble.py` | `src.model_evaluation.specialist_ensemble` | `specialist_ensemble_metrics.json`, confusion matrix | Current post-processing script |
| `scripts/run_confidence_intervals.py` | Bootstrap confidence intervals from per-chemical LOEO outputs | `python scripts/run_confidence_intervals.py` | `src.model_evaluation.confidence_intervals` | `confidence_intervals.csv`, `confidence_interval_report.md` | Pre-existing untracked current/experimental script |
| `scripts/run_repeated_runs.py` | Random-seed robustness analysis | `python scripts/run_repeated_runs.py` | `src.model_evaluation.repeated_runs` | `repeated_run_metrics.csv`, `repeated_run_summary.csv`, report, boxplot | Pre-existing untracked current/experimental script |
| Test suite | Validation and regression checks | `python -m pytest`; collect-only not run in Phase 1 | `tests/` imports many `src/` modules | Test temp workspaces and figures when run | Existing test entry point |

## 4. Current Execution Flow

Evidence-supported main workflow, centered on `scripts/run_real_analysis.py`:

1. Input discovery
   - File/module: `scripts/run_real_analysis.py`
   - Function: `run_real_analysis`
   - Input: `data/raw/*.csv`
   - Output: sorted list of raw file paths
   - Configuration: hard-coded `RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"`

2. File loading
   - File/module: `src.data_ingestion.loader`
   - Functions: `load_csv`, `load_multiple_csv`
   - Input: CSV paths
   - Output: concatenated DataFrame with `source_file`
   - Notes: `pd.read_csv`; fallback to latin1 exists in `src.pipeline.run_pipeline._load_multiple_csv_with_encoding_fallback`

3. Validation
   - File/module: `src.data_validation.validator`
   - Function: `validate_schema`
   - Input: harmonized DataFrame
   - Output: `SchemaValidationResult`
   - Configuration: required columns in `src.pipeline.run_pipeline.REQUIRED_COLUMNS`

4. Schema harmonization
   - File/module: `src.preprocessing.schema_harmonizer`
   - Function: `harmonize_schema`
   - Input: loaded DataFrame
   - Output: copied DataFrame with renamed columns and `Unnamed*` columns dropped
   - Configuration: `COLUMN_RENAMES = {"bacteria_id": "strain", "antibiotic": "chemical", "Experiment": "experiment", "time_min": "time"}`

5. Strain identity fill
   - File/module: `src.pipeline.run_pipeline`
   - Function: `_fill_missing_strain_from_source_file`
   - Input: harmonized DataFrame with possible missing `strain`
   - Output: copied DataFrame with missing strain values filled from `Path(source_file).stem`
   - Purpose: needed for files whose first header is `BL030` or `BL031` rather than `bacteria_id`

6. Cleaning and standardisation
   - File/module: `src.preprocessing.cleaner` and `scripts.run_real_analysis`
   - Functions: `standardize_strain_names`, `_normalize_real_chemical_aliases`, `standardize_chemical_names`, `remove_excluded_chemicals`, `filter_target_chemicals`, `parse_concentration`
   - Input: harmonized raw DataFrame
   - Output: cleaned copied DataFrame
   - Configuration: hard-coded target chemicals, Monensin exclusion, BL027ab alias, real chemical aliases in `run_real_analysis`

7. Missing-value handling
   - File/module: `src.pipeline.run_pipeline` and `scripts.run_real_analysis`
   - Operation: `dropna(subset=list(REQUIRED_COLUMNS)).reset_index(drop=True)`
   - Input: cleaned DataFrame
   - Output: rows with required nulls removed

8. Replicate/time aggregation and minimum curve filtering
   - File/module: `src.pipeline.run_pipeline`
   - Function: `_prepare_feature_input`
   - Input: cleaned long-form table
   - Output: processed table with mean luminescence grouped by `strain`, `chemical`, `concentration`, `experiment`, `replicate`, `time`; groups with fewer than two unique time points removed

9. Feature extraction
   - File/module: `src.feature_engineering.features`
   - Function: `extract_features`
   - Input: processed long-form table
   - Output: `features.csv` style feature table, one row per `strain`, `chemical`, `concentration`, `experiment`, `replicate`

10. Table generation
   - File/module: `scripts.run_real_analysis`
   - Outputs: `outputs/tables/cleaned_data.csv`, `processed_data.csv`, `features.csv`, optional `model_metrics.json`

11. Machine-learning analysis
   - File/module: `scripts.run_real_analysis`, `src.model_training.models`, `src.model_evaluation.evaluate`
   - Functions: `_train_and_evaluate_if_possible`, `train_classifier`, `predict_classifier`, `train_regressor`, `predict_regressor`, `evaluate_classification`, `evaluate_regression`
   - Input: `feature_data`
   - Output: in-memory metrics saved to `model_metrics.json`
   - Important risk: the baseline `run_real_analysis` metrics train and predict on the same feature table; LOEO scripts are separate.

12. Figure generation
   - File/module: `scripts.run_real_analysis`, `src.visualization.plots`
   - Functions: `plot_heatmap`, `plot_pca`, `plot_dose_response`, `plot_time_course`
   - Outputs: `heatmap.png`, `pca.png`, `dose_response.png`, `time_course.png`

13. Report generation
   - File/module: `src.reporting.report`
   - Functions: `generate_validation_summary`, `generate_markdown_report`
   - Output: `outputs/reports/scientific_performance_report.md`

14. Model saving
   - No model persistence was found in current code. No `joblib.dump`, `pickle`, or saved model path was found.

15. GUI status reporting
   - No GUI implementation was found, so no GUI status reporting flow can be proven.

## 5. Data Import Architecture

- Supported file types in code: CSV only.
- Excel support in code: none found. `requirements.txt` includes `openpyxl`, but no `read_excel`, `to_excel`, `.xlsx`, or `.xls` handling was found in source/scripts/tests.
- Expected real-data directory: `data/raw/`.
- Expected raw discovery pattern: `RAW_DATA_DIR.glob("*.csv")`.
- Expected raw headers observed:
  - `BL011.csv`: `bacteria_id,antibiotic,concentration,Experiment,replicate,time_min,luminescence,...`
  - `BL027ab.csv`: same base columns plus trailing blank columns
  - `BL029.csv`: same base columns
  - `BL030.csv`: `BL030,antibiotic,concentration,Experiment,replicate,time_min,luminescence`
  - `BL031.csv`: `BL031,antibiotic,concentration,Experiment,replicate,time_min,luminescence`
  - `BL032.csv`: same base columns
- Canonical columns after harmonization:
  - `strain`
  - `chemical`
  - `concentration`
  - `experiment`
  - `replicate`
  - `time`
  - `luminescence`
- Strain identity:
  - Usually from `bacteria_id`, renamed to `strain`.
  - For missing strain values, filled from `source_file` stem.
  - `BL027ab` standardized to `BL027`.
- Chemical identity:
  - `antibiotic` renamed to `chemical`.
  - Real aliases normalized in `scripts/run_real_analysis.py`.
  - Names then canonicalized in `src.preprocessing.cleaner`.
- Concentration parsing:
  - `parse_concentration` extracts the first numeric value from strings via regex and converts to `float`.
  - Invalid values become `NaN`.
  - Controls with `concentration = Control` therefore parse to `NaN`, but controls are already removed by target-chemical filtering in the main flow.
- Replicate parsing:
  - Existing code expects a `replicate` column; no complex replicate parsing logic was found.
- Time detection:
  - Existing code expects `time_min` in raw CSV and renames it to `time`.
  - Advanced feature windows infer hours vs minutes by checking `max(time_values) <= 24`.
- Luminescence detection:
  - Existing code expects a `luminescence` column.
- Controls:
  - Raw files contain `Control` rows.
  - Tests define `CONTROL_LABELS = {"Control"}`.
  - Main processing does not normalize to control rows; it filters to target chemicals, so controls are removed.
- Format:
  - Existing pipeline expects long-form rows, not wide plate layouts.
- Configurability:
  - Import mapping is hard-coded in Python constants.
  - No YAML/JSON/TOML config for schema mappings or chemical/strain lists was found.
- Canonical/intermediate tables:
  - `outputs/tables/cleaned_data.csv`
  - `outputs/tables/processed_data.csv`
  - `outputs/tables/features.csv`
  - `outputs/tables/features_advanced.csv`
  - `outputs/tables/features_normalized.csv`

## 6. Transformation and Normalisation

| Method | File/function | Purpose | When applied | Raw or copied data | Configurable | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Drop blank `Unnamed*` columns | `schema_harmonizer.harmonize_schema` | Remove trailing blank raw CSV columns | Before schema validation | Copy | Hard-coded | Important for BL011/BL027ab headers |
| Column rename | `schema_harmonizer.harmonize_schema` | Convert raw labels to canonical schema | Before validation | Copy | Hard-coded | CSV-specific assumptions embedded |
| Strain alias replacement | `cleaner.standardize_strain_names` | Convert `BL027ab` to `BL027` | Cleaning | Copy | Hard-coded | Only one alias |
| Chemical alias normalization | `run_real_analysis._normalize_real_chemical_aliases` | Normalize long chemical names for real data | Real workflow before cleaner canonicalization | Copy | Hard-coded in script | Separate from cleaner map |
| Chemical canonicalization | `cleaner.standardize_chemical_names` | Strip whitespace and canonicalize known names | Cleaning | Copy | Hard-coded | Unknown names preserved until filtering |
| Monensin exclusion | `cleaner.remove_excluded_chemicals` | Remove excluded chemical | Cleaning | Copy | Hard-coded | `EXCLUDED_CHEMICALS = ("Monensin",)` |
| Target filtering | `cleaner.filter_target_chemicals` | Keep six target chemicals | Cleaning | Copy | Hard-coded | Removes controls and non-targets |
| Concentration conversion | `cleaner.parse_concentration` | Convert concentration to numeric | Cleaning | Copy | Hard-coded regex | Units not preserved |
| Required-field drop | pipeline and real script | Remove incomplete rows | After concentration parsing | Copy via DataFrame operations | Hard-coded required columns | Drops rows with null required fields |
| Replicate/time aggregation | `_prepare_feature_input` | Average duplicate measurements at same group/time | Before feature extraction | New DataFrame | Hard-coded group columns | Aggregates luminescence mean |
| Curve minimum filter | `_prepare_feature_input` | Require at least two time points | Before feature extraction | New DataFrame | Hard-coded | Prevents slope/AUC errors |
| Experiment z-score | `normalized_features.add_experiment_zscore_features` | Normalize feature columns within experiment | Post-feature script | Copy | Hard-coded base columns | Output columns end `_zexp` |
| Strain-experiment z-score | `normalized_features.add_strain_experiment_zscore_features` | Normalize feature columns within strain and experiment | Post-feature script | Copy | Hard-coded base columns | Output columns end `_zstrain_exp` |
| PCA scaling | `visualization.plots._pca_projection`; `feature_importance._pca_projection` | Standardize features before PCA | Plot generation | New arrays | Hard-coded | Two implementations |
| Baseline-derived fold features | `advanced_features.calculate_peak_to_baseline_ratio`, `calculate_fold_change` | First signal as baseline proxy | Advanced feature extraction | New feature table | Hard-coded | Not control normalization |

Not found in active code:

- Explicit baseline subtraction.
- Control normalization from `Control` rows.
- Log transformation or log2 fold change.
- Outlier detection/removal.
- General common-window selection outside segmented advanced AUC windows.
- Configurable time conversion.
- Explicit curve truncation before base features.

## 7. Feature Extraction

Feature rows are generated per `strain`, `chemical`, `concentration`, `experiment`, and `replicate`.

| Feature | File/function | Definition | Units | Duration dependent | 12h/24h suitability |
| --- | --- | --- | --- | --- | --- |
| `auc` | `features.calculate_auc` | Trapezoidal integral over observed `time` and `luminescence` | luminescence x time | Yes | Not directly comparable across 12h and 24h unless window-normalized |
| `max_signal` | `features.calculate_max_signal` | Maximum luminescence in curve | luminescence | Somewhat | More comparable than AUC, but longer windows can expose later maxima |
| `min_signal` | `features.calculate_min_signal` | Minimum luminescence in curve | luminescence | Somewhat | Longer windows can expose later minima |
| `time_to_peak` | `features.calculate_time_to_peak` | Time value at maximum signal | time units from input | Yes | Depends on available duration and time units |
| `initial_slope` | `features.calculate_initial_slope` | `(signal[1] - signal[0]) / (time[1] - time[0])` | luminescence per time | Less, but depends on sampling interval | Likely comparable only with identical early sampling |
| `final_signal` | `features.calculate_final_signal` | Last signal value in sorted curve | luminescence | Yes | Not comparable between 12h endpoint and 24h endpoint without window labeling |
| `peak_to_baseline_ratio` | `advanced_features.calculate_peak_to_baseline_ratio` | `max(signal) / signal[0]` | ratio | Somewhat | Sensitive to duration because peak search window changes |
| `fold_change` | `advanced_features.calculate_fold_change` | `(signal[-1] - signal[0]) / signal[0]` | ratio | Yes | Endpoint-duration dependent |
| `max_derivative` | `advanced_features.calculate_max_derivative` | Maximum adjacent slope | luminescence per time | Somewhat | Depends on sampling/window |
| `min_derivative` | `advanced_features.calculate_min_derivative` | Minimum adjacent slope | luminescence per time | Somewhat | Depends on sampling/window |
| `signal_decay_rate` | `advanced_features.calculate_signal_decay_rate` | Slope from peak signal to final signal; `0.0` if peak is final or time delta zero | luminescence per time | Yes | Endpoint-duration dependent |
| `auc_early` | `advanced_features.calculate_auc_early` | AUC in 0-6 hours or 0-360 minutes | luminescence x time | Window-specific | Suitable if both datasets share this window and units are normalized |
| `auc_mid` | `advanced_features.calculate_auc_mid` | AUC in 6-12 hours or 360-720 minutes | luminescence x time | Window-specific | Suitable for 0-12h comparison if units align |
| `auc_late` | `advanced_features.calculate_auc_late` | AUC in 12-24 hours or 720-1440 minutes | luminescence x time | Window-specific | Not available for 12h-only datasets except as zero/no overlap |
| `*_zexp` | `normalized_features` | Z-score of base features within experiment | standard deviations | Depends on source feature | Normalized statistically, not duration-normalized |
| `*_zstrain_exp` | `normalized_features` | Z-score of base features within strain and experiment | standard deviations | Depends on source feature | Normalized statistically, not duration-normalized |

Feature gaps relative to requested attention list:

- Endpoint exists as `final_signal`.
- Baseline exists only as first-signal proxy inside advanced ratio/fold features; no `baseline` output column.
- Area above or below baseline is not implemented.
- Log2 fold change is not implemented.
- Duration-normalized AUC alternatives are not implemented.

## 8. Fingerprint Analysis

Current fingerprint representation:

- The main fingerprint table is `outputs/tables/features.csv`.
- Each row is a replicate-level kinetic fingerprint for one `strain` x `chemical` x `concentration` x `experiment` x `replicate`.
- Base vector columns are `auc`, `max_signal`, `min_signal`, `time_to_peak`, `initial_slope`, `final_signal`.
- Advanced vectors can add `peak_to_baseline_ratio`, `fold_change`, `max_derivative`, `min_derivative`, `signal_decay_rate`, `auc_early`, `auc_mid`, `auc_late`.

Heatmaps and fingerprint outputs:

- `src.visualization.plots.plot_heatmap`
  - Input: feature table.
  - Grouping: chemical by strain.
  - Aggregation: mean `auc`.
  - Output: `outputs/figures/heatmap.png`.
  - Supports arbitrary chemicals/strains present in data.
- `src.model_evaluation.chemical_specific_strains.plot_chemical_specific_strain_heatmap`
  - Input: chemical-specific ranking table.
  - Grouping: fixed chemical by fixed strain.
  - Aggregation: mean `f1`.
  - Output: `outputs/figures/chemical_specific_strain_heatmap.png`.
  - Hard-coded to six chemicals and six strains.
- Reduced-array fingerprints are represented through panel and ablation workflows, not through a separate reusable fingerprint object.
- Centroid fingerprints and similarity/distance matrices are not implemented as production outputs. `validate_fingerprint_distinctiveness` can evaluate distances if a matrix is supplied, but it is not wired into the main pipeline.

Hard-coding support:

- Core feature extraction is mostly arbitrary with respect to chemical and strain labels.
- Cleaning and validation restrict to six target chemicals and six expected strains.
- Chemical-specific ranking and specialist ensemble use hard-coded chemical/strain mappings.
- Candidate panels are hard-coded.

## 9. Statistical Analysis

| Method | File/function | Input | Grouping | Output | Active in main pipeline | Tests | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AUC heatmap descriptive aggregation | `plots.plot_heatmap` | `features.csv` style table | chemical x strain, mean AUC | PNG | Yes, via `run_real_analysis.py` | `test_plots.py` | Current |
| Dose-response plot | `plots.plot_dose_response` | feature table | by chemical if present | PNG | Yes | `test_plots.py` | Current, plotting only |
| PCA plot | `plots.plot_pca` | feature table | none; scatter only | PNG | Yes | `test_plots.py` | Current |
| PCA by chemical | `feature_importance.generate_pca_by_chemical` | `features.csv` | colored by chemical | PNG | No, post-script | `test_feature_importance.py` | Current |
| PCA by experiment | `feature_importance.generate_pca_by_experiment` | `features.csv` | colored by experiment | PNG | No, post-script | `test_feature_importance.py` | Current |
| Feature importance | `feature_importance.calculate_random_forest_feature_importance` | `features.csv` | RF classifier | CSV, PNG | No, post-script | `test_feature_importance.py` | Current |
| Confidence intervals | `confidence_intervals.bootstrap_confidence_interval`, `summarize_metric_confidence_intervals` | per-chemical metrics | numeric metric columns | CSV, report | No, post-script | `test_confidence_intervals.py` | Pre-existing untracked current/experimental |
| Repeated-run analysis | `repeated_runs.run_repeated_seed_evaluation`, `summarize_repeated_run_metrics` | `features.csv` | random seeds | CSVs, report, boxplot | No, post-script | `test_repeated_runs.py` | Pre-existing untracked current/experimental |
| Scientific cluster validation | `scientific_validation.validate_cluster_separation` | cluster labels | labels only | result object | No wiring found | `test_scientific_validation.py` | Utility only |
| Concentration dependence | `scientific_validation.validate_concentration_dependence` | concentration/response sequences | paired observations | result object | No wiring found | `test_scientific_validation.py` | Utility only |
| Replicate reproducibility | `scientific_validation.validate_reproducibility` | replicate matrix | pairwise correlations | result object | No wiring found | `test_scientific_validation.py` | Utility only |
| Fingerprint distinctiveness | `scientific_validation.validate_fingerprint_distinctiveness` | fingerprint matrix | pairwise distances | result object | No wiring found | `test_scientific_validation.py` | Utility only |

Not found as active implementations:

- Centroid PCA.
- Clustering model generation.
- Correlation or distance matrix output.
- Early-versus-late statistical comparison, beyond advanced segmented AUC features.
- Batch-effect analysis, beyond PCA by experiment and experiment z-scores.

## 10. Machine-Learning Architecture

| Workflow | Entry script | Core module | Target | Features | Split/CV | Models | Metrics | Outputs | Leakage notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline chemical classification | `run_real_analysis.py` | `model_training.models`, `evaluate` | `chemical` | base numeric features | No split in script; train and predict on same feature table | RandomForestClassifier, 100 trees, seed 42 | accuracy, macro precision/recall/F1, confusion matrix | `model_metrics.json` | High leakage/optimism risk because evaluation uses training data |
| Baseline concentration regression | `run_real_analysis.py` | `model_training.models`, `evaluate` | `concentration` | base numeric features | No split in script; train and predict on same feature table | RandomForestRegressor, 100 trees, seed 42 | R2, RMSE, MAE | `model_metrics.json` | High leakage/optimism risk because evaluation uses training data |
| LOEO classification | `run_loeo_validation.py` | `loeo_validation.run_loeo_classification` | `chemical` | base or supplied feature columns | Leave one experiment out | RandomForestClassifier | mean and per-experiment accuracy, macro precision/recall/F1 | `loeo_metrics.json` | Experiment leakage controlled by split; replicate linkage within held-out experiment is preserved |
| LOEO regression | `run_loeo_validation.py` | `loeo_validation.run_loeo_regression` | `concentration` | base or supplied columns | Leave one experiment out | RandomForestRegressor | R2, RMSE, MAE | `loeo_metrics.json` | Experiment leakage controlled |
| Normalized LOEO | `run_normalized_loeo.py` | `normalized_features`, `loeo_validation` | chemical and concentration | z-score feature columns | Leave one experiment out | Random Forest | LOEO metrics | `features_normalized.csv`, `loeo_metrics_normalized.json` | Normalization computed across full feature table before LOEO, so fold leakage should be reviewed |
| Per-chemical LOEO | `run_per_chemical_analysis.py` | `per_chemical_analysis` | chemical | base features | Leave one experiment out within selected strains | RandomForestClassifier | precision, recall, F1, support | `per_chemical_loeo.csv`, confusion matrix | Experiment split controlled |
| Advanced per-chemical LOEO | `run_advanced_per_chemical_analysis.py` | `advanced_per_chemical_analysis` | chemical | base + advanced features | Leave one experiment out, selected BL027 | RandomForestClassifier | precision, recall, F1, support | advanced table and confusion matrix | Experiment split controlled |
| Panel optimization | `run_panel_optimization.py` | `panel_optimization` | chemical | base features | LOEO per candidate panel | RandomForestClassifier | mean LOEO metrics | panel CSV and figures | Candidate list is hard-coded |
| Advanced panel comparison | `run_advanced_loeo_comparison.py` | `advanced_loeo_comparison` | chemical | base + advanced features | LOEO per candidate panel | RandomForestClassifier | mean LOEO metrics | advanced panel CSV and figure | Candidate list is hard-coded |
| Strain ablation | `run_strain_ablation.py` | `strain_ablation` | chemical | base features | LOEO after selecting/removing strains | RandomForestClassifier | mean LOEO metrics | ablation CSVs and figures | Experiment split controlled |
| Chemical-specific strain ranking | `run_chemical_specific_strains.py` | `chemical_specific_strains` | target chemical vs Other | available base + advanced features | LOEO by experiment per strain and chemical | RandomForestClassifier | precision, recall, F1, support | ranking CSV and heatmap | Fixed six chemicals/strains |
| Specialist ensemble | `run_specialist_ensemble.py` | `specialist_ensemble` | chemical | available base + advanced features | LOEO one-vs-rest specialists by experiment | RandomForestClassifier with probabilities | accuracy, macro precision/recall/F1 | metrics JSON and confusion matrix | Fixed specialist map; no saved models |
| Repeated-run robustness | `run_repeated_runs.py` | `repeated_runs` | chemical | base features unless supplied | random train/test split, optional stratification, repeated seeds | RandomForestClassifier | accuracy, precision, recall, F1; summary stats | metrics CSV, summary CSV, report, boxplot | Related replicates/experiments may appear in both train and test |
| Feature importance | `analyze_feature_importance.py` | `feature_importance` | chemical | base features | No validation split | RandomForestClassifier | feature importances | CSV and PNG | Interpretability only; trained on all rows |

Not found:

- Unknown-sample prediction API.
- Model persistence or saved model path.
- scikit-learn `Pipeline` preprocessing for models.
- Hyperparameter tuning.
- GroupKFold/LeaveOneGroupOut classes; LOEO is implemented manually by experiment value.
- XGBoost usage, despite `xgboost` in `requirements.txt`.

## 11. GUI Architecture

No GUI architecture was found in the inspected repository.

- GUI framework: none detected.
- Main GUI file: none found.
- Main window class: none found.
- Buttons and commands: none found.
- Pipeline launch style: scripts and direct imports only; no GUI subprocess/direct-import launcher found.
- Progress handling: none found.
- Logging: no GUI logging found.
- Error handling: script-level exceptions and return dictionaries only.
- Cancellation handling: none found.
- Output/report opening: none found.
- Raw-data selection: none found; real workflow hard-codes `data/raw`.
- Multiple input folders: not supported by GUI because GUI is absent.
- Future dual-source import support would logically connect below any future UI layer, at data import/schema harmonization rather than in model code.

## 12. Report Generation

Current generated report mechanisms:

- `src.reporting.report.generate_markdown_report`
  - Writes a Markdown file with fixed title and caller-supplied sections.
  - Used by `src.pipeline.run_pipeline` and `scripts.run_real_analysis`.
- `src.reporting.report.generate_validation_summary`
  - Converts classification/regression/scientific validation metric dictionaries into Markdown bullets.
  - Used by `scripts.run_real_analysis`.
- `scripts/run_confidence_intervals.py`
  - Builds a Markdown confidence-interval report directly with `_build_report`.
- `scripts/run_repeated_runs.py`
  - Builds a Markdown repeated-run report directly with `_build_report`.

Report outputs:

- `outputs/reports/analysis_report.md`
- `outputs/reports/scientific_performance_report.md`
- `outputs/reports/confidence_interval_report.md`
- `outputs/reports/repeated_run_analysis.md`

Not found:

- Markdown templates.
- Programmatic DOCX generation.
- Programmatic PDF generation.
- A report registry that verifies every referenced output exists.

Risk notes:

- Some project docs appear manually authored and may contain conclusions independent of generated outputs.
- Generated reports are assembled from in-memory metrics or existing output tables; missing primary output files usually raise `FileNotFoundError` in scripts.
- Report generation writes fixed output paths and can overwrite prior reports.

## 13. Configuration and Hard-Coded Assumptions

No YAML, TOML, or JSON configuration mechanism was found for pipeline behavior. Most configuration is Python constants.

| Assumption | Location | Impact on future 12h/24h integration |
| --- | --- | --- |
| Raw data directory is `data/raw` | `run_real_analysis.py`, `inspect_dataset.py`, e2e tests | Needs abstraction for dual CSV/Excel source folders |
| Raw files are `*.csv` | `run_real_analysis.py`, `inspect_dataset.py` | Blocks Excel import until importer is added |
| Canonical schema is fixed seven long-form columns | `run_pipeline.REQUIRED_COLUMNS`, `schema_harmonizer.REQUIRED_HARMONIZED_COLUMNS` | Good target schema, but import adapters must map Excel/wide forms |
| Raw column aliases are fixed | `schema_harmonizer.COLUMN_RENAMES` | Needs configurable schema mapping for 12h Excel |
| Target chemicals are six fixed values | `cleaner.TARGET_CHEMICALS`, `validator.ALLOWED_CHEMICALS`, docs/tests | Needs update/config for ten-chemical analysis |
| Expected strains are six fixed values | `validator.ALLOWED_STRAINS`, `chemical_specific_strains.STRAINS`, tests/docs | Future strain additions require code changes |
| Expected concentrations are five fixed values | `validator.EXPECTED_CONCENTRATIONS` | Future concentration ranges may fail validation |
| Monensin excluded | `cleaner.EXCLUDED_CHEMICALS` | Should be policy/config, not embedded |
| `BL027ab` maps to `BL027` | `cleaner.STRAIN_NAME_MAP` | More aliases need config |
| Chemical aliases are in script | `run_real_analysis.REAL_CHEMICAL_ALIASES` | Alias logic is split from cleaner |
| Time unit inferred by `max(time) <= 24` | `advanced_features._segment_bounds` | Fragile for mixed-hour/minute data and 12h/24h integration |
| Base model features are six fixed columns | `model_training.NUMERIC_FEATURE_COLUMNS` | New duration-normalized features require explicit wiring |
| Random Forest settings are fixed | `model_training`, `loeo_validation`, `feature_importance`, `repeated_runs` | No central ML config |
| Candidate panels fixed | `panel_optimization.CANDIDATE_PANELS`, `advanced_loeo_comparison.ADVANCED_PANEL_CANDIDATES` | Reduced-array search is not data-driven |
| Best panels fixed in scripts | `run_per_chemical_analysis.py`, `run_advanced_per_chemical_analysis.py` | Results depend on script constants |
| Specialist map fixed | `specialist_ensemble.get_specialist_mapping` | Ten-chemical specialist mapping requires code changes |
| Output paths fixed under `outputs` | all scripts | Running scripts overwrites prior outputs |
| Current report title fixed | `report.REPORT_TITLE` | Fine for current project; not config-driven |

## 14. Testing Architecture

- Test framework: pytest.
- Test files: 35 `test_*.py` files.
- Python files in `src`, `scripts`, and `tests`: 78.
- Fixtures:
  - `tests/fixtures/blackbox/*.csv`
  - `tests/fixtures/e2e/raw_sample.csv`
  - `tests/fixtures/integration/*.csv`
  - `tests/fixtures/regression/*.csv` and `.json`

Coverage areas by folder:

- `tests/unit/`
  - Loader, schema harmonizer, cleaner, base and advanced features, normalized features, models, evaluation, LOEO, panel optimization, per-chemical analysis, strain ablation, chemical-specific strains, specialist ensemble, plots, reports, validation utilities, confidence intervals, repeated runs.
- `tests/integration/`
  - Interface-like flow from cleaned rows to feature-like rows and classifier/regressor field requirements.
- `tests/e2e/`
  - Synthetic end-to-end flow.
  - Real raw CSV smoke tests.
  - Real pipeline tests that check raw files are not modified.
  - One e2e test runs `scripts/run_real_analysis.py` via subprocess and asserts generated outputs exist.
- `tests/regression/`
  - Feature schema and expected metric ranges from fixtures.
- `tests/blackbox/`
  - CSV validation behavior including missing columns, unknown chemicals, missing controls.
- `tests/contract/`
  - Placeholder contract tests that skip if production callables are not found.

Safety notes:

- Many unit tests use `tests/tmp` local workspaces and delete them with `shutil.rmtree`.
- Several tests create figures or reports in temporary folders.
- Real-data e2e tests read `data/raw` and compare file metadata before and after.
- `tests/e2e/test_real_analysis_outputs.py` runs the full real analysis script and can regenerate tracked/ignored outputs; it was not run.
- Test discovery was not run in Phase 1 to avoid import/cache side effects.

Coverage gaps or concerns:

- No test for Excel import.
- No test for 12h/24h common-window integration.
- No test for duration-normalized AUC.
- No GUI tests because no GUI was found.
- Grouped splitting is tested for LOEO and overlap utility, but random repeated runs can still split related replicates/experiments.
- No test proving control normalization, because control normalization is not implemented.
- No explicit test for model persistence, because model persistence is absent.

## 15. Duplication and Technical Debt

- Multiple scripts repeat `PROJECT_ROOT`, `sys.path` insertion, `outputs/tables`, `outputs/figures`, and JSON printing patterns.
- Multiple report builders exist: shared `src.reporting.report` plus ad hoc builders in confidence-interval and repeated-run scripts.
- PCA is implemented twice:
  - SVD manually in `src.visualization.plots`.
  - `StandardScaler` + sklearn PCA in `src.model_evaluation.feature_importance`.
- Plotting logic is duplicated across `src.visualization`, model-evaluation modules, and scripts.
- Confusion-matrix plotting is duplicated across per-chemical, advanced per-chemical, and specialist ensemble modules.
- Chemical/strain constants are repeated in cleaner, validator, tests, and chemical-specific modules.
- Panel definitions are hard-coded and split between base and advanced modules.
- Some feature and evaluation modules are current but untracked in Git.
- Fixed output paths make repeated runs overwrite artifacts.
- Generated outputs are tracked in Git, while some large outputs are ignored; output policy is inconsistent.
- No central configuration layer separates data, analysis, and presentation settings.
- No package metadata file was found (`pyproject.toml`, `setup.py`, etc.).
- `src/` lacks obvious `__init__.py` files; imports rely on namespace-package behavior and script `sys.path` insertion.
- `__pycache__` directories are present in the project tree.
- No circular imports were obvious from static import scans, but a full import graph was not generated.

## 16. Scientific and Software Risks

- Baseline model metrics in `run_real_analysis.py` train and evaluate on the same feature table, so they should not be interpreted as independent validation.
- LOEO scripts are stronger for experiment-level generalization, but they are not the first report metrics in the main real-analysis script.
- Control rows exist, and requirements mention control normalization, but main code filters controls out rather than normalizing to them.
- Duration-dependent features are not yet safe for direct 12h-vs-24h comparison.
- Time-unit inference by maximum time is fragile.
- Output overwrite risk is high because scripts write fixed filenames.
- Generated outputs tracked in Git may become stale relative to source.
- Dirty Git state pre-existed Phase 1.
- Excel dependency exists (`openpyxl`), but no Excel importer exists.
- Multiple docs contain scientific claims; this audit did not regenerate or validate those claims.

## 17. Future Integration Points

Recommended locations, based strictly on current architecture:

| Future need | Safest likely integration point | Rationale |
| --- | --- | --- |
| 12-hour Excel import | Add a new importer beside `src.data_ingestion.loader`, then feed canonical long-form output into `schema_harmonizer` or a new canonicalizer | Keeps file-format logic separate from analysis |
| Canonical long-format schema | Strengthen `src.preprocessing.schema_harmonizer` or add `src.data_ingestion.canonical_schema` | Existing downstream code already expects long form |
| Common 0-12h windows | Add explicit window selection before feature extraction, likely near `_prepare_feature_input` or a new preprocessing module | Prevents duration leakage before feature calculation |
| 12-24h late windows | Add explicit late-window feature path in `advanced_features` with clear missing-window semantics | Current `auc_late` already names the concept but needs source-duration awareness |
| Complete 0-24h windows | Parameterize feature extraction by duration/window | Avoids mixing 12h and 24h endpoints |
| Duration-normalized features | Add new feature functions in `src.feature_engineering` and wire selected columns through model constants/config | Avoids altering existing base features |
| Ten-chemical analysis | Move chemical lists from Python constants into config and update cleaner/validator/tests | Current lists are repeated and hard-coded |
| Combined reporting | Extend `src.reporting.report` into a shared report layer used by all scripts | Reduces ad hoc report duplication |
| GUI input-folder selection | No GUI exists; if added, connect it to a data-import facade that returns canonical long-form tables | Keeps UI away from scientific transformations |

## 18. Uncertainties Requiring Later Verification

- Whether VS Code is currently selecting `.venv`.
- Whether docs claiming control normalization reflect older intended behavior or missing implementation.
- Whether BL030/BL031 header handling is fully robust for all raw files beyond current data.
- Whether output files in Git are current relative to source.
- Whether existing untracked confidence-interval and repeated-run files should become formal project files.
- Whether the external 12-hour Excel files are wide-form plate exports or long-form tables.
- Whether future analysis should compare by absolute time, normalized duration fraction, or both.
- Whether a GUI exists outside this repository.

## 19. Phase 1 Conclusion

Phase 1 is complete as an inspection and architecture mapping pass.

Decision: PASS WITH BLOCKERS.

Blockers before Phase 2:

- Confirm and preserve the pre-existing dirty Git state.
- Decide whether the untracked confidence-interval and repeated-run modules are part of the intended baseline.
- Define a protected output/versioning plan before any regenerated outputs.
- Confirm intended interpreter selection in VS Code.
- Design the 12h Excel import around a canonical long-format schema before touching scientific analysis code.

Verification:

- No source code was modified.
- No tests were modified.
- No configuration files were modified.
- No raw data was modified intentionally or through pipeline execution.
- No full pipeline, tests, model training, package installation, or output regeneration was run.
- The only intended Phase 1 files are:
  - `docs/phase_1_project_architecture_audit.md`
  - `docs/phase_1_architecture_map.md`
