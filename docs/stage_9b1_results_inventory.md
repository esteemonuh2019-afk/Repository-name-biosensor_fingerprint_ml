# Stage 9B.1 Results Inventory and Selection Engine

## Purpose

Stage 9B.1 builds a read-only inventory of generated project outputs. It scans
the output tree recursively, classifies every discovered file, detects repeated
analysis runs, selects the newest complete run for each analysis category, and
assesses which evidence is available for a future supervisor report.

This stage does not generate the supervisor report. It does not rerun scientific
analyses and does not modify or delete existing scientific outputs.

## Inventory Architecture

The implementation lives in `src/results_inventory/`.

- `inventory_models.py` defines structured dataclasses for files, runs,
  duplicate candidates, obsolete candidates, missing results, selected report
  rows, and the top-level `ResultsInventory`.
- `inventory_scanner.py` recursively scans `outputs/`, records deterministic
  file metadata, applies the large-file hash guard, and orchestrates the full
  inventory build.
- `result_classifier.py` classifies files using folder path, filename,
  extension, nearby companion filenames, and known stage conventions.
- `run_selector.py` detects run directories, applies explicit successful-run
  rules, selects preferred runs, and reports duplicate or obsolete candidates.
- `completeness_checker.py` maps selected outputs onto required
  supervisor-report sections and creates the project-health summary.
- `inventory_report.py` writes the required CSV, JSON, and Markdown inventory
  files.

The command-line entry point is:

```bash
python scripts/build_results_inventory.py --project-root "." --output-dir "outputs/results_inventory"
```

## Classification Rules

Classification uses multiple signals rather than filenames alone.

- Stage folders identify primary analysis type: `qc`, `features`,
  `feature_validation`, `fingerprints`, `exploratory`, `classification`,
  `regression`, `feature_engineering`, `feature_selection*`, and
  `blind_prediction`.
- Stage-like run folders such as `stage_8a`, `stage_8a_2`,
  `stage_7b_3`, `stage_8b_2`, and `feature_selection_3` are preserved as run
  names and version hints.
- File extensions identify machine-readable tables, JSON summaries, figures,
  reports, and model-metric outputs.
- Nearby companion filenames help distinguish complete run bundles from
  isolated legacy tables.
- Top-level legacy `outputs/tables`, `outputs/figures`, and `outputs/reports`
  artifacts are classified by known project conventions but remain lower
  priority than complete stage run directories.

Scientific roles include dataset summary, canonical QC, feature QC,
fingerprint QC, PCA, clustering, heatmaps, chemical similarity, concentration
trajectories, classification performance, confusion matrix, per-class metrics,
classification feature importance, regression performance,
prediction-versus-actual, residual analysis, regression feature importance,
feature-family benchmark, feature selection, strain ablation,
reduced-array optimisation, blind prediction, scientific report,
supplementary material, and unknown.

Unknown files are retained in the inventory as `unknown`; they are not forced
into a scientific category.

## Successful-Run Rules

Runs are evaluated with explicit required-output rules.

Classification runs require:

- `classification_summary.csv`
- `best_model_metrics.json`
- `confusion_matrix.csv`
- `per_class_metrics.csv`
- `model_rankings.csv`
- `classification_report.md`

Regression runs require:

- `regression_summary.csv`
- `best_regression_model.json`
- `prediction_vs_actual.csv`
- `residuals.csv`
- `model_rankings.csv`
- `regression_report.md`

Exploratory runs require PCA scores, loadings, explained variance, cluster
assignments, an exploratory report or summary, and at least one major
exploratory figure.

Feature-engineering runs require the Stage 8C advanced feature tables, summary,
feature-family benchmark tables, and Stage 8C report.

Feature-selection runs require the Stage 8D selected-feature, ranking,
summary, after-selection benchmark, performance-versus-feature-count, and
report outputs.

Fingerprint, feature-validation, feature-extraction, canonical-QC, and
blind-prediction runs have analogous required-output rules based on their
writer modules and stage documentation.

Optional figures and audit tables improve completeness and report suitability
but do not by themselves block a run when the required machine-readable and
report artifacts are present.

## Latest-Valid-Run Selection

For each analysis category, the selector chooses one preferred run using this
priority order:

1. Complete runs before incomplete runs.
2. Non-diagnostic runs before smoke/debug/test runs.
3. Runs with required machine-readable outputs.
4. Runs with companion figures or reports.
5. Newest modification time.
6. Most complete required-output coverage.
7. Explicit stage/version hints.

A newer incomplete run cannot replace an older complete run. Every selected run
records the exact selection reason in `detected_runs.csv`,
`selected_results.csv`, and `output_inventory.json`.

## Duplicate Handling

Duplicate detection reports exact filename duplicates across different run
directories. It does not delete or merge files.

Obsolete/review candidates include:

- superseded run directories;
- partial run directories;
- diagnostic or smoke run directories;
- empty run directories;
- large generated files that should remain excluded from Git.

The project `.gitignore` excludes generated `outputs/` and `models/` artifacts,
so the inventory reports these candidates as review information only.

## Completeness Rules

The supervisor-report readiness check evaluates these sections:

1. Dataset summary.
2. Data-quality summary.
3. Fingerprint summary.
4. PCA/exploratory analysis.
5. Chemical similarity or clustering.
6. Classification results.
7. Regression results.
8. Feature engineering results.
9. Feature-selection results.
10. Strain contribution or ablation, if available.
11. Limitations.
12. Blind validation status.

Section statuses are `FOUND`, `PARTIAL`, `MISSING`, or
`NOT YET APPLICABLE`. Real blind validation is marked `NOT YET APPLICABLE`
unless truth-reveal or evaluation outputs are detected. Blind-prediction
infrastructure can still be marked complete in the project-health summary when
the model bundle metadata and prediction tooling exist.

The machine-readable project-health summary uses `COMPLETE`, `PARTIAL`,
`MISSING`, and `NOT_APPLICABLE` for ingestion, canonical QC, feature
extraction, feature validation, fingerprint generation, exploratory analysis,
classification, regression, feature engineering, feature selection,
blind-prediction infrastructure, real blind validation, and supervisor report.

## Output Files

The writer creates exactly these filenames in the requested inventory output
directory:

- `output_inventory.csv`
- `output_inventory.json`
- `detected_runs.csv`
- `selected_results.csv`
- `duplicate_candidates.csv`
- `obsolete_candidates.csv`
- `missing_results.csv`
- `project_results_health.json`
- `results_inventory_report.md`

Existing inventory output directories are not overwritten unless `--overwrite`
is supplied. The writer refuses to write directly into the broad `outputs/`
directory.

## Limitations

The inventory engine does not interpret metric values, compare model
performance numerically, validate figures visually, or judge biological
significance. It selects by completeness, role, machine readability, companion
artifacts, modification time, and known run naming.

Hashing is used only for files at or below the configured hash-size threshold.
Files above the threshold are still inventoried but are not hashed unless the
caller explicitly requests large-file hashing with `--include-large-files`.

## Transition to Supervisor-Report Generation

Stage 9B.1 produces the evidence map needed by a later supervisor-report stage.
The next stage should consume `selected_results.csv`,
`project_results_health.json`, and `results_inventory_report.md`, then assemble
the supervisor report from selected files only. It should preserve the blind
validation caveat unless real truth-reveal validation results have been added.
