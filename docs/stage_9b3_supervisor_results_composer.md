# Stage 9B.3 Supervisor Results Composer

## Purpose

Stage 9B.3 builds a supervisor-ready results package from the preferred outputs listed in
`outputs/results_inventory_2/selected_results.csv`.

The composer does not rerun analyses, regenerate benchmarks, or rediscover files. It uses
only selected inventory entries and their listed companion files.

## Architecture

The implementation lives in `src/supervisor_report/`.

- `authoritative_source_loader.py` reads selected inventory rows and resolves only listed paths.
- `results_composer.py` builds the structured `SupervisorResultsPackage`, validates coherence, and writes artifacts.
- `report_models.py` defines package and source data models.
- `table_builder.py` writes the supervisor workbook and selected-table index.
- `figure_selector.py` copies readable listed figures into the package figure directory.
- `provenance_tracker.py` records quantitative claim provenance.
- `scientific_interpreter.py` adds conservative, source-grounded interpretation text.

The command-line entry point is:

```bash
python scripts/build_supervisor_results_package.py --project-root "." --selected-results "outputs/results_inventory_2/selected_results.csv" --output-dir "outputs/supervisor_results"
```

Use `--overwrite` to replace an existing non-empty output directory.

## Authoritative Model Coherence

Classification model identity is loaded from the selected classification best-model metadata
file, currently `classification/stage_8a/best_model_metrics.json`.

Regression model identity is loaded from the selected regression best-model metadata file,
currently `regression/stage_8b_2/best_regression_model.json`.

Primary classification and regression metrics are populated only for those selected models.
Comparison rows from model-ranking tables remain separate and are not used to replace metrics
for the selected model. If a selected-model metric is unavailable, it is marked `MISSING`.

## Figure Selection

The figure selector considers only inventory-listed image files with readable image suffixes
such as `.png`, `.jpg`, or `.jpeg`. PDF figures may remain source references, but they are not
copied as embedded report images by this stage.

Selection follows a fixed preference order covering consensus fingerprint heatmaps, chemical
similarity heatmaps, dendrograms, strain contribution figures, and related benchmark figures.
Duplicate files are skipped.

## Interpretation Policy

The report text is intentionally conservative. It states which benchmark outputs selected the
models, reports quantitative metrics, and separates benchmark, feature-family,
feature-selection, and strain-ablation results.

The selected blind-prediction output is reported as prediction context only when true labels
are absent. The package does not claim real blind-validation performance without true labels.

## Validation

The package must pass validation before DOCX and PDF outputs are written.

Validation checks include:

- classification primary metrics belong to one selected classifier
- regression primary metrics belong to one selected regressor
- no primary model-metric mixing
- copied figures exist
- populated tables have source files
- supported quantitative claims have provenance
- real blind validation is not claimed unless true labels are present
- chemical and strain names are preserved from selected sources
- available units are reported for target-scale regression metrics
- missing information is marked

## Provenance

Every quantitative metric added to the report is recorded in `provenance_index.csv` with:

- section
- claim text
- metric name
- metric value
- units where available
- model name where relevant
- source file
- source run
- table or figure reference
- support status

This allows supervisor-facing claims to be traced back to the selected inventory sources.

## Outputs

The standard output directory is `outputs/supervisor_results/`.

Generated files are:

- `supervisor_results_summary.json`
- `supervisor_results_tables.xlsx`
- `supervisor_results_report.md`
- `supervisor_results_report.docx`
- `supervisor_results_report.pdf`
- `selected_figures.csv`
- `selected_tables.csv`
- `provenance_index.csv`
- `report_validation.json`
- `figures/`

## Limitations

The composer reports limitations from selected QC, feature, blind-prediction, and project
documentation sources. It does not perform new scientific analysis or modify source outputs.

The current DOCX and PDF writers are dependency-light exporters. They prioritize reproducible
text, traceability, and artifact availability over advanced page-design features.

## Future Updates

Future stages may add richer document rendering, embedded images in DOCX, extended figure
caption handling, and direct export from a styled document renderer. Those changes should
preserve the current authoritative-model and provenance validation rules.
