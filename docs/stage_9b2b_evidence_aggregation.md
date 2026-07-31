# Stage 9B.2B Evidence Aggregation

Stage 9B.2B reduces the Stage 9B.2A evidence database into compact, traceable scientific summary tables. It does not rerun analyses, regenerate results, modify upstream evidence files, or write Results/Discussion prose.

## Inputs

The aggregation engine reads one required input:

```text
outputs/scientific_narrative_2/scientific_evidence.csv
```

If present beside the CSV, the engine also reads:

```text
outputs/scientific_narrative_2/scientific_evidence.json
```

The JSON is used only for run metadata such as unsupported or unreadable file counts. Metadata-only values are not inserted into summary evidence rows because every non-missing summary row must remain traceable to source evidence IDs from the CSV.

## Command

```text
python scripts/aggregate_scientific_evidence.py --evidence-file "outputs/scientific_narrative_2/scientific_evidence.csv" --output-dir "outputs/scientific_aggregation"
```

Supported options:

```text
--evidence-file
--output-dir
--overwrite
--maximum-source-ids-per-cell
```

`--overwrite` is required when the output directory already exists and is not empty.

## Outputs

The engine writes exactly these files:

```text
aggregated_evidence.csv
aggregated_evidence.json
dataset_summary.csv
qc_summary.csv
exploratory_summary.csv
classification_summary.csv
regression_summary.csv
feature_engineering_summary.csv
feature_selection_summary.csv
strain_summary.csv
limitations_summary.csv
blind_validation_status.csv
evidence_traceability.csv
aggregation_report.md
```

`fingerprint_summary` is retained inside `aggregated_evidence.json` and `aggregated_evidence.csv`, but no separate `fingerprint_summary.csv` is written in this stage.

## Summary Schema

Each summary row contains:

```text
summary_id
analysis_section
summary_type
metric_name
metric_value
metric_units
model_name
biological_entity
comparison_group
rank
direction
evidence_record_count
source_evidence_ids
source_files
aggregation_method
confidence
status
notes
```

`status` can be:

```text
OK
CONFLICT
MISSING
```

`MISSING` rows are used when an expected summary cannot be formed from the input evidence. `CONFLICT` rows preserve conflicting values rather than resolving or interpreting them.

## Aggregation Rules

Classification model rankings are sorted by macro F1, balanced accuracy, accuracy, then model name.

Regression model rankings are sorted by R2, RMSE, MAE, then model name.

Duplicate scalar evidence is selected using deterministic source priority. Best-model JSON files and model-ranking CSV files are preferred over broader summary files. If preferred records still contain conflicting values, all conflicting values are preserved as `CONFLICT`.

Feature selection summaries identify best observed subset rows and the smallest subset within 1 percent of the best observed performance where the required evidence columns are available.

Feature engineering summaries aggregate direct feature-family ablation values and non-interpretive extrema such as highest R2 and lowest RMSE.

Exploratory summaries include PCA variance rows, top matrix values, and matrix extrema where the evidence database contains machine-readable tables.

Blind validation status separates prediction infrastructure evidence from real experimental blind-validation availability.

## Traceability

Every non-missing summary row carries one or more source evidence IDs. `evidence_traceability.csv` maps each summary row back to:

```text
source_evidence_id
source_file
source_run
analysis_type
original_metric_name
original_metric_value
```

Traceability coverage is reported as the fraction of non-missing summary rows with source evidence IDs.

## Non-Interpretive Boundary

The engine extracts and aggregates existing evidence only. It does not:

```text
perform scientific analyses
regenerate results
create figures
write Results or Discussion sections
generate DOCX or PDF reports
modify original evidence outputs
```

## Transition To Scientific Interpretation

The next stage can consume `aggregated_evidence.json`, section CSVs, and `evidence_traceability.csv` as the evidence substrate for narrative drafting. Before interpretation, summaries with `status=CONFLICT` or `status=MISSING` should be reviewed as evidence constraints rather than resolved by this aggregation stage.
