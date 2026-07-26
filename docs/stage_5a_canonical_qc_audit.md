# Stage 5A Canonical Dataset QC Audit

This stage adds a read-only quality-control audit for canonical biosensor
DataFrames. It classifies duplicate measurements, missing identifiers, numeric
measurement problems, and time-series issues without deleting records or
modifying any source, builder, schema, GUI, feature extraction, PCA, plotting,
statistics, or machine-learning code.

## Files Added

- `src/quality_control/__init__.py`
- `src/quality_control/canonical_qc.py`
- `scripts/audit_canonical_dataset.py`
- `tests/unit/test_canonical_qc.py`
- `tests/integration/test_canonical_qc_pipeline.py`
- `docs/stage_5a_canonical_qc_audit.md`

## Audit Entry Points

- `audit_canonical_dataframe(dataframe)` audits an in-memory canonical
  DataFrame and returns a `CanonicalQCResult`.
- `write_qc_outputs(result, output_dir, overwrite=False)` optionally writes a
  report folder. It refuses to overwrite an existing directory unless
  `overwrite=True`.
- `python scripts/audit_canonical_dataset.py "<source_folder>"` discovers source
  files, reads them, builds the canonical DataFrame in memory, and audits it.
- `python scripts/audit_canonical_dataset.py --canonical-file "<file.csv>"`
  audits an existing canonical CSV.

The script prints a concise summary by default. It writes no outputs unless
`--output-dir` is provided.

## Duplicate Keys

The existing schema logical duplicate key is:

```text
Experiment_ID, Plate_ID, Strain_Original, Chemical_Name_Original, Concentration_Label, Replicate_ID, Well_ID, Time_Minutes
```

Stage 5A reports this key separately from a source-aware variant:

```text
Source_File, Experiment_ID, Plate_ID, Strain_Original, Chemical_Name_Original, Concentration_Label, Replicate_ID, Well_ID, Time_Minutes
```

`Plate_ID` and `Well_ID` are currently often unavailable in canonical data, so
duplicate groups involving those missing identifiers remain ambiguous until
the import design or source metadata provides a way to distinguish physical
wells or plates.

## Checks

- Exact duplicate rows across all canonical columns.
- Duplicate `Source_File` + `Source_Row_ID` provenance keys.
- Current schema logical duplicate rows and duplicate groups.
- Source-aware logical duplicate rows and duplicate groups.
- Duplicate groups with identical versus conflicting `Luminescence_Raw` values.
- Ambiguous duplicate groups caused by missing `Replicate_ID`, `Plate_ID`, or
  `Well_ID`.
- Missing required canonical fields.
- Missing identifier counts for experiment, plate, well, replicate, and source
  row ID.
- Negative concentration, negative time, inconsistent hour/minute time values,
  negative luminescence, and infinite luminescence.
- Non-monotonic measurement series and duplicate time-point groups.

## Real Dataset Audit

The real source-folder audit is run with:

```powershell
python scripts/audit_canonical_dataset.py "C:\Users\USER\Desktop\biosensor_phase2_source_files"
```

The Stage 5A run completed against 12 discovered source files and produced these
read-only in-memory audit counts:

- Total canonical rows: 1,834,346
- Exact duplicate rows: 0
- `Source_File` + `Source_Row_ID` duplicate rows: 0
- Current schema logical duplicate rows: 30,697
- Current schema logical duplicate groups: 14,438
- Source-aware logical duplicate rows: 30,697
- Source-aware logical duplicate groups: 14,438
- Identical-value duplicate rows: 0
- Conflicting-value duplicate rows: 30,697
- Ambiguous duplicate rows: 30,697
- Ambiguous duplicate groups: 14,438
- Duplicate time-point groups: 14,438
- Non-monotonic measurement-series groups: 379
- Missing `Experiment_ID`: 0
- Missing `Plate_ID`: 1,834,346
- Missing `Well_ID`: 1,834,346
- Missing `Replicate_ID`: 11
- Missing `Source_Row_ID`: 0
- Negative luminescence rows: 0
- Infinite luminescence rows: 0
- QC passed: false

No canonical dataset or QC output folder is created unless the script is run
with `--output-dir`.

The earlier 1,296-record duplicate figure was not reproduced by the current
Stage 5A audit using the logical key documented above. The current in-memory
run reports 30,697 duplicate logical rows. Because adding `Source_File` to the
key does not change the count, the duplicate groups are not explained by
cross-file collisions alone. They remain unresolved measurement-key collisions
with conflicting raw luminescence values and missing plate/well identifiers.

## Importer Follow-Up Questions

- Are duplicate logical measurements intended technical repeats, or do they
  indicate missing well/plate identifiers?
- Should the official duplicate key include `Source_File`, or is
  `Experiment_ID` guaranteed to encode source identity in every future import?
- How should duplicate rows with conflicting raw luminescence be represented
  downstream if they cannot be separated by `Well_ID` or `Plate_ID`?
- Should missing `Well_ID` and `Plate_ID` be treated as acceptable source
  limitations, or as blocking metadata defects for normalization and model
  training?
- Should `BL027` and `BL027ab` remain separate throughout QC, or should a later
  approved mapping relate them while preserving originals?
- Should `Lambda Cyclotherin` be verified as written before any chemical-name
  mapping is approved?
