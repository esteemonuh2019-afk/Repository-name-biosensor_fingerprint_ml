# Stage 4B Canonical Builder

## Purpose

Stage 4B maps structured CSV and Excel reader results into the canonical long-format biosensor schema. It builds canonical tables and runs schema validation only. It does not run preprocessing, feature extraction, machine learning, reporting, the GUI, or the scientific pipeline.

## Data Flow

```text
file discovery
  -> CSV or Excel reader
  -> source-specific canonical mapper
  -> combined canonical DataFrame
  -> validate_canonical_schema
```

The builder consumes reader result objects. It does not reopen source files during mapping and does not modify raw data.

## CSV Mapping

CSV reader results are mapped from the 24-hour long-format columns:

- `antibiotic` -> `Chemical_Name_Original`
- `concentration` -> `Concentration_Label`
- `replicate` -> `Replicate_ID`
- `time_min` -> `Time_Original`, `Time_Minutes`, `Time_Hours`
- `luminescence` -> `Luminescence_Raw`

`Strain_Original` uses the filename-inferred strain label from file discovery/reader metadata so `BL027ab` remains preserved. If a row-level source strain column disagrees with the filename-inferred label, the record is flagged for review.

CSV rows use `Source_Type = csv` and `Data_Source = 24_hour_csv`.

## Excel Mapping

Excel reader results are mapped from the active worksheet table. The current 12-hour workbooks are already long-format tables with a blank first column followed by:

- `bacteria_id`
- `antibiotic`
- `concentration`
- `Experiment`
- `replicate`
- `time_min`
- `luminescence`

The blank leading worksheet column is not used as measurement metadata. Labels such as `Lambda Cyclotherin` and `Monesin sodium` are preserved exactly and are not corrected.

Excel rows use `Source_Type = xlsx`, `Data_Source = 12_hour_excel`, and the active worksheet name is recorded in `Worksheet`.

## Canonical Output

The output DataFrame uses the exact ordered columns from `CANONICAL_COLUMNS` in `src/data_schema/canonical_schema.py`.

The builder sets:

- `Analysis_Window` to `unassigned`
- `Luminescence_Normalized` to null
- `Normalization_Method` to null
- `Replicate_Type` to `unspecified`
- `Plate_ID` and `Well_ID` to null when unavailable
- `Measurement_Unit_ID` from source-supported unit evidence

`Experiment_ID` is generated deterministically from source type, source filename stem, and the source `Experiment` value when no explicit experiment ID is supplied. `Source_Row_ID` is the deterministic one-based reader data-row position and is not overloaded with source-column identity.

`Measurement_Unit_ID` uses the best available source evidence:

- `Plate_ID` plus `Well_ID` when both are present
- `Well_ID` when present without a plate
- stable source measurement-column position when multiple measurement columns exist
- deterministic source-position curve starts when the source is long format and lacks well/plate identifiers

Source-position unit IDs are flagged with `measurement_unit_id_synthetic`. When duplicate measurement columns with the same display name are present, each source column is retained as a separate measurement unit and flagged with `measurement_unit_id_uses_source_column`.

Missing `Replicate_ID` values are repaired only when adjacent source rows have matching experiment, strain, chemical, concentration, worksheet context, and the same non-missing replicate label. Repaired values are flagged with `replicate_id_inferred_from_source_position`. Unresolved replicate labels remain null and are reported as ambiguous.

## QC Behaviour

The builder never silently deletes rows. Conversion problems are represented through:

- `QC_Status`
- `QC_Flags`
- `Record_Valid`
- `Notes`

Invalid records remain in the output with `Record_Valid = False`. Negative raw luminescence values are retained and warned. Infinite luminescence values are invalid. Negative concentrations are invalid. Numeric concentrations are parsed separately from the original concentration labels and flagged as unit-unverified until source units are confirmed.

## Command Example

```powershell
python scripts/build_canonical_dataset.py "C:\Users\USER\Desktop\biosensor_phase2_source_files"
```

This prints a concise summary and does not save a dataset.

To save explicitly:

```powershell
python scripts/build_canonical_dataset.py "C:\Users\USER\Desktop\biosensor_phase2_source_files" --output outputs/canonical/canonical_dataset.csv
```

The script refuses to overwrite an existing file unless `--overwrite` is supplied. It also refuses to save inside the raw source folder.

## Limitations

- No chemical or strain standardization is performed.
- No canonical dataset is saved unless explicitly requested.
- Concentration units are not confirmed by the raw sources, so numeric concentration parsing is flagged.
- `Plate_ID` and `Well_ID` are unavailable in the inspected CSV and Excel sources and remain null.
- Source-position-derived `Measurement_Unit_ID` values are synthetic because the inspected sources do not provide physical well/plate identifiers.
- The builder does not invent missing time, concentration, or luminescence values.

## Unresolved Mappings

The following source-label questions remain unresolved and are intentionally not corrected:

- `BL027` versus `BL027ab`
- `Lambda Cyclotherin`
- `Monesin sodium` versus `Monensin`
- `Trimethoprim` versus `Trimetropin`
- `Boric Acid` versus `Boric acid`
- `N,N-Diethyl-m-Toluamide` versus `N,N-Diethyl-m-Toluamide (DEET)`
- `BL032` appearing as a chemical value in one CSV row
