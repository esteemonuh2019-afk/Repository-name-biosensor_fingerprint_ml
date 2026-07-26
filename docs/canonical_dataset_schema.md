# Canonical Biosensor Dataset Schema

Schema version: `1.1.0`

## Purpose

The canonical biosensor schema defines the long-format internal data contract for future CSV ingestion, Excel ingestion, preprocessing, quality control, time-window extraction, feature extraction, fingerprint analysis, statistics, machine learning, reporting, and any future GUI.

This document defines the schema only. It does not build a canonical dataset, combine CSV and Excel sources, standardize labels, extract features, or make scientific claims.

## Long-Format Definition

Each canonical row represents one luminescence value for one experimental unit at one measured time point. `Measurement_Unit_ID` identifies the experimental unit; strain, chemical, concentration, and replicate labels must remain internally consistent within that unit, but they do not substitute for the unit identifier.

The raw measurement is preserved in `Luminescence_Raw`. Derived normalized values must be stored separately in `Luminescence_Normalized` and must never overwrite raw signal values.

## Required Fields

These fields must be populated for every canonical measurement row:

- `Experiment_ID`
- `Source_File`
- `Source_Type`
- `Measurement_Unit_ID`
- `Strain_Original`
- `Chemical_Name_Original`
- `Concentration_Label`
- `Time_Minutes`
- `Time_Hours`
- `Luminescence_Raw`
- `QC_Status`
- `Record_Valid`

All 35 canonical columns are expected to be present in a canonical table. Optional fields may contain nulls when unavailable.

## Optional and Nullable Fields

Nullable fields include source identifiers not present in the raw files, later-stage standardized labels, optional timing provenance, normalized measurements, and free-text notes. Missing optional identifiers must remain null; importers must not invent `Plate_ID`, `Well_ID`, standardized labels, or control types.

Examples of explicitly nullable fields:

- `Plate_ID`
- `Worksheet`
- `Well_ID`
- `Replicate_ID`
- `Control_Type`
- `Replicate_Type`
- `Luminescence_Normalized`
- `Normalization_Method`
- `Notes`

## Controlled Values

`Source_Type`: `csv`, `xlsx`

`Data_Source`: `24_hour_csv`, `12_hour_excel`, `unknown`

`Control_Status`: `treatment`, `control`, `unknown`

`Control_Type`: `untreated`, `solvent`, `blank`, `zero_concentration`, `shared_control`, `unknown`

`Replicate_Type`: `technical`, `biological`, `unspecified`

`QC_Status`: `pass`, `warning`, `fail`, `not_evaluated`

`Record_Valid`: `True`, `False`

`Analysis_Window`: `unassigned`, `0-12h_Common`, `0-12h_Early`, `12-24h_Late`, `0-24h_Full`, `0-12h_Full`

Unknown controlled values must be reported by validation. They must not be silently replaced.

## Data Dictionary

| Field | Required | Type | Missingness | Allowed values | Scientific meaning | Validation rules | Example |
|---|---:|---|---|---|---|---|---|
| `Experiment_ID` | Yes | `string` | Not nullable | Free text | Separates independent experimental runs. | Must be present and non-missing. | `EXP-001` |
| `Plate_ID` | No | `string` | Nullable | Free text | Identifies a physical plate when known. | Do not invent missing plate identifiers. | `Plate-01` |
| `Source_File` | Yes | `string` | Not nullable | Free text | Preserves raw filename provenance. | Must preserve original filename. | `BL027ab.csv` |
| `Source_Path` | No | `string` | Nullable | Free text | Records raw-file path when available. | Should not be the only biological identifier. | `C:\data\BL011.csv` |
| `Source_Type` | Yes | `string` | Not nullable | `csv`; `xlsx` | Distinguishes raw file family. | Must match controlled values. | `csv` |
| `Worksheet` | No | `string` | Nullable | Free text | Preserves Excel sheet provenance. | Null for CSV rows is allowed. | `Sheet1` |
| `Data_Source` | No | `string` | Nullable or `unknown` | `24_hour_csv`; `12_hour_excel`; `unknown` | Describes source-duration family. | Must match controlled values when present. | `24_hour_csv` |
| `Time_Series_Duration_Hours` | No | `Float64` | Nullable | Non-negative numeric | Observed measured duration. | Derive from measured data; do not extrapolate. | `24.0` |
| `Analysis_Window` | No | `string` | Nullable, initially `unassigned` preferred | Controlled window labels | Records future analysis-window assignment. | Must be controlled; import should start as `unassigned`. | `unassigned` |
| `Import_Timestamp` | No | `datetime64[ns, UTC]` | Nullable | UTC timestamp | Captures import audit timing. | Must be UTC when present. | `2026-07-25T12:00:00Z` |
| `Source_Row_ID` | No | `Int64` | Nullable | Non-negative integer | Supports traceability and fallback identity. | Should preserve source row identity. | `42` |
| `Measurement_Unit_ID` | Yes | `string` | Not nullable for valid records | Free text | Identifies one experimental unit measured over time. | Must be deterministic; must not use luminescence, current time, or random UUIDs; synthetic IDs must be flagged. | `unit_r0002377__col007_luminescence` |
| `Strain_Original` | Yes | `string` | Not nullable | Free text | Preserves source strain label. | Must not be automatically standardized. | `BL027ab` |
| `Strain_Standardized` | No | `string` | Nullable | Approved mapping only | Stores future verified strain mapping. | Must not overwrite `Strain_Original`. | `BL027` |
| `Chemical_Name_Original` | Yes | `string` | Not nullable | Free text | Preserves source chemical label. | Preserve labels exactly, including `Lambda Cyclotherin`. | `Lambda Cyclotherin` |
| `Chemical_Name_Standardized` | No | `string` | Nullable | Approved mapping only | Stores future verified chemical mapping. | Remain null or unchanged until mapping approval. | null |
| `Concentration_Label` | Yes | `string` | Not nullable | Free text | Preserves source concentration/control label. | Must be preserved exactly. | `Control` |
| `Concentration_ug_mL` | No | `Float64` | Nullable | Non-negative numeric | Parsed numeric concentration in ug/mL. | Negative values fail validation; unknown units must be flagged. | `5.0` |
| `Control_Status` | No | `string` | Nullable or `unknown` | `treatment`; `control`; `unknown` | Indicates treatment/control status. | Must match controlled values when present. | `treatment` |
| `Control_Type` | No | `string` | Nullable | Controlled control labels | Distinguishes control classes. | Zero concentration is not automatically every control type. | `shared_control` |
| `Replicate_ID` | No | `string` | Nullable | Free text | Identifies replicate within source context when source evidence supports it. | Preserve explicit labels; generated labels must come from clear source structure and be flagged; not globally unique by itself. | `1` |
| `Replicate_Type` | No | `string` | Nullable | `technical`; `biological`; `unspecified` | Describes replicate class when known. | Must match controlled values when present. | `unspecified` |
| `Well_ID` | No | `string` | Nullable | Free text | Physical well identifier when available. | Do not invent missing well IDs. | `A01` |
| `Time_Original` | No | `string` | Nullable | Free text | Preserves source time label before conversion. | Must not be extrapolated. | `120` |
| `Time_Unit_Original` | No | `string` | Nullable | Free text | Preserves source time unit. | Unknown units must be flagged before conversion. | `min` |
| `Time_Minutes` | Yes | `Float64` | Not nullable | Non-negative numeric | Primary minute-scale time axis. | Must agree with `Time_Hours` within tolerance. | `120.0` |
| `Time_Hours` | Yes | `Float64` | Not nullable | Non-negative numeric | Hour-scale time axis for windows/reports. | Must agree with `Time_Minutes` within tolerance. | `2.0` |
| `Timepoint_Index` | No | `Int64` | Nullable | Integer | Within-curve time ordering. | Should increase within each independent curve. | `24` |
| `Luminescence_Raw` | Yes | `Float64` | Not nullable | Numeric, finite | Unmodified raw signal. | Infinite values fail; negative values are retained but warned. | `12345.0` |
| `Luminescence_Normalized` | No | `Float64` | Nullable | Numeric, finite | Derived normalized signal. | Remains null until normalization; never overwrites raw signal. | null |
| `Normalization_Method` | No | `string` | Nullable | Free text | Documents normalization method. | Required only once normalized values exist. | `baseline_ratio` |
| `QC_Status` | Yes | `string` | Not nullable | `pass`; `warning`; `fail`; `not_evaluated` | Quality-control status. | Must match controlled values. | `not_evaluated` |
| `QC_Flags` | No | `string` | Nullable | Free text or encoded flags | Stores non-fatal QC issues. | Must not silently delete rows. | `negative_luminescence` |
| `Record_Valid` | Yes | `boolean` | Not nullable | `True`; `False` | Downstream eligibility flag. | Must be boolean. | `True` |
| `Notes` | No | `string` | Nullable | Free text | Human-readable unresolved context. | Should not replace machine-readable QC flags. | `Source label pending verification` |

## Validation Rules

Schema validation reports errors, warnings, missing columns, unexpected columns, invalid controlled values, and row-level problem counts. Validation must not mutate the input DataFrame and must not delete rows.

Required values must be present for required fields in valid measurement records. Invalid retained rows may carry missing measurement fields when the source row is malformed; those missing values are counted and warned rather than silently deleted. Optional fields may be null. Unexpected columns are reported.

Controlled-value fields must match their enumerations when values are present.

## Record Identity Rules

The canonical measurement key is:

- `Experiment_ID`
- `Source_File`
- `Measurement_Unit_ID`
- `Time_Minutes`

The time-series grouping key is:

- `Experiment_ID`
- `Source_File`
- `Measurement_Unit_ID`

`Measurement_Unit_ID` should use the best source-specific evidence available: plate plus well, well, explicit sample/replicate structure, stable source measurement-column identity, then deterministic source-position-derived IDs. Replicate labels are preserved but are not assumed to be globally unique.

## Missing-Data Rules

Missing optional identifiers remain null. The schema must not convert missing optional identifiers to empty strings. Missing measurements are reported by validation and must not be silently deleted.

## Name-Preservation Rules

Original and standardized names are separate fields:

- `Strain_Original` may contain `BL027ab`.
- `Strain_Standardized` may later contain `BL027` only after an explicit approved mapping.
- `Chemical_Name_Original` must preserve `Lambda Cyclotherin` exactly when that is how it appears.
- `Chemical_Name_Standardized` remains null or unchanged until a verified mapping is applied.

The schema does not automatically correct `Lambda Cyclotherin`, `BL027ab`, `Monensin Sodium`, or chemical spelling variants.

## Time Rules

`Time_Minutes` and `Time_Hours` must be numerically consistent within the schema tolerance. Time values must not be negative unless a future explicit permission and QC flag are defined. `Timepoint_Index` should increase within each independent curve. Time must not be extrapolated. Duplicate time points are flagged, not removed. Duration should be derived from measured data where possible. `Analysis_Window` should initially be `unassigned` during import.

## Concentration Rules

`Concentration_Label` is preserved exactly. Parsed numeric concentration is stored separately in `Concentration_ug_mL`. Controls may have null numeric concentration. Zero concentration is not automatically identical to every control type. Negative concentrations fail validation. Units must ultimately be expressed as ug/mL; unknown units are flagged rather than converted without evidence.

## Luminescence Rules

`Luminescence_Raw` must be numeric and finite for valid measurement rows. Negative raw values are retained but warned. Missing measurements are reported and must not be silently deleted. Infinite values fail validation. `Luminescence_Normalized` remains null until normalization. Raw values must never be overwritten by normalized values.

## Example Canonical Rows

| Experiment_ID | Source_File | Measurement_Unit_ID | Source_Type | Data_Source | Strain_Original | Strain_Standardized | Chemical_Name_Original | Chemical_Name_Standardized | Concentration_Label | Concentration_ug_mL | Replicate_ID | Time_Minutes | Time_Hours | Luminescence_Raw | Luminescence_Normalized | QC_Status | Record_Valid |
|---|---|---|---|---|---|---|---|---|---|---:|---|---:|---:|---:|---|---|---|
| `EXP-001` | `BL027ab.csv` | `unit_r000001__col007_luminescence` | `csv` | `24_hour_csv` | `BL027ab` | null | `Trimethoprim` | null | `5` | `5.0` | `1` | `120.0` | `2.0` | `12345.0` | null | `not_evaluated` | `True` |
| `EXP-002` | `BL011.12hrs.xlsx` | `unit_r000001__col008_luminescence` | `xlsx` | `12_hour_excel` | `BL011` | null | `Lambda Cyclotherin` | null | `Control` | null | `1` | `0.0` | `0.0` | `7196.0` | null | `not_evaluated` | `True` |

## Known Unresolved Mappings

These labels are intentionally unresolved until source-owner approval:

- `BL027` versus `BL027ab`
- `Lambda Cyclotherin`
- `Monesin sodium` versus `Monensin`
- `Trimethoprim` versus `Trimetropin`
- `Boric Acid` versus `Boric acid`
- `N,N-Diethyl-m-Toluamide` versus `N,N-Diethyl-m-Toluamide (DEET)`
- `BL032` appearing as a chemical value in one inspected CSV row

## Future Importer Requirements

Future importers must:

- create all canonical columns in the documented order
- populate required fields without inventing missing identifiers
- assign deterministic `Measurement_Unit_ID` values from source-supported unit evidence
- preserve original labels exactly
- write standardized labels only after approved mapping rules exist
- keep raw and normalized luminescence separate
- set `Analysis_Window` to `unassigned` during initial import
- report invalid values and duplicate identity problems without deleting rows
- use `coerce_canonical_dtypes` and `validate_canonical_schema` before handing data to preprocessing or downstream analysis
