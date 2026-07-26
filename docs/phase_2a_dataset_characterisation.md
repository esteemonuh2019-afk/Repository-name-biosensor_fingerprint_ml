# Phase 2A Dataset Characterisation

Inspection date: 2026-07-25

Source folder inspected: `C:\Users\USER\Desktop\biosensor_phase2_source_files`

Scope: inspection only. No source Excel/CSV files, Python files, tests, configuration files, GUI files, or existing reports were modified. The biosensor pipeline was not run, no models were trained, and no importer code was written.

## Filename Check

The exact filenames named in the task were not present in the source folder. The 12 logical files were present with suffix-free names and were inspected read-only:

| Requested filename | Actual filename inspected | Requested file present | Actual file opened |
|---|---|---:|---:|
| `BL011.12hrs(2).xlsx` | `BL011.12hrs.xlsx` | No | Yes |
| `BL027.12hrs(2).xlsx` | `BL027.12hrs.xlsx` | No | Yes |
| `BL029.12hrs(2).xlsx` | `BL029.12hrs.xlsx` | No | Yes |
| `BL030.12hrs(2).xlsx` | `BL030.12hrs.xlsx` | No | Yes |
| `BL031.12hrs(2).xlsx` | `BL031.12hrs.xlsx` | No | Yes |
| `BL032.12hrs(2).xlsx` | `BL032.12hrs.xlsx` | No | Yes |
| `BL011(9).csv` | `BL011.csv` | No | Yes |
| `BL027ab(10).csv` | `BL027ab.csv` | No | Yes |
| `BL029(9).csv` | `BL029.csv` | No | Yes |
| `BL030(9).csv` | `BL030.csv` | No | Yes |
| `BL031(9).csv` | `BL031.csv` | No | Yes |
| `BL032(9).csv` | `BL032.csv` | No | Yes |

## Summary of All 12 Files

Row counts below are data rows, excluding the detected header row. Column counts report data columns first, with raw worksheet/file columns in parentheses where they differ.

| Actual file | Opens | Sheet names | Rows x cols | Strain name(s) | Replicates | Well IDs | Time | Luminescence area | Format |
|---|---:|---|---:|---|---|---|---|---|---|
| `BL011.12hrs.xlsx` | Yes | `Sheet1` | 69,600 x 7 (raw 69,603 x 8) | `BL011` | 4: `1`, `2`, `3`, `4` | No | `time_min`; 0 to 720 min; 145 points; 5 min interval | Column H / 8, `luminescence` | Long format |
| `BL027.12hrs.xlsx` | Yes | `Sheet1` | 69,600 x 7 (raw 69,603 x 8) | `BL027` | 4: `1`, `2`, `3`, `4` | No | `time_min`; 0 to 720 min; 145 points; 5 min interval | Column H / 8, `luminescence` | Long format |
| `BL029.12hrs.xlsx` | Yes | `Sheet1` | 69,600 x 7 (raw 69,603 x 8) | `BL029` | 4: `1`, `2`, `3`, `4` | No | `time_min`; 0 to 720 min; 145 points; 5 min interval | Column H / 8, `luminescence` | Long format |
| `BL030.12hrs.xlsx` | Yes | `Sheet1` | 69,600 x 7 (raw 69,603 x 8) | `BL030` | 4: `1`, `2`, `3`, `4` | No | `time_min`; 0 to 720 min; 145 points; 5 min interval | Column H / 8, `luminescence` | Long format |
| `BL031.12hrs.xlsx` | Yes | `Sheet1` | 69,600 x 7 (raw 69,603 x 8) | `BL031` | 4: `1`, `2`, `3`, `4` | No | `time_min`; 0 to 720 min; 145 points; 5 min interval | Column H / 8, `luminescence` | Long format |
| `BL032.12hrs.xlsx` | Yes | `Sheet1` | 69,600 x 7 (raw 69,603 x 8) | `BL032` | 4: `1`, `2`, `3`, `4` | No | `time_min`; 0 to 720 min; 145 points; 5 min interval | Column H / 8, `luminescence` | Long format |
| `BL011.csv` | Yes | N/A | 215,856 x 7 (raw 18 cols) | `BL011` | 4: `1`, `2`, `3`, `4` | No | `time_min`; 0 to 1440 min; 289 points; 5 min interval | Column 7, `luminescence` | Long format with trailing blank columns |
| `BL027ab.csv` | Yes | N/A | 223,200 x 7 (raw 16 cols) | `BL027` | 4: `1`, `2`, `3`, `4`; 4 rows missing replicate | No | `time_min`; 0 to 1440 min; 289 points; 5 min interval | Column 7, `luminescence` | Long format with trailing blank columns |
| `BL029.csv` | Yes | N/A | 215,856 x 7 | `BL029`, `BL29` | 4: `1`, `2`, `3`, `4` | No | `time_min`; 0 to 1440 min; 289 points; 5 min interval | Column 7, `luminescence` | Long format |
| `BL030.csv` | Yes | N/A | 246,096 x 7 | `BL030` | 4: `1`, `2`, `3`, `4`; 6 rows missing replicate | No | `time_min`; 0 to 1440 min; 289 points; 5 min interval | Column 7, `luminescence` | Long format with nonstandard first header |
| `BL031.csv` | Yes | N/A | 243,936 x 7 | `BL031` | 4: `1`, `2`, `3`, `4` | No | `time_min`; 0 to 1440 min; 289 points; 5 min interval | Column 7, `luminescence` | Long format with nonstandard first header |
| `BL032.csv` | Yes | N/A | 271,802 x 7 | `BL032`; 2 rows blank strain | 4: `1`, `2`, `3`, `4`; 1 row missing replicate | No | `time_min`; 0 to 1440 min; 289 points; 5 min interval | Column 7, `luminescence` | Long format with row-level metadata defects |

## Exact Chemical Names and Concentration Labels

| Actual file | Chemical names exactly as written | Concentration labels exactly as written | Control labels |
|---|---|---|---|
| `BL011.12hrs.xlsx` | `Bixafen`; `Flonicamid`; `Lambda Cyclotherin`; `Monesin sodium` | `0.05`; `0.5`; `5`; `50`; `500`; `Control` | `Control` in `concentration` |
| `BL027.12hrs.xlsx` | `Bixafen`; `Flonicamid`; `Lambda Cyclotherin`; `Monesin sodium` | `0.05`; `0.5`; `5`; `50`; `500`; `Control` | `Control` in `concentration` |
| `BL029.12hrs.xlsx` | `Bixafen`; `Flonicamid`; `Lambda Cyclotherin`; `Monesin sodium` | `0.05`; `0.5`; `5`; `50`; `500`; `Control` | `Control` in `concentration` |
| `BL030.12hrs.xlsx` | `Bixafen`; `Flonicamid`; `Lambda Cyclotherin`; `Monesin sodium` | `0.05`; `0.5`; `5`; `50`; `500`; `Control` | `Control` in `concentration` |
| `BL031.12hrs.xlsx` | `Bixafen`; `Flonicamid`; `Lambda Cyclotherin`; `Monesin sodium` | `0.05`; `0.5`; `5`; `50`; `500`; `Control` | `Control` in `concentration` |
| `BL032.12hrs.xlsx` | `Bixafen`; `Flonicamid`; `Lambda Cyclotherin`; `Monesin sodium` | `0.05`; `0.5`; `5`; `50`; `500`; `Control` | `Control` in `concentration` |
| `BL011.csv` | `Boric Acid`; `Chloramphenicol`; `Kanamycin`; `Metaldehyde`; `Monensin`; `N,N-Diethyl-m-Toluamide (DEET)`; `Novobiocin`; `O,O-Diethyl O-(2-Isopropyl-6-Methylpyrimidinyl) (Diazinon)`; `Polymyxin B`; `Propoxur`; `Tetracycline`; `Trimethoprim` | `0.005`; `0.05`; `0.5`; `5`; `50`; `500`; `Control` | `Control` in `concentration` |
| `BL027ab.csv` | `Boric Acid`; `Chloramphenicol`; `Erythromycin`; `Metaldehyde`; `N,N-Diethyl-m-Toluamide (DEET)`; `Novobiocin`; `O,O-Diethyl O-(2-Isopropyl-6-Methylpyrimidinyl) (Diazinon)`; `Polymyxin B`; `Propoxur`; `Rifampicin`; `Trimethoprim`; `Trimetropin` | `0.005`; `0.05`; `0.5`; `5`; `50`; `500`; `Control` | `Control` in `concentration` |
| `BL029.csv` | `Boric Acid`; `Chloramphenicol`; `Hygromycin B`; `Metaldehyde`; `N,N-Diethyl-m-Toluamide (DEET)`; `Novobiocin`; `O,O-Diethyl O-(2-Isopropyl-6-Methylpyrimidinyl) (Diazinon)`; `Polymyxin B`; `Propoxur`; `Rifampicin`; `Streptomycin`; `Tetracycline`; `Trimethoprim` | `0.005`; `0.05`; `0.5`; `5`; `50`; `500`; `Control` | `Control` in `concentration` |
| `BL030.csv` | `Boric Acid`; `Chloramphenicol`; `Erythromycin`; `Hygromycin B`; `Metaldehyde`; `N,N-Diethyl-m-Toluamide (DEET)`; `Novobiocin`; `O,O-Diethyl O-(2-Isopropyl-6-Methylpyrimidinyl) (Diazinon)`; `Polymyxin B`; `Propoxur`; `Rifampicin`; `Streptomycin`; `Tetracycline`; `Trimethoprim` | `0.005`; `0.05`; `0.5`; `5`; `50`; `500`; `Control` | `Control` in `concentration` |
| `BL031.csv` | `Boric acid`; `Chloramphenicol`; `Hygromycin B`; `Metaldehyde`; `N,N-Diethyl-m-Toluamide`; `Novobiocin`; `O,O-Diethyl O-(2-Isopropyl-6-Methylpyrimidinyl) (Diazinon)`; `Polymyxin B`; `Propoxur`; `Rifampicin`; `Streptomycin`; `Tetracycline`; `Trimethoprim`; `Trimetropin` | `0.005`; `0.05`; `0.5`; `5`; `50`; `500`; `Control` | `Control` in `concentration` |
| `BL032.csv` | `Ampicillin`; `BL032`; `Boric acid`; `Chloramphenicol`; `Erythromycin`; `Hygromycin B`; `Metaldehyde`; `Monensin`; `N,N-Diethyl-m-Toluamide`; `Novobiocin`; `O,O-Diethyl O-(2-Isopropyl-6-Methylpyrimidinyl) (Diazinon)`; `Polymyxin B`; `Propoxur`; `Rifampicin`; `Streptomycin`; `Tetracycline`; `Trimethoprim` | `0.005`; `0.05`; `0.5`; `5`; `50`; `500`; `Control` | `Control` in `concentration` |

## Excel vs CSV Structures

All inspected files are already row-wise long format, not plate-reader matrix format. The Excel files do not contain visible plate layouts or well IDs; they contain normalized rows.

The 12-hour Excel files are structurally uniform:

- One sheet named `Sheet1`.
- Header detected on worksheet row 3.
- Column A is blank; data fields are in columns B-H.
- Raw worksheet dimensions are 69,603 rows x 8 columns; detected data dimensions are 69,600 rows x 7 columns.
- Time uses `time_min` from 0 to 720 at 5-minute intervals, giving 145 global time points.
- Each file has 480 strain/chemical/concentration/experiment/replicate groups, each with 145 time points.

The 24-hour CSV files are less uniform:

- Header detected on row 1.
- Time uses `time_min` from 0 to 1440 at 5-minute intervals, giving 289 global time points.
- `BL011.csv` has 18 raw columns but only 7 named data columns, with one duplicated header fragment in columns 12-18.
- `BL027ab.csv` has 16 raw columns but only 7 named data columns, with one duplicated header fragment in columns 10-16.
- `BL030.csv` and `BL031.csv` use the strain value itself as the first header (`BL030` or `BL031`) instead of `bacteria_id`.
- `BL027ab.csv`, `BL030.csv`, and `BL032.csv` required cp1252 decoding; UTF-8 decoding failed because of non-UTF-8 bytes such as `0xA0`.
- Several CSVs have condition groups with fewer than the full 289 global time points. This may be intentional experimental coverage or a data export issue; it should not be assumed either way.

## Naming Inconsistencies

- All exact requested filenames are absent. Actual source filenames omit the `(2)`, `(9)`, and `(10)` suffixes.
- `BL027.12hrs.xlsx` is paired logically with `BL027ab.csv`, while the row-level strain value in `BL027ab.csv` is `BL027`. This `BL027` versus `BL027ab` difference must be resolved before importer development.
- `BL029.csv` contains both `BL029` and `BL29` as strain names. Counts observed: `BL029` = 208,920 rows; `BL29` = 6,936 rows.
- `BL030.csv` and `BL031.csv` place `BL030` and `BL031` in the first header cell rather than using the standard `bacteria_id` header.
- `BL032.csv` contains two blank strain rows, and one row has `BL032` in the `antibiotic` column.

## Chemical-Name Issues

The exact label `Lambda Cyclotherin` appears in all six 12-hour Excel files and has been preserved exactly. It is flagged for later verification.

Other exact chemical-name inconsistencies observed:

- `Monesin sodium` appears in the 12-hour Excel files, while `Monensin` appears in some 24-hour CSV files.
- `Boric Acid` and `Boric acid` both appear across CSV files.
- `N,N-Diethyl-m-Toluamide (DEET)` appears in some CSV files, while `N,N-Diethyl-m-Toluamide` appears in others.
- `Trimethoprim` and `Trimetropin` both appear across CSV files.
- `BL032` appears as an `antibiotic` value in `BL032.csv`, which is likely metadata drift but should not be corrected without source confirmation.
- The chemical set differs by strain/file; for example, `Ampicillin` appears in `BL032.csv`, `Kanamycin` appears in `BL011.csv`, and `Erythromycin` appears in `BL027ab.csv`, `BL030.csv`, and `BL032.csv`.

## Concentration Issues

- The 12-hour Excel files use `0.05`, `0.5`, `5`, `50`, `500`, and `Control`.
- The 24-hour CSV files use `0.005`, `0.05`, `0.5`, `5`, `50`, `500`, and `Control`.
- `Control` is stored in the `concentration` column, not as a separate control field.
- Concentration units are not present in the inspected files.
- No source file states whether `Control` should be treated as a concentration label, a control condition, or both.

## Time-Series Issues

The global time grids are regular:

- 12-hour Excel files: 0 to 720 minutes, 145 time points, 5-minute interval.
- 24-hour CSV files: 0 to 1440 minutes, 289 time points, 5-minute interval.

Condition-level coverage differs:

| File | Condition group time-count distribution |
|---|---|
| All six 12-hour Excel files | 480 groups x 145 time points |
| `BL011.csv` | 720 groups x 289; 198 groups x 36 |
| `BL027ab.csv` | 716 groups x 289; 4 groups x 288; 108 groups x 140; 4 groups x 1 |
| `BL029.csv` | 720 groups x 289; 147 groups x 48 |
| `BL030.csv` | 714 groups x 289; 6 groups x 288; 366 groups x 48; 49 groups x 95; 101 groups x 96; 6 groups x 1 |
| `BL031.csv` | 720 groups x 289; 216 groups x 142; 90 groups x 48 |
| `BL032.csv` | 719 groups x 289; 1 group x 288; 108 groups x 140; 100 groups x 96; 50 groups x 95; 356 groups x 48; 31 groups x 47; 216 groups x 36; 2 groups x 1 |

These differences need source-owner interpretation before importer development. The importer should not assume that every condition has the full global time grid.

## Missing or Inconsistent Metadata

- No inspected file contains well IDs.
- No inspected file contains concentration units.
- No inspected file contains luminescence units or instrument metadata.
- No inspected file contains explicit plate-reader well layout metadata.
- Control conditions are represented by the `Control` value in the `concentration` column.
- `BL027ab.csv`, `BL030.csv`, and `BL032.csv` require non-UTF-8-aware handling or an encoding decision.
- `BL027ab.csv` has 4 rows with missing `replicate`.
- `BL030.csv` has 6 rows with missing `replicate` and includes experiment value `6`, while most files use experiments `1` through `5`.
- `BL032.csv` has 2 blank-strain rows, 1 row missing `concentration`, `Experiment`, `replicate`, `time_min`, and `luminescence`, and includes experiment labels `1` through `10` plus one blank experiment value.
- `BL011.csv` and `BL027ab.csv` each contain a duplicated header fragment in later blank columns, indicating side-pasted or residual export content.

## Questions Before Importer Development

1. Should the importer use the actual source filenames found now, or should the source files be renamed to match the requested `(2)`, `(9)`, and `(10)` filenames?
2. Is `BL027ab.csv` a separate strain/sample from `BL027`, or is it the 24-hour file for strain `BL027`?
3. Should `BL29` in `BL029.csv` be treated as `BL029`, excluded, or preserved as a distinct strain label?
4. Is `Lambda Cyclotherin` the intended chemical name, or should it map to another canonical label later?
5. Is `Monesin sodium` intended to map to `Monensin`, or should both labels remain distinct?
6. Should `Trimetropin` be treated as a distinct chemical label or verified against `Trimethoprim`?
7. Should `Boric Acid` and `Boric acid` be canonicalized, or preserved exactly?
8. Should `N,N-Diethyl-m-Toluamide` and `N,N-Diethyl-m-Toluamide (DEET)` be canonicalized?
9. What concentration units apply to the numeric labels?
10. Should `Control` remain in the `concentration` field, or should it become a separate condition type?
11. Are partial condition time series in the CSV files expected experimental design, early truncation, or export defects?
12. Should non-UTF-8 CSVs be decoded as cp1252 in importer logic, or should the raw CSV files be re-exported as UTF-8?
13. Should rows with missing replicate/strain/time/luminescence metadata be excluded, repaired, or quarantined?
14. How should the duplicated header fragments in trailing CSV columns be handled?

## End-of-Phase 2A Report

1. Files inspected: 12 actual source files in `C:\Users\USER\Desktop\biosensor_phase2_source_files`.
2. Files that failed to open: none of the 12 actual files. The exact requested source filenames were absent, but suffix-free counterparts opened successfully.
3. Main structural differences: Excel files are uniform 12-hour long-format workbooks with one sheet and header on row 3; CSV files are 24-hour long-format files with row-1 headers, but several have encoding, header, trailing-column, strain-label, and row-level metadata inconsistencies.
4. Chemical-name issues: `Lambda Cyclotherin` preserved and flagged; additional exact-label issues include `Monesin sodium`/`Monensin`, `Trimethoprim`/`Trimetropin`, `Boric Acid`/`Boric acid`, DEET suffix differences, and `BL032` appearing as a chemical value.
5. Concentration issues: Excel files lack `0.005`; CSV files include `0.005`; `Control` appears in the `concentration` column; concentration units are missing.
6. Time-series issues: global time grids are regular at 5-minute intervals, but several CSV condition groups have partial time coverage.
7. Files created: `docs/phase_2a_dataset_characterisation.md`; `outputs/tables/phase_2a_file_inventory.csv`.
8. Files modified: only the two newly created Phase 2A output files.
9. Whether any source file changed: no source Excel or CSV file was modified.
10. Phase 2A result: inspection completed with importer blockers/questions documented; importer development should not proceed until naming, chemical labels, concentration semantics, encoding, and partial time-series handling are resolved.
