# Stage 3.1 File Discovery

## Purpose

Stage 3.1 adds a read-only discovery layer for biosensor source files. It scans a user-provided folder and reports supported source-file candidates without opening CSV or Excel measurement contents.

This stage does not parse CSV rows, parse workbook sheets, build a canonical dataset, modify the GUI, run the analysis pipeline, or train models.

## Supported Extensions

The discovery component considers only these extensions:

- `.csv`
- `.xlsx`

Extension matching is case-insensitive. Directories, hidden files, temporary Excel files beginning with `~$`, and unrelated file types are ignored.

## Discovery Rules

Use `discover_biosensor_files(folder_path)` from `src.data_ingestion.file_discovery`.

The function:

- verifies that the supplied path exists
- rejects paths that are not directories
- scans only immediate files in the folder
- records absolute path, filename, extension, source type, file size, inferred strain label, and inferred duration hint
- returns records in deterministic filename-sorted order
- returns warnings for non-fatal discovery issues

The source-type labels are candidates, not confirmed content types:

- `csv_24h_candidate`
- `excel_12h_candidate`
- `unknown_supported_file`

Candidate classification is based on filename and extension evidence only. A CSV file is not treated as confirmed 24-hour data, and an Excel workbook is not treated as confirmed 12-hour data until later content-aware stages.

## Strain Inference Rules

The filename is searched for biosensor-style strain labels such as:

- `BL011`
- `BL027`
- `BL027ab`
- `BL029`
- `BL030`
- `BL031`
- `BL032`

The exact matched label is preserved. For example, `BL027ab` remains `BL027ab` and is not silently converted to `BL027`.

If no strain-like label is detected, discovery still returns the file and adds a warning.

## Duration-Hint Rules

The filename is searched for duration hints such as:

- `12hrs`
- `12hr`
- `12h`
- `24hrs`
- `24hr`
- `24h`

The detected hint is recorded from the filename. Suffixes such as `(2)`, `(9)`, and `(10)` do not affect strain or duration detection.

## Command-Line Example

```powershell
python scripts/discover_biosensor_files.py "C:\Users\USER\Desktop\biosensor_phase2_source_files"
```

The script prints a readable table containing filename, source type, inferred strain, duration hint, and file size. It writes no output file unless `--output-csv` is provided explicitly.

## Warnings

Warnings are returned when:

- no supported files are found
- a supported file has no detectable strain label
- duplicate strain/source-type candidates are present

Warnings are non-fatal. They are intended to guide later importer decisions without changing source data.

## Limitations

This stage does not inspect experimental contents. It does not validate sheet names, row counts, chemical names, concentration labels, time columns, luminescence columns, or well identifiers. Those checks belong to later content-aware importer stages.
