"""Build a canonical biosensor dataset from discovered source files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_ingestion.canonical_builder import (
    build_canonical_dataset,
    save_canonical_dataset,
)
from src.data_ingestion.csv_reader import read_biosensor_csv
from src.data_ingestion.excel_reader import read_biosensor_excel
from src.data_ingestion.file_discovery import discover_biosensor_files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_folder", help="Folder containing biosensor source files.")
    parser.add_argument("--output", type=Path, help="Optional canonical CSV output path.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting --output.")
    args = parser.parse_args(argv)

    source_folder = Path(args.source_folder).expanduser()
    try:
        discovery = discover_biosensor_files(source_folder)
    except (FileNotFoundError, NotADirectoryError) as error:
        print(f"Discovery failed: {error}", file=sys.stderr)
        return 1

    reader_results = []
    imported_files = []
    failed_files = []

    for record in discovery.files:
        try:
            if record.extension == ".csv":
                reader_results.append(read_biosensor_csv(record.absolute_path))
            elif record.extension == ".xlsx":
                reader_results.append(read_biosensor_excel(record.absolute_path))
            else:
                continue
            imported_files.append(record.filename)
        except Exception as error:  # noqa: BLE001 - CLI should report all import failures.
            failed_files.append(f"{record.filename}: {type(error).__name__}: {error}")

    build_result = build_canonical_dataset(reader_results)
    dataframe = build_result.dataframe

    print(f"source files discovered: {len(discovery.files)}")
    print(f"source files successfully imported: {len(imported_files)}")
    print(f"source files failed: {len(failed_files)}")
    print(f"rows generated: {build_result.row_count}")
    print(f"valid rows: {build_result.valid_record_count}")
    print(f"warning rows: {build_result.warning_record_count}")
    print(f"invalid rows: {build_result.invalid_record_count}")
    print(f"measurement units: {dataframe['Measurement_Unit_ID'].dropna().nunique()}")
    print("strains detected: " + _format_values(dataframe["Strain_Original"].dropna().unique()))
    print("chemicals detected: " + _format_values(dataframe["Chemical_Name_Original"].dropna().unique()))
    print("time range by source type:")
    print(_time_range_summary(dataframe))
    print(f"schema validation valid: {build_result.schema_valid}")
    print(f"schema errors: {len(build_result.errors)}")
    print(f"schema/build warnings: {len(build_result.warnings)}")

    if discovery.warnings:
        print("discovery warnings:")
        for warning in discovery.warnings:
            print(f"- {warning}")
    if failed_files:
        print("failed files:")
        for failure in failed_files:
            print(f"- {failure}")
    if build_result.errors:
        print("schema errors:")
        for error in build_result.errors:
            print(f"- {error}")
    if build_result.warnings:
        print("warnings:")
        for warning in build_result.warnings[:20]:
            print(f"- {warning}")
        if len(build_result.warnings) > 20:
            print(f"- ... {len(build_result.warnings) - 20} additional warnings")

    if args.output is not None:
        try:
            saved_path = save_canonical_dataset(
                build_result,
                args.output,
                overwrite=args.overwrite,
                source_folder=source_folder,
            )
        except (FileExistsError, ValueError) as error:
            print(f"Save failed: {error}", file=sys.stderr)
            return 1
        print(f"saved canonical dataset: {saved_path}")

    return 0 if not failed_files else 1


def _format_values(values) -> str:
    formatted = sorted(str(value) for value in values)
    return f"{len(formatted)} ({'; '.join(formatted)})"


def _time_range_summary(dataframe) -> str:
    if dataframe.empty:
        return "  none"

    lines = []
    for source_type, group in dataframe.groupby("Source_Type", dropna=False):
        times = group["Time_Minutes"].dropna()
        if times.empty:
            lines.append(f"  {source_type}: unavailable")
            continue
        lines.append(
            f"  {source_type}: {float(times.min())} to {float(times.max())} minutes"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
