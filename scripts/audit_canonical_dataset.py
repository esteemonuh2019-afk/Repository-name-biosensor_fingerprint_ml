"""Run a read-only QC audit against canonical biosensor data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_ingestion.canonical_builder import build_canonical_dataset
from src.data_ingestion.csv_reader import read_biosensor_csv
from src.data_ingestion.excel_reader import read_biosensor_excel
from src.data_ingestion.file_discovery import discover_biosensor_files
from src.quality_control.canonical_qc import (
    audit_canonical_dataframe,
    write_qc_outputs,
)


def main() -> int:
    args = _parse_args()
    if bool(args.canonical_file) == bool(args.source_folder):
        print(
            "Provide exactly one input: a source folder or --canonical-file.",
            file=sys.stderr,
        )
        return 2

    if args.canonical_file:
        dataframe = pd.read_csv(args.canonical_file)
    else:
        dataframe = _build_canonical_from_source(args.source_folder)

    result = audit_canonical_dataframe(dataframe)
    if args.output_dir:
        created = write_qc_outputs(result, args.output_dir, overwrite=args.overwrite)
        print(f"qc output files: {len(created)}")

    _print_summary(result)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a canonical biosensor dataset without modifying inputs."
    )
    parser.add_argument(
        "source_folder",
        nargs="?",
        type=Path,
        help="Folder containing source biosensor CSV and Excel files.",
    )
    parser.add_argument(
        "--canonical-file",
        type=Path,
        help="Existing canonical CSV file to audit instead of reading source files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional new directory for QC report tables.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing QC output directory.",
    )
    return parser.parse_args()


def _build_canonical_from_source(source_folder: Path) -> pd.DataFrame:
    discovery = discover_biosensor_files(source_folder)
    read_results = []
    for record in discovery.files:
        if record.extension == ".csv":
            read_results.append(read_biosensor_csv(record.absolute_path))
        elif record.extension == ".xlsx":
            read_results.append(read_biosensor_excel(record.absolute_path))
    build_result = build_canonical_dataset(read_results)
    return build_result.dataframe


def _print_summary(result) -> None:
    print(f"total canonical rows: {result.row_count}")
    print(f"source files: {len(result.source_files)}")
    print(f"measurement units: {result.measurement_unit_count}")
    print(f"synthetic measurement units: {result.synthetic_measurement_unit_count}")
    print(f"unresolved measurement unit rows: {result.unresolved_measurement_unit_count}")
    print(f"exact duplicate rows: {result.exact_duplicate_count}")
    print(f"source_row_id duplicate rows: {result.source_row_id_duplicate_count}")
    print(f"corrected measurement-key duplicate rows: {result.logical_duplicate_count}")
    print(f"corrected measurement-key duplicate groups: {result.duplicate_group_count}")
    print(f"legacy logical duplicate rows: {result.legacy_logical_duplicate_count}")
    print(f"legacy logical duplicate groups: {result.legacy_duplicate_group_count}")
    print(
        "source-aware legacy logical duplicate rows: "
        f"{result.source_aware_logical_duplicate_count}"
    )
    print(
        "source-aware legacy logical duplicate groups: "
        f"{result.source_aware_duplicate_group_count}"
    )
    print(f"identical-value duplicate rows: {result.identical_value_duplicate_count}")
    print(f"conflicting-value duplicate rows: {result.conflicting_value_duplicate_count}")
    print(f"ambiguous measurement identity rows: {result.ambiguous_measurement_identity_count}")
    print(f"ambiguous measurement identity groups: {result.ambiguous_measurement_identity_group_count}")
    print(f"separate replicate measurement rows: {result.separate_replicate_measurement_count}")
    print(f"separate replicate measurement groups: {result.separate_replicate_measurement_group_count}")
    print(f"ambiguous replicate rows: {result.ambiguous_replicate_count}")
    print(f"ambiguous replicate groups: {result.ambiguous_replicate_group_count}")
    print(f"negative luminescence rows: {result.negative_luminescence_count}")
    print(f"infinite luminescence rows: {result.infinite_luminescence_count}")
    print(f"duplicate timepoint groups: {result.duplicate_timepoint_group_count}")
    print(f"non-monotonic time groups: {result.non_monotonic_time_group_count}")
    print(f"qc passed: {result.qc_passed}")
    print("missing identifiers:")
    for key, value in result.missing_identifier_counts.items():
        print(f"  {key}: {value}")
    print("warnings:")
    if result.warnings:
        for warning in result.warnings:
            print(f"  - {warning}")
    else:
        print("  - none")
    print("errors:")
    if result.errors:
        for error in result.errors:
            print(f"  - {error}")
    else:
        print("  - none")


if __name__ == "__main__":
    raise SystemExit(main())
