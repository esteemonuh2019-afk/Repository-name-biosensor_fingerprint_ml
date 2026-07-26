"""Command-line discovery of biosensor source files."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_ingestion.file_discovery import BiosensorFileRecord, discover_biosensor_files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", help="Folder to scan for supported biosensor source files.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional path for writing the discovery table as CSV.",
    )
    args = parser.parse_args(argv)

    try:
        result = discover_biosensor_files(args.folder)
    except (FileNotFoundError, NotADirectoryError) as error:
        print(f"Discovery failed: {error}", file=sys.stderr)
        return 1

    print(_format_table(result.files))

    if result.warnings:
        print()
        print("Warnings:")
        for warning in result.warnings:
            print(f"- {warning}")

    if args.output_csv is not None:
        _write_csv(args.output_csv, result.files)

    return 0


def _format_table(records: list[BiosensorFileRecord]) -> str:
    headers = ["filename", "source type", "inferred strain", "duration hint", "file size"]
    rows = [
        [
            record.filename,
            record.source_type,
            record.strain_label_from_filename or "",
            record.duration_hint_from_filename or "",
            str(record.file_size_bytes),
        ]
        for record in records
    ]
    table_rows = [headers, *rows]

    widths = [
        max(len(row[column_index]) for row in table_rows)
        for column_index in range(len(headers))
    ]

    formatted_rows = [
        "  ".join(
            value.ljust(widths[column_index])
            for column_index, value in enumerate(row)
        ).rstrip()
        for row in table_rows
    ]
    separator = "  ".join("-" * width for width in widths).rstrip()
    return "\n".join([formatted_rows[0], separator, *formatted_rows[1:]])


def _write_csv(output_path: Path, records: list[BiosensorFileRecord]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "absolute_path",
                "filename",
                "extension",
                "source_type",
                "file_size_bytes",
                "strain_label_from_filename",
                "duration_hint_from_filename",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.absolute_path,
                    record.filename,
                    record.extension,
                    record.source_type,
                    record.file_size_bytes,
                    record.strain_label_from_filename or "",
                    record.duration_hint_from_filename or "",
                ]
            )


if __name__ == "__main__":
    raise SystemExit(main())
