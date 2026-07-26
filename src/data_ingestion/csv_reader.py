"""Read-only reader for raw 24-hour biosensor CSV candidate files."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data_ingestion.file_discovery import (
    _classify_source_type,
    _infer_duration_hint,
    _infer_strain_label,
)


ENCODINGS_TO_TRY: tuple[str, ...] = ("utf-8", "utf-8-sig", "cp1252")
DELIMITERS_TO_TRY: tuple[str, ...] = (",", ";", "\t", "|")
EXPECTED_MEASUREMENT_COLUMNS: tuple[str, ...] = (
    "antibiotic",
    "concentration",
    "Experiment",
    "replicate",
    "time_min",
    "luminescence",
)


@dataclass(frozen=True)
class CsvReadResult:
    """Raw CSV data plus read-time metadata and non-fatal warnings."""

    source_file: str
    absolute_path: str
    source_type: str
    encoding: str
    delimiter: str
    strain_label_original: str | None
    row_count: int
    column_count: int
    original_columns: list[str]
    dataframe: pd.DataFrame
    warnings: list[str]


def read_biosensor_csv(file_path: str | Path) -> CsvReadResult:
    """Read one biosensor CSV candidate without modifying or cleaning source data."""

    path = Path(file_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(str(path))
    if not path.is_file():
        raise IsADirectoryError(str(path))
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv file: {path}")

    raw_bytes = path.read_bytes()
    encoding, text = _decode_csv_bytes(raw_bytes, path)
    if not text.strip():
        raise ValueError(f"CSV file is empty: {path}")

    delimiter = _detect_delimiter(text)
    header, rows, malformed_row_numbers = _parse_csv_text(text, delimiter, path)
    columns = _columns_for_dataframe(header, rows)
    normalized_rows = [_normalize_row_length(row, len(columns)) for row in rows]
    dataframe = pd.DataFrame(normalized_rows, columns=columns)

    strain_label = _infer_strain_label(path.name)
    duration_hint = _infer_duration_hint(path.name)
    source_type = _classify_source_type(path.suffix.lower(), duration_hint)
    warnings = _build_warnings(
        columns=columns,
        rows=normalized_rows,
        malformed_row_numbers=malformed_row_numbers,
        strain_label=strain_label,
    )

    return CsvReadResult(
        source_file=path.name,
        absolute_path=str(path.resolve()),
        source_type=source_type,
        encoding=encoding,
        delimiter=delimiter,
        strain_label_original=strain_label,
        row_count=len(normalized_rows),
        column_count=len(columns),
        original_columns=columns,
        dataframe=dataframe,
        warnings=warnings,
    )


def _decode_csv_bytes(raw_bytes: bytes, path: Path) -> tuple[str, str]:
    if not raw_bytes:
        raise ValueError(f"CSV file is empty: {path}")

    failures: list[str] = []
    for encoding in ENCODINGS_TO_TRY:
        try:
            text = raw_bytes.decode(encoding)
        except UnicodeDecodeError as error:
            failures.append(f"{encoding}: {error}")
            continue

        if encoding == "utf-8" and text.startswith("\ufeff"):
            continue
        return encoding, text

    joined_failures = "; ".join(failures)
    raise UnicodeError(
        "Could not decode CSV using supported encodings "
        f"{ENCODINGS_TO_TRY}: {path}. {joined_failures}"
    )


def _detect_delimiter(text: str) -> str:
    sample = "\n".join(text.splitlines()[:20])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="".join(DELIMITERS_TO_TRY))
        return dialect.delimiter
    except csv.Error:
        first_line = text.splitlines()[0] if text.splitlines() else ""
        delimiter_counts = {
            delimiter: first_line.count(delimiter)
            for delimiter in DELIMITERS_TO_TRY
        }
        best_delimiter, best_count = max(
            delimiter_counts.items(),
            key=lambda item: item[1],
        )
        return best_delimiter if best_count else ","


def _parse_csv_text(
    text: str,
    delimiter: str,
    path: Path,
) -> tuple[list[str], list[list[str]], list[int]]:
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)

    try:
        header = next(reader)
    except StopIteration as error:
        raise ValueError(f"CSV file is empty: {path}") from error

    if not header and not text.strip():
        raise ValueError(f"CSV file is empty: {path}")

    rows: list[list[str]] = []
    malformed_row_numbers: list[int] = []
    expected_length = len(header)

    for row_number, row in enumerate(reader, start=2):
        if not row:
            continue
        if len(row) != expected_length:
            malformed_row_numbers.append(row_number)
        rows.append(row)

    return header, rows, malformed_row_numbers


def _columns_for_dataframe(header: list[str], rows: list[list[str]]) -> list[str]:
    max_width = max([len(header), *(len(row) for row in rows)] or [len(header)])
    columns = list(header)

    for index in range(len(header), max_width):
        columns.append(f"__extra_column_{index - len(header) + 1}")

    return columns


def _normalize_row_length(row: list[str], width: int) -> list[str]:
    if len(row) < width:
        return [*row, *([""] * (width - len(row)))]
    return row[:width]


def _build_warnings(
    columns: list[str],
    rows: list[list[str]],
    malformed_row_numbers: list[int],
    strain_label: str | None,
) -> list[str]:
    warnings: list[str] = []

    duplicate_column_warnings = _duplicate_column_warnings(columns)
    warnings.extend(duplicate_column_warnings)

    empty_columns = _empty_column_descriptions(columns, rows)
    if empty_columns:
        warnings.append(f"Completely empty columns detected: {', '.join(empty_columns)}")

    if malformed_row_numbers:
        first_examples = ", ".join(str(row_number) for row_number in malformed_row_numbers[:5])
        warnings.append(
            "Malformed row lengths detected: "
            f"{len(malformed_row_numbers)} rows differ from the header width "
            f"(first row numbers: {first_examples})"
        )

    missing_columns = [
        column for column in EXPECTED_MEASUREMENT_COLUMNS if column not in columns
    ]
    has_strain_column = "bacteria_id" in columns or (
        strain_label is not None and columns and columns[0] == strain_label
    )
    if not has_strain_column:
        missing_columns.insert(0, "bacteria_id")

    if missing_columns:
        warnings.append(
            "Missing expected measurement structure columns: "
            + ", ".join(missing_columns)
        )

    if strain_label is not None and columns and columns[0] == strain_label:
        warnings.append(
            "First column header is the inferred strain label "
            f"{strain_label!r} rather than 'bacteria_id'."
        )

    return warnings


def _duplicate_column_warnings(columns: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    for column in columns:
        counts[column] = counts.get(column, 0) + 1

    warnings = []
    for column, count in sorted(counts.items(), key=lambda item: (item[0], item[1])):
        if count > 1:
            label = column if column else "<empty>"
            warnings.append(f"Duplicate column name detected: {label} ({count} occurrences)")
    return warnings


def _empty_column_descriptions(columns: list[str], rows: list[list[str]]) -> list[str]:
    empty_columns = []
    for index, column in enumerate(columns):
        values = [row[index] for row in rows if index < len(row)]
        if column == "" and all(value == "" for value in values):
            empty_columns.append(f"{index + 1} (<empty>)")
        elif column and all(value == "" for value in values):
            empty_columns.append(f"{index + 1} ({column})")
    return empty_columns
