"""Inspect raw biosensor CSV files without modifying them."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError


DEFAULT_RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
CSV_ENCODINGS = ("utf-8", "utf-8-sig", "latin1")


def inspect_csv(file_path: str | Path) -> dict[str, Any]:
    """Return structural and missing-value details for one CSV file."""

    path = Path(file_path)
    dataframe = _read_csv_for_inspection(path)
    return {
        "filename": path.name,
        "row_count": len(dataframe),
        "column_count": len(dataframe.columns),
        "column_names": list(dataframe.columns),
        "missing_value_count_per_column": dataframe.isna().sum().to_dict(),
        "detected_data_types": {
            column: str(dtype) for column, dtype in dataframe.dtypes.items()
        },
    }


def inspect_folder(folder_path: str | Path) -> list[dict[str, Any]]:
    """Inspect every CSV file in a folder and print a readable summary."""

    folder = Path(folder_path)
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {folder}")
        return []

    summaries = [inspect_csv(csv_file) for csv_file in csv_files]
    for summary in summaries:
        _print_summary(summary)
    return summaries


def _read_csv_for_inspection(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(str(file_path))

    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(file_path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as error:
            last_error = error
        except EmptyDataError as error:
            raise ValueError(f"CSV file is empty: {file_path}") from error

    raise ValueError(f"Could not decode CSV file: {file_path}") from last_error


def _print_summary(summary: dict[str, Any]) -> None:
    print(f"\nFile: {summary['filename']}")
    print(f"Rows: {summary['row_count']}")
    print(f"Columns: {summary['column_count']}")
    print(f"Column names: {summary['column_names']}")
    print("Detected data types:")
    for column, dtype in summary["detected_data_types"].items():
        print(f"  - {column}: {dtype}")
    print("Missing values:")
    for column, missing_count in summary["missing_value_count_per_column"].items():
        print(f"  - {column}: {missing_count}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect raw biosensor CSV files without modifying them."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=DEFAULT_RAW_DATA_DIR,
        help="Folder containing raw CSV files. Defaults to data/raw.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    inspect_folder(args.folder)
