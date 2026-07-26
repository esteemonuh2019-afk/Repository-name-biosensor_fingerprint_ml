"""Build a Stage 6B feature dataset from canonical-builder source inputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_ingestion.canonical_builder import build_canonical_dataset
from src.data_ingestion.csv_reader import read_biosensor_csv
from src.data_ingestion.excel_reader import read_biosensor_excel
from src.data_ingestion.file_discovery import discover_biosensor_files
from src.data_schema.canonical_schema import SERIES_GROUPING_KEY_COLUMNS
from src.feature_engine import extract_features


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "features"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    source_folder = Path(args.source_folder).expanduser()

    try:
        canonical_dataframe = _build_canonical_dataframe(source_folder)
    except (FileNotFoundError, NotADirectoryError, RuntimeError) as error:
        print(f"Feature dataset build failed: {error}", file=sys.stderr)
        return 1

    feature_dataset = extract_features(canonical_dataframe)
    try:
        output_paths = feature_dataset.write_outputs(
            args.output_dir,
            overwrite=not args.no_overwrite,
        )
    except FileExistsError as error:
        print(f"Feature output save failed: {error}", file=sys.stderr)
        return 1

    _print_summary(canonical_dataframe, feature_dataset, output_paths)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_folder",
        type=Path,
        help="Folder containing biosensor CSV and Excel source files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for feature_dataset.csv, feature_summary.json, and feature_qc_report.md.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Refuse to replace existing feature output files.",
    )
    return parser.parse_args(argv)


def _build_canonical_dataframe(source_folder: Path) -> pd.DataFrame:
    discovery = discover_biosensor_files(source_folder)
    read_results = []
    failed_files: list[str] = []

    for record in discovery.files:
        try:
            if record.extension == ".csv":
                read_results.append(read_biosensor_csv(record.absolute_path))
            elif record.extension == ".xlsx":
                read_results.append(read_biosensor_excel(record.absolute_path))
        except Exception as error:  # noqa: BLE001 - CLI should report all file failures.
            failed_files.append(f"{record.filename}: {type(error).__name__}: {error}")

    if failed_files:
        formatted = "\n".join(f"- {failure}" for failure in failed_files)
        raise RuntimeError(f"One or more source files failed to import:\n{formatted}")

    return build_canonical_dataset(read_results).dataframe


def _print_summary(canonical_dataframe: pd.DataFrame, feature_dataset, output_paths: list[Path]) -> None:
    dataframe = feature_dataset.dataframe
    qc_summary = feature_dataset.qc.summary
    status_counts = dataframe["QC_Status"].value_counts().to_dict() if not dataframe.empty else {}

    print(f"canonical rows: {len(canonical_dataframe)}")
    print(f"measurement series: {_measurement_series_count(canonical_dataframe)}")
    print(f"feature rows: {len(dataframe)}")
    print(f"valid feature rows: {int(status_counts.get('pass', 0))}")
    print(f"warning rows: {int(status_counts.get('warning', 0))}")
    print(f"failed rows: {int(status_counts.get('fail', 0))}")
    print(f"zero-baseline rows: {qc_summary['zero_baseline_count']}")
    print(f"duplicate unit IDs: {qc_summary['duplicated_measurement_unit_count']}")
    print("output paths:")
    for path in output_paths:
        print(f"- {path}")

    if feature_dataset.qc.warnings:
        print("feature QC warnings:")
        for warning in feature_dataset.qc.warnings:
            print(f"- {warning}")
    if feature_dataset.qc.errors:
        print("feature QC errors:")
        for error in feature_dataset.qc.errors:
            print(f"- {error}")


def _measurement_series_count(canonical_dataframe: pd.DataFrame) -> int:
    if canonical_dataframe.empty:
        return 0
    return int(
        canonical_dataframe.groupby(
            list(SERIES_GROUPING_KEY_COLUMNS),
            dropna=False,
            sort=False,
        ).ngroups
    )


if __name__ == "__main__":
    raise SystemExit(main())
