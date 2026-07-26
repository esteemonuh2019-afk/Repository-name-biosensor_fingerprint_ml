"""Validate Stage 6B core feature datasets without supervised feature selection."""

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
from src.feature_engine import extract_features
from src.feature_validation import validate_features, write_validation_outputs


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "feature_validation" / "stage_6c"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        feature_input = _load_feature_input(args)
    except (FileNotFoundError, NotADirectoryError, RuntimeError, ValueError) as error:
        print(f"Feature validation failed: {error}", file=sys.stderr)
        return 1

    result = validate_features(feature_input)
    output_paths: list[Path] = []
    if args.output_dir:
        try:
            output_paths = write_validation_outputs(
                result,
                args.output_dir,
                overwrite=args.overwrite,
            )
        except FileExistsError as error:
            print(f"Feature validation output failed: {error}", file=sys.stderr)
            return 1

    _print_summary(result, output_paths)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_folder",
        nargs="?",
        type=Path,
        help="Folder containing biosensor source files; canonical and feature datasets are built in memory.",
    )
    parser.add_argument(
        "--feature-file",
        type=Path,
        help="Existing feature_dataset.csv to validate instead of building from source.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for Stage 6C validation outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    return parser.parse_args(argv)


def _load_feature_input(args: argparse.Namespace):
    if bool(args.source_folder) == bool(args.feature_file):
        raise ValueError("Provide exactly one input: source_folder or --feature-file.")
    if args.feature_file:
        return pd.read_csv(args.feature_file)
    canonical_dataframe = _build_canonical_dataframe(args.source_folder)
    return extract_features(canonical_dataframe)


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
        except Exception as error:  # noqa: BLE001 - CLI should aggregate file failures.
            failed_files.append(f"{record.filename}: {type(error).__name__}: {error}")
    if failed_files:
        formatted = "\n".join(f"- {failure}" for failure in failed_files)
        raise RuntimeError(f"One or more source files failed to import:\n{formatted}")
    return build_canonical_dataset(read_results).dataframe


def _print_summary(result, output_paths: list[Path]) -> None:
    recommendations = result.feature_recommendations
    recommendation_counts = recommendations["recommendation"].value_counts().to_dict() if not recommendations.empty else {}
    nonfinite = result.infinite_value_summary
    highly_correlated = result.correlation_summary.get("highly_correlated_pairs", pd.DataFrame())

    print(f"total feature rows: {result.metadata['feature_rows']}")
    print(f"valid feature rows: {result.metadata['valid_feature_rows']}")
    print(f"feature columns assessed: {result.metadata['feature_columns_assessed']}")
    print("missing-value counts by feature:")
    _print_feature_count_table(result.missing_value_summary, "missing_count")
    print("non-finite counts by feature:")
    _print_feature_count_table(nonfinite, "nonfinite_count")
    print("constant features: " + _format_feature_list(result.constant_feature_summary))
    print("low-variance features: " + _format_feature_list(result.low_variance_feature_summary))
    print(f"highly correlated feature pairs: {len(highly_correlated)}")
    print("replicate consistency findings:")
    _print_replicate_summary(result.replicate_reproducibility_summary)
    print(f"features recommended for retention: {int(recommendation_counts.get('Retain', 0))}")
    print(
        "features recommended for caution: "
        f"{int(recommendation_counts.get('Retain with caution', 0))}"
    )
    print(f"features recommended for review: {int(recommendation_counts.get('Review', 0))}")
    print(f"features recommended for exclusion: {int(recommendation_counts.get('Exclude', 0))}")
    print("known upstream QC problems that affect interpretation:")
    if result.errors or result.warnings:
        for message in [*result.errors, *result.warnings]:
            print(f"- {message}")
    else:
        print("- none")
    if output_paths:
        print("output paths:")
        for path in output_paths:
            print(f"- {path}")


def _print_feature_count_table(dataframe: pd.DataFrame, column: str) -> None:
    if dataframe.empty or column not in dataframe.columns:
        print("- none")
        return
    for row in dataframe.loc[:, ["feature", column]].to_dict("records"):
        print(f"- {row['feature']}: {int(row[column])}")


def _format_feature_list(dataframe: pd.DataFrame) -> str:
    if dataframe.empty or "feature" not in dataframe.columns:
        return "None"
    values = sorted(dataframe["feature"].astype(str).unique().tolist())
    return "; ".join(values) if values else "None"


def _print_replicate_summary(dataframe: pd.DataFrame) -> None:
    if dataframe.empty:
        print("- none")
        return
    for row in dataframe.to_dict("records"):
        print(
            "- {feature}: stable={stable}, acceptable={acceptable}, "
            "unstable={unstable}, insufficient={insufficient}".format(
                feature=row["feature"],
                stable=int(row["stable_group_count"]),
                acceptable=int(row["acceptable_group_count"]),
                unstable=int(row["unstable_group_count"]),
                insufficient=int(row["insufficient_data_group_count"]),
            )
        )


if __name__ == "__main__":
    raise SystemExit(main())
