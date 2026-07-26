"""Build Stage 7A fingerprint datasets from validated core features."""

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
from src.feature_validation import validate_features
from src.fingerprint import build_fingerprint_dataset


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "fingerprints"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        _ensure_output_dir_allowed(args.output_dir, overwrite=args.overwrite)
        feature_input = _load_feature_input(args)
        validation_result = validate_features(feature_input)
        fingerprint_dataset = build_fingerprint_dataset(
            validation_result,
            normalization=args.normalization,
        )
        output_paths = fingerprint_dataset.write_outputs(
            args.output_dir,
            overwrite=args.overwrite,
            write_distances=True,
            distance_chunk_size=args.distance_chunk_size,
        )
    except (FileExistsError, FileNotFoundError, NotADirectoryError, RuntimeError, TypeError, ValueError) as error:
        print(f"Fingerprint build failed: {error}", file=sys.stderr)
        return 1

    _print_summary(fingerprint_dataset, output_paths)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_folder",
        nargs="?",
        type=Path,
        help="Folder containing biosensor source files; canonical/features are built before validation.",
    )
    parser.add_argument(
        "--feature-file",
        type=Path,
        help="Existing Stage 6B feature_dataset.csv to validate and convert to fingerprints.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for fingerprint outputs.",
    )
    parser.add_argument(
        "--normalization",
        choices=["none", "zscore", "minmax", "robust"],
        default="zscore",
        help="Normalisation method for fingerprint_dataset_normalized.csv and distance matrices.",
    )
    parser.add_argument(
        "--distance-chunk-size",
        type=int,
        default=128,
        help="Rows per chunk when writing large distance matrices.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing files in the output directory.",
    )
    return parser.parse_args(argv)


def _ensure_output_dir_allowed(output_dir: Path, *, overwrite: bool) -> None:
    if not output_dir.exists() or overwrite:
        return
    if any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")


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
        except Exception as error:  # noqa: BLE001 - CLI should report all import failures.
            failed_files.append(f"{record.filename}: {type(error).__name__}: {error}")
    if failed_files:
        formatted = "\n".join(f"- {failure}" for failure in failed_files)
        raise RuntimeError(f"One or more source files failed to import:\n{formatted}")
    return build_canonical_dataset(read_results).dataframe


def _print_summary(fingerprint_dataset, output_paths: list[Path]) -> None:
    summary = fingerprint_dataset.summary
    qc = fingerprint_dataset.qc.summary
    normalization = fingerprint_dataset.normalization_parameters

    print(f"feature rows: {summary['feature_rows']}")
    print(f"fingerprint rows: {summary['fingerprint_rows']}")
    print(f"excluded rows: {summary['excluded_rows']}")
    print(f"duplicate fingerprints: {qc['duplicate_fingerprint_row_count']}")
    print(f"duplicated Measurement_Unit_ID rows: {qc['duplicated_measurement_unit_row_count']}")
    print("QC warnings:")
    if fingerprint_dataset.warnings:
        for warning in fingerprint_dataset.warnings:
            print(f"- {warning}")
    else:
        print("- none")
    print("normalisation summary:")
    print(f"- method: {normalization['method']}")
    print(f"- zero-scale features: {len(normalization.get('zero_scale_features', []))}")
    print(
        "distance matrix dimensions: "
        f"{summary['distance_matrix_rows']} x {summary['distance_matrix_columns']}"
    )
    print("distance metrics: euclidean, cosine, manhattan, correlation")
    print("output paths:")
    for path in output_paths:
        print(f"- {path}")


if __name__ == "__main__":
    raise SystemExit(main())
