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
from src.fingerprint import (
    DEFAULT_MAX_INDIVIDUAL_DISTANCE_ROWS,
    build_fingerprint_dataset,
)


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
            consensus_group_columns=_parse_consensus_group_columns(args.consensus_group_columns),
        )
        _print_distance_estimates(
            fingerprint_dataset,
            distance_mode=args.distance_mode,
            max_individual_distance_rows=args.max_individual_distance_rows,
            allow_large_distance_matrix=args.allow_large_distance_matrix,
        )
        output_paths = fingerprint_dataset.write_outputs(
            args.output_dir,
            overwrite=args.overwrite,
            distance_mode=args.distance_mode,
            max_individual_distance_rows=args.max_individual_distance_rows,
            allow_large_distance_matrix=args.allow_large_distance_matrix,
            distance_chunk_size=args.distance_chunk_size,
        )
    except (FileExistsError, FileNotFoundError, NotADirectoryError, RuntimeError, TypeError, ValueError) as error:
        print(f"Fingerprint build failed: {error}", file=sys.stderr)
        return 1

    _print_summary(fingerprint_dataset, output_paths, distance_mode=args.distance_mode)
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
        "--distance-mode",
        choices=["none", "consensus", "individual"],
        default="consensus",
        help="Distance output mode. Consensus is the safe default.",
    )
    parser.add_argument(
        "--max-individual-distance-rows",
        type=int,
        default=DEFAULT_MAX_INDIVIDUAL_DISTANCE_ROWS,
        help="Maximum fingerprint rows allowed for individual-level distance matrices.",
    )
    parser.add_argument(
        "--allow-large-distance-matrix",
        action="store_true",
        help="Explicitly allow individual distance matrices above the row threshold.",
    )
    parser.add_argument(
        "--consensus-group-columns",
        default="Strain,Chemical,Concentration",
        help="Comma-separated columns for consensus grouping.",
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


def _parse_consensus_group_columns(value: str) -> list[str]:
    columns = [part.strip() for part in str(value).split(",") if part.strip()]
    if not columns:
        raise ValueError("At least one consensus grouping column is required.")
    return columns


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


def _print_summary(fingerprint_dataset, output_paths: list[Path], *, distance_mode: str) -> None:
    summary = fingerprint_dataset.summary
    qc = fingerprint_dataset.qc.summary
    normalization = fingerprint_dataset.normalization_parameters

    print(f"feature rows: {summary['feature_rows']}")
    print(f"fingerprint rows: {summary['fingerprint_rows']}")
    print(f"consensus fingerprint rows: {summary['consensus_fingerprint_rows']}")
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
    print(f"distance mode: {distance_mode}")
    print("distance metrics: euclidean, cosine, manhattan, correlation")
    print("output paths:")
    for path in output_paths:
        print(f"- {path}")


def _print_distance_estimates(
    fingerprint_dataset,
    *,
    distance_mode: str,
    max_individual_distance_rows: int,
    allow_large_distance_matrix: bool,
) -> None:
    estimates = fingerprint_dataset.distance_estimates()
    selected = estimates[distance_mode] if distance_mode in estimates else None
    print(f"distance mode requested: {distance_mode}")
    if selected is None:
        print("distance size estimate: no distance matrices requested")
        return
    print("distance size estimate:")
    print(f"- rows: {selected['rows']}")
    print(f"- matrix dimensions: {selected['rows']} x {selected['columns']}")
    print(f"- cells: {selected['cells']}")
    print(f"- estimated memory bytes: {selected['estimated_memory_bytes']}")
    print(f"- estimated CSV bytes per matrix: {selected['estimated_csv_bytes']}")
    if distance_mode == "individual":
        print(f"- max individual distance rows: {max_individual_distance_rows}")
        print(f"- large matrix override: {allow_large_distance_matrix}")


if __name__ == "__main__":
    raise SystemExit(main())
