"""Run Stage 7B exploratory analysis on validated biosensor fingerprints."""

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
from src.exploratory_analysis import run_exploratory_analysis


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "exploratory" / "stage_7b"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        individual, consensus, upstream_warnings, upstream_errors = _load_inputs(args)
        result = run_exploratory_analysis(
            individual,
            consensus,
            scaling=args.scaling,
            distance=args.distance,
            linkage_method=args.linkage,
            individual_pca=args.individual_pca,
            upstream_warnings=upstream_warnings,
            upstream_errors=upstream_errors,
        )
        output_paths = result.write_outputs(args.output_dir, overwrite=args.overwrite)
    except (FileExistsError, FileNotFoundError, NotADirectoryError, RuntimeError, TypeError, ValueError) as error:
        print(f"Exploratory analysis failed: {error}", file=sys.stderr)
        return 1

    _print_summary(result, output_paths)
    return 0 if result.analysis_passed else 1


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_folder",
        nargs="?",
        type=Path,
        help="Source folder; canonical, feature, validation, and fingerprint datasets are built first.",
    )
    parser.add_argument("--fingerprint-file", type=Path, help="Existing individual fingerprint dataset CSV.")
    parser.add_argument("--consensus-file", type=Path, help="Existing consensus fingerprint dataset CSV.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--scaling", choices=["zscore", "robust", "minmax", "none"], default="zscore")
    parser.add_argument("--distance", choices=["euclidean", "cosine", "correlation"], default="euclidean")
    parser.add_argument("--linkage", choices=["ward", "average", "complete"], default="ward")
    parser.add_argument("--individual-pca", action="store_true")
    return parser.parse_args(argv)


def _load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    file_mode = bool(args.fingerprint_file or args.consensus_file)
    source_mode = bool(args.source_folder)
    if source_mode == file_mode:
        raise ValueError("Provide either source_folder or both --fingerprint-file and --consensus-file.")
    if file_mode:
        if not args.fingerprint_file or not args.consensus_file:
            raise ValueError("Both --fingerprint-file and --consensus-file are required in file mode.")
        return pd.read_csv(args.fingerprint_file), pd.read_csv(args.consensus_file), [], []

    canonical = _build_canonical_dataframe(args.source_folder)
    features = extract_features(canonical)
    validation = validate_features(features)
    fingerprints = build_fingerprint_dataset(validation)
    return (
        fingerprints.dataframe,
        fingerprints.consensus_dataframe,
        list(fingerprints.warnings),
        list(fingerprints.errors),
    )


def _build_canonical_dataframe(source_folder: Path) -> pd.DataFrame:
    discovery = discover_biosensor_files(source_folder)
    read_results = []
    failures: list[str] = []
    for record in discovery.files:
        try:
            if record.extension == ".csv":
                read_results.append(read_biosensor_csv(record.absolute_path))
            elif record.extension == ".xlsx":
                read_results.append(read_biosensor_excel(record.absolute_path))
        except Exception as error:  # noqa: BLE001 - CLI reports aggregated failures.
            failures.append(f"{record.filename}: {type(error).__name__}: {error}")
    if failures:
        formatted = "\n".join(f"- {failure}" for failure in failures)
        raise RuntimeError(f"One or more source files failed to import:\n{formatted}")
    return build_canonical_dataset(read_results).dataframe


def _print_summary(result, output_paths: list[Path]) -> None:
    metadata = result.metadata
    explained = result.explained_variance.set_index("component") if not result.explained_variance.empty else pd.DataFrame()
    clusters = result.clustering_results.get("cluster_assignments", pd.DataFrame())
    replicate = result.replicate_to_consensus_distances

    print(f"individual fingerprint count: {metadata.get('individual_fingerprint_count', 0)}")
    print(f"consensus fingerprint count: {metadata.get('consensus_fingerprint_count', 0)}")
    print(f"features included: {metadata.get('feature_count', 0)}")
    print(f"rows excluded for QC/analysis: {metadata.get('excluded_for_analysis_count', 0)}")
    for component in ("PC1", "PC2", "PC3"):
        if not explained.empty and component in explained.index:
            print(
                f"{component} explained variance: "
                f"{float(explained.loc[component, 'explained_variance_ratio']):.6f}"
            )
    print("top PCA contributors:")
    for row in result.top_component_features.head(10).to_dict("records"):
        print(f"- {row['component']} rank {row['rank']}: {row['feature']} ({row['loading']:.6f})")
    print(f"clusters generated: {clusters['cluster_id'].nunique() if not clusters.empty else 0}")
    if not replicate.empty:
        print(
            "replicate-to-consensus distance summary: "
            f"median={replicate['distance_to_consensus'].median():.6f}, "
            f"max={replicate['distance_to_consensus'].max():.6f}"
        )
    print(f"concentration trajectory rows: {len(result.concentration_trajectories)}")
    print(f"strain dispersion rows: {len(result.strain_dispersion)}")
    print("warnings:")
    if result.warnings:
        for warning in result.warnings:
            print(f"- {warning}")
    else:
        print("- none")
    print("output paths:")
    for path in output_paths:
        print(f"- {path}")


if __name__ == "__main__":
    raise SystemExit(main())
