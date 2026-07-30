"""Run Stage 8C Feature Engine V2 and feature-family ablation benchmarks."""

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
from src.feature_engine_v2 import run_feature_family_ablation


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "feature_engineering" / "stage_8c"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        _ensure_output_dir_allowed(args.output_dir, overwrite=args.overwrite)
        canonical_dataframe = _load_canonical_input(args)
        result = run_feature_family_ablation(
            canonical_dataframe,
            classification_models=_parse_models(args.classification_models),
            regression_models=_parse_models(args.regression_models),
            n_splits=args.n_splits,
            n_repeats=args.n_repeats,
            preprocessing=args.preprocessing,
            random_state=args.random_state,
            permutation_repeats=args.permutation_repeats,
        )
        output_paths = result.write_outputs(args.output_dir, overwrite=args.overwrite)
    except (
        FileExistsError,
        FileNotFoundError,
        NotADirectoryError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"Stage 8C feature-engineering benchmark failed: {error}", file=sys.stderr)
        return 1

    _print_summary(result, output_paths)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_folder",
        nargs="?",
        type=Path,
        help="Folder containing biosensor source files; canonical rows are built before V2 feature extraction.",
    )
    parser.add_argument(
        "--canonical-file",
        type=Path,
        help="Existing canonical dataset CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for Stage 8C outputs.",
    )
    parser.add_argument(
        "--classification-models",
        default="extra_trees",
        help="Comma-separated Stage 8A model ids used for ablation reruns; use all for all available models.",
    )
    parser.add_argument(
        "--regression-models",
        default="extra_trees",
        help="Comma-separated Stage 8B model ids used for ablation reruns; use all for all available models.",
    )
    parser.add_argument("--n-splits", type=int, default=3, help="Cross-validation split count for Stage 8C screening.")
    parser.add_argument("--n-repeats", type=int, default=1, help="Cross-validation repeats for Stage 8C screening.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for deterministic ablation.")
    parser.add_argument("--permutation-repeats", type=int, default=2, help="Permutation importance repeats.")
    parser.add_argument(
        "--preprocessing",
        choices=["none", "zscore", "robust", "minmax"],
        default="zscore",
        help="Preprocessing passed into Stage 8A/8B sklearn pipelines.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing files in the output directory.")
    return parser.parse_args(argv)


def _ensure_output_dir_allowed(output_dir: Path, *, overwrite: bool) -> None:
    if not output_dir.exists() or overwrite:
        return
    if any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")


def _parse_models(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    if str(value).strip().casefold() == "all":
        return None
    models = tuple(part.strip() for part in str(value).split(",") if part.strip())
    return models or None


def _load_canonical_input(args: argparse.Namespace) -> pd.DataFrame:
    if bool(args.source_folder) == bool(args.canonical_file):
        raise ValueError("Provide exactly one input: source_folder or --canonical-file.")
    if args.canonical_file:
        if not args.canonical_file.exists():
            raise FileNotFoundError(args.canonical_file)
        return pd.read_csv(args.canonical_file)
    return _build_canonical_dataframe(args.source_folder)


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
        except Exception as error:  # noqa: BLE001 - CLI should report every import failure.
            failed_files.append(f"{record.filename}: {type(error).__name__}: {error}")
    if failed_files:
        formatted = "\n".join(f"- {failure}" for failure in failed_files)
        raise RuntimeError(f"One or more source files failed to import:\n{formatted}")
    return build_canonical_dataset(read_results).dataframe


def _print_summary(result, output_paths: list[Path]) -> None:
    metadata = result.metadata
    print(f"advanced feature rows: {metadata['advanced_feature_rows']}")
    print(f"new feature count: {metadata['new_feature_count']}")
    print(f"feature families: {metadata['feature_family_count']}")
    print(f"feature sets benchmarked: {metadata['feature_set_count']}")
    print(f"best feature family: {metadata['best_feature_family']}")
    print(f"worst feature family: {metadata['worst_feature_family']}")
    print(f"classification improvement: {metadata['best_classification_gain']}")
    print(f"regression improvement: {metadata['best_regression_r2_gain']}")
    print(f"all-family runtime increase seconds: {metadata['all_families_runtime_increase_seconds']}")
    print("ablation summary:")
    for row in result.ablation_summary.itertuples(index=False):
        print(
            f"- {row.feature_set}: macro_f1={float(row.classification_macro_f1):.6g}, "
            f"r2={float(row.regression_r2):.6g}, rmse={float(row.regression_rmse):.6g}, "
            f"runtime={float(row.total_runtime_seconds):.6g}s"
        )
    print("warnings:")
    if result.warnings:
        for warning in result.warnings[:30]:
            print(f"- {warning}")
        if len(result.warnings) > 30:
            print(f"- ... {len(result.warnings) - 30} additional warnings")
    else:
        print("- none")
    print("output paths:")
    for path in output_paths:
        print(f"- {path}")


if __name__ == "__main__":
    raise SystemExit(main())
