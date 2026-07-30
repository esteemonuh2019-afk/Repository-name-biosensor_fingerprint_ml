"""Run the Stage 8B concentration regression benchmark."""

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
from src.regression_benchmark import RegressionBenchmarkConfig, run_regression_benchmark


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "regression"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        _ensure_output_dir_allowed(args.output_dir, overwrite=args.overwrite)
        fingerprint_input = _load_fingerprint_input(args)
        config = RegressionBenchmarkConfig(
            validation_strategy=args.validation,
            preprocessing=args.preprocessing,
            n_splits=args.n_splits,
            n_repeats=args.n_repeats,
            random_state=args.random_state,
            target_column=args.target_column,
            model_ids=_parse_models(args.models),
            permutation_repeats=args.permutation_repeats,
            run_permutation_importance=not args.skip_permutation_importance,
            run_leave_one_strain_importance=not args.skip_leave_one_strain_importance,
        )
        result = run_regression_benchmark(fingerprint_input, config=config)
        output_paths = result.write_outputs(args.output_dir, overwrite=args.overwrite)
    except (
        FileExistsError,
        FileNotFoundError,
        NotADirectoryError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"Regression benchmark failed: {error}", file=sys.stderr)
        return 1

    _print_summary(result, output_paths)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_folder",
        nargs="?",
        type=Path,
        help="Folder containing biosensor source files; fingerprints are built through canonical, feature, and validation stages.",
    )
    parser.add_argument(
        "--fingerprint-file",
        type=Path,
        help="Existing Stage 7A fingerprint_dataset.csv file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for regression benchmark outputs.",
    )
    parser.add_argument(
        "--validation",
        choices=["repeated_kfold", "leave_one_strain_out", "leave_one_chemical_out"],
        default="repeated_kfold",
        help="Validation strategy. Leave-one-chemical-out is research mode.",
    )
    parser.add_argument(
        "--preprocessing",
        choices=["none", "zscore", "robust", "minmax"],
        default="zscore",
        help="Preprocessing applied inside sklearn pipelines.",
    )
    parser.add_argument(
        "--target-column",
        default="Concentration",
        help="Fingerprint metadata column used to derive numeric concentration targets.",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="Requested repeated K-fold split count.")
    parser.add_argument("--n-repeats", type=int, default=2, help="Repeated K-fold repeats.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for deterministic benchmarks.")
    parser.add_argument(
        "--models",
        help="Comma-separated model ids. Omit to evaluate all available supported regressors.",
    )
    parser.add_argument(
        "--permutation-repeats",
        type=int,
        default=5,
        help="Permutation importance repeats for the selected best model.",
    )
    parser.add_argument(
        "--skip-permutation-importance",
        action="store_true",
        help="Skip permutation importance for quicker diagnostic runs.",
    )
    parser.add_argument(
        "--skip-leave-one-strain-importance",
        action="store_true",
        help="Skip tree-model leave-one-strain-out importance.",
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


def _parse_models(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    models = tuple(part.strip() for part in value.split(",") if part.strip())
    return models or None


def _load_fingerprint_input(args: argparse.Namespace):
    if bool(args.source_folder) == bool(args.fingerprint_file):
        raise ValueError("Provide exactly one input: source_folder or --fingerprint-file.")
    if args.fingerprint_file:
        if "normalized" in args.fingerprint_file.name.casefold():
            raise ValueError(
                "Use original fingerprint_dataset.csv, not fingerprint_dataset_normalized.csv; "
                "benchmark scaling must occur inside validation folds."
            )
        if not args.fingerprint_file.exists():
            raise FileNotFoundError(args.fingerprint_file)
        return pd.read_csv(args.fingerprint_file)
    canonical_dataframe = _build_canonical_dataframe(args.source_folder)
    feature_dataset = extract_features(canonical_dataframe)
    validation_result = validate_features(feature_dataset)
    return build_fingerprint_dataset(validation_result, normalization="none")


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
    best = result.best_model_metrics
    print(f"sample count: {metadata['sample_count']}")
    print(
        "concentration range: "
        f"{metadata['concentration_min']:.6g} to {metadata['concentration_max']:.6g} "
        f"{metadata['target_units']}"
    )
    print(f"unique concentrations: {metadata['unique_concentration_count']}")
    print(f"validation strategy: {metadata['validation_strategy']}")
    print(f"preprocessing: {metadata['preprocessing']}")
    print(f"cross-validation folds: {metadata['fold_count']}")
    print("models evaluated:")
    for model in metadata["models_evaluated"]:
        print(f"- {model}")
    print("models skipped:")
    if metadata["models_skipped"]:
        for model in metadata["models_skipped"]:
            print(f"- {model}")
    else:
        print("- none")
    print("model rankings:")
    for row in result.rankings.itertuples(index=False):
        print(
            f"- {row.rank}. {row.model_name}: "
            f"R2={float(row.r2_mean):.6g}, RMSE={float(row.rmse_mean):.6g}, "
            f"MAE={float(row.mae_mean):.6g}"
        )
    print(f"best regression model: {best['model_name']}")
    print(f"best R2 mean: {best['r2_mean']}")
    print(f"best RMSE mean: {best['rmse_mean']}")
    print(f"best MAE mean: {best['mae_mean']}")
    print("most informative features:")
    for feature, value in _top_features(result):
        print(f"- {feature}: {value:.6g}")
    print("warnings:")
    if result.warnings:
        for warning in result.warnings:
            print(f"- {warning}")
    else:
        print("- none")
    print("output paths:")
    for path in output_paths:
        print(f"- {path}")


def _top_features(result) -> list[tuple[str, float]]:
    if not result.permutation_importance.empty:
        table = result.permutation_importance.sort_values(
            ["importance_mean", "feature"],
            ascending=[False, True],
        ).head(10)
        return [(str(row.feature), float(row.importance_mean)) for row in table.itertuples(index=False)]
    if result.feature_importance.empty:
        return []
    table = (
        result.feature_importance.groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values(["importance", "feature"], ascending=[False, True])
        .head(10)
    )
    return [(str(row.feature), float(row.importance)) for row in table.itertuples(index=False)]


if __name__ == "__main__":
    raise SystemExit(main())
