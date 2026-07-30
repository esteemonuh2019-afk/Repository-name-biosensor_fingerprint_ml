"""Train a frozen Stage 9A blind-prediction model bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.blind_prediction import (  # noqa: E402
    BlindTrainingConfig,
    load_feature_profile,
    run_simulated_blind_test,
    train_blind_prediction_bundle,
)
from src.data_ingestion.canonical_builder import build_canonical_dataset  # noqa: E402
from src.data_ingestion.csv_reader import read_biosensor_csv  # noqa: E402
from src.data_ingestion.excel_reader import read_biosensor_excel  # noqa: E402
from src.data_ingestion.file_discovery import discover_biosensor_files  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "blind_prediction" / "v1"
DEFAULT_FEATURE_SELECTION_DIR = PROJECT_ROOT / "outputs" / "feature_selection"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        canonical = _load_canonical_input(args)
        profile = load_feature_profile(args.feature_selection_dir)
        config = BlindTrainingConfig(
            classifier_model_id=args.classifier_model,
            regressor_model_id=args.regressor_model,
            preprocessing=args.preprocessing,
            random_state=args.random_state,
            min_chemical_specific_rows=args.min_chemical_specific_rows,
            min_chemical_specific_concentrations=args.min_chemical_specific_concentrations,
            required_strains=tuple(_parse_csv(args.required_strains)),
        )
        bundle = train_blind_prediction_bundle(canonical, feature_profile=profile, config=config)
        paths = bundle.save(args.output_dir, overwrite=args.overwrite)
        simulation = None
        if args.simulate_blind:
            simulation = run_simulated_blind_test(
                canonical,
                feature_profile=profile,
                group_column=args.holdout_group_column,
                holdout_group=args.holdout_group,
                config=config,
            )
    except (
        FileExistsError,
        FileNotFoundError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"Stage 9A model training failed: {error}", file=sys.stderr)
        return 1

    _print_summary(bundle, paths, simulation)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_folder", nargs="?", type=Path)
    parser.add_argument("--canonical-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-selection-dir", type=Path, default=DEFAULT_FEATURE_SELECTION_DIR)
    parser.add_argument("--classifier-model", default="extra_trees")
    parser.add_argument("--regressor-model", default="extra_trees")
    parser.add_argument("--preprocessing", choices=["none", "zscore", "robust", "minmax"], default="zscore")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-chemical-specific-rows", type=int, default=6)
    parser.add_argument("--min-chemical-specific-concentrations", type=int, default=2)
    parser.add_argument("--required-strains", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--simulate-blind", action="store_true")
    parser.add_argument("--holdout-group-column", default="Source_File")
    parser.add_argument("--holdout-group")
    return parser.parse_args(argv)


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
        except Exception as error:  # noqa: BLE001
            failed_files.append(f"{record.filename}: {type(error).__name__}: {error}")
    if failed_files:
        formatted = "\n".join(f"- {failure}" for failure in failed_files)
        raise RuntimeError(f"One or more source files failed to import:\n{formatted}")
    return build_canonical_dataset(read_results).dataframe


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _print_summary(bundle, paths: list[Path], simulation) -> None:
    print(f"bundle version: {bundle.bundle_version}")
    print(f"pipeline version: {bundle.pipeline_version}")
    print(f"classification features: {len(bundle.classification_features)}")
    print(f"regression features: {len(bundle.regression_features)}")
    print(f"class labels: {len(bundle.class_labels)}")
    print(f"chemical-specific regressors: {len(bundle.chemical_regressors)}")
    print(f"regression strategy: {bundle.regression_strategy}")
    print(f"time window: {bundle.time_window.get('label')}")
    if simulation is not None:
        evaluation = simulation["evaluation"]
        print("simulated blind test:")
        print(f"- holdout group: {simulation['holdout_group']}")
        print(f"- group leakage prevented: {simulation['group_leakage_prevented']}")
        print(f"- chemical correct: {evaluation['chemical_prediction_correct']}")
        print(f"- concentration absolute error: {evaluation['concentration_absolute_error']}")
        print(f"- novelty status: {evaluation['novelty_status']}")
    print("output paths:")
    for path in paths:
        print(f"- {path}")


if __name__ == "__main__":
    raise SystemExit(main())
