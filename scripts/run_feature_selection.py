"""Run Stage 8D automatic feature selection and benchmark reruns."""

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
from src.feature_selection import FeatureSelectionConfig, run_feature_selection


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "feature_selection"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        _ensure_output_dir_allowed(args.output_dir, overwrite=args.overwrite)
        canonical_dataframe = _load_canonical_input(args)
        result = run_feature_selection(
            canonical_dataframe,
            config=FeatureSelectionConfig(
                selector_methods=_parse_methods(args.selector_methods),
                reduction_levels=_parse_levels(args.reduction_levels),
                classification_model_ids=_parse_models(args.classification_models),
                regression_model_ids=_parse_models(args.regression_models),
                preprocessing=args.preprocessing,
                n_splits=args.n_splits,
                n_repeats=args.n_repeats,
                random_state=args.random_state,
                benchmark_permutation_importance=args.benchmark_permutation_importance,
                selection_permutation_repeats=args.selection_permutation_repeats,
                selection_tree_estimators=args.selection_tree_estimators,
                max_sequential_greedy_steps=args.max_sequential_greedy_steps,
                sequential_candidate_pool=args.sequential_candidate_pool,
                selection_cv_splits=args.selection_cv_splits,
                include_boruta=not args.skip_boruta,
            ),
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
        print(f"Stage 8D feature selection failed: {error}", file=sys.stderr)
        return 1

    _print_summary(result, output_paths)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_folder",
        nargs="?",
        type=Path,
        help="Folder containing biosensor source files; canonical rows are built before feature selection.",
    )
    parser.add_argument("--canonical-file", type=Path, help="Existing canonical dataset CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for Stage 8D outputs.",
    )
    parser.add_argument(
        "--selector-methods",
        default="rfe,sequential_forward,sequential_backward,permutation,tree_importance",
        help="Comma-separated selectors: rfe, sequential_forward, sequential_backward, permutation, tree_importance.",
    )
    parser.add_argument(
        "--reduction-levels",
        default="100,75,50,25,10",
        help="Comma-separated feature percentages to evaluate.",
    )
    parser.add_argument(
        "--classification-models",
        default="extra_trees",
        help="Comma-separated Stage 8A model ids for selected-subset reruns.",
    )
    parser.add_argument(
        "--regression-models",
        default="extra_trees",
        help="Comma-separated Stage 8B model ids for selected-subset reruns.",
    )
    parser.add_argument(
        "--preprocessing",
        choices=["none", "zscore", "robust", "minmax"],
        default="zscore",
        help="Preprocessing passed into Stage 8A/8B sklearn pipelines.",
    )
    parser.add_argument("--n-splits", type=int, default=3, help="Cross-validation split count.")
    parser.add_argument("--n-repeats", type=int, default=1, help="Cross-validation repeats.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--benchmark-permutation-importance",
        action="store_true",
        help="Also run Stage 8A/8B benchmark permutation importance for every selected subset.",
    )
    parser.add_argument("--selection-permutation-repeats", type=int, default=3)
    parser.add_argument("--selection-tree-estimators", type=int, default=100)
    parser.add_argument("--max-sequential-greedy-steps", type=int, default=16)
    parser.add_argument("--sequential-candidate-pool", type=int, default=12)
    parser.add_argument("--selection-cv-splits", type=int, default=2)
    parser.add_argument("--skip-boruta", action="store_true", help="Do not attempt optional Boruta selection.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing files in the output directory.")
    return parser.parse_args(argv)


def _ensure_output_dir_allowed(output_dir: Path, *, overwrite: bool) -> None:
    if not output_dir.exists() or overwrite:
        return
    if any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")


def _parse_models(value: str) -> tuple[str, ...]:
    models = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if not models:
        raise ValueError("At least one model id is required.")
    return models


def _parse_methods(value: str) -> tuple[str, ...]:
    methods = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if not methods:
        raise ValueError("At least one selector method is required.")
    return methods


def _parse_levels(value: str) -> tuple[int, ...]:
    levels = tuple(int(part.strip()) for part in str(value).split(",") if part.strip())
    if not levels:
        raise ValueError("At least one reduction level is required.")
    invalid = [level for level in levels if level <= 0 or level > 100]
    if invalid:
        raise ValueError("Reduction levels must be percentages in the range 1..100.")
    return tuple(sorted(set(levels), reverse=True))


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
    class_rec = metadata["default_classification_feature_set"]
    reg_rec = metadata["default_regression_feature_set"]
    research = metadata["research_feature_set"]
    print(f"generated feature rows: {metadata['generated_feature_rows']}")
    print(f"available features: {metadata['available_feature_count']}")
    print(f"selector methods completed: {', '.join(metadata['selector_methods_completed'])}")
    print(f"boruta status: {metadata['boruta_status']}")
    print(
        "default classification feature set: "
        f"{class_rec['selector_method']} {class_rec['reduction_level_percent']}% "
        f"({class_rec['feature_count']} features), macro_f1={class_rec['macro_f1_mean']:.6g}, "
        f"balanced_accuracy={class_rec['balanced_accuracy_mean']:.6g}"
    )
    print(
        "default regression feature set: "
        f"{reg_rec['selector_method']} {reg_rec['reduction_level_percent']}% "
        f"({reg_rec['feature_count']} features), r2={reg_rec['r2_mean']:.6g}, "
        f"rmse={reg_rec['rmse_mean']:.6g}, mae={reg_rec['mae_mean']:.6g}"
    )
    print(f"research feature set: {research['feature_count']} features")
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
