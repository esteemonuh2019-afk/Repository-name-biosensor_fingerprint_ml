"""Predict a blind biosensor sample using a frozen Stage 9A model bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.blind_prediction import load_model_bundle, predict_blind_sample  # noqa: E402
from src.data_ingestion.canonical_builder import build_canonical_dataset  # noqa: E402
from src.data_ingestion.csv_reader import read_biosensor_csv  # noqa: E402
from src.data_ingestion.excel_reader import read_biosensor_excel  # noqa: E402
from src.data_ingestion.file_discovery import discover_biosensor_files  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        canonical, source_files = _load_blind_canonical(args)
        bundle = load_model_bundle(args.model_dir)
        result = predict_blind_sample(canonical, bundle=bundle, source_files=source_files)
        paths = result.write_outputs(args.output_dir, overwrite=args.overwrite)
    except (
        FileExistsError,
        FileNotFoundError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"Stage 9A blind prediction failed: {error}", file=sys.stderr)
        return 1

    _print_summary(result, paths)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("blind_sample_folder", nargs="?", type=Path)
    parser.add_argument("--canonical-file", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _load_blind_canonical(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    if bool(args.blind_sample_folder) == bool(args.canonical_file):
        raise ValueError("Provide exactly one blind input: blind_sample_folder or --canonical-file.")
    if args.canonical_file:
        if not args.canonical_file.exists():
            raise FileNotFoundError(args.canonical_file)
        dataframe = pd.read_csv(args.canonical_file)
        source_files = sorted(dataframe.get("Source_File", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
        return dataframe, source_files
    discovery = discover_biosensor_files(args.blind_sample_folder)
    read_results = []
    source_files = []
    failed_files: list[str] = []
    for record in discovery.files:
        try:
            source_files.append(record.filename)
            if record.extension == ".csv":
                read_results.append(read_biosensor_csv(record.absolute_path))
            elif record.extension == ".xlsx":
                read_results.append(read_biosensor_excel(record.absolute_path))
        except Exception as error:  # noqa: BLE001
            failed_files.append(f"{record.filename}: {type(error).__name__}: {error}")
    if failed_files:
        formatted = "\n".join(f"- {failure}" for failure in failed_files)
        raise RuntimeError(f"One or more blind source files failed to import:\n{formatted}")
    return build_canonical_dataset(read_results).dataframe, source_files


def _print_summary(result, paths: list[Path]) -> None:
    print(f"predicted chemical: {result.predicted_chemical}")
    print(f"chemical confidence: {result.chemical_confidence}")
    print(f"prediction margin: {result.prediction_margin}")
    print(f"predicted concentration: {result.predicted_concentration}")
    print(f"concentration units: {result.concentration_units}")
    print(f"novelty status: {result.novelty_status}")
    print(f"novelty score: {result.novelty_score}")
    print(f"prediction passed: {result.prediction_passed}")
    print("top three candidates:")
    for row in result.top_three_candidates:
        print(f"- {row['chemical']}: {row['probability']:.6g}")
    print("warnings:")
    if result.warnings:
        for warning in result.warnings[:30]:
            print(f"- {warning}")
        if len(result.warnings) > 30:
            print(f"- ... {len(result.warnings) - 30} additional warnings")
    else:
        print("- none")
    print("errors:")
    if result.errors:
        for error in result.errors[:30]:
            print(f"- {error}")
        if len(result.errors) > 30:
            print(f"- ... {len(result.errors) - 30} additional errors")
    else:
        print("- none")
    print("output paths:")
    for path in paths:
        print(f"- {path}")


if __name__ == "__main__":
    raise SystemExit(main())
