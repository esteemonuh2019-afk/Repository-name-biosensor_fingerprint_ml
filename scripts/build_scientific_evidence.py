"""Build the Stage 9B.2A scientific evidence database from selected results."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scientific_narrative import build_scientific_evidence, write_evidence_outputs


DEFAULT_SELECTED_RESULTS = Path("outputs") / "results_inventory" / "selected_results.csv"
DEFAULT_OUTPUT_DIR = Path("outputs") / "scientific_narrative"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        project_root = args.project_root.resolve()
        database = build_scientific_evidence(
            project_root=project_root,
            selected_results_path=args.selected_results,
        )
        output_paths = write_evidence_outputs(
            database,
            _resolve(project_root, args.output_dir),
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, NotADirectoryError, RuntimeError, TypeError, ValueError) as error:
        print(f"Scientific evidence extraction failed: {error}", file=sys.stderr)
        return 1

    _print_summary(database, output_paths)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Project root containing outputs/results_inventory/selected_results.csv.",
    )
    parser.add_argument(
        "--selected-results",
        type=Path,
        default=DEFAULT_SELECTED_RESULTS,
        help="Selected results CSV produced by Stage 9B.1.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for Stage 9B.2A evidence outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing Stage 9B.2A evidence outputs.",
    )
    return parser.parse_args(argv)


def _resolve(project_root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def _print_summary(database, output_paths: list[Path]) -> None:
    print(f"files parsed: {len(database.parsed_files)}")
    print(f"evidence records extracted: {len(database.records)}")
    print(f"unsupported files: {len(database.unsupported_files)}")
    print(f"unreadable files: {len(database.unreadable_files)}")
    print(f"missing expected evidence records: {len(database.missing_evidence)}")
    print(f"extraction success: {database.extraction_success}")
    print("output paths:")
    for path in output_paths:
        print(f"- {path}")


if __name__ == "__main__":
    raise SystemExit(main())
