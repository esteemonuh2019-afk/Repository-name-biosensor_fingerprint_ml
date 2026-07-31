"""Build the Stage 9B.1 generated-results inventory."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.results_inventory import build_results_inventory
from src.results_inventory.inventory_report import write_inventory_outputs


DEFAULT_OUTPUT_DIR = Path("outputs") / "results_inventory"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        project_root = args.project_root.resolve()
        inventory = build_results_inventory(
            project_root=project_root,
            outputs_dir=args.outputs_dir,
            output_dir=args.output_dir,
            include_large_files=args.include_large_files,
            minimum_file_size=args.minimum_file_size,
            maximum_hash_size_mb=args.maximum_hash_size_mb,
        )
        output_dir = _resolve(project_root, args.output_dir)
        output_paths = write_inventory_outputs(
            inventory,
            output_dir,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, NotADirectoryError, RuntimeError, TypeError, ValueError) as error:
        print(f"Results inventory failed: {error}", file=sys.stderr)
        return 1

    _print_summary(inventory, output_paths)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Project root containing outputs/, docs/, src/, and scripts/.",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Output tree to scan recursively. Relative paths are resolved under --project-root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for Stage 9B.1 inventory outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing inventory outputs in --output-dir.",
    )
    parser.add_argument(
        "--include-large-files",
        action="store_true",
        help="Explicitly allow hashing files above --maximum-hash-size-mb.",
    )
    parser.add_argument(
        "--minimum-file-size",
        type=int,
        default=0,
        help="Only inventory files at least this many bytes. Default inventories all files.",
    )
    parser.add_argument(
        "--maximum-hash-size-mb",
        type=float,
        default=100.0,
        help="Maximum file size hashed by default. Larger files are still inventoried.",
    )
    return parser.parse_args(argv)


def _resolve(project_root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def _print_summary(inventory, output_paths: list[Path]) -> None:
    metadata = inventory.scan_metadata
    print(f"total files inventoried: {metadata.get('total_files', 0)}")
    print(f"total output size bytes: {metadata.get('total_size_bytes', 0)}")
    print("analysis categories detected:")
    for category in metadata.get("analysis_categories_found", []):
        print(f"- {category}")
    print(f"runs detected: {len(inventory.detected_runs)}")
    for analysis_type in (
        "classification",
        "regression",
        "exploratory analysis",
        "advanced feature engineering",
        "feature selection",
    ):
        run = inventory.selected_runs.get(analysis_type)
        print(f"preferred {analysis_type} run: {run.run_name if run else 'none'}")
    missing = [
        item.report_section
        for item in inventory.missing_required_results
        if item.status in {"MISSING", "PARTIAL"}
    ]
    print("missing supervisor-report sections:")
    if missing:
        for section in missing:
            print(f"- {section}")
    else:
        print("- none")
    print(f"duplicate candidates: {len(inventory.duplicate_candidates)}")
    print(f"obsolete candidates: {len(inventory.obsolete_candidates)}")
    print(f"large-file warnings: {metadata.get('large_file_warning_count', 0)}")
    print(f"report generation can proceed: {inventory.project_health.get('report_generation_can_proceed')}")
    print("output paths:")
    for path in output_paths:
        print(f"- {path}")


if __name__ == "__main__":
    raise SystemExit(main())
