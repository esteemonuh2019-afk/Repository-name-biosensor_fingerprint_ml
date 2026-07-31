"""Aggregate Stage 9B.2A scientific evidence into compact traceable summaries."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scientific_narrative import aggregate_scientific_evidence, write_aggregation_outputs


DEFAULT_EVIDENCE_FILE = Path("outputs") / "scientific_narrative_2" / "scientific_evidence.csv"
DEFAULT_OUTPUT_DIR = Path("outputs") / "scientific_aggregation"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        evidence_path = _resolve(args.project_root, args.evidence_file)
        aggregation = aggregate_scientific_evidence(evidence_path)
        output_paths = write_aggregation_outputs(
            aggregation,
            _resolve(args.project_root, args.output_dir),
            overwrite=args.overwrite,
            maximum_source_ids_per_cell=args.maximum_source_ids_per_cell,
        )
    except (FileExistsError, FileNotFoundError, NotADirectoryError, RuntimeError, TypeError, ValueError) as error:
        print(f"Scientific evidence aggregation failed: {error}", file=sys.stderr)
        return 1

    _print_summary(aggregation, output_paths)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Project root containing the Stage 9B.2A scientific evidence file.",
    )
    parser.add_argument(
        "--evidence-file",
        type=Path,
        default=DEFAULT_EVIDENCE_FILE,
        help="Stage 9B.2A scientific_evidence.csv to aggregate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for Stage 9B.2B aggregation outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing Stage 9B.2B aggregation outputs.",
    )
    parser.add_argument(
        "--maximum-source-ids-per-cell",
        type=int,
        default=50,
        help="Maximum source evidence IDs to write into one CSV cell.",
    )
    args = parser.parse_args(argv)
    if args.maximum_source_ids_per_cell < 1:
        parser.error("--maximum-source-ids-per-cell must be at least 1")
    return args


def _resolve(project_root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def _print_summary(aggregation, output_paths: list[Path]) -> None:
    metadata = aggregation.metadata
    print(f"evidence records received: {metadata.get('evidence_records_received', 0)}")
    print(f"evidence records used: {metadata.get('evidence_records_used', 0)}")
    print(f"summary records created: {metadata.get('summary_records_created', 0)}")
    print(f"unsupported/null evidence rows excluded: {metadata.get('unsupported_or_null_evidence_records', 0)}")
    print(f"unsupported files from evidence JSON: {metadata.get('unsupported_file_count_from_json', 0)}")
    print(f"conflicting summaries: {metadata.get('conflicting_summary_count', 0)}")
    print(f"missing expected summaries: {metadata.get('missing_summary_count', 0)}")
    print(f"traceability coverage: {metadata.get('traceability_coverage', 0)}")
    print(f"aggregation success: {aggregation.aggregation_passed}")
    print(f"scientific interpretation can proceed: {metadata.get('interpretation_can_proceed', False)}")
    print("output paths:")
    for path in output_paths:
        print(f"- {path}")


if __name__ == "__main__":
    raise SystemExit(main())
