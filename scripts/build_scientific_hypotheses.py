"""Build BSIP scientific hypotheses from validated interpretations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.scientific_reasoning.hypothesis import DEFAULT_SOFTWARE_VERSION, HypothesisEngine


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build explicit scientific hypotheses from Interpretation Engine outputs."
    )
    parser.add_argument("--project-root", default=".", help="Project root directory.")
    parser.add_argument(
        "--interpretations-dir",
        default="outputs/scientific_interpretations",
        help="Interpretation Engine output directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/scientific_hypotheses",
        help="Hypothesis Engine output directory.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace the specified output directory.")
    parser.add_argument(
        "--software-version",
        default=DEFAULT_SOFTWARE_VERSION,
        help="Software version string recorded in hypothesis outputs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    engine = HypothesisEngine(
        project_root=Path(args.project_root),
        interpretations_dir=Path(args.interpretations_dir),
        output_dir=Path(args.output_dir),
        overwrite=args.overwrite,
        software_version=args.software_version,
    )
    try:
        result = engine.run()
    except (FileExistsError, ValueError, OSError) as exc:
        print(json.dumps({"validation_passed": False, "error": str(exc)}, indent=2, sort_keys=True))
        return 2

    summary = {
        "validation_passed": bool(result.metadata.get("validation_passed")),
        "hypothesis_count": result.metadata.get("hypothesis_count", 0),
        "competing_hypothesis_count": result.metadata.get("competing_hypothesis_count", 0),
        "source_interpretation_count": result.metadata.get("source_interpretation_count", 0),
        "critical_issue_count": result.metadata.get("critical_issue_count", 0),
        "warning_count": result.metadata.get("warning_count", 0),
        "outputs": [str(path) for path in result.output_paths],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["validation_passed"] and summary["outputs"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
