"""Build BSIP Scientific Observation Engine outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.scientific_reasoning.observation.engine import (  # noqa: E402
    DEFAULT_SOFTWARE_VERSION,
    ScientificObservationEngine,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build factual scientific observations from supervisor results.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--supervisor-results", default="outputs/supervisor_results_2")
    parser.add_argument("--output-dir", default="outputs/scientific_observations")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--software-version", default=DEFAULT_SOFTWARE_VERSION)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    engine = ScientificObservationEngine(
        project_root=args.project_root,
        supervisor_results_dir=args.supervisor_results,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        software_version=args.software_version,
    )
    try:
        result = engine.run()
    except FileExistsError as exc:
        print(json.dumps({"validation_passed": False, "error": str(exc)}, indent=2))
        return 2
    validation_summary = result.write_result.validation_summary if result.write_result else {}
    observation_summary = result.write_result.observation_summary if result.write_result else {}
    summary = {
        "validation_passed": validation_summary.get("validation_passed", result.validation_passed),
        "observation_count": len(result.observations),
        "incomplete_observation_count": validation_summary.get("incomplete_observation_count", 0),
        "critical_issue_count": validation_summary.get("critical_issue_count", 0),
        "warning_count": validation_summary.get("warning_count", 0),
        "selected_classifier": observation_summary.get("selected_classifier"),
        "selected_regressor": observation_summary.get("selected_regressor"),
        "outputs": [str(path) for path in result.output_paths],
    }
    print(json.dumps(summary, indent=2))
    return 0 if summary["validation_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
