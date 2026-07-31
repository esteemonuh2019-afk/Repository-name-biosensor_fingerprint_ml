"""Build BSIP v4.1.0 scientific review findings from existing downstream artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.scientific_reasoning.reviewer import REVIEW_SOFTWARE_VERSION, ReviewerEngine  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic BSIP scientific review findings.")
    parser.add_argument("--project-root", default=".", help="Project root directory.")
    parser.add_argument("--claims-dir", default="outputs/scientific_claims", help="Claim Engine output directory.")
    parser.add_argument("--evidence-scoring-dir", default="outputs/evidence_scoring", help="Evidence Scoring Engine output directory.")
    parser.add_argument("--reasoning-graph-dir", default="outputs/reasoning_graph", help="Reasoning Graph Engine output directory.")
    parser.add_argument("--supervisor-dir", default="outputs/supervisor_results_2", help="Optional supervisor output directory.")
    parser.add_argument("--output-dir", default="outputs/scientific_review", help="Reviewer Engine output directory.")
    parser.add_argument("--overwrite", action="store_true", help="Replace the specified review output directory.")
    parser.add_argument("--strict", action="store_true", help="Return non-zero when warnings are present.")
    parser.add_argument("--software-version", default=REVIEW_SOFTWARE_VERSION, help="Reviewer Engine software version.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = ReviewerEngine(
            project_root=Path(args.project_root),
            claims_dir=Path(args.claims_dir),
            evidence_scoring_dir=Path(args.evidence_scoring_dir),
            reasoning_graph_dir=Path(args.reasoning_graph_dir),
            supervisor_dir=Path(args.supervisor_dir),
            output_dir=Path(args.output_dir),
            overwrite=args.overwrite,
            strict=args.strict,
            software_version=args.software_version,
        ).run()
    except (OSError, ValueError, TypeError, RuntimeError, FileExistsError) as exc:
        print(json.dumps({"overall_status": "FAILED", "error": str(exc)}, indent=2, sort_keys=True))
        return 2

    critical_count = int(result.metadata.get("critical_issue_count") or 0)
    warning_count = int(result.metadata.get("warning_count") or 0)
    summary = {
        "findings_generated": result.metadata.get("findings_generated", 0),
        "blocking_findings": result.metadata.get("blocking_findings", 0),
        "overall_recommendation": result.metadata.get("overall_recommendation"),
        "overall_readiness_score": result.metadata.get("overall_readiness_score"),
        "validation_status": result.metadata.get("validation_passed", False),
        "output_directory": str(Path(args.output_dir)),
        "critical_issue_count": critical_count,
        "warning_count": warning_count,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if critical_count:
        return 2
    if args.strict and warning_count:
        return 2
    return 0 if result.output_paths else 2


if __name__ == "__main__":
    raise SystemExit(main())
