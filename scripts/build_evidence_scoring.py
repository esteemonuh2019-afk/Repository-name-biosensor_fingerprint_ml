"""Build BSIP v4.0.0 evidence scores from validated claims and reasoning graph outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.scientific_reasoning.evidence_scoring import EVIDENCE_SCORING_SOFTWARE_VERSION, EvidenceScoringEngine  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic BSIP evidence-support scores.")
    parser.add_argument("--project-root", default=".", help="Project root directory.")
    parser.add_argument("--claims-dir", default="outputs/scientific_claims", help="Claim Engine output directory.")
    parser.add_argument("--graph-dir", default="outputs/reasoning_graph", help="Reasoning Graph Engine output directory.")
    parser.add_argument("--output-dir", default="outputs/evidence_scoring", help="Evidence scoring output directory.")
    parser.add_argument("--overwrite", action="store_true", help="Replace the specified evidence scoring output directory.")
    parser.add_argument("--strict", action="store_true", help="Return non-zero when warnings are present.")
    parser.add_argument("--software-version", default=EVIDENCE_SCORING_SOFTWARE_VERSION, help="Evidence Scoring Engine software version.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = EvidenceScoringEngine(
            project_root=Path(args.project_root),
            claims_dir=Path(args.claims_dir),
            graph_dir=Path(args.graph_dir),
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
        "claims_loaded": result.metadata.get("claims_loaded", 0),
        "claims_scored": result.metadata.get("claims_scored", 0),
        "claims_withheld": result.metadata.get("claims_withheld", 0),
        "mean_evidence_score": result.metadata.get("mean_evidence_score"),
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
