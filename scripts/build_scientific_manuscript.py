"""Build a conservative BSIP v4.2.0 scientific manuscript draft."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.scientific_reasoning.manuscript import DEFAULT_TITLE, MANUSCRIPT_SOFTWARE_VERSION, ManuscriptEngine  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an evidence-traceable BSIP manuscript draft.")
    parser.add_argument("--project-root", default=".", help="Project root directory.")
    parser.add_argument("--observations-dir", default="outputs/scientific_observations", help="Observation Engine output directory.")
    parser.add_argument("--interpretations-dir", default="outputs/scientific_interpretations", help="Interpretation Engine output directory.")
    parser.add_argument("--hypotheses-dir", default="outputs/scientific_hypotheses", help="Hypothesis Engine output directory.")
    parser.add_argument("--claims-dir", default="outputs/scientific_claims", help="Claim Engine output directory.")
    parser.add_argument("--evidence-dir", default="outputs/evidence_scoring", help="Evidence Scoring Engine output directory.")
    parser.add_argument("--review-dir", default="outputs/scientific_review", help="Reviewer Engine output directory.")
    parser.add_argument("--graph-dir", default="outputs/reasoning_graph", help="Reasoning Graph Engine output directory.")
    parser.add_argument("--supervisor-results", default="outputs/supervisor_results_2", help="Supervisor selected results directory.")
    parser.add_argument("--output-dir", default="outputs/scientific_manuscript", help="Manuscript Engine output directory.")
    parser.add_argument("--overwrite", action="store_true", help="Replace the manuscript output directory.")
    parser.add_argument("--strict", action="store_true", help="Return non-zero when warnings are present.")
    parser.add_argument("--software-version", default=MANUSCRIPT_SOFTWARE_VERSION, help="Manuscript Engine software version.")
    parser.add_argument("--title", default=DEFAULT_TITLE, help="Manuscript draft title.")
    parser.add_argument("--author", default=None, help="Optional author label for the draft title page.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = ManuscriptEngine(
            project_root=Path(args.project_root),
            observations_dir=Path(args.observations_dir),
            interpretations_dir=Path(args.interpretations_dir),
            hypotheses_dir=Path(args.hypotheses_dir),
            claims_dir=Path(args.claims_dir),
            evidence_dir=Path(args.evidence_dir),
            review_dir=Path(args.review_dir),
            graph_dir=Path(args.graph_dir),
            supervisor_results=Path(args.supervisor_results),
            output_dir=Path(args.output_dir),
            overwrite=args.overwrite,
            strict=args.strict,
            software_version=args.software_version,
            title=args.title,
            author=args.author,
        ).run()
    except (OSError, ValueError, TypeError, RuntimeError, FileExistsError) as exc:
        print(json.dumps({"overall_status": "FAILED", "error": str(exc)}, indent=2, sort_keys=True))
        return 2

    critical_count = int(result.metadata.get("critical_issue_count") or 0)
    warning_count = int(result.metadata.get("warning_count") or 0)
    summary = {
        "document_status": result.metadata.get("document_status"),
        "sentence_count": result.metadata.get("sentence_count"),
        "figure_caption_count": result.metadata.get("figure_caption_count"),
        "table_caption_count": result.metadata.get("table_caption_count"),
        "unresolved_revision_flag_count": result.metadata.get("unresolved_revision_flag_count"),
        "overall_reviewer_recommendation": result.metadata.get("overall_reviewer_recommendation"),
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
