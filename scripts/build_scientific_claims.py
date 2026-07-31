"""Build BSIP v3.2.0 scientific claims from validated hypothesis and graph outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.scientific_reasoning.claim import ClaimEngine, DEFAULT_CLAIM_SOFTWARE_VERSION  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build evidence-bounded BSIP scientific claims.")
    parser.add_argument("--project-root", default=".", help="Project root directory.")
    parser.add_argument("--hypotheses-dir", default="outputs/scientific_hypotheses", help="Hypothesis Engine output directory.")
    parser.add_argument("--reasoning-graph-dir", default="outputs/reasoning_graph", help="Reasoning Graph Engine output directory.")
    parser.add_argument("--output-dir", default="outputs/scientific_claims", help="Claim Engine output directory.")
    parser.add_argument("--overwrite", action="store_true", help="Replace the specified claim output directory.")
    parser.add_argument("--software-version", default=DEFAULT_CLAIM_SOFTWARE_VERSION, help="Claim Engine software version.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = ClaimEngine(
            project_root=Path(args.project_root),
            hypotheses_dir=Path(args.hypotheses_dir),
            reasoning_graph_dir=Path(args.reasoning_graph_dir),
            output_dir=Path(args.output_dir),
            overwrite=args.overwrite,
            software_version=args.software_version,
        ).run()
    except (OSError, ValueError, TypeError, RuntimeError, FileExistsError) as exc:
        print(json.dumps({"overall_status": "FAILED", "error": str(exc)}, indent=2, sort_keys=True))
        return 2

    critical_count = int(result.metadata.get("critical_issue_count") or 0)
    summary = {
        "overall_status": "COMPLETED" if critical_count == 0 else "FAILED_VALIDATION",
        "claim_count": result.metadata.get("claim_count", len(result.claims)),
        "withheld_claim_count": result.metadata.get("withheld_claim_count", 0),
        "critical_issue_count": critical_count,
        "warning_count": result.metadata.get("warning_count", 0),
        "validation_passed": result.metadata.get("validation_passed", False),
        "outputs": [str(path) for path in result.output_paths],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if critical_count == 0 and result.output_paths else 2


if __name__ == "__main__":
    raise SystemExit(main())
