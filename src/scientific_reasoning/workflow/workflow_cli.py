"""Command-line entry point for BSIP workflow orchestration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .workflow_engine import WorkflowEngine
from .workflow_models import WORKFLOW_SOFTWARE_VERSION


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BSIP scientific reasoning workflow.")
    parser.add_argument("--project-root", default=".", help="Project root directory.")
    parser.add_argument("--output-root", default="outputs", help="Root directory for reasoning outputs.")
    parser.add_argument("--overwrite", action="store_true", help="Replace stage output directories before running.")
    parser.add_argument("--resume", action="store_true", help="Skip stages whose outputs already validate successfully.")
    parser.add_argument(
        "--software-version",
        default=WORKFLOW_SOFTWARE_VERSION,
        help="Workflow software version recorded in the manifest.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    engine = WorkflowEngine(
        project_root=Path(args.project_root),
        output_root=Path(args.output_root),
        overwrite=args.overwrite,
        resume=args.resume,
        software_version=args.software_version,
    )
    try:
        result = engine.run()
    except (OSError, ValueError) as exc:
        print(json.dumps({"overall_status": "FAILED", "error": str(exc)}, indent=2, sort_keys=True))
        return 2

    summary = {
        "workflow_id": result.workflow_id,
        "overall_status": result.overall_status.value,
        "completed_stages": [
            record.stage_name.value
            for record in result.stage_records
            if record.status.value in {"COMPLETED", "SKIPPED"}
        ],
        "failed_stages": [
            record.stage_name.value
            for record in result.stage_records
            if record.status.value == "FAILED"
        ],
        "manifest": None if result.manifest_path is None else str(result.manifest_path),
        "report": None if result.report_path is None else str(result.report_path),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result.overall_status.value == "COMPLETED" else 2
