"""Build the Stage 9B.3 supervisor results package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.supervisor_report import build_supervisor_results_package, write_supervisor_results_package


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a supervisor-ready results package.")
    parser.add_argument("--project-root", required=True, help="Project root containing outputs/ and docs/.")
    parser.add_argument("--selected-results", required=True, help="Selected results inventory CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for supervisor package outputs.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing non-empty output directory.")
    parser.add_argument("--title", default="Biosensor Fingerprint ML Supervisor Results")
    parser.add_argument("--author", default="")
    parser.add_argument("--supervisor-name", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    selected_results = Path(args.selected_results)
    if not selected_results.is_absolute():
        selected_results = project_root / selected_results
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    package = build_supervisor_results_package(
        project_root=project_root,
        selected_results_path=selected_results,
        title=args.title,
        author=args.author,
        supervisor_name=args.supervisor_name,
    )
    result = write_supervisor_results_package(package, output_dir, overwrite=args.overwrite)
    print(json.dumps(result, indent=2))
    return 0 if result.get("package_passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
