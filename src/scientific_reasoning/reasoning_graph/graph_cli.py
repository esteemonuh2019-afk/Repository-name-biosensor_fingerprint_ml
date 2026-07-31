"""Command-line entry point for the BSIP reasoning graph engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .graph_builder import ReasoningGraphBuilder
from .graph_export import DEFAULT_OUTPUT_DIR, write_reasoning_graph_outputs
from .graph_models import GRAPH_SOFTWARE_VERSION
from .graph_validator import graph_validation_summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the BSIP reasoning graph from existing validated outputs.")
    parser.add_argument("--project-root", default=".", help="Project root directory.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Reasoning graph output directory.")
    parser.add_argument("--observations-dir", default="outputs/scientific_observations", help="Observation output directory.")
    parser.add_argument(
        "--interpretations-dir",
        default="outputs/scientific_interpretations",
        help="Interpretation output directory.",
    )
    parser.add_argument("--hypotheses-dir", default="outputs/scientific_hypotheses", help="Hypothesis output directory.")
    parser.add_argument("--workflow-dir", default="outputs/workflow", help="Workflow output directory.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing reasoning graph outputs.")
    parser.add_argument(
        "--software-version",
        default=GRAPH_SOFTWARE_VERSION,
        help="Reasoning graph software version recorded in graph metadata.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = Path(args.project_root)
    try:
        graph = ReasoningGraphBuilder(
            project_root=project_root,
            observations_dir=Path(args.observations_dir),
            interpretations_dir=Path(args.interpretations_dir),
            hypotheses_dir=Path(args.hypotheses_dir),
            workflow_dir=Path(args.workflow_dir),
            software_version=args.software_version,
        ).build()
        output_dir = _resolve_output(project_root, args.output_dir)
        paths = write_reasoning_graph_outputs(graph, output_dir=output_dir, overwrite=args.overwrite)
        validation = graph_validation_summary(graph)
    except (OSError, ValueError, TypeError, KeyError, FileExistsError) as exc:
        print(json.dumps({"overall_status": "FAILED", "error": str(exc)}, indent=2, sort_keys=True))
        return 2

    summary = {
        "overall_status": "COMPLETED" if validation["validation_passed"] else "FAILED_VALIDATION",
        "graph_id": graph.graph_id,
        "node_count": validation["node_count"],
        "edge_count": validation["edge_count"],
        "orphan_count": validation["orphan_count"],
        "cycle_count": validation["cycle_count"],
        "validation_passed": validation["validation_passed"],
        "outputs": {key: str(path) for key, path in sorted(paths.items())},
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if validation["validation_passed"] else 2


def _resolve_output(project_root: Path, output_dir: str) -> Path:
    candidate = Path(output_dir)
    if candidate.is_absolute():
        return candidate
    return project_root / candidate
