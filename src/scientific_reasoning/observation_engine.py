"""Observation Engine for converting validated outputs into factual observations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .observation_models import OBSERVATION_FIELDS, Observation, ObservationDatabase
from .observation_registry import DEFAULT_OUTPUT_DIR, DEFAULT_SUMMARY_PATH, DEFAULT_VALIDATION_PATH
from .observation_rules import build_observations_from_summary, incomplete_observations


def build_observation_database(
    project_root: str | Path = ".",
    *,
    summary_path: str | Path | None = None,
    validation_path: str | Path | None = None,
) -> ObservationDatabase:
    """Read validated supervisor outputs and build factual observations."""

    root = Path(project_root)
    summary_file = _resolve_input(root, summary_path or DEFAULT_SUMMARY_PATH)
    validation_file = _resolve_input(root, validation_path or DEFAULT_VALIDATION_PATH)
    warnings: list[str] = []
    errors: list[str] = []

    validation = _read_json(validation_file)
    if validation is None:
        errors.append(f"Validation file is missing or unreadable: {validation_file}")
        observations = incomplete_observations(
            "Validated supervisor report was unavailable.",
            [str(_relative_to_project(root, validation_file)), str(_relative_to_project(root, summary_file))],
        )
        return _finalize_database(observations, warnings, errors, root, summary_file, validation_file, validated=False)

    if validation.get("passed") is not True:
        errors.append(f"Validated supervisor report did not pass: {validation_file}")
        observations = incomplete_observations(
            "Report validation did not pass, so observations were not extracted.",
            [str(_relative_to_project(root, validation_file)), str(_relative_to_project(root, summary_file))],
        )
        return _finalize_database(observations, warnings, errors, root, summary_file, validation_file, validated=False)

    summary = _read_json(summary_file)
    if summary is None:
        errors.append(f"Supervisor summary file is missing or unreadable: {summary_file}")
        observations = incomplete_observations(
            "Validated summary was unavailable.",
            [str(_relative_to_project(root, validation_file)), str(_relative_to_project(root, summary_file))],
        )
        return _finalize_database(observations, warnings, errors, root, summary_file, validation_file, validated=False)

    if summary.get("package_passed") is not True:
        errors.append("Supervisor summary package_passed is not true.")
        observations = incomplete_observations(
            "Supervisor summary was not marked as passed.",
            [str(_relative_to_project(root, validation_file)), str(_relative_to_project(root, summary_file))],
        )
        return _finalize_database(observations, warnings, errors, root, summary_file, validation_file, validated=False)

    observations = build_observations_from_summary(summary)
    _validate_supporting_files(observations, root, warnings)
    return _finalize_database(observations, warnings, errors, root, summary_file, validation_file, validated=True)


def write_observation_outputs(
    database: ObservationDatabase,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Write observations JSON, CSV, and Markdown outputs."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    outputs = [
        target / "observations.json",
        target / "observations.csv",
        target / "observations.md",
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Observation outputs already exist: {names}")

    outputs[0].write_text(json.dumps(database.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(database.observations, outputs[1])
    outputs[2].write_text(render_markdown(database), encoding="utf-8")
    return outputs


def render_markdown(database: ObservationDatabase) -> str:
    lines = [
        "# Scientific Observations",
        "",
        f"Observation count: {len(database.observations)}",
        f"Complete observations: {database.complete_count}",
        f"Incomplete observations: {database.incomplete_count}",
        "",
    ]
    categories = []
    for observation in database.observations:
        if observation.category not in categories:
            categories.append(observation.category)
    for category in categories:
        lines.extend([f"## {category}", ""])
        for observation in [item for item in database.observations if item.category == category]:
            lines.extend(
                [
                    f"### {observation.id}",
                    "",
                    f"Observation: {observation.statement}",
                    "",
                    "Evidence:",
                ]
            )
            if observation.supporting_files:
                for source_file in observation.supporting_files:
                    lines.append(f"- File: {source_file}")
            else:
                lines.append("- File: MISSING")
            if observation.supporting_metrics:
                for metric in observation.supporting_metrics:
                    units = f" {metric.get('metric_units')}" if metric.get("metric_units") else ""
                    value = metric.get("metric_value")
                    value_text = "MISSING" if value is None else _format_value(value)
                    lines.append(f"- Metric: {metric.get('metric_name')} = {value_text}{units}")
            else:
                lines.append("- Metric: MISSING")
            lines.extend(
                [
                    f"Confidence: {observation.confidence}",
                    f"Status: {observation.status}",
                ]
            )
            if observation.notes:
                lines.append(f"Notes: {observation.notes}")
            lines.append("")
    if database.warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in database.warnings)
        lines.append("")
    if database.errors:
        lines.extend(["## Errors", ""])
        lines.extend(f"- {error}" for error in database.errors)
        lines.append("")
    return "\n".join(lines)


def _finalize_database(
    observations: list[Observation],
    warnings: list[str],
    errors: list[str],
    project_root: Path,
    summary_file: Path,
    validation_file: Path,
    *,
    validated: bool,
) -> ObservationDatabase:
    for index, observation in enumerate(observations, start=1):
        observation.id = f"OBS-{index:04d}"
    metadata = {
        "engine": "Observation Engine",
        "validated_input_used": validated,
        "summary_path": str(_relative_to_project(project_root, summary_file)),
        "validation_path": str(_relative_to_project(project_root, validation_file)),
        "category_count": len({observation.category for observation in observations}),
    }
    return ObservationDatabase(observations=observations, warnings=warnings, errors=errors, metadata=metadata)


def _validate_supporting_files(observations: list[Observation], project_root: Path, warnings: list[str]) -> None:
    for observation in observations:
        missing = [source for source in observation.supporting_files if not _supporting_file_exists(project_root, source)]
        if missing:
            observation.status = "INCOMPLETE"
            observation.confidence = "Low"
            message = f"Missing supporting files for {observation.id or observation.category}: {', '.join(missing)}"
            observation.notes = f"{observation.notes} {message}".strip()
            warnings.append(message)


def _supporting_file_exists(project_root: Path, source_file: str) -> bool:
    source = Path(source_file)
    if source.is_absolute():
        return source.exists()
    return (project_root / "outputs" / source).exists() or (project_root / source).exists()


def _resolve_input(project_root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return project_root / candidate


def _relative_to_project(project_root: Path, path: Path) -> Path:
    try:
        return path.resolve().relative_to(project_root.resolve())
    except ValueError:
        return path


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _write_csv(observations: list[Observation], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OBSERVATION_FIELDS)
        writer.writeheader()
        for observation in observations:
            row = observation.to_dict()
            row["supporting_files"] = "; ".join(row.get("supporting_files", []))
            row["supporting_metrics"] = json.dumps(row.get("supporting_metrics", []), ensure_ascii=False)
            writer.writerow({field: row.get(field) for field in OBSERVATION_FIELDS})


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build factual scientific observations from validated outputs.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--validation", default=str(DEFAULT_VALIDATION_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    database = build_observation_database(
        args.project_root,
        summary_path=args.summary,
        validation_path=args.validation,
    )
    paths = write_observation_outputs(database, args.output_dir, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "observation_count": len(database.observations),
                "complete_observation_count": database.complete_count,
                "incomplete_observation_count": database.incomplete_count,
                "extraction_success": database.extraction_success,
                "outputs": [str(path) for path in paths],
                "warnings": database.warnings,
                "errors": database.errors,
            },
            indent=2,
        )
    )
    return 0 if database.extraction_success else 2


if __name__ == "__main__":
    raise SystemExit(main())
