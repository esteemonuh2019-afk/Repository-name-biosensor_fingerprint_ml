"""Deterministic output writers for Scientific Observation Engine results."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import Observation, ProvenanceRecord, ValidationIssue
from .source_loader import SupervisorSourcePayload


OUTPUT_FILENAMES: tuple[str, ...] = (
    "observations.json",
    "observations.csv",
    "observations.md",
    "observation_validation.json",
    "observation_provenance.csv",
    "observation_summary.json",
)


@dataclass(frozen=True)
class ObservationWriteResult:
    output_dir: Path
    output_paths: tuple[Path, ...]
    validation_summary: dict[str, Any]
    observation_summary: dict[str, Any]


def write_observation_outputs(
    *,
    project_root: str | Path,
    output_dir: str | Path,
    supervisor_payload: SupervisorSourcePayload,
    observations: tuple[Observation, ...],
    validation_issues: tuple[ValidationIssue, ...],
    software_version: str,
    generated_at: str,
    overwrite: bool = False,
) -> ObservationWriteResult:
    root = Path(project_root).resolve()
    target = _resolve_output_dir(root, output_dir)
    _prepare_output_dir(root, target, overwrite)

    observation_summary = build_observation_summary(supervisor_payload, observations, validation_issues)
    validation_summary = build_validation_summary(observations, validation_issues, {})

    paths = {
        "observations.json": target / "observations.json",
        "observations.csv": target / "observations.csv",
        "observations.md": target / "observations.md",
        "observation_validation.json": target / "observation_validation.json",
        "observation_provenance.csv": target / "observation_provenance.csv",
        "observation_summary.json": target / "observation_summary.json",
    }

    observations_payload = {
        "schema_version": "bsip_observation_v2",
        "software_version": software_version,
        "source_supervisor_results_directory": str(supervisor_payload.supervisor_results_dir),
        "generated_at": generated_at,
        "observations": [observation.to_dict() for observation in observations],
        "validation_summary": validation_summary,
    }
    _write_json(paths["observations.json"], observations_payload)
    _write_observations_csv(paths["observations.csv"], observations)
    paths["observations.md"].write_text(render_markdown(observations), encoding="utf-8")
    _write_provenance_csv(paths["observation_provenance.csv"], observations)
    _write_json(paths["observation_summary.json"], observation_summary)

    readability = _readability_checks(paths)
    validation_summary = build_validation_summary(observations, validation_issues, readability)
    observations_payload["validation_summary"] = validation_summary
    _write_json(paths["observations.json"], observations_payload)
    _write_json(paths["observation_validation.json"], validation_summary)
    readability = _readability_checks(paths)
    validation_summary = build_validation_summary(observations, validation_issues, readability)
    observations_payload["validation_summary"] = validation_summary
    observation_summary["validation_passed"] = validation_summary["validation_passed"]
    _write_json(paths["observations.json"], observations_payload)
    _write_json(paths["observation_validation.json"], validation_summary)
    _write_json(paths["observation_summary.json"], observation_summary)

    return ObservationWriteResult(
        output_dir=target,
        output_paths=tuple(paths[name] for name in OUTPUT_FILENAMES),
        validation_summary=validation_summary,
        observation_summary=observation_summary,
    )


def build_observation_summary(
    supervisor_payload: SupervisorSourcePayload,
    observations: tuple[Observation, ...],
    validation_issues: tuple[ValidationIssue, ...],
) -> dict[str, Any]:
    count_by_category: dict[str, int] = {}
    count_by_status: dict[str, int] = {}
    count_by_confidence: dict[str, int] = {}
    for observation in observations:
        count_by_category[observation.category.value] = count_by_category.get(observation.category.value, 0) + 1
        count_by_status[observation.status.value] = count_by_status.get(observation.status.value, 0) + 1
        count_by_confidence[observation.confidence.value] = count_by_confidence.get(observation.confidence.value, 0) + 1

    quantitative = [observation for observation in observations if observation.supporting_metrics]
    provenance_backed = [
        observation
        for observation in quantitative
        if all(metric.metric_value is None or metric.provenance_id for metric in observation.supporting_metrics)
    ]
    summary = supervisor_payload.summary
    return {
        "total_observations": len(observations),
        "count_by_category": dict(sorted(count_by_category.items())),
        "count_by_status": dict(sorted(count_by_status.items())),
        "count_by_confidence": dict(sorted(count_by_confidence.items())),
        "quantitative_observation_count": len(quantitative),
        "provenance_backed_observation_count": len(provenance_backed),
        "incomplete_observation_count": count_by_status.get("INCOMPLETE", 0),
        "source_files_loaded": list(supervisor_payload.loaded_files),
        "source_files_missing": list(supervisor_payload.missing_required_files + supervisor_payload.missing_optional_files),
        "selected_classifier": summary.get("classification_results", {}).get("selected_model", {}).get("model_name"),
        "selected_regressor": summary.get("regression_results", {}).get("selected_model", {}).get("model_name"),
        "blind_labels_available": summary.get("project_summary", {}).get("blind_prediction_context", {}).get("true_labels_included"),
        "validation_passed": not _has_blocking_issue(observations, validation_issues, {}),
    }


def build_validation_summary(
    observations: tuple[Observation, ...],
    issues: tuple[ValidationIssue, ...],
    readability: dict[str, Any],
) -> dict[str, Any]:
    critical_count = sum(1 for issue in issues if issue.severity == "CRITICAL")
    error_count = sum(1 for issue in issues if issue.severity == "ERROR")
    warning_count = sum(1 for issue in issues if issue.severity == "WARNING")
    incomplete_count = sum(1 for observation in observations if observation.status.value == "INCOMPLETE")
    duplicate_id_count = sum(1 for issue in issues if issue.code == "DUPLICATE_OBSERVATION_ID")
    missing_provenance_count = sum(1 for issue in issues if issue.code == "MISSING_PROVENANCE")
    model_coherence_count = sum(1 for issue in issues if "MODEL" in issue.code)
    blind_wording_count = sum(1 for issue in issues if issue.code == "BLIND_VALIDATION_WORDING")
    validation_passed = not _has_blocking_issue(observations, issues, readability)
    return {
        "validation_passed": validation_passed,
        "critical_issue_count": critical_count,
        "warning_count": warning_count,
        "incomplete_observation_count": incomplete_count,
        "duplicate_id_count": duplicate_id_count,
        "missing_provenance_count": missing_provenance_count,
        "model_coherence_issue_count": model_coherence_count,
        "blind_validation_wording_issue_count": blind_wording_count,
        "output_readability_checks": readability,
        "structured_validation_issues": [issue.to_dict() for issue in issues],
        "error_count": error_count,
    }


def render_markdown(observations: tuple[Observation, ...]) -> str:
    lines = ["# Scientific Observations", ""]
    categories = []
    for observation in observations:
        if observation.category.value not in categories:
            categories.append(observation.category.value)
    for category in categories:
        lines.extend([f"## {category}", ""])
        for observation in [item for item in observations if item.category.value == category]:
            provenance_ids = [
                metric.provenance_id
                for metric in observation.supporting_metrics
                if metric.provenance_id
            ]
            lines.extend(
                [
                    f"### {observation.observation_id}",
                    "",
                    f"Title: {observation.title}",
                    "",
                    f"Statement: {observation.statement}",
                    "",
                    f"Status: {observation.status.value}",
                    f"Confidence: {observation.confidence.value}",
                    f"Analysis stage: {observation.analysis_stage}",
                    "",
                    "Supporting metrics:",
                ]
            )
            if observation.supporting_metrics:
                for metric in observation.supporting_metrics:
                    value = "null" if metric.metric_value is None else _format_value(metric.metric_value)
                    units = f" {metric.units}" if metric.units else ""
                    lines.append(f"- {metric.metric_name}: {value}{units} (provenance: {metric.provenance_id or 'MISSING'})")
            else:
                lines.append("- None")
            lines.extend(["", "Supporting files:"])
            if observation.supporting_files:
                lines.extend(f"- {source_file}" for source_file in observation.supporting_files)
            else:
                lines.append("- None")
            lines.extend(["", "Provenance IDs:"])
            lines.append("- " + (", ".join(provenance_ids) if provenance_ids else "None"))
            lines.extend(["", "Limitations:"])
            if observation.limitations:
                lines.extend(f"- {limitation}" for limitation in observation.limitations)
            else:
                lines.append("- None")
            lines.append("")
    return "\n".join(lines)


def _prepare_output_dir(project_root: Path, output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and output_dir.is_file():
        raise FileExistsError(f"Output path exists as a file: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_dir}")
        _verify_safe_delete_target(project_root, output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _verify_safe_delete_target(project_root: Path, output_dir: Path) -> None:
    resolved_root = project_root.resolve()
    resolved_output = output_dir.resolve()
    if resolved_output == resolved_root:
        raise ValueError("Refusing to overwrite the project root.")
    try:
        resolved_output.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Refusing to overwrite a directory outside the project root: {output_dir}") from exc


def _resolve_output_dir(project_root: Path, output_dir: str | Path) -> Path:
    candidate = Path(output_dir)
    if candidate.is_absolute():
        return candidate.resolve()
    return (project_root / candidate).resolve()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def _write_observations_csv(path: Path, observations: tuple[Observation, ...]) -> None:
    fieldnames = [
        "observation_id",
        "category",
        "title",
        "statement",
        "status",
        "analysis_stage",
        "supporting_metrics",
        "supporting_files",
        "provenance_records",
        "confidence",
        "limitations",
        "created_at",
        "software_version",
        "source_run",
        "tags",
        "metadata",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for observation in observations:
            record = observation.to_dict()
            row = {}
            for field in fieldnames:
                value = record.get(field)
                if isinstance(value, (dict, list)):
                    row[field] = json.dumps(value, sort_keys=True, ensure_ascii=False)
                else:
                    row[field] = value
            writer.writerow(row)


def _write_provenance_csv(path: Path, observations: tuple[Observation, ...]) -> None:
    fieldnames = [
        "observation_id",
        "provenance_id",
        "source_file",
        "source_run",
        "section",
        "claim_text",
        "metric_name",
        "metric_value",
        "units",
        "model_name",
        "table_or_figure_reference",
        "support_status",
    ]
    rows: list[dict[str, Any]] = []
    seen = set()
    for observation in observations:
        for record in observation.provenance_records:
            key = (observation.observation_id, record.provenance_id)
            if key in seen:
                continue
            seen.add(key)
            row = record.to_dict()
            row["observation_id"] = observation.observation_id
            rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["observation_id"], item["provenance_id"])):
            writer.writerow({field: row.get(field) for field in fieldnames})


def _readability_checks(paths: dict[str, Path]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for name, path in paths.items():
        if not path.exists():
            checks[name] = {"readable": False, "reason": "missing"}
            continue
        try:
            if path.suffix == ".json":
                json.loads(path.read_text(encoding="utf-8"))
            elif path.suffix == ".csv":
                with path.open("r", encoding="utf-8-sig", newline="") as handle:
                    next(csv.reader(handle), None)
            else:
                path.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - readability summary must capture all failures
            checks[name] = {"readable": False, "reason": str(exc)}
        else:
            checks[name] = {"readable": True, "reason": ""}
    return checks


def _has_blocking_issue(
    observations: tuple[Observation, ...],
    issues: tuple[ValidationIssue, ...],
    readability: dict[str, Any],
) -> bool:
    if any(issue.severity in {"CRITICAL", "ERROR"} for issue in issues):
        return True
    if any(observation.status.value == "INCOMPLETE" for observation in observations):
        return True
    if readability and any(not check.get("readable") for check in readability.values()):
        return True
    return False


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
