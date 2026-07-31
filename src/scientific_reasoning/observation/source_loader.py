"""Strict source loading for the production Scientific Observation Engine."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import ProvenanceRecord, ValidationIssue


REQUIRED_SOURCE_FILES: tuple[str, ...] = (
    "supervisor_results_summary.json",
    "provenance_index.csv",
    "report_validation.json",
)

OPTIONAL_SOURCE_FILES: tuple[str, ...] = (
    "selected_tables.csv",
    "selected_figures.csv",
)


@dataclass(frozen=True)
class SupervisorSourcePayload:
    """Loaded supervisor-results source payloads."""

    project_root: Path
    supervisor_results_dir: Path
    summary: dict[str, Any] = field(default_factory=dict)
    provenance_rows: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    provenance_records: tuple[ProvenanceRecord, ...] = field(default_factory=tuple)
    report_validation: dict[str, Any] = field(default_factory=dict)
    selected_tables: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    selected_figures: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    loaded_files: tuple[str, ...] = field(default_factory=tuple)
    missing_required_files: tuple[str, ...] = field(default_factory=tuple)
    missing_optional_files: tuple[str, ...] = field(default_factory=tuple)
    validation_issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def has_critical_issues(self) -> bool:
        return any(issue.severity == "CRITICAL" for issue in self.validation_issues)


def load_supervisor_sources(
    project_root: str | Path,
    supervisor_results_dir: str | Path,
) -> SupervisorSourcePayload:
    """Load only files in the supplied supervisor-results directory."""

    root = Path(project_root).resolve()
    source_dir = _resolve_under_project(root, supervisor_results_dir)
    loaded_files: list[str] = []
    missing_required: list[str] = []
    missing_optional: list[str] = []
    issues: list[ValidationIssue] = []
    warnings: list[str] = []

    if not source_dir.exists() or not source_dir.is_dir():
        issue = ValidationIssue(
            code="SUPERVISOR_RESULTS_DIR_MISSING",
            severity="CRITICAL",
            message=f"Supervisor-results directory is missing: {source_dir}",
            observation_id=None,
            field="supervisor_results_dir",
            source_file=str(source_dir),
        )
        return SupervisorSourcePayload(
            project_root=root,
            supervisor_results_dir=source_dir,
            missing_required_files=REQUIRED_SOURCE_FILES,
            validation_issues=(issue,),
        )

    for filename in REQUIRED_SOURCE_FILES:
        path = source_dir / filename
        if not path.exists():
            missing_required.append(filename)
            issues.append(
                ValidationIssue(
                    code="REQUIRED_SOURCE_MISSING",
                    severity="CRITICAL",
                    message=f"Required supervisor-results source is missing: {filename}",
                    observation_id=None,
                    field="source_file",
                    source_file=str(path),
                )
            )
    for filename in OPTIONAL_SOURCE_FILES:
        path = source_dir / filename
        if not path.exists():
            missing_optional.append(filename)
            warnings.append(f"Optional supervisor-results source is missing: {filename}")

    summary: dict[str, Any] = {}
    report_validation: dict[str, Any] = {}
    provenance_rows: list[dict[str, Any]] = []
    selected_tables: list[dict[str, Any]] = []
    selected_figures: list[dict[str, Any]] = []

    if "supervisor_results_summary.json" not in missing_required:
        summary, issue = _read_json(source_dir / "supervisor_results_summary.json", "summary")
        if issue:
            issues.append(issue)
        else:
            loaded_files.append("supervisor_results_summary.json")
    if "report_validation.json" not in missing_required:
        report_validation, issue = _read_json(source_dir / "report_validation.json", "report_validation")
        if issue:
            issues.append(issue)
        else:
            loaded_files.append("report_validation.json")
    if "provenance_index.csv" not in missing_required:
        provenance_rows, issue = _read_csv(source_dir / "provenance_index.csv", "provenance_index")
        if issue:
            issues.append(issue)
        else:
            loaded_files.append("provenance_index.csv")
    if "selected_tables.csv" not in missing_optional:
        selected_tables, issue = _read_csv(source_dir / "selected_tables.csv", "selected_tables")
        if issue:
            warnings.append(issue.message)
        else:
            loaded_files.append("selected_tables.csv")
    if "selected_figures.csv" not in missing_optional:
        selected_figures, issue = _read_csv(source_dir / "selected_figures.csv", "selected_figures")
        if issue:
            warnings.append(issue.message)
        else:
            loaded_files.append("selected_figures.csv")

    provenance_records = tuple(_provenance_record_from_row(row) for row in provenance_rows)
    if report_validation and report_validation.get("passed") is not True:
        issues.append(
            ValidationIssue(
                code="SUPERVISOR_REPORT_VALIDATION_FAILED",
                severity="CRITICAL",
                message="Supervisor report validation did not pass.",
                observation_id=None,
                field="report_validation.passed",
                source_file=str(source_dir / "report_validation.json"),
            )
        )
    if summary and summary.get("package_passed") is not True:
        issues.append(
            ValidationIssue(
                code="SUPERVISOR_PACKAGE_NOT_PASSED",
                severity="CRITICAL",
                message="Supervisor summary package_passed is not true.",
                observation_id=None,
                field="summary.package_passed",
                source_file=str(source_dir / "supervisor_results_summary.json"),
            )
        )

    return SupervisorSourcePayload(
        project_root=root,
        supervisor_results_dir=source_dir,
        summary=summary,
        provenance_rows=tuple(provenance_rows),
        provenance_records=provenance_records,
        report_validation=report_validation,
        selected_tables=tuple(selected_tables),
        selected_figures=tuple(selected_figures),
        loaded_files=tuple(loaded_files),
        missing_required_files=tuple(missing_required),
        missing_optional_files=tuple(missing_optional),
        validation_issues=tuple(issues),
        warnings=tuple(warnings),
    )


def _resolve_under_project(project_root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (project_root / candidate).resolve()


def _read_json(path: Path, source_name: str) -> tuple[dict[str, Any], ValidationIssue | None]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return {}, ValidationIssue(
            code="SOURCE_JSON_UNREADABLE",
            severity="CRITICAL",
            message=f"Unable to read {source_name} JSON: {exc}",
            observation_id=None,
            field=source_name,
            source_file=str(path),
        )
    if not isinstance(payload, dict):
        return {}, ValidationIssue(
            code="SOURCE_JSON_NOT_OBJECT",
            severity="CRITICAL",
            message=f"{source_name} JSON must contain an object.",
            observation_id=None,
            field=source_name,
            source_file=str(path),
        )
    return payload, None


def _read_csv(path: Path, source_name: str) -> tuple[list[dict[str, Any]], ValidationIssue | None]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        return [], ValidationIssue(
            code="SOURCE_CSV_UNREADABLE",
            severity="CRITICAL",
            message=f"Unable to read {source_name} CSV: {exc}",
            observation_id=None,
            field=source_name,
            source_file=str(path),
        )
    return rows, None


def _provenance_record_from_row(row: dict[str, Any]) -> ProvenanceRecord:
    reference = row.get("table_or_figure_reference") or row.get("table_reference") or row.get("figure_reference") or None
    return ProvenanceRecord(
        provenance_id=row.get("provenance_id", ""),
        source_file=row.get("source_file") or None,
        source_run=row.get("source_run") or None,
        section=row.get("section") or None,
        claim_text=row.get("claim_text") or row.get("claim") or None,
        metric_name=row.get("metric_name") or None,
        metric_value=parse_scalar(row.get("metric_value")),
        units=row.get("units") or row.get("metric_units") or None,
        model_name=row.get("model_name") or None,
        table_or_figure_reference=reference,
        support_status=row.get("support_status") or row.get("status") or "SUPPORTED",
    )


def parse_scalar(value: Any) -> Any:
    """Parse common CSV scalar strings while preserving missing values as None."""

    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return value
    text = str(value).strip()
    if text == "":
        return None
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer() and "." not in text and "e" not in lowered:
        return int(number)
    return number
