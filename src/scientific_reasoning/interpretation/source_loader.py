"""Load validated Observation Engine output packages for interpretation."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.scientific_reasoning.observation import (
    Observation,
    ProvenanceRecord,
    SupportingMetric,
)

from .models import InterpretationValidationIssue


REQUIRED_OBSERVATION_FILES: tuple[str, ...] = (
    "observations.json",
    "observation_validation.json",
    "observation_summary.json",
    "observation_provenance.csv",
)


@dataclass(frozen=True)
class ObservationSourcePackage:
    observations_dir: Path
    observations: tuple[Observation, ...] = field(default_factory=tuple)
    observations_document: dict[str, Any] = field(default_factory=dict)
    validation_document: dict[str, Any] = field(default_factory=dict)
    summary_document: dict[str, Any] = field(default_factory=dict)
    provenance_rows: tuple[dict[str, str], ...] = field(default_factory=tuple)
    source_files_loaded: tuple[str, ...] = field(default_factory=tuple)
    source_files_missing: tuple[str, ...] = field(default_factory=tuple)
    validation_issues: tuple[InterpretationValidationIssue, ...] = field(default_factory=tuple)

    @property
    def schema_version(self) -> str | None:
        schema = self.observations_document.get("schema_version")
        return None if schema is None else str(schema)

    @property
    def validation_passed(self) -> bool:
        return not any(issue.severity.value == "CRITICAL" for issue in self.validation_issues)


def load_observation_package(
    project_root: Path | str,
    observations_dir: Path | str,
) -> ObservationSourcePackage:
    """Read a validated Observation Engine output package.

    The loader only reads the Observation Engine package files named in the
    Interpretation Engine contract. It does not search for or parse raw
    analysis outputs.
    """

    root = Path(project_root).resolve()
    directory = _resolve_input_directory(root, observations_dir)
    issues: list[InterpretationValidationIssue] = []
    missing = tuple(name for name in REQUIRED_OBSERVATION_FILES if not (directory / name).exists())
    if missing:
        for name in missing:
            issues.append(
                _source_issue(
                    "MISSING_OBSERVATION_SOURCE_FILE",
                    f"Required Observation Engine output is missing: {name}",
                    field="source_file",
                )
            )
        return ObservationSourcePackage(
            observations_dir=directory,
            source_files_missing=missing,
            validation_issues=tuple(issues),
        )

    observations_document: dict[str, Any] = {}
    validation_document: dict[str, Any] = {}
    summary_document: dict[str, Any] = {}
    provenance_rows: tuple[dict[str, str], ...] = tuple()
    observations: tuple[Observation, ...] = tuple()
    loaded: list[str] = []

    try:
        observations_document = _read_json(directory / "observations.json")
        loaded.append("observations.json")
        validation_document = _read_json(directory / "observation_validation.json")
        loaded.append("observation_validation.json")
        summary_document = _read_json(directory / "observation_summary.json")
        loaded.append("observation_summary.json")
        provenance_rows = _read_csv(directory / "observation_provenance.csv")
        loaded.append("observation_provenance.csv")
        observations = _parse_observations(observations_document)
    except (OSError, json.JSONDecodeError, csv.Error, TypeError, ValueError) as exc:
        issues.append(
            _source_issue(
                "UNREADABLE_OBSERVATION_PACKAGE",
                f"Observation Engine output package could not be read: {exc}",
                field="observations_dir",
            )
        )

    issues.extend(_package_validation_issues(validation_document))
    return ObservationSourcePackage(
        observations_dir=directory,
        observations=observations,
        observations_document=observations_document,
        validation_document=validation_document,
        summary_document=summary_document,
        provenance_rows=provenance_rows,
        source_files_loaded=tuple(loaded),
        source_files_missing=missing,
        validation_issues=tuple(issues),
    )


def _resolve_input_directory(project_root: Path, observations_dir: Path | str) -> Path:
    path = Path(observations_dir)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return payload


def _read_csv(path: Path) -> tuple[dict[str, str], ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return tuple(dict(row) for row in reader)


def _parse_observations(document: dict[str, Any]) -> tuple[Observation, ...]:
    records = document.get("observations", document)
    if isinstance(records, dict):
        records = records.get("observations", ())
    if not isinstance(records, list):
        raise TypeError("observations.json must contain an observations list")
    observations = tuple(_observation_from_record(record) for record in records)
    return tuple(sorted(observations, key=lambda observation: observation.observation_id))


def _observation_from_record(record: dict[str, Any]) -> Observation:
    metrics = tuple(SupportingMetric(**metric) for metric in record.get("supporting_metrics", ()))
    provenance = tuple(ProvenanceRecord(**item) for item in record.get("provenance_records", ()))
    return Observation(
        observation_id=record["observation_id"],
        category=record["category"],
        title=record["title"],
        statement=record["statement"],
        status=record["status"],
        analysis_stage=record["analysis_stage"],
        supporting_metrics=metrics,
        supporting_files=tuple(record.get("supporting_files", ())),
        provenance_records=provenance,
        confidence=record["confidence"],
        limitations=tuple(record.get("limitations", ())),
        created_at=record.get("created_at"),
        software_version=record.get("software_version", "BSIP-2.0"),
        source_run=record.get("source_run"),
        tags=tuple(record.get("tags", ())),
        metadata=record.get("metadata", {}),
    )


def _package_validation_issues(document: dict[str, Any]) -> tuple[InterpretationValidationIssue, ...]:
    if not document:
        return tuple()
    issues: list[InterpretationValidationIssue] = []
    critical_count = int(document.get("critical_issue_count") or 0)
    if critical_count > 0:
        issues.append(
            _source_issue(
                "CRITICALLY_INVALID_OBSERVATION_PACKAGE",
                f"Observation package reports {critical_count} critical validation issue(s).",
                field="observation_validation.json",
            )
        )
    if document.get("validation_passed") is False:
        issues.append(
            _source_issue(
                "OBSERVATION_PACKAGE_VALIDATION_FAILED",
                "Observation package validation_passed is false.",
                field="observation_validation.json",
            )
        )
    for issue in document.get("structured_validation_issues", ()) or ():
        severity = str(issue.get("severity", "")).upper()
        if severity == "CRITICAL":
            issues.append(
                _source_issue(
                    "CRITICAL_OBSERVATION_VALIDATION_ISSUE",
                    str(issue.get("message", "Observation package contains a critical validation issue.")),
                    field=str(issue.get("field") or "observation_validation.json"),
                    observation_id=issue.get("observation_id"),
                )
            )
    return tuple(issues)


def _source_issue(
    code: str,
    message: str,
    *,
    field: str | None,
    observation_id: str | None = None,
) -> InterpretationValidationIssue:
    return InterpretationValidationIssue(
        code=code,
        severity="CRITICAL",
        message=message,
        interpretation_id=None,
        field=field,
        observation_id=observation_id,
        rule_id=None,
    )
