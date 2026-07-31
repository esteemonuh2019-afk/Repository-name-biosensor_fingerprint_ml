"""Concrete BSIP v2.2.0 Scientific Hypothesis Engine."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.scientific_reasoning.interpretation import (
    Interpretation,
    InterpretationEvidenceLink,
    validate_interpretations,
)

from .models import Hypothesis, HypothesisRunResult, HypothesisValidationIssue, utc_now_iso
from .rules import build_hypotheses
from .validators import validate_hypotheses
from .writers import summarize_hypotheses, summarize_validation, write_hypothesis_outputs


DEFAULT_SOFTWARE_VERSION = "BSIP-2.2.0-hypothesis-engine"
HYPOTHESIS_SCHEMA_VERSION = "BSIP-2.2.0"

REQUIRED_INTERPRETATION_FILES: tuple[str, ...] = (
    "interpretations.json",
    "interpretation_validation.json",
    "interpretation_summary.json",
    "interpretation_dependencies.csv",
)


@dataclass(frozen=True)
class InterpretationSourcePackage:
    interpretations_dir: Path
    interpretations: tuple[Interpretation, ...] = field(default_factory=tuple)
    interpretations_document: dict[str, Any] = field(default_factory=dict)
    validation_document: dict[str, Any] = field(default_factory=dict)
    summary_document: dict[str, Any] = field(default_factory=dict)
    dependency_rows: tuple[dict[str, str], ...] = field(default_factory=tuple)
    source_files_loaded: tuple[str, ...] = field(default_factory=tuple)
    source_files_missing: tuple[str, ...] = field(default_factory=tuple)
    validation_issues: tuple[HypothesisValidationIssue, ...] = field(default_factory=tuple)

    @property
    def schema_version(self) -> str | None:
        schema = self.interpretations_document.get("schema_version")
        return None if schema is None else str(schema)


class HypothesisEngine:
    """Generate explicit, testable hypotheses from validated interpretations."""

    def __init__(
        self,
        *,
        project_root: Path | str = ".",
        interpretations_dir: Path | str = "outputs/scientific_interpretations",
        output_dir: Path | str = "outputs/scientific_hypotheses",
        overwrite: bool = False,
        software_version: str = DEFAULT_SOFTWARE_VERSION,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.interpretations_dir = Path(interpretations_dir)
        self.output_dir = Path(output_dir)
        self.overwrite = overwrite
        self.software_version = software_version
        self._source_package: InterpretationSourcePackage | None = None
        self._generated_at: str | None = None

    def load_interpretations(self) -> tuple[Interpretation, ...]:
        self._source_package = load_interpretation_package(self.project_root, self.interpretations_dir)
        return self._source_package.interpretations

    def validate_input_interpretations(
        self,
        interpretations: tuple[Interpretation, ...],
    ) -> tuple[HypothesisValidationIssue, ...]:
        issues: list[HypothesisValidationIssue] = []
        if self._source_package is not None:
            issues.extend(self._source_package.validation_issues)
        issues.extend(_convert_interpretation_issues(validate_interpretations(interpretations)))
        return tuple(issues)

    def build_hypotheses(self, interpretations: tuple[Interpretation, ...]) -> tuple[Hypothesis, ...]:
        package = self._source_package
        self._generated_at = self._generated_at or utc_now_iso()
        return build_hypotheses(
            interpretations,
            software_version=self.software_version,
            source_interpretation_schema_version=None if package is None else package.schema_version,
            created_at=self._generated_at,
            metadata={
                "interpretation_validation_passed": None
                if package is None
                else package.validation_document.get("validation_passed"),
                "source_interpretation_count": len(interpretations),
            },
        )

    def validate_hypotheses(
        self,
        hypotheses: tuple[Hypothesis, ...],
        interpretations: tuple[Interpretation, ...],
    ) -> tuple[HypothesisValidationIssue, ...]:
        return validate_hypotheses(hypotheses, interpretations)

    def write_outputs(self, hypotheses: tuple[Hypothesis, ...]) -> tuple[Path, ...]:
        if self._source_package is None:
            raise RuntimeError("Interpretation package must be loaded before writing outputs.")
        validation_issues = validate_hypotheses(hypotheses, self._source_package.interpretations)
        return write_hypothesis_outputs(
            project_root=self.project_root,
            output_dir=self.output_dir,
            hypotheses=hypotheses,
            validation_issues=validation_issues,
            schema_version=HYPOTHESIS_SCHEMA_VERSION,
            software_version=self.software_version,
            source_interpretation_dir=self._source_package.interpretations_dir,
            generated_at=self._generated_at or utc_now_iso(),
            source_interpretations_loaded=tuple(
                interpretation.interpretation_id for interpretation in self._source_package.interpretations
            ),
            source_interpretations_missing=self._source_package.source_files_missing,
            overwrite=self.overwrite,
        )

    def run(self) -> HypothesisRunResult:
        self._generated_at = utc_now_iso()
        interpretations = self.load_interpretations()
        input_issues = self.validate_input_interpretations(interpretations)
        if _has_critical_issues(input_issues):
            return HypothesisRunResult(
                hypotheses=tuple(),
                validation_issues=input_issues,
                output_paths=tuple(),
                metadata=_run_metadata(
                    interpretations=interpretations,
                    hypotheses=tuple(),
                    validation_issues=input_issues,
                    package=self._source_package,
                    output_paths=tuple(),
                ),
            )

        hypotheses = self.build_hypotheses(interpretations)
        hypothesis_issues = self.validate_hypotheses(hypotheses, interpretations)
        all_issues = input_issues + hypothesis_issues
        if _has_critical_issues(hypothesis_issues):
            return HypothesisRunResult(
                hypotheses=hypotheses,
                validation_issues=all_issues,
                output_paths=tuple(),
                metadata=_run_metadata(
                    interpretations=interpretations,
                    hypotheses=hypotheses,
                    validation_issues=all_issues,
                    package=self._source_package,
                    output_paths=tuple(),
                ),
            )

        output_paths = write_hypothesis_outputs(
            project_root=self.project_root,
            output_dir=self.output_dir,
            hypotheses=hypotheses,
            validation_issues=all_issues,
            schema_version=HYPOTHESIS_SCHEMA_VERSION,
            software_version=self.software_version,
            source_interpretation_dir=self._source_package.interpretations_dir
            if self._source_package
            else self.interpretations_dir,
            generated_at=self._generated_at,
            source_interpretations_loaded=tuple(interpretation.interpretation_id for interpretation in interpretations),
            source_interpretations_missing=self._source_package.source_files_missing if self._source_package else tuple(),
            overwrite=self.overwrite,
        )
        return HypothesisRunResult(
            hypotheses=hypotheses,
            validation_issues=all_issues,
            output_paths=output_paths,
            metadata=_run_metadata(
                interpretations=interpretations,
                hypotheses=hypotheses,
                validation_issues=all_issues,
                package=self._source_package,
                output_paths=output_paths,
            ),
        )


ScientificHypothesisEngine = HypothesisEngine


def load_interpretation_package(
    project_root: Path | str,
    interpretations_dir: Path | str,
) -> InterpretationSourcePackage:
    root = Path(project_root).resolve()
    directory = _resolve_input_directory(root, interpretations_dir)
    issues: list[HypothesisValidationIssue] = []
    missing = tuple(name for name in REQUIRED_INTERPRETATION_FILES if not (directory / name).exists())
    if missing:
        for name in missing:
            issues.append(
                _source_issue(
                    "MISSING_INTERPRETATION_SOURCE_FILE",
                    f"Required Interpretation Engine output is missing: {name}",
                    field="source_file",
                )
            )
        return InterpretationSourcePackage(
            interpretations_dir=directory,
            source_files_missing=missing,
            validation_issues=tuple(issues),
        )

    interpretations_document: dict[str, Any] = {}
    validation_document: dict[str, Any] = {}
    summary_document: dict[str, Any] = {}
    dependency_rows: tuple[dict[str, str], ...] = tuple()
    interpretations: tuple[Interpretation, ...] = tuple()
    loaded: list[str] = []

    try:
        interpretations_document = _read_json(directory / "interpretations.json")
        loaded.append("interpretations.json")
        validation_document = _read_json(directory / "interpretation_validation.json")
        loaded.append("interpretation_validation.json")
        summary_document = _read_json(directory / "interpretation_summary.json")
        loaded.append("interpretation_summary.json")
        dependency_rows = _read_csv(directory / "interpretation_dependencies.csv")
        loaded.append("interpretation_dependencies.csv")
        interpretations = _parse_interpretations(interpretations_document)
    except (OSError, json.JSONDecodeError, csv.Error, TypeError, ValueError) as exc:
        issues.append(
            _source_issue(
                "UNREADABLE_INTERPRETATION_PACKAGE",
                f"Interpretation Engine output package could not be read: {exc}",
                field="interpretations_dir",
            )
        )

    issues.extend(_package_validation_issues(validation_document))
    return InterpretationSourcePackage(
        interpretations_dir=directory,
        interpretations=interpretations,
        interpretations_document=interpretations_document,
        validation_document=validation_document,
        summary_document=summary_document,
        dependency_rows=dependency_rows,
        source_files_loaded=tuple(loaded),
        source_files_missing=missing,
        validation_issues=tuple(issues),
    )


def _parse_interpretations(document: dict[str, Any]) -> tuple[Interpretation, ...]:
    records = document.get("interpretations", document)
    if isinstance(records, dict):
        records = records.get("interpretations", ())
    if not isinstance(records, list):
        raise TypeError("interpretations.json must contain an interpretations list")
    interpretations = tuple(_interpretation_from_record(record) for record in records)
    return tuple(sorted(interpretations, key=lambda interpretation: interpretation.interpretation_id))


def _interpretation_from_record(record: dict[str, Any]) -> Interpretation:
    evidence_summary = tuple(
        InterpretationEvidenceLink(**link)
        for link in record.get("evidence_summary", ())
    )
    return Interpretation(
        interpretation_id=record["interpretation_id"],
        category=record["category"],
        title=record["title"],
        claim=record["claim"],
        status=record["status"],
        confidence=record["confidence"],
        supporting_observation_ids=tuple(record.get("supporting_observation_ids", ())),
        contradicting_observation_ids=tuple(record.get("contradicting_observation_ids", ())),
        assumptions=tuple(record.get("assumptions", ())),
        limitations=tuple(record.get("limitations", ())),
        evidence_summary=evidence_summary,
        reasoning_rule_ids=tuple(record.get("reasoning_rule_ids", ())),
        created_at=record.get("created_at"),
        software_version=record.get("software_version", "BSIP-2.1.0"),
        source_observation_schema_version=record.get("source_observation_schema_version"),
        tags=tuple(record.get("tags", ())),
        metadata=record.get("metadata", {}),
    )


def _package_validation_issues(document: dict[str, Any]) -> tuple[HypothesisValidationIssue, ...]:
    if not document:
        return tuple()
    issues: list[HypothesisValidationIssue] = []
    critical_count = int(document.get("critical_issue_count") or 0)
    if critical_count > 0:
        issues.append(
            _source_issue(
                "CRITICALLY_INVALID_INTERPRETATION_PACKAGE",
                f"Interpretation package reports {critical_count} critical validation issue(s).",
                field="interpretation_validation.json",
            )
        )
    if document.get("validation_passed") is False:
        issues.append(
            _source_issue(
                "INTERPRETATION_PACKAGE_VALIDATION_FAILED",
                "Interpretation package validation_passed is false.",
                field="interpretation_validation.json",
            )
        )
    for issue in document.get("structured_validation_issues", ()) or ():
        if str(issue.get("severity", "")).upper() == "CRITICAL":
            issues.append(
                _source_issue(
                    "CRITICAL_INTERPRETATION_VALIDATION_ISSUE",
                    str(issue.get("message", "Interpretation package contains a critical validation issue.")),
                    field=str(issue.get("field") or "interpretation_validation.json"),
                    interpretation_id=issue.get("interpretation_id"),
                )
            )
    return tuple(issues)


def _convert_interpretation_issues(interpretation_issues) -> tuple[HypothesisValidationIssue, ...]:
    converted = []
    for issue in interpretation_issues:
        severity = "WARNING" if str(issue.severity).upper() == "WARNING" else "CRITICAL"
        converted.append(
            HypothesisValidationIssue(
                code=f"INTERPRETATION_{issue.code}",
                severity=severity,
                message=issue.message,
                hypothesis_id=None,
                field=issue.field,
                interpretation_id=issue.interpretation_id,
                rule_id=issue.rule_id,
            )
        )
    return tuple(converted)


def _has_critical_issues(issues: tuple[HypothesisValidationIssue, ...]) -> bool:
    return any(issue.severity.value == "CRITICAL" for issue in issues)


def _run_metadata(
    *,
    interpretations: tuple[Interpretation, ...],
    hypotheses: tuple[Hypothesis, ...],
    validation_issues: tuple[HypothesisValidationIssue, ...],
    package: InterpretationSourcePackage | None,
    output_paths: tuple[Path, ...],
) -> dict[str, Any]:
    validation_summary = summarize_validation(validation_issues, output_readability_checks={})
    hypothesis_summary = summarize_hypotheses(
        hypotheses,
        source_interpretations_loaded=tuple(interpretation.interpretation_id for interpretation in interpretations),
        source_interpretations_missing=tuple() if package is None else package.source_files_missing,
        validation_passed=validation_summary["validation_passed"],
    )
    return {
        "validation_passed": validation_summary["validation_passed"],
        "critical_issue_count": validation_summary["critical_issue_count"],
        "warning_count": validation_summary["warning_count"],
        "hypothesis_count": len(hypotheses),
        "competing_hypothesis_count": hypothesis_summary["competing_count"],
        "source_interpretation_count": len(interpretations),
        "source_interpretations_dir": None if package is None else str(package.interpretations_dir),
        "output_paths": [str(path) for path in output_paths],
        "hypothesis_summary": hypothesis_summary,
        "validation_summary": validation_summary,
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return payload


def _read_csv(path: Path) -> tuple[dict[str, str], ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return tuple(dict(row) for row in reader)


def _resolve_input_directory(project_root: Path, interpretations_dir: Path | str) -> Path:
    path = Path(interpretations_dir)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _source_issue(
    code: str,
    message: str,
    *,
    field: str | None,
    interpretation_id: str | None = None,
) -> HypothesisValidationIssue:
    return HypothesisValidationIssue(
        code=code,
        severity="CRITICAL",
        message=message,
        hypothesis_id=None,
        field=field,
        interpretation_id=interpretation_id,
        rule_id=None,
    )
