"""Service layer for the BSIP v4.0.0 Evidence Scoring Engine."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import (
    EVIDENCE_SCORING_SOFTWARE_VERSION,
    EvidenceScoreRecord,
    EvidenceScoringRunResult,
    EvidenceScoringValidationIssue,
)
from .scorer import score_claims
from .serializers import summarize_records, write_evidence_scoring_outputs
from .validation import validate_evidence_score_records, validate_source_documents, validation_summary


REQUIRED_CLAIM_FILES: tuple[str, ...] = ("claims.json", "claim_validation.json", "claim_summary.json")
OPTIONAL_CLAIM_FILES: tuple[str, ...] = (
    "claim_dependencies.csv",
    "claim_evidence_scores.csv",
    "claim_publication_matrix.csv",
    "claims.csv",
)
REQUIRED_GRAPH_FILES: tuple[str, ...] = (
    "reasoning_graph.json",
    "reasoning_graph_validation.json",
    "reasoning_graph_summary.json",
)


@dataclass(frozen=True)
class EvidenceScoringSourcePackage:
    claims_dir: Path
    graph_dir: Path
    claims_document: dict[str, Any] = field(default_factory=dict)
    claim_validation_document: dict[str, Any] = field(default_factory=dict)
    claim_summary_document: dict[str, Any] = field(default_factory=dict)
    graph_document: dict[str, Any] = field(default_factory=dict)
    graph_validation_document: dict[str, Any] = field(default_factory=dict)
    graph_summary_document: dict[str, Any] = field(default_factory=dict)
    optional_claim_rows: dict[str, tuple[dict[str, str], ...]] = field(default_factory=dict)
    source_files_loaded: tuple[str, ...] = field(default_factory=tuple)
    source_files_missing: tuple[str, ...] = field(default_factory=tuple)
    validation_issues: tuple[EvidenceScoringValidationIssue, ...] = field(default_factory=tuple)

    @property
    def claim_schema_version(self) -> str | None:
        schema = self.claims_document.get("schema_version")
        return None if schema is None else str(schema)

    @property
    def graph_schema_version(self) -> str | None:
        schema = self.graph_document.get("schema_version")
        return None if schema is None else str(schema)

    @property
    def claims(self) -> tuple[dict[str, Any], ...]:
        records = self.claims_document.get("claims", ()) or ()
        return tuple(sorted((dict(record) for record in records), key=lambda record: str(record.get("claim_id"))))

    @property
    def source_validation_passed(self) -> bool:
        return (
            self.claim_validation_document.get("validation_passed") is True
            and int(self.claim_validation_document.get("critical_issue_count") or 0) == 0
            and self.graph_validation_document.get("validation_passed") is True
            and int(self.graph_validation_document.get("critical_issue_count") or 0) == 0
            and not self.validation_issues
        )


class EvidenceScoringEngine:
    """Evaluate evidence support for existing scientific claims."""

    def __init__(
        self,
        *,
        project_root: Path | str = ".",
        claims_dir: Path | str = "outputs/scientific_claims",
        graph_dir: Path | str = "outputs/reasoning_graph",
        output_dir: Path | str = "outputs/evidence_scoring",
        overwrite: bool = False,
        strict: bool = False,
        software_version: str = EVIDENCE_SCORING_SOFTWARE_VERSION,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.claims_dir = Path(claims_dir)
        self.graph_dir = Path(graph_dir)
        self.output_dir = Path(output_dir)
        self.overwrite = overwrite
        self.strict = strict
        self.software_version = software_version
        self._source_package: EvidenceScoringSourcePackage | None = None
        self._generated_at: str | None = None

    def load_sources(self) -> EvidenceScoringSourcePackage:
        self._source_package = load_source_package(self.project_root, self.claims_dir, self.graph_dir)
        return self._source_package

    def validate_inputs(self) -> tuple[EvidenceScoringValidationIssue, ...]:
        if self._source_package is None:
            raise RuntimeError("Sources must be loaded before validation.")
        return self._source_package.validation_issues

    def build_scores(self) -> tuple[EvidenceScoreRecord, ...]:
        if self._source_package is None:
            raise RuntimeError("Sources must be loaded before scoring.")
        source = self._source_package
        source_validated = source.source_validation_passed
        return score_claims(
            source.claims,
            source.graph_document,
            claim_validation_passed=source_validated,
            graph_validation_passed=source_validated,
            source_claim_schema_version=source.claim_schema_version,
            source_graph_schema_version=source.graph_schema_version,
            software_version=self.software_version,
        )

    def validate_scores(self, records: tuple[EvidenceScoreRecord, ...]) -> tuple[EvidenceScoringValidationIssue, ...]:
        if self._source_package is None:
            raise RuntimeError("Sources must be loaded before score validation.")
        return validate_evidence_score_records(
            records,
            source_claim_ids=tuple(str(claim.get("claim_id")) for claim in self._source_package.claims),
        )

    def write_outputs(
        self,
        records: tuple[EvidenceScoreRecord, ...],
        validation_issues: tuple[EvidenceScoringValidationIssue, ...],
    ) -> tuple[Path, ...]:
        if self._source_package is None:
            raise RuntimeError("Sources must be loaded before writing outputs.")
        return write_evidence_scoring_outputs(
            project_root=self.project_root,
            output_dir=self.output_dir,
            records=records,
            validation_issues=validation_issues,
            generated_at=self._generated_at or utc_now_iso(),
            source_validation_status=_source_validation_status(self._source_package, software_version=self.software_version),
            overwrite=self.overwrite,
        )

    def run(self) -> EvidenceScoringRunResult:
        self._generated_at = utc_now_iso()
        source = self.load_sources()
        input_issues = self.validate_inputs()
        if _has_fatal_load_issue(input_issues):
            return EvidenceScoringRunResult(
                records=tuple(),
                validation_issues=input_issues,
                output_paths=tuple(),
                metadata=_run_metadata(tuple(), input_issues, source, software_version=self.software_version),
            )
        records = self.build_scores()
        score_issues = self.validate_scores(records)
        all_issues = input_issues + score_issues
        output_paths = self.write_outputs(records, all_issues)
        return EvidenceScoringRunResult(
            records=records,
            validation_issues=all_issues,
            output_paths=output_paths,
            metadata=_run_metadata(records, all_issues, source, output_paths=output_paths, software_version=self.software_version),
        )


def load_source_package(
    project_root: Path | str,
    claims_dir: Path | str,
    graph_dir: Path | str,
) -> EvidenceScoringSourcePackage:
    root = Path(project_root).resolve()
    claim_directory = _resolve_input_directory(root, claims_dir)
    graph_directory = _resolve_input_directory(root, graph_dir)
    issues: list[EvidenceScoringValidationIssue] = []
    missing = []
    for name in REQUIRED_CLAIM_FILES:
        if not (claim_directory / name).exists():
            missing.append(f"claims:{name}")
            issues.append(_source_issue("MISSING_SOURCE_FILE", f"Required claim source file is missing: {name}", field=name))
    for name in REQUIRED_GRAPH_FILES:
        if not (graph_directory / name).exists():
            missing.append(f"graph:{name}")
            issues.append(_source_issue("MISSING_SOURCE_FILE", f"Required reasoning graph source file is missing: {name}", field=name))
    if missing:
        return EvidenceScoringSourcePackage(
            claims_dir=claim_directory,
            graph_dir=graph_directory,
            source_files_missing=tuple(missing),
            validation_issues=tuple(issues),
        )

    loaded: list[str] = []
    optional_rows: dict[str, tuple[dict[str, str], ...]] = {}
    try:
        claims_document = _read_json(claim_directory / "claims.json")
        loaded.append(str(claim_directory / "claims.json"))
        claim_validation = _read_json(claim_directory / "claim_validation.json")
        loaded.append(str(claim_directory / "claim_validation.json"))
        claim_summary = _read_json(claim_directory / "claim_summary.json")
        loaded.append(str(claim_directory / "claim_summary.json"))
        graph_document = _read_json(graph_directory / "reasoning_graph.json")
        loaded.append(str(graph_directory / "reasoning_graph.json"))
        graph_validation = _read_json(graph_directory / "reasoning_graph_validation.json")
        loaded.append(str(graph_directory / "reasoning_graph_validation.json"))
        graph_summary = _read_json(graph_directory / "reasoning_graph_summary.json")
        loaded.append(str(graph_directory / "reasoning_graph_summary.json"))
        for name in OPTIONAL_CLAIM_FILES:
            path = claim_directory / name
            if path.exists():
                optional_rows[name] = _read_csv(path)
                loaded.append(str(path))
    except (OSError, json.JSONDecodeError, csv.Error, TypeError, ValueError) as exc:
        issues.append(_source_issue("UNREADABLE_SOURCE_FILE", f"Evidence scoring source package could not be read: {exc}", field="source_package"))
        return EvidenceScoringSourcePackage(
            claims_dir=claim_directory,
            graph_dir=graph_directory,
            source_files_loaded=tuple(loaded),
            validation_issues=tuple(issues),
        )
    source_issues = validate_source_documents(
        claims_document=claims_document,
        claim_validation_document=claim_validation,
        graph_document=graph_document,
        graph_validation_document=graph_validation,
    )
    issues.extend(source_issues)
    return EvidenceScoringSourcePackage(
        claims_dir=claim_directory,
        graph_dir=graph_directory,
        claims_document=claims_document,
        claim_validation_document=claim_validation,
        claim_summary_document=claim_summary,
        graph_document=graph_document,
        graph_validation_document=graph_validation,
        graph_summary_document=graph_summary,
        optional_claim_rows=optional_rows,
        source_files_loaded=tuple(loaded),
        source_files_missing=tuple(missing),
        validation_issues=tuple(issues),
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _source_validation_status(source: EvidenceScoringSourcePackage, *, software_version: str) -> dict[str, Any]:
    return {
        "claims_loaded": len(source.claims),
        "claim_schema_version": source.claim_schema_version,
        "graph_schema_version": source.graph_schema_version,
        "claim_validation_passed": source.claim_validation_document.get("validation_passed") is True,
        "graph_validation_passed": source.graph_validation_document.get("validation_passed") is True,
        "source_validation_passed": source.source_validation_passed,
        "source_files_loaded": list(source.source_files_loaded),
        "source_files_missing": list(source.source_files_missing),
        "software_version": software_version,
    }


def _run_metadata(
    records: tuple[EvidenceScoreRecord, ...],
    issues: tuple[EvidenceScoringValidationIssue, ...],
    source: EvidenceScoringSourcePackage,
    *,
    output_paths: tuple[Path, ...] = tuple(),
    software_version: str,
) -> dict[str, Any]:
    summary_validation = validation_summary(records, issues, output_readability_checks={})
    summary = summarize_records(
        records,
        validation_passed=summary_validation["validation_passed"],
        source_validation_status=_source_validation_status(source, software_version=software_version),
    )
    return {
        "validation_passed": summary_validation["validation_passed"],
        "critical_issue_count": summary_validation["critical_issue_count"],
        "warning_count": summary_validation["warning_count"],
        "claims_loaded": len(source.claims),
        "claims_scored": len(records),
        "claims_withheld": sum(1 for record in records if record.is_withheld),
        "mean_evidence_score": summary["mean_normalized_score"],
        "output_paths": [str(path) for path in output_paths],
        "summary": summary,
        "validation_summary": summary_validation,
    }


def _has_fatal_load_issue(issues: tuple[EvidenceScoringValidationIssue, ...]) -> bool:
    return any(issue.code in {"MISSING_SOURCE_FILE", "UNREADABLE_SOURCE_FILE"} for issue in issues)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return payload


def _read_csv(path: Path) -> tuple[dict[str, str], ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return tuple(dict(row) for row in csv.DictReader(handle))


def _resolve_input_directory(project_root: Path, directory: Path | str) -> Path:
    path = Path(directory)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _source_issue(code: str, message: str, *, field: str) -> EvidenceScoringValidationIssue:
    return EvidenceScoringValidationIssue(
        code=code,
        severity="CRITICAL",
        message=message,
        field=field,
    )


EvidenceScoringService = EvidenceScoringEngine
