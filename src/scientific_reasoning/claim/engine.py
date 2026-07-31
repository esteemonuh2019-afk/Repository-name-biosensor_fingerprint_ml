"""Concrete BSIP v3.2.0 Claim Engine."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import (
    CLAIM_SCHEMA_VERSION,
    DEFAULT_CLAIM_SOFTWARE_VERSION,
    ClaimRunResult,
    ClaimValidationIssue,
    ScientificClaim,
    utc_now_iso,
)
from .rules import build_claims_from_sources
from .validators import validate_claims
from .writers import summarize_claims, summarize_validation, write_claim_outputs


REQUIRED_HYPOTHESIS_FILES: tuple[str, ...] = (
    "hypotheses.json",
    "hypothesis_validation.json",
    "hypothesis_summary.json",
    "hypothesis_dependencies.csv",
    "hypothesis_competition_map.csv",
)

REQUIRED_GRAPH_FILES: tuple[str, ...] = (
    "reasoning_graph.json",
    "reasoning_graph_validation.json",
    "reasoning_graph_summary.json",
)


@dataclass(frozen=True)
class HypothesisClaimSourcePackage:
    hypotheses_dir: Path
    hypotheses: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    hypotheses_document: dict[str, Any] = field(default_factory=dict)
    validation_document: dict[str, Any] = field(default_factory=dict)
    summary_document: dict[str, Any] = field(default_factory=dict)
    dependency_rows: tuple[dict[str, str], ...] = field(default_factory=tuple)
    competition_rows: tuple[dict[str, str], ...] = field(default_factory=tuple)
    source_files_loaded: tuple[str, ...] = field(default_factory=tuple)
    source_files_missing: tuple[str, ...] = field(default_factory=tuple)
    validation_issues: tuple[ClaimValidationIssue, ...] = field(default_factory=tuple)

    @property
    def schema_version(self) -> str | None:
        schema = self.hypotheses_document.get("schema_version")
        return None if schema is None else str(schema)

    @property
    def validation_passed(self) -> bool:
        return bool(self.validation_document.get("validation_passed")) and int(self.validation_document.get("critical_issue_count") or 0) == 0


@dataclass(frozen=True)
class ReasoningGraphClaimSourcePackage:
    reasoning_graph_dir: Path
    graph_document: dict[str, Any] = field(default_factory=dict)
    validation_document: dict[str, Any] = field(default_factory=dict)
    summary_document: dict[str, Any] = field(default_factory=dict)
    source_files_loaded: tuple[str, ...] = field(default_factory=tuple)
    source_files_missing: tuple[str, ...] = field(default_factory=tuple)
    validation_issues: tuple[ClaimValidationIssue, ...] = field(default_factory=tuple)

    @property
    def schema_version(self) -> str | None:
        schema = self.graph_document.get("schema_version")
        return None if schema is None else str(schema)

    @property
    def validation_passed(self) -> bool:
        return bool(self.validation_document.get("validation_passed")) and int(self.validation_document.get("critical_issue_count") or 0) == 0


class ClaimEngine:
    """Transform validated hypotheses into explicit evidence-bounded claims."""

    def __init__(
        self,
        *,
        project_root: Path | str = ".",
        hypotheses_dir: Path | str = "outputs/scientific_hypotheses",
        reasoning_graph_dir: Path | str = "outputs/reasoning_graph",
        output_dir: Path | str = "outputs/scientific_claims",
        overwrite: bool = False,
        software_version: str = DEFAULT_CLAIM_SOFTWARE_VERSION,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.hypotheses_dir = Path(hypotheses_dir)
        self.reasoning_graph_dir = Path(reasoning_graph_dir)
        self.output_dir = Path(output_dir)
        self.overwrite = overwrite
        self.software_version = software_version
        self._hypothesis_package: HypothesisClaimSourcePackage | None = None
        self._graph_package: ReasoningGraphClaimSourcePackage | None = None
        self._generated_at: str | None = None

    def load_hypotheses(self) -> tuple[dict[str, Any], ...]:
        self._hypothesis_package = load_hypothesis_claim_package(self.project_root, self.hypotheses_dir)
        return self._hypothesis_package.hypotheses

    def load_reasoning_graph(self) -> dict[str, Any]:
        self._graph_package = load_reasoning_graph_claim_package(self.project_root, self.reasoning_graph_dir)
        return self._graph_package.graph_document

    def validate_inputs(self) -> tuple[ClaimValidationIssue, ...]:
        issues: list[ClaimValidationIssue] = []
        if self._hypothesis_package is not None:
            issues.extend(self._hypothesis_package.validation_issues)
        if self._graph_package is not None:
            issues.extend(self._graph_package.validation_issues)
        return tuple(issues)

    def build_claims(self) -> tuple[ScientificClaim, ...]:
        if self._hypothesis_package is None or self._graph_package is None:
            raise RuntimeError("Hypotheses and reasoning graph must be loaded before building claims.")
        self._generated_at = self._generated_at or utc_now_iso()
        return build_claims_from_sources(
            self._hypothesis_package.hypotheses,
            self._graph_package.graph_document,
            hypothesis_validation_passed=self._hypothesis_package.validation_passed,
            graph_validation_passed=self._graph_package.validation_passed,
            created_at=self._generated_at,
            software_version=self.software_version,
            source_hypothesis_schema_version=self._hypothesis_package.schema_version,
            source_graph_schema_version=self._graph_package.schema_version,
        )

    def validate_claims(self, claims: tuple[ScientificClaim, ...]) -> tuple[ClaimValidationIssue, ...]:
        hypotheses_by_id = {}
        graph_document = {}
        if self._hypothesis_package is not None:
            hypotheses_by_id = {str(record.get("hypothesis_id")): record for record in self._hypothesis_package.hypotheses}
        if self._graph_package is not None:
            graph_document = self._graph_package.graph_document
        return validate_claims(claims, hypotheses_by_id=hypotheses_by_id, graph_document=graph_document)

    def write_outputs(
        self,
        claims: tuple[ScientificClaim, ...],
        validation_issues: tuple[ClaimValidationIssue, ...],
    ) -> tuple[Path, ...]:
        if self._hypothesis_package is None or self._graph_package is None:
            raise RuntimeError("Source packages must be loaded before writing outputs.")
        return write_claim_outputs(
            project_root=self.project_root,
            output_dir=self.output_dir,
            claims=claims,
            validation_issues=validation_issues,
            schema_version=CLAIM_SCHEMA_VERSION,
            software_version=self.software_version,
            generated_at=self._generated_at or utc_now_iso(),
            hypotheses_dir=self._hypothesis_package.hypotheses_dir,
            reasoning_graph_dir=self._graph_package.reasoning_graph_dir,
            source_hypotheses_loaded=tuple(record["hypothesis_id"] for record in self._hypothesis_package.hypotheses),
            source_hypotheses_missing=_required_hypotheses_missing(self._hypothesis_package.hypotheses),
            graph_nodes_loaded=len(self._graph_package.graph_document.get("nodes", ())),
            overwrite=self.overwrite,
        )

    def run(self) -> ClaimRunResult:
        self._generated_at = utc_now_iso()
        hypotheses = self.load_hypotheses()
        graph_document = self.load_reasoning_graph()
        input_issues = self.validate_inputs()
        if _has_fatal_load_issue(input_issues):
            return ClaimRunResult(
                claims=tuple(),
                validation_issues=input_issues,
                output_paths=tuple(),
                metadata=_run_metadata(
                    claims=tuple(),
                    validation_issues=input_issues,
                    hypothesis_package=self._hypothesis_package,
                    graph_package=self._graph_package,
                    output_paths=tuple(),
                ),
            )

        claims = self.build_claims()
        claim_issues = self.validate_claims(claims)
        all_issues = input_issues + claim_issues
        output_paths = self.write_outputs(claims, all_issues)
        return ClaimRunResult(
            claims=claims,
            validation_issues=all_issues,
            output_paths=output_paths,
            metadata=_run_metadata(
                claims=claims,
                validation_issues=all_issues,
                hypothesis_package=self._hypothesis_package,
                graph_package=self._graph_package,
                output_paths=output_paths,
            ),
        )


ScientificClaimEngine = ClaimEngine


def load_hypothesis_claim_package(
    project_root: Path | str,
    hypotheses_dir: Path | str,
) -> HypothesisClaimSourcePackage:
    root = Path(project_root).resolve()
    directory = _resolve_input_directory(root, hypotheses_dir)
    missing = tuple(name for name in REQUIRED_HYPOTHESIS_FILES if not (directory / name).exists())
    issues: list[ClaimValidationIssue] = []
    if missing:
        for name in missing:
            issues.append(_source_issue("MISSING_CLAIM_SOURCE_FILE", f"Required hypothesis output is missing: {name}", field=name))
        return HypothesisClaimSourcePackage(hypotheses_dir=directory, source_files_missing=missing, validation_issues=tuple(issues))

    loaded: list[str] = []
    hypotheses_document: dict[str, Any] = {}
    validation_document: dict[str, Any] = {}
    summary_document: dict[str, Any] = {}
    dependency_rows: tuple[dict[str, str], ...] = tuple()
    competition_rows: tuple[dict[str, str], ...] = tuple()
    hypotheses: tuple[dict[str, Any], ...] = tuple()
    try:
        hypotheses_document = _read_json(directory / "hypotheses.json")
        loaded.append("hypotheses.json")
        validation_document = _read_json(directory / "hypothesis_validation.json")
        loaded.append("hypothesis_validation.json")
        summary_document = _read_json(directory / "hypothesis_summary.json")
        loaded.append("hypothesis_summary.json")
        dependency_rows = _read_csv(directory / "hypothesis_dependencies.csv")
        loaded.append("hypothesis_dependencies.csv")
        competition_rows = _read_csv(directory / "hypothesis_competition_map.csv")
        loaded.append("hypothesis_competition_map.csv")
        hypotheses = _parse_hypotheses(hypotheses_document)
    except (OSError, json.JSONDecodeError, csv.Error, TypeError, ValueError) as exc:
        issues.append(_source_issue("UNREADABLE_CLAIM_SOURCE_PACKAGE", f"Hypothesis package could not be read: {exc}", field="hypotheses_dir"))
    issues.extend(_source_validation_issues(validation_document, source_name="hypothesis", field="hypothesis_validation.json"))
    return HypothesisClaimSourcePackage(
        hypotheses_dir=directory,
        hypotheses=hypotheses,
        hypotheses_document=hypotheses_document,
        validation_document=validation_document,
        summary_document=summary_document,
        dependency_rows=dependency_rows,
        competition_rows=competition_rows,
        source_files_loaded=tuple(loaded),
        source_files_missing=missing,
        validation_issues=tuple(issues),
    )


def load_reasoning_graph_claim_package(
    project_root: Path | str,
    reasoning_graph_dir: Path | str,
) -> ReasoningGraphClaimSourcePackage:
    root = Path(project_root).resolve()
    directory = _resolve_input_directory(root, reasoning_graph_dir)
    missing = tuple(name for name in REQUIRED_GRAPH_FILES if not (directory / name).exists())
    issues: list[ClaimValidationIssue] = []
    if missing:
        for name in missing:
            issues.append(_source_issue("MISSING_CLAIM_SOURCE_FILE", f"Required reasoning graph output is missing: {name}", field=name))
        return ReasoningGraphClaimSourcePackage(reasoning_graph_dir=directory, source_files_missing=missing, validation_issues=tuple(issues))

    loaded: list[str] = []
    graph_document: dict[str, Any] = {}
    validation_document: dict[str, Any] = {}
    summary_document: dict[str, Any] = {}
    try:
        graph_document = _read_json(directory / "reasoning_graph.json")
        loaded.append("reasoning_graph.json")
        validation_document = _read_json(directory / "reasoning_graph_validation.json")
        loaded.append("reasoning_graph_validation.json")
        summary_document = _read_json(directory / "reasoning_graph_summary.json")
        loaded.append("reasoning_graph_summary.json")
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        issues.append(_source_issue("UNREADABLE_CLAIM_SOURCE_PACKAGE", f"Reasoning graph package could not be read: {exc}", field="reasoning_graph_dir"))
    issues.extend(_source_validation_issues(validation_document, source_name="reasoning graph", field="reasoning_graph_validation.json"))
    return ReasoningGraphClaimSourcePackage(
        reasoning_graph_dir=directory,
        graph_document=graph_document,
        validation_document=validation_document,
        summary_document=summary_document,
        source_files_loaded=tuple(loaded),
        source_files_missing=missing,
        validation_issues=tuple(issues),
    )


def _parse_hypotheses(document: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    records = document.get("hypotheses", ())
    if not isinstance(records, list):
        raise TypeError("hypotheses.json must contain a hypotheses list")
    return tuple(sorted((dict(record) for record in records), key=lambda record: str(record.get("hypothesis_id"))))


def _source_validation_issues(document: dict[str, Any], *, source_name: str, field: str) -> tuple[ClaimValidationIssue, ...]:
    if not document:
        return tuple()
    issues: list[ClaimValidationIssue] = []
    critical_count = int(document.get("critical_issue_count") or 0)
    if critical_count > 0:
        issues.append(
            _source_issue(
                "SOURCE_VALIDATION_FAILURE",
                f"{source_name.title()} package reports {critical_count} critical validation issue(s).",
                field=field,
            )
        )
    if document.get("validation_passed") is False:
        issues.append(_source_issue("SOURCE_VALIDATION_FAILURE", f"{source_name.title()} package validation_passed is false.", field=field))
    for issue in document.get("structured_validation_issues", ()) or ():
        if str(issue.get("severity", "")).upper() == "CRITICAL":
            issues.append(
                ClaimValidationIssue(
                    code="SOURCE_VALIDATION_FAILURE",
                    severity="CRITICAL",
                    message=str(issue.get("message", f"{source_name.title()} package contains a critical validation issue.")),
                    claim_id=None,
                    field=field,
                    hypothesis_id=issue.get("hypothesis_id"),
                    graph_node_id=issue.get("node_id"),
                    rule_id=issue.get("rule_id"),
                )
            )
    return tuple(issues)


def _required_hypotheses_missing(hypotheses: tuple[dict[str, Any], ...]) -> tuple[str, ...]:
    present = {str(record.get("hypothesis_id")) for record in hypotheses}
    required = {
        "HYP-CHEMICAL_DISCRIMINATION-0001",
        "HYP-CHEMICAL_DISCRIMINATION-0002",
        "HYP-CONCENTRATION_ENCODING-0001",
        "HYP-CONCENTRATION_ENCODING-0002",
        "HYP-DATA_QUALITY_EFFECT-0001",
        "HYP-FEATURE_REPRESENTATION-0001",
        "HYP-FEATURE_REPRESENTATION-0002",
        "HYP-GENERALIZATION-0001",
        "HYP-OVERALL_SYSTEM_BEHAVIOR-0001",
        "HYP-STRAIN_CONTRIBUTION-0001",
        "HYP-STRAIN_CONTRIBUTION-0002",
        "HYP-TEMPORAL_INFORMATION-0001",
    }
    return tuple(sorted(required - present))


def _has_fatal_load_issue(issues: tuple[ClaimValidationIssue, ...]) -> bool:
    return any(issue.code in {"MISSING_CLAIM_SOURCE_FILE", "UNREADABLE_CLAIM_SOURCE_PACKAGE"} for issue in issues)


def _has_critical_issues(issues: tuple[ClaimValidationIssue, ...]) -> bool:
    return any(issue.severity.value == "CRITICAL" for issue in issues)


def _run_metadata(
    *,
    claims: tuple[ScientificClaim, ...],
    validation_issues: tuple[ClaimValidationIssue, ...],
    hypothesis_package: HypothesisClaimSourcePackage | None,
    graph_package: ReasoningGraphClaimSourcePackage | None,
    output_paths: tuple[Path, ...],
) -> dict[str, Any]:
    validation_summary = summarize_validation(claims, validation_issues, output_readability_checks={})
    summary = summarize_claims(
        claims,
        source_hypotheses_loaded=tuple()
        if hypothesis_package is None
        else tuple(record["hypothesis_id"] for record in hypothesis_package.hypotheses),
        source_hypotheses_missing=tuple()
        if hypothesis_package is None
        else _required_hypotheses_missing(hypothesis_package.hypotheses),
        graph_nodes_loaded=0 if graph_package is None else len(graph_package.graph_document.get("nodes", ())),
        validation_passed=validation_summary["validation_passed"],
    )
    return {
        "validation_passed": validation_summary["validation_passed"],
        "critical_issue_count": validation_summary["critical_issue_count"],
        "warning_count": validation_summary["warning_count"],
        "claim_count": len(claims),
        "withheld_claim_count": validation_summary["withheld_claim_count"],
        "output_paths": [str(path) for path in output_paths],
        "claim_summary": summary,
        "validation_summary": validation_summary,
    }


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


def _source_issue(code: str, message: str, *, field: str) -> ClaimValidationIssue:
    return ClaimValidationIssue(
        code=code,
        severity="CRITICAL",
        message=message,
        claim_id=None,
        field=field,
        hypothesis_id=None,
        graph_node_id=None,
        rule_id=None,
    )
