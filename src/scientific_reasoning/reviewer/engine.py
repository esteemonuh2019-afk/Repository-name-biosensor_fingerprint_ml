"""Service layer for the BSIP v4.1.0 Reviewer Engine."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from . import (
    evidence_reviewer,
    figure_reviewer,
    publication_reviewer,
    reproducibility_reviewer,
    scientific_reviewer,
    statistical_reviewer,
    validation_reviewer,
    writing_reviewer,
)
from .models import REVIEW_SOFTWARE_VERSION, ReviewContext, ReviewFinding, ReviewRunResult, ReviewValidationIssue, utc_now_iso
from .policies import sort_findings
from .validators import validate_source_documents
from .writers import publication_assessment, summarize_findings, write_review_outputs


REQUIRED_CLAIM_FILES: tuple[str, ...] = (
    "claims.json",
    "claim_validation.json",
    "claim_summary.json",
    "claim_publication_matrix.csv",
)
REQUIRED_EVIDENCE_FILES: tuple[str, ...] = (
    "evidence_scores.json",
    "evidence_scoring_validation.json",
    "evidence_scoring_summary.json",
    "reviewer_confidence_summary.json",
    "uncertainty_report.json",
    "evidence_traceability.json",
)
REQUIRED_GRAPH_FILES: tuple[str, ...] = (
    "reasoning_graph.json",
    "reasoning_graph_validation.json",
    "reasoning_graph_summary.json",
)
OPTIONAL_SUPERVISOR_FILES: tuple[str, ...] = (
    "selected_figures.csv",
    "selected_tables.csv",
    "report_validation.json",
)


class ReviewerEngine:
    """Review existing BSIP reasoning artifacts without performing scientific analysis."""

    def __init__(
        self,
        *,
        project_root: Path | str = ".",
        claims_dir: Path | str = "outputs/scientific_claims",
        evidence_scoring_dir: Path | str = "outputs/evidence_scoring",
        reasoning_graph_dir: Path | str = "outputs/reasoning_graph",
        supervisor_dir: Path | str = "outputs/supervisor_results_2",
        output_dir: Path | str = "outputs/scientific_review",
        overwrite: bool = False,
        strict: bool = False,
        software_version: str = REVIEW_SOFTWARE_VERSION,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.claims_dir = Path(claims_dir)
        self.evidence_scoring_dir = Path(evidence_scoring_dir)
        self.reasoning_graph_dir = Path(reasoning_graph_dir)
        self.supervisor_dir = Path(supervisor_dir)
        self.output_dir = Path(output_dir)
        self.overwrite = overwrite
        self.strict = strict
        self.software_version = software_version
        self._source_package: ReviewContext | None = None
        self._generated_at: str | None = None

    def load_sources(self) -> ReviewContext:
        self._source_package = load_source_package(
            self.project_root,
            claims_dir=self.claims_dir,
            evidence_scoring_dir=self.evidence_scoring_dir,
            reasoning_graph_dir=self.reasoning_graph_dir,
            supervisor_dir=self.supervisor_dir,
        )
        return self._source_package

    def validate_inputs(self) -> tuple[ReviewValidationIssue, ...]:
        if self._source_package is None:
            raise RuntimeError("Sources must be loaded before validation.")
        return self._source_package.validation_issues

    def build_findings(self) -> tuple[ReviewFinding, ...]:
        if self._source_package is None:
            raise RuntimeError("Sources must be loaded before reviewer execution.")
        generated_at = self._generated_at or utc_now_iso()
        context = self._source_package
        prior = []
        for reviewer in (
            scientific_reviewer,
            statistical_reviewer,
            evidence_reviewer,
            validation_reviewer,
            reproducibility_reviewer,
            figure_reviewer,
            writing_reviewer,
        ):
            prior.extend(reviewer.review(context, created_at=generated_at, software_version=self.software_version))
        ordered_prior = sort_findings(prior)
        publication_findings = publication_reviewer.review(
            context,
            ordered_prior,
            created_at=generated_at,
            software_version=self.software_version,
        )
        return sort_findings((*ordered_prior, *publication_findings))

    def write_outputs(
        self,
        findings: tuple[ReviewFinding, ...],
    ) -> tuple[tuple[Path, ...], tuple[ReviewValidationIssue, ...], dict[str, Any], dict[str, Any]]:
        if self._source_package is None:
            raise RuntimeError("Sources must be loaded before writing outputs.")
        return write_review_outputs(
            project_root=self.project_root,
            output_dir=self.output_dir,
            findings=findings,
            context=self._source_package,
            generated_at=self._generated_at or utc_now_iso(),
            overwrite=self.overwrite,
            software_version=self.software_version,
        )

    def run(self) -> ReviewRunResult:
        self._generated_at = utc_now_iso()
        source = self.load_sources()
        input_issues = self.validate_inputs()
        if _has_fatal_load_issue(input_issues):
            metadata = _run_metadata(tuple(), input_issues, source, software_version=self.software_version)
            return ReviewRunResult(findings=tuple(), validation_issues=input_issues, output_paths=tuple(), metadata=metadata)
        findings = self.build_findings()
        output_paths, validation_issues, summary, assessment = self.write_outputs(findings)
        metadata = _run_metadata(
            findings,
            validation_issues,
            source,
            output_paths=output_paths,
            summary=summary,
            assessment=assessment,
            software_version=self.software_version,
        )
        return ReviewRunResult(findings=findings, validation_issues=validation_issues, output_paths=output_paths, metadata=metadata)


def load_source_package(
    project_root: Path | str,
    *,
    claims_dir: Path | str,
    evidence_scoring_dir: Path | str,
    reasoning_graph_dir: Path | str,
    supervisor_dir: Path | str,
) -> ReviewContext:
    root = Path(project_root).resolve()
    claim_directory = _resolve_input_directory(root, claims_dir)
    evidence_directory = _resolve_input_directory(root, evidence_scoring_dir)
    graph_directory = _resolve_input_directory(root, reasoning_graph_dir)
    supervisor_directory = _resolve_input_directory(root, supervisor_dir)
    issues: list[ReviewValidationIssue] = []
    missing = []
    for name in REQUIRED_CLAIM_FILES:
        if not (claim_directory / name).exists():
            missing.append(f"claims:{name}")
            issues.append(_source_issue("MISSING_SOURCE_FILE", f"Required claim source file is missing: {name}", source_file=name))
    for name in REQUIRED_EVIDENCE_FILES:
        if not (evidence_directory / name).exists():
            missing.append(f"evidence:{name}")
            issues.append(_source_issue("MISSING_SOURCE_FILE", f"Required evidence-scoring source file is missing: {name}", source_file=name))
    for name in REQUIRED_GRAPH_FILES:
        if not (graph_directory / name).exists():
            missing.append(f"graph:{name}")
            issues.append(_source_issue("MISSING_SOURCE_FILE", f"Required reasoning-graph source file is missing: {name}", source_file=name))
    if missing:
        return ReviewContext(
            source_files_missing=tuple(missing),
            validation_issues=tuple(issues),
        )
    loaded: list[str] = []
    try:
        claims_document = _read_json(claim_directory / "claims.json")
        loaded.append(str(claim_directory / "claims.json"))
        claim_validation = _read_json(claim_directory / "claim_validation.json")
        loaded.append(str(claim_directory / "claim_validation.json"))
        claim_summary = _read_json(claim_directory / "claim_summary.json")
        loaded.append(str(claim_directory / "claim_summary.json"))
        publication_rows = _read_csv(claim_directory / "claim_publication_matrix.csv")
        loaded.append(str(claim_directory / "claim_publication_matrix.csv"))

        evidence_scores = _read_json(evidence_directory / "evidence_scores.json")
        loaded.append(str(evidence_directory / "evidence_scores.json"))
        evidence_validation = _read_json(evidence_directory / "evidence_scoring_validation.json")
        loaded.append(str(evidence_directory / "evidence_scoring_validation.json"))
        evidence_summary = _read_json(evidence_directory / "evidence_scoring_summary.json")
        loaded.append(str(evidence_directory / "evidence_scoring_summary.json"))
        reviewer_confidence = _read_json(evidence_directory / "reviewer_confidence_summary.json")
        loaded.append(str(evidence_directory / "reviewer_confidence_summary.json"))
        uncertainty = _read_json(evidence_directory / "uncertainty_report.json")
        loaded.append(str(evidence_directory / "uncertainty_report.json"))
        evidence_traceability = _read_json(evidence_directory / "evidence_traceability.json")
        loaded.append(str(evidence_directory / "evidence_traceability.json"))

        graph_document = _read_json(graph_directory / "reasoning_graph.json")
        loaded.append(str(graph_directory / "reasoning_graph.json"))
        graph_validation = _read_json(graph_directory / "reasoning_graph_validation.json")
        loaded.append(str(graph_directory / "reasoning_graph_validation.json"))
        graph_summary = _read_json(graph_directory / "reasoning_graph_summary.json")
        loaded.append(str(graph_directory / "reasoning_graph_summary.json"))

        selected_figures: tuple[dict[str, str], ...] = tuple()
        selected_tables: tuple[dict[str, str], ...] = tuple()
        supervisor_validation: dict[str, Any] = {}
        if (supervisor_directory / "selected_figures.csv").exists():
            selected_figures = _read_csv(supervisor_directory / "selected_figures.csv")
            loaded.append(str(supervisor_directory / "selected_figures.csv"))
        if (supervisor_directory / "selected_tables.csv").exists():
            selected_tables = _read_csv(supervisor_directory / "selected_tables.csv")
            loaded.append(str(supervisor_directory / "selected_tables.csv"))
        if (supervisor_directory / "report_validation.json").exists():
            supervisor_validation = _read_json(supervisor_directory / "report_validation.json")
            loaded.append(str(supervisor_directory / "report_validation.json"))
    except (OSError, json.JSONDecodeError, csv.Error, TypeError, ValueError) as exc:
        issues.append(_source_issue("UNREADABLE_SOURCE_FILE", f"Reviewer source package could not be read: {exc}", source_file="source_package"))
        return ReviewContext(source_files_loaded=tuple(loaded), validation_issues=tuple(issues))

    source_issues = validate_source_documents(
        claims_document=claims_document,
        claim_validation_document=claim_validation,
        evidence_scores_document=evidence_scores,
        evidence_validation_document=evidence_validation,
        graph_document=graph_document,
        graph_validation_document=graph_validation,
        supervisor_validation_document=supervisor_validation,
    )
    issues.extend(source_issues)
    return ReviewContext(
        claims_document=claims_document,
        claim_validation_document=claim_validation,
        claim_summary_document=claim_summary,
        claim_publication_rows=publication_rows,
        evidence_scores_document=evidence_scores,
        evidence_validation_document=evidence_validation,
        evidence_summary_document=evidence_summary,
        reviewer_confidence_document=reviewer_confidence,
        uncertainty_document=uncertainty,
        evidence_traceability_document=evidence_traceability,
        graph_document=graph_document,
        graph_validation_document=graph_validation,
        graph_summary_document=graph_summary,
        selected_figures=selected_figures,
        selected_tables=selected_tables,
        supervisor_validation_document=supervisor_validation,
        source_files_loaded=tuple(loaded),
        source_files_missing=tuple(missing),
        validation_issues=tuple(issues),
    )


def _run_metadata(
    findings: tuple[ReviewFinding, ...],
    issues: tuple[ReviewValidationIssue, ...],
    source: ReviewContext,
    *,
    output_paths: tuple[Path, ...] = tuple(),
    summary: dict[str, Any] | None = None,
    assessment: dict[str, Any] | None = None,
    software_version: str,
) -> dict[str, Any]:
    if assessment is None:
        assessment = publication_assessment(source, findings, generated_at=utc_now_iso(), software_version=software_version)
    if summary is None:
        validation_passed = not any(issue.severity.value == "CRITICAL" for issue in issues)
        summary = summarize_findings(findings, validation_passed=validation_passed, assessment=assessment, context=source, software_version=software_version)
    return {
        "validation_passed": summary.get("validation_passed", False),
        "critical_issue_count": sum(1 for issue in issues if issue.severity.value == "CRITICAL"),
        "warning_count": sum(1 for issue in issues if issue.severity.value == "WARNING"),
        "findings_generated": len(findings),
        "blocking_findings": sum(1 for finding in findings if getattr(finding, "blocking", False)),
        "overall_recommendation": assessment.get("overall_recommendation"),
        "overall_readiness_score": assessment.get("overall_readiness_score"),
        "claims_loaded": len(source.claims),
        "evidence_scores_loaded": len(source.evidence_scores),
        "graph_nodes_loaded": len(source.graph_node_ids),
        "source_files_loaded": list(source.source_files_loaded),
        "source_files_missing": list(source.source_files_missing),
        "output_paths": [str(path) for path in output_paths],
        "summary": summary,
        "publication_assessment": assessment,
    }


def _has_fatal_load_issue(issues: tuple[ReviewValidationIssue, ...]) -> bool:
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


def _source_issue(code: str, message: str, *, source_file: str) -> ReviewValidationIssue:
    return ReviewValidationIssue(
        code=code,
        severity="CRITICAL",
        message=message,
        source_file=source_file,
    )


ScientificReviewerEngine = ReviewerEngine
