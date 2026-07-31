"""Source loading and source-validation gate for BSIP Manuscript Engine."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from .enums import ManuscriptIssueSeverity
from .models import ManuscriptSourcePackage, ManuscriptValidationIssue


REQUIRED_GROUPS: dict[str, tuple[str, ...]] = {
    "observations": ("observations.json", "observation_validation.json", "observation_summary.json"),
    "interpretations": ("interpretations.json", "interpretation_validation.json", "interpretation_summary.json"),
    "hypotheses": ("hypotheses.json", "hypothesis_validation.json", "hypothesis_summary.json"),
    "claims": ("claims.json", "claim_validation.json", "claim_summary.json", "claim_publication_matrix.csv"),
    "evidence": (
        "evidence_scores.json",
        "evidence_scoring_validation.json",
        "evidence_scoring_summary.json",
        "reviewer_confidence_summary.json",
        "uncertainty_report.json",
        "evidence_traceability.json",
    ),
    "review": (
        "review_findings.json",
        "reviewer_validation.json",
        "reviewer_summary.json",
        "reviewer_publication_assessment.json",
        "reviewer_claim_matrix.csv",
        "reviewer_revision_requirements.csv",
    ),
    "graph": ("reasoning_graph.json", "reasoning_graph_validation.json", "reasoning_graph_summary.json"),
    "supervisor": ("selected_figures.csv", "selected_tables.csv", "report_validation.json"),
}


def load_source_package(
    *,
    project_root: Path | str,
    observations_dir: Path | str,
    interpretations_dir: Path | str,
    hypotheses_dir: Path | str,
    claims_dir: Path | str,
    evidence_dir: Path | str,
    review_dir: Path | str,
    graph_dir: Path | str,
    supervisor_dir: Path | str,
) -> ManuscriptSourcePackage:
    root = Path(project_root).resolve()
    directories = {
        "observations": _resolve_input_directory(root, observations_dir),
        "interpretations": _resolve_input_directory(root, interpretations_dir),
        "hypotheses": _resolve_input_directory(root, hypotheses_dir),
        "claims": _resolve_input_directory(root, claims_dir),
        "evidence": _resolve_input_directory(root, evidence_dir),
        "review": _resolve_input_directory(root, review_dir),
        "graph": _resolve_input_directory(root, graph_dir),
        "supervisor": _resolve_input_directory(root, supervisor_dir),
    }
    issues: list[ManuscriptValidationIssue] = []
    missing: list[str] = []
    for group, filenames in REQUIRED_GROUPS.items():
        for filename in filenames:
            if not (directories[group] / filename).exists():
                missing.append(f"{group}:{filename}")
                issues.append(_source_issue("MISSING_SOURCE_FILE", f"Required manuscript source file is missing: {filename}", source_file=filename))
    if missing:
        return ManuscriptSourcePackage(
            observations_dir=directories["observations"],
            interpretations_dir=directories["interpretations"],
            hypotheses_dir=directories["hypotheses"],
            claims_dir=directories["claims"],
            evidence_dir=directories["evidence"],
            review_dir=directories["review"],
            graph_dir=directories["graph"],
            supervisor_dir=directories["supervisor"],
            source_files_missing=tuple(missing),
            validation_issues=tuple(issues),
        )

    loaded: list[str] = []
    try:
        observations = _read_json(directories["observations"] / "observations.json", loaded)
        observation_validation = _read_json(directories["observations"] / "observation_validation.json", loaded)
        observation_summary = _read_json(directories["observations"] / "observation_summary.json", loaded)
        interpretations = _read_json(directories["interpretations"] / "interpretations.json", loaded)
        interpretation_validation = _read_json(directories["interpretations"] / "interpretation_validation.json", loaded)
        interpretation_summary = _read_json(directories["interpretations"] / "interpretation_summary.json", loaded)
        hypotheses = _read_json(directories["hypotheses"] / "hypotheses.json", loaded)
        hypothesis_validation = _read_json(directories["hypotheses"] / "hypothesis_validation.json", loaded)
        hypothesis_summary = _read_json(directories["hypotheses"] / "hypothesis_summary.json", loaded)
        claims = _read_json(directories["claims"] / "claims.json", loaded)
        claim_validation = _read_json(directories["claims"] / "claim_validation.json", loaded)
        claim_summary = _read_json(directories["claims"] / "claim_summary.json", loaded)
        claim_publication_rows = _read_csv(directories["claims"] / "claim_publication_matrix.csv", loaded)
        evidence_scores = _read_json(directories["evidence"] / "evidence_scores.json", loaded)
        evidence_validation = _read_json(directories["evidence"] / "evidence_scoring_validation.json", loaded)
        evidence_summary = _read_json(directories["evidence"] / "evidence_scoring_summary.json", loaded)
        reviewer_confidence = _read_json(directories["evidence"] / "reviewer_confidence_summary.json", loaded)
        uncertainty = _read_json(directories["evidence"] / "uncertainty_report.json", loaded)
        evidence_traceability = _read_json(directories["evidence"] / "evidence_traceability.json", loaded)
        review_findings = _read_json(directories["review"] / "review_findings.json", loaded)
        reviewer_validation = _read_json(directories["review"] / "reviewer_validation.json", loaded)
        reviewer_summary = _read_json(directories["review"] / "reviewer_summary.json", loaded)
        reviewer_assessment = _read_json(directories["review"] / "reviewer_publication_assessment.json", loaded)
        reviewer_claim_rows = _read_csv(directories["review"] / "reviewer_claim_matrix.csv", loaded)
        reviewer_revision_rows = _read_csv(directories["review"] / "reviewer_revision_requirements.csv", loaded)
        graph = _read_json(directories["graph"] / "reasoning_graph.json", loaded)
        graph_validation = _read_json(directories["graph"] / "reasoning_graph_validation.json", loaded)
        graph_summary = _read_json(directories["graph"] / "reasoning_graph_summary.json", loaded)
        selected_figures = _read_csv(directories["supervisor"] / "selected_figures.csv", loaded)
        selected_tables = _read_csv(directories["supervisor"] / "selected_tables.csv", loaded)
        supervisor_validation = _read_json(directories["supervisor"] / "report_validation.json", loaded)
    except (OSError, json.JSONDecodeError, csv.Error, TypeError, ValueError) as exc:
        issues.append(_source_issue("UNREADABLE_SOURCE_FILE", f"Manuscript source package could not be read: {exc}", source_file="source_package"))
        return ManuscriptSourcePackage(
            observations_dir=directories["observations"],
            interpretations_dir=directories["interpretations"],
            hypotheses_dir=directories["hypotheses"],
            claims_dir=directories["claims"],
            evidence_dir=directories["evidence"],
            review_dir=directories["review"],
            graph_dir=directories["graph"],
            supervisor_dir=directories["supervisor"],
            source_files_loaded=tuple(loaded),
            validation_issues=tuple(issues),
        )

    package = ManuscriptSourcePackage(
        observations_dir=directories["observations"],
        interpretations_dir=directories["interpretations"],
        hypotheses_dir=directories["hypotheses"],
        claims_dir=directories["claims"],
        evidence_dir=directories["evidence"],
        review_dir=directories["review"],
        graph_dir=directories["graph"],
        supervisor_dir=directories["supervisor"],
        observations_document=observations,
        observation_validation_document=observation_validation,
        observation_summary_document=observation_summary,
        interpretations_document=interpretations,
        interpretation_validation_document=interpretation_validation,
        interpretation_summary_document=interpretation_summary,
        hypotheses_document=hypotheses,
        hypothesis_validation_document=hypothesis_validation,
        hypothesis_summary_document=hypothesis_summary,
        claims_document=claims,
        claim_validation_document=claim_validation,
        claim_summary_document=claim_summary,
        claim_publication_rows=claim_publication_rows,
        evidence_scores_document=evidence_scores,
        evidence_validation_document=evidence_validation,
        evidence_summary_document=evidence_summary,
        reviewer_confidence_document=reviewer_confidence,
        uncertainty_document=uncertainty,
        evidence_traceability_document=evidence_traceability,
        review_findings_document=review_findings,
        reviewer_validation_document=reviewer_validation,
        reviewer_summary_document=reviewer_summary,
        reviewer_publication_assessment_document=reviewer_assessment,
        reviewer_claim_rows=reviewer_claim_rows,
        reviewer_revision_rows=reviewer_revision_rows,
        graph_document=graph,
        graph_validation_document=graph_validation,
        graph_summary_document=graph_summary,
        selected_figures=selected_figures,
        selected_tables=selected_tables,
        supervisor_validation_document=supervisor_validation,
        source_files_loaded=tuple(loaded),
        source_files_missing=tuple(missing),
    )
    source_issues = validate_source_gate(package)
    return replace(package, validation_issues=tuple(source_issues))


def validate_source_gate(package: ManuscriptSourcePackage) -> tuple[ManuscriptValidationIssue, ...]:
    issues: list[ManuscriptValidationIssue] = []
    validation_docs = (
        ("observation_validation.json", package.observation_validation_document),
        ("interpretation_validation.json", package.interpretation_validation_document),
        ("hypothesis_validation.json", package.hypothesis_validation_document),
        ("claim_validation.json", package.claim_validation_document),
        ("evidence_scoring_validation.json", package.evidence_validation_document),
        ("reviewer_validation.json", package.reviewer_validation_document),
        ("reasoning_graph_validation.json", package.graph_validation_document),
        ("report_validation.json", package.supervisor_validation_document),
    )
    for filename, document in validation_docs:
        issues.extend(_validation_status_issues(filename, document))

    claim_ids = [str(claim.get("claim_id")) for claim in package.claims]
    for claim_id, count in sorted(Counter(claim_ids).items()):
        if count > 1:
            issues.append(_source_issue("DUPLICATE_CLAIM_ID", f"Duplicate claim ID in source claims: {claim_id}", source_file="claims.json", claim_id=claim_id, field="claim_id"))

    graph_node_ids = {str(node.get("node_id")) for node in package.graph_nodes}
    for claim in package.claims:
        claim_id = str(claim.get("claim_id"))
        if claim.get("claim_type") == "WITHHELD":
            continue
        if claim.get("publication_use") in {"RESULTS_ELIGIBLE", "DISCUSSION_ELIGIBLE", "LIMITATION_ONLY"} and not claim.get("reasoning_graph_node_ids"):
            issues.append(
                _source_issue(
                    "MISSING_TRACEABILITY",
                    f"Manuscript-eligible claim lacks reasoning graph traceability: {claim_id}",
                    source_file="claims.json",
                    claim_id=claim_id,
                    field="reasoning_graph_node_ids",
                )
            )
        for node_id in tuple(str(node) for node in claim.get("reasoning_graph_node_ids", ()) or ()):
            if node_id not in graph_node_ids:
                issues.append(
                    ManuscriptValidationIssue(
                        code="MISSING_GRAPH_REFERENCE",
                        severity=ManuscriptIssueSeverity.CRITICAL,
                        message=f"Claim references graph node absent from reasoning_graph.json: {node_id}",
                        claim_id=claim_id,
                        source_file="claims.json",
                        field="reasoning_graph_node_ids",
                    )
                )

    if package.reviewer_publication_assessment_document.get("manuscript_drafting_allowed") is not True:
        issues.append(
            ManuscriptValidationIssue(
                code="MANUSCRIPT_DRAFTING_PROHIBITED",
                severity=ManuscriptIssueSeverity.CRITICAL,
                message="Reviewer Engine publication assessment does not allow manuscript drafting.",
                source_file="reviewer_publication_assessment.json",
                field="manuscript_drafting_allowed",
            )
        )
    return tuple(issues)


def has_fatal_source_issue(issues: tuple[ManuscriptValidationIssue, ...]) -> bool:
    return any(
        issue.severity is ManuscriptIssueSeverity.CRITICAL
        and issue.code in {
            "MISSING_SOURCE_FILE",
            "UNREADABLE_SOURCE_FILE",
            "SOURCE_VALIDATION_FAILURE",
            "DUPLICATE_CLAIM_ID",
            "MISSING_TRACEABILITY",
            "MISSING_GRAPH_REFERENCE",
            "MANUSCRIPT_DRAFTING_PROHIBITED",
        }
        for issue in issues
    )


def _read_json(path: Path, loaded: list[str]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    loaded.append(str(path))
    return payload


def _read_csv(path: Path, loaded: list[str]) -> tuple[dict[str, str], ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = tuple(dict(row) for row in csv.DictReader(handle))
    loaded.append(str(path))
    return rows


def _resolve_input_directory(project_root: Path, directory: Path | str) -> Path:
    path = Path(directory)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _validation_status_issues(filename: str, document: Any) -> tuple[ManuscriptValidationIssue, ...]:
    if not isinstance(document, dict):
        return (_source_issue("SOURCE_VALIDATION_FAILURE", f"Validation document is not a JSON object: {filename}", source_file=filename),)
    passed = document.get("validation_passed")
    if passed is None and "passed" in document:
        passed = document.get("passed")
    critical_count = int(document.get("critical_issue_count") or 0)
    issues = []
    if passed is not True:
        issues.append(_source_issue("SOURCE_VALIDATION_FAILURE", f"Required validation did not pass: {filename}", source_file=filename, field="validation_passed"))
    if critical_count > 0:
        issues.append(_source_issue("SOURCE_VALIDATION_FAILURE", f"Required validation reports critical issues: {filename}", source_file=filename, field="critical_issue_count"))
    return tuple(issues)


def _source_issue(
    code: str,
    message: str,
    *,
    source_file: str,
    claim_id: str | None = None,
    field: str | None = None,
) -> ManuscriptValidationIssue:
    return ManuscriptValidationIssue(
        code=code,
        severity=ManuscriptIssueSeverity.CRITICAL,
        message=message,
        claim_id=claim_id,
        source_file=source_file,
        field=field,
    )
