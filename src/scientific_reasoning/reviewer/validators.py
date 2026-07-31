"""Validation helpers for BSIP reviewer sources and outputs."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from .enums import OverallRecommendation, ReviewerType, ReviewIssueSeverity, Severity
from .models import ReviewContext, ReviewFinding, ReviewValidationIssue
from .policies import (
    JOURNAL_PREDICTION_TERMS,
    NEW_CLAIM_PATTERNS,
    NOVELTY_TERMS,
    PROHIBITED_PROTOCOL_TERMS,
    SUPPORTED_CLAIM_SCHEMA_VERSIONS,
    SUPPORTED_EVIDENCE_SCHEMA_VERSIONS,
    SUPPORTED_GRAPH_SCHEMA_VERSIONS,
    default_blocking,
    determine_recommendation,
    text_contains_any,
)


def validate_source_documents(
    *,
    claims_document: dict[str, Any],
    claim_validation_document: dict[str, Any],
    evidence_scores_document: dict[str, Any],
    evidence_validation_document: dict[str, Any],
    graph_document: dict[str, Any],
    graph_validation_document: dict[str, Any],
    supervisor_validation_document: dict[str, Any] | None = None,
) -> tuple[ReviewValidationIssue, ...]:
    issues: list[ReviewValidationIssue] = []
    if claims_document.get("schema_version") not in SUPPORTED_CLAIM_SCHEMA_VERSIONS:
        issues.append(_source_issue("UNSUPPORTED_SCHEMA_VERSION", "Unsupported claim schema version.", "claims.json"))
    if evidence_scores_document.get("schema_version") not in SUPPORTED_EVIDENCE_SCHEMA_VERSIONS:
        issues.append(_source_issue("UNSUPPORTED_SCHEMA_VERSION", "Unsupported evidence scoring schema version.", "evidence_scores.json"))
    if graph_document.get("schema_version") not in SUPPORTED_GRAPH_SCHEMA_VERSIONS:
        issues.append(_source_issue("UNSUPPORTED_SCHEMA_VERSION", "Unsupported reasoning graph schema version.", "reasoning_graph.json"))
    issues.extend(_validation_status_issues(claim_validation_document, source_file="claim_validation.json", label="Claim validation"))
    issues.extend(_validation_status_issues(evidence_validation_document, source_file="evidence_scoring_validation.json", label="Evidence scoring validation"))
    issues.extend(_validation_status_issues(graph_validation_document, source_file="reasoning_graph_validation.json", label="Reasoning graph validation"))
    if supervisor_validation_document:
        issues.extend(
            _validation_status_issues(
                supervisor_validation_document,
                source_file="report_validation.json",
                label="Supervisor report validation",
                severity=ReviewIssueSeverity.WARNING,
            )
        )

    claims = tuple(claims_document.get("claims", ()) or ())
    evidence_scores = tuple(evidence_scores_document.get("evidence_scores", ()) or ())
    claim_ids = [str(claim.get("claim_id")) for claim in claims]
    evidence_claim_ids = [str(record.get("claim_id")) for record in evidence_scores]
    for claim_id, count in sorted(Counter(claim_ids).items()):
        if count > 1:
            issues.append(_source_issue("DUPLICATE_CLAIM_ID", f"Duplicate claim ID in source claims: {claim_id}", "claims.json", claim_id=claim_id, field="claim_id"))
    for claim_id, count in sorted(Counter(evidence_claim_ids).items()):
        if count > 1:
            issues.append(
                _source_issue(
                    "DUPLICATE_EVIDENCE_SCORE_ID",
                    f"Duplicate evidence score claim ID: {claim_id}",
                    "evidence_scores.json",
                    claim_id=claim_id,
                    field="claim_id",
                )
            )
    missing_scores = sorted(set(claim_ids) - set(evidence_claim_ids))
    for claim_id in missing_scores:
        issues.append(
            _source_issue(
                "MISSING_EVIDENCE_SCORE",
                f"Claim is not represented in evidence_scores.json: {claim_id}",
                "evidence_scores.json",
                claim_id=claim_id,
                field="claim_id",
            )
        )
    missing_claims = sorted(set(evidence_claim_ids) - set(claim_ids))
    for claim_id in missing_claims:
        issues.append(
            _source_issue(
                "MISSING_CLAIM_REFERENCE",
                f"Evidence score references an unknown claim: {claim_id}",
                "evidence_scores.json",
                claim_id=claim_id,
                field="claim_id",
            )
        )
    return tuple(issues)


def validate_review_package(
    findings: Iterable[ReviewFinding],
    *,
    context: ReviewContext,
    overall_recommendation: OverallRecommendation | str,
    has_results_ready_claim: bool,
    output_readability_checks: dict[str, dict[str, Any]],
) -> tuple[ReviewValidationIssue, ...]:
    ordered = tuple(findings)
    issues: list[ReviewValidationIssue] = []
    issues.extend(validate_review_findings(ordered, context=context))
    issues.extend(
        validate_recommendation_policy(
            ordered,
            overall_recommendation=overall_recommendation,
            has_results_ready_claim=has_results_ready_claim,
        )
    )
    issues.extend(_output_readability_issues(output_readability_checks))
    return tuple(issues)


def validate_review_findings(
    findings: Iterable[ReviewFinding],
    *,
    context: ReviewContext,
) -> tuple[ReviewValidationIssue, ...]:
    ordered = tuple(findings)
    issues: list[ReviewValidationIssue] = []
    claim_ids = set(context.claim_by_id)
    evidence_score_ids = set(context.evidence_by_claim_id)
    graph_node_ids = context.graph_node_ids
    for finding_id, count in sorted(Counter(finding.finding_id for finding in ordered).items()):
        if count > 1:
            issues.append(_finding_issue("DUPLICATE_FINDING_ID", f"Duplicate review finding ID: {finding_id}", finding_id=finding_id, field="finding_id"))
    for finding in ordered:
        for claim_id in finding.affected_claim_ids:
            if claim_id not in claim_ids:
                issues.append(
                    _finding_issue(
                        "MISSING_CLAIM_REFERENCE",
                        f"Review finding references an unknown claim: {claim_id}",
                        finding_id=finding.finding_id,
                        claim_id=claim_id,
                        field="affected_claim_ids",
                    )
                )
        for score_id in finding.evidence_score_ids:
            if score_id not in evidence_score_ids:
                issues.append(
                    _finding_issue(
                        "MISSING_EVIDENCE_SCORE_REFERENCE",
                        f"Review finding references an unknown evidence score: {score_id}",
                        finding_id=finding.finding_id,
                        claim_id=score_id,
                        field="evidence_score_ids",
                    )
                )
        for node_id in finding.reasoning_graph_node_ids:
            if node_id not in graph_node_ids:
                issues.append(
                    _finding_issue(
                        "MISSING_GRAPH_REFERENCE",
                        f"Review finding references an unknown graph node: {node_id}",
                        finding_id=finding.finding_id,
                        graph_node_id=node_id,
                        field="reasoning_graph_node_ids",
                    )
                )
        if finding.severity is not Severity.INFORMATION and not finding.revision_requirement:
            issues.append(
                _finding_issue(
                    "MISSING_REVISION_REQUIREMENT",
                    "Non-information review finding lacks a revision requirement.",
                    finding_id=finding.finding_id,
                    field="revision_requirement",
                )
            )
        if finding.severity is Severity.CRITICAL and not finding.blocking:
            issues.append(_finding_issue("BLOCKING_POLICY_ISSUE", "Critical findings must be blocking.", finding_id=finding.finding_id, field="blocking"))
        if finding.severity is Severity.INFORMATION and finding.blocking:
            issues.append(_finding_issue("BLOCKING_POLICY_ISSUE", "Information findings cannot be blocking.", finding_id=finding.finding_id, field="blocking"))
        expected_blocking = default_blocking(finding.category, finding.severity, finding.rule_ids)
        if finding.blocking is False and expected_blocking:
            issues.append(_finding_issue("BLOCKING_POLICY_ISSUE", "Finding blocking flag is weaker than policy.", finding_id=finding.finding_id, field="blocking"))
        issues.extend(_boundary_language_issues(finding))
    issues.extend(validate_deterministic_ordering(ordered))
    return tuple(issues)


def validate_recommendation_policy(
    findings: Iterable[ReviewFinding],
    *,
    overall_recommendation: OverallRecommendation | str,
    has_results_ready_claim: bool = False,
) -> tuple[ReviewValidationIssue, ...]:
    non_publication_findings = tuple(finding for finding in findings if finding.reviewer_type is not ReviewerType.PUBLICATION)
    expected = determine_recommendation(non_publication_findings, has_results_ready_claim=has_results_ready_claim)
    actual = OverallRecommendation(overall_recommendation)
    if expected is actual:
        return tuple()
    return (
        ReviewValidationIssue(
            code="RECOMMENDATION_POLICY_ISSUE",
            severity=ReviewIssueSeverity.CRITICAL,
            message=f"Overall recommendation {actual.value} does not match policy-derived {expected.value}.",
            field="overall_recommendation",
            rule_id="REVIEW-PUBLICATION-OVERALL-001",
        ),
    )


def validate_deterministic_ordering(findings: tuple[ReviewFinding, ...]) -> tuple[ReviewValidationIssue, ...]:
    actual = [finding.finding_id for finding in findings]
    if actual == sorted(actual):
        return tuple()
    return (
        ReviewValidationIssue(
            code="DETERMINISTIC_ORDERING_ISSUE",
            severity=ReviewIssueSeverity.WARNING,
            message="Review findings are not ordered deterministically by finding_id.",
            field="review_findings",
        ),
    )


def validation_summary(
    findings: Iterable[ReviewFinding],
    issues: Iterable[ReviewValidationIssue],
    *,
    output_readability_checks: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    ordered = tuple(findings)
    issue_tuple = tuple(issues)
    code_counts = Counter(issue.code for issue in issue_tuple)
    critical_count = sum(1 for issue in issue_tuple if issue.severity is ReviewIssueSeverity.CRITICAL)
    warning_count = sum(1 for issue in issue_tuple if issue.severity is ReviewIssueSeverity.WARNING)
    readability_failed = sum(1 for check in output_readability_checks.values() if not check.get("readable"))
    return {
        "validation_passed": critical_count == 0 and readability_failed == 0,
        "critical_issue_count": critical_count,
        "warning_count": warning_count,
        "duplicate_finding_id_count": code_counts["DUPLICATE_FINDING_ID"],
        "missing_claim_reference_count": code_counts["MISSING_CLAIM_REFERENCE"],
        "missing_graph_reference_count": code_counts["MISSING_GRAPH_REFERENCE"],
        "missing_revision_requirement_count": code_counts["MISSING_REVISION_REQUIREMENT"],
        "severity_policy_issue_count": code_counts["SEVERITY_POLICY_ISSUE"],
        "blocking_policy_issue_count": code_counts["BLOCKING_POLICY_ISSUE"],
        "recommendation_policy_issue_count": code_counts["RECOMMENDATION_POLICY_ISSUE"],
        "new_claim_issue_count": code_counts["NEW_CLAIM_ISSUE"],
        "experimental_protocol_issue_count": code_counts["EXPERIMENTAL_PROTOCOL_ISSUE"],
        "novelty_language_issue_count": code_counts["NOVELTY_LANGUAGE_ISSUE"],
        "journal_prediction_issue_count": code_counts["JOURNAL_PREDICTION_ISSUE"],
        "deterministic_ordering_issue_count": code_counts["DETERMINISTIC_ORDERING_ISSUE"],
        "missing_evidence_score_reference_count": code_counts["MISSING_EVIDENCE_SCORE_REFERENCE"],
        "finding_count": len(ordered),
        "output_readability_checks": output_readability_checks,
        "structured_validation_issues": [issue.to_record() for issue in issue_tuple],
    }


def _validation_status_issues(
    document: dict[str, Any],
    *,
    source_file: str,
    label: str,
    severity: ReviewIssueSeverity = ReviewIssueSeverity.CRITICAL,
) -> tuple[ReviewValidationIssue, ...]:
    issues = []
    passed = document.get("validation_passed")
    if passed is None and "passed" in document:
        passed = document.get("passed")
    if passed is not True:
        issues.append(
            ReviewValidationIssue(
                code="SOURCE_VALIDATION_FAILURE",
                severity=severity,
                message=f"{label} did not pass.",
                source_file=source_file,
                field="validation_passed",
                rule_id="REVIEW-VALIDATION-SOURCE-001",
            )
        )
    if int(document.get("critical_issue_count") or 0) > 0:
        issues.append(
            ReviewValidationIssue(
                code="SOURCE_VALIDATION_FAILURE",
                severity=severity,
                message=f"{label} reports critical issues.",
                source_file=source_file,
                field="critical_issue_count",
                rule_id="REVIEW-VALIDATION-SOURCE-001",
            )
        )
    return tuple(issues)


def _source_issue(
    code: str,
    message: str,
    source_file: str,
    *,
    claim_id: str | None = None,
    field: str | None = None,
) -> ReviewValidationIssue:
    return ReviewValidationIssue(
        code=code,
        severity=ReviewIssueSeverity.CRITICAL,
        message=message,
        claim_id=claim_id,
        field=field,
        source_file=source_file,
    )


def _finding_issue(
    code: str,
    message: str,
    *,
    finding_id: str | None = None,
    claim_id: str | None = None,
    graph_node_id: str | None = None,
    field: str | None = None,
) -> ReviewValidationIssue:
    return ReviewValidationIssue(
        code=code,
        severity=ReviewIssueSeverity.CRITICAL if code != "DETERMINISTIC_ORDERING_ISSUE" else ReviewIssueSeverity.WARNING,
        message=message,
        finding_id=finding_id,
        claim_id=claim_id,
        graph_node_id=graph_node_id,
        field=field,
    )


def _boundary_language_issues(finding: ReviewFinding) -> tuple[ReviewValidationIssue, ...]:
    text = " ".join(
        (
            finding.title,
            finding.finding_text,
            finding.rationale,
            finding.evidence_summary,
            finding.revision_requirement,
            " ".join(finding.limitations),
        )
    ).lower()
    issues: list[ReviewValidationIssue] = []
    if text_contains_any(text, PROHIBITED_PROTOCOL_TERMS):
        issues.append(_finding_issue("EXPERIMENTAL_PROTOCOL_ISSUE", "Review finding contains prohibited experimental-protocol language.", finding_id=finding.finding_id))
    if _contains_unqualified_novelty(text):
        issues.append(_finding_issue("NOVELTY_LANGUAGE_ISSUE", "Review finding contains novelty language.", finding_id=finding.finding_id))
    if text_contains_any(text, JOURNAL_PREDICTION_TERMS):
        issues.append(_finding_issue("JOURNAL_PREDICTION_ISSUE", "Review finding contains journal-prediction language.", finding_id=finding.finding_id))
    if text_contains_any(text, NEW_CLAIM_PATTERNS):
        issues.append(_finding_issue("NEW_CLAIM_ISSUE", "Review finding may introduce a new scientific claim.", finding_id=finding.finding_id))
    return tuple(issues)


def _contains_unqualified_novelty(text: str) -> bool:
    if not text_contains_any(text, NOVELTY_TERMS):
        return False
    qualified = (
        "unsupported novelty" in text
        or "novelty wording" in text
        or "novelty language" in text
        or "not novelty" in text
    )
    return not qualified


def _output_readability_issues(checks: dict[str, dict[str, Any]]) -> tuple[ReviewValidationIssue, ...]:
    issues = []
    for filename, check in sorted(checks.items()):
        if not check.get("readable"):
            issues.append(
                ReviewValidationIssue(
                    code="OUTPUT_READABILITY_FAILURE",
                    severity=ReviewIssueSeverity.CRITICAL,
                    message=f"Reviewer output is not readable: {filename}",
                    source_file=filename,
                    field="output_readability_checks",
                )
            )
    return tuple(issues)
