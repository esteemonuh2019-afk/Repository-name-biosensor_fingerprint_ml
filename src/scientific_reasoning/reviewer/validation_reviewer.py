"""Validation-boundary reviewer for BSIP downstream artifacts."""

from __future__ import annotations

from .enums import ReviewCategory, ReviewerConfidence, ReviewerType, Severity
from .models import REVIEW_SOFTWARE_VERSION, ReviewContext, ReviewFinding
from .policies import build_finding


def review(
    context: ReviewContext,
    *,
    created_at: str,
    software_version: str = REVIEW_SOFTWARE_VERSION,
) -> tuple[ReviewFinding, ...]:
    findings: list[ReviewFinding] = []
    index = 1
    source_failures = tuple(
        issue
        for issue in context.validation_issues
        if issue.code == "SOURCE_VALIDATION_FAILURE" and issue.severity.value == "CRITICAL"
    )
    if source_failures:
        findings.append(
            build_finding(
                reviewer_type=ReviewerType.VALIDATION,
                index=index,
                category=ReviewCategory.TRACEABILITY,
                title="Upstream validation failure is present",
                finding_text=f"{len(source_failures)} upstream source-validation failure(s) were recorded before reviewer assessment.",
                severity=Severity.CRITICAL,
                confidence=ReviewerConfidence.HIGH,
                source_validation_ids=context.source_validation_ids,
                rationale="Reviewer assessment cannot treat failed upstream validation as publication-ready evidence.",
                evidence_summary="source_validation_failure_count=" + str(len(source_failures)),
                limitations=("The reviewer does not repair upstream validation failures.",),
                rule_ids=("REVIEW-VALIDATION-SOURCE-001",),
                created_at=created_at,
                software_version=software_version,
                tags=("source-validation",),
            )
        )
        index += 1
    external_count = int(context.evidence_summary_document.get("claims_with_external_validation") or 0)
    if context.evidence_scores and external_count == 0:
        generalization_claim_ids = tuple(
            str(claim.get("claim_id"))
            for claim in context.claims
            if claim.get("category") in {"GENERALIZATION", "CHEMICAL_DISCRIMINATION", "SYSTEM_LEVEL_PERFORMANCE"}
        )
        findings.append(
            build_finding(
                reviewer_type=ReviewerType.VALIDATION,
                index=index,
                category=ReviewCategory.EXTERNAL_VALIDATION,
                title="True blind-label validation is absent",
                finding_text=(
                    "External validation performance has not been established because independently labelled unknown samples "
                    "were not evaluated."
                ),
                severity=Severity.MAJOR,
                confidence=ReviewerConfidence.HIGH,
                affected_claim_ids=generalization_claim_ids,
                evidence_score_ids=generalization_claim_ids,
                reasoning_graph_node_ids=_nodes_for_claims(context, generalization_claim_ids),
                source_validation_ids=context.source_validation_ids,
                rationale="The evidence-scoring summary reports zero claims with genuine external-validation support.",
                evidence_summary=f"claims_with_external_validation={external_count}",
                limitations=("Absence of external validation does not prevent draft use, but it blocks definitive generalization claims.",),
                rule_ids=("REVIEW-VALIDATION-EXTERNAL-001",),
                created_at=created_at,
                software_version=software_version,
                tags=("external-validation", "blind-labels"),
            )
        )
    return tuple(findings)


def _nodes_for_claims(context: ReviewContext, claim_ids: tuple[str, ...]) -> tuple[str, ...]:
    nodes: set[str] = set()
    for claim_id in claim_ids:
        claim = context.claim_by_id.get(claim_id, {})
        nodes.update(str(item) for item in claim.get("reasoning_graph_node_ids", ()) or ())
    return tuple(sorted(nodes))
