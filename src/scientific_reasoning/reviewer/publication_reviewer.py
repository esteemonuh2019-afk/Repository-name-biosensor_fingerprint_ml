"""Publication-readiness aggregation reviewer."""

from __future__ import annotations

from collections import Counter

from .enums import OverallRecommendation, ReviewCategory, ReviewerConfidence, ReviewerType, Severity
from .models import REVIEW_SOFTWARE_VERSION, ReviewContext, ReviewFinding
from .policies import build_finding, determine_recommendation


def review(
    context: ReviewContext,
    prior_findings: tuple[ReviewFinding, ...],
    *,
    created_at: str,
    software_version: str = REVIEW_SOFTWARE_VERSION,
) -> tuple[ReviewFinding, ...]:
    has_results_ready = any(
        score.get("publication_readiness") in {"RESULTS_READY", "HIGH_CONFIDENCE_RESULTS_READY"}
        for score in context.evidence_scores
    )
    recommendation = determine_recommendation(prior_findings, has_results_ready_claim=has_results_ready)
    if recommendation is OverallRecommendation.READY_FOR_DRAFT_MANUSCRIPT:
        return tuple()
    severity = _severity_for_recommendation(recommendation)
    counts = Counter(finding.severity.value for finding in prior_findings)
    blockers = tuple(finding.finding_id for finding in prior_findings if finding.blocking)
    claim_ids = tuple(str(claim.get("claim_id")) for claim in context.claims)
    return (
        build_finding(
            reviewer_type=ReviewerType.PUBLICATION,
            index=1,
            category=ReviewCategory.PUBLICATION_READINESS,
            title="Publication readiness requires revision",
            finding_text=f"Policy-derived overall recommendation is {recommendation.value}.",
            severity=severity,
            confidence=ReviewerConfidence.HIGH,
            blocking=recommendation is OverallRecommendation.INTERNAL_REVIEW_ONLY,
            affected_claim_ids=claim_ids,
            evidence_score_ids=claim_ids,
            source_validation_ids=context.source_validation_ids,
            rationale="The publication reviewer applies the deterministic severity and blocking policy to prior reviewer findings.",
            evidence_summary=(
                f"critical={counts['CRITICAL']}; major={counts['MAJOR']}; moderate={counts['MODERATE']}; "
                f"minor={counts['MINOR']}; blockers={len(blockers)}"
            ),
            limitations=("This assessment is not a journal outcome prediction.",),
            rule_ids=("REVIEW-PUBLICATION-OVERALL-001",),
            created_at=created_at,
            software_version=software_version,
            tags=("publication-readiness",),
            metadata={"overall_recommendation": recommendation.value, "blocking_finding_ids": list(blockers)},
        ),
    )


def _severity_for_recommendation(recommendation: OverallRecommendation) -> Severity:
    if recommendation is OverallRecommendation.INTERNAL_REVIEW_ONLY:
        return Severity.CRITICAL
    if recommendation is OverallRecommendation.NEEDS_MAJOR_REVISION:
        return Severity.MAJOR
    if recommendation is OverallRecommendation.NEEDS_MODERATE_REVISION:
        return Severity.MODERATE
    if recommendation is OverallRecommendation.NEEDS_MINOR_REVISION:
        return Severity.MINOR
    return Severity.INFORMATION
