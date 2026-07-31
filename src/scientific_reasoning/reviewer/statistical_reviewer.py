"""Statistical-boundary reviewer for BSIP claims and evidence scores."""

from __future__ import annotations

from .enums import ReviewCategory, ReviewerConfidence, ReviewerType, Severity
from .models import REVIEW_SOFTWARE_VERSION, ReviewContext, ReviewFinding
from .policies import build_finding, text_contains_any


def review(
    context: ReviewContext,
    *,
    created_at: str,
    software_version: str = REVIEW_SOFTWARE_VERSION,
) -> tuple[ReviewFinding, ...]:
    findings: list[ReviewFinding] = []
    index = 1
    summary = context.evidence_summary_document
    external_count = int(summary.get("claims_with_external_validation") or 0)
    primary_claims = tuple(
        claim
        for claim in context.claims
        if claim.get("claim_type") == "PRIMARY_FINDING" and claim.get("publication_use") == "RESULTS_ELIGIBLE"
    )
    if primary_claims and external_count == 0:
        claim_ids = tuple(str(claim.get("claim_id")) for claim in primary_claims)
        findings.append(
            build_finding(
                reviewer_type=ReviewerType.STATISTICAL,
                index=index,
                category=ReviewCategory.EXTERNAL_VALIDATION,
                title="Primary performance claims remain internally validated",
                finding_text=(
                    f"{len(claim_ids)} primary RESULTS_ELIGIBLE claim(s) are present, and the evidence summary reports "
                    "0 claims with genuine external validation."
                ),
                severity=Severity.MAJOR,
                confidence=ReviewerConfidence.HIGH,
                affected_claim_ids=claim_ids,
                evidence_score_ids=claim_ids,
                reasoning_graph_node_ids=_combined_nodes(primary_claims),
                source_validation_ids=context.source_validation_ids,
                rationale="Internal validation and external validation are separate policy categories.",
                evidence_summary=f"claims_with_external_validation={external_count}",
                limitations=("External-validity conclusions are not established by internal validation alone.",),
                rule_ids=("REVIEW-STAT-EXTERNAL-001",),
                created_at=created_at,
                software_version=software_version,
                tags=("external-validation", "statistical-boundary"),
            )
        )
        index += 1
    regression_scores = tuple(
        score
        for score in context.evidence_scores
        if score.get("claim_category") == "CONCENTRATION_INFORMATION"
        and score.get("uncertainty_level") in {"HIGH", "VERY_HIGH"}
    )
    if regression_scores:
        claim_ids = tuple(str(score.get("claim_id")) for score in regression_scores)
        findings.append(
            build_finding(
                reviewer_type=ReviewerType.STATISTICAL,
                index=index,
                category=ReviewCategory.STATISTICAL_INTERPRETATION,
                title="Regression evidence carries high uncertainty",
                finding_text=f"{len(claim_ids)} concentration-information claim(s) have HIGH or VERY_HIGH uncertainty in evidence scoring.",
                severity=Severity.MODERATE,
                confidence=ReviewerConfidence.HIGH,
                affected_claim_ids=claim_ids,
                affected_hypothesis_ids=_score_ids(regression_scores, "supporting_hypothesis_ids"),
                affected_interpretation_ids=_score_ids(regression_scores, "supporting_interpretation_ids"),
                affected_observation_ids=_score_ids(regression_scores, "supporting_observation_ids"),
                evidence_score_ids=claim_ids,
                reasoning_graph_node_ids=_score_ids(regression_scores, "reasoning_graph_node_ids"),
                rationale="The finding preserves the evidence scorer's uncertainty label for concentration-related claims.",
                evidence_summary="uncertainty_levels=" + ", ".join(sorted({str(score.get("uncertainty_level")) for score in regression_scores})),
                limitations=("Regression support is assessed separately from classification support.",),
                rule_ids=("REVIEW-STAT-REGRESSION-001",),
                created_at=created_at,
                software_version=software_version,
                tags=("regression", "uncertainty"),
            )
        )
        index += 1
    system_claims = tuple(
        claim
        for claim in context.claims
        if claim.get("category") == "SYSTEM_LEVEL_PERFORMANCE"
        and _has_metric_family_boundary(claim, context.evidence_by_claim_id.get(str(claim.get("claim_id")), {}))
    )
    if system_claims:
        claim_ids = tuple(str(claim.get("claim_id")) for claim in system_claims)
        findings.append(
            build_finding(
                reviewer_type=ReviewerType.STATISTICAL,
                index=index,
                category=ReviewCategory.STATISTICAL_INTERPRETATION,
                title="System-level claim spans task-specific metrics",
                finding_text="System-level performance evidence records a metric-family boundary across classification and regression tasks.",
                severity=Severity.MODERATE,
                confidence=ReviewerConfidence.MODERATE,
                affected_claim_ids=claim_ids,
                evidence_score_ids=claim_ids,
                reasoning_graph_node_ids=_combined_nodes(system_claims),
                rationale="The source claim records that tasks use different metrics.",
                evidence_summary="metric_family_boundary=present",
                limitations=("Classification and regression metrics are not interchangeable.",),
                rule_ids=("REVIEW-STAT-METRIC-COMPAT-001",),
                created_at=created_at,
                software_version=software_version,
                tags=("metric-boundary",),
            )
        )
    return tuple(findings)


def _has_metric_family_boundary(claim: dict, score: dict) -> bool:
    values = []
    for field in ("limitations", "evidence_gap_ids"):
        values.extend(str(item) for item in claim.get(field, ()) or ())
    for field in ("limitations", "evidence_gaps", "negative_factors"):
        values.extend(str(item) for item in score.get(field, ()) or ())
    return text_contains_any(" ".join(values), ("different metrics", "task-specific metrics", "classification and regression"))


def _combined_nodes(records: tuple[dict, ...]) -> tuple[str, ...]:
    return tuple(sorted({str(node) for record in records for node in record.get("reasoning_graph_node_ids", ()) or ()}))


def _score_ids(records: tuple[dict, ...], field: str) -> tuple[str, ...]:
    return tuple(sorted({str(item) for record in records for item in record.get(field, ()) or ()}))
