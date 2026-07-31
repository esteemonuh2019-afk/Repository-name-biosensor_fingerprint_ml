"""Evidence-score reviewer for BSIP support, uncertainty, and gap boundaries."""

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
    high_uncertainty_strong = tuple(
        score
        for score in context.evidence_scores
        if score.get("evidence_level") in {"STRONG", "VERY_STRONG"}
        and score.get("uncertainty_level") in {"HIGH", "VERY_HIGH"}
    )
    if high_uncertainty_strong:
        claim_ids = tuple(str(score.get("claim_id")) for score in high_uncertainty_strong)
        findings.append(
            build_finding(
                reviewer_type=ReviewerType.EVIDENCE,
                index=index,
                category=ReviewCategory.CLAIM_SUPPORT,
                title="Strong evidence scores carry high uncertainty",
                finding_text=(
                    f"{len(claim_ids)} claim(s) have STRONG or VERY_STRONG evidence scores while retaining "
                    "HIGH or VERY_HIGH uncertainty."
                ),
                severity=Severity.MODERATE,
                confidence=ReviewerConfidence.HIGH,
                affected_claim_ids=claim_ids,
                affected_hypothesis_ids=_ids(high_uncertainty_strong, "supporting_hypothesis_ids"),
                affected_interpretation_ids=_ids(high_uncertainty_strong, "supporting_interpretation_ids"),
                affected_observation_ids=_ids(high_uncertainty_strong, "supporting_observation_ids"),
                evidence_score_ids=claim_ids,
                reasoning_graph_node_ids=_ids(high_uncertainty_strong, "reasoning_graph_node_ids"),
                rationale="Evidence level and uncertainty level are separate fields in evidence_scores.json.",
                evidence_summary="uncertainty_levels=" + ", ".join(sorted({str(score.get("uncertainty_level")) for score in high_uncertainty_strong})),
                limitations=("High evidence support does not remove recorded uncertainty.",),
                rule_ids=("REVIEW-EVIDENCE-UNCERTAINTY-001",),
                created_at=created_at,
                software_version=software_version,
                tags=("evidence-score", "uncertainty"),
            )
        )
        index += 1
    downgraded_readiness = tuple(
        score
        for score in context.evidence_scores
        if score.get("claim_publication_use") == "RESULTS_ELIGIBLE"
        and score.get("publication_readiness") not in {"RESULTS_READY", "HIGH_CONFIDENCE_RESULTS_READY"}
    )
    if downgraded_readiness:
        claim_ids = tuple(str(score.get("claim_id")) for score in downgraded_readiness)
        findings.append(
            build_finding(
                reviewer_type=ReviewerType.EVIDENCE,
                index=index,
                category=ReviewCategory.PUBLICATION_READINESS,
                title="Results-eligible claims do not reach results readiness",
                finding_text=f"{len(claim_ids)} RESULTS_ELIGIBLE claim(s) are scored below RESULTS_READY publication readiness.",
                severity=Severity.MAJOR,
                confidence=ReviewerConfidence.HIGH,
                affected_claim_ids=claim_ids,
                evidence_score_ids=claim_ids,
                reasoning_graph_node_ids=_ids(downgraded_readiness, "reasoning_graph_node_ids"),
                rationale="The reviewer preserves the Evidence Scoring Engine publication-readiness labels.",
                evidence_summary="publication_readiness=" + ", ".join(sorted({str(score.get("publication_readiness")) for score in downgraded_readiness})),
                limitations=("Claim Engine eligibility is a ceiling, not a guarantee of Results placement.",),
                rule_ids=("REVIEW-EVIDENCE-READINESS-001",),
                created_at=created_at,
                software_version=software_version,
                tags=("publication-readiness",),
            )
        )
        index += 1
    claims_with_gaps = tuple(score for score in context.evidence_scores if score.get("evidence_gaps"))
    if claims_with_gaps:
        claim_ids = tuple(str(score.get("claim_id")) for score in claims_with_gaps)
        findings.append(
            build_finding(
                reviewer_type=ReviewerType.EVIDENCE,
                index=index,
                category=ReviewCategory.MISSING_EVIDENCE,
                title="Evidence gaps remain recorded",
                finding_text=f"{len(claim_ids)} scored claim(s) retain at least one evidence gap.",
                severity=Severity.MODERATE,
                confidence=ReviewerConfidence.HIGH,
                affected_claim_ids=claim_ids,
                evidence_score_ids=claim_ids,
                reasoning_graph_node_ids=_ids(claims_with_gaps, "reasoning_graph_node_ids"),
                rationale="Evidence-gap counts are taken directly from evidence_scores.json.",
                evidence_summary=f"claims_with_evidence_gaps={len(claim_ids)}",
                limitations=("Recorded gaps are not filled or reweighted by reviewer output.",),
                rule_ids=("REVIEW-EVIDENCE-GAP-001",),
                created_at=created_at,
                software_version=software_version,
                tags=("evidence-gaps",),
            )
        )
    return tuple(findings)


def _ids(records: tuple[dict, ...], field: str) -> tuple[str, ...]:
    return tuple(sorted({str(item) for record in records for item in record.get(field, ()) or ()}))
