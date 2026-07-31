"""Language-boundary reviewer for structured claim text."""

from __future__ import annotations

from .enums import ReviewCategory, ReviewerConfidence, ReviewerType, Severity
from .models import REVIEW_SOFTWARE_VERSION, ReviewContext, ReviewFinding
from .policies import build_finding, writing_overclaim_labels


def review(
    context: ReviewContext,
    *,
    created_at: str,
    software_version: str = REVIEW_SOFTWARE_VERSION,
) -> tuple[ReviewFinding, ...]:
    findings: list[ReviewFinding] = []
    index = 1
    for claim in context.claims:
        claim_text = str(claim.get("claim_text") or "")
        labels = writing_overclaim_labels(claim_text)
        if not labels:
            continue
        claim_id = str(claim.get("claim_id"))
        score = context.evidence_by_claim_id.get(claim_id, {})
        severity = Severity.MAJOR if claim.get("publication_use") == "RESULTS_ELIGIBLE" else Severity.MODERATE
        findings.append(
            build_finding(
                reviewer_type=ReviewerType.WRITING,
                index=index,
                category=ReviewCategory.LANGUAGE_STRENGTH,
                title="Claim wording exceeds supported language boundary",
                finding_text=f"Claim {claim_id} contains {', '.join(labels)} wording in claim_text.",
                severity=severity,
                confidence=ReviewerConfidence.HIGH,
                affected_claim_ids=(claim_id,),
                affected_hypothesis_ids=tuple(str(item) for item in claim.get("supporting_hypothesis_ids", ()) or ()),
                affected_interpretation_ids=tuple(str(item) for item in claim.get("supporting_interpretation_ids", ()) or ()),
                affected_observation_ids=tuple(str(item) for item in claim.get("supporting_observation_ids", ()) or ()),
                evidence_score_ids=(claim_id,),
                reasoning_graph_node_ids=tuple(str(item) for item in claim.get("reasoning_graph_node_ids", ()) or ()),
                rationale="Claim text is reviewed against the BSIP language-strength policy.",
                evidence_summary=f"detected_language_labels={', '.join(labels)}; uncertainty={score.get('uncertainty_level')}",
                limitations=("The writing reviewer flags language boundaries but does not rewrite prose.",),
                rule_ids=("REVIEW-WRITING-FORBIDDEN-001", "REVIEW-WRITING-LANGUAGE-001"),
                created_at=created_at,
                software_version=software_version,
                tags=("language",),
            )
        )
        index += 1
    return tuple(findings)
