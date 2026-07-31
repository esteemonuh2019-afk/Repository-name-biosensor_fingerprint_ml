"""Publication-readiness and reviewer-confidence policies."""

from __future__ import annotations

from .enums import EvidenceLevel, PublicationReadiness, ReviewerConfidence, UncertaintyLevel


PUBLICATION_CEILINGS = {
    "INTERNAL_REVIEW_ONLY": PublicationReadiness.NOT_READY,
    "NOT_ELIGIBLE": PublicationReadiness.NOT_READY,
    "LIMITATION_ONLY": PublicationReadiness.LIMITATION_ONLY,
    "DISCUSSION_ELIGIBLE": PublicationReadiness.DISCUSSION_READY,
    "RESULTS_ELIGIBLE": PublicationReadiness.HIGH_CONFIDENCE_RESULTS_READY,
}

READINESS_RANK = {
    PublicationReadiness.NOT_READY: 0,
    PublicationReadiness.LIMITATION_ONLY: 1,
    PublicationReadiness.DISCUSSION_READY: 2,
    PublicationReadiness.RESULTS_READY: 3,
    PublicationReadiness.HIGH_CONFIDENCE_RESULTS_READY: 4,
}

UNCERTAINTY_RANK = {
    UncertaintyLevel.VERY_HIGH: 0,
    UncertaintyLevel.HIGH: 1,
    UncertaintyLevel.MODERATE: 2,
    UncertaintyLevel.LOW: 3,
    UncertaintyLevel.VERY_LOW: 4,
}


def reviewer_confidence_for(
    evidence_level: EvidenceLevel,
    uncertainty_level: UncertaintyLevel,
    *,
    traceable: bool,
    source_validated: bool,
    claim_status: str,
) -> tuple[ReviewerConfidence, str]:
    evidence_level = EvidenceLevel(evidence_level)
    uncertainty_level = UncertaintyLevel(uncertainty_level)
    if not traceable or not source_validated or evidence_level is EvidenceLevel.INSUFFICIENT:
        return ReviewerConfidence.LOW, "Evidence is insufficient for publication-facing use."
    if claim_status == "TENTATIVE" or uncertainty_level in (UncertaintyLevel.VERY_HIGH, UncertaintyLevel.HIGH):
        return ReviewerConfidence.GUARDED, "The claim is suitable only for cautious use because uncertainty remains high."
    if claim_status == "CONFLICTED":
        return ReviewerConfidence.GUARDED, "The claim remains guarded because unresolved conflict is present."
    if evidence_level in (EvidenceLevel.STRONG, EvidenceLevel.VERY_STRONG) and uncertainty_level in (
        UncertaintyLevel.LOW,
        UncertaintyLevel.VERY_LOW,
    ):
        return ReviewerConfidence.HIGH, "The claim has strong traceable support under the applicable validation boundary."
    return ReviewerConfidence.MODERATE, "The claim is adequately supported for cautious reporting under internal evaluation conditions."


def publication_readiness_for(
    *,
    claim_publication_use: str,
    claim_type: str,
    claim_status: str,
    evidence_level: EvidenceLevel,
    uncertainty_level: UncertaintyLevel,
    has_external_validation: bool,
    traceable: bool,
    is_withheld: bool,
) -> tuple[PublicationReadiness, tuple[str, ...], str]:
    ceilings: list[str] = []
    if is_withheld or not traceable:
        return (
            PublicationReadiness.NOT_READY,
            ("critical traceability or withholding policy ceiling",),
            "The evidence is insufficient for publication-facing use.",
        )

    evidence_level = EvidenceLevel(evidence_level)
    uncertainty_level = UncertaintyLevel(uncertainty_level)
    if evidence_level in (EvidenceLevel.INSUFFICIENT, EvidenceLevel.LIMITED):
        base = PublicationReadiness.DISCUSSION_READY if claim_publication_use == "DISCUSSION_ELIGIBLE" else PublicationReadiness.NOT_READY
    elif evidence_level is EvidenceLevel.MODERATE:
        base = PublicationReadiness.DISCUSSION_READY
    elif evidence_level is EvidenceLevel.STRONG:
        base = PublicationReadiness.RESULTS_READY if uncertainty_level not in (UncertaintyLevel.VERY_HIGH, UncertaintyLevel.HIGH) else PublicationReadiness.DISCUSSION_READY
    else:
        if uncertainty_level is UncertaintyLevel.VERY_LOW and has_external_validation:
            base = PublicationReadiness.HIGH_CONFIDENCE_RESULTS_READY
        else:
            base = PublicationReadiness.RESULTS_READY
            if not has_external_validation:
                ceilings.append("no genuine external validation")

    if claim_type == "LIMITATION":
        base = PublicationReadiness.LIMITATION_ONLY
        ceilings.append("limitation claims cannot become results-ready")
    if claim_status == "CONFLICTED" and base is PublicationReadiness.HIGH_CONFIDENCE_RESULTS_READY:
        base = PublicationReadiness.RESULTS_READY
        ceilings.append("conflicted claims cannot reach high-confidence results readiness")
    if claim_status == "TENTATIVE" and base in (PublicationReadiness.RESULTS_READY, PublicationReadiness.HIGH_CONFIDENCE_RESULTS_READY):
        base = PublicationReadiness.DISCUSSION_READY
        ceilings.append("tentative claims cannot exceed discussion readiness")
    if not has_external_validation and base is PublicationReadiness.HIGH_CONFIDENCE_RESULTS_READY:
        base = PublicationReadiness.RESULTS_READY
        ceilings.append("no genuine external validation")

    ceiling = PUBLICATION_CEILINGS.get(claim_publication_use, PublicationReadiness.NOT_READY)
    if READINESS_RANK[base] > READINESS_RANK[ceiling]:
        base = ceiling
        ceilings.append(f"Claim Engine publication_use ceiling: {claim_publication_use}")

    explanation = _readiness_explanation(base)
    return base, tuple(ceilings), explanation


def _readiness_explanation(readiness: PublicationReadiness) -> str:
    if readiness is PublicationReadiness.RESULTS_READY:
        return "The claim is adequately supported for cautious reporting in the Results section under internal evaluation conditions."
    if readiness is PublicationReadiness.DISCUSSION_READY:
        return "The claim is suitable for Discussion but should not be presented as a definitive result."
    if readiness is PublicationReadiness.LIMITATION_ONLY:
        return "The claim should be reported only as a limitation."
    if readiness is PublicationReadiness.HIGH_CONFIDENCE_RESULTS_READY:
        return "The claim has unusually strong support and genuine external-validation evidence."
    return "The evidence is insufficient for publication-facing use."
