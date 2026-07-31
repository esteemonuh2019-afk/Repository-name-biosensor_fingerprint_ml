from src.scientific_reasoning.evidence_scoring import EvidenceLevel, PublicationReadiness, UncertaintyLevel
from src.scientific_reasoning.evidence_scoring.publication_policy import publication_readiness_for, reviewer_confidence_for


def test_publication_readiness_respects_claim_engine_ceiling() -> None:
    readiness, ceilings, _ = publication_readiness_for(
        claim_publication_use="DISCUSSION_ELIGIBLE",
        claim_type="PRIMARY_FINDING",
        claim_status="PARTIALLY_SUPPORTED",
        evidence_level=EvidenceLevel.VERY_STRONG,
        uncertainty_level=UncertaintyLevel.VERY_LOW,
        has_external_validation=True,
        traceable=True,
        is_withheld=False,
    )

    assert readiness is PublicationReadiness.DISCUSSION_READY
    assert ceilings


def test_high_confidence_results_ready_requires_external_validation() -> None:
    readiness, ceilings, _ = publication_readiness_for(
        claim_publication_use="RESULTS_ELIGIBLE",
        claim_type="PRIMARY_FINDING",
        claim_status="PARTIALLY_SUPPORTED",
        evidence_level=EvidenceLevel.VERY_STRONG,
        uncertainty_level=UncertaintyLevel.VERY_LOW,
        has_external_validation=False,
        traceable=True,
        is_withheld=False,
    )

    assert readiness is PublicationReadiness.RESULTS_READY
    assert "no genuine external validation" in ceilings


def test_tentative_claim_cannot_receive_high_reviewer_confidence() -> None:
    confidence, _ = reviewer_confidence_for(
        EvidenceLevel.STRONG,
        UncertaintyLevel.LOW,
        traceable=True,
        source_validated=True,
        claim_status="TENTATIVE",
    )
    assert confidence.value != "HIGH"
