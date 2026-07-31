from src.scientific_reasoning.claim import (
    ClaimStatus,
    ClaimType,
    EvidenceStrength,
    PublicationUse,
)
from src.scientific_reasoning.claim.policies import (
    calculate_evidence_score,
    confidence_label_from_strength,
    evidence_strength_from_score,
    publication_use_for_claim,
)


def test_evidence_strength_mapping_boundaries() -> None:
    assert evidence_strength_from_score(0) is EvidenceStrength.NOT_ASSESSABLE
    assert evidence_strength_from_score(1) is EvidenceStrength.INSUFFICIENT
    assert evidence_strength_from_score(30) is EvidenceStrength.LIMITED
    assert evidence_strength_from_score(60) is EvidenceStrength.MODERATE
    assert evidence_strength_from_score(80) is EvidenceStrength.STRONG


def test_evidence_score_is_clamped_and_penalizes_gaps_and_competitors() -> None:
    score = calculate_evidence_score(
        ({"status": "PLAUSIBLE", "confidence": "MODERATE"},),
        supporting_interpretation_count=10,
        supporting_observation_count=10,
        competing_hypothesis_count=99,
        evidence_gap_count=99,
        graph_traceable=True,
        source_validation_passed=True,
    )

    assert 0 <= score <= 100
    assert score < 80


def test_publication_use_policy() -> None:
    assert publication_use_for_claim(
        claim_type=ClaimType.PRIMARY_FINDING,
        claim_status=ClaimStatus.PARTIALLY_SUPPORTED,
        evidence_strength=EvidenceStrength.MODERATE,
        has_critical_issue=False,
        category="CHEMICAL_DISCRIMINATION",
    ) is PublicationUse.RESULTS_ELIGIBLE
    assert publication_use_for_claim(
        claim_type=ClaimType.LIMITATION,
        claim_status=ClaimStatus.TENTATIVE,
        evidence_strength=EvidenceStrength.LIMITED,
        has_critical_issue=False,
        category="GENERALIZATION",
    ) is PublicationUse.LIMITATION_ONLY
    assert confidence_label_from_strength(EvidenceStrength.LIMITED).value == "LOW"
