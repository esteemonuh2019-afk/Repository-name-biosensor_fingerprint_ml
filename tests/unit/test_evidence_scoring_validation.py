from src.scientific_reasoning.evidence_scoring import (
    DIMENSION_WEIGHTS,
    DimensionScore,
    EvidenceDimension,
    EvidenceLevel,
    EvidenceScoreRecord,
    PublicationReadiness,
    ReviewerConfidence,
    UncertaintyLevel,
    validate_evidence_score_records,
)


def record(**overrides) -> EvidenceScoreRecord:
    dimensions = {
        dimension: DimensionScore(dimension, 70, weight, 70 * weight, explanation="dimension")
        for dimension, weight in DIMENSION_WEIGHTS.items()
    }
    payload = {
        "claim_id": "CLM-TEST-0001",
        "claim_category": "TEST",
        "claim_type": "PRIMARY_FINDING",
        "claim_status": "PARTIALLY_SUPPORTED",
        "claim_publication_use": "RESULTS_ELIGIBLE",
        "dimension_scores": dimensions,
        "weighted_score": 70,
        "normalized_score": 70,
        "evidence_level": EvidenceLevel.STRONG,
        "uncertainty_level": UncertaintyLevel.LOW,
        "reviewer_confidence": ReviewerConfidence.MODERATE,
        "publication_readiness": PublicationReadiness.RESULTS_READY,
        "supporting_hypothesis_ids": ("HYP-1",),
        "supporting_interpretation_ids": ("INT-1",),
        "supporting_observation_ids": ("OBS-1",),
        "reasoning_graph_node_ids": ("HYP-1", "INT-1", "OBS-1"),
        "score_explanation": "Scores are deterministic support indices, not probabilities.",
    }
    payload.update(overrides)
    return EvidenceScoreRecord(**payload)


def test_validation_detects_missing_dimension() -> None:
    dimensions = dict(record().dimension_scores)
    dimensions.pop(EvidenceDimension.TRACEABILITY)
    issues = validate_evidence_score_records((record(dimension_scores=dimensions),))
    assert any(issue.code == "INVALID_DIMENSION" for issue in issues)


def test_validation_detects_publication_ceiling_violation() -> None:
    issues = validate_evidence_score_records((record(claim_publication_use="LIMITATION_ONLY"),))
    assert any(issue.code == "PUBLICATION_POLICY_ISSUE" for issue in issues)


def test_validation_detects_duplicate_claim_ids() -> None:
    issues = validate_evidence_score_records((record(), record()))
    assert any(issue.code == "DUPLICATE_CLAIM_ID" for issue in issues)
