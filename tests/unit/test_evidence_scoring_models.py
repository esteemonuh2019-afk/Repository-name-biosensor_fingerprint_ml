from dataclasses import FrozenInstanceError

import pytest

from src.scientific_reasoning.evidence_scoring import (
    DIMENSION_WEIGHTS,
    DimensionScore,
    EvidenceDimension,
    EvidenceLevel,
    EvidenceScoreRecord,
    PublicationReadiness,
    ReviewerConfidence,
    UncertaintyLevel,
)


def record(**overrides) -> EvidenceScoreRecord:
    dimensions = {
        dimension: DimensionScore(
            dimension=dimension,
            raw_score=80,
            weight=weight,
            weighted_contribution=80 * weight,
            positive_factors=("factor",),
            explanation="dimension explanation",
        )
        for dimension, weight in DIMENSION_WEIGHTS.items()
    }
    payload = {
        "claim_id": "CLM-TEST-0001",
        "claim_category": "TEST",
        "claim_type": "PRIMARY_FINDING",
        "claim_status": "PARTIALLY_SUPPORTED",
        "claim_publication_use": "RESULTS_ELIGIBLE",
        "dimension_scores": dimensions,
        "weighted_score": 80,
        "normalized_score": 80,
        "evidence_level": EvidenceLevel.VERY_STRONG,
        "uncertainty_level": UncertaintyLevel.LOW,
        "reviewer_confidence": ReviewerConfidence.HIGH,
        "publication_readiness": PublicationReadiness.RESULTS_READY,
        "supporting_hypothesis_ids": ("HYP-1",),
        "supporting_interpretation_ids": ("INT-1",),
        "supporting_observation_ids": ("OBS-1",),
        "reasoning_graph_node_ids": ("OBS-1", "INT-1", "HYP-1", "VAL:workflow"),
        "score_explanation": "Scores are deterministic support indices, not probabilities.",
    }
    payload.update(overrides)
    return EvidenceScoreRecord(**payload)


def test_evidence_score_record_is_immutable_and_serializes_enums() -> None:
    item = record()

    assert item.to_dict()["evidence_level"] == "VERY_STRONG"
    assert "TRACEABILITY" in item.to_dict()["dimension_scores"]
    with pytest.raises(FrozenInstanceError):
        item.normalized_score = 0


def test_dimension_score_serializes_rule_context() -> None:
    score = DimensionScore(
        EvidenceDimension.TRACEABILITY,
        raw_score=75,
        weight=0.12,
        weighted_contribution=9,
        rule_ids=("B", "A"),
        source_node_ids=("OBS-2", "OBS-1"),
        explanation="Traceability score.",
    )

    assert score.to_dict()["rule_ids"] == ["A", "B"]
    assert score.to_dict()["source_node_ids"] == ["OBS-1", "OBS-2"]
