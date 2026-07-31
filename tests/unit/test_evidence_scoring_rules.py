from src.scientific_reasoning.evidence_scoring import DIMENSION_WEIGHTS, EvidenceLevel, validate_weights
from src.scientific_reasoning.evidence_scoring.rules import evidence_level_from_score


def test_weights_sum_to_one() -> None:
    validate_weights()
    assert round(sum(DIMENSION_WEIGHTS.values()), 10) == 1.0


def test_evidence_level_thresholds() -> None:
    assert evidence_level_from_score(0) is EvidenceLevel.INSUFFICIENT
    assert evidence_level_from_score(25) is EvidenceLevel.LIMITED
    assert evidence_level_from_score(45) is EvidenceLevel.MODERATE
    assert evidence_level_from_score(65) is EvidenceLevel.STRONG
    assert evidence_level_from_score(80) is EvidenceLevel.VERY_STRONG
