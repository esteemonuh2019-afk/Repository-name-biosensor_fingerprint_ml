from src.scientific_reasoning.interpretation import (
    EvidenceDirection,
    Interpretation,
    InterpretationCategory,
    InterpretationConfidence,
    InterpretationEvidenceLink,
    InterpretationStatus,
)
from src.scientific_reasoning.hypothesis import (
    HypothesisCategory,
    HypothesisConfidence,
    HypothesisPriority,
    assign_confidence,
    falsifiability_is_valid,
    find_forbidden_hypothesis_terms,
    find_recommendation_terms,
    priority_from_score,
    priority_score,
    supports_confidence_assignment,
)


def interpretation(
    interpretation_id: str,
    confidence: InterpretationConfidence = InterpretationConfidence.MODERATE,
) -> Interpretation:
    return Interpretation(
        interpretation_id=interpretation_id,
        category=InterpretationCategory.CHEMICAL_CLASSIFICATION,
        title="Synthetic interpretation",
        claim="Synthetic interpretation suggests evidence.",
        status=InterpretationStatus.SUPPORTED,
        confidence=confidence,
        supporting_observation_ids=("OBS-CLASSIFICATION-0001",),
        evidence_summary=(
            InterpretationEvidenceLink(
                observation_id="OBS-CLASSIFICATION-0001",
                direction=EvidenceDirection.SUPPORTING,
                rationale="Synthetic evidence.",
            ),
        ),
        reasoning_rule_ids=("RULE-SYNTHETIC-001",),
        created_at="2026-07-31T00:00:00+00:00",
        software_version="BSIP-2.1.0-test",
        source_observation_schema_version="BSIP-2.0",
    )


def test_language_policy_helpers() -> None:
    assert "definitely" in find_forbidden_hypothesis_terms("This definitely establishes the claim.")
    assert "should test" in find_recommendation_terms("The team should test a method.")


def test_falsifiability_policy_requires_weakening_evidence() -> None:
    assert falsifiability_is_valid("This hypothesis would be weakened if the pattern is not reproducible.")
    assert not falsifiability_is_valid("")
    assert not falsifiability_is_valid("Perform an experiment using protocol details.")


def test_confidence_policy_high_requires_multiple_coherent_interpretations_without_major_gap() -> None:
    supporting = (
        interpretation("INT-CHEMICAL_CLASSIFICATION-0001", InterpretationConfidence.HIGH),
        interpretation("INT-FINGERPRINT_STRUCTURE-0001", InterpretationConfidence.HIGH),
        interpretation("INT-FEATURE_ENGINEERING-0001", InterpretationConfidence.MODERATE),
    )
    assert assign_confidence(supporting, evidence_gap_count=0) is HypothesisConfidence.HIGH


def test_confidence_policy_downgrades_external_validation_gap() -> None:
    supporting = (
        interpretation("INT-CHEMICAL_CLASSIFICATION-0001"),
        interpretation("INT-FINGERPRINT_STRUCTURE-0001"),
    )
    assert (
        assign_confidence(supporting, evidence_gap_count=2, external_validation_gap=True)
        is HypothesisConfidence.MODERATE
    )


def test_confidence_policy_low_for_single_supporting_interpretation() -> None:
    assert assign_confidence((interpretation("INT-CONCENTRATION_REGRESSION-0001"),), evidence_gap_count=2) is HypothesisConfidence.LOW


def test_confidence_assignment_supports_no_stronger_than_expected() -> None:
    assert supports_confidence_assignment(HypothesisConfidence.LOW, HypothesisConfidence.MODERATE)
    assert not supports_confidence_assignment(HypothesisConfidence.HIGH, HypothesisConfidence.MODERATE)


def test_priority_score_boundaries() -> None:
    supporting = (
        interpretation("INT-CHEMICAL_CLASSIFICATION-0001"),
        interpretation("INT-FINGERPRINT_STRUCTURE-0001"),
    )
    score = priority_score(
        HypothesisCategory.CHEMICAL_DISCRIMINATION,
        supporting,
        confidence=HypothesisConfidence.MODERATE,
        evidence_gap_count=2,
    )
    assert 0 <= score <= 100
    assert priority_from_score(0) is HypothesisPriority.NOT_ASSESSABLE
    assert priority_from_score(20) is HypothesisPriority.LOW
    assert priority_from_score(55) is HypothesisPriority.MEDIUM
    assert priority_from_score(80) is HypothesisPriority.HIGH
