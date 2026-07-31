from src.scientific_reasoning.interpretation import (
    EvidenceDirection,
    Interpretation,
    InterpretationCategory,
    InterpretationConfidence,
    InterpretationEvidenceLink,
    InterpretationStatus,
)
from src.scientific_reasoning.hypothesis import (
    Hypothesis,
    HypothesisCategory,
    HypothesisConfidence,
    HypothesisPriority,
    HypothesisStatus,
    validate_hypothesis,
    validate_hypotheses,
)


def make_interpretation(
    interpretation_id: str = "INT-CHEMICAL_CLASSIFICATION-0001",
    category: InterpretationCategory = InterpretationCategory.CHEMICAL_CLASSIFICATION,
    confidence: InterpretationConfidence = InterpretationConfidence.MODERATE,
) -> Interpretation:
    return Interpretation(
        interpretation_id=interpretation_id,
        category=category,
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


def make_hypothesis(**overrides) -> Hypothesis:
    payload = {
        "hypothesis_id": "HYP-CHEMICAL_DISCRIMINATION-0001",
        "category": HypothesisCategory.CHEMICAL_DISCRIMINATION,
        "title": "Synthetic hypothesis",
        "statement": "Different chemical classes may produce distinct response patterns.",
        "status": HypothesisStatus.PLAUSIBLE,
        "confidence": HypothesisConfidence.LOW,
        "supporting_interpretation_ids": ("INT-CHEMICAL_CLASSIFICATION-0001",),
        "supporting_observation_ids": ("OBS-CLASSIFICATION-0001",),
        "assumptions": ("Interpretations are validated.",),
        "evidence_gaps": ("No external validation is available.",),
        "falsifiability_statement": "This hypothesis would be weakened if classification is not reproducible.",
        "rationale": "Synthetic rationale.",
        "reasoning_rule_ids": ("RULE-CHEMICAL-DISCRIMINATION-001",),
        "priority_score": 34,
        "priority": HypothesisPriority.LOW,
        "created_at": "2026-07-31T00:00:00+00:00",
        "software_version": "BSIP-2.2.0-test",
        "source_interpretation_schema_version": "BSIP-2.1.0",
    }
    payload.update(overrides)
    return Hypothesis(**payload)


def codes(issues):
    return {issue.code for issue in issues}


def test_invalid_hypothesis_id() -> None:
    assert "INVALID_HYPOTHESIS_ID" in codes(validate_hypothesis(make_hypothesis(hypothesis_id="HYP-BAD-001")))


def test_missing_interpretation_dependency() -> None:
    hypothesis = make_hypothesis(supporting_interpretation_ids=("INT-MISSING-0001",))
    assert "MISSING_INTERPRETATION_DEPENDENCY" in codes(validate_hypothesis(hypothesis, interpretations=()))


def test_unsupported_hypothesis_without_support() -> None:
    hypothesis = make_hypothesis(supporting_interpretation_ids=())
    assert "UNSUPPORTED_HYPOTHESIS" in codes(validate_hypothesis(hypothesis))


def test_causal_overclaim_rejection() -> None:
    hypothesis = make_hypothesis(statement="The biosensor definitely establishes chemical identity.")
    assert "CAUSAL_OVERCLAIM" in codes(validate_hypothesis(hypothesis))


def test_recommendation_language_rejection() -> None:
    hypothesis = make_hypothesis(statement="The team should test a new protocol.")
    assert "RECOMMENDATION_LANGUAGE" in codes(validate_hypothesis(hypothesis))


def test_missing_falsifiability_statement() -> None:
    hypothesis = make_hypothesis(falsifiability_statement="")
    assert "MISSING_FALSIFIABILITY" in codes(validate_hypothesis(hypothesis))


def test_confidence_policy_behavior() -> None:
    interpretation = make_interpretation()
    hypothesis = make_hypothesis(confidence=HypothesisConfidence.HIGH)
    assert "CONFIDENCE_POLICY_ISSUE" in codes(validate_hypothesis(hypothesis, interpretations=(interpretation,)))


def test_priority_score_boundaries_and_priority_policy() -> None:
    assert "PRIORITY_SCORE_OUT_OF_RANGE" in codes(validate_hypothesis(make_hypothesis(priority_score=101)))
    assert "PRIORITY_POLICY_ISSUE" in codes(
        validate_hypothesis(make_hypothesis(priority_score=75, priority=HypothesisPriority.LOW))
    )


def test_competing_hypothesis_links() -> None:
    primary = make_hypothesis(
        hypothesis_id="HYP-CHEMICAL_DISCRIMINATION-0001",
        alternative_hypothesis_ids=("HYP-CHEMICAL_DISCRIMINATION-0002",),
    )
    competing = make_hypothesis(
        hypothesis_id="HYP-CHEMICAL_DISCRIMINATION-0002",
        status=HypothesisStatus.COMPETING,
        alternative_hypothesis_ids=("HYP-CHEMICAL_DISCRIMINATION-0001",),
        priority_score=34,
        priority=HypothesisPriority.LOW,
    )
    assert not codes(validate_hypotheses((primary, competing), interpretations=(make_interpretation(),)))


def test_deterministic_ordering_issue() -> None:
    hypothesis = make_hypothesis(
        supporting_interpretation_ids=("INT-FINGERPRINT_STRUCTURE-0001", "INT-CHEMICAL_CLASSIFICATION-0001")
    )
    assert "DETERMINISTIC_ORDERING_ISSUE" in codes(validate_hypothesis(hypothesis))
