from src.scientific_reasoning.hypothesis import (
    Hypothesis,
    HypothesisCategory,
    HypothesisConfidence,
    HypothesisPriority,
    HypothesisStatus,
    validate_hypothesis,
)
from src.scientific_reasoning.hypothesis.engine import HypothesisEngine

from tests.integration.hypothesis_fixture import write_interpretation_package


def generated_by_id(tmp_path):
    interpretations_dir = write_interpretation_package(tmp_path / "interpretations")
    result = HypothesisEngine(
        project_root=tmp_path,
        interpretations_dir=interpretations_dir,
        output_dir=tmp_path / "hypotheses",
        overwrite=True,
    ).run()
    return {hypothesis.hypothesis_id: hypothesis for hypothesis in result.hypotheses}


def test_temporal_information_hypothesis_wording(tmp_path) -> None:
    hypothesis = generated_by_id(tmp_path)["HYP-TEMPORAL_INFORMATION-0001"]
    assert "may contribute" in hypothesis.statement
    assert "proves" not in hypothesis.statement.lower()
    assert hypothesis.falsifiability_statement


def test_generalization_not_high_confidence(tmp_path) -> None:
    hypothesis = generated_by_id(tmp_path)["HYP-GENERALIZATION-0001"]
    assert hypothesis.status is HypothesisStatus.WEAKLY_SUPPORTED
    assert hypothesis.confidence is HypothesisConfidence.LOW
    assert "independently labelled unknown samples" in hypothesis.statement


def test_overall_system_behavior_cites_classification_and_regression(tmp_path) -> None:
    hypothesis = generated_by_id(tmp_path)["HYP-OVERALL_SYSTEM_BEHAVIOR-0001"]
    assert "INT-CHEMICAL_CLASSIFICATION-0001" in hypothesis.supporting_interpretation_ids
    assert "INT-CONCENTRATION_REGRESSION-0001" in hypothesis.supporting_interpretation_ids
    assert any("external validation" in gap.lower() for gap in hypothesis.evidence_gaps)


def test_competing_hypotheses_are_linked(tmp_path) -> None:
    by_id = generated_by_id(tmp_path)
    pairs = (
        ("HYP-CHEMICAL_DISCRIMINATION-0001", "HYP-CHEMICAL_DISCRIMINATION-0002"),
        ("HYP-CONCENTRATION_ENCODING-0001", "HYP-CONCENTRATION_ENCODING-0002"),
        ("HYP-FEATURE_REPRESENTATION-0001", "HYP-FEATURE_REPRESENTATION-0002"),
        ("HYP-STRAIN_CONTRIBUTION-0001", "HYP-STRAIN_CONTRIBUTION-0002"),
    )
    for first, second in pairs:
        assert second in by_id[first].alternative_hypothesis_ids
        assert first in by_id[second].alternative_hypothesis_ids


def test_generated_hypotheses_do_not_contain_recommendations_or_established_fact_language(tmp_path) -> None:
    restricted = ("proves", "confirms", "definitely", "certainly", "should test", "recommend")
    for hypothesis in generated_by_id(tmp_path).values():
        statement = hypothesis.statement.lower()
        falsifiability = (hypothesis.falsifiability_statement or "").lower()
        assert all(term not in statement for term in restricted)
        assert "should test" not in falsifiability
        assert "recommend" not in falsifiability


def test_causal_overclaim_rejection() -> None:
    assert "CAUSAL_OVERCLAIM" in codes(validate_hypothesis(boundary_hypothesis("The mechanism is definitely known.")))


def test_recommendation_language_rejection() -> None:
    assert "RECOMMENDATION_LANGUAGE" in codes(
        validate_hypothesis(boundary_hypothesis("The team should test an added assay."))
    )


def test_missing_falsifiability_rejection() -> None:
    assert "MISSING_FALSIFIABILITY" in codes(
        validate_hypothesis(boundary_hypothesis("A pattern may exist.", falsifiability_statement=""))
    )


def boundary_hypothesis(statement: str, *, falsifiability_statement: str | None = None) -> Hypothesis:
    return Hypothesis(
        hypothesis_id="HYP-CHEMICAL_DISCRIMINATION-0001",
        category=HypothesisCategory.CHEMICAL_DISCRIMINATION,
        title="Boundary fixture",
        statement=statement,
        status=HypothesisStatus.PLAUSIBLE,
        confidence=HypothesisConfidence.LOW,
        supporting_interpretation_ids=("INT-CHEMICAL_CLASSIFICATION-0001",),
        supporting_observation_ids=("OBS-CLASSIFICATION-0001",),
        assumptions=("Synthetic boundary fixture.",),
        evidence_gaps=("No external validation is available.",),
        falsifiability_statement=falsifiability_statement
        if falsifiability_statement is not None
        else "This hypothesis would be weakened if the pattern is not reproducible.",
        rationale="Synthetic rationale.",
        reasoning_rule_ids=("RULE-BOUNDARY-001",),
        priority_score=34,
        priority=HypothesisPriority.LOW,
        created_at="2026-07-31T00:00:00+00:00",
        software_version="BSIP-2.2.0-test",
        source_interpretation_schema_version="BSIP-2.1.0",
    )


def codes(issues):
    return {issue.code for issue in issues}
