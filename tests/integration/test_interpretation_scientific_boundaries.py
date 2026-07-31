from src.scientific_reasoning.interpretation import (
    EvidenceDirection,
    Interpretation,
    InterpretationCategory,
    InterpretationConfidence,
    InterpretationEvidenceLink,
    InterpretationStatus,
    ScientificInterpretationEngine,
    validate_interpretation,
)

from tests.integration.interpretation_fixture import realistic_observations, write_observation_package


def generated_by_id(tmp_path, *, regression_r2=0.28, blind_labels=False):
    observations_dir = write_observation_package(
        tmp_path / "observations",
        realistic_observations(regression_r2=regression_r2, blind_labels=blind_labels),
    )
    result = ScientificInterpretationEngine(
        project_root=tmp_path,
        observations_dir=observations_dir,
        output_dir=tmp_path / "interpretations",
        overwrite=True,
    ).run()
    return {interpretation.interpretation_id: interpretation for interpretation in result.interpretations}


def test_classification_interpretation_uses_conservative_wording(tmp_path) -> None:
    interpretation = generated_by_id(tmp_path)["INT-CHEMICAL_CLASSIFICATION-0001"]
    assert "suggest" in interpretation.claim.lower()
    assert "chemical-class discrimination" in interpretation.claim
    assert "publication" not in interpretation.claim.lower()
    assert "good" not in interpretation.claim.lower()


def test_regression_wording_for_positive_r2_below_half(tmp_path) -> None:
    interpretation = generated_by_id(tmp_path, regression_r2=0.28)["INT-CONCENTRATION_REGRESSION-0001"]
    assert "concentration-related information is present" in interpretation.claim
    assert "substantial proportion of target variance remains unaccounted for" in interpretation.claim
    assert interpretation.status is InterpretationStatus.SUPPORTED


def test_regression_missing_r2_yields_insufficient_evidence(tmp_path) -> None:
    interpretation = generated_by_id(tmp_path, regression_r2=None)["INT-CONCENTRATION_REGRESSION-0001"]
    assert interpretation.status is InterpretationStatus.INSUFFICIENT_EVIDENCE
    assert "insufficient" in interpretation.claim


def test_blind_label_absence_uses_required_boundary_wording(tmp_path) -> None:
    interpretation = generated_by_id(tmp_path, blind_labels=False)["INT-BLIND_VALIDATION-0001"]
    assert (
        interpretation.claim
        == "The available blind-prediction observations do not establish external validation performance because true labels were absent."
    )
    assert interpretation.status is InterpretationStatus.PARTIALLY_SUPPORTED


def test_generated_claims_do_not_use_restricted_scientific_language(tmp_path) -> None:
    interpretations = generated_by_id(tmp_path).values()
    restricted = ("proves", "confirms", "mechanism", "pathway", "publication-ready", "excellent", "strong")
    for interpretation in interpretations:
        claim = interpretation.claim.lower()
        assert all(term not in claim for term in restricted)


def test_causal_language_rejection() -> None:
    assert "FORBIDDEN_CAUSAL_LANGUAGE" in issue_codes(
        validate_interpretation(make_boundary_interpretation("The observation proves chemical identity."))
    )


def test_recommendation_language_rejection() -> None:
    assert "RECOMMENDATION_LANGUAGE" in issue_codes(
        validate_interpretation(make_boundary_interpretation("The team should test a new panel."))
    )


def test_hypothesis_language_rejection() -> None:
    assert "HYPOTHESIS_LANGUAGE" in issue_codes(
        validate_interpretation(make_boundary_interpretation("We hypothesize that the mechanism is receptor binding."))
    )


def test_blind_validation_overclaim_rejection() -> None:
    interpretation = make_boundary_interpretation(
        "External validation was achieved using blind-prediction outputs.",
        interpretation_id="INT-BLIND_VALIDATION-0001",
        category=InterpretationCategory.BLIND_VALIDATION,
        observation_id="OBS-BLIND_PREDICTION-0001",
    )
    assert "BLIND_VALIDATION_OVERCLAIM" in issue_codes(validate_interpretation(interpretation))


def make_boundary_interpretation(
    claim: str,
    *,
    interpretation_id: str = "INT-CHEMICAL_CLASSIFICATION-0001",
    category: InterpretationCategory = InterpretationCategory.CHEMICAL_CLASSIFICATION,
    observation_id: str = "OBS-CLASSIFICATION-0001",
) -> Interpretation:
    return Interpretation(
        interpretation_id=interpretation_id,
        category=category,
        title="Boundary fixture",
        claim=claim,
        status=InterpretationStatus.SUPPORTED,
        confidence=InterpretationConfidence.MODERATE,
        supporting_observation_ids=(observation_id,),
        evidence_summary=(
            InterpretationEvidenceLink(
                observation_id=observation_id,
                direction=EvidenceDirection.SUPPORTING,
                rationale="Synthetic boundary fixture.",
            ),
        ),
        reasoning_rule_ids=("RULE-BOUNDARY-001",),
        created_at="2026-07-31T00:00:00+00:00",
        software_version="BSIP-2.1.0-test",
        source_observation_schema_version="BSIP-2.0",
    )


def issue_codes(issues):
    return {issue.code for issue in issues}
