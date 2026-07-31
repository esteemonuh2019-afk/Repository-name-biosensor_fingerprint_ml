from src.scientific_reasoning.observation import (
    ConfidenceLevel,
    Observation,
    ObservationCategory,
    ObservationStatus,
    ProvenanceRecord,
    SupportingMetric,
)
from src.scientific_reasoning.interpretation import (
    EvidenceDirection,
    Interpretation,
    InterpretationCategory,
    InterpretationConfidence,
    InterpretationEvidenceLink,
    InterpretationStatus,
    validate_interpretation,
    validate_interpretations,
)


def make_observation(
    observation_id: str = "OBS-CLASSIFICATION-0001",
    category: ObservationCategory = ObservationCategory.CLASSIFICATION,
    *,
    confidence: ConfidenceLevel = ConfidenceLevel.HIGH,
    status: ObservationStatus = ObservationStatus.COMPLETE,
) -> Observation:
    provenance = ProvenanceRecord(
        provenance_id=f"P-{observation_id}",
        source_file="observations.json",
        source_run="scientific_observations",
        section=category.value,
        claim_text="Synthetic observation fixture",
        metric_name="fixture_metric",
        metric_value=1.0,
        support_status="SUPPORTED",
    )
    metric = SupportingMetric(
        metric_name="fixture_metric",
        metric_value=1.0,
        source_file="observations.json",
        source_run="scientific_observations",
        provenance_id=f"P-{observation_id}",
    )
    return Observation(
        observation_id=observation_id,
        category=category,
        title="Synthetic observation",
        statement="Synthetic observation reports one fixture metric.",
        status=status,
        analysis_stage="Synthetic",
        supporting_metrics=(metric,),
        supporting_files=("observations.json",),
        provenance_records=(provenance,),
        confidence=confidence,
        software_version="BSIP-2.0",
    )


def make_interpretation(**overrides) -> Interpretation:
    link = InterpretationEvidenceLink(
        observation_id="OBS-CLASSIFICATION-0001",
        direction=EvidenceDirection.SUPPORTING,
        rationale="Classification observation supports the conservative claim.",
        metric_names=("accuracy_mean",),
        provenance_ids=("P-OBS-CLASSIFICATION-0001",),
        source_files=("observations.json",),
    )
    payload = {
        "interpretation_id": "INT-CHEMICAL_CLASSIFICATION-0001",
        "category": InterpretationCategory.CHEMICAL_CLASSIFICATION,
        "title": "Chemical-class discrimination evidence",
        "claim": (
            "The available classification observations suggest that biosensor fingerprints "
            "contain information associated with chemical-class discrimination."
        ),
        "status": InterpretationStatus.SUPPORTED,
        "confidence": InterpretationConfidence.MODERATE,
        "supporting_observation_ids": ("OBS-CLASSIFICATION-0001",),
        "contradicting_observation_ids": (),
        "assumptions": ("Observation inputs passed validation.",),
        "limitations": ("External blind labels were absent.",),
        "evidence_summary": (link,),
        "reasoning_rule_ids": ("RULE-CLASSIFICATION-0001",),
        "software_version": "BSIP-2.1.0",
        "source_observation_schema_version": "BSIP-2.0",
    }
    payload.update(overrides)
    return Interpretation(**payload)


def issue_codes(issues):
    return {issue.code for issue in issues}


def test_invalid_id_rejection_contract() -> None:
    interpretation = make_interpretation(interpretation_id="INT-CHEMICAL_CLASSIFICATION-001")
    assert "INVALID_INTERPRETATION_ID" in issue_codes(validate_interpretation(interpretation))


def test_id_category_mismatch_issue() -> None:
    interpretation = make_interpretation(interpretation_id="INT-DATA_QUALITY-0001")
    assert "INTERPRETATION_ID_CATEGORY_MISMATCH" in issue_codes(validate_interpretation(interpretation))


def test_duplicate_id_issue() -> None:
    first = make_interpretation(interpretation_id="INT-CHEMICAL_CLASSIFICATION-0001")
    second = make_interpretation(interpretation_id="INT-CHEMICAL_CLASSIFICATION-0001")
    assert "DUPLICATE_INTERPRETATION_ID" in issue_codes(validate_interpretations((first, second)))


def test_missing_supporting_observation_dependency_issue() -> None:
    link = InterpretationEvidenceLink(
        observation_id="OBS-CLASSIFICATION-0001",
        direction=EvidenceDirection.SUPPORTING,
        rationale="Evidence link is present but the supporting ID list is empty.",
    )
    interpretation = make_interpretation(supporting_observation_ids=(), evidence_summary=(link,))
    assert "MISSING_SUPPORTING_OBSERVATION" in issue_codes(validate_interpretation(interpretation))


def test_nonexistent_observation_dependency_issue() -> None:
    interpretation = make_interpretation(supporting_observation_ids=("OBS-CLASSIFICATION-9999",))
    assert "NONEXISTENT_OBSERVATION_DEPENDENCY" in issue_codes(
        validate_interpretation(interpretation, observations=())
    )


def test_interpretation_without_evidence_issue() -> None:
    interpretation = make_interpretation(supporting_observation_ids=(), evidence_summary=())
    assert "INTERPRETATION_WITHOUT_EVIDENCE" in issue_codes(validate_interpretation(interpretation))


def test_unsupported_confidence_assignment_issue() -> None:
    observation = make_observation()
    interpretation = make_interpretation(confidence=InterpretationConfidence.HIGH)
    assert "UNSUPPORTED_CONFIDENCE_ASSIGNMENT" in issue_codes(
        validate_interpretation(interpretation, observations=(observation,))
    )


def test_causal_language_rejection() -> None:
    interpretation = make_interpretation(claim="The biosensor result proves chemical identity.")
    assert "FORBIDDEN_CAUSAL_LANGUAGE" in issue_codes(validate_interpretation(interpretation))


def test_recommendation_language_rejection() -> None:
    interpretation = make_interpretation(claim="The team should test a future experiment next.")
    assert "RECOMMENDATION_LANGUAGE" in issue_codes(validate_interpretation(interpretation))


def test_hypothesis_language_rejection() -> None:
    interpretation = make_interpretation(claim="We hypothesize that the mechanism is receptor binding.")
    assert "HYPOTHESIS_LANGUAGE" in issue_codes(validate_interpretation(interpretation))


def test_literature_comparison_language_rejection() -> None:
    interpretation = make_interpretation(claim="The result is better than previous studies.")
    assert "LITERATURE_COMPARISON_LANGUAGE" in issue_codes(validate_interpretation(interpretation))


def test_blind_validation_overclaim_rejection() -> None:
    link = InterpretationEvidenceLink(
        observation_id="OBS-BLIND_PREDICTION-0001",
        direction=EvidenceDirection.SUPPORTING,
        rationale="Blind-prediction observation provides the source dependency.",
    )
    interpretation = make_interpretation(
        interpretation_id="INT-BLIND_VALIDATION-0001",
        category=InterpretationCategory.BLIND_VALIDATION,
        claim="External validation was achieved using the blind-prediction output.",
        supporting_observation_ids=("OBS-BLIND_PREDICTION-0001",),
        evidence_summary=(link,),
    )
    assert "BLIND_VALIDATION_OVERCLAIM" in issue_codes(validate_interpretation(interpretation))


def test_contradiction_not_recorded_issue() -> None:
    supporting = InterpretationEvidenceLink(
        observation_id="OBS-CLASSIFICATION-0001",
        direction=EvidenceDirection.SUPPORTING,
        rationale="Supporting fixture.",
    )
    contradicting = InterpretationEvidenceLink(
        observation_id="OBS-QC-0001",
        direction=EvidenceDirection.CONTRADICTING,
        rationale="Contradicting fixture.",
    )
    interpretation = make_interpretation(evidence_summary=(supporting, contradicting))
    assert "CONTRADICTION_NOT_RECORDED" in issue_codes(validate_interpretation(interpretation))


def test_non_serializable_metadata_issue() -> None:
    interpretation = make_interpretation(metadata={"bad": object()})
    assert "NON_SERIALIZABLE_METADATA" in issue_codes(validate_interpretation(interpretation))


def test_deterministic_interpretation_field_ordering_issue() -> None:
    link = InterpretationEvidenceLink(
        observation_id="OBS-CLASSIFICATION-0001",
        direction=EvidenceDirection.SUPPORTING,
        rationale="Supporting fixture.",
    )
    interpretation = make_interpretation(
        supporting_observation_ids=("OBS-CLASSIFICATION-0002", "OBS-CLASSIFICATION-0001"),
        evidence_summary=(link,),
    )
    assert "NON_DETERMINISTIC_ORDER" in issue_codes(validate_interpretation(interpretation))


def test_deterministic_batch_ordering_issue() -> None:
    first = make_interpretation(
        interpretation_id="INT-DATASET_SCOPE-0001",
        category=InterpretationCategory.DATASET_SCOPE,
        supporting_observation_ids=("OBS-DATASET-0001",),
        evidence_summary=(
            InterpretationEvidenceLink(
                observation_id="OBS-DATASET-0001",
                direction=EvidenceDirection.SUPPORTING,
                rationale="Dataset fixture.",
            ),
        ),
    )
    second = make_interpretation()
    assert "NON_DETERMINISTIC_ORDER" in issue_codes(validate_interpretations((first, second)))
