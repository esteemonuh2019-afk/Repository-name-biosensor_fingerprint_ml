from src.scientific_reasoning.observation import (
    ConfidenceLevel,
    Observation,
    ObservationCategory,
    ObservationStatus,
    ProvenanceRecord,
    SupportingMetric,
)
from src.scientific_reasoning.interpretation import (
    InterpretationConfidence,
    InterpretationStatus,
    assign_confidence,
    assign_status,
    find_blind_validation_overclaim_terms,
    find_forbidden_causal_terms,
    find_hypothesis_terms,
    find_literature_comparison_terms,
    find_recommendation_terms,
    supports_confidence_assignment,
)


def make_observation(
    observation_id: str,
    category: ObservationCategory = ObservationCategory.CLASSIFICATION,
    *,
    confidence: ConfidenceLevel = ConfidenceLevel.HIGH,
    status: ObservationStatus = ObservationStatus.COMPLETE,
) -> Observation:
    provenance = ProvenanceRecord(
        provenance_id=f"P-{observation_id}",
        source_file="observations.json",
        metric_name="fixture_metric",
        metric_value=1,
    )
    metric = SupportingMetric(
        metric_name="fixture_metric",
        metric_value=1,
        source_file="observations.json",
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


def test_policy_helpers_identify_restricted_language() -> None:
    assert "proves" in find_forbidden_causal_terms("The pattern proves a mechanism.")
    assert "recommend" in find_recommendation_terms("We recommend another assay.")
    assert "we hypothesize" in find_hypothesis_terms("We hypothesize a pathway.")
    assert "previous studies" in find_literature_comparison_terms("This exceeds previous studies.")
    assert "external validation was achieved" in find_blind_validation_overclaim_terms(
        "External validation was achieved."
    )


def test_confidence_policy_assigns_high_for_two_coherent_high_observations() -> None:
    observations = (
        make_observation("OBS-CLASSIFICATION-0001"),
        make_observation("OBS-FINGERPRINT-0001", ObservationCategory.FINGERPRINT),
    )
    assert assign_confidence(observations) is InterpretationConfidence.HIGH


def test_confidence_policy_assigns_moderate_for_one_strong_observation() -> None:
    assert assign_confidence((make_observation("OBS-CLASSIFICATION-0001"),)) is InterpretationConfidence.MODERATE


def test_confidence_policy_assigns_low_for_contextual_limitations() -> None:
    observations = (make_observation("OBS-QC-0001", ObservationCategory.QUALITY_CONTROL),)
    assert assign_confidence(observations, critical_qc_limitation=True) is InterpretationConfidence.LOW


def test_confidence_policy_assigns_low_for_incomplete_or_lower_confidence_evidence() -> None:
    observations = (
        make_observation(
            "OBS-FEATURE_SELECTION-0001",
            ObservationCategory.FEATURE_SELECTION,
            confidence=ConfidenceLevel.MODERATE,
            status=ObservationStatus.INCOMPLETE,
        ),
    )
    assert assign_confidence(observations) is InterpretationConfidence.LOW


def test_confidence_policy_assigns_not_assessable_when_evidence_absent() -> None:
    assert assign_confidence(()) is InterpretationConfidence.NOT_ASSESSABLE
    assert (
        assign_confidence(
            (make_observation("OBS-CLASSIFICATION-0001"),),
            observation_validation_failed_critically=True,
        )
        is InterpretationConfidence.NOT_ASSESSABLE
    )


def test_confidence_assignment_supports_no_stronger_than_expected() -> None:
    assert supports_confidence_assignment(InterpretationConfidence.LOW, InterpretationConfidence.MODERATE)
    assert not supports_confidence_assignment(InterpretationConfidence.HIGH, InterpretationConfidence.MODERATE)


def test_status_policy_contract() -> None:
    assert assign_status(2, minimum_supporting_observations=2) is InterpretationStatus.SUPPORTED
    assert assign_status(1, minimum_supporting_observations=2) is InterpretationStatus.PARTIALLY_SUPPORTED
    assert assign_status(1, contradicting_evidence_count=1) is InterpretationStatus.CONFLICTED
    assert assign_status(0) is InterpretationStatus.INSUFFICIENT_EVIDENCE
    assert assign_status(1, dependencies_valid=False) is InterpretationStatus.NOT_ASSESSABLE
