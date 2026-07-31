from src.scientific_reasoning.observation import (
    ConfidenceLevel,
    Observation,
    ObservationCategory,
    ObservationStatus,
    ProvenanceRecord,
    SupportingMetric,
    validate_observation,
    validate_observations,
)


def make_observation(**overrides):
    provenance = ProvenanceRecord(
        provenance_id="P-0001",
        source_file="regression/stage_8b_2/best_regression_model.json",
        source_run="regression/stage_8b_2",
        section="Regression",
        claim_text="Extra Trees Regressor selected",
        metric_name="r2_mean",
        metric_value=0.288059,
        model_name="Extra Trees Regressor",
        support_status="SUPPORTED",
    )
    metric = SupportingMetric(
        metric_name="r2_mean",
        metric_value=0.288059,
        model_name="Extra Trees Regressor",
        source_file="regression/stage_8b_2/best_regression_model.json",
        source_run="regression/stage_8b_2",
        provenance_id="P-0001",
    )
    payload = {
        "observation_id": "OBS-REGRESSION-0001",
        "category": ObservationCategory.REGRESSION,
        "title": "Selected regression model",
        "statement": "Regression metadata lists Extra Trees Regressor as rank 1.",
        "status": ObservationStatus.COMPLETE,
        "analysis_stage": "Stage 8B",
        "supporting_metrics": (metric,),
        "supporting_files": ("regression/stage_8b_2/best_regression_model.json",),
        "provenance_records": (provenance,),
        "confidence": ConfidenceLevel.HIGH,
        "software_version": "BSIP-2.0",
    }
    payload.update(overrides)
    return Observation(**payload)


def issue_codes(issues):
    return {issue.code for issue in issues}


def test_invalid_id_rejection_contract() -> None:
    observation = make_observation(observation_id="OBS-REGRESSION-001")
    assert "INVALID_OBSERVATION_ID" in issue_codes(validate_observation(observation))


def test_id_category_mismatch_issue() -> None:
    observation = make_observation(observation_id="OBS-CLASSIFICATION-0001")
    assert "OBSERVATION_ID_CATEGORY_MISMATCH" in issue_codes(validate_observation(observation))


def test_missing_provenance_issue() -> None:
    metric = SupportingMetric(
        metric_name="r2_mean",
        metric_value=0.288059,
        provenance_id="P-MISSING",
        source_file="regression/stage_8b_2/best_regression_model.json",
    )
    observation = make_observation(supporting_metrics=(metric,))
    assert "MISSING_PROVENANCE" in issue_codes(validate_observation(observation))


def test_model_metric_mismatch_issue() -> None:
    provenance = ProvenanceRecord(
        provenance_id="P-0001",
        source_file="regression/stage_8b_2/best_regression_model.json",
        metric_name="r2_mean",
        metric_value=0.288059,
        model_name="XGBoost Regressor",
    )
    observation = make_observation(provenance_records=(provenance,))
    assert "MODEL_METRIC_MISMATCH" in issue_codes(validate_observation(observation))


def test_blind_validation_wording_issue() -> None:
    observation = make_observation(
        observation_id="OBS-BLIND_PREDICTION-0001",
        category=ObservationCategory.BLIND_PREDICTION,
        statement="Real blind validation confirmed the predicted chemical.",
    )
    assert "BLIND_VALIDATION_WORDING" in issue_codes(validate_observation(observation))


def test_duplicate_id_issue() -> None:
    first = make_observation(observation_id="OBS-REGRESSION-0001")
    second = make_observation(observation_id="OBS-REGRESSION-0001")
    assert "DUPLICATE_OBSERVATION_ID" in issue_codes(validate_observations((first, second)))


def test_deterministic_ordering_issue() -> None:
    first = make_observation(observation_id="OBS-REGRESSION-0002")
    second = make_observation(observation_id="OBS-REGRESSION-0001")
    assert "NON_DETERMINISTIC_ORDER" in issue_codes(validate_observations((first, second)))
