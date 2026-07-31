import json
from dataclasses import FrozenInstanceError

import pytest

from src.scientific_reasoning.observation import (
    ConfidenceLevel,
    Observation,
    ObservationCategory,
    ObservationStatus,
    ProvenanceRecord,
    SupportingMetric,
)


def valid_observation(**overrides):
    provenance = ProvenanceRecord(
        provenance_id="P-0001",
        source_file="classification/stage_8a/best_model_metrics.json",
        source_run="classification/stage_8a",
        section="Classification",
        claim_text="Extra Trees selected",
        metric_name="accuracy_mean",
        metric_value=0.740959,
        units=None,
        model_name="Extra Trees",
        support_status="SUPPORTED",
    )
    metric = SupportingMetric(
        metric_name="accuracy_mean",
        metric_value=0.740959,
        model_name="Extra Trees",
        fold_count=10,
        sample_count=9485,
        source_file="classification/stage_8a/best_model_metrics.json",
        source_run="classification/stage_8a",
        provenance_id="P-0001",
    )
    payload = {
        "observation_id": "OBS-CLASSIFICATION-0001",
        "category": ObservationCategory.CLASSIFICATION,
        "title": "Selected classification model",
        "statement": "Classification metadata lists Extra Trees as rank 1.",
        "status": ObservationStatus.COMPLETE,
        "analysis_stage": "Stage 8A",
        "supporting_metrics": (metric,),
        "supporting_files": ("classification/stage_8a/best_model_metrics.json",),
        "provenance_records": (provenance,),
        "confidence": ConfidenceLevel.HIGH,
        "limitations": (),
        "software_version": "BSIP-2.0",
        "source_run": "classification/stage_8a",
        "tags": ("classification", "contract"),
        "metadata": {"schema": "bsip_observation_v2"},
    }
    payload.update(overrides)
    return Observation(**payload)


def test_valid_observation_construction() -> None:
    observation = valid_observation()
    assert observation.observation_id == "OBS-CLASSIFICATION-0001"
    assert observation.category is ObservationCategory.CLASSIFICATION
    assert observation.status is ObservationStatus.COMPLETE
    assert observation.confidence is ConfidenceLevel.HIGH


def test_observation_is_immutable() -> None:
    observation = valid_observation()
    with pytest.raises(FrozenInstanceError):
        observation.title = "Changed"


def test_supporting_metric_accepts_missing_values() -> None:
    metric = SupportingMetric(metric_name="median_absolute_error_mean", metric_value=None)
    assert metric.metric_value is None
    assert metric.to_record()["metric_value"] is None


def test_enum_serialization_uses_string_values() -> None:
    record = valid_observation().to_record()
    assert record["category"] == "CLASSIFICATION"
    assert record["status"] == "COMPLETE"
    assert record["confidence"] == "HIGH"


def test_observation_record_is_json_serializable_with_sorted_keys() -> None:
    payload = valid_observation().to_record()
    encoded = json.dumps(payload, sort_keys=True)
    assert "OBS-CLASSIFICATION-0001" in encoded
    assert json.loads(encoded)["supporting_metrics"][0]["metric_name"] == "accuracy_mean"
