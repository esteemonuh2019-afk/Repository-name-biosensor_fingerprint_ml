import json

import pytest

from src.scientific_reasoning.observation import (
    ConfidenceLevel,
    DuplicateObservationError,
    Observation,
    ObservationCategory,
    ObservationRegistry,
    ObservationStatus,
    ProvenanceRecord,
    SupportingMetric,
)


def make_observation(observation_id: str, category=ObservationCategory.DATASET) -> Observation:
    provenance = ProvenanceRecord(
        provenance_id=f"P-{observation_id}",
        source_file="features/feature_summary.json",
        source_run="features",
        section="Dataset",
        claim_text="Dataset row count",
        metric_name="feature_rows",
        metric_value=9762,
        support_status="SUPPORTED",
    )
    metric = SupportingMetric(
        metric_name="feature_rows",
        metric_value=9762,
        units="rows",
        source_file="features/feature_summary.json",
        source_run="features",
        provenance_id=f"P-{observation_id}",
    )
    return Observation(
        observation_id=observation_id,
        category=category,
        title="Dataset row count",
        statement="Dataset summary reports 9762 feature rows.",
        status=ObservationStatus.COMPLETE,
        analysis_stage="Stage 6B",
        supporting_metrics=(metric,),
        supporting_files=("features/feature_summary.json",),
        provenance_records=(provenance,),
        confidence=ConfidenceLevel.HIGH,
        software_version="BSIP-2.0",
    )


def test_registry_registers_and_returns_observation() -> None:
    registry = ObservationRegistry()
    observation = make_observation("OBS-DATASET-0001")
    registry.register(observation)
    assert registry.get("OBS-DATASET-0001") == observation


def test_registry_rejects_duplicate_ids() -> None:
    registry = ObservationRegistry()
    observation = make_observation("OBS-DATASET-0001")
    registry.register(observation)
    with pytest.raises(DuplicateObservationError):
        registry.register(observation)
    assert registry.validation_issues[-1].code == "DUPLICATE_OBSERVATION_ID"


def test_registry_rejects_invalid_ids() -> None:
    registry = ObservationRegistry()
    with pytest.raises(ValueError):
        registry.register(make_observation("OBS-DATASET-001"))
    assert registry.validation_issues[-1].code == "INVALID_OBSERVATION_ID"


def test_registry_returns_deterministic_ordering() -> None:
    registry = ObservationRegistry()
    registry.register(make_observation("OBS-DATASET-0002"))
    registry.register(make_observation("OBS-DATASET-0001"))
    assert [item.observation_id for item in registry.ordered()] == [
        "OBS-DATASET-0001",
        "OBS-DATASET-0002",
    ]


def test_registry_returns_observations_by_category() -> None:
    registry = ObservationRegistry()
    registry.register(make_observation("OBS-DATASET-0001"))
    registry.register(make_observation("OBS-QC-0001", ObservationCategory.QUALITY_CONTROL))
    assert [item.observation_id for item in registry.by_category(ObservationCategory.QUALITY_CONTROL)] == [
        "OBS-QC-0001"
    ]


def test_registry_exports_json_serializable_records() -> None:
    registry = ObservationRegistry()
    registry.register(make_observation("OBS-DATASET-0001"))
    records = registry.to_records()
    assert records[0]["category"] == "DATASET"
    assert json.loads(json.dumps(records, sort_keys=True))[0]["observation_id"] == "OBS-DATASET-0001"


def test_registry_exposes_validation_issues() -> None:
    registry = ObservationRegistry()
    with pytest.raises(ValueError):
        registry.register(make_observation("BAD-ID"))
    assert registry.validation_issues
