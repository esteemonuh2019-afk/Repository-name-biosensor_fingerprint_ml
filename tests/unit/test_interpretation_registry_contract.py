import json

import pytest

from src.scientific_reasoning.interpretation import (
    DuplicateInterpretationError,
    EvidenceDirection,
    Interpretation,
    InterpretationCategory,
    InterpretationConfidence,
    InterpretationEvidenceLink,
    InterpretationRegistry,
    InterpretationStatus,
)


def make_interpretation(
    interpretation_id: str,
    category: InterpretationCategory = InterpretationCategory.DATASET_SCOPE,
    observation_id: str = "OBS-DATASET-0001",
) -> Interpretation:
    link = InterpretationEvidenceLink(
        observation_id=observation_id,
        direction=EvidenceDirection.SUPPORTING,
        rationale="Observation provides the factual source dependency.",
        metric_names=("sample_count",),
        provenance_ids=("P-DATASET",),
        source_files=("observations.json",),
    )
    return Interpretation(
        interpretation_id=interpretation_id,
        category=category,
        title="Dataset scope interpretation",
        claim="The dataset observations indicate the scope of the analyzed biosensor records.",
        status=InterpretationStatus.SUPPORTED,
        confidence=InterpretationConfidence.MODERATE,
        supporting_observation_ids=(observation_id,),
        evidence_summary=(link,),
        reasoning_rule_ids=("RULE-DATASET-0001",),
        software_version="BSIP-2.1.0",
        source_observation_schema_version="BSIP-2.0",
    )


def test_registry_registers_and_returns_interpretation() -> None:
    registry = InterpretationRegistry()
    interpretation = make_interpretation("INT-DATASET_SCOPE-0001")
    registry.register(interpretation)
    assert registry.get("INT-DATASET_SCOPE-0001") == interpretation


def test_registry_rejects_duplicate_ids() -> None:
    registry = InterpretationRegistry()
    interpretation = make_interpretation("INT-DATASET_SCOPE-0001")
    registry.register(interpretation)
    with pytest.raises(DuplicateInterpretationError):
        registry.register(interpretation)
    assert registry.validation_issues[-1].code == "DUPLICATE_INTERPRETATION_ID"


def test_registry_rejects_invalid_ids() -> None:
    registry = InterpretationRegistry()
    with pytest.raises(ValueError):
        registry.register(make_interpretation("INT-DATASET_SCOPE-001"))
    assert registry.validation_issues[-1].code == "INVALID_INTERPRETATION_ID"


def test_registry_returns_deterministic_ordering() -> None:
    registry = InterpretationRegistry()
    registry.register(make_interpretation("INT-DATASET_SCOPE-0002", observation_id="OBS-DATASET-0002"))
    registry.register(make_interpretation("INT-DATASET_SCOPE-0001", observation_id="OBS-DATASET-0001"))
    assert [item.interpretation_id for item in registry.ordered()] == [
        "INT-DATASET_SCOPE-0001",
        "INT-DATASET_SCOPE-0002",
    ]


def test_registry_returns_interpretations_by_category() -> None:
    registry = InterpretationRegistry()
    registry.register(make_interpretation("INT-DATASET_SCOPE-0001"))
    registry.register(
        make_interpretation(
            "INT-DATA_QUALITY-0001",
            InterpretationCategory.DATA_QUALITY,
            observation_id="OBS-QC-0001",
        )
    )
    assert [item.interpretation_id for item in registry.by_category(InterpretationCategory.DATA_QUALITY)] == [
        "INT-DATA_QUALITY-0001"
    ]


def test_registry_returns_interpretations_by_supporting_observation_id() -> None:
    registry = InterpretationRegistry()
    registry.register(make_interpretation("INT-DATASET_SCOPE-0001", observation_id="OBS-DATASET-0001"))
    registry.register(make_interpretation("INT-DATASET_SCOPE-0002", observation_id="OBS-DATASET-0002"))
    assert [item.interpretation_id for item in registry.by_supporting_observation_id("OBS-DATASET-0002")] == [
        "INT-DATASET_SCOPE-0002"
    ]


def test_registry_exports_json_serializable_records() -> None:
    registry = InterpretationRegistry()
    registry.register(make_interpretation("INT-DATASET_SCOPE-0001"))
    records = registry.to_records()
    assert records[0]["category"] == "DATASET_SCOPE"
    assert json.loads(json.dumps(records, sort_keys=True))[0]["interpretation_id"] == "INT-DATASET_SCOPE-0001"


def test_registry_exposes_validation_issues() -> None:
    registry = InterpretationRegistry()
    with pytest.raises(ValueError):
        registry.register(make_interpretation("BAD-ID"))
    assert registry.validation_issues
