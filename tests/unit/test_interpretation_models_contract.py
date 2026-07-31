import json
from dataclasses import FrozenInstanceError

import pytest

from src.scientific_reasoning.interpretation import (
    EvidenceDirection,
    Interpretation,
    InterpretationCategory,
    InterpretationConfidence,
    InterpretationEvidenceLink,
    InterpretationStatus,
    ReasoningRule,
)


def valid_interpretation(**overrides) -> Interpretation:
    link = InterpretationEvidenceLink(
        observation_id="OBS-CLASSIFICATION-0001",
        direction=EvidenceDirection.SUPPORTING,
        rationale="Classification observation reports selected model and benchmark metrics.",
        metric_names=("accuracy_mean", "balanced_accuracy_mean", "f1_macro_mean"),
        provenance_ids=("P0001", "P0002"),
        source_files=("classification/stage_8a/best_model_metrics.json",),
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
        "tags": ("classification", "contract"),
        "metadata": {"schema": "bsip_interpretation_v2_1_0"},
    }
    payload.update(overrides)
    return Interpretation(**payload)


def test_valid_interpretation_construction() -> None:
    interpretation = valid_interpretation()
    assert interpretation.interpretation_id == "INT-CHEMICAL_CLASSIFICATION-0001"
    assert interpretation.category is InterpretationCategory.CHEMICAL_CLASSIFICATION
    assert interpretation.status is InterpretationStatus.SUPPORTED
    assert interpretation.confidence is InterpretationConfidence.MODERATE


def test_interpretation_is_immutable() -> None:
    interpretation = valid_interpretation()
    with pytest.raises(FrozenInstanceError):
        interpretation.title = "Changed"


def test_evidence_link_is_immutable_and_serializes_enums() -> None:
    link = valid_interpretation().evidence_summary[0]
    with pytest.raises(FrozenInstanceError):
        link.rationale = "Changed"
    assert link.to_record()["direction"] == "SUPPORTING"


def test_enum_serialization_uses_string_values() -> None:
    record = valid_interpretation().to_record()
    assert record["category"] == "CHEMICAL_CLASSIFICATION"
    assert record["status"] == "SUPPORTED"
    assert record["confidence"] == "MODERATE"
    assert record["evidence_summary"][0]["direction"] == "SUPPORTING"


def test_interpretation_record_is_json_serializable_with_sorted_dependencies() -> None:
    interpretation = valid_interpretation(
        supporting_observation_ids=("OBS-CLASSIFICATION-0002", "OBS-CLASSIFICATION-0001")
    )
    payload = interpretation.to_record()
    encoded = json.dumps(payload, sort_keys=True)
    assert json.loads(encoded)["supporting_observation_ids"] == [
        "OBS-CLASSIFICATION-0001",
        "OBS-CLASSIFICATION-0002",
    ]


def test_reasoning_rule_serializes_contract_fields() -> None:
    rule = ReasoningRule(
        rule_id="RULE-CLASSIFICATION-0001",
        name="Classification evidence wording",
        description="Allow conservative classification interpretations from validated observations.",
        required_categories=(InterpretationCategory.CHEMICAL_CLASSIFICATION,),
        optional_categories=(InterpretationCategory.BLIND_VALIDATION,),
        minimum_supporting_observations=1,
        allowed_claim_template="The available observations suggest ...",
        forbidden_terms=("proves", "confirms"),
        confidence_policy="observation_coherence",
        limitation_policy="inherit_observation_limitations",
        enabled=True,
    )
    record = rule.to_record()
    assert record["required_categories"] == ["CHEMICAL_CLASSIFICATION"]
    assert record["optional_categories"] == ["BLIND_VALIDATION"]
    assert json.loads(json.dumps(record, sort_keys=True))["rule_id"] == "RULE-CLASSIFICATION-0001"
