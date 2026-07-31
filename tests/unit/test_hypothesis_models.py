import json
from dataclasses import FrozenInstanceError

import pytest

from src.scientific_reasoning.hypothesis import (
    Hypothesis,
    HypothesisCategory,
    HypothesisConfidence,
    HypothesisPriority,
    HypothesisStatus,
)


def valid_hypothesis(**overrides) -> Hypothesis:
    payload = {
        "hypothesis_id": "HYP-CHEMICAL_DISCRIMINATION-0001",
        "category": HypothesisCategory.CHEMICAL_DISCRIMINATION,
        "title": "Chemical discrimination fixture",
        "statement": "Different chemical classes may produce partially distinct response patterns.",
        "status": HypothesisStatus.PLAUSIBLE,
        "confidence": HypothesisConfidence.MODERATE,
        "supporting_interpretation_ids": ("INT-CHEMICAL_CLASSIFICATION-0001",),
        "contradicting_interpretation_ids": (),
        "supporting_observation_ids": ("OBS-CLASSIFICATION-0001",),
        "assumptions": ("Interpretations are validated.",),
        "alternative_hypothesis_ids": (),
        "evidence_gaps": ("No external validation is available.",),
        "falsifiability_statement": "This hypothesis would be weakened if classification is not reproducible.",
        "rationale": "Synthetic rationale.",
        "reasoning_rule_ids": ("RULE-CHEMICAL-DISCRIMINATION-001",),
        "priority_score": 66,
        "priority": HypothesisPriority.MEDIUM,
        "software_version": "BSIP-2.2.0-test",
        "source_interpretation_schema_version": "BSIP-2.1.0",
        "tags": ("synthetic",),
        "metadata": {"fixture": True},
    }
    payload.update(overrides)
    return Hypothesis(**payload)


def test_valid_immutable_hypothesis_construction() -> None:
    hypothesis = valid_hypothesis()
    assert hypothesis.hypothesis_id == "HYP-CHEMICAL_DISCRIMINATION-0001"
    assert hypothesis.category is HypothesisCategory.CHEMICAL_DISCRIMINATION
    assert hypothesis.status is HypothesisStatus.PLAUSIBLE
    assert hypothesis.confidence is HypothesisConfidence.MODERATE
    with pytest.raises(FrozenInstanceError):
        hypothesis.title = "Changed"


def test_hypothesis_enum_serialization() -> None:
    record = valid_hypothesis().to_record()
    assert record["category"] == "CHEMICAL_DISCRIMINATION"
    assert record["status"] == "PLAUSIBLE"
    assert record["confidence"] == "MODERATE"
    assert record["priority"] == "MEDIUM"


def test_hypothesis_json_serializable_with_sorted_dependencies() -> None:
    hypothesis = valid_hypothesis(
        supporting_interpretation_ids=("INT-FINGERPRINT_STRUCTURE-0001", "INT-CHEMICAL_CLASSIFICATION-0001"),
        supporting_observation_ids=("OBS-FINGERPRINT-0001", "OBS-CLASSIFICATION-0001"),
    )
    record = hypothesis.to_record()
    encoded = json.dumps(record, sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded["supporting_interpretation_ids"] == [
        "INT-CHEMICAL_CLASSIFICATION-0001",
        "INT-FINGERPRINT_STRUCTURE-0001",
    ]
    assert decoded["supporting_observation_ids"] == ["OBS-CLASSIFICATION-0001", "OBS-FINGERPRINT-0001"]
