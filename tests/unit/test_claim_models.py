from dataclasses import FrozenInstanceError

import pytest

from src.scientific_reasoning.claim import (
    ClaimCategory,
    ClaimStatus,
    ClaimType,
    EvidenceStrength,
    PublicationUse,
    ScientificClaim,
    validate_claim,
)


def claim(**overrides) -> ScientificClaim:
    payload = {
        "claim_id": "CLM-CHEMICAL_DISCRIMINATION-0001",
        "category": ClaimCategory.CHEMICAL_DISCRIMINATION,
        "title": "Claim",
        "claim_text": "The current evidence supports partial discrimination under internal evaluation.",
        "claim_type": ClaimType.PRIMARY_FINDING,
        "claim_status": ClaimStatus.PARTIALLY_SUPPORTED,
        "evidence_strength": EvidenceStrength.MODERATE,
        "publication_use": PublicationUse.RESULTS_ELIGIBLE,
        "supporting_hypothesis_ids": ("HYP-1",),
        "supporting_interpretation_ids": ("INT-1",),
        "supporting_observation_ids": ("OBS-1",),
        "evidence_gap_ids": ("GAP-1",),
        "validation_summary_ids": ("VAL:workflow",),
        "reasoning_graph_node_ids": ("GAP-1", "HYP-1", "INT-1", "OBS-1", "VAL:workflow"),
        "limitations": ("No independent external validation is available.",),
        "rationale": "Fixture rationale.",
        "evidence_score": 60,
        "confidence_label": "MODERATE",
        "created_at": "2026-07-31T00:00:00+00:00",
        "source_hypothesis_schema_version": "BSIP-2.2.0",
        "source_graph_schema_version": "BSIP-3.1.0",
    }
    payload.update(overrides)
    return ScientificClaim(**payload)


def test_valid_immutable_claim_construction_and_serialization() -> None:
    item = claim(reasoning_graph_node_ids=("VAL:workflow", "OBS-1", "INT-1", "HYP-1", "GAP-1"))

    assert item.to_dict()["category"] == "CHEMICAL_DISCRIMINATION"
    assert item.reasoning_graph_node_ids == ("GAP-1", "HYP-1", "INT-1", "OBS-1", "VAL:workflow")
    with pytest.raises(FrozenInstanceError):
        item.title = "Changed"


def test_invalid_claim_id_is_reported() -> None:
    issues = validate_claim(claim(claim_id="BAD-ID"))
    assert any(issue.code == "INVALID_CLAIM_ID" for issue in issues)


def test_deterministic_ordering_issue_is_normalized_by_model() -> None:
    item = claim(supporting_observation_ids=("OBS-2", "OBS-1"))
    assert item.supporting_observation_ids == ("OBS-1", "OBS-2")
