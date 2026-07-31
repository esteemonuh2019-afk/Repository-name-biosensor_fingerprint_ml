from dataclasses import FrozenInstanceError

import pytest

from src.scientific_reasoning.reviewer import ReviewCategory, ReviewFinding, ReviewerConfidence, ReviewerType, Severity


def test_review_finding_is_immutable_and_serializes_enums() -> None:
    finding = ReviewFinding(
        finding_id="REV-SCIENTIFIC-0001",
        reviewer_type=ReviewerType.SCIENTIFIC,
        category=ReviewCategory.CLAIM_SUPPORT,
        title="Traceable finding",
        finding_text="A source claim is linked to evidence.",
        severity=Severity.MODERATE,
        blocking=False,
        confidence=ReviewerConfidence.HIGH,
        affected_claim_ids=("CLM-B", "CLM-A"),
        rule_ids=("RULE-B", "RULE-A"),
    )

    record = finding.to_dict()

    assert record["reviewer_type"] == "SCIENTIFIC"
    assert record["category"] == "CLAIM_SUPPORT"
    assert record["severity"] == "MODERATE"
    assert record["affected_claim_ids"] == ["CLM-A", "CLM-B"]
    assert record["rule_ids"] == ["RULE-A", "RULE-B"]
    with pytest.raises(FrozenInstanceError):
        finding.title = "mutated"
