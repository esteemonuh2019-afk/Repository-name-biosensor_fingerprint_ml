from src.scientific_reasoning.evidence_scoring import UncertaintyLevel
from src.scientific_reasoning.evidence_scoring.traceability import ClaimTraceability
from src.scientific_reasoning.evidence_scoring.uncertainty import assess_uncertainty


def trace(*, external: bool = False, complete: bool = True) -> ClaimTraceability:
    return ClaimTraceability(
        claim_id="CLM-TEST-0001",
        referenced_node_ids=("OBS-1", "INT-1", "HYP-1"),
        missing_node_ids=(),
        supporting_hypothesis_ids=("HYP-1",),
        supporting_interpretation_ids=("INT-1",),
        supporting_observation_ids=("OBS-1",),
        evidence_gap_ids=("GAP-1",),
        validation_summary_ids=("VAL:workflow",),
        complete_support_chain=complete,
        has_external_validation=external,
    )


def test_uncertainty_is_not_inverse_of_evidence_score() -> None:
    claim = {
        "claim_status": "PARTIALLY_SUPPORTED",
        "competing_hypothesis_ids": ["HYP-2"],
        "limitations": ["No independent external validation is available."],
        "evidence_gaps": ["external validation gap"],
    }

    assessment = assess_uncertainty(claim, trace(external=False))

    assert assessment.uncertainty_level in {UncertaintyLevel.HIGH, UncertaintyLevel.VERY_HIGH}
    assert "no_external_validation" in assessment.uncertainty_penalties


def test_conflicted_claim_cannot_have_very_low_uncertainty() -> None:
    assessment = assess_uncertainty({"claim_status": "CONFLICTED"}, trace(external=True))
    assert assessment.uncertainty_level is not UncertaintyLevel.VERY_LOW
