from src.scientific_reasoning.claim.policies import calculate_evidence_score


def test_one_deterministic_score_for_same_inputs() -> None:
    kwargs = {
        "supporting_interpretation_count": 2,
        "supporting_observation_count": 3,
        "competing_hypothesis_count": 1,
        "evidence_gap_count": 2,
        "graph_traceable": True,
        "source_validation_passed": True,
    }
    hypotheses = (
        {"status": "PLAUSIBLE", "confidence": "MODERATE"},
        {"status": "COMPETING", "confidence": "MODERATE"},
    )

    assert calculate_evidence_score(hypotheses, **kwargs) == calculate_evidence_score(hypotheses, **kwargs)


def test_missing_support_scores_zero() -> None:
    assert calculate_evidence_score(
        (),
        supporting_interpretation_count=0,
        supporting_observation_count=0,
        competing_hypothesis_count=0,
        evidence_gap_count=0,
        graph_traceable=False,
        source_validation_passed=False,
    ) == 0
