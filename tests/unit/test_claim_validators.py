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


def graph() -> dict:
    return {
        "nodes": [
            {"node_id": "OBS-1", "node_type": "Observation"},
            {"node_id": "INT-1", "node_type": "Interpretation"},
            {"node_id": "HYP-1", "node_type": "Hypothesis"},
            {"node_id": "GAP-1", "node_type": "EvidenceGap"},
            {"node_id": "VAL:workflow", "node_type": "ValidationSummary"},
        ],
        "edges": [
            {"source_id": "OBS-1", "target_id": "INT-1", "edge_type": "supports"},
            {"source_id": "INT-1", "target_id": "HYP-1", "edge_type": "supports"},
        ],
    }


def test_missing_hypothesis_and_graph_dependencies_are_reported() -> None:
    issues = validate_claim(claim(), hypotheses_by_id={}, graph_document={"nodes": [], "edges": []})
    assert any(issue.code == "MISSING_HYPOTHESIS_DEPENDENCY" for issue in issues)
    assert any(issue.code == "MISSING_GRAPH_DEPENDENCY" for issue in issues)


def test_unsupported_and_missing_limitation_claims_are_rejected() -> None:
    item = claim(supporting_hypothesis_ids=(), limitations=())
    issues = validate_claim(item, hypotheses_by_id={}, graph_document=graph())

    assert any(issue.code == "UNSUPPORTED_CLAIM" for issue in issues)
    assert any(issue.code == "MISSING_LIMITATION" for issue in issues)


def test_language_overclaim_rejections() -> None:
    texts_and_codes = [
        ("This proves the result.", "CAUSAL_OVERCLAIM"),
        ("A molecular mechanism explains the result.", "MECHANISM_OVERCLAIM"),
        ("This novel result is publication-ready.", "NOVELTY_CLAIM_ISSUE"),
        ("The model is externally validated and clinically useful.", "EXTERNAL_VALIDATION_OVERCLAIM"),
        ("Researchers should test a future experiment.", "UNSUPPORTED_CLAIM"),
    ]
    for text, code in texts_and_codes:
        assert any(issue.code == code for issue in validate_claim(claim(claim_text=text)))


def test_complete_traceability_is_required_for_active_claims() -> None:
    broken_graph = {"nodes": graph()["nodes"], "edges": []}
    issues = validate_claim(claim(), hypotheses_by_id={"HYP-1": {}}, graph_document=broken_graph)
    assert any(issue.code == "MISSING_TRACEABILITY" for issue in issues)


def test_withheld_claim_policy() -> None:
    item = claim(
        claim_type=ClaimType.WITHHELD,
        claim_status=ClaimStatus.WITHHELD,
        evidence_strength=EvidenceStrength.NOT_ASSESSABLE,
        publication_use=PublicationUse.NOT_ELIGIBLE,
        evidence_score=0,
        supporting_hypothesis_ids=(),
        reasoning_graph_node_ids=(),
        validation_summary_ids=(),
        limitations=("Required source missing.",),
        rationale="Required source missing.",
    )
    assert not validate_claim(item, hypotheses_by_id={}, graph_document=graph())
