from src.scientific_reasoning.evidence_scoring import EvidenceLevel
from src.scientific_reasoning.evidence_scoring.scorer import score_claim
from src.scientific_reasoning.evidence_scoring.traceability import ReasoningGraphIndex


def claim(**overrides) -> dict:
    payload = {
        "claim_id": "CLM-TEST-0001",
        "category": "TEST",
        "claim_type": "PRIMARY_FINDING",
        "claim_status": "PARTIALLY_SUPPORTED",
        "publication_use": "RESULTS_ELIGIBLE",
        "confidence_label": "MODERATE",
        "evidence_score": 5,
        "supporting_hypothesis_ids": ["HYP-1"],
        "supporting_interpretation_ids": ["INT-1"],
        "supporting_observation_ids": ["OBS-1"],
        "evidence_gap_ids": ["GAP-1"],
        "validation_summary_ids": ["VAL:workflow"],
        "reasoning_graph_node_ids": ["OBS-1", "INT-1", "HYP-1", "GAP-1", "VAL:workflow"],
        "limitations": ["No independent external validation is available."],
    }
    payload.update(overrides)
    return payload


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


def test_scorer_calculates_independent_score_not_claim_score_copy() -> None:
    record = score_claim(
        claim(),
        ReasoningGraphIndex(graph()),
        claim_validation_passed=True,
        graph_validation_passed=True,
        source_claim_schema_version="BSIP-3.2.0",
        source_graph_schema_version="BSIP-3.1.0",
    )

    assert record.normalized_score != 5
    assert set(record.dimension_scores)
    assert record.evidence_level in {EvidenceLevel.MODERATE, EvidenceLevel.STRONG}


def test_scorer_withholds_missing_traceability() -> None:
    record = score_claim(
        claim(reasoning_graph_node_ids=["MISSING"]),
        ReasoningGraphIndex(graph()),
        claim_validation_passed=True,
        graph_validation_passed=True,
        source_claim_schema_version="BSIP-3.2.0",
        source_graph_schema_version="BSIP-3.1.0",
    )

    assert record.is_withheld is True
    assert record.normalized_score == 0
