from src.scientific_reasoning.evidence_scoring.traceability import ReasoningGraphIndex, trace_claim


def graph() -> dict:
    return {
        "nodes": [
            {"node_id": "OBS-1", "node_type": "Observation"},
            {"node_id": "INT-1", "node_type": "Interpretation"},
            {"node_id": "HYP-1", "node_type": "Hypothesis"},
            {"node_id": "VAL:workflow", "node_type": "ValidationSummary"},
        ],
        "edges": [
            {"source_id": "OBS-1", "target_id": "INT-1", "edge_type": "supports"},
            {"source_id": "INT-1", "target_id": "HYP-1", "edge_type": "supports"},
        ],
    }


def test_trace_claim_finds_complete_support_chain() -> None:
    claim = {
        "claim_id": "CLM-TEST-0001",
        "supporting_hypothesis_ids": ["HYP-1"],
        "supporting_interpretation_ids": ["INT-1"],
        "supporting_observation_ids": ["OBS-1"],
        "validation_summary_ids": ["VAL:workflow"],
        "reasoning_graph_node_ids": ["OBS-1", "INT-1", "HYP-1", "VAL:workflow"],
    }

    trace = trace_claim(claim, ReasoningGraphIndex(graph()))

    assert trace.complete_support_chain is True
    assert trace.support_paths == (("OBS-1", "INT-1", "HYP-1"),)


def test_trace_claim_reports_missing_graph_dependency() -> None:
    claim = {
        "claim_id": "CLM-TEST-0001",
        "supporting_hypothesis_ids": ["HYP-1"],
        "reasoning_graph_node_ids": ["MISSING"],
    }

    assert trace_claim(claim, ReasoningGraphIndex(graph())).missing_node_ids == ("MISSING",)
