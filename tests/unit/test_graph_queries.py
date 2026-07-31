from src.scientific_reasoning.reasoning_graph import (
    GRAPH_SCHEMA_VERSION,
    GRAPH_SOFTWARE_VERSION,
    GraphEdge,
    GraphEdgeType,
    GraphNode,
    GraphNodeType,
    ReasoningGraph,
    ReasoningGraphQueries,
)


def query_graph() -> ReasoningGraph:
    nodes = (
        GraphNode("DATASET-test", GraphNodeType.DATASET, "Dataset"),
        GraphNode("BSIP-WF-test", GraphNodeType.WORKFLOW, "Workflow"),
        GraphNode("VAL-test", GraphNodeType.VALIDATION_SUMMARY, "Validation"),
        GraphNode("OBS-0001", GraphNodeType.OBSERVATION, "Observation"),
        GraphNode("INT-0001", GraphNodeType.INTERPRETATION, "Interpretation"),
        GraphNode("HYP-0001", GraphNodeType.HYPOTHESIS, "Hypothesis"),
        GraphNode("HYP-0002", GraphNodeType.HYPOTHESIS, "Alternative hypothesis"),
        GraphNode("GAP-HYP-0001-0001", GraphNodeType.EVIDENCE_GAP, "Gap"),
    )
    edges = (
        GraphEdge("E-01", "OBS-0001", "INT-0001", GraphEdgeType.SUPPORTS),
        GraphEdge("E-02", "INT-0001", "HYP-0001", GraphEdgeType.SUPPORTS),
        GraphEdge("E-03", "INT-0001", "HYP-0002", GraphEdgeType.SUPPORTS),
        GraphEdge("E-04", "HYP-0001", "GAP-HYP-0001-0001", GraphEdgeType.LIMITED_BY),
        GraphEdge("E-05", "HYP-0001", "HYP-0002", GraphEdgeType.COMPETES_WITH),
        GraphEdge("E-06", "OBS-0001", "DATASET-test", GraphEdgeType.DERIVED_FROM),
        GraphEdge("E-07", "DATASET-test", "BSIP-WF-test", GraphEdgeType.BELONGS_TO),
        GraphEdge("E-08", "VAL-test", "BSIP-WF-test", GraphEdgeType.BELONGS_TO),
        GraphEdge("E-09", "HYP-0002", "VAL-test", GraphEdgeType.VALIDATED_BY),
    )
    return ReasoningGraph(
        graph_id="GRAPH-query",
        schema_version=GRAPH_SCHEMA_VERSION,
        software_version=GRAPH_SOFTWARE_VERSION,
        generated_at="2026-07-31T00:00:00+00:00",
        nodes=nodes,
        edges=edges,
    )


def test_find_support_chain_orders_observation_to_hypothesis() -> None:
    assert ReasoningGraphQueries(query_graph()).find_support_chain("HYP-0001") == (
        "OBS-0001",
        "INT-0001",
        "HYP-0001",
    )


def test_find_downstream_and_upstream_are_deterministic() -> None:
    queries = ReasoningGraphQueries(query_graph())

    assert queries.find_downstream("OBS-0001") == (
        "DATASET-test",
        "INT-0001",
        "HYP-0001",
        "HYP-0002",
        "GAP-HYP-0001-0001",
        "VAL-test",
        "BSIP-WF-test",
    )
    assert queries.find_upstream("GAP-HYP-0001-0001") == (
        "OBS-0001",
        "INT-0001",
        "HYP-0001",
    )


def test_find_competing_hypotheses_and_evidence_gaps() -> None:
    queries = ReasoningGraphQueries(query_graph())

    assert queries.find_competing_hypotheses("HYP-0002") == ("HYP-0001",)
    assert queries.find_evidence_gaps("HYP-0001") == ("GAP-HYP-0001-0001",)
