from src.scientific_reasoning.reasoning_graph import (
    GRAPH_SCHEMA_VERSION,
    GRAPH_SOFTWARE_VERSION,
    GraphEdge,
    GraphEdgeType,
    GraphNode,
    GraphNodeType,
    ReasoningGraph,
    graph_validation_summary,
    validate_reasoning_graph,
)


def valid_graph() -> ReasoningGraph:
    nodes = (
        GraphNode("DATASET-test", GraphNodeType.DATASET, "Dataset"),
        GraphNode("BSIP-WF-test", GraphNodeType.WORKFLOW, "Workflow"),
        GraphNode("VAL-test", GraphNodeType.VALIDATION_SUMMARY, "Validation"),
        GraphNode("OBS-0001", GraphNodeType.OBSERVATION, "Observation"),
        GraphNode("INT-0001", GraphNodeType.INTERPRETATION, "Interpretation"),
        GraphNode("HYP-0001", GraphNodeType.HYPOTHESIS, "Hypothesis"),
        GraphNode("GAP-HYP-0001-0001", GraphNodeType.EVIDENCE_GAP, "Gap"),
    )
    edges = (
        GraphEdge("E-01", "OBS-0001", "INT-0001", GraphEdgeType.SUPPORTS),
        GraphEdge("E-02", "INT-0001", "HYP-0001", GraphEdgeType.SUPPORTS),
        GraphEdge("E-03", "OBS-0001", "DATASET-test", GraphEdgeType.DERIVED_FROM),
        GraphEdge("E-04", "HYP-0001", "GAP-HYP-0001-0001", GraphEdgeType.LIMITED_BY),
        GraphEdge("E-05", "OBS-0001", "VAL-test", GraphEdgeType.VALIDATED_BY),
        GraphEdge("E-06", "INT-0001", "VAL-test", GraphEdgeType.VALIDATED_BY),
        GraphEdge("E-07", "HYP-0001", "VAL-test", GraphEdgeType.VALIDATED_BY),
        GraphEdge("E-08", "GAP-HYP-0001-0001", "BSIP-WF-test", GraphEdgeType.BELONGS_TO),
        GraphEdge("E-09", "DATASET-test", "BSIP-WF-test", GraphEdgeType.BELONGS_TO),
        GraphEdge("E-10", "VAL-test", "BSIP-WF-test", GraphEdgeType.BELONGS_TO),
    )
    return ReasoningGraph(
        graph_id="GRAPH-test",
        schema_version=GRAPH_SCHEMA_VERSION,
        software_version=GRAPH_SOFTWARE_VERSION,
        generated_at="2026-07-31T00:00:00+00:00",
        nodes=nodes,
        edges=edges,
    )


def test_valid_graph_has_directional_edges_and_passes_validation() -> None:
    graph = valid_graph()
    summary = graph_validation_summary(graph)

    assert summary["validation_passed"] is True
    assert summary["orphan_count"] == 0
    assert summary["cycle_count"] == 0
    assert ("OBS-0001", "INT-0001") in {(edge.source_id, edge.target_id) for edge in graph.edges}


def test_missing_edge_target_is_reported() -> None:
    graph = ReasoningGraph(
        graph_id="GRAPH-bad",
        schema_version=GRAPH_SCHEMA_VERSION,
        software_version=GRAPH_SOFTWARE_VERSION,
        generated_at="2026-07-31T00:00:00+00:00",
        nodes=valid_graph().nodes,
        edges=valid_graph().edges + (GraphEdge("E-missing", "OBS-0001", "MISSING", GraphEdgeType.SUPPORTS),),
    )

    issues = validate_reasoning_graph(graph)
    assert any(issue.code == "MISSING_EDGE_TARGET" for issue in issues)


def test_evidence_cycles_are_detected() -> None:
    graph = ReasoningGraph(
        graph_id="GRAPH-cycle",
        schema_version=GRAPH_SCHEMA_VERSION,
        software_version=GRAPH_SOFTWARE_VERSION,
        generated_at="2026-07-31T00:00:00+00:00",
        nodes=valid_graph().nodes,
        edges=valid_graph().edges + (GraphEdge("E-cycle", "HYP-0001", "OBS-0001", GraphEdgeType.SUPPORTS),),
    )

    issues = validate_reasoning_graph(graph)
    assert any(issue.code == "EVIDENCE_CYCLE_DETECTED" for issue in issues)
