from dataclasses import FrozenInstanceError

import pytest

from src.scientific_reasoning.reasoning_graph import (
    GRAPH_SCHEMA_VERSION,
    GRAPH_SOFTWARE_VERSION,
    GraphEdge,
    GraphEdgeType,
    GraphNode,
    GraphNodeType,
    ReasoningGraph,
)


def test_graph_node_is_immutable_and_serializes_enum_values() -> None:
    node = GraphNode(
        node_id="OBS-0001",
        node_type=GraphNodeType.OBSERVATION,
        label="Observation",
        attributes={"metric_names": ("accuracy_mean", "f1_macro_mean")},
    )

    assert node.to_dict()["node_type"] == "Observation"
    assert node.to_dict()["attributes"]["metric_names"] == ["accuracy_mean", "f1_macro_mean"]
    with pytest.raises(FrozenInstanceError):
        node.label = "Changed"


def test_reasoning_graph_orders_nodes_and_edges_deterministically() -> None:
    graph = ReasoningGraph(
        graph_id="GRAPH-test",
        schema_version=GRAPH_SCHEMA_VERSION,
        software_version=GRAPH_SOFTWARE_VERSION,
        generated_at="2026-07-31T00:00:00+00:00",
        nodes=(
            GraphNode("HYP-0001", GraphNodeType.HYPOTHESIS, "Hypothesis"),
            GraphNode("OBS-0001", GraphNodeType.OBSERVATION, "Observation"),
        ),
        edges=(
            GraphEdge("EDGE-2", "HYP-0001", "OBS-0001", GraphEdgeType.LIMITED_BY),
            GraphEdge("EDGE-1", "OBS-0001", "HYP-0001", GraphEdgeType.SUPPORTS),
        ),
    )

    assert [node.node_id for node in graph.nodes] == ["HYP-0001", "OBS-0001"]
    assert [edge.edge_id for edge in graph.edges] == ["EDGE-2", "EDGE-1"]
    assert graph.to_dict()["edges"][0]["edge_type"] == "limited_by"
