import shutil
from pathlib import Path

from src.scientific_reasoning.reasoning_graph import (
    GraphEdgeType,
    GraphNodeType,
    ReasoningGraph,
    ReasoningGraphBuilder,
    graph_validation_summary,
    validate_reasoning_graph,
)


def copy_reasoning_outputs(tmp_path: Path) -> Path:
    source_root = Path.cwd()
    project_root = tmp_path / "project"
    for directory in (
        "scientific_observations",
        "scientific_interpretations",
        "scientific_hypotheses",
        "workflow",
    ):
        shutil.copytree(source_root / "outputs" / directory, project_root / "outputs" / directory)
    return project_root


def test_real_reasoning_graph_validation_has_no_orphans_or_cycles(tmp_path: Path) -> None:
    project_root = copy_reasoning_outputs(tmp_path)
    graph = ReasoningGraphBuilder(project_root=project_root).build()
    validation = graph_validation_summary(graph)

    assert validation["validation_passed"] is True
    assert validation["orphan_count"] == 0
    assert validation["cycle_count"] == 0
    assert validation["critical_issue_count"] == 0


def test_missing_hypothesis_interpretation_parent_is_critical(tmp_path: Path) -> None:
    project_root = copy_reasoning_outputs(tmp_path)
    graph = ReasoningGraphBuilder(project_root=project_root).build()
    hypothesis_id = next(node.node_id for node in graph.nodes if node.node_type == GraphNodeType.HYPOTHESIS)
    without_hypothesis_support = tuple(
        edge
        for edge in graph.edges
        if not (
            edge.target_id == hypothesis_id
            and edge.edge_type == GraphEdgeType.SUPPORTS
            and graph.node_by_id()[edge.source_id].node_type == GraphNodeType.INTERPRETATION
        )
    )
    broken = ReasoningGraph(
        graph_id=graph.graph_id,
        schema_version=graph.schema_version,
        software_version=graph.software_version,
        generated_at=graph.generated_at,
        nodes=graph.nodes,
        edges=without_hypothesis_support,
        metadata=graph.metadata,
    )

    issues = validate_reasoning_graph(broken)
    assert any(issue.code == "HYPOTHESIS_MISSING_INTERPRETATION_PARENT" for issue in issues)
    assert graph_validation_summary(broken)["validation_passed"] is False
