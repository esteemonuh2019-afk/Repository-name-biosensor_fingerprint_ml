"""Validation for deterministic BSIP reasoning graphs."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Any

from .graph_models import (
    GraphEdgeType,
    GraphNodeType,
    GraphValidationIssue,
    GraphValidationSeverity,
    ReasoningGraph,
)


EVIDENCE_EDGE_TYPES = {
    GraphEdgeType.SUPPORTS,
    GraphEdgeType.DERIVED_FROM,
    GraphEdgeType.LIMITED_BY,
}


def validate_reasoning_graph(graph: ReasoningGraph) -> tuple[GraphValidationIssue, ...]:
    issues: list[GraphValidationIssue] = []
    node_ids = {node.node_id for node in graph.nodes}
    node_types = {node.node_id: node.node_type for node in graph.nodes}

    issues.extend(_validate_unique_ids(graph))
    issues.extend(_validate_edge_references(graph, node_ids))
    issues.extend(_validate_interpretation_parents(graph, node_types))
    issues.extend(_validate_hypothesis_parents(graph, node_types))
    issues.extend(_validate_hypothesis_reachability(graph, node_types))
    issues.extend(_validate_orphans(graph))
    issues.extend(_validate_evidence_cycles(graph))
    issues.extend(_validate_deterministic_ordering(graph))
    return tuple(issues)


def graph_validation_summary(graph: ReasoningGraph) -> dict[str, Any]:
    issues = validate_reasoning_graph(graph)
    node_counts = Counter(node.node_type.value for node in graph.nodes)
    edge_counts = Counter(edge.edge_type.value for edge in graph.edges)
    orphan_count = sum(1 for issue in issues if issue.code == "ORPHAN_NODE")
    cycle_count = sum(1 for issue in issues if issue.code == "EVIDENCE_CYCLE_DETECTED")
    critical_count = sum(1 for issue in issues if issue.severity == GraphValidationSeverity.CRITICAL)
    warning_count = sum(1 for issue in issues if issue.severity == GraphValidationSeverity.WARNING)
    return {
        "validation_passed": critical_count == 0,
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "node_count_by_type": dict(sorted(node_counts.items())),
        "edge_count_by_type": dict(sorted(edge_counts.items())),
        "orphan_count": orphan_count,
        "cycle_count": cycle_count,
        "critical_issue_count": critical_count,
        "warning_count": warning_count,
        "structured_validation_issues": [issue.to_dict() for issue in issues],
    }


def _validate_unique_ids(graph: ReasoningGraph) -> tuple[GraphValidationIssue, ...]:
    issues = []
    for label, values in (
        ("node", [node.node_id for node in graph.nodes]),
        ("edge", [edge.edge_id for edge in graph.edges]),
    ):
        counts = Counter(values)
        for item_id, count in sorted(counts.items()):
            if count > 1:
                issues.append(
                    GraphValidationIssue(
                        code=f"DUPLICATE_{label.upper()}_ID",
                        severity=GraphValidationSeverity.CRITICAL,
                        message=f"Duplicate {label} ID: {item_id}",
                        node_id=item_id if label == "node" else None,
                        edge_id=item_id if label == "edge" else None,
                    )
                )
    return tuple(issues)


def _validate_edge_references(graph: ReasoningGraph, node_ids: set[str]) -> tuple[GraphValidationIssue, ...]:
    issues = []
    for edge in graph.edges:
        if edge.source_id not in node_ids:
            issues.append(
                GraphValidationIssue(
                    code="MISSING_EDGE_SOURCE",
                    severity=GraphValidationSeverity.CRITICAL,
                    message=f"Edge source does not exist: {edge.source_id}",
                    node_id=edge.source_id,
                    edge_id=edge.edge_id,
                )
            )
        if edge.target_id not in node_ids:
            issues.append(
                GraphValidationIssue(
                    code="MISSING_EDGE_TARGET",
                    severity=GraphValidationSeverity.CRITICAL,
                    message=f"Edge target does not exist: {edge.target_id}",
                    node_id=edge.target_id,
                    edge_id=edge.edge_id,
                )
            )
    return tuple(issues)


def _validate_interpretation_parents(
    graph: ReasoningGraph,
    node_types: dict[str, GraphNodeType],
) -> tuple[GraphValidationIssue, ...]:
    incoming_support = _incoming_by_type(graph, GraphEdgeType.SUPPORTS)
    issues = []
    for node in graph.nodes:
        if node.node_type != GraphNodeType.INTERPRETATION:
            continue
        parents = incoming_support.get(node.node_id, set())
        if not any(node_types.get(parent) == GraphNodeType.OBSERVATION for parent in parents):
            issues.append(
                GraphValidationIssue(
                    code="INTERPRETATION_MISSING_OBSERVATION_PARENT",
                    severity=GraphValidationSeverity.CRITICAL,
                    message=f"Interpretation has no observation parent: {node.node_id}",
                    node_id=node.node_id,
                )
            )
    return tuple(issues)


def _validate_hypothesis_parents(
    graph: ReasoningGraph,
    node_types: dict[str, GraphNodeType],
) -> tuple[GraphValidationIssue, ...]:
    incoming_support = _incoming_by_type(graph, GraphEdgeType.SUPPORTS)
    issues = []
    for node in graph.nodes:
        if node.node_type != GraphNodeType.HYPOTHESIS:
            continue
        parents = incoming_support.get(node.node_id, set())
        if not any(node_types.get(parent) == GraphNodeType.INTERPRETATION for parent in parents):
            issues.append(
                GraphValidationIssue(
                    code="HYPOTHESIS_MISSING_INTERPRETATION_PARENT",
                    severity=GraphValidationSeverity.CRITICAL,
                    message=f"Hypothesis has no interpretation parent: {node.node_id}",
                    node_id=node.node_id,
                )
            )
    return tuple(issues)


def _validate_hypothesis_reachability(
    graph: ReasoningGraph,
    node_types: dict[str, GraphNodeType],
) -> tuple[GraphValidationIssue, ...]:
    support_adjacency = _adjacency(graph, edge_types={GraphEdgeType.SUPPORTS})
    reachable_from_observations: set[str] = set()
    observation_ids = sorted(node_id for node_id, node_type in node_types.items() if node_type == GraphNodeType.OBSERVATION)
    for observation_id in observation_ids:
        reachable_from_observations.update(_bfs(support_adjacency, observation_id))
    issues = []
    for node_id, node_type in sorted(node_types.items()):
        if node_type == GraphNodeType.HYPOTHESIS and node_id not in reachable_from_observations:
            issues.append(
                GraphValidationIssue(
                    code="HYPOTHESIS_NOT_REACHABLE_FROM_OBSERVATION",
                    severity=GraphValidationSeverity.CRITICAL,
                    message=f"Hypothesis is not reachable from any observation: {node_id}",
                    node_id=node_id,
                )
            )
    return tuple(issues)


def _validate_orphans(graph: ReasoningGraph) -> tuple[GraphValidationIssue, ...]:
    degree = Counter()
    for edge in graph.edges:
        degree[edge.source_id] += 1
        degree[edge.target_id] += 1
    issues = []
    for node in graph.nodes:
        if degree[node.node_id] == 0:
            issues.append(
                GraphValidationIssue(
                    code="ORPHAN_NODE",
                    severity=GraphValidationSeverity.CRITICAL,
                    message=f"Node has no incident edges: {node.node_id}",
                    node_id=node.node_id,
                )
            )
    return tuple(issues)


def _validate_evidence_cycles(graph: ReasoningGraph) -> tuple[GraphValidationIssue, ...]:
    adjacency = _adjacency(graph, edge_types=EVIDENCE_EDGE_TYPES)
    visited: set[str] = set()
    active: set[str] = set()
    cycles: list[str] = []

    def visit(node_id: str) -> None:
        if node_id in active:
            cycles.append(node_id)
            return
        if node_id in visited:
            return
        active.add(node_id)
        for child in sorted(adjacency.get(node_id, ())):
            visit(child)
        active.remove(node_id)
        visited.add(node_id)

    for node in graph.nodes:
        visit(node.node_id)
    return tuple(
        GraphValidationIssue(
            code="EVIDENCE_CYCLE_DETECTED",
            severity=GraphValidationSeverity.CRITICAL,
            message=f"Evidence-edge cycle detected near node: {node_id}",
            node_id=node_id,
        )
        for node_id in sorted(set(cycles))
    )


def _validate_deterministic_ordering(graph: ReasoningGraph) -> tuple[GraphValidationIssue, ...]:
    issues = []
    if [node.node_id for node in graph.nodes] != sorted(node.node_id for node in graph.nodes):
        issues.append(
            GraphValidationIssue(
                code="NON_DETERMINISTIC_NODE_ORDER",
                severity=GraphValidationSeverity.CRITICAL,
                message="Graph nodes are not sorted deterministically by node_id.",
            )
        )
    edge_keys = [(edge.source_id, edge.target_id, edge.edge_type.value, edge.edge_id) for edge in graph.edges]
    if edge_keys != sorted(edge_keys):
        issues.append(
            GraphValidationIssue(
                code="NON_DETERMINISTIC_EDGE_ORDER",
                severity=GraphValidationSeverity.CRITICAL,
                message="Graph edges are not sorted deterministically.",
            )
        )
    return tuple(issues)


def _incoming_by_type(graph: ReasoningGraph, edge_type: GraphEdgeType) -> dict[str, set[str]]:
    incoming: dict[str, set[str]] = defaultdict(set)
    for edge in graph.edges:
        if edge.edge_type == edge_type:
            incoming[edge.target_id].add(edge.source_id)
    return incoming


def _adjacency(
    graph: ReasoningGraph,
    *,
    edge_types: set[GraphEdgeType],
) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in graph.edges:
        if edge.edge_type in edge_types:
            adjacency[edge.source_id].add(edge.target_id)
    return adjacency


def _bfs(adjacency: dict[str, set[str]], start: str) -> set[str]:
    seen = {start}
    queue = deque([start])
    while queue:
        node_id = queue.popleft()
        for child in sorted(adjacency.get(node_id, ())):
            if child in seen:
                continue
            seen.add(child)
            queue.append(child)
    return seen
