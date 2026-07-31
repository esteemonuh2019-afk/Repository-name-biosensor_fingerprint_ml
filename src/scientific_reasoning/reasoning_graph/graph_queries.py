"""Deterministic query helpers for BSIP reasoning graphs."""

from __future__ import annotations

from collections import defaultdict, deque

from .graph_models import GraphEdgeType, GraphNodeType, ReasoningGraph


class ReasoningGraphQueries:
    def __init__(self, graph: ReasoningGraph) -> None:
        self.graph = graph
        self.nodes = graph.node_by_id()
        self.outgoing = _outgoing(graph)
        self.incoming = _incoming(graph)

    def find_support_chain(self, node_id: str) -> tuple[str, ...]:
        """Return support ancestors followed by the requested node."""

        self._require_node(node_id)
        ancestors = self._collect_upstream(node_id, edge_type=GraphEdgeType.SUPPORTS)
        ordered = _sort_by_type_then_id(self.graph, ancestors)
        return tuple(ordered + [node_id])

    def find_downstream(self, node_id: str) -> tuple[str, ...]:
        self._require_node(node_id)
        return tuple(_sort_by_type_then_id(self.graph, _reachable(self.outgoing, node_id)))

    def find_upstream(self, node_id: str) -> tuple[str, ...]:
        self._require_node(node_id)
        return tuple(_sort_by_type_then_id(self.graph, _reachable(self.incoming, node_id)))

    def find_competing_hypotheses(self, hypothesis_id: str) -> tuple[str, ...]:
        self._require_node(hypothesis_id)
        competitors = set()
        for edge in self.graph.edges:
            if edge.edge_type != GraphEdgeType.COMPETES_WITH:
                continue
            if edge.source_id == hypothesis_id:
                competitors.add(edge.target_id)
            if edge.target_id == hypothesis_id:
                competitors.add(edge.source_id)
        return tuple(sorted(competitors))

    def find_evidence_gaps(self, hypothesis_id: str) -> tuple[str, ...]:
        self._require_node(hypothesis_id)
        gaps = [
            edge.target_id
            for edge in self.graph.edges
            if edge.source_id == hypothesis_id
            and edge.edge_type == GraphEdgeType.LIMITED_BY
            and self.nodes.get(edge.target_id)
            and self.nodes[edge.target_id].node_type == GraphNodeType.EVIDENCE_GAP
        ]
        return tuple(sorted(gaps))

    def _collect_upstream(self, node_id: str, *, edge_type: GraphEdgeType) -> set[str]:
        adjacency: dict[str, set[str]] = defaultdict(set)
        for edge in self.graph.edges:
            if edge.edge_type == edge_type:
                adjacency[edge.target_id].add(edge.source_id)
        return _reachable(adjacency, node_id)

    def _require_node(self, node_id: str) -> None:
        if node_id not in self.nodes:
            raise KeyError(f"Node not found: {node_id}")


def find_support_chain(graph: ReasoningGraph, node_id: str) -> tuple[str, ...]:
    return ReasoningGraphQueries(graph).find_support_chain(node_id)


def find_downstream(graph: ReasoningGraph, node_id: str) -> tuple[str, ...]:
    return ReasoningGraphQueries(graph).find_downstream(node_id)


def find_upstream(graph: ReasoningGraph, node_id: str) -> tuple[str, ...]:
    return ReasoningGraphQueries(graph).find_upstream(node_id)


def find_competing_hypotheses(graph: ReasoningGraph, hypothesis_id: str) -> tuple[str, ...]:
    return ReasoningGraphQueries(graph).find_competing_hypotheses(hypothesis_id)


def find_evidence_gaps(graph: ReasoningGraph, hypothesis_id: str) -> tuple[str, ...]:
    return ReasoningGraphQueries(graph).find_evidence_gaps(hypothesis_id)


def _outgoing(graph: ReasoningGraph) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in graph.edges:
        adjacency[edge.source_id].add(edge.target_id)
    return adjacency


def _incoming(graph: ReasoningGraph) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in graph.edges:
        adjacency[edge.target_id].add(edge.source_id)
    return adjacency


def _reachable(adjacency: dict[str, set[str]], node_id: str) -> set[str]:
    seen = set()
    queue = deque(sorted(adjacency.get(node_id, ())))
    while queue:
        current = queue.popleft()
        if current in seen:
            continue
        seen.add(current)
        for child in sorted(adjacency.get(current, ())):
            if child not in seen:
                queue.append(child)
    return seen


def _sort_by_type_then_id(graph: ReasoningGraph, node_ids: set[str]) -> list[str]:
    type_order = {
        GraphNodeType.DATASET: 0,
        GraphNodeType.OBSERVATION: 1,
        GraphNodeType.INTERPRETATION: 2,
        GraphNodeType.HYPOTHESIS: 3,
        GraphNodeType.EVIDENCE_GAP: 4,
        GraphNodeType.VALIDATION_SUMMARY: 5,
        GraphNodeType.WORKFLOW: 6,
    }
    nodes = graph.node_by_id()
    return sorted(node_ids, key=lambda node_id: (type_order[nodes[node_id].node_type], node_id))
