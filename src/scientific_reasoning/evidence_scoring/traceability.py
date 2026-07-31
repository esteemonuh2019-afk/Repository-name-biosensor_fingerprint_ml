"""Traceability helpers for claim-to-graph evidence scoring."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ClaimTraceability:
    claim_id: str
    referenced_node_ids: tuple[str, ...]
    missing_node_ids: tuple[str, ...]
    supporting_hypothesis_ids: tuple[str, ...]
    supporting_interpretation_ids: tuple[str, ...]
    supporting_observation_ids: tuple[str, ...]
    evidence_gap_ids: tuple[str, ...]
    validation_summary_ids: tuple[str, ...]
    support_paths: tuple[tuple[str, ...], ...] = field(default_factory=tuple)
    complete_support_chain: bool = False
    has_external_validation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "referenced_node_ids": list(self.referenced_node_ids),
            "missing_node_ids": list(self.missing_node_ids),
            "supporting_hypothesis_ids": list(self.supporting_hypothesis_ids),
            "supporting_interpretation_ids": list(self.supporting_interpretation_ids),
            "supporting_observation_ids": list(self.supporting_observation_ids),
            "evidence_gap_ids": list(self.evidence_gap_ids),
            "validation_summary_ids": list(self.validation_summary_ids),
            "support_paths": [list(path) for path in self.support_paths],
            "complete_support_chain": self.complete_support_chain,
            "has_external_validation": self.has_external_validation,
        }


class ReasoningGraphIndex:
    def __init__(self, graph_document: dict[str, Any]) -> None:
        self.graph_document = graph_document
        self.nodes_by_id = {str(node.get("node_id")): dict(node) for node in graph_document.get("nodes", ())}
        self.edges = tuple(dict(edge) for edge in graph_document.get("edges", ()))
        self.parents_by_target: dict[str, set[str]] = defaultdict(set)
        self.children_by_source: dict[str, set[str]] = defaultdict(set)
        for edge in self.edges:
            source = str(edge.get("source_id"))
            target = str(edge.get("target_id"))
            if edge.get("edge_type") == "supports":
                self.parents_by_target[target].add(source)
                self.children_by_source[source].add(target)

    def node_type(self, node_id: str) -> str | None:
        node = self.nodes_by_id.get(node_id)
        return None if node is None else str(node.get("node_type"))

    def node_exists(self, node_id: str) -> bool:
        return node_id in self.nodes_by_id

    def validation_summary_ids(self) -> tuple[str, ...]:
        return tuple(sorted(node_id for node_id, node in self.nodes_by_id.items() if node.get("node_type") == "ValidationSummary"))

    def support_ancestors(self, node_id: str) -> tuple[str, ...]:
        seen: set[str] = set()
        queue = deque(sorted(self.parents_by_target.get(node_id, ())))
        while queue:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            for parent in sorted(self.parents_by_target.get(current, ())):
                if parent not in seen:
                    queue.append(parent)
        return tuple(sorted(seen))

    def support_paths_to_observations(self, hypothesis_id: str) -> tuple[tuple[str, ...], ...]:
        paths: list[tuple[str, ...]] = []

        def visit(node_id: str, path: tuple[str, ...]) -> None:
            parents = sorted(self.parents_by_target.get(node_id, ()))
            if not parents:
                if self.node_type(node_id) == "Observation":
                    paths.append(tuple(reversed(path)))
                return
            for parent in parents:
                if parent in path:
                    continue
                visit(parent, path + (parent,))

        visit(hypothesis_id, (hypothesis_id,))
        return tuple(sorted(paths))

    def has_complete_support_chain(self, hypothesis_id: str) -> bool:
        ancestors = self.support_ancestors(hypothesis_id)
        return any(self.node_type(node_id) == "Interpretation" for node_id in ancestors) and any(
            self.node_type(node_id) == "Observation" for node_id in ancestors
        )


def trace_claim(claim: dict[str, Any], graph_index: ReasoningGraphIndex) -> ClaimTraceability:
    referenced = set(str(item) for item in claim.get("reasoning_graph_node_ids", ()) or ())
    referenced.update(str(item) for item in claim.get("supporting_hypothesis_ids", ()) or ())
    referenced.update(str(item) for item in claim.get("supporting_interpretation_ids", ()) or ())
    referenced.update(str(item) for item in claim.get("supporting_observation_ids", ()) or ())
    referenced.update(str(item) for item in claim.get("evidence_gap_ids", ()) or ())
    referenced.update(str(item) for item in claim.get("validation_summary_ids", ()) or ())
    missing = tuple(sorted(node_id for node_id in referenced if not graph_index.node_exists(node_id)))
    hypothesis_ids = tuple(sorted(str(item) for item in claim.get("supporting_hypothesis_ids", ()) or ()))
    support_paths = tuple(path for hypothesis_id in hypothesis_ids for path in graph_index.support_paths_to_observations(hypothesis_id))
    complete = bool(hypothesis_ids) and all(graph_index.has_complete_support_chain(hypothesis_id) for hypothesis_id in hypothesis_ids)
    external_validation = _has_genuine_external_validation_signal(claim, graph_index)
    return ClaimTraceability(
        claim_id=str(claim.get("claim_id")),
        referenced_node_ids=tuple(sorted(referenced)),
        missing_node_ids=missing,
        supporting_hypothesis_ids=hypothesis_ids,
        supporting_interpretation_ids=tuple(sorted(str(item) for item in claim.get("supporting_interpretation_ids", ()) or ())),
        supporting_observation_ids=tuple(sorted(str(item) for item in claim.get("supporting_observation_ids", ()) or ())),
        evidence_gap_ids=tuple(sorted(str(item) for item in claim.get("evidence_gap_ids", ()) or ())),
        validation_summary_ids=tuple(sorted(str(item) for item in claim.get("validation_summary_ids", ()) or ())),
        support_paths=support_paths,
        complete_support_chain=complete and not missing,
        has_external_validation=external_validation,
    )


def _has_genuine_external_validation_signal(claim: dict[str, Any], graph_index: ReasoningGraphIndex) -> bool:
    text = " ".join(
        [
            " ".join(str(item) for item in claim.get("limitations", ()) or ()),
            " ".join(str(item) for item in claim.get("evidence_gaps", ()) or ()),
        ]
    ).lower()
    if "no independent external validation" in text or "no true external validation" in text:
        return False
    for node_id in claim.get("supporting_observation_ids", ()) or ():
        node = graph_index.nodes_by_id.get(str(node_id), {})
        haystack = " ".join(str(value) for value in node.values()).lower()
        if "external" in haystack and "independent" in haystack and "true_labels" in haystack:
            return True
    return False
