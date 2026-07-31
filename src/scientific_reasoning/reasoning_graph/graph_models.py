"""Immutable graph models for BSIP v3.1.0 reasoning graph."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


GRAPH_SOFTWARE_VERSION = "BSIP-3.1.0-reasoning-graph-engine"
GRAPH_SCHEMA_VERSION = "BSIP-3.1.0"


class GraphNodeType(str, Enum):
    DATASET = "Dataset"
    OBSERVATION = "Observation"
    INTERPRETATION = "Interpretation"
    HYPOTHESIS = "Hypothesis"
    EVIDENCE_GAP = "EvidenceGap"
    VALIDATION_SUMMARY = "ValidationSummary"
    WORKFLOW = "Workflow"


class GraphEdgeType(str, Enum):
    SUPPORTS = "supports"
    DERIVED_FROM = "derived_from"
    LIMITED_BY = "limited_by"
    COMPETES_WITH = "competes_with"
    VALIDATED_BY = "validated_by"
    GENERATED_BY = "generated_by"
    BELONGS_TO = "belongs_to"


class GraphValidationSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    node_type: GraphNodeType
    label: str
    source_id: str | None = None
    source_file: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_type", GraphNodeType(self.node_type))
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "label": self.label,
            "source_id": self.source_id,
            "source_file": self.source_file,
            "attributes": json_ready(dict(self.attributes)),
        }


@dataclass(frozen=True)
class GraphEdge:
    edge_id: str
    source_id: str
    target_id: str
    edge_type: GraphEdgeType
    label: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "edge_type", GraphEdgeType(self.edge_type))
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "label": self.label,
            "attributes": json_ready(dict(self.attributes)),
        }


@dataclass(frozen=True)
class ReasoningGraph:
    graph_id: str
    schema_version: str
    software_version: str
    generated_at: str
    nodes: tuple[GraphNode, ...] = field(default_factory=tuple)
    edges: tuple[GraphEdge, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(sorted(self.nodes, key=lambda node: node.node_id)))
        object.__setattr__(
            self,
            "edges",
            tuple(sorted(self.edges, key=lambda edge: (edge.source_id, edge.target_id, edge.edge_type.value, edge.edge_id))),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def node_by_id(self) -> dict[str, GraphNode]:
        return {node.node_id: node for node in self.nodes}

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "schema_version": self.schema_version,
            "software_version": self.software_version,
            "generated_at": self.generated_at,
            "metadata": json_ready(dict(self.metadata)),
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }


@dataclass(frozen=True)
class GraphValidationIssue:
    code: str
    severity: GraphValidationSeverity
    message: str
    node_id: str | None = None
    edge_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "severity", GraphValidationSeverity(self.severity))

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "node_id": self.node_id,
            "edge_id": self.edge_id,
        }


def json_ready(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return json_ready(asdict(value))
    if isinstance(value, MappingProxyType):
        return json_ready(dict(value))
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    return value
