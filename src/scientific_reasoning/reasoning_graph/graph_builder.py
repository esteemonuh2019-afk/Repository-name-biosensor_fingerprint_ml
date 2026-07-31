"""Build a deterministic directed reasoning graph from validated BSIP artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .graph_models import (
    GRAPH_SCHEMA_VERSION,
    GRAPH_SOFTWARE_VERSION,
    GraphEdge,
    GraphEdgeType,
    GraphNode,
    GraphNodeType,
    ReasoningGraph,
)


class ReasoningGraphBuilder:
    """Construct a graph from existing outputs without rerunning reasoning engines."""

    def __init__(
        self,
        *,
        project_root: Path | str = ".",
        observations_dir: Path | str = "outputs/scientific_observations",
        interpretations_dir: Path | str = "outputs/scientific_interpretations",
        hypotheses_dir: Path | str = "outputs/scientific_hypotheses",
        workflow_dir: Path | str = "outputs/workflow",
        software_version: str = GRAPH_SOFTWARE_VERSION,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.observations_dir = self._resolve(observations_dir)
        self.interpretations_dir = self._resolve(interpretations_dir)
        self.hypotheses_dir = self._resolve(hypotheses_dir)
        self.workflow_dir = self._resolve(workflow_dir)
        self.software_version = software_version

    def build(self) -> ReasoningGraph:
        observations_doc = _read_json(self.observations_dir / "observations.json")
        interpretations_doc = _read_json(self.interpretations_dir / "interpretations.json")
        hypotheses_doc = _read_json(self.hypotheses_dir / "hypotheses.json")
        workflow_manifest = _read_json(self.workflow_dir / "workflow_manifest.json")
        observation_validation = _read_json(self.observations_dir / "observation_validation.json")
        interpretation_validation = _read_json(self.interpretations_dir / "interpretation_validation.json")
        hypothesis_validation = _read_json(self.hypotheses_dir / "hypothesis_validation.json")
        interpretation_dependencies = _read_csv(self.interpretations_dir / "interpretation_dependencies.csv")
        hypothesis_dependencies = _read_csv(self.hypotheses_dir / "hypothesis_dependencies.csv")
        competition_map = _read_csv(self.hypotheses_dir / "hypothesis_competition_map.csv")

        observations = tuple(observations_doc.get("observations", ()))
        interpretations = tuple(interpretations_doc.get("interpretations", ()))
        hypotheses = tuple(hypotheses_doc.get("hypotheses", ()))

        workflow_id = str(workflow_manifest.get("workflow_id") or "BSIP-WF-UNKNOWN")
        dataset_source = str(
            observations_doc.get("source_supervisor_results_directory")
            or workflow_manifest.get("source_dataset")
            or "unknown_dataset"
        )
        dataset_id = "DATASET:" + _stable_token(dataset_source)
        workflow_node_id = workflow_id
        generated_at = str(workflow_manifest.get("timestamp") or workflow_manifest.get("completed_at") or "")

        nodes: dict[str, GraphNode] = {}
        edges: dict[str, GraphEdge] = {}

        _add_node(
            nodes,
            GraphNode(
                node_id=dataset_id,
                node_type=GraphNodeType.DATASET,
                label="Source dataset package",
                source_id=dataset_source,
                source_file=dataset_source,
                attributes={"source_dataset": dataset_source},
            ),
        )
        _add_node(
            nodes,
            GraphNode(
                node_id=workflow_node_id,
                node_type=GraphNodeType.WORKFLOW,
                label="BSIP workflow run",
                source_id=workflow_id,
                source_file=str(self.workflow_dir / "workflow_manifest.json"),
                attributes={
                    "overall_status": workflow_manifest.get("overall_status"),
                    "software_version": workflow_manifest.get("software_version"),
                    "completed_stages": workflow_manifest.get("completed_stages", []),
                    "failed_stages": workflow_manifest.get("failed_stages", []),
                },
            ),
        )
        _add_edge(edges, dataset_id, workflow_node_id, GraphEdgeType.BELONGS_TO)

        validation_nodes = {
            "observation": self._validation_node("VAL:observation", "Observation validation summary", observation_validation, "observation_validation.json"),
            "interpretation": self._validation_node(
                "VAL:interpretation",
                "Interpretation validation summary",
                interpretation_validation,
                "interpretation_validation.json",
            ),
            "hypothesis": self._validation_node("VAL:hypothesis", "Hypothesis validation summary", hypothesis_validation, "hypothesis_validation.json"),
            "workflow": self._validation_node("VAL:workflow", "Workflow validation summary", workflow_manifest, "workflow_manifest.json"),
        }
        for node in validation_nodes.values():
            _add_node(nodes, node)
            _add_edge(edges, node.node_id, workflow_node_id, GraphEdgeType.BELONGS_TO)
        _add_edge(edges, workflow_node_id, "VAL:workflow", GraphEdgeType.VALIDATED_BY)

        for observation in observations:
            observation_id = observation["observation_id"]
            _add_node(
                nodes,
                GraphNode(
                    node_id=observation_id,
                    node_type=GraphNodeType.OBSERVATION,
                    label=observation.get("title") or observation_id,
                    source_id=observation_id,
                    source_file=str(self.observations_dir / "observations.json"),
                    attributes={
                        "category": observation.get("category"),
                        "status": observation.get("status"),
                        "confidence": observation.get("confidence"),
                        "metric_names": [metric.get("metric_name") for metric in observation.get("supporting_metrics", [])],
                    },
                ),
            )
            _add_edge(edges, observation_id, dataset_id, GraphEdgeType.DERIVED_FROM)
            _add_edge(edges, observation_id, "VAL:observation", GraphEdgeType.VALIDATED_BY)
            _add_edge(edges, observation_id, workflow_node_id, GraphEdgeType.GENERATED_BY)

        for interpretation in interpretations:
            interpretation_id = interpretation["interpretation_id"]
            _add_node(
                nodes,
                GraphNode(
                    node_id=interpretation_id,
                    node_type=GraphNodeType.INTERPRETATION,
                    label=interpretation.get("title") or interpretation_id,
                    source_id=interpretation_id,
                    source_file=str(self.interpretations_dir / "interpretations.json"),
                    attributes={
                        "category": interpretation.get("category"),
                        "status": interpretation.get("status"),
                        "confidence": interpretation.get("confidence"),
                        "reasoning_rule_ids": interpretation.get("reasoning_rule_ids", []),
                    },
                ),
            )
            _add_edge(edges, interpretation_id, "VAL:interpretation", GraphEdgeType.VALIDATED_BY)
            _add_edge(edges, interpretation_id, workflow_node_id, GraphEdgeType.GENERATED_BY)
            for observation_id in interpretation.get("supporting_observation_ids", []):
                _add_edge(edges, observation_id, interpretation_id, GraphEdgeType.SUPPORTS)
            for observation_id in interpretation.get("contradicting_observation_ids", []):
                _add_edge(edges, interpretation_id, observation_id, GraphEdgeType.LIMITED_BY)

        # Preserve dependency-table traceability even if future JSON differs.
        for row in interpretation_dependencies:
            observation_id = row.get("observation_id")
            interpretation_id = row.get("interpretation_id")
            if observation_id and interpretation_id and row.get("dependency_type") == "supporting":
                _add_edge(edges, observation_id, interpretation_id, GraphEdgeType.SUPPORTS, attributes={"source": "interpretation_dependencies.csv"})

        for hypothesis in hypotheses:
            hypothesis_id = hypothesis["hypothesis_id"]
            _add_node(
                nodes,
                GraphNode(
                    node_id=hypothesis_id,
                    node_type=GraphNodeType.HYPOTHESIS,
                    label=hypothesis.get("title") or hypothesis_id,
                    source_id=hypothesis_id,
                    source_file=str(self.hypotheses_dir / "hypotheses.json"),
                    attributes={
                        "category": hypothesis.get("category"),
                        "status": hypothesis.get("status"),
                        "confidence": hypothesis.get("confidence"),
                        "priority": hypothesis.get("priority"),
                        "priority_score": hypothesis.get("priority_score"),
                        "reasoning_rule_ids": hypothesis.get("reasoning_rule_ids", []),
                    },
                ),
            )
            _add_edge(edges, hypothesis_id, "VAL:hypothesis", GraphEdgeType.VALIDATED_BY)
            _add_edge(edges, hypothesis_id, workflow_node_id, GraphEdgeType.GENERATED_BY)
            for interpretation_id in hypothesis.get("supporting_interpretation_ids", []):
                _add_edge(edges, interpretation_id, hypothesis_id, GraphEdgeType.SUPPORTS)
            for interpretation_id in hypothesis.get("contradicting_interpretation_ids", []):
                _add_edge(edges, hypothesis_id, interpretation_id, GraphEdgeType.LIMITED_BY)
            for index, gap in enumerate(hypothesis.get("evidence_gaps", []), start=1):
                gap_id = f"GAP-{hypothesis_id}-{index:04d}"
                _add_node(
                    nodes,
                    GraphNode(
                        node_id=gap_id,
                        node_type=GraphNodeType.EVIDENCE_GAP,
                        label=str(gap),
                        source_id=hypothesis_id,
                        source_file=str(self.hypotheses_dir / "hypotheses.json"),
                        attributes={"hypothesis_id": hypothesis_id, "gap_index": index, "text": str(gap)},
                    ),
                )
                _add_edge(edges, hypothesis_id, gap_id, GraphEdgeType.LIMITED_BY)
                _add_edge(edges, gap_id, workflow_node_id, GraphEdgeType.BELONGS_TO)
            for alternative_id in hypothesis.get("alternative_hypothesis_ids", []):
                _add_edge(edges, hypothesis_id, alternative_id, GraphEdgeType.COMPETES_WITH)

        for row in hypothesis_dependencies:
            hypothesis_id = row.get("hypothesis_id")
            interpretation_id = row.get("interpretation_id")
            if hypothesis_id and interpretation_id and row.get("dependency_type") == "supporting":
                _add_edge(edges, interpretation_id, hypothesis_id, GraphEdgeType.SUPPORTS, attributes={"source": "hypothesis_dependencies.csv"})
        for row in competition_map:
            hypothesis_id = row.get("hypothesis_id")
            alternative_id = row.get("alternative_hypothesis_id")
            if hypothesis_id and alternative_id:
                _add_edge(edges, hypothesis_id, alternative_id, GraphEdgeType.COMPETES_WITH, attributes={"source": "hypothesis_competition_map.csv"})

        return ReasoningGraph(
            graph_id=f"BSIP-GRAPH-{workflow_id}",
            schema_version=GRAPH_SCHEMA_VERSION,
            software_version=self.software_version,
            generated_at=generated_at,
            nodes=tuple(nodes.values()),
            edges=tuple(edges.values()),
            metadata={
                "project_root": str(self.project_root),
                "observations_dir": str(self.observations_dir),
                "interpretations_dir": str(self.interpretations_dir),
                "hypotheses_dir": str(self.hypotheses_dir),
                "workflow_dir": str(self.workflow_dir),
                "observation_count": len(observations),
                "interpretation_count": len(interpretations),
                "hypothesis_count": len(hypotheses),
            },
        )

    def _validation_node(self, node_id: str, label: str, payload: dict[str, Any], filename: str) -> GraphNode:
        return GraphNode(
            node_id=node_id,
            node_type=GraphNodeType.VALIDATION_SUMMARY,
            label=label,
            source_id=node_id,
            source_file=str(self._validation_source_file(filename)),
            attributes={
                "validation_passed": payload.get("validation_passed", payload.get("overall_status") == "COMPLETED"),
                "critical_issue_count": payload.get("critical_issue_count", 0),
                "warning_count": payload.get("warning_count", 0),
                "overall_status": payload.get("overall_status"),
            },
        )

    def _validation_source_file(self, filename: str) -> Path:
        if filename.startswith("observation"):
            return self.observations_dir / filename
        if filename.startswith("interpretation"):
            return self.interpretations_dir / filename
        if filename.startswith("hypothesis"):
            return self.hypotheses_dir / filename
        return self.workflow_dir / filename

    def _resolve(self, path: Path | str) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate.resolve()
        return (self.project_root / candidate).resolve()


def build_reasoning_graph(**kwargs: Any) -> ReasoningGraph:
    return ReasoningGraphBuilder(**kwargs).build()


def _add_node(nodes: dict[str, GraphNode], node: GraphNode) -> None:
    nodes.setdefault(node.node_id, node)


def _add_edge(
    edges: dict[str, GraphEdge],
    source_id: str,
    target_id: str,
    edge_type: GraphEdgeType,
    *,
    attributes: dict[str, Any] | None = None,
) -> None:
    edge_id = f"EDGE:{edge_type.value}:{source_id}->{target_id}"
    edges.setdefault(
        edge_id,
        GraphEdge(
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            label=edge_type.value,
            attributes=attributes or {},
        ),
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return payload


def _read_csv(path: Path) -> tuple[dict[str, str], ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return tuple(dict(row) for row in csv.DictReader(handle))


def _stable_token(value: str) -> str:
    return (
        value.replace("\\", "/")
        .rstrip("/")
        .split("/")[-1]
        .replace(" ", "_")
        or "dataset"
    )
