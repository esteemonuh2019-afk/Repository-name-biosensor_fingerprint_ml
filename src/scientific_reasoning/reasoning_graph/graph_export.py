"""Export helpers for BSIP reasoning graphs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape, quoteattr

from .graph_models import GraphEdgeType, GraphNodeType, ReasoningGraph, json_ready
from .graph_validator import graph_validation_summary


DEFAULT_OUTPUT_DIR = Path("outputs/reasoning_graph")
GRAPH_OUTPUT_FILENAMES = {
    "reasoning_graph.json",
    "reasoning_graph.graphml",
    "reasoning_graph_summary.json",
    "reasoning_graph_validation.json",
    "reasoning_graph_statistics.csv",
}


def write_reasoning_graph_outputs(
    graph: ReasoningGraph,
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write all deterministic graph artifacts and return their paths."""

    resolved_output_dir = Path(output_dir).resolve()
    _prepare_output_dir(resolved_output_dir, overwrite=overwrite)

    validation = graph_validation_summary(graph)
    summary = _graph_summary(graph, validation)

    paths = {
        "graph_json": resolved_output_dir / "reasoning_graph.json",
        "graphml": resolved_output_dir / "reasoning_graph.graphml",
        "summary_json": resolved_output_dir / "reasoning_graph_summary.json",
        "validation_json": resolved_output_dir / "reasoning_graph_validation.json",
        "statistics_csv": resolved_output_dir / "reasoning_graph_statistics.csv",
    }

    _write_json(paths["graph_json"], graph.to_dict())
    _write_text(paths["graphml"], _to_graphml(graph))
    _write_json(paths["summary_json"], summary)
    _write_json(paths["validation_json"], validation)
    _write_statistics_csv(paths["statistics_csv"], graph, validation)
    return paths


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_graph_outputs = [output_dir / filename for filename in sorted(GRAPH_OUTPUT_FILENAMES) if (output_dir / filename).exists()]
    if existing_graph_outputs and not overwrite:
        raise FileExistsError(f"Reasoning graph outputs already exist in {output_dir}. Use overwrite=True to replace them.")
    for path in existing_graph_outputs:
        path.unlink()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, payload: str) -> None:
    path.write_text(payload, encoding="utf-8")


def _graph_summary(graph: ReasoningGraph, validation: dict[str, Any]) -> dict[str, Any]:
    return {
        "graph_id": graph.graph_id,
        "schema_version": graph.schema_version,
        "software_version": graph.software_version,
        "generated_at": graph.generated_at,
        "node_count": validation["node_count"],
        "edge_count": validation["edge_count"],
        "node_count_by_type": validation["node_count_by_type"],
        "edge_count_by_type": validation["edge_count_by_type"],
        "orphan_count": validation["orphan_count"],
        "cycle_count": validation["cycle_count"],
        "validation_passed": validation["validation_passed"],
        "critical_issue_count": validation["critical_issue_count"],
        "warning_count": validation["warning_count"],
    }


def _write_statistics_csv(path: Path, graph: ReasoningGraph, validation: dict[str, Any]) -> None:
    rows: list[dict[str, str]] = [
        {"statistic": "node_count", "category": "all", "value": str(len(graph.nodes))},
        {"statistic": "edge_count", "category": "all", "value": str(len(graph.edges))},
        {"statistic": "orphan_count", "category": "validation", "value": str(validation["orphan_count"])},
        {"statistic": "cycle_count", "category": "validation", "value": str(validation["cycle_count"])},
        {
            "statistic": "critical_issue_count",
            "category": "validation",
            "value": str(validation["critical_issue_count"]),
        },
        {"statistic": "warning_count", "category": "validation", "value": str(validation["warning_count"])},
    ]
    rows.extend(
        {"statistic": "node_count_by_type", "category": node_type, "value": str(count)}
        for node_type, count in sorted(validation["node_count_by_type"].items())
    )
    rows.extend(
        {"statistic": "edge_count_by_type", "category": edge_type, "value": str(count)}
        for edge_type, count in sorted(validation["edge_count_by_type"].items())
    )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("statistic", "category", "value"))
        writer.writeheader()
        writer.writerows(rows)


def _to_graphml(graph: ReasoningGraph) -> str:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
        '  <key id="node_type" for="node" attr.name="node_type" attr.type="string"/>',
        '  <key id="label" for="node" attr.name="label" attr.type="string"/>',
        '  <key id="source_id" for="node" attr.name="source_id" attr.type="string"/>',
        '  <key id="source_file" for="node" attr.name="source_file" attr.type="string"/>',
        '  <key id="edge_type" for="edge" attr.name="edge_type" attr.type="string"/>',
        '  <key id="edge_label" for="edge" attr.name="label" attr.type="string"/>',
        f'  <graph id={quoteattr(graph.graph_id)} edgedefault="directed">',
    ]

    for node in graph.nodes:
        lines.append(f"    <node id={quoteattr(node.node_id)}>")
        lines.append(f"      <data key=\"node_type\">{escape(node.node_type.value)}</data>")
        lines.append(f"      <data key=\"label\">{escape(node.label)}</data>")
        if node.source_id is not None:
            lines.append(f"      <data key=\"source_id\">{escape(str(node.source_id))}</data>")
        if node.source_file is not None:
            lines.append(f"      <data key=\"source_file\">{escape(str(node.source_file))}</data>")
        for key, value in sorted(node.attributes.items(), key=lambda item: str(item[0])):
            key_id = _attribute_key("node_attr", key)
            lines.insert(2, f'  <key id="{key_id}" for="node" attr.name="{escape(str(key))}" attr.type="string"/>')
            lines.append(f"      <data key=\"{key_id}\">{escape(_graphml_value(value))}</data>")
        lines.append("    </node>")

    for edge in graph.edges:
        lines.append(
            f"    <edge id={quoteattr(edge.edge_id)} source={quoteattr(edge.source_id)} target={quoteattr(edge.target_id)}>"
        )
        lines.append(f"      <data key=\"edge_type\">{escape(edge.edge_type.value)}</data>")
        if edge.label is not None:
            lines.append(f"      <data key=\"edge_label\">{escape(edge.label)}</data>")
        for key, value in sorted(edge.attributes.items(), key=lambda item: str(item[0])):
            key_id = _attribute_key("edge_attr", key)
            lines.insert(2, f'  <key id="{key_id}" for="edge" attr.name="{escape(str(key))}" attr.type="string"/>')
            lines.append(f"      <data key=\"{key_id}\">{escape(_graphml_value(value))}</data>")
        lines.append("    </edge>")

    lines.extend(["  </graph>", "</graphml>", ""])
    return "\n".join(_dedupe_graphml_keys(lines))


def _attribute_key(prefix: str, key: object) -> str:
    safe = "".join(character if character.isalnum() else "_" for character in str(key))
    return f"{prefix}_{safe}"


def _graphml_value(value: object) -> str:
    if isinstance(value, (GraphNodeType, GraphEdgeType)):
        return value.value
    if isinstance(value, (list, tuple)):
        return ", ".join(_graphml_value(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(json_ready(value), sort_keys=True)
    if value is None:
        return ""
    return str(value)


def _dedupe_graphml_keys(lines: list[str]) -> list[str]:
    seen_keys = set()
    deduped = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("<key "):
            if stripped in seen_keys:
                continue
            seen_keys.add(stripped)
        deduped.append(line)
    return deduped
