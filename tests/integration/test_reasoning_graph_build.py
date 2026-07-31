import json
import shutil
from pathlib import Path

from src.scientific_reasoning.reasoning_graph import (
    ReasoningGraphBuilder,
    graph_validation_summary,
    write_reasoning_graph_outputs,
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
        source = source_root / "outputs" / directory
        assert source.exists(), f"Required validated artifact directory is missing: {source}"
        shutil.copytree(source, project_root / "outputs" / directory)
    return project_root


def test_reasoning_graph_builds_from_existing_validated_outputs(tmp_path: Path) -> None:
    project_root = copy_reasoning_outputs(tmp_path)
    graph = ReasoningGraphBuilder(project_root=project_root).build()
    validation = graph_validation_summary(graph)

    assert validation["validation_passed"] is True
    assert validation["node_count_by_type"]["Observation"] > 0
    assert validation["node_count_by_type"]["Interpretation"] > 0
    assert validation["node_count_by_type"]["Hypothesis"] > 0
    assert validation["edge_count_by_type"]["supports"] > 0

    paths = write_reasoning_graph_outputs(
        graph,
        output_dir=project_root / "outputs" / "reasoning_graph",
        overwrite=True,
    )
    assert set(paths) == {"graph_json", "graphml", "summary_json", "validation_json", "statistics_csv"}
    assert all(path.exists() for path in paths.values())

    exported_graph = json.loads(paths["graph_json"].read_text(encoding="utf-8"))
    assert exported_graph["graph_id"] == graph.graph_id
    assert exported_graph["nodes"]
    assert exported_graph["edges"]
    assert "<graphml" in paths["graphml"].read_text(encoding="utf-8")


def test_reasoning_graph_build_is_deterministic_across_runs(tmp_path: Path) -> None:
    project_root = copy_reasoning_outputs(tmp_path)
    first = ReasoningGraphBuilder(project_root=project_root).build().to_dict()
    second = ReasoningGraphBuilder(project_root=project_root).build().to_dict()

    assert first == second
