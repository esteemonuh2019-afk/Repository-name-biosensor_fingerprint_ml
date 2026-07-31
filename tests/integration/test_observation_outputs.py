import csv
import json
from pathlib import Path

import pytest

from src.scientific_reasoning.observation.engine import ScientificObservationEngine
from tests.integration.observation_fixture import FIXED_TIMESTAMP, create_supervisor_fixture


def run_engine(project_root: Path, supervisor: Path, output_dir: str = "outputs/scientific_observations", overwrite: bool = True):
    return ScientificObservationEngine(
        project_root=project_root,
        supervisor_results_dir=supervisor.relative_to(project_root),
        output_dir=output_dir,
        overwrite=overwrite,
        generated_at=FIXED_TIMESTAMP,
    ).run()


def test_json_csv_and_markdown_readability(tmp_path: Path) -> None:
    project_root, supervisor = create_supervisor_fixture(tmp_path)
    result = run_engine(project_root, supervisor)
    output_dir = result.write_result.output_dir
    json.loads((output_dir / "observations.json").read_text(encoding="utf-8"))
    json.loads((output_dir / "observation_validation.json").read_text(encoding="utf-8"))
    json.loads((output_dir / "observation_summary.json").read_text(encoding="utf-8"))
    with (output_dir / "observations.csv").open("r", encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 11
    assert "## CLASSIFICATION" in (output_dir / "observations.md").read_text(encoding="utf-8")


def test_observation_summary_count_consistency(tmp_path: Path) -> None:
    project_root, supervisor = create_supervisor_fixture(tmp_path)
    result = run_engine(project_root, supervisor)
    summary = json.loads((result.write_result.output_dir / "observation_summary.json").read_text(encoding="utf-8"))
    assert summary["total_observations"] == sum(summary["count_by_category"].values())
    assert summary["total_observations"] == sum(summary["count_by_status"].values())
    assert summary["provenance_backed_observation_count"] == summary["quantitative_observation_count"]


def test_validation_summary_count_consistency(tmp_path: Path) -> None:
    project_root, supervisor = create_supervisor_fixture(tmp_path)
    result = run_engine(project_root, supervisor)
    validation = json.loads((result.write_result.output_dir / "observation_validation.json").read_text(encoding="utf-8"))
    issues = validation["structured_validation_issues"]
    assert validation["critical_issue_count"] == sum(1 for issue in issues if issue["severity"] == "CRITICAL")
    assert validation["warning_count"] == sum(1 for issue in issues if issue["severity"] == "WARNING")
    assert validation["missing_provenance_count"] == 0


def test_non_overwrite_behavior(tmp_path: Path) -> None:
    project_root, supervisor = create_supervisor_fixture(tmp_path)
    output_dir = project_root / "outputs" / "scientific_observations"
    output_dir.mkdir(parents=True)
    (output_dir / "existing.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_engine(project_root, supervisor, overwrite=False)


def test_overwrite_behavior_replaces_only_output_directory(tmp_path: Path) -> None:
    project_root, supervisor = create_supervisor_fixture(tmp_path)
    output_dir = project_root / "outputs" / "scientific_observations"
    output_dir.mkdir(parents=True)
    (output_dir / "existing.txt").write_text("remove", encoding="utf-8")
    result = run_engine(project_root, supervisor, overwrite=True)
    assert result.validation_passed is True
    assert not (output_dir / "existing.txt").exists()
    assert (supervisor / "supervisor_results_summary.json").exists()
