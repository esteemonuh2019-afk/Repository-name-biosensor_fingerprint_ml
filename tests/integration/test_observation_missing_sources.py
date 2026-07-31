import json
from pathlib import Path

from src.scientific_reasoning.observation.engine import ScientificObservationEngine
from tests.integration.observation_fixture import FIXED_TIMESTAMP, create_supervisor_fixture


def run_engine(project_root: Path, supervisor: Path):
    return ScientificObservationEngine(
        project_root=project_root,
        supervisor_results_dir=supervisor.relative_to(project_root),
        output_dir="outputs/scientific_observations",
        overwrite=True,
        generated_at=FIXED_TIMESTAMP,
    ).run()


def test_required_file_absence_creates_critical_issue(tmp_path: Path) -> None:
    project_root, supervisor = create_supervisor_fixture(tmp_path, missing_required="provenance_index.csv")
    result = run_engine(project_root, supervisor)
    assert result.validation_passed is False
    validation = result.write_result.validation_summary
    assert validation["critical_issue_count"] == 1
    assert any(issue.code == "REQUIRED_SOURCE_MISSING" for issue in result.validation_issues)


def test_optional_file_absence_creates_warning_not_failure(tmp_path: Path) -> None:
    project_root, supervisor = create_supervisor_fixture(tmp_path, missing_optional="selected_figures.csv")
    result = run_engine(project_root, supervisor)
    assert result.validation_passed is True
    assert result.write_result.validation_summary["warning_count"] == 1
    assert any(issue.code == "OPTIONAL_SOURCE_MISSING" for issue in result.validation_issues)


def test_quantitative_provenance_enforcement(tmp_path: Path) -> None:
    project_root, supervisor = create_supervisor_fixture(tmp_path, remove_provenance_metric="accuracy_mean")
    result = run_engine(project_root, supervisor)
    assert result.validation_passed is False
    validation = json.loads((result.write_result.output_dir / "observation_validation.json").read_text(encoding="utf-8"))
    assert validation["missing_provenance_count"] >= 1
    classification = next(observation for observation in result.observations if observation.category.value == "CLASSIFICATION")
    assert classification.status.value == "INCOMPLETE"
