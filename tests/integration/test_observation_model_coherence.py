from pathlib import Path

from src.scientific_reasoning.observation.engine import ScientificObservationEngine
from tests.integration.observation_fixture import FIXED_TIMESTAMP, create_supervisor_fixture


def run_engine(project_root: Path, supervisor: Path):
    engine = ScientificObservationEngine(
        project_root=project_root,
        supervisor_results_dir=supervisor.relative_to(project_root),
        output_dir="outputs/scientific_observations",
        overwrite=True,
        generated_at=FIXED_TIMESTAMP,
    )
    return engine.run(), engine


def test_duplicate_id_prevention(tmp_path: Path) -> None:
    project_root, supervisor = create_supervisor_fixture(tmp_path)
    result, engine = run_engine(project_root, supervisor)
    duplicated = result.observations + (result.observations[0],)
    issues = engine.validate_observations(duplicated)
    assert any(issue.code == "DUPLICATE_OBSERVATION_ID" for issue in issues)


def test_classification_model_metric_coherence(tmp_path: Path) -> None:
    project_root, supervisor = create_supervisor_fixture(tmp_path, classification_metric_model="Random Forest")
    result, _engine = run_engine(project_root, supervisor)
    assert result.validation_passed is False
    assert result.write_result.validation_summary["model_coherence_issue_count"] >= 1


def test_regression_model_metric_coherence(tmp_path: Path) -> None:
    project_root, supervisor = create_supervisor_fixture(tmp_path, regression_metric_model="XGBoost Regressor")
    result, _engine = run_engine(project_root, supervisor)
    assert result.validation_passed is False
    assert result.write_result.validation_summary["model_coherence_issue_count"] >= 1


def test_blind_prediction_without_true_labels_wording(tmp_path: Path) -> None:
    project_root, supervisor = create_supervisor_fixture(tmp_path)
    result, _engine = run_engine(project_root, supervisor)
    blind = next(observation for observation in result.observations if observation.category.value == "BLIND_PREDICTION")
    assert "Validation performance was not calculated because true labels were absent." in blind.statement
    assert result.write_result.validation_summary["blind_validation_wording_issue_count"] == 0


def test_units_preservation(tmp_path: Path) -> None:
    project_root, supervisor = create_supervisor_fixture(tmp_path)
    result, _engine = run_engine(project_root, supervisor)
    regression = next(observation for observation in result.observations if observation.category.value == "REGRESSION")
    rmse = next(metric for metric in regression.supporting_metrics if metric.metric_name == "rmse_mean")
    assert rmse.units == "ug/mL"
