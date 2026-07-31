from pathlib import Path

from src.scientific_reasoning.interpretation import ScientificInterpretationEngine
from src.scientific_reasoning.observation import ScientificObservationEngine
from src.scientific_reasoning.workflow import WorkflowEngine
from tests.integration.observation_fixture import create_supervisor_fixture


def test_failed_observation_stage_stops_workflow(tmp_path: Path) -> None:
    project_root, supervisor = create_supervisor_fixture(tmp_path, missing_required="provenance_index.csv")
    result = WorkflowEngine(
        project_root=project_root,
        output_root=project_root / "outputs",
        supervisor_results_dir=supervisor,
        overwrite=True,
    ).run()
    assert result.overall_status.value == "FAILED"
    assert len(result.stage_records) == 1
    assert result.stage_records[0].stage_name.value == "observation"
    assert result.stage_records[0].status.value == "FAILED"


def test_failed_interpretation_stage_stops_before_hypothesis(tmp_path: Path) -> None:
    project_root, supervisor = create_supervisor_fixture(tmp_path)
    outputs = project_root / "outputs"
    ScientificObservationEngine(
        project_root=project_root,
        supervisor_results_dir=supervisor,
        output_dir=outputs / "scientific_observations",
        overwrite=True,
    ).run()
    interpretation_dir = outputs / "scientific_interpretations"
    interpretation_dir.mkdir()
    (interpretation_dir / "placeholder.txt").write_text("blocks overwrite", encoding="utf-8")

    result = WorkflowEngine(
        project_root=project_root,
        output_root=outputs,
        supervisor_results_dir=supervisor,
        resume=True,
        overwrite=False,
    ).run()
    assert [record.stage_name.value for record in result.stage_records] == ["observation", "interpretation"]
    assert result.stage_records[0].status.value == "SKIPPED"
    assert result.stage_records[1].status.value == "FAILED"
    assert not (outputs / "scientific_hypotheses").exists()


def test_failed_hypothesis_stage_records_failure(tmp_path: Path) -> None:
    project_root, supervisor = create_supervisor_fixture(tmp_path)
    outputs = project_root / "outputs"
    ScientificObservationEngine(
        project_root=project_root,
        supervisor_results_dir=supervisor,
        output_dir=outputs / "scientific_observations",
        overwrite=True,
    ).run()
    ScientificInterpretationEngine(
        project_root=project_root,
        observations_dir=outputs / "scientific_observations",
        output_dir=outputs / "scientific_interpretations",
        overwrite=True,
    ).run()
    hypothesis_dir = outputs / "scientific_hypotheses"
    hypothesis_dir.mkdir()
    (hypothesis_dir / "placeholder.txt").write_text("blocks overwrite", encoding="utf-8")

    result = WorkflowEngine(
        project_root=project_root,
        output_root=outputs,
        supervisor_results_dir=supervisor,
        resume=True,
        overwrite=False,
    ).run()
    assert [record.stage_name.value for record in result.stage_records] == [
        "observation",
        "interpretation",
        "hypothesis",
    ]
    assert result.stage_records[0].status.value == "SKIPPED"
    assert result.stage_records[1].status.value == "SKIPPED"
    assert result.stage_records[2].status.value == "FAILED"
    assert result.overall_status.value == "FAILED"
