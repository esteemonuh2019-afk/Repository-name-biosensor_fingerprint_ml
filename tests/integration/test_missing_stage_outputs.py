from pathlib import Path

from src.scientific_reasoning.workflow import WorkflowEngine
from tests.integration.observation_fixture import create_supervisor_fixture


def test_resume_detects_missing_stage_output_and_stops(tmp_path: Path) -> None:
    project_root, supervisor = create_supervisor_fixture(tmp_path)
    outputs = project_root / "outputs"
    first = WorkflowEngine(
        project_root=project_root,
        output_root=outputs,
        supervisor_results_dir=supervisor,
        overwrite=True,
    ).run()
    assert first.overall_status.value == "COMPLETED"
    (outputs / "scientific_interpretations" / "interpretations.json").unlink()

    second = WorkflowEngine(
        project_root=project_root,
        output_root=outputs,
        supervisor_results_dir=supervisor,
        resume=True,
        overwrite=False,
    ).run()
    assert second.overall_status.value == "FAILED"
    assert [record.stage_name.value for record in second.stage_records] == ["observation", "interpretation"]
    assert second.stage_records[0].status.value == "SKIPPED"
    assert second.stage_records[1].status.value == "FAILED"
    validation = second.stage_records[1].metadata["workflow_stage_validation"]
    assert "interpretations.json" in validation["missing_files"]
