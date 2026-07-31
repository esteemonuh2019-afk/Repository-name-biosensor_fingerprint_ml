import json
from pathlib import Path

from src.scientific_reasoning.workflow import WorkflowEngine
from tests.integration.observation_fixture import create_supervisor_fixture


def test_successful_complete_workflow_generates_manifest_and_report(tmp_path: Path) -> None:
    project_root, supervisor = create_supervisor_fixture(tmp_path)
    result = WorkflowEngine(
        project_root=project_root,
        output_root=project_root / "outputs",
        supervisor_results_dir=supervisor,
        overwrite=True,
    ).run()
    assert result.overall_status.value == "COMPLETED"
    assert [record.stage_name.value for record in result.stage_records] == [
        "observation",
        "interpretation",
        "hypothesis",
    ]
    assert all(record.validation_passed for record in result.stage_records)
    assert result.manifest_path and result.manifest_path.exists()
    assert result.report_path and result.report_path.exists()

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["overall_status"] == "COMPLETED"
    assert manifest["completed_stages"] == ["observation", "interpretation", "hypothesis"]
    assert manifest["failed_stages"] == []
    assert set(manifest["stage_durations"]) == {"observation", "interpretation", "hypothesis"}
    assert manifest["source_dataset"] == str(supervisor.resolve())
    assert all(manifest["generated_files"][stage] for stage in manifest["completed_stages"])


def test_workflow_manifest_stage_records_are_deterministically_ordered(tmp_path: Path) -> None:
    project_root, supervisor = create_supervisor_fixture(tmp_path)
    result = WorkflowEngine(
        project_root=project_root,
        output_root=project_root / "outputs",
        supervisor_results_dir=supervisor,
        overwrite=True,
    ).run()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert [record["stage_name"] for record in manifest["stage_records"]] == [
        "observation",
        "interpretation",
        "hypothesis",
    ]
