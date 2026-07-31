import json
from pathlib import Path

from src.scientific_reasoning.workflow import WorkflowEngine
from tests.integration.observation_fixture import create_supervisor_fixture


def test_resume_skips_already_validated_successful_stages(tmp_path: Path) -> None:
    project_root, supervisor = create_supervisor_fixture(tmp_path)
    outputs = project_root / "outputs"
    first = WorkflowEngine(
        project_root=project_root,
        output_root=outputs,
        supervisor_results_dir=supervisor,
        overwrite=True,
    ).run()
    assert first.overall_status.value == "COMPLETED"

    second = WorkflowEngine(
        project_root=project_root,
        output_root=outputs,
        supervisor_results_dir=supervisor,
        resume=True,
        overwrite=False,
    ).run()
    assert second.overall_status.value == "COMPLETED"
    assert [record.status.value for record in second.stage_records] == ["SKIPPED", "SKIPPED", "SKIPPED"]
    assert all(record.metadata.get("resume_skipped") is True for record in second.stage_records)
    manifest = json.loads(second.manifest_path.read_text(encoding="utf-8"))
    assert manifest["completed_stages"] == ["observation", "interpretation", "hypothesis"]
