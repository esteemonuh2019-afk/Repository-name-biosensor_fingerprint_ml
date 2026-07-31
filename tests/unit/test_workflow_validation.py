import json
from pathlib import Path

from src.scientific_reasoning.workflow import WorkflowStageName, WorkflowStageStatus
from src.scientific_reasoning.workflow.workflow_models import WorkflowStageRecord
from src.scientific_reasoning.workflow.workflow_validator import (
    STAGE_OUTPUT_FILES,
    validate_stage_outputs,
    validate_workflow_stage_sequence,
)


def write_stage_outputs(directory: Path, stage: WorkflowStageName, *, validation_passed: bool = True) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for filename in STAGE_OUTPUT_FILES[stage]:
        path = directory / filename
        if filename.endswith(".json"):
            payload = {
                "validation_passed": validation_passed,
                "critical_issue_count": 0 if validation_passed else 1,
                "warning_count": 0,
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
        else:
            path.write_text("placeholder", encoding="utf-8")


def test_validate_stage_outputs_accepts_complete_valid_package(tmp_path: Path) -> None:
    output_dir = tmp_path / "scientific_observations"
    write_stage_outputs(output_dir, WorkflowStageName.OBSERVATION)
    result = validate_stage_outputs(WorkflowStageName.OBSERVATION, output_dir)
    assert result.validation_passed is True
    assert len(result.generated_files) == len(STAGE_OUTPUT_FILES[WorkflowStageName.OBSERVATION])


def test_validate_stage_outputs_reports_missing_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "scientific_hypotheses"
    output_dir.mkdir()
    result = validate_stage_outputs(WorkflowStageName.HYPOTHESIS, output_dir)
    assert result.validation_passed is False
    assert "hypotheses.json" in result.missing_files
    assert result.critical_issue_count > 0


def test_validate_stage_outputs_reports_failed_validation_json(tmp_path: Path) -> None:
    output_dir = tmp_path / "scientific_interpretations"
    write_stage_outputs(output_dir, WorkflowStageName.INTERPRETATION, validation_passed=False)
    result = validate_stage_outputs(WorkflowStageName.INTERPRETATION, output_dir)
    assert result.validation_passed is False
    assert {issue.code for issue in result.issues} >= {
        "STAGE_VALIDATION_FAILED",
        "STAGE_CRITICAL_VALIDATION_ISSUES",
    }


def test_validate_workflow_stage_sequence_detects_out_of_order_records() -> None:
    records = (
        record(WorkflowStageName.HYPOTHESIS),
        record(WorkflowStageName.OBSERVATION),
    )
    issues = validate_workflow_stage_sequence(records)
    assert issues[0].code == "WORKFLOW_STAGE_ORDER_INVALID"


def record(stage: WorkflowStageName) -> WorkflowStageRecord:
    return WorkflowStageRecord(
        stage_name=stage,
        status=WorkflowStageStatus.COMPLETED,
        started_at="2026-07-31T00:00:00+00:00",
        completed_at="2026-07-31T00:00:01+00:00",
        duration_seconds=1,
        software_version="test",
        input_directory="input",
        output_directory="output",
        validation_passed=True,
    )
