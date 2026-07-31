from dataclasses import FrozenInstanceError

import pytest

from src.scientific_reasoning.workflow import (
    WorkflowOverallStatus,
    WorkflowRunResult,
    WorkflowStageName,
    WorkflowStageRecord,
    WorkflowStageStatus,
)
from src.scientific_reasoning.workflow.workflow_models import workflow_id_from_timestamp


def stage_record(**overrides) -> WorkflowStageRecord:
    payload = {
        "stage_name": WorkflowStageName.OBSERVATION,
        "status": WorkflowStageStatus.COMPLETED,
        "started_at": "2026-07-31T00:00:00+00:00",
        "completed_at": "2026-07-31T00:00:01+00:00",
        "duration_seconds": 1.0,
        "software_version": "BSIP-test",
        "input_directory": "inputs",
        "output_directory": "outputs/scientific_observations",
        "generated_files": ("outputs/scientific_observations/observations.json",),
        "validation_passed": True,
        "critical_issue_count": 0,
        "warning_count": 0,
        "validation_summary": {"validation_passed": True},
    }
    payload.update(overrides)
    return WorkflowStageRecord(**payload)


def test_workflow_stage_record_is_immutable_and_serializes_enums() -> None:
    record = stage_record()
    assert record.stage_name is WorkflowStageName.OBSERVATION
    assert record.status is WorkflowStageStatus.COMPLETED
    assert record.to_dict()["stage_name"] == "observation"
    with pytest.raises(FrozenInstanceError):
        record.status = WorkflowStageStatus.FAILED


def test_workflow_run_result_serializes_stage_records() -> None:
    result = WorkflowRunResult(
        workflow_id="BSIP-WF-test",
        started_at="2026-07-31T00:00:00+00:00",
        completed_at="2026-07-31T00:00:01+00:00",
        software_version="BSIP-3.0.0-test",
        overall_status=WorkflowOverallStatus.COMPLETED,
        stage_records=(stage_record(),),
    )
    record = result.to_dict()
    assert record["overall_status"] == "COMPLETED"
    assert record["stage_records"][0]["validation_passed"] is True


def test_workflow_id_from_timestamp_is_stable() -> None:
    assert workflow_id_from_timestamp("2026-07-31T00:00:00+00:00").startswith("BSIP-WF-20260731T000000Z0000")
