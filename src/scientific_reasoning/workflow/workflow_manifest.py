"""Workflow manifest construction and writing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .workflow_models import (
    WorkflowOverallStatus,
    WorkflowRunResult,
    WorkflowStageStatus,
    json_ready,
)


def build_workflow_manifest(
    result: WorkflowRunResult,
    *,
    source_dataset: str,
) -> dict[str, Any]:
    completed = [
        record.stage_name.value
        for record in result.stage_records
        if record.status in (WorkflowStageStatus.COMPLETED, WorkflowStageStatus.SKIPPED)
    ]
    failed = [
        record.stage_name.value
        for record in result.stage_records
        if record.status == WorkflowStageStatus.FAILED
    ]
    return {
        "workflow_id": result.workflow_id,
        "timestamp": result.started_at,
        "completed_at": result.completed_at,
        "software_version": result.software_version,
        "completed_stages": completed,
        "failed_stages": failed,
        "stage_durations": {
            record.stage_name.value: record.duration_seconds for record in result.stage_records
        },
        "output_directories": {
            record.stage_name.value: record.output_directory for record in result.stage_records
        },
        "validation_summaries": {
            record.stage_name.value: json_ready(dict(record.validation_summary)) for record in result.stage_records
        },
        "source_dataset": source_dataset,
        "generated_files": {
            record.stage_name.value: list(record.generated_files) for record in result.stage_records
        },
        "overall_status": result.overall_status.value,
        "stage_records": [record.to_dict() for record in result.stage_records],
        "metadata": json_ready(dict(result.metadata)),
    }


def write_workflow_manifest(
    workflow_dir: Path,
    result: WorkflowRunResult,
    *,
    source_dataset: str,
) -> Path:
    workflow_dir.mkdir(parents=True, exist_ok=True)
    path = workflow_dir / "workflow_manifest.json"
    payload = build_workflow_manifest(result, source_dataset=source_dataset)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def overall_status_from_stage_records(stage_records) -> WorkflowOverallStatus:
    if any(record.status == WorkflowStageStatus.FAILED for record in stage_records):
        return WorkflowOverallStatus.FAILED
    if len(stage_records) == 3 and all(
        record.status in (WorkflowStageStatus.COMPLETED, WorkflowStageStatus.SKIPPED)
        for record in stage_records
    ):
        return WorkflowOverallStatus.COMPLETED
    return WorkflowOverallStatus.PARTIAL
