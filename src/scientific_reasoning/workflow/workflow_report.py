"""Markdown report for BSIP workflow execution."""

from __future__ import annotations

from pathlib import Path

from .workflow_models import WorkflowOverallStatus, WorkflowRunResult


def render_workflow_report(result: WorkflowRunResult) -> str:
    readiness = (
        "Reasoning workflow outputs are available for downstream review."
        if result.overall_status == WorkflowOverallStatus.COMPLETED
        else "Reasoning workflow outputs are incomplete because at least one stage did not pass validation."
    )
    lines = [
        "# BSIP Workflow Report",
        "",
        f"Workflow ID: {result.workflow_id}",
        f"Overall status: {result.overall_status.value}",
        f"Software version: {result.software_version}",
        "",
        "## Completed Stages",
        "",
    ]
    completed = [
        record
        for record in result.stage_records
        if record.status.value in {"COMPLETED", "SKIPPED"}
    ]
    if completed:
        for record in completed:
            lines.append(f"- {record.stage_name.value}: {record.status.value}")
    else:
        lines.append("- None")
    lines.extend(["", "## Outputs", ""])
    for record in result.stage_records:
        lines.append(f"### {record.stage_name.value}")
        lines.append(f"- Output directory: {record.output_directory}")
        lines.append(f"- Generated files: {len(record.generated_files)}")
        lines.append(f"- Duration seconds: {record.duration_seconds}")
        lines.append("")
    lines.extend(["## Validation", ""])
    for record in result.stage_records:
        lines.append(
            f"- {record.stage_name.value}: validation_passed={record.validation_passed}; "
            f"critical={record.critical_issue_count}; warnings={record.warning_count}"
        )
        if record.error:
            lines.append(f"  Error: {record.error}")
    lines.extend(["", "## Warnings", ""])
    warnings = [
        f"{record.stage_name.value}: {record.warning_count}"
        for record in result.stage_records
        if record.warning_count
    ]
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- None reported by workflow validation.")
    lines.extend(["", "## Overall Scientific Readiness", "", readiness, ""])
    return "\n".join(lines)


def write_workflow_report(workflow_dir: Path, result: WorkflowRunResult) -> Path:
    workflow_dir.mkdir(parents=True, exist_ok=True)
    path = workflow_dir / "workflow_report.md"
    path.write_text(render_workflow_report(result), encoding="utf-8")
    return path
