"""File-based validation for workflow stage output packages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .workflow_models import (
    StageOutputValidation,
    WorkflowIssueSeverity,
    WorkflowStageName,
    WorkflowValidationIssue,
)


STAGE_OUTPUT_FILES: dict[WorkflowStageName, tuple[str, ...]] = {
    WorkflowStageName.OBSERVATION: (
        "observations.json",
        "observations.csv",
        "observations.md",
        "observation_validation.json",
        "observation_provenance.csv",
        "observation_summary.json",
    ),
    WorkflowStageName.INTERPRETATION: (
        "interpretations.json",
        "interpretations.csv",
        "interpretations.md",
        "interpretation_validation.json",
        "interpretation_summary.json",
        "interpretation_dependencies.csv",
    ),
    WorkflowStageName.HYPOTHESIS: (
        "hypotheses.json",
        "hypotheses.csv",
        "hypotheses.md",
        "hypothesis_validation.json",
        "hypothesis_summary.json",
        "hypothesis_dependencies.csv",
        "hypothesis_competition_map.csv",
    ),
}

STAGE_VALIDATION_FILE: dict[WorkflowStageName, str] = {
    WorkflowStageName.OBSERVATION: "observation_validation.json",
    WorkflowStageName.INTERPRETATION: "interpretation_validation.json",
    WorkflowStageName.HYPOTHESIS: "hypothesis_validation.json",
}


def validate_stage_outputs(
    stage_name: WorkflowStageName | str,
    output_dir: Path | str,
) -> StageOutputValidation:
    stage = WorkflowStageName(stage_name)
    directory = Path(output_dir)
    expected_files = STAGE_OUTPUT_FILES[stage]
    issues: list[WorkflowValidationIssue] = []
    missing = tuple(filename for filename in expected_files if not (directory / filename).exists())
    for filename in missing:
        issues.append(
            WorkflowValidationIssue(
                code="STAGE_OUTPUT_FILE_MISSING",
                severity=WorkflowIssueSeverity.CRITICAL,
                message=f"Required {stage.value} output file is missing: {filename}",
                stage_name=stage,
                file=str(directory / filename),
            )
        )

    validation_summary: dict[str, Any] = {}
    validation_file = directory / STAGE_VALIDATION_FILE[stage]
    if validation_file.exists():
        try:
            validation_summary = _read_json(validation_file)
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            issues.append(
                WorkflowValidationIssue(
                    code="STAGE_VALIDATION_UNREADABLE",
                    severity=WorkflowIssueSeverity.CRITICAL,
                    message=f"Unable to read stage validation file: {exc}",
                    stage_name=stage,
                    file=str(validation_file),
                )
            )
    if validation_summary:
        if validation_summary.get("validation_passed") is not True:
            issues.append(
                WorkflowValidationIssue(
                    code="STAGE_VALIDATION_FAILED",
                    severity=WorkflowIssueSeverity.CRITICAL,
                    message=f"{stage.value} output validation did not pass.",
                    stage_name=stage,
                    file=str(validation_file),
                )
            )
        if int(validation_summary.get("critical_issue_count") or 0) > 0:
            issues.append(
                WorkflowValidationIssue(
                    code="STAGE_CRITICAL_VALIDATION_ISSUES",
                    severity=WorkflowIssueSeverity.CRITICAL,
                    message=f"{stage.value} validation reports critical issues.",
                    stage_name=stage,
                    file=str(validation_file),
                )
            )

    generated_files = tuple(directory / filename for filename in expected_files if (directory / filename).exists())
    validation_passed = not any(issue.severity == WorkflowIssueSeverity.CRITICAL for issue in issues)
    return StageOutputValidation(
        stage_name=stage,
        output_dir=directory,
        validation_passed=validation_passed,
        missing_files=missing,
        generated_files=generated_files,
        validation_summary=validation_summary,
        issues=tuple(issues),
    )


def validate_workflow_stage_sequence(stage_records) -> tuple[WorkflowValidationIssue, ...]:
    """Return a deterministic-ordering issue if stage records are out of workflow order."""

    order = {
        WorkflowStageName.OBSERVATION: 0,
        WorkflowStageName.INTERPRETATION: 1,
        WorkflowStageName.HYPOTHESIS: 2,
    }
    actual = [order[record.stage_name] for record in stage_records]
    if actual == sorted(actual):
        return tuple()
    return (
        WorkflowValidationIssue(
            code="WORKFLOW_STAGE_ORDER_INVALID",
            severity=WorkflowIssueSeverity.CRITICAL,
            message="Workflow stage records are not ordered observation -> interpretation -> hypothesis.",
            stage_name=None,
            file=None,
        ),
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return payload
