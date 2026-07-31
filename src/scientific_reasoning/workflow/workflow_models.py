"""Immutable models for BSIP v3.0.0 workflow orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


WORKFLOW_SOFTWARE_VERSION = "BSIP-3.0.0-workflow-engine"


class WorkflowStageName(str, Enum):
    OBSERVATION = "observation"
    INTERPRETATION = "interpretation"
    HYPOTHESIS = "hypothesis"


class WorkflowStageStatus(str, Enum):
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"


class WorkflowOverallStatus(str, Enum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"


class WorkflowIssueSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class WorkflowValidationIssue:
    code: str
    severity: WorkflowIssueSeverity
    message: str
    stage_name: WorkflowStageName | str | None = None
    file: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "severity", WorkflowIssueSeverity(self.severity))
        if self.stage_name is not None:
            object.__setattr__(self, "stage_name", WorkflowStageName(self.stage_name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "stage_name": None if self.stage_name is None else self.stage_name.value,
            "file": self.file,
        }


@dataclass(frozen=True)
class StageOutputValidation:
    stage_name: WorkflowStageName
    output_dir: Path
    validation_passed: bool
    missing_files: tuple[str, ...] = field(default_factory=tuple)
    generated_files: tuple[Path, ...] = field(default_factory=tuple)
    validation_summary: Mapping[str, Any] = field(default_factory=dict)
    issues: tuple[WorkflowValidationIssue, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage_name", WorkflowStageName(self.stage_name))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "missing_files", tuple(self.missing_files))
        object.__setattr__(self, "generated_files", tuple(Path(path) for path in self.generated_files))
        object.__setattr__(self, "validation_summary", MappingProxyType(dict(self.validation_summary)))
        object.__setattr__(self, "issues", tuple(self.issues))

    @property
    def critical_issue_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == WorkflowIssueSeverity.CRITICAL)

    @property
    def warning_count(self) -> int:
        return int(self.validation_summary.get("warning_count") or 0) + sum(
            1 for issue in self.issues if issue.severity == WorkflowIssueSeverity.WARNING
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_name": self.stage_name.value,
            "output_dir": str(self.output_dir),
            "validation_passed": self.validation_passed,
            "missing_files": list(self.missing_files),
            "generated_files": [str(path) for path in self.generated_files],
            "validation_summary": json_ready(dict(self.validation_summary)),
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class WorkflowStageRecord:
    stage_name: WorkflowStageName
    status: WorkflowStageStatus
    started_at: str
    completed_at: str
    duration_seconds: float
    software_version: str
    input_directory: str | None
    output_directory: str
    generated_files: tuple[str, ...] = field(default_factory=tuple)
    validation_passed: bool = False
    critical_issue_count: int = 0
    warning_count: int = 0
    validation_summary: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage_name", WorkflowStageName(self.stage_name))
        object.__setattr__(self, "status", WorkflowStageStatus(self.status))
        object.__setattr__(self, "generated_files", tuple(self.generated_files))
        object.__setattr__(self, "validation_summary", MappingProxyType(dict(self.validation_summary)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_name": self.stage_name.value,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
            "software_version": self.software_version,
            "input_directory": self.input_directory,
            "output_directory": self.output_directory,
            "generated_files": list(self.generated_files),
            "validation_passed": self.validation_passed,
            "critical_issue_count": self.critical_issue_count,
            "warning_count": self.warning_count,
            "validation_summary": json_ready(dict(self.validation_summary)),
            "error": self.error,
            "metadata": json_ready(dict(self.metadata)),
        }


@dataclass(frozen=True)
class WorkflowRunResult:
    workflow_id: str
    started_at: str
    completed_at: str
    software_version: str
    overall_status: WorkflowOverallStatus
    stage_records: tuple[WorkflowStageRecord, ...] = field(default_factory=tuple)
    manifest_path: Path | None = None
    report_path: Path | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "overall_status", WorkflowOverallStatus(self.overall_status))
        object.__setattr__(self, "stage_records", tuple(self.stage_records))
        object.__setattr__(self, "manifest_path", None if self.manifest_path is None else Path(self.manifest_path))
        object.__setattr__(self, "report_path", None if self.report_path is None else Path(self.report_path))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "software_version": self.software_version,
            "overall_status": self.overall_status.value,
            "stage_records": [record.to_dict() for record in self.stage_records],
            "manifest_path": None if self.manifest_path is None else str(self.manifest_path),
            "report_path": None if self.report_path is None else str(self.report_path),
            "metadata": json_ready(dict(self.metadata)),
        }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def workflow_id_from_timestamp(timestamp: str) -> str:
    token = timestamp.replace(":", "").replace("-", "").replace("+", "Z")
    return f"BSIP-WF-{token}"


def json_ready(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return json_ready(asdict(value))
    if isinstance(value, MappingProxyType):
        return json_ready(dict(value))
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    return value
