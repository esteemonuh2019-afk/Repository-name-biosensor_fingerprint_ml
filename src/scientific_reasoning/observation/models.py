"""Immutable public models for factual scientific observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Mapping

from .enums import ConfidenceLevel, ObservationCategory, ObservationStatus


MetricValue = int | float | str | bool | None
JsonRecord = dict[str, Any]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class SupportingMetric:
    metric_name: str
    metric_value: MetricValue = None
    units: str | None = None
    model_name: str | None = None
    fold_count: int | None = None
    sample_count: int | None = None
    source_file: str | None = None
    source_run: str | None = None
    provenance_id: str | None = None

    def to_dict(self) -> JsonRecord:
        return json_ready(asdict(self))

    def to_record(self) -> JsonRecord:
        return self.to_dict()


@dataclass(frozen=True)
class ProvenanceRecord:
    provenance_id: str
    source_file: str | None = None
    source_run: str | None = None
    section: str | None = None
    claim_text: str | None = None
    metric_name: str | None = None
    metric_value: MetricValue = None
    units: str | None = None
    model_name: str | None = None
    table_or_figure_reference: str | None = None
    support_status: str = "SUPPORTED"

    def to_dict(self) -> JsonRecord:
        return json_ready(asdict(self))

    def to_record(self) -> JsonRecord:
        return self.to_dict()


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    severity: str
    message: str
    observation_id: str | None = None
    field: str | None = None
    source_file: str | None = None

    def to_dict(self) -> JsonRecord:
        return json_ready(asdict(self))

    def to_record(self) -> JsonRecord:
        return self.to_dict()


@dataclass(frozen=True)
class Observation:
    observation_id: str
    category: ObservationCategory
    title: str
    statement: str
    status: ObservationStatus
    analysis_stage: str
    supporting_metrics: tuple[SupportingMetric, ...] = field(default_factory=tuple)
    supporting_files: tuple[str, ...] = field(default_factory=tuple)
    provenance_records: tuple[ProvenanceRecord, ...] = field(default_factory=tuple)
    confidence: ConfidenceLevel = ConfidenceLevel.NOT_ASSESSABLE
    limitations: tuple[str, ...] = field(default_factory=tuple)
    created_at: str = field(default_factory=utc_now_iso)
    software_version: str = "BSIP-2.0"
    source_run: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "category", ObservationCategory(self.category))
        object.__setattr__(self, "status", ObservationStatus(self.status))
        object.__setattr__(self, "confidence", ConfidenceLevel(self.confidence))
        object.__setattr__(self, "supporting_metrics", tuple(self.supporting_metrics))
        object.__setattr__(self, "supporting_files", tuple(self.supporting_files))
        object.__setattr__(self, "provenance_records", tuple(self.provenance_records))
        object.__setattr__(self, "limitations", tuple(self.limitations))
        object.__setattr__(self, "tags", tuple(self.tags))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> JsonRecord:
        return {
            "observation_id": self.observation_id,
            "category": self.category.value,
            "title": self.title,
            "statement": self.statement,
            "status": self.status.value,
            "analysis_stage": self.analysis_stage,
            "supporting_metrics": [metric.to_record() for metric in self.supporting_metrics],
            "supporting_files": list(self.supporting_files),
            "provenance_records": [record.to_record() for record in self.provenance_records],
            "confidence": self.confidence.value,
            "limitations": list(self.limitations),
            "created_at": self.created_at,
            "software_version": self.software_version,
            "source_run": self.source_run,
            "tags": list(self.tags),
            "metadata": json_ready(dict(self.metadata)),
        }

    def to_record(self) -> JsonRecord:
        return self.to_dict()


def json_ready(value: Any) -> Any:
    """Return a canonical JSON-serializable representation."""

    if isinstance(value, ObservationCategory | ObservationStatus | ConfidenceLevel):
        return value.value
    if is_dataclass(value):
        return json_ready(asdict(value))
    if isinstance(value, MappingProxyType):
        return json_ready(dict(value))
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    return value
