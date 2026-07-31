"""Immutable public models for BSIP scientific interpretations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .enums import (
    EvidenceDirection,
    InterpretationCategory,
    InterpretationConfidence,
    InterpretationStatus,
    ReasoningSeverity,
)


JsonRecord = dict[str, Any]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class InterpretationEvidenceLink:
    observation_id: str
    direction: EvidenceDirection
    rationale: str
    metric_names: tuple[str, ...] = field(default_factory=tuple)
    provenance_ids: tuple[str, ...] = field(default_factory=tuple)
    source_files: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "direction", EvidenceDirection(self.direction))
        object.__setattr__(self, "metric_names", tuple(self.metric_names))
        object.__setattr__(self, "provenance_ids", tuple(self.provenance_ids))
        object.__setattr__(self, "source_files", tuple(self.source_files))

    def to_dict(self) -> JsonRecord:
        return {
            "observation_id": self.observation_id,
            "direction": self.direction.value,
            "rationale": self.rationale,
            "metric_names": sorted(self.metric_names),
            "provenance_ids": sorted(self.provenance_ids),
            "source_files": sorted(self.source_files),
        }

    def to_record(self) -> JsonRecord:
        return self.to_dict()


@dataclass(frozen=True)
class ReasoningRule:
    rule_id: str
    name: str
    description: str
    required_categories: tuple[InterpretationCategory, ...] = field(default_factory=tuple)
    optional_categories: tuple[InterpretationCategory, ...] = field(default_factory=tuple)
    minimum_supporting_observations: int = 1
    allowed_claim_template: str | None = None
    forbidden_terms: tuple[str, ...] = field(default_factory=tuple)
    confidence_policy: str = "rule_based"
    limitation_policy: str = "inherit_observation_limitations"
    enabled: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "required_categories",
            tuple(InterpretationCategory(category) for category in self.required_categories),
        )
        object.__setattr__(
            self,
            "optional_categories",
            tuple(InterpretationCategory(category) for category in self.optional_categories),
        )
        object.__setattr__(self, "forbidden_terms", tuple(self.forbidden_terms))

    def to_dict(self) -> JsonRecord:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "required_categories": [category.value for category in self.required_categories],
            "optional_categories": [category.value for category in self.optional_categories],
            "minimum_supporting_observations": self.minimum_supporting_observations,
            "allowed_claim_template": self.allowed_claim_template,
            "forbidden_terms": sorted(self.forbidden_terms),
            "confidence_policy": self.confidence_policy,
            "limitation_policy": self.limitation_policy,
            "enabled": self.enabled,
        }

    def to_record(self) -> JsonRecord:
        return self.to_dict()


@dataclass(frozen=True)
class InterpretationValidationIssue:
    code: str
    severity: ReasoningSeverity
    message: str
    interpretation_id: str | None = None
    field: str | None = None
    observation_id: str | None = None
    rule_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "severity", ReasoningSeverity(self.severity))

    def to_dict(self) -> JsonRecord:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "interpretation_id": self.interpretation_id,
            "field": self.field,
            "observation_id": self.observation_id,
            "rule_id": self.rule_id,
        }

    def to_record(self) -> JsonRecord:
        return self.to_dict()


@dataclass(frozen=True)
class Interpretation:
    interpretation_id: str
    category: InterpretationCategory
    title: str
    claim: str
    status: InterpretationStatus
    confidence: InterpretationConfidence
    supporting_observation_ids: tuple[str, ...] = field(default_factory=tuple)
    contradicting_observation_ids: tuple[str, ...] = field(default_factory=tuple)
    assumptions: tuple[str, ...] = field(default_factory=tuple)
    limitations: tuple[str, ...] = field(default_factory=tuple)
    evidence_summary: tuple[InterpretationEvidenceLink, ...] = field(default_factory=tuple)
    reasoning_rule_ids: tuple[str, ...] = field(default_factory=tuple)
    created_at: str = field(default_factory=utc_now_iso)
    software_version: str = "BSIP-2.1.0-interpretation-contract"
    source_observation_schema_version: str | None = "BSIP-2.0"
    tags: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "category", InterpretationCategory(self.category))
        object.__setattr__(self, "status", InterpretationStatus(self.status))
        object.__setattr__(self, "confidence", InterpretationConfidence(self.confidence))
        object.__setattr__(self, "supporting_observation_ids", tuple(self.supporting_observation_ids))
        object.__setattr__(self, "contradicting_observation_ids", tuple(self.contradicting_observation_ids))
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        object.__setattr__(self, "limitations", tuple(self.limitations))
        object.__setattr__(self, "evidence_summary", tuple(self.evidence_summary))
        object.__setattr__(self, "reasoning_rule_ids", tuple(self.reasoning_rule_ids))
        object.__setattr__(self, "tags", tuple(self.tags))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> JsonRecord:
        return {
            "interpretation_id": self.interpretation_id,
            "category": self.category.value,
            "title": self.title,
            "claim": self.claim,
            "status": self.status.value,
            "confidence": self.confidence.value,
            "supporting_observation_ids": sorted(self.supporting_observation_ids),
            "contradicting_observation_ids": sorted(self.contradicting_observation_ids),
            "assumptions": list(self.assumptions),
            "limitations": list(self.limitations),
            "evidence_summary": [link.to_record() for link in self.evidence_summary],
            "reasoning_rule_ids": sorted(self.reasoning_rule_ids),
            "created_at": self.created_at,
            "software_version": self.software_version,
            "source_observation_schema_version": self.source_observation_schema_version,
            "tags": sorted(self.tags),
            "metadata": json_ready(dict(self.metadata)),
        }

    def to_record(self) -> JsonRecord:
        return self.to_dict()


def json_ready(value: Any) -> Any:
    """Return a canonical JSON-serializable representation."""

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
