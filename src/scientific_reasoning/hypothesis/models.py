"""Immutable public models for BSIP scientific hypotheses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .enums import (
    HypothesisCategory,
    HypothesisConfidence,
    HypothesisPriority,
    HypothesisSeverity,
    HypothesisStatus,
)


JsonRecord = dict[str, Any]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class HypothesisValidationIssue:
    code: str
    severity: HypothesisSeverity
    message: str
    hypothesis_id: str | None = None
    field: str | None = None
    interpretation_id: str | None = None
    rule_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "severity", HypothesisSeverity(self.severity))

    def to_dict(self) -> JsonRecord:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "hypothesis_id": self.hypothesis_id,
            "field": self.field,
            "interpretation_id": self.interpretation_id,
            "rule_id": self.rule_id,
        }

    def to_record(self) -> JsonRecord:
        return self.to_dict()


@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: str
    category: HypothesisCategory
    title: str
    statement: str
    status: HypothesisStatus
    confidence: HypothesisConfidence
    supporting_interpretation_ids: tuple[str, ...] = field(default_factory=tuple)
    contradicting_interpretation_ids: tuple[str, ...] = field(default_factory=tuple)
    supporting_observation_ids: tuple[str, ...] = field(default_factory=tuple)
    assumptions: tuple[str, ...] = field(default_factory=tuple)
    alternative_hypothesis_ids: tuple[str, ...] = field(default_factory=tuple)
    evidence_gaps: tuple[str, ...] = field(default_factory=tuple)
    falsifiability_statement: str | None = None
    rationale: str = ""
    reasoning_rule_ids: tuple[str, ...] = field(default_factory=tuple)
    priority_score: float = 0.0
    priority: HypothesisPriority = HypothesisPriority.NOT_ASSESSABLE
    created_at: str = field(default_factory=utc_now_iso)
    software_version: str = "BSIP-2.2.0-hypothesis-engine"
    source_interpretation_schema_version: str | None = "BSIP-2.1.0"
    tags: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "category", HypothesisCategory(self.category))
        object.__setattr__(self, "status", HypothesisStatus(self.status))
        object.__setattr__(self, "confidence", HypothesisConfidence(self.confidence))
        object.__setattr__(self, "priority", HypothesisPriority(self.priority))
        object.__setattr__(
            self,
            "supporting_interpretation_ids",
            tuple(self.supporting_interpretation_ids),
        )
        object.__setattr__(
            self,
            "contradicting_interpretation_ids",
            tuple(self.contradicting_interpretation_ids),
        )
        object.__setattr__(self, "supporting_observation_ids", tuple(self.supporting_observation_ids))
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        object.__setattr__(self, "alternative_hypothesis_ids", tuple(self.alternative_hypothesis_ids))
        object.__setattr__(self, "evidence_gaps", tuple(self.evidence_gaps))
        object.__setattr__(self, "reasoning_rule_ids", tuple(self.reasoning_rule_ids))
        object.__setattr__(self, "priority_score", float(self.priority_score))
        object.__setattr__(self, "tags", tuple(self.tags))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> JsonRecord:
        return {
            "hypothesis_id": self.hypothesis_id,
            "category": self.category.value,
            "title": self.title,
            "statement": self.statement,
            "status": self.status.value,
            "confidence": self.confidence.value,
            "supporting_interpretation_ids": sorted(self.supporting_interpretation_ids),
            "contradicting_interpretation_ids": sorted(self.contradicting_interpretation_ids),
            "supporting_observation_ids": sorted(self.supporting_observation_ids),
            "assumptions": list(self.assumptions),
            "alternative_hypothesis_ids": sorted(self.alternative_hypothesis_ids),
            "evidence_gaps": list(self.evidence_gaps),
            "falsifiability_statement": self.falsifiability_statement,
            "rationale": self.rationale,
            "reasoning_rule_ids": sorted(self.reasoning_rule_ids),
            "priority_score": self.priority_score,
            "priority": self.priority.value,
            "created_at": self.created_at,
            "software_version": self.software_version,
            "source_interpretation_schema_version": self.source_interpretation_schema_version,
            "tags": sorted(self.tags),
            "metadata": json_ready(dict(self.metadata)),
        }

    def to_record(self) -> JsonRecord:
        return self.to_dict()


@dataclass(frozen=True)
class HypothesisRunResult:
    hypotheses: tuple[Hypothesis, ...] = field(default_factory=tuple)
    validation_issues: tuple[HypothesisValidationIssue, ...] = field(default_factory=tuple)
    output_paths: tuple[Path, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)


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
