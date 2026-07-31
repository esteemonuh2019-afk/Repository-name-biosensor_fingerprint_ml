"""Immutable public models for BSIP scientific claims."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .enums import (
    ClaimCategory,
    ClaimIssueSeverity,
    ClaimStatus,
    ClaimType,
    ConfidenceLabel,
    EvidenceStrength,
    PublicationUse,
)


CLAIM_SCHEMA_VERSION = "BSIP-3.2.0"
DEFAULT_CLAIM_SOFTWARE_VERSION = "BSIP-3.2.0-claim-engine"
JsonRecord = dict[str, Any]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class ClaimValidationIssue:
    code: str
    severity: ClaimIssueSeverity
    message: str
    claim_id: str | None = None
    field: str | None = None
    hypothesis_id: str | None = None
    graph_node_id: str | None = None
    rule_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "severity", ClaimIssueSeverity(self.severity))

    def to_dict(self) -> JsonRecord:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "claim_id": self.claim_id,
            "field": self.field,
            "hypothesis_id": self.hypothesis_id,
            "graph_node_id": self.graph_node_id,
            "rule_id": self.rule_id,
        }

    def to_record(self) -> JsonRecord:
        return self.to_dict()


@dataclass(frozen=True)
class ScientificClaim:
    claim_id: str
    category: ClaimCategory
    title: str
    claim_text: str
    claim_type: ClaimType
    claim_status: ClaimStatus
    evidence_strength: EvidenceStrength
    publication_use: PublicationUse
    supporting_hypothesis_ids: tuple[str, ...] = field(default_factory=tuple)
    competing_hypothesis_ids: tuple[str, ...] = field(default_factory=tuple)
    supporting_interpretation_ids: tuple[str, ...] = field(default_factory=tuple)
    supporting_observation_ids: tuple[str, ...] = field(default_factory=tuple)
    evidence_gap_ids: tuple[str, ...] = field(default_factory=tuple)
    validation_summary_ids: tuple[str, ...] = field(default_factory=tuple)
    reasoning_graph_node_ids: tuple[str, ...] = field(default_factory=tuple)
    assumptions: tuple[str, ...] = field(default_factory=tuple)
    limitations: tuple[str, ...] = field(default_factory=tuple)
    rationale: str = ""
    evidence_score: float = 0.0
    confidence_label: ConfidenceLabel = ConfidenceLabel.NOT_ASSESSABLE
    language_policy_rule_ids: tuple[str, ...] = field(default_factory=tuple)
    reasoning_rule_ids: tuple[str, ...] = field(default_factory=tuple)
    created_at: str = field(default_factory=utc_now_iso)
    software_version: str = DEFAULT_CLAIM_SOFTWARE_VERSION
    source_hypothesis_schema_version: str | None = None
    source_graph_schema_version: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "category", ClaimCategory(self.category))
        object.__setattr__(self, "claim_type", ClaimType(self.claim_type))
        object.__setattr__(self, "claim_status", ClaimStatus(self.claim_status))
        object.__setattr__(self, "evidence_strength", EvidenceStrength(self.evidence_strength))
        object.__setattr__(self, "publication_use", PublicationUse(self.publication_use))
        object.__setattr__(self, "confidence_label", ConfidenceLabel(self.confidence_label))
        for field_name in (
            "supporting_hypothesis_ids",
            "competing_hypothesis_ids",
            "supporting_interpretation_ids",
            "supporting_observation_ids",
            "evidence_gap_ids",
            "validation_summary_ids",
            "reasoning_graph_node_ids",
            "assumptions",
            "limitations",
            "language_policy_rule_ids",
            "reasoning_rule_ids",
            "tags",
        ):
            values = tuple(getattr(self, field_name))
            if field_name not in {"assumptions", "limitations"}:
                values = tuple(sorted(values))
            object.__setattr__(self, field_name, values)
        object.__setattr__(self, "evidence_score", float(self.evidence_score))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> JsonRecord:
        return {
            "claim_id": self.claim_id,
            "category": self.category.value,
            "title": self.title,
            "claim_text": self.claim_text,
            "claim_type": self.claim_type.value,
            "claim_status": self.claim_status.value,
            "evidence_strength": self.evidence_strength.value,
            "publication_use": self.publication_use.value,
            "supporting_hypothesis_ids": list(self.supporting_hypothesis_ids),
            "competing_hypothesis_ids": list(self.competing_hypothesis_ids),
            "supporting_interpretation_ids": list(self.supporting_interpretation_ids),
            "supporting_observation_ids": list(self.supporting_observation_ids),
            "evidence_gap_ids": list(self.evidence_gap_ids),
            "validation_summary_ids": list(self.validation_summary_ids),
            "reasoning_graph_node_ids": list(self.reasoning_graph_node_ids),
            "assumptions": list(self.assumptions),
            "limitations": list(self.limitations),
            "rationale": self.rationale,
            "evidence_score": self.evidence_score,
            "confidence_label": self.confidence_label.value,
            "language_policy_rule_ids": list(self.language_policy_rule_ids),
            "reasoning_rule_ids": list(self.reasoning_rule_ids),
            "created_at": self.created_at,
            "software_version": self.software_version,
            "source_hypothesis_schema_version": self.source_hypothesis_schema_version,
            "source_graph_schema_version": self.source_graph_schema_version,
            "tags": list(self.tags),
            "metadata": json_ready(dict(self.metadata)),
        }

    def to_record(self) -> JsonRecord:
        return self.to_dict()


@dataclass(frozen=True)
class ClaimRunResult:
    claims: tuple[ScientificClaim, ...] = field(default_factory=tuple)
    validation_issues: tuple[ClaimValidationIssue, ...] = field(default_factory=tuple)
    output_paths: tuple[Path, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)


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
