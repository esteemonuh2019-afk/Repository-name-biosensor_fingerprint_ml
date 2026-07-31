"""Immutable public models for the BSIP v4.1.0 Reviewer Engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .enums import (
    PublicationRisk,
    ReviewCategory,
    ReviewerConfidence,
    ReviewerType,
    ReviewIssueSeverity,
    Severity,
)


REVIEW_SCHEMA_VERSION = "BSIP-4.1.0"
REVIEW_SOFTWARE_VERSION = "BSIP-4.1.0-reviewer-engine"
REVIEW_RULE_VERSION = "BSIP-REVIEW-RULES-4.1.0"
JsonRecord = dict[str, Any]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class ReviewValidationIssue:
    code: str
    severity: ReviewIssueSeverity
    message: str
    finding_id: str | None = None
    claim_id: str | None = None
    graph_node_id: str | None = None
    field: str | None = None
    source_file: str | None = None
    rule_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "severity", ReviewIssueSeverity(self.severity))

    def to_dict(self) -> JsonRecord:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "finding_id": self.finding_id,
            "claim_id": self.claim_id,
            "graph_node_id": self.graph_node_id,
            "field": self.field,
            "source_file": self.source_file,
            "rule_id": self.rule_id,
        }

    def to_record(self) -> JsonRecord:
        return self.to_dict()


@dataclass(frozen=True)
class ReviewFinding:
    finding_id: str
    reviewer_type: ReviewerType
    category: ReviewCategory
    title: str
    finding_text: str
    severity: Severity
    blocking: bool
    confidence: ReviewerConfidence
    affected_claim_ids: tuple[str, ...] = field(default_factory=tuple)
    affected_hypothesis_ids: tuple[str, ...] = field(default_factory=tuple)
    affected_interpretation_ids: tuple[str, ...] = field(default_factory=tuple)
    affected_observation_ids: tuple[str, ...] = field(default_factory=tuple)
    affected_figure_ids: tuple[str, ...] = field(default_factory=tuple)
    affected_table_ids: tuple[str, ...] = field(default_factory=tuple)
    evidence_score_ids: tuple[str, ...] = field(default_factory=tuple)
    reasoning_graph_node_ids: tuple[str, ...] = field(default_factory=tuple)
    source_validation_ids: tuple[str, ...] = field(default_factory=tuple)
    rationale: str = ""
    evidence_summary: str = ""
    publication_risk: PublicationRisk = PublicationRisk.NONE
    revision_requirement: str = ""
    limitations: tuple[str, ...] = field(default_factory=tuple)
    rule_ids: tuple[str, ...] = field(default_factory=tuple)
    created_at: str = field(default_factory=utc_now_iso)
    software_version: str = REVIEW_SOFTWARE_VERSION
    tags: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "reviewer_type", ReviewerType(self.reviewer_type))
        object.__setattr__(self, "category", ReviewCategory(self.category))
        object.__setattr__(self, "severity", Severity(self.severity))
        object.__setattr__(self, "confidence", ReviewerConfidence(self.confidence))
        object.__setattr__(self, "publication_risk", PublicationRisk(self.publication_risk))
        for field_name in (
            "affected_claim_ids",
            "affected_hypothesis_ids",
            "affected_interpretation_ids",
            "affected_observation_ids",
            "affected_figure_ids",
            "affected_table_ids",
            "evidence_score_ids",
            "reasoning_graph_node_ids",
            "source_validation_ids",
            "rule_ids",
            "tags",
        ):
            object.__setattr__(self, field_name, tuple(sorted(str(item) for item in getattr(self, field_name))))
        object.__setattr__(self, "limitations", tuple(str(item) for item in self.limitations))
        object.__setattr__(self, "blocking", bool(self.blocking))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> JsonRecord:
        return {
            "finding_id": self.finding_id,
            "reviewer_type": self.reviewer_type.value,
            "category": self.category.value,
            "title": self.title,
            "finding_text": self.finding_text,
            "severity": self.severity.value,
            "blocking": self.blocking,
            "confidence": self.confidence.value,
            "affected_claim_ids": list(self.affected_claim_ids),
            "affected_hypothesis_ids": list(self.affected_hypothesis_ids),
            "affected_interpretation_ids": list(self.affected_interpretation_ids),
            "affected_observation_ids": list(self.affected_observation_ids),
            "affected_figure_ids": list(self.affected_figure_ids),
            "affected_table_ids": list(self.affected_table_ids),
            "evidence_score_ids": list(self.evidence_score_ids),
            "reasoning_graph_node_ids": list(self.reasoning_graph_node_ids),
            "source_validation_ids": list(self.source_validation_ids),
            "rationale": self.rationale,
            "evidence_summary": self.evidence_summary,
            "publication_risk": self.publication_risk.value,
            "revision_requirement": self.revision_requirement,
            "limitations": list(self.limitations),
            "rule_ids": list(self.rule_ids),
            "created_at": self.created_at,
            "software_version": self.software_version,
            "tags": list(self.tags),
            "metadata": json_ready(dict(self.metadata)),
        }

    def to_record(self) -> JsonRecord:
        return self.to_dict()


@dataclass(frozen=True)
class ReviewContext:
    claims_document: Mapping[str, Any] = field(default_factory=dict)
    claim_validation_document: Mapping[str, Any] = field(default_factory=dict)
    claim_summary_document: Mapping[str, Any] = field(default_factory=dict)
    claim_publication_rows: tuple[Mapping[str, str], ...] = field(default_factory=tuple)
    evidence_scores_document: Mapping[str, Any] = field(default_factory=dict)
    evidence_validation_document: Mapping[str, Any] = field(default_factory=dict)
    evidence_summary_document: Mapping[str, Any] = field(default_factory=dict)
    reviewer_confidence_document: Mapping[str, Any] = field(default_factory=dict)
    uncertainty_document: Mapping[str, Any] = field(default_factory=dict)
    evidence_traceability_document: Mapping[str, Any] = field(default_factory=dict)
    graph_document: Mapping[str, Any] = field(default_factory=dict)
    graph_validation_document: Mapping[str, Any] = field(default_factory=dict)
    graph_summary_document: Mapping[str, Any] = field(default_factory=dict)
    selected_figures: tuple[Mapping[str, str], ...] = field(default_factory=tuple)
    selected_tables: tuple[Mapping[str, str], ...] = field(default_factory=tuple)
    supervisor_validation_document: Mapping[str, Any] = field(default_factory=dict)
    source_files_loaded: tuple[str, ...] = field(default_factory=tuple)
    source_files_missing: tuple[str, ...] = field(default_factory=tuple)
    validation_issues: tuple[ReviewValidationIssue, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_publication_rows", tuple(dict(row) for row in self.claim_publication_rows))
        object.__setattr__(self, "selected_figures", tuple(dict(row) for row in self.selected_figures))
        object.__setattr__(self, "selected_tables", tuple(dict(row) for row in self.selected_tables))
        object.__setattr__(self, "source_files_loaded", tuple(sorted(str(path) for path in self.source_files_loaded)))
        object.__setattr__(self, "source_files_missing", tuple(sorted(str(path) for path in self.source_files_missing)))
        object.__setattr__(self, "validation_issues", tuple(self.validation_issues))

    @property
    def claims(self) -> tuple[dict[str, Any], ...]:
        records = self.claims_document.get("claims", ()) or ()
        return tuple(sorted((dict(record) for record in records), key=lambda record: str(record.get("claim_id"))))

    @property
    def evidence_scores(self) -> tuple[dict[str, Any], ...]:
        records = self.evidence_scores_document.get("evidence_scores", ()) or ()
        return tuple(sorted((dict(record) for record in records), key=lambda record: str(record.get("claim_id"))))

    @property
    def claim_by_id(self) -> dict[str, dict[str, Any]]:
        return {str(claim.get("claim_id")): claim for claim in self.claims}

    @property
    def evidence_by_claim_id(self) -> dict[str, dict[str, Any]]:
        return {str(score.get("claim_id")): score for score in self.evidence_scores}

    @property
    def graph_node_ids(self) -> set[str]:
        return {str(node.get("node_id")) for node in self.graph_document.get("nodes", ()) or () if node.get("node_id") is not None}

    @property
    def source_validation_ids(self) -> tuple[str, ...]:
        ids = [
            "claim_validation.json",
            "evidence_scoring_validation.json",
            "reasoning_graph_validation.json",
        ]
        if self.supervisor_validation_document:
            ids.append("report_validation.json")
        return tuple(ids)

    @property
    def source_validation_passed(self) -> bool:
        return (
            self.claim_validation_document.get("validation_passed") is True
            and int(self.claim_validation_document.get("critical_issue_count") or 0) == 0
            and self.evidence_validation_document.get("validation_passed") is True
            and int(self.evidence_validation_document.get("critical_issue_count") or 0) == 0
            and self.graph_validation_document.get("validation_passed") is True
            and int(self.graph_validation_document.get("critical_issue_count") or 0) == 0
            and not self.validation_issues
        )


@dataclass(frozen=True)
class ReviewRunResult:
    findings: tuple[ReviewFinding, ...] = field(default_factory=tuple)
    validation_issues: tuple[ReviewValidationIssue, ...] = field(default_factory=tuple)
    output_paths: tuple[Path, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)


def json_ready(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, MappingProxyType):
        return json_ready(dict(value))
    if is_dataclass(value):
        return json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list, set)):
        return [json_ready(item) for item in value]
    return value
