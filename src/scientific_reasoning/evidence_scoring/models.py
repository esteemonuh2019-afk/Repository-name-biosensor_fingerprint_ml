"""Immutable public models for BSIP evidence scoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .enums import (
    EvidenceDimension,
    EvidenceLevel,
    EvidenceScoringIssueSeverity,
    PublicationReadiness,
    ReviewerConfidence,
    UncertaintyLevel,
)


EVIDENCE_SCORING_SCHEMA_VERSION = "BSIP-4.0.0"
EVIDENCE_SCORING_SOFTWARE_VERSION = "BSIP-4.0.0-evidence-scoring-engine"
EVIDENCE_SCORING_RULE_VERSION = "BSIP-EVIDENCE-RULES-4.0.0"
JsonRecord = dict[str, Any]


@dataclass(frozen=True)
class EvidenceScoringValidationIssue:
    code: str
    severity: EvidenceScoringIssueSeverity
    message: str
    claim_id: str | None = None
    field: str | None = None
    graph_node_id: str | None = None
    rule_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "severity", EvidenceScoringIssueSeverity(self.severity))

    def to_dict(self) -> JsonRecord:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "claim_id": self.claim_id,
            "field": self.field,
            "graph_node_id": self.graph_node_id,
            "rule_id": self.rule_id,
        }

    def to_record(self) -> JsonRecord:
        return self.to_dict()


@dataclass(frozen=True)
class DimensionScore:
    dimension: EvidenceDimension
    raw_score: float
    weight: float
    weighted_contribution: float
    positive_factors: tuple[str, ...] = field(default_factory=tuple)
    penalties: tuple[str, ...] = field(default_factory=tuple)
    rule_ids: tuple[str, ...] = field(default_factory=tuple)
    source_node_ids: tuple[str, ...] = field(default_factory=tuple)
    ceilings: tuple[str, ...] = field(default_factory=tuple)
    explanation: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "dimension", EvidenceDimension(self.dimension))
        object.__setattr__(self, "raw_score", float(self.raw_score))
        object.__setattr__(self, "weight", float(self.weight))
        object.__setattr__(self, "weighted_contribution", float(self.weighted_contribution))
        object.__setattr__(self, "positive_factors", tuple(self.positive_factors))
        object.__setattr__(self, "penalties", tuple(self.penalties))
        object.__setattr__(self, "rule_ids", tuple(sorted(self.rule_ids)))
        object.__setattr__(self, "source_node_ids", tuple(sorted(self.source_node_ids)))
        object.__setattr__(self, "ceilings", tuple(self.ceilings))

    def to_dict(self) -> JsonRecord:
        return {
            "dimension": self.dimension.value,
            "raw_score": self.raw_score,
            "weight": self.weight,
            "weighted_contribution": self.weighted_contribution,
            "positive_factors": list(self.positive_factors),
            "penalties": list(self.penalties),
            "rule_ids": list(self.rule_ids),
            "source_node_ids": list(self.source_node_ids),
            "ceilings": list(self.ceilings),
            "explanation": self.explanation,
        }


@dataclass(frozen=True)
class UncertaintyAssessment:
    uncertainty_level: UncertaintyLevel
    uncertainty_sources: tuple[str, ...] = field(default_factory=tuple)
    uncertainty_penalties: tuple[str, ...] = field(default_factory=tuple)
    uncertainty_explanation: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "uncertainty_level", UncertaintyLevel(self.uncertainty_level))
        object.__setattr__(self, "uncertainty_sources", tuple(self.uncertainty_sources))
        object.__setattr__(self, "uncertainty_penalties", tuple(self.uncertainty_penalties))

    def to_dict(self) -> JsonRecord:
        return {
            "uncertainty_level": self.uncertainty_level.value,
            "uncertainty_sources": list(self.uncertainty_sources),
            "uncertainty_penalties": list(self.uncertainty_penalties),
            "uncertainty_explanation": self.uncertainty_explanation,
        }


@dataclass(frozen=True)
class EvidenceScoreRecord:
    claim_id: str
    claim_category: str
    claim_type: str
    claim_status: str
    claim_publication_use: str
    dimension_scores: Mapping[EvidenceDimension, DimensionScore]
    weighted_score: float
    normalized_score: float
    evidence_level: EvidenceLevel
    uncertainty_level: UncertaintyLevel
    reviewer_confidence: ReviewerConfidence
    publication_readiness: PublicationReadiness
    positive_factors: tuple[str, ...] = field(default_factory=tuple)
    negative_factors: tuple[str, ...] = field(default_factory=tuple)
    evidence_gaps: tuple[str, ...] = field(default_factory=tuple)
    limitations: tuple[str, ...] = field(default_factory=tuple)
    competing_hypothesis_ids: tuple[str, ...] = field(default_factory=tuple)
    supporting_observation_ids: tuple[str, ...] = field(default_factory=tuple)
    supporting_interpretation_ids: tuple[str, ...] = field(default_factory=tuple)
    supporting_hypothesis_ids: tuple[str, ...] = field(default_factory=tuple)
    reasoning_graph_node_ids: tuple[str, ...] = field(default_factory=tuple)
    score_explanation: str = ""
    withholding_reasons: tuple[str, ...] = field(default_factory=tuple)
    is_withheld: bool = False
    uncertainty_sources: tuple[str, ...] = field(default_factory=tuple)
    uncertainty_penalties: tuple[str, ...] = field(default_factory=tuple)
    uncertainty_explanation: str = ""
    reviewer_confidence_explanation: str = ""
    publication_readiness_explanation: str = ""
    source_claim_schema_version: str | None = None
    source_graph_schema_version: str | None = None
    evidence_scoring_rule_version: str = EVIDENCE_SCORING_RULE_VERSION
    software_version: str = EVIDENCE_SCORING_SOFTWARE_VERSION

    def __post_init__(self) -> None:
        normalized_dimensions = {
            EvidenceDimension(dimension): score
            for dimension, score in sorted(
                self.dimension_scores.items(),
                key=lambda item: EvidenceDimension(item[0]).value,
            )
        }
        object.__setattr__(self, "dimension_scores", MappingProxyType(normalized_dimensions))
        object.__setattr__(self, "weighted_score", float(self.weighted_score))
        object.__setattr__(self, "normalized_score", float(self.normalized_score))
        object.__setattr__(self, "evidence_level", EvidenceLevel(self.evidence_level))
        object.__setattr__(self, "uncertainty_level", UncertaintyLevel(self.uncertainty_level))
        object.__setattr__(self, "reviewer_confidence", ReviewerConfidence(self.reviewer_confidence))
        object.__setattr__(self, "publication_readiness", PublicationReadiness(self.publication_readiness))
        for field_name in (
            "positive_factors",
            "negative_factors",
            "evidence_gaps",
            "limitations",
            "competing_hypothesis_ids",
            "supporting_observation_ids",
            "supporting_interpretation_ids",
            "supporting_hypothesis_ids",
            "reasoning_graph_node_ids",
            "withholding_reasons",
            "uncertainty_sources",
            "uncertainty_penalties",
        ):
            object.__setattr__(self, field_name, tuple(sorted(getattr(self, field_name))))

    def to_dict(self) -> JsonRecord:
        return {
            "claim_id": self.claim_id,
            "claim_category": self.claim_category,
            "claim_type": self.claim_type,
            "claim_status": self.claim_status,
            "claim_publication_use": self.claim_publication_use,
            "dimension_scores": {
                dimension.value: score.to_dict()
                for dimension, score in sorted(self.dimension_scores.items(), key=lambda item: item[0].value)
            },
            "weighted_score": self.weighted_score,
            "normalized_score": self.normalized_score,
            "evidence_level": self.evidence_level.value,
            "uncertainty_level": self.uncertainty_level.value,
            "reviewer_confidence": self.reviewer_confidence.value,
            "publication_readiness": self.publication_readiness.value,
            "positive_factors": list(self.positive_factors),
            "negative_factors": list(self.negative_factors),
            "evidence_gaps": list(self.evidence_gaps),
            "limitations": list(self.limitations),
            "competing_hypothesis_ids": list(self.competing_hypothesis_ids),
            "supporting_observation_ids": list(self.supporting_observation_ids),
            "supporting_interpretation_ids": list(self.supporting_interpretation_ids),
            "supporting_hypothesis_ids": list(self.supporting_hypothesis_ids),
            "reasoning_graph_node_ids": list(self.reasoning_graph_node_ids),
            "score_explanation": self.score_explanation,
            "withholding_reasons": list(self.withholding_reasons),
            "is_withheld": self.is_withheld,
            "uncertainty_sources": list(self.uncertainty_sources),
            "uncertainty_penalties": list(self.uncertainty_penalties),
            "uncertainty_explanation": self.uncertainty_explanation,
            "reviewer_confidence_explanation": self.reviewer_confidence_explanation,
            "publication_readiness_explanation": self.publication_readiness_explanation,
            "source_claim_schema_version": self.source_claim_schema_version,
            "source_graph_schema_version": self.source_graph_schema_version,
            "evidence_scoring_rule_version": self.evidence_scoring_rule_version,
            "software_version": self.software_version,
        }

    def to_record(self) -> JsonRecord:
        return self.to_dict()


@dataclass(frozen=True)
class EvidenceScoringRunResult:
    records: tuple[EvidenceScoreRecord, ...] = field(default_factory=tuple)
    validation_issues: tuple[EvidenceScoringValidationIssue, ...] = field(default_factory=tuple)
    output_paths: tuple[Path, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)


def json_ready(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return json_ready(value.to_dict())
    if is_dataclass(value):
        return json_ready(asdict(value))
    if isinstance(value, MappingProxyType):
        return json_ready(dict(value))
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    return value
