"""Immutable public models for the BSIP v4.2.0 Manuscript Engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .enums import (
    DocumentStatus,
    ManuscriptIssueSeverity,
    PublicationBoundary,
    SectionType,
    SentenceStatus,
    SentenceType,
    TraceabilityStatus,
)


MANUSCRIPT_SCHEMA_VERSION = "BSIP-4.2.0"
MANUSCRIPT_SOFTWARE_VERSION = "BSIP-4.2.0-manuscript-engine"
MANUSCRIPT_RULE_VERSION = "BSIP-MANUSCRIPT-RULES-4.2.0"
JsonRecord = dict[str, Any]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class ManuscriptValidationIssue:
    code: str
    severity: ManuscriptIssueSeverity
    message: str
    sentence_id: str | None = None
    section_id: str | None = None
    claim_id: str | None = None
    reviewer_finding_id: str | None = None
    source_file: str | None = None
    field: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "severity", ManuscriptIssueSeverity(self.severity))

    def to_dict(self) -> JsonRecord:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "sentence_id": self.sentence_id,
            "section_id": self.section_id,
            "claim_id": self.claim_id,
            "reviewer_finding_id": self.reviewer_finding_id,
            "source_file": self.source_file,
            "field": self.field,
        }

    def to_record(self) -> JsonRecord:
        return self.to_dict()


@dataclass(frozen=True)
class ManuscriptSentence:
    sentence_id: str
    text: str
    sentence_type: SentenceType
    section_id: str
    claim_ids: tuple[str, ...] = field(default_factory=tuple)
    observation_ids: tuple[str, ...] = field(default_factory=tuple)
    interpretation_ids: tuple[str, ...] = field(default_factory=tuple)
    hypothesis_ids: tuple[str, ...] = field(default_factory=tuple)
    evidence_score_ids: tuple[str, ...] = field(default_factory=tuple)
    reviewer_finding_ids: tuple[str, ...] = field(default_factory=tuple)
    figure_ids: tuple[str, ...] = field(default_factory=tuple)
    table_ids: tuple[str, ...] = field(default_factory=tuple)
    reasoning_graph_node_ids: tuple[str, ...] = field(default_factory=tuple)
    traceability_status: TraceabilityStatus = TraceabilityStatus.COMPLETE
    language_policy_status: str = "PASSED"
    limitations: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sentence_type", SentenceType(self.sentence_type))
        object.__setattr__(self, "traceability_status", TraceabilityStatus(self.traceability_status))
        for field_name in (
            "claim_ids",
            "observation_ids",
            "interpretation_ids",
            "hypothesis_ids",
            "evidence_score_ids",
            "reviewer_finding_ids",
            "figure_ids",
            "table_ids",
            "reasoning_graph_node_ids",
        ):
            object.__setattr__(self, field_name, tuple(sorted(str(item) for item in getattr(self, field_name))))
        object.__setattr__(self, "limitations", tuple(str(item) for item in self.limitations))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    *self.claim_ids,
                    *self.observation_ids,
                    *self.interpretation_ids,
                    *self.hypothesis_ids,
                    *self.evidence_score_ids,
                    *self.reviewer_finding_ids,
                    *self.figure_ids,
                    *self.table_ids,
                    *self.reasoning_graph_node_ids,
                }
            )
        )

    def to_dict(self) -> JsonRecord:
        return {
            "sentence_id": self.sentence_id,
            "text": self.text,
            "sentence_type": self.sentence_type.value,
            "section_id": self.section_id,
            "source_ids": list(self.source_ids),
            "claim_ids": list(self.claim_ids),
            "observation_ids": list(self.observation_ids),
            "interpretation_ids": list(self.interpretation_ids),
            "hypothesis_ids": list(self.hypothesis_ids),
            "evidence_score_ids": list(self.evidence_score_ids),
            "reviewer_finding_ids": list(self.reviewer_finding_ids),
            "figure_ids": list(self.figure_ids),
            "table_ids": list(self.table_ids),
            "reasoning_graph_node_ids": list(self.reasoning_graph_node_ids),
            "traceability_status": self.traceability_status.value,
            "language_policy_status": self.language_policy_status,
            "limitations": list(self.limitations),
            "metadata": json_ready(dict(self.metadata)),
        }

    def to_record(self) -> JsonRecord:
        return self.to_dict()


@dataclass(frozen=True)
class ManuscriptParagraph:
    paragraph_id: str
    text: str
    sentence_ids: tuple[str, ...]
    source_ids: tuple[str, ...] = field(default_factory=tuple)
    claim_ids: tuple[str, ...] = field(default_factory=tuple)
    reviewer_finding_ids: tuple[str, ...] = field(default_factory=tuple)
    confidence: str = "GUARDED"
    publication_use: str = "INTERNAL"
    status: SentenceStatus = SentenceStatus.ALLOWED

    def __post_init__(self) -> None:
        object.__setattr__(self, "sentence_ids", tuple(sorted(str(item) for item in self.sentence_ids)))
        object.__setattr__(self, "source_ids", tuple(sorted(str(item) for item in self.source_ids)))
        object.__setattr__(self, "claim_ids", tuple(sorted(str(item) for item in self.claim_ids)))
        object.__setattr__(self, "reviewer_finding_ids", tuple(sorted(str(item) for item in self.reviewer_finding_ids)))
        object.__setattr__(self, "status", SentenceStatus(self.status))

    def to_dict(self) -> JsonRecord:
        return {
            "paragraph_id": self.paragraph_id,
            "text": self.text,
            "sentence_ids": list(self.sentence_ids),
            "source_ids": list(self.source_ids),
            "claim_ids": list(self.claim_ids),
            "reviewer_finding_ids": list(self.reviewer_finding_ids),
            "confidence": self.confidence,
            "publication_use": self.publication_use,
            "status": self.status.value,
        }


@dataclass(frozen=True)
class ManuscriptSection:
    section_id: str
    section_type: SectionType
    title: str
    paragraphs: tuple[ManuscriptParagraph, ...] = field(default_factory=tuple)
    sentences: tuple[ManuscriptSentence, ...] = field(default_factory=tuple)
    source_claim_ids: tuple[str, ...] = field(default_factory=tuple)
    source_observation_ids: tuple[str, ...] = field(default_factory=tuple)
    source_interpretation_ids: tuple[str, ...] = field(default_factory=tuple)
    source_hypothesis_ids: tuple[str, ...] = field(default_factory=tuple)
    source_figure_ids: tuple[str, ...] = field(default_factory=tuple)
    source_table_ids: tuple[str, ...] = field(default_factory=tuple)
    reviewer_finding_ids: tuple[str, ...] = field(default_factory=tuple)
    publication_boundary: PublicationBoundary = PublicationBoundary.INTERNAL_ONLY
    limitations: tuple[str, ...] = field(default_factory=tuple)
    status: SentenceStatus = SentenceStatus.ALLOWED

    def __post_init__(self) -> None:
        object.__setattr__(self, "section_type", SectionType(self.section_type))
        object.__setattr__(self, "publication_boundary", PublicationBoundary(self.publication_boundary))
        object.__setattr__(self, "status", SentenceStatus(self.status))
        for field_name in (
            "source_claim_ids",
            "source_observation_ids",
            "source_interpretation_ids",
            "source_hypothesis_ids",
            "source_figure_ids",
            "source_table_ids",
            "reviewer_finding_ids",
        ):
            object.__setattr__(self, field_name, tuple(sorted(str(item) for item in getattr(self, field_name))))
        object.__setattr__(self, "paragraphs", tuple(self.paragraphs))
        object.__setattr__(self, "sentences", tuple(self.sentences))
        object.__setattr__(self, "limitations", tuple(str(item) for item in self.limitations))

    def to_dict(self) -> JsonRecord:
        return {
            "section_id": self.section_id,
            "section_type": self.section_type.value,
            "title": self.title,
            "paragraphs": [paragraph.to_dict() for paragraph in self.paragraphs],
            "sentences": [sentence.to_dict() for sentence in self.sentences],
            "source_claim_ids": list(self.source_claim_ids),
            "source_observation_ids": list(self.source_observation_ids),
            "source_interpretation_ids": list(self.source_interpretation_ids),
            "source_hypothesis_ids": list(self.source_hypothesis_ids),
            "source_figure_ids": list(self.source_figure_ids),
            "source_table_ids": list(self.source_table_ids),
            "reviewer_finding_ids": list(self.reviewer_finding_ids),
            "publication_boundary": self.publication_boundary.value,
            "limitations": list(self.limitations),
            "status": self.status.value,
        }


@dataclass(frozen=True)
class Caption:
    caption_id: str
    caption_type: SentenceType
    title: str
    text: str
    source_id: str
    source_file: str | None = None
    source_run: str | None = None
    sentence_id: str | None = None
    claim_ids: tuple[str, ...] = field(default_factory=tuple)
    observation_ids: tuple[str, ...] = field(default_factory=tuple)
    limitations: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "caption_type", SentenceType(self.caption_type))
        object.__setattr__(self, "claim_ids", tuple(sorted(str(item) for item in self.claim_ids)))
        object.__setattr__(self, "observation_ids", tuple(sorted(str(item) for item in self.observation_ids)))
        object.__setattr__(self, "limitations", tuple(str(item) for item in self.limitations))

    def to_dict(self) -> JsonRecord:
        return {
            "caption_id": self.caption_id,
            "caption_type": self.caption_type.value,
            "title": self.title,
            "text": self.text,
            "source_id": self.source_id,
            "source_file": self.source_file,
            "source_run": self.source_run,
            "sentence_id": self.sentence_id,
            "claim_ids": list(self.claim_ids),
            "observation_ids": list(self.observation_ids),
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class RevisionFlag:
    flag_id: str
    reviewer_finding_id: str
    severity: str
    blocking: bool
    affected_section: str
    affected_sentence_ids: tuple[str, ...]
    applied_action: str
    resolution_status: str
    author_action_required: bool
    notes: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "affected_sentence_ids", tuple(sorted(str(item) for item in self.affected_sentence_ids)))
        object.__setattr__(self, "blocking", bool(self.blocking))
        object.__setattr__(self, "author_action_required", bool(self.author_action_required))

    def to_dict(self) -> JsonRecord:
        return {
            "flag_id": self.flag_id,
            "reviewer_finding_id": self.reviewer_finding_id,
            "severity": self.severity,
            "blocking": self.blocking,
            "affected_section": self.affected_section,
            "affected_sentence_ids": list(self.affected_sentence_ids),
            "applied_action": self.applied_action,
            "resolution_status": self.resolution_status,
            "author_action_required": self.author_action_required,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ManuscriptDocument:
    manuscript_id: str
    title: str
    document_status: DocumentStatus
    sections: tuple[ManuscriptSection, ...]
    figure_captions: tuple[Caption, ...] = field(default_factory=tuple)
    table_captions: tuple[Caption, ...] = field(default_factory=tuple)
    unresolved_flags: tuple[RevisionFlag, ...] = field(default_factory=tuple)
    source_schema_versions: Mapping[str, Any] = field(default_factory=dict)
    software_version: str = MANUSCRIPT_SOFTWARE_VERSION
    created_at: str = field(default_factory=utc_now_iso)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_status", DocumentStatus(self.document_status))
        object.__setattr__(self, "sections", tuple(self.sections))
        object.__setattr__(self, "figure_captions", tuple(self.figure_captions))
        object.__setattr__(self, "table_captions", tuple(self.table_captions))
        object.__setattr__(self, "unresolved_flags", tuple(self.unresolved_flags))
        object.__setattr__(self, "source_schema_versions", MappingProxyType(dict(self.source_schema_versions)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def sentences(self) -> tuple[ManuscriptSentence, ...]:
        return tuple(sentence for section in self.sections for sentence in section.sentences)

    @property
    def paragraphs(self) -> tuple[ManuscriptParagraph, ...]:
        return tuple(paragraph for section in self.sections for paragraph in section.paragraphs)

    def to_dict(self) -> JsonRecord:
        return {
            "manuscript_id": self.manuscript_id,
            "title": self.title,
            "document_status": self.document_status.value,
            "sections": [section.to_dict() for section in self.sections],
            "figure_captions": [caption.to_dict() for caption in self.figure_captions],
            "table_captions": [caption.to_dict() for caption in self.table_captions],
            "unresolved_flags": [flag.to_dict() for flag in self.unresolved_flags],
            "source_schema_versions": json_ready(dict(self.source_schema_versions)),
            "software_version": self.software_version,
            "created_at": self.created_at,
            "metadata": json_ready(dict(self.metadata)),
        }

    def to_record(self) -> JsonRecord:
        return self.to_dict()


@dataclass(frozen=True)
class ManuscriptSourcePackage:
    observations_dir: Path
    interpretations_dir: Path
    hypotheses_dir: Path
    claims_dir: Path
    evidence_dir: Path
    review_dir: Path
    graph_dir: Path
    supervisor_dir: Path
    observations_document: Mapping[str, Any] = field(default_factory=dict)
    observation_validation_document: Mapping[str, Any] = field(default_factory=dict)
    observation_summary_document: Mapping[str, Any] = field(default_factory=dict)
    interpretations_document: Mapping[str, Any] = field(default_factory=dict)
    interpretation_validation_document: Mapping[str, Any] = field(default_factory=dict)
    interpretation_summary_document: Mapping[str, Any] = field(default_factory=dict)
    hypotheses_document: Mapping[str, Any] = field(default_factory=dict)
    hypothesis_validation_document: Mapping[str, Any] = field(default_factory=dict)
    hypothesis_summary_document: Mapping[str, Any] = field(default_factory=dict)
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
    review_findings_document: Mapping[str, Any] = field(default_factory=dict)
    reviewer_validation_document: Mapping[str, Any] = field(default_factory=dict)
    reviewer_summary_document: Mapping[str, Any] = field(default_factory=dict)
    reviewer_publication_assessment_document: Mapping[str, Any] = field(default_factory=dict)
    reviewer_claim_rows: tuple[Mapping[str, str], ...] = field(default_factory=tuple)
    reviewer_revision_rows: tuple[Mapping[str, str], ...] = field(default_factory=tuple)
    graph_document: Mapping[str, Any] = field(default_factory=dict)
    graph_validation_document: Mapping[str, Any] = field(default_factory=dict)
    graph_summary_document: Mapping[str, Any] = field(default_factory=dict)
    selected_figures: tuple[Mapping[str, str], ...] = field(default_factory=tuple)
    selected_tables: tuple[Mapping[str, str], ...] = field(default_factory=tuple)
    supervisor_validation_document: Mapping[str, Any] = field(default_factory=dict)
    source_files_loaded: tuple[str, ...] = field(default_factory=tuple)
    source_files_missing: tuple[str, ...] = field(default_factory=tuple)
    validation_issues: tuple[ManuscriptValidationIssue, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for field_name in (
            "claim_publication_rows",
            "reviewer_claim_rows",
            "reviewer_revision_rows",
            "selected_figures",
            "selected_tables",
        ):
            object.__setattr__(self, field_name, tuple(dict(row) for row in getattr(self, field_name)))
        object.__setattr__(self, "source_files_loaded", tuple(sorted(str(path) for path in self.source_files_loaded)))
        object.__setattr__(self, "source_files_missing", tuple(sorted(str(path) for path in self.source_files_missing)))
        object.__setattr__(self, "validation_issues", tuple(self.validation_issues))

    @property
    def observations(self) -> tuple[dict[str, Any], ...]:
        return _ordered_records(self.observations_document, "observations", "observation_id")

    @property
    def interpretations(self) -> tuple[dict[str, Any], ...]:
        return _ordered_records(self.interpretations_document, "interpretations", "interpretation_id")

    @property
    def hypotheses(self) -> tuple[dict[str, Any], ...]:
        return _ordered_records(self.hypotheses_document, "hypotheses", "hypothesis_id")

    @property
    def claims(self) -> tuple[dict[str, Any], ...]:
        return _ordered_records(self.claims_document, "claims", "claim_id")

    @property
    def evidence_scores(self) -> tuple[dict[str, Any], ...]:
        return _ordered_records(self.evidence_scores_document, "evidence_scores", "claim_id")

    @property
    def review_findings(self) -> tuple[dict[str, Any], ...]:
        return _ordered_records(self.review_findings_document, "review_findings", "finding_id")

    @property
    def graph_nodes(self) -> tuple[dict[str, Any], ...]:
        return _ordered_records(self.graph_document, "nodes", "node_id")

    @property
    def claim_by_id(self) -> dict[str, dict[str, Any]]:
        return {str(record.get("claim_id")): record for record in self.claims}

    @property
    def evidence_by_claim_id(self) -> dict[str, dict[str, Any]]:
        return {str(record.get("claim_id")): record for record in self.evidence_scores}

    @property
    def observation_by_id(self) -> dict[str, dict[str, Any]]:
        return {str(record.get("observation_id")): record for record in self.observations}

    @property
    def interpretation_by_id(self) -> dict[str, dict[str, Any]]:
        return {str(record.get("interpretation_id")): record for record in self.interpretations}

    @property
    def hypothesis_by_id(self) -> dict[str, dict[str, Any]]:
        return {str(record.get("hypothesis_id")): record for record in self.hypotheses}

    @property
    def reviewer_finding_by_id(self) -> dict[str, dict[str, Any]]:
        return {str(record.get("finding_id")): record for record in self.review_findings}

    @property
    def graph_node_by_id(self) -> dict[str, dict[str, Any]]:
        return {str(record.get("node_id")): record for record in self.graph_nodes}

    @property
    def manuscript_drafting_allowed(self) -> bool:
        return self.reviewer_publication_assessment_document.get("manuscript_drafting_allowed") is True


@dataclass(frozen=True)
class ManuscriptRunResult:
    document: ManuscriptDocument | None = None
    validation_issues: tuple[ManuscriptValidationIssue, ...] = field(default_factory=tuple)
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


def _ordered_records(document: Mapping[str, Any], key: str, sort_key: str) -> tuple[dict[str, Any], ...]:
    records = document.get(key, ()) or ()
    return tuple(sorted((dict(record) for record in records), key=lambda record: str(record.get(sort_key))))
