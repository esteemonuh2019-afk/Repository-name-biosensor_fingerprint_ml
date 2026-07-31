"""BSIP v4.2.0 Manuscript Engine."""

from .composer import DEFAULT_TITLE, compose_manuscript
from .engine import ManuscriptEngine, ManuscriptService
from .enums import (
    DocumentStatus,
    ManuscriptIssueSeverity,
    PublicationBoundary,
    ResolutionStatus,
    SectionType,
    SentenceStatus,
    SentenceType,
    TraceabilityStatus,
)
from .models import (
    MANUSCRIPT_RULE_VERSION,
    MANUSCRIPT_SCHEMA_VERSION,
    MANUSCRIPT_SOFTWARE_VERSION,
    Caption,
    ManuscriptDocument,
    ManuscriptParagraph,
    ManuscriptRunResult,
    ManuscriptSection,
    ManuscriptSentence,
    ManuscriptSourcePackage,
    ManuscriptValidationIssue,
    RevisionFlag,
)
from .source_loader import load_source_package, validate_source_gate
from .validators import validate_manuscript_document

__all__ = [
    "Caption",
    "DEFAULT_TITLE",
    "DocumentStatus",
    "MANUSCRIPT_RULE_VERSION",
    "MANUSCRIPT_SCHEMA_VERSION",
    "MANUSCRIPT_SOFTWARE_VERSION",
    "ManuscriptDocument",
    "ManuscriptEngine",
    "ManuscriptIssueSeverity",
    "ManuscriptParagraph",
    "ManuscriptRunResult",
    "ManuscriptSection",
    "ManuscriptSentence",
    "ManuscriptService",
    "ManuscriptSourcePackage",
    "ManuscriptValidationIssue",
    "PublicationBoundary",
    "ResolutionStatus",
    "RevisionFlag",
    "SectionType",
    "SentenceStatus",
    "SentenceType",
    "TraceabilityStatus",
    "compose_manuscript",
    "load_source_package",
    "validate_manuscript_document",
    "validate_source_gate",
]
