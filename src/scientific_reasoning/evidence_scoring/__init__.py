"""BSIP v4.0.0 Evidence Scoring Engine."""

from .enums import (
    EvidenceDimension,
    EvidenceLevel,
    EvidenceScoringIssueSeverity,
    PublicationReadiness,
    ReviewerConfidence,
    UncertaintyLevel,
)
from .models import (
    EVIDENCE_SCORING_RULE_VERSION,
    EVIDENCE_SCORING_SCHEMA_VERSION,
    EVIDENCE_SCORING_SOFTWARE_VERSION,
    DimensionScore,
    EvidenceScoreRecord,
    EvidenceScoringRunResult,
    EvidenceScoringValidationIssue,
    UncertaintyAssessment,
)
from .rules import DIMENSION_WEIGHTS, EVIDENCE_LEVEL_THRESHOLDS, validate_weights
from .scorer import score_claim, score_claims
from .service import EvidenceScoringEngine, EvidenceScoringService
from .validation import validate_evidence_score_records, validate_source_documents

__all__ = [
    "DIMENSION_WEIGHTS",
    "EVIDENCE_LEVEL_THRESHOLDS",
    "EVIDENCE_SCORING_RULE_VERSION",
    "EVIDENCE_SCORING_SCHEMA_VERSION",
    "EVIDENCE_SCORING_SOFTWARE_VERSION",
    "DimensionScore",
    "EvidenceDimension",
    "EvidenceLevel",
    "EvidenceScoreRecord",
    "EvidenceScoringEngine",
    "EvidenceScoringIssueSeverity",
    "EvidenceScoringRunResult",
    "EvidenceScoringService",
    "EvidenceScoringValidationIssue",
    "PublicationReadiness",
    "ReviewerConfidence",
    "UncertaintyAssessment",
    "UncertaintyLevel",
    "score_claim",
    "score_claims",
    "validate_evidence_score_records",
    "validate_source_documents",
    "validate_weights",
]
