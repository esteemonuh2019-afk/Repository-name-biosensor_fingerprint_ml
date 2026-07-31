"""Stable enums for the BSIP 2.1.0 Scientific Interpretation contract."""

from __future__ import annotations

from enum import Enum


class InterpretationCategory(str, Enum):
    DATASET_SCOPE = "DATASET_SCOPE"
    DATA_QUALITY = "DATA_QUALITY"
    FINGERPRINT_STRUCTURE = "FINGERPRINT_STRUCTURE"
    EXPLORATORY_STRUCTURE = "EXPLORATORY_STRUCTURE"
    CHEMICAL_CLASSIFICATION = "CHEMICAL_CLASSIFICATION"
    CONCENTRATION_REGRESSION = "CONCENTRATION_REGRESSION"
    FEATURE_ENGINEERING = "FEATURE_ENGINEERING"
    FEATURE_SELECTION = "FEATURE_SELECTION"
    STRAIN_CONTRIBUTION = "STRAIN_CONTRIBUTION"
    BLIND_VALIDATION = "BLIND_VALIDATION"
    OVERALL_EVIDENCE = "OVERALL_EVIDENCE"


class InterpretationStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    CONFLICTED = "CONFLICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NOT_ASSESSABLE = "NOT_ASSESSABLE"


class InterpretationConfidence(str, Enum):
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    NOT_ASSESSABLE = "NOT_ASSESSABLE"


class EvidenceDirection(str, Enum):
    SUPPORTING = "SUPPORTING"
    CONTRADICTING = "CONTRADICTING"
    CONTEXTUAL = "CONTEXTUAL"


class ReasoningSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


INTERPRETATION_ID_TOKENS: dict[InterpretationCategory, str] = {
    category: category.value for category in InterpretationCategory
}


def category_id_token(category: InterpretationCategory | str) -> str:
    """Return the canonical category token used in interpretation IDs."""

    return INTERPRETATION_ID_TOKENS[to_interpretation_category(category)]


def to_interpretation_category(value: InterpretationCategory | str) -> InterpretationCategory:
    if isinstance(value, InterpretationCategory):
        return value
    return InterpretationCategory(str(value))


def to_interpretation_status(value: InterpretationStatus | str) -> InterpretationStatus:
    if isinstance(value, InterpretationStatus):
        return value
    return InterpretationStatus(str(value))


def to_interpretation_confidence(value: InterpretationConfidence | str) -> InterpretationConfidence:
    if isinstance(value, InterpretationConfidence):
        return value
    return InterpretationConfidence(str(value))


def to_evidence_direction(value: EvidenceDirection | str) -> EvidenceDirection:
    if isinstance(value, EvidenceDirection):
        return value
    return EvidenceDirection(str(value))


def to_reasoning_severity(value: ReasoningSeverity | str) -> ReasoningSeverity:
    if isinstance(value, ReasoningSeverity):
        return value
    return ReasoningSeverity(str(value))
