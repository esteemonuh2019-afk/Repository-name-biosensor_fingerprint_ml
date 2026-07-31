"""Stable enums for the BSIP 2.2.0 Scientific Hypothesis contract."""

from __future__ import annotations

from enum import Enum


class HypothesisCategory(str, Enum):
    TEMPORAL_INFORMATION = "TEMPORAL_INFORMATION"
    CHEMICAL_DISCRIMINATION = "CHEMICAL_DISCRIMINATION"
    CONCENTRATION_ENCODING = "CONCENTRATION_ENCODING"
    FEATURE_REPRESENTATION = "FEATURE_REPRESENTATION"
    STRAIN_CONTRIBUTION = "STRAIN_CONTRIBUTION"
    DATA_QUALITY_EFFECT = "DATA_QUALITY_EFFECT"
    GENERALIZATION = "GENERALIZATION"
    OVERALL_SYSTEM_BEHAVIOR = "OVERALL_SYSTEM_BEHAVIOR"


class HypothesisStatus(str, Enum):
    PLAUSIBLE = "PLAUSIBLE"
    COMPETING = "COMPETING"
    WEAKLY_SUPPORTED = "WEAKLY_SUPPORTED"
    CONFLICTED = "CONFLICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NOT_ASSESSABLE = "NOT_ASSESSABLE"


class HypothesisConfidence(str, Enum):
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    NOT_ASSESSABLE = "NOT_ASSESSABLE"


class HypothesisPriority(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NOT_ASSESSABLE = "NOT_ASSESSABLE"


class HypothesisSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


HYPOTHESIS_ID_TOKENS: dict[HypothesisCategory, str] = {
    category: category.value for category in HypothesisCategory
}


def category_id_token(category: HypothesisCategory | str) -> str:
    return HYPOTHESIS_ID_TOKENS[to_hypothesis_category(category)]


def to_hypothesis_category(value: HypothesisCategory | str) -> HypothesisCategory:
    if isinstance(value, HypothesisCategory):
        return value
    return HypothesisCategory(str(value))


def to_hypothesis_status(value: HypothesisStatus | str) -> HypothesisStatus:
    if isinstance(value, HypothesisStatus):
        return value
    return HypothesisStatus(str(value))


def to_hypothesis_confidence(value: HypothesisConfidence | str) -> HypothesisConfidence:
    if isinstance(value, HypothesisConfidence):
        return value
    return HypothesisConfidence(str(value))


def to_hypothesis_priority(value: HypothesisPriority | str) -> HypothesisPriority:
    if isinstance(value, HypothesisPriority):
        return value
    return HypothesisPriority(str(value))


def to_hypothesis_severity(value: HypothesisSeverity | str) -> HypothesisSeverity:
    if isinstance(value, HypothesisSeverity):
        return value
    return HypothesisSeverity(str(value))
