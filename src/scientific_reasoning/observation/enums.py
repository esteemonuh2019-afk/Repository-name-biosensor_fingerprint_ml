"""Stable enums for the BSIP 2.0 Observation contract."""

from __future__ import annotations

from enum import Enum


class ObservationCategory(str, Enum):
    DATASET = "DATASET"
    QUALITY_CONTROL = "QUALITY_CONTROL"
    FINGERPRINT = "FINGERPRINT"
    EXPLORATORY_ANALYSIS = "EXPLORATORY_ANALYSIS"
    CLASSIFICATION = "CLASSIFICATION"
    REGRESSION = "REGRESSION"
    FEATURE_ENGINEERING = "FEATURE_ENGINEERING"
    FEATURE_SELECTION = "FEATURE_SELECTION"
    STRAIN_CONTRIBUTION = "STRAIN_CONTRIBUTION"
    BLIND_PREDICTION = "BLIND_PREDICTION"
    VALIDATION = "VALIDATION"


class ObservationStatus(str, Enum):
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    ACTIVE = "ACTIVE"
    FAILED = "FAILED"
    NOT_ASSESSABLE = "NOT_ASSESSABLE"


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    NOT_ASSESSABLE = "NOT_ASSESSABLE"


OBSERVATION_ID_TOKENS: dict[ObservationCategory, str] = {
    ObservationCategory.DATASET: "DATASET",
    ObservationCategory.QUALITY_CONTROL: "QC",
    ObservationCategory.FINGERPRINT: "FINGERPRINT",
    ObservationCategory.EXPLORATORY_ANALYSIS: "EXPLORATORY_ANALYSIS",
    ObservationCategory.CLASSIFICATION: "CLASSIFICATION",
    ObservationCategory.REGRESSION: "REGRESSION",
    ObservationCategory.FEATURE_ENGINEERING: "FEATURE_ENGINEERING",
    ObservationCategory.FEATURE_SELECTION: "FEATURE_SELECTION",
    ObservationCategory.STRAIN_CONTRIBUTION: "STRAIN_CONTRIBUTION",
    ObservationCategory.BLIND_PREDICTION: "BLIND_PREDICTION",
    ObservationCategory.VALIDATION: "VALIDATION",
}


def category_id_token(category: ObservationCategory | str) -> str:
    """Return the canonical token used in observation IDs."""

    return OBSERVATION_ID_TOKENS[to_observation_category(category)]


def to_observation_category(value: ObservationCategory | str) -> ObservationCategory:
    if isinstance(value, ObservationCategory):
        return value
    return ObservationCategory(str(value))


def to_observation_status(value: ObservationStatus | str) -> ObservationStatus:
    if isinstance(value, ObservationStatus):
        return value
    return ObservationStatus(str(value))


def to_confidence_level(value: ConfidenceLevel | str) -> ConfidenceLevel:
    if isinstance(value, ConfidenceLevel):
        return value
    return ConfidenceLevel(str(value))
