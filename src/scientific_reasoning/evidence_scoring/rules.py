"""Versioned evidence scoring rules and thresholds."""

from __future__ import annotations

from types import MappingProxyType

from .enums import EvidenceDimension, EvidenceLevel
from .models import EVIDENCE_SCORING_RULE_VERSION


DIMENSION_WEIGHTS = MappingProxyType(
    {
        EvidenceDimension.TRACEABILITY: 0.12,
        EvidenceDimension.SOURCE_VALIDATION: 0.10,
        EvidenceDimension.OBSERVATION_SUPPORT: 0.15,
        EvidenceDimension.INTERPRETATION_SUPPORT: 0.10,
        EvidenceDimension.HYPOTHESIS_SUPPORT: 0.12,
        EvidenceDimension.COMPETING_HYPOTHESIS_CONTROL: 0.08,
        EvidenceDimension.EVIDENCE_GAP_BURDEN: 0.10,
        EvidenceDimension.LIMITATION_COMPLETENESS: 0.07,
        EvidenceDimension.INTERNAL_CONSISTENCY: 0.08,
        EvidenceDimension.GENERALIZATION_SUPPORT: 0.05,
        EvidenceDimension.REPRODUCIBILITY_SUPPORT: 0.03,
    }
)

WEIGHTED_DIMENSIONS: tuple[EvidenceDimension, ...] = tuple(DIMENSION_WEIGHTS)

EVIDENCE_LEVEL_THRESHOLDS: tuple[tuple[EvidenceLevel, float, float], ...] = (
    (EvidenceLevel.INSUFFICIENT, 0.0, 24.99),
    (EvidenceLevel.LIMITED, 25.0, 44.99),
    (EvidenceLevel.MODERATE, 45.0, 64.99),
    (EvidenceLevel.STRONG, 65.0, 79.99),
    (EvidenceLevel.VERY_STRONG, 80.0, 100.0),
)

SUPPORTED_CLAIM_SCHEMA_VERSIONS = ("BSIP-3.2.0",)
SUPPORTED_GRAPH_SCHEMA_VERSIONS = ("BSIP-3.1.0",)

RULE_IDS = MappingProxyType(
    {
        EvidenceDimension.TRACEABILITY: ("EVIDENCE-TRACEABILITY-001",),
        EvidenceDimension.SOURCE_VALIDATION: ("EVIDENCE-SOURCE-VALIDATION-001",),
        EvidenceDimension.OBSERVATION_SUPPORT: ("EVIDENCE-OBSERVATION-SUPPORT-001",),
        EvidenceDimension.INTERPRETATION_SUPPORT: ("EVIDENCE-INTERPRETATION-SUPPORT-001",),
        EvidenceDimension.HYPOTHESIS_SUPPORT: ("EVIDENCE-HYPOTHESIS-SUPPORT-001",),
        EvidenceDimension.COMPETING_HYPOTHESIS_CONTROL: ("EVIDENCE-COMPETITION-CONTROL-001",),
        EvidenceDimension.EVIDENCE_GAP_BURDEN: ("EVIDENCE-GAP-BURDEN-001",),
        EvidenceDimension.LIMITATION_COMPLETENESS: ("EVIDENCE-LIMITATION-COMPLETENESS-001",),
        EvidenceDimension.INTERNAL_CONSISTENCY: ("EVIDENCE-INTERNAL-CONSISTENCY-001",),
        EvidenceDimension.GENERALIZATION_SUPPORT: ("EVIDENCE-GENERALIZATION-SUPPORT-001",),
        EvidenceDimension.REPRODUCIBILITY_SUPPORT: ("EVIDENCE-REPRODUCIBILITY-SUPPORT-001",),
    }
)

MAJOR_EVIDENCE_GAP_TERMS = (
    "external validation",
    "independent",
    "causal",
    "causality",
    "mechanism",
    "reproducibility",
    "replication",
    "confounding",
    "batch",
    "sampling",
    "controls",
)

EXTERNAL_VALIDATION_TERMS = (
    "independently labelled",
    "external validation",
    "external dataset",
    "independent external",
)

INTERNAL_ONLY_TERMS = (
    "internal evaluation",
    "current dataset",
    "internal analyses",
    "current feature representation",
)


def validate_weights() -> None:
    total = round(sum(DIMENSION_WEIGHTS.values()), 10)
    if total != 1.0:
        raise ValueError(f"{EVIDENCE_SCORING_RULE_VERSION} weights must sum to exactly 1.0, found {total}.")
    for dimension, weight in DIMENSION_WEIGHTS.items():
        if weight < 0:
            raise ValueError(f"Evidence dimension weight cannot be negative: {dimension.value}")


def evidence_level_from_score(score: float) -> EvidenceLevel:
    bounded = max(0.0, min(100.0, float(score)))
    for level, minimum, maximum in EVIDENCE_LEVEL_THRESHOLDS:
        if minimum <= bounded <= maximum:
            return level
    return EvidenceLevel.VERY_STRONG
