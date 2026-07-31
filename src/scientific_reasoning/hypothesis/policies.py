"""Scientific language, falsifiability, confidence, and priority policies."""

from __future__ import annotations

from collections.abc import Iterable

from src.scientific_reasoning.interpretation import (
    Interpretation,
    InterpretationConfidence,
    InterpretationStatus,
)

from .enums import HypothesisCategory, HypothesisConfidence, HypothesisPriority


ALLOWED_MODAL_TERMS: tuple[str, ...] = (
    "may",
    "might",
    "could",
    "is consistent with the possibility that",
    "remains plausible",
    "cannot yet distinguish between",
)

FORBIDDEN_HYPOTHESIS_TERMS: tuple[str, ...] = (
    "proves",
    "confirms",
    "demonstrates conclusively",
    "is caused by",
    "mechanism is",
    "definitely",
    "certainly",
    "establishes",
    "publication-ready",
    "publication ready",
    "clinically useful",
    "field-ready",
    "field ready",
    "regulatory",
    "deployment-ready",
)

RECOMMENDATION_TERMS: tuple[str, ...] = (
    "should test",
    "recommend",
    "future experiment",
    "ought to perform",
    "should perform",
)

LITERATURE_COMPARISON_TERMS: tuple[str, ...] = (
    "compared with literature",
    "compared to literature",
    "previous studies",
    "published studies",
    "state of the art",
    "state-of-the-art",
    "literature benchmark",
)

PROTOCOL_TERMS: tuple[str, ...] = (
    "using protocol",
    "incubate",
    "pipette",
    "plate layout",
    "randomize wells",
    "run an experiment",
    "perform an experiment",
)

FALSIFIABILITY_WEAKENING_TERMS: tuple[str, ...] = (
    "weakened",
    "contradicted",
    "not reproducibly",
    "does not reproducibly",
    "do not reproducibly",
    "fails to",
    "cannot distinguish",
)

CONFIDENCE_RANK: dict[HypothesisConfidence, int] = {
    HypothesisConfidence.NOT_ASSESSABLE: 0,
    HypothesisConfidence.LOW: 1,
    HypothesisConfidence.MODERATE: 2,
    HypothesisConfidence.HIGH: 3,
}

INTERPRETATION_CONFIDENCE_RANK: dict[InterpretationConfidence, int] = {
    InterpretationConfidence.NOT_ASSESSABLE: 0,
    InterpretationConfidence.LOW: 1,
    InterpretationConfidence.MODERATE: 2,
    InterpretationConfidence.HIGH: 3,
}

RESEARCH_RELEVANCE: dict[HypothesisCategory, int] = {
    HypothesisCategory.TEMPORAL_INFORMATION: 18,
    HypothesisCategory.CHEMICAL_DISCRIMINATION: 20,
    HypothesisCategory.CONCENTRATION_ENCODING: 18,
    HypothesisCategory.FEATURE_REPRESENTATION: 16,
    HypothesisCategory.STRAIN_CONTRIBUTION: 14,
    HypothesisCategory.DATA_QUALITY_EFFECT: 16,
    HypothesisCategory.GENERALIZATION: 20,
    HypothesisCategory.OVERALL_SYSTEM_BEHAVIOR: 20,
}


def find_terms(text: str, terms: Iterable[str]) -> tuple[str, ...]:
    lowered = text.lower()
    return tuple(term for term in terms if term in lowered)


def find_forbidden_hypothesis_terms(text: str) -> tuple[str, ...]:
    return find_terms(text, FORBIDDEN_HYPOTHESIS_TERMS)


def find_recommendation_terms(text: str) -> tuple[str, ...]:
    return find_terms(text, RECOMMENDATION_TERMS)


def find_literature_comparison_terms(text: str) -> tuple[str, ...]:
    return find_terms(text, LITERATURE_COMPARISON_TERMS)


def find_protocol_terms(text: str) -> tuple[str, ...]:
    return find_terms(text, PROTOCOL_TERMS)


def has_allowed_hypothesis_modality(text: str) -> bool:
    return bool(find_terms(text, ALLOWED_MODAL_TERMS))


def falsifiability_is_valid(statement: str | None) -> bool:
    if not statement or not statement.strip():
        return False
    lowered = statement.lower()
    return bool(find_terms(lowered, FALSIFIABILITY_WEAKENING_TERMS)) and not find_protocol_terms(lowered)


def assign_confidence(
    supporting_interpretations: Iterable[Interpretation],
    contradicting_interpretations: Iterable[Interpretation] = (),
    *,
    evidence_gap_count: int = 0,
    external_validation_gap: bool = False,
    mechanistic_explanation: bool = False,
) -> HypothesisConfidence:
    supporting = tuple(supporting_interpretations)
    contradicting = tuple(contradicting_interpretations)
    if not supporting:
        return HypothesisConfidence.NOT_ASSESSABLE
    if contradicting or mechanistic_explanation:
        return HypothesisConfidence.LOW
    if external_validation_gap or evidence_gap_count >= 2:
        return HypothesisConfidence.MODERATE if len(supporting) >= 2 else HypothesisConfidence.LOW
    all_usable = all(
        interpretation.status in (InterpretationStatus.SUPPORTED, InterpretationStatus.PARTIALLY_SUPPORTED)
        for interpretation in supporting
    )
    all_moderate_or_high = all(
        INTERPRETATION_CONFIDENCE_RANK[interpretation.confidence] >= INTERPRETATION_CONFIDENCE_RANK[
            InterpretationConfidence.MODERATE
        ]
        for interpretation in supporting
    )
    if len(supporting) >= 3 and all_usable and all_moderate_or_high:
        return HypothesisConfidence.HIGH
    if len(supporting) >= 2 and all_usable:
        return HypothesisConfidence.MODERATE
    return HypothesisConfidence.LOW


def supports_confidence_assignment(
    assigned: HypothesisConfidence,
    expected: HypothesisConfidence,
) -> bool:
    return CONFIDENCE_RANK[assigned] <= CONFIDENCE_RANK[expected]


def priority_score(
    category: HypothesisCategory,
    supporting_interpretations: Iterable[Interpretation],
    contradicting_interpretations: Iterable[Interpretation] = (),
    *,
    confidence: HypothesisConfidence,
    evidence_gap_count: int,
) -> float:
    supporting = tuple(supporting_interpretations)
    contradicting = tuple(contradicting_interpretations)
    support_points = min(len(supporting) * 10, 30)
    interpretation_confidence_points = min(
        sum(INTERPRETATION_CONFIDENCE_RANK[item.confidence] * 4 for item in supporting),
        24,
    )
    hypothesis_confidence_points = CONFIDENCE_RANK[confidence] * 8
    contradiction_penalty = len(contradicting) * 20
    gap_penalty = evidence_gap_count * 5
    score = (
        RESEARCH_RELEVANCE[category]
        + support_points
        + interpretation_confidence_points
        + hypothesis_confidence_points
        - contradiction_penalty
        - gap_penalty
    )
    return float(max(0, min(100, round(score, 2))))


def priority_from_score(score: float) -> HypothesisPriority:
    if score <= 0:
        return HypothesisPriority.NOT_ASSESSABLE
    if 1 <= score <= 39:
        return HypothesisPriority.LOW
    if 40 <= score <= 69:
        return HypothesisPriority.MEDIUM
    return HypothesisPriority.HIGH
