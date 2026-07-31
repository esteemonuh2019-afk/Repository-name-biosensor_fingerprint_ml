"""Scientific language, status, and confidence policies for interpretations."""

from __future__ import annotations

from collections.abc import Iterable

from src.scientific_reasoning.observation import ConfidenceLevel, Observation, ObservationStatus

from .enums import InterpretationConfidence, InterpretationStatus


FORBIDDEN_CAUSAL_TERMS: tuple[str, ...] = (
    "proves",
    "confirms",
    "demonstrates conclusively",
    "causes",
    "results in",
    "biologically explains",
    "publication-ready",
    "publication ready",
    "clinically useful",
    "field-ready",
    "field ready",
)

RECOMMENDATION_TERMS: tuple[str, ...] = (
    "should test",
    "should perform",
    "recommend",
    "future experiment",
    "ought to",
)

HYPOTHESIS_TERMS: tuple[str, ...] = (
    "we hypothesize",
    "may be caused by",
    "mechanism is",
    "pathway explains",
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

BLIND_VALIDATION_OVERCLAIM_TERMS: tuple[str, ...] = (
    "establishes external validation",
    "established external validation",
    "external validation was achieved",
    "external validation performance is available",
    "validated externally",
    "blind validation accuracy",
    "blind validation f1",
    "blind validation performance was measured",
    "true-label validation performance",
)

PREFERRED_CONSERVATIVE_TERMS: tuple[str, ...] = (
    "suggests",
    "indicates",
    "is consistent with",
    "is associated with",
    "supports the presence of",
    "remains limited by",
    "cannot yet establish",
)

CONFIDENCE_RANK: dict[InterpretationConfidence, int] = {
    InterpretationConfidence.NOT_ASSESSABLE: 0,
    InterpretationConfidence.LOW: 1,
    InterpretationConfidence.MODERATE: 2,
    InterpretationConfidence.HIGH: 3,
}


def find_terms(text: str, terms: Iterable[str]) -> tuple[str, ...]:
    lowered = text.lower()
    return tuple(term for term in terms if term in lowered)


def find_forbidden_causal_terms(text: str) -> tuple[str, ...]:
    return find_terms(text, FORBIDDEN_CAUSAL_TERMS)


def find_recommendation_terms(text: str) -> tuple[str, ...]:
    return find_terms(text, RECOMMENDATION_TERMS)


def find_hypothesis_terms(text: str) -> tuple[str, ...]:
    return find_terms(text, HYPOTHESIS_TERMS)


def find_literature_comparison_terms(text: str) -> tuple[str, ...]:
    return find_terms(text, LITERATURE_COMPARISON_TERMS)


def find_blind_validation_overclaim_terms(text: str) -> tuple[str, ...]:
    return find_terms(text, BLIND_VALIDATION_OVERCLAIM_TERMS)


def assign_confidence(
    supporting_observations: Iterable[Observation],
    contradicting_observations: Iterable[Observation] = (),
    *,
    critical_qc_limitation: bool = False,
    observation_validation_failed_critically: bool = False,
    evidence_is_indirect: bool = False,
) -> InterpretationConfidence:
    """Assign confidence from observation coherence, not metric magnitude."""

    supporting = tuple(supporting_observations)
    contradicting = tuple(contradicting_observations)
    if observation_validation_failed_critically or not supporting:
        return InterpretationConfidence.NOT_ASSESSABLE
    if contradicting or critical_qc_limitation or evidence_is_indirect:
        return InterpretationConfidence.LOW

    all_complete = all(observation.status == ObservationStatus.COMPLETE for observation in supporting)
    all_high = all(observation.confidence == ConfidenceLevel.HIGH for observation in supporting)
    if len(supporting) >= 2 and all_complete and all_high:
        return InterpretationConfidence.HIGH
    if all_complete and any(observation.confidence == ConfidenceLevel.HIGH for observation in supporting):
        return InterpretationConfidence.MODERATE
    return InterpretationConfidence.LOW


def assign_status(
    supporting_evidence_count: int,
    contradicting_evidence_count: int = 0,
    *,
    minimum_supporting_observations: int = 1,
    dependencies_valid: bool = True,
) -> InterpretationStatus:
    if not dependencies_valid:
        return InterpretationStatus.NOT_ASSESSABLE
    if supporting_evidence_count and contradicting_evidence_count:
        return InterpretationStatus.CONFLICTED
    if supporting_evidence_count >= minimum_supporting_observations:
        return InterpretationStatus.SUPPORTED
    if supporting_evidence_count > 0:
        return InterpretationStatus.PARTIALLY_SUPPORTED
    return InterpretationStatus.INSUFFICIENT_EVIDENCE


def supports_confidence_assignment(
    assigned: InterpretationConfidence,
    expected: InterpretationConfidence,
) -> bool:
    """Return whether an assigned confidence is no stronger than policy support."""

    return CONFIDENCE_RANK[assigned] <= CONFIDENCE_RANK[expected]
