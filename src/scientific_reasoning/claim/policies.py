"""Claim scoring, publication-use, and scientific-language policies."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .enums import ClaimStatus, ClaimType, ConfidenceLabel, EvidenceStrength, PublicationUse


LANGUAGE_POLICY_RULE_IDS: tuple[str, ...] = (
    "CLAIM-LANGUAGE-DESCRIPTIVE-001",
    "CLAIM-LANGUAGE-NO-CAUSAL-OVERCLAIM-001",
    "CLAIM-LANGUAGE-NO-MECHANISM-001",
    "CLAIM-LANGUAGE-NO-NOVELTY-001",
    "CLAIM-LANGUAGE-NO-EXTERNAL-VALIDATION-OVERCLAIM-001",
)

ALLOWED_CLAIM_TERMS: tuple[str, ...] = (
    "supports",
    "is associated with",
    "is consistent with",
    "provides evidence for",
    "under the current dataset",
    "under internal evaluation",
    "partially supports",
    "cannot yet establish",
    "remains uncertain",
    "may reflect",
)

CAUSAL_OVERCLAIM_TERMS: tuple[str, ...] = (
    "proves",
    "confirms conclusively",
    "establishes causation",
    "caused",
    "causes",
    "definitely",
    "certainly",
    "universally",
)

MECHANISTIC_OVERCLAIM_TERMS: tuple[str, ...] = (
    "molecular mechanism",
    "biological mechanism",
    "mechanism explains",
    "mechanism is",
)

NOVELTY_OVERCLAIM_TERMS: tuple[str, ...] = (
    "novel",
    "groundbreaking",
    "publication-ready",
)

EXTERNAL_VALIDATION_OVERCLAIM_TERMS: tuple[str, ...] = (
    "externally validated",
    "external validation performance",
    "field-ready",
    "field ready",
    "deployment-ready",
    "deployment ready",
    "clinically useful",
    "regulatory compliant",
    "regulatory suitability",
)

RECOMMENDATION_TERMS: tuple[str, ...] = (
    "should test",
    "recommend",
    "future experiment",
    "ought to perform",
    "should perform",
)

HYPOTHESIS_STATUS_POINTS = {
    "PLAUSIBLE": 18.0,
    "WEAKLY_SUPPORTED": 10.0,
    "COMPETING": 6.0,
    "CONFLICTED": 4.0,
    "INSUFFICIENT_EVIDENCE": 0.0,
    "NOT_ASSESSABLE": 0.0,
}

HYPOTHESIS_CONFIDENCE_POINTS = {
    "HIGH": 4.0,
    "MODERATE": 3.0,
    "LOW": 1.0,
    "NOT_ASSESSABLE": 0.0,
}


def find_terms(text: str, terms: Iterable[str]) -> tuple[str, ...]:
    lowered = text.lower()
    return tuple(term for term in terms if term in lowered)


def causal_overclaim_terms(text: str) -> tuple[str, ...]:
    return find_terms(text, CAUSAL_OVERCLAIM_TERMS)


def mechanistic_overclaim_terms(text: str) -> tuple[str, ...]:
    return find_terms(text, MECHANISTIC_OVERCLAIM_TERMS)


def novelty_overclaim_terms(text: str) -> tuple[str, ...]:
    return find_terms(text, NOVELTY_OVERCLAIM_TERMS)


def external_validation_overclaim_terms(text: str) -> tuple[str, ...]:
    lowered = text.lower()
    if "cannot yet" in lowered or "no external validation" in lowered or "not be assumed" in lowered:
        return tuple(term for term in EXTERNAL_VALIDATION_OVERCLAIM_TERMS if term in lowered and "validated" in term)
    return find_terms(text, EXTERNAL_VALIDATION_OVERCLAIM_TERMS)


def recommendation_terms(text: str) -> tuple[str, ...]:
    return find_terms(text, RECOMMENDATION_TERMS)


def evidence_strength_from_score(score: float) -> EvidenceStrength:
    if score <= 0:
        return EvidenceStrength.NOT_ASSESSABLE
    if score <= 29:
        return EvidenceStrength.INSUFFICIENT
    if score <= 59:
        return EvidenceStrength.LIMITED
    if score <= 79:
        return EvidenceStrength.MODERATE
    return EvidenceStrength.STRONG


def confidence_label_from_strength(strength: EvidenceStrength) -> ConfidenceLabel:
    strength = EvidenceStrength(strength)
    if strength is EvidenceStrength.STRONG:
        return ConfidenceLabel.HIGH
    if strength is EvidenceStrength.MODERATE:
        return ConfidenceLabel.MODERATE
    if strength in (EvidenceStrength.LIMITED, EvidenceStrength.INSUFFICIENT):
        return ConfidenceLabel.LOW
    return ConfidenceLabel.NOT_ASSESSABLE


def calculate_evidence_score(
    hypotheses: Iterable[dict[str, Any]],
    *,
    supporting_interpretation_count: int,
    supporting_observation_count: int,
    competing_hypothesis_count: int,
    evidence_gap_count: int,
    graph_traceable: bool,
    source_validation_passed: bool,
) -> float:
    """Return a deterministic 0-100 evidence-bounded score.

    The score is a structured evidence-support index, not a probability that
    the claim is true.
    """

    records = tuple(hypotheses)
    if not records:
        return 0.0
    hypothesis_support = min(
        sum(
            HYPOTHESIS_STATUS_POINTS.get(str(record.get("status", "NOT_ASSESSABLE")), 0.0)
            + HYPOTHESIS_CONFIDENCE_POINTS.get(str(record.get("confidence", "NOT_ASSESSABLE")), 0.0)
            for record in records
        ),
        30.0,
    )
    interpretation_support = min(max(supporting_interpretation_count, 0) * 5.0, 15.0)
    observation_support = min(max(supporting_observation_count, 0) * 5.0, 15.0)
    traceability_support = 15.0 if graph_traceable else 0.0
    validation_support = 10.0 if source_validation_passed else 0.0
    competing_penalty = min(max(competing_hypothesis_count, 0) * 5.0, 10.0)
    evidence_gap_penalty = min(max(evidence_gap_count, 0) * 1.0, 15.0)
    low_confidence_penalty = sum(4.0 for record in records if record.get("confidence") == "LOW")
    weak_support_penalty = sum(3.0 for record in records if record.get("status") == "WEAKLY_SUPPORTED")
    not_assessable_penalty = sum(
        20.0
        for record in records
        if record.get("confidence") == "NOT_ASSESSABLE" or record.get("status") == "NOT_ASSESSABLE"
    )
    insufficient_penalty = sum(20.0 for record in records if record.get("status") == "INSUFFICIENT_EVIDENCE")
    raw_score = (
        hypothesis_support
        + interpretation_support
        + observation_support
        + traceability_support
        + validation_support
        - competing_penalty
        - evidence_gap_penalty
        - low_confidence_penalty
        - weak_support_penalty
        - not_assessable_penalty
        - insufficient_penalty
    )
    return float(max(0.0, min(100.0, round(raw_score, 2))))


def publication_use_for_claim(
    *,
    claim_type: ClaimType,
    claim_status: ClaimStatus,
    evidence_strength: EvidenceStrength,
    has_critical_issue: bool,
    category: str,
) -> PublicationUse:
    claim_type = ClaimType(claim_type)
    claim_status = ClaimStatus(claim_status)
    evidence_strength = EvidenceStrength(evidence_strength)
    if claim_type is ClaimType.WITHHELD or claim_status is ClaimStatus.WITHHELD:
        return PublicationUse.NOT_ELIGIBLE
    if claim_type is ClaimType.LIMITATION:
        return PublicationUse.LIMITATION_ONLY
    if has_critical_issue:
        return PublicationUse.NOT_ELIGIBLE
    if evidence_strength in (EvidenceStrength.STRONG, EvidenceStrength.MODERATE):
        if claim_type is ClaimType.PRIMARY_FINDING and claim_status in (
            ClaimStatus.SUPPORTED,
            ClaimStatus.PARTIALLY_SUPPORTED,
        ):
            return PublicationUse.RESULTS_ELIGIBLE
        return PublicationUse.DISCUSSION_ELIGIBLE
    if evidence_strength is EvidenceStrength.LIMITED:
        if category in {"CONCENTRATION_INFORMATION", "GENERALIZATION"}:
            return PublicationUse.DISCUSSION_ELIGIBLE
        return PublicationUse.INTERNAL_REVIEW_ONLY
    return PublicationUse.NOT_ELIGIBLE
