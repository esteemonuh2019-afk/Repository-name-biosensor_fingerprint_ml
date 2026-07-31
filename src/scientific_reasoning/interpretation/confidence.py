"""Rule-facing confidence helpers for the Scientific Interpretation Engine."""

from __future__ import annotations

from collections.abc import Iterable

from src.scientific_reasoning.observation import Observation

from .enums import InterpretationConfidence
from .policies import assign_confidence


def rule_based_confidence(
    supporting_observations: Iterable[Observation],
    contradicting_observations: Iterable[Observation] = (),
    *,
    critical_qc_limitation: bool = False,
    external_validation_absent: bool = False,
    evidence_is_indirect: bool = False,
    observation_validation_failed_critically: bool = False,
) -> InterpretationConfidence:
    """Apply the contract confidence policy with rule-level context."""

    confidence = assign_confidence(
        supporting_observations,
        contradicting_observations,
        critical_qc_limitation=critical_qc_limitation,
        evidence_is_indirect=evidence_is_indirect,
        observation_validation_failed_critically=observation_validation_failed_critically,
    )
    if external_validation_absent and confidence == InterpretationConfidence.HIGH:
        return InterpretationConfidence.MODERATE
    return confidence
