"""Uncertainty assessment for evidence score records."""

from __future__ import annotations

from typing import Any

from .enums import UncertaintyLevel
from .models import UncertaintyAssessment
from .rules import MAJOR_EVIDENCE_GAP_TERMS
from .traceability import ClaimTraceability


def assess_uncertainty(claim: dict[str, Any], traceability: ClaimTraceability) -> UncertaintyAssessment:
    sources: list[str] = []
    penalties: list[str] = []
    points = 0
    evidence_gaps = tuple(str(item) for item in claim.get("evidence_gaps", ()) or ())
    limitations = tuple(str(item) for item in claim.get("limitations", ()) or ())
    text = " ".join([str(claim.get("claim_text", "")), " ".join(evidence_gaps), " ".join(limitations)]).lower()

    if claim.get("competing_hypothesis_ids"):
        sources.append("Unresolved competing hypotheses are preserved.")
        penalties.append("competing_hypotheses")
        points += 20
    if evidence_gaps:
        sources.append("Evidence gaps remain attached to the claim.")
        penalties.append("evidence_gaps")
        points += min(25, len(evidence_gaps) * 5)
    if any(term in text for term in MAJOR_EVIDENCE_GAP_TERMS):
        sources.append("At least one major evidence gap concerns validation, controls, confounding, causality, or reproducibility.")
        penalties.append("major_evidence_gap")
        points += 20
    if "no independent external validation" in text or "no true external validation" in text:
        sources.append("No genuine external validation is available.")
        penalties.append("no_external_validation")
        points += 20
    if "quality-control" in text or "quality control" in text or claim.get("category") == "DATA_QUALITY":
        sources.append("Quality-control limitations contribute uncertainty.")
        penalties.append("quality_control_limitations")
        points += 10
    if claim.get("claim_status") in {"TENTATIVE", "CONFLICTED"}:
        sources.append(f"Claim status is {claim.get('claim_status')}.")
        penalties.append("tentative_or_conflicted_status")
        points += 15
    if not traceability.complete_support_chain:
        sources.append("Complete support-chain traceability is missing.")
        penalties.append("incomplete_traceability")
        points += 40
    if not traceability.has_external_validation:
        sources.append("The traceability package does not contain a genuine external-validation signal.")
        penalties.append("no_graph_external_validation_signal")
        points += 10

    level = _level_from_points(points, claim_status=str(claim.get("claim_status", "")))
    explanation = (
        "Uncertainty is assessed from competing hypotheses, evidence gaps, validation boundaries, "
        "source consistency, and graph traceability; it is not the inverse of the evidence score."
    )
    return UncertaintyAssessment(
        uncertainty_level=level,
        uncertainty_sources=tuple(sources),
        uncertainty_penalties=tuple(penalties),
        uncertainty_explanation=explanation,
    )


def _level_from_points(points: int, *, claim_status: str) -> UncertaintyLevel:
    if points >= 70:
        return UncertaintyLevel.VERY_HIGH
    if points >= 50:
        return UncertaintyLevel.HIGH
    if points >= 30:
        return UncertaintyLevel.MODERATE
    if points >= 15:
        return UncertaintyLevel.LOW if claim_status != "CONFLICTED" else UncertaintyLevel.MODERATE
    return UncertaintyLevel.VERY_LOW if claim_status != "CONFLICTED" else UncertaintyLevel.MODERATE
