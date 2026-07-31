"""Deterministic multidimensional evidence scoring."""

from __future__ import annotations

from typing import Any

from .enums import EvidenceDimension, EvidenceLevel, PublicationReadiness
from .models import (
    EVIDENCE_SCORING_RULE_VERSION,
    EVIDENCE_SCORING_SOFTWARE_VERSION,
    DimensionScore,
    EvidenceScoreRecord,
)
from .publication_policy import publication_readiness_for, reviewer_confidence_for
from .rules import (
    DIMENSION_WEIGHTS,
    EXTERNAL_VALIDATION_TERMS,
    INTERNAL_ONLY_TERMS,
    MAJOR_EVIDENCE_GAP_TERMS,
    RULE_IDS,
    WEIGHTED_DIMENSIONS,
    evidence_level_from_score,
    validate_weights,
)
from .traceability import ClaimTraceability, ReasoningGraphIndex, trace_claim
from .uncertainty import assess_uncertainty


def score_claims(
    claims: tuple[dict[str, Any], ...],
    graph_document: dict[str, Any],
    *,
    claim_validation_passed: bool,
    graph_validation_passed: bool,
    source_claim_schema_version: str | None,
    source_graph_schema_version: str | None,
    software_version: str = EVIDENCE_SCORING_SOFTWARE_VERSION,
) -> tuple[EvidenceScoreRecord, ...]:
    validate_weights()
    graph_index = ReasoningGraphIndex(graph_document)
    records = [
        score_claim(
            claim,
            graph_index,
            claim_validation_passed=claim_validation_passed,
            graph_validation_passed=graph_validation_passed,
            source_claim_schema_version=source_claim_schema_version,
            source_graph_schema_version=source_graph_schema_version,
            software_version=software_version,
        )
        for claim in claims
    ]
    return tuple(sorted(records, key=lambda record: record.claim_id))


def score_claim(
    claim: dict[str, Any],
    graph_index: ReasoningGraphIndex,
    *,
    claim_validation_passed: bool,
    graph_validation_passed: bool,
    source_claim_schema_version: str | None,
    source_graph_schema_version: str | None,
    software_version: str = EVIDENCE_SCORING_SOFTWARE_VERSION,
) -> EvidenceScoreRecord:
    traceability = trace_claim(claim, graph_index)
    source_validated = claim_validation_passed and graph_validation_passed
    withholding_reasons = _withholding_reasons(claim, traceability, source_validated)
    dimension_scores = _dimension_scores(claim, traceability, source_validated=source_validated)
    weighted_score = round(sum(score.weighted_contribution for score in dimension_scores.values()), 2)
    normalized_score = 0.0 if withholding_reasons else max(0.0, min(100.0, weighted_score))
    evidence_level = EvidenceLevel.INSUFFICIENT if withholding_reasons else evidence_level_from_score(normalized_score)
    uncertainty = assess_uncertainty(claim, traceability)
    readiness, ceilings, readiness_explanation = publication_readiness_for(
        claim_publication_use=str(claim.get("publication_use", "")),
        claim_type=str(claim.get("claim_type", "")),
        claim_status=str(claim.get("claim_status", "")),
        evidence_level=evidence_level,
        uncertainty_level=uncertainty.uncertainty_level,
        has_external_validation=traceability.has_external_validation,
        traceable=traceability.complete_support_chain,
        is_withheld=bool(withholding_reasons),
    )
    if not traceability.has_external_validation and evidence_level is EvidenceLevel.VERY_STRONG:
        evidence_level = EvidenceLevel.STRONG
        normalized_score = min(normalized_score, 79.99)
        readiness, ceilings, readiness_explanation = publication_readiness_for(
            claim_publication_use=str(claim.get("publication_use", "")),
            claim_type=str(claim.get("claim_type", "")),
            claim_status=str(claim.get("claim_status", "")),
            evidence_level=evidence_level,
            uncertainty_level=uncertainty.uncertainty_level,
            has_external_validation=False,
            traceable=traceability.complete_support_chain,
            is_withheld=bool(withholding_reasons),
        )
    reviewer_confidence, reviewer_explanation = reviewer_confidence_for(
        evidence_level,
        uncertainty.uncertainty_level,
        traceable=traceability.complete_support_chain,
        source_validated=source_validated,
        claim_status=str(claim.get("claim_status", "")),
    )
    positive = _positive_factors(dimension_scores)
    negative = _negative_factors(dimension_scores) + tuple(ceilings)
    return EvidenceScoreRecord(
        claim_id=str(claim.get("claim_id", "")),
        claim_category=str(claim.get("category", "")),
        claim_type=str(claim.get("claim_type", "")),
        claim_status=str(claim.get("claim_status", "")),
        claim_publication_use=str(claim.get("publication_use", "")),
        dimension_scores=dimension_scores,
        weighted_score=weighted_score,
        normalized_score=round(normalized_score, 2),
        evidence_level=evidence_level,
        uncertainty_level=uncertainty.uncertainty_level,
        reviewer_confidence=reviewer_confidence,
        publication_readiness=readiness,
        positive_factors=positive,
        negative_factors=negative,
        evidence_gaps=traceability.evidence_gap_ids,
        limitations=tuple(str(item) for item in claim.get("limitations", ()) or ()),
        competing_hypothesis_ids=tuple(str(item) for item in claim.get("competing_hypothesis_ids", ()) or ()),
        supporting_observation_ids=tuple(str(item) for item in claim.get("supporting_observation_ids", ()) or ()),
        supporting_interpretation_ids=tuple(str(item) for item in claim.get("supporting_interpretation_ids", ()) or ()),
        supporting_hypothesis_ids=tuple(str(item) for item in claim.get("supporting_hypothesis_ids", ()) or ()),
        reasoning_graph_node_ids=traceability.referenced_node_ids,
        score_explanation=(
            "Scores are deterministic evidence-support indices aggregated from independently inspectable dimensions; "
            "they are not probabilities, p-values, causal certainty, mechanism proof, novelty evidence, or external-validity evidence."
        ),
        withholding_reasons=withholding_reasons,
        is_withheld=bool(withholding_reasons),
        uncertainty_sources=uncertainty.uncertainty_sources,
        uncertainty_penalties=uncertainty.uncertainty_penalties,
        uncertainty_explanation=uncertainty.uncertainty_explanation,
        reviewer_confidence_explanation=reviewer_explanation,
        publication_readiness_explanation=readiness_explanation,
        source_claim_schema_version=source_claim_schema_version,
        source_graph_schema_version=source_graph_schema_version,
        evidence_scoring_rule_version=EVIDENCE_SCORING_RULE_VERSION,
        software_version=software_version,
    )


def _dimension_scores(
    claim: dict[str, Any],
    traceability: ClaimTraceability,
    *,
    source_validated: bool,
) -> dict[EvidenceDimension, DimensionScore]:
    scores = {
        EvidenceDimension.TRACEABILITY: _traceability_score(claim, traceability),
        EvidenceDimension.SOURCE_VALIDATION: _source_validation_score(claim, source_validated=source_validated),
        EvidenceDimension.OBSERVATION_SUPPORT: _observation_score(claim, traceability),
        EvidenceDimension.INTERPRETATION_SUPPORT: _interpretation_score(claim, traceability),
        EvidenceDimension.HYPOTHESIS_SUPPORT: _hypothesis_score(claim, traceability),
        EvidenceDimension.COMPETING_HYPOTHESIS_CONTROL: _competition_score(claim),
        EvidenceDimension.EVIDENCE_GAP_BURDEN: _gap_score(claim, traceability),
        EvidenceDimension.LIMITATION_COMPLETENESS: _limitation_score(claim),
        EvidenceDimension.INTERNAL_CONSISTENCY: _internal_consistency_score(claim, traceability),
        EvidenceDimension.GENERALIZATION_SUPPORT: _generalization_score(claim, traceability),
        EvidenceDimension.REPRODUCIBILITY_SUPPORT: _reproducibility_score(claim, traceability, source_validated=source_validated),
    }
    return {dimension: _weighted(score) for dimension, score in scores.items()}


def _weighted(score: DimensionScore) -> DimensionScore:
    contribution = round(score.raw_score * score.weight, 4)
    return DimensionScore(
        dimension=score.dimension,
        raw_score=round(score.raw_score, 2),
        weight=score.weight,
        weighted_contribution=contribution,
        positive_factors=score.positive_factors,
        penalties=score.penalties,
        rule_ids=score.rule_ids,
        source_node_ids=score.source_node_ids,
        ceilings=score.ceilings,
        explanation=score.explanation,
    )


def _score(
    dimension: EvidenceDimension,
    raw_score: float,
    *,
    positive: tuple[str, ...] = (),
    penalties: tuple[str, ...] = (),
    source_nodes: tuple[str, ...] = (),
    ceilings: tuple[str, ...] = (),
    explanation: str,
) -> DimensionScore:
    return DimensionScore(
        dimension=dimension,
        raw_score=max(0.0, min(100.0, raw_score)),
        weight=DIMENSION_WEIGHTS[dimension],
        weighted_contribution=0.0,
        positive_factors=positive,
        penalties=penalties,
        rule_ids=RULE_IDS[dimension],
        source_node_ids=source_nodes,
        ceilings=ceilings,
        explanation=explanation,
    )


def _traceability_score(claim: dict[str, Any], traceability: ClaimTraceability) -> DimensionScore:
    score = 100.0
    positive = []
    penalties = []
    if traceability.supporting_hypothesis_ids:
        positive.append("claim links to supporting hypotheses")
    else:
        score -= 35
        penalties.append("missing supporting hypothesis")
    if traceability.supporting_interpretation_ids:
        positive.append("claim links to interpretations")
    else:
        score -= 20
        penalties.append("missing supporting interpretation")
    if traceability.supporting_observation_ids:
        positive.append("claim links to observations")
    else:
        score -= 20
        penalties.append("missing supporting observation")
    if traceability.validation_summary_ids:
        positive.append("validation-summary nodes are referenced")
    else:
        score -= 15
        penalties.append("missing validation-summary nodes")
    if traceability.missing_node_ids:
        score -= 50
        penalties.append("referenced graph nodes are missing")
    if not traceability.complete_support_chain:
        score -= 40
        penalties.append("incomplete observation-to-interpretation-to-hypothesis support chain")
    return _score(
        EvidenceDimension.TRACEABILITY,
        score,
        positive=tuple(positive),
        penalties=tuple(penalties),
        source_nodes=traceability.referenced_node_ids,
        explanation="Traceability rewards complete graph paths from observations through interpretations to hypotheses and validation nodes.",
    )


def _source_validation_score(claim: dict[str, Any], *, source_validated: bool) -> DimensionScore:
    score = 100.0 if source_validated else 0.0
    positive = ("Claim and reasoning-graph source validation passed.",) if source_validated else ()
    penalties = () if source_validated else ("source validation failed",)
    return _score(
        EvidenceDimension.SOURCE_VALIDATION,
        score,
        positive=positive,
        penalties=penalties,
        source_nodes=tuple(str(item) for item in claim.get("validation_summary_ids", ()) or ()),
        explanation="Source validation checks claim package validation, reasoning-graph validation, readability, and critical source issues.",
    )


def _observation_score(claim: dict[str, Any], traceability: ClaimTraceability) -> DimensionScore:
    observations = traceability.supporting_observation_ids
    diversity = {_observation_family(obs) for obs in observations}
    score = 0.0
    positive = []
    penalties = []
    if observations:
        score = 35 + min(len(observations), 3) * 12 + min(len(diversity), 3) * 8
        positive.append("supporting observations are present")
        if len(diversity) > 1:
            positive.append("more than one observation family is represented")
    else:
        penalties.append("no supporting observations")
    if diversity == {"EXPLORATORY_ANALYSIS"}:
        score -= 25
        penalties.append("observation support is exploratory only")
    if "QC" in diversity or claim.get("category") == "DATA_QUALITY":
        score -= 5
        penalties.append("quality-control context increases uncertainty")
    return _score(
        EvidenceDimension.OBSERVATION_SUPPORT,
        score,
        positive=tuple(positive),
        penalties=tuple(penalties),
        source_nodes=observations,
        explanation="Observation support considers number, relevance, and coarse diversity of supporting observation nodes without treating count as independent replication.",
    )


def _interpretation_score(claim: dict[str, Any], traceability: ClaimTraceability) -> DimensionScore:
    interpretations = traceability.supporting_interpretation_ids
    score = 0.0
    positive = []
    penalties = []
    if interpretations:
        score = 45 + min(len(interpretations), 3) * 12
        positive.append("supporting interpretations are present")
    else:
        penalties.append("no supporting interpretations")
    if claim.get("claim_status") == "CONFLICTED":
        score -= 20
        penalties.append("claim status is conflicted")
    if claim.get("claim_status") == "TENTATIVE":
        score -= 10
        penalties.append("claim status is tentative")
    if claim.get("competing_hypothesis_ids"):
        positive.append("alternative explanations are preserved")
    return _score(
        EvidenceDimension.INTERPRETATION_SUPPORT,
        score,
        positive=tuple(positive),
        penalties=tuple(penalties),
        source_nodes=interpretations,
        explanation="Interpretation support rewards traceable interpretations while preserving conditional or conflicted support.",
    )


def _hypothesis_score(claim: dict[str, Any], traceability: ClaimTraceability) -> DimensionScore:
    score = 0.0
    positive = []
    penalties = []
    if traceability.supporting_hypothesis_ids:
        score = 45 + min(len(traceability.supporting_hypothesis_ids), 2) * 10
        positive.append("supporting hypotheses are present")
    else:
        penalties.append("no supporting hypotheses")
    confidence = claim.get("confidence_label")
    status = claim.get("claim_status")
    if confidence == "MODERATE":
        score += 10
        positive.append("claim confidence label is moderate")
    elif confidence == "HIGH":
        score += 18
        positive.append("claim confidence label is high")
    elif confidence == "LOW":
        score -= 10
        penalties.append("claim confidence label is low")
    if status == "TENTATIVE":
        score -= 18
        penalties.append("claim depends on tentative support")
    if status == "CONFLICTED":
        score -= 25
        penalties.append("claim depends on conflicted support")
    return _score(
        EvidenceDimension.HYPOTHESIS_SUPPORT,
        score,
        positive=tuple(positive),
        penalties=tuple(penalties),
        source_nodes=traceability.supporting_hypothesis_ids,
        explanation="Hypothesis support evaluates support status, confidence label, and whether claim support is tentative or conflicted.",
    )


def _competition_score(claim: dict[str, Any]) -> DimensionScore:
    competitors = tuple(str(item) for item in claim.get("competing_hypothesis_ids", ()) or ())
    text = _claim_text_bundle(claim)
    positive = []
    penalties = []
    if not competitors:
        score = 85.0
        positive.append("no explicit competing hypothesis is attached to this claim")
    else:
        score = 78.0
        positive.append("competing hypotheses are explicitly preserved")
        if any(term in text for term in ("plausible alternatives", "remain plausible", "alternative explanation", "may not be")):
            positive.append("limitations acknowledge unresolved alternatives")
        else:
            score -= 20
            penalties.append("competition is present but not explicitly bounded in limitations")
        if claim.get("claim_status") == "CONFLICTED":
            score -= 25
            penalties.append("claim status is conflicted")
        if any(term in text for term in ("concentration", "batch", "dimensionality", "sampling", "experimental structure")):
            score -= 8
            penalties.append("unresolved confounding remains plausible")
    return _score(
        EvidenceDimension.COMPETING_HYPOTHESIS_CONTROL,
        score,
        positive=tuple(positive),
        penalties=tuple(penalties),
        source_nodes=competitors,
        explanation="Competition control rewards explicit preservation and partial bounding of alternatives while penalizing unresolved confounding.",
    )


def _gap_score(claim: dict[str, Any], traceability: ClaimTraceability) -> DimensionScore:
    gaps = traceability.evidence_gap_ids
    text = _claim_text_bundle(claim)
    score = 100.0 - min(len(gaps) * 8, 55)
    positive = []
    penalties = []
    if not gaps:
        positive.append("no evidence-gap nodes are attached")
    else:
        penalties.append("evidence gaps are attached to the claim")
    major_terms = tuple(term for term in MAJOR_EVIDENCE_GAP_TERMS if term in text)
    if major_terms:
        score -= min(len(major_terms) * 5, 30)
        penalties.append("major evidence gaps concern validation, confounding, controls, causality, or reproducibility")
    if "external validation" in text:
        penalties.append("external-validation gap remains")
    return _score(
        EvidenceDimension.EVIDENCE_GAP_BURDEN,
        score,
        positive=tuple(positive),
        penalties=tuple(penalties),
        source_nodes=gaps,
        explanation="Evidence-gap burden penalizes gap count and severity, with larger penalties for validation, controls, confounding, causality, mechanism, or reproducibility gaps.",
    )


def _limitation_score(claim: dict[str, Any]) -> DimensionScore:
    limitations = tuple(str(item) for item in claim.get("limitations", ()) or ())
    text = _claim_text_bundle(claim)
    score = 0.0
    positive = []
    penalties = []
    if limitations:
        score = 45 + min(len(limitations), 3) * 12
        positive.append("explicit limitations are present")
    else:
        penalties.append("no limitations are present")
    if any(term in text for term in INTERNAL_ONLY_TERMS):
        score += 10
        positive.append("internal-only evaluation boundary is explicit")
    if "external validation" in text:
        score += 10
        positive.append("external-validation limitation is explicit")
    if any(term in text for term in ("plausible alternatives", "remain plausible", "may not be", "different metrics")):
        score += 10
        positive.append("major limitations match evidence gaps or alternatives")
    return _score(
        EvidenceDimension.LIMITATION_COMPLETENESS,
        score,
        positive=tuple(positive),
        penalties=tuple(penalties),
        source_nodes=tuple(str(item) for item in claim.get("evidence_gap_ids", ()) or ()),
        explanation="Limitation completeness rewards explicit limitations that match actual evidence gaps and validation boundaries.",
    )


def _internal_consistency_score(claim: dict[str, Any], traceability: ClaimTraceability) -> DimensionScore:
    score = 92.0
    positive = ["claim dependencies and status fields are present"]
    penalties = []
    claim_engine_score = float(claim.get("evidence_score") or 0)
    if claim.get("publication_use") == "RESULTS_ELIGIBLE" and claim_engine_score < 60:
        score -= 25
        penalties.append("results-eligible claim has low upstream evidence score")
    if claim.get("claim_type") == "LIMITATION" and claim.get("publication_use") != "LIMITATION_ONLY":
        score -= 30
        penalties.append("limitation claim is not limitation-only")
    if claim.get("claim_status") == "TENTATIVE" and claim.get("confidence_label") == "HIGH":
        score -= 25
        penalties.append("tentative claim has high confidence label")
    if traceability.missing_node_ids:
        score -= 35
        penalties.append("claim references missing graph nodes")
    if "probability" in str(claim.get("rationale", "")).lower() and "not a probability" not in str(claim.get("rationale", "")).lower():
        score -= 20
        penalties.append("rationale could imply probabilistic interpretation")
    return _score(
        EvidenceDimension.INTERNAL_CONSISTENCY,
        score,
        positive=tuple(positive),
        penalties=tuple(penalties),
        source_nodes=traceability.referenced_node_ids,
        explanation="Internal consistency checks claim status, confidence, publication use, limitations, graph links, and upstream evidence-score alignment.",
    )


def _generalization_score(claim: dict[str, Any], traceability: ClaimTraceability) -> DimensionScore:
    text = _claim_text_bundle(claim)
    if traceability.has_external_validation:
        score = 100.0
        positive = ("genuine external-validation signal is traceable",)
        penalties = ()
    else:
        score = 20.0
        positive = ()
        penalties = ("no genuine external validation is traceable",)
        if any(term in text for term in EXTERNAL_VALIDATION_TERMS):
            score += 10
            positive = ("external-validation boundary is explicitly acknowledged",)
        if claim.get("category") == "GENERALIZATION":
            score -= 5
            penalties = penalties + ("generalization claim remains bounded by missing external validation",)
    return _score(
        EvidenceDimension.GENERALIZATION_SUPPORT,
        score,
        positive=positive,
        penalties=penalties,
        source_nodes=traceability.supporting_observation_ids,
        explanation="Generalization support rewards only genuine external or independently labelled unknown-sample evidence; internal validation is not external validation.",
    )


def _reproducibility_score(claim: dict[str, Any], traceability: ClaimTraceability, *, source_validated: bool) -> DimensionScore:
    score = 30.0
    positive = []
    penalties = []
    if source_validated:
        score += 25
        positive.append("source packages validate successfully")
    else:
        penalties.append("source validation failed")
    if claim.get("software_version") and claim.get("source_graph_schema_version") and claim.get("source_hypothesis_schema_version"):
        score += 20
        positive.append("software and schema versions are preserved")
    else:
        penalties.append("software or schema version metadata is incomplete")
    if traceability.complete_support_chain:
        score += 15
        positive.append("support-chain traceability is repeatable")
    if tuple(sorted(traceability.referenced_node_ids)) == traceability.referenced_node_ids:
        score += 5
        positive.append("dependency ordering is deterministic")
    return _score(
        EvidenceDimension.REPRODUCIBILITY_SUPPORT,
        score,
        positive=tuple(positive),
        penalties=tuple(penalties),
        source_nodes=traceability.referenced_node_ids,
        explanation="Reproducibility support evaluates deterministic source generation, validation status, schema/software metadata, and traceable ordered dependencies.",
    )


def _withholding_reasons(
    claim: dict[str, Any],
    traceability: ClaimTraceability,
    source_validated: bool,
) -> tuple[str, ...]:
    reasons = []
    if not source_validated:
        reasons.append("source validation failed")
    if str(claim.get("claim_type", "")) == "WITHHELD" or str(claim.get("claim_status", "")) == "WITHHELD":
        reasons.append("claim is withheld by upstream Claim Engine")
    if not traceability.supporting_hypothesis_ids:
        reasons.append("no valid supporting hypothesis exists")
    if traceability.missing_node_ids:
        reasons.append("critical reasoning-graph dependency is missing")
    if not traceability.complete_support_chain:
        reasons.append("no valid observation-to-interpretation-to-hypothesis traceability path exists")
    return tuple(sorted(set(reasons)))


def _positive_factors(dimension_scores: dict[EvidenceDimension, DimensionScore]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                factor
                for dimension in WEIGHTED_DIMENSIONS
                for factor in dimension_scores[dimension].positive_factors
            }
        )
    )


def _negative_factors(dimension_scores: dict[EvidenceDimension, DimensionScore]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                penalty
                for dimension in WEIGHTED_DIMENSIONS
                for penalty in dimension_scores[dimension].penalties
            }
        )
    )


def _claim_text_bundle(claim: dict[str, Any]) -> str:
    return " ".join(
        [
            " ".join(str(item) for item in claim.get("limitations", ()) or ()),
            " ".join(str(item) for item in claim.get("evidence_gaps", ()) or ()),
            " ".join(str(item) for item in claim.get("tags", ()) or ()),
        ]
    ).lower()


def _observation_family(observation_id: str) -> str:
    token = observation_id.replace("OBS-", "").split("-000", 1)[0]
    return token or observation_id
