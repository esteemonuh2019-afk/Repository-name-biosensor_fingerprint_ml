"""Deterministic rule set for BSIP scientific interpretations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from src.scientific_reasoning.observation import Observation, ObservationCategory

from .confidence import rule_based_confidence
from .enums import EvidenceDirection, InterpretationCategory, InterpretationStatus
from .models import Interpretation, InterpretationEvidenceLink, ReasoningRule
from .policies import (
    FORBIDDEN_CAUSAL_TERMS,
    HYPOTHESIS_TERMS,
    LITERATURE_COMPARISON_TERMS,
    RECOMMENDATION_TERMS,
    assign_status,
)


DEFAULT_SOURCE_OBSERVATION_SCHEMA_VERSION = "BSIP-2.0"


RULE_DATA_QUALITY = "RULE-DATA-QUALITY-001"
RULE_FINGERPRINT_STRUCTURE = "RULE-FINGERPRINT-STRUCTURE-001"
RULE_CHEMICAL_CLASSIFICATION = "RULE-CHEMICAL-CLASSIFICATION-001"
RULE_CONCENTRATION_REGRESSION = "RULE-CONCENTRATION-REGRESSION-001"
RULE_FEATURE_ENGINEERING = "RULE-FEATURE-ENGINEERING-001"
RULE_FEATURE_SELECTION = "RULE-FEATURE-SELECTION-001"
RULE_STRAIN_CONTRIBUTION = "RULE-STRAIN-CONTRIBUTION-001"
RULE_BLIND_VALIDATION = "RULE-BLIND-VALIDATION-001"
RULE_OVERALL_EVIDENCE = "RULE-OVERALL-EVIDENCE-001"


def _all_forbidden_terms() -> tuple[str, ...]:
    return tuple(
        sorted(
            set(FORBIDDEN_CAUSAL_TERMS)
            | set(RECOMMENDATION_TERMS)
            | set(HYPOTHESIS_TERMS)
            | set(LITERATURE_COMPARISON_TERMS)
        )
    )


INTERPRETATION_RULES: tuple[ReasoningRule, ...] = (
    ReasoningRule(
        rule_id=RULE_DATA_QUALITY,
        name="Data-quality limitations",
        description="Interpret active QC and package-validation limitations from observations.",
        required_categories=(InterpretationCategory.DATA_QUALITY,),
        optional_categories=(InterpretationCategory.OVERALL_EVIDENCE,),
        minimum_supporting_observations=1,
        allowed_claim_template=(
            "The available quality-control observations indicate that the analysis package contains "
            "active data-quality limitations that should be considered when interpreting downstream results."
        ),
        forbidden_terms=_all_forbidden_terms(),
    ),
    ReasoningRule(
        rule_id=RULE_FINGERPRINT_STRUCTURE,
        name="Fingerprint and exploratory structure",
        description="Interpret structured variation from fingerprint and exploratory observations.",
        required_categories=(InterpretationCategory.FINGERPRINT_STRUCTURE,),
        optional_categories=(InterpretationCategory.EXPLORATORY_STRUCTURE,),
        minimum_supporting_observations=1,
        allowed_claim_template=(
            "The fingerprint and exploratory observations indicate that structured variation is present "
            "in the derived biosensor response profiles."
        ),
        forbidden_terms=_all_forbidden_terms(),
    ),
    ReasoningRule(
        rule_id=RULE_CHEMICAL_CLASSIFICATION,
        name="Chemical-class discrimination",
        description="Interpret chemical-class discrimination evidence from classification observations.",
        required_categories=(InterpretationCategory.CHEMICAL_CLASSIFICATION,),
        minimum_supporting_observations=1,
        allowed_claim_template=(
            "The classification observations suggest that the derived biosensor fingerprints contain "
            "information associated with chemical-class discrimination."
        ),
        forbidden_terms=_all_forbidden_terms(),
    ),
    ReasoningRule(
        rule_id=RULE_CONCENTRATION_REGRESSION,
        name="Concentration-regression signal",
        description="Interpret concentration-related signal from regression observations.",
        required_categories=(InterpretationCategory.CONCENTRATION_REGRESSION,),
        minimum_supporting_observations=1,
        allowed_claim_template=(
            "The regression observations indicate that concentration-related information is present, "
            "while a substantial proportion of target variance remains unaccounted for."
        ),
        forbidden_terms=_all_forbidden_terms(),
    ),
    ReasoningRule(
        rule_id=RULE_FEATURE_ENGINEERING,
        name="Feature-engineering association",
        description="Interpret reported feature-engineering benchmark associations.",
        required_categories=(InterpretationCategory.FEATURE_ENGINEERING,),
        minimum_supporting_observations=1,
        allowed_claim_template=(
            "The feature-engineering observations indicate that the selected temporal feature family was "
            "associated with higher reported benchmark values than the reference feature configuration."
        ),
        forbidden_terms=_all_forbidden_terms(),
    ),
    ReasoningRule(
        rule_id=RULE_FEATURE_SELECTION,
        name="Feature-selection evaluation",
        description="Interpret the presence of reduced or selected feature-set evaluation.",
        required_categories=(InterpretationCategory.FEATURE_SELECTION,),
        minimum_supporting_observations=1,
        allowed_claim_template=(
            "The feature-selection outputs document that reduced or selected feature sets were evaluated."
        ),
        forbidden_terms=_all_forbidden_terms(),
    ),
    ReasoningRule(
        rule_id=RULE_STRAIN_CONTRIBUTION,
        name="Strain-contribution evaluation",
        description="Interpret the presence of differential strain-contribution evaluation.",
        required_categories=(InterpretationCategory.STRAIN_CONTRIBUTION,),
        minimum_supporting_observations=1,
        allowed_claim_template=(
            "The strain-contribution outputs indicate that differential strain contribution was evaluated "
            "across the biosensor array."
        ),
        forbidden_terms=_all_forbidden_terms(),
    ),
    ReasoningRule(
        rule_id=RULE_BLIND_VALIDATION,
        name="Blind-label availability",
        description="Interpret blind-prediction validation boundaries from blind-prediction observations.",
        required_categories=(InterpretationCategory.BLIND_VALIDATION,),
        minimum_supporting_observations=1,
        allowed_claim_template=(
            "The available blind-prediction observations do not establish external validation performance "
            "because true labels were absent."
        ),
        forbidden_terms=_all_forbidden_terms(),
    ),
    ReasoningRule(
        rule_id=RULE_OVERALL_EVIDENCE,
        name="Overall evidence boundary",
        description="Integrate observation dependencies from classification, regression, QC, and blind validation.",
        required_categories=(InterpretationCategory.OVERALL_EVIDENCE,),
        optional_categories=(
            InterpretationCategory.CHEMICAL_CLASSIFICATION,
            InterpretationCategory.CONCENTRATION_REGRESSION,
            InterpretationCategory.DATA_QUALITY,
            InterpretationCategory.BLIND_VALIDATION,
        ),
        minimum_supporting_observations=1,
        allowed_claim_template=(
            "The available evidence supports further evaluation of biosensor fingerprint classification, "
            "while quantitative concentration prediction and external validation remain limited by the current "
            "evidence base."
        ),
        forbidden_terms=_all_forbidden_terms(),
    ),
)


def build_interpretations(
    observations: Iterable[Observation],
    *,
    software_version: str,
    source_observation_schema_version: str | None = DEFAULT_SOURCE_OBSERVATION_SCHEMA_VERSION,
    created_at: str,
    metadata: dict[str, Any] | None = None,
) -> tuple[Interpretation, ...]:
    """Build deterministic interpretations from validated observations only."""

    ordered_observations = tuple(sorted(observations, key=lambda observation: observation.observation_id))
    grouped = _group_by_observation_category(ordered_observations)
    context = dict(metadata or {})
    interpretations: list[Interpretation] = []

    data_quality = _build_data_quality(
        grouped,
        software_version=software_version,
        schema_version=source_observation_schema_version,
        created_at=created_at,
        metadata=context,
    )
    if data_quality:
        interpretations.append(data_quality)

    fingerprint = _build_fingerprint_structure(
        grouped,
        software_version=software_version,
        schema_version=source_observation_schema_version,
        created_at=created_at,
    )
    if fingerprint:
        interpretations.append(fingerprint)

    classification = _build_classification(
        grouped,
        software_version=software_version,
        schema_version=source_observation_schema_version,
        created_at=created_at,
    )
    if classification:
        interpretations.append(classification)

    regression = _build_regression(
        grouped,
        software_version=software_version,
        schema_version=source_observation_schema_version,
        created_at=created_at,
    )
    if regression:
        interpretations.append(regression)

    feature_engineering = _build_feature_engineering(
        grouped,
        software_version=software_version,
        schema_version=source_observation_schema_version,
        created_at=created_at,
    )
    if feature_engineering:
        interpretations.append(feature_engineering)

    feature_selection = _build_feature_selection(
        grouped,
        software_version=software_version,
        schema_version=source_observation_schema_version,
        created_at=created_at,
    )
    if feature_selection:
        interpretations.append(feature_selection)

    strain = _build_strain_contribution(
        grouped,
        software_version=software_version,
        schema_version=source_observation_schema_version,
        created_at=created_at,
    )
    if strain:
        interpretations.append(strain)

    blind = _build_blind_validation(
        grouped,
        software_version=software_version,
        schema_version=source_observation_schema_version,
        created_at=created_at,
    )
    if blind:
        interpretations.append(blind)

    overall = _build_overall_evidence(
        interpretations,
        software_version=software_version,
        schema_version=source_observation_schema_version,
        created_at=created_at,
    )
    if overall:
        interpretations.append(overall)

    return tuple(sorted(interpretations, key=lambda interpretation: interpretation.interpretation_id))


def _build_data_quality(
    grouped: dict[ObservationCategory, tuple[Observation, ...]],
    *,
    software_version: str,
    schema_version: str | None,
    created_at: str,
    metadata: dict[str, Any],
) -> Interpretation | None:
    qc_observations = grouped.get(ObservationCategory.QUALITY_CONTROL, ())
    validation_observations = grouped.get(ObservationCategory.VALIDATION, ())
    if not qc_observations or not _has_active_data_quality_limitation(qc_observations + validation_observations):
        return None
    supporting = _dedupe_observations(qc_observations + validation_observations)
    return _make_interpretation(
        interpretation_id="INT-DATA_QUALITY-0001",
        category=InterpretationCategory.DATA_QUALITY,
        title="Data-quality limitations",
        claim=(
            "The available quality-control observations indicate that the analysis package contains "
            "active data-quality limitations that should be considered when interpreting downstream results."
        ),
        status=assign_status(len(supporting)),
        confidence=rule_based_confidence(supporting),
        supporting=supporting,
        assumptions=("Validated Observation Engine outputs are the authoritative source for this interpretation.",),
        limitations=_collect_limitations(supporting),
        rule_id=RULE_DATA_QUALITY,
        software_version=software_version,
        schema_version=schema_version,
        created_at=created_at,
        tags=("data-quality", "quality-control"),
        metadata={
            "source_validation_passed": metadata.get("observation_validation_passed"),
            "source_observation_count": metadata.get("source_observation_count"),
        },
    )


def _build_fingerprint_structure(
    grouped: dict[ObservationCategory, tuple[Observation, ...]],
    *,
    software_version: str,
    schema_version: str | None,
    created_at: str,
) -> Interpretation | None:
    supporting = _dedupe_observations(
        grouped.get(ObservationCategory.FINGERPRINT, ())
        + grouped.get(ObservationCategory.EXPLORATORY_ANALYSIS, ())
    )
    if not supporting:
        return None
    return _make_interpretation(
        interpretation_id="INT-FINGERPRINT_STRUCTURE-0001",
        category=InterpretationCategory.FINGERPRINT_STRUCTURE,
        title="Fingerprint response-profile structure",
        claim=(
            "The fingerprint and exploratory observations indicate that structured variation is present "
            "in the derived biosensor response profiles."
        ),
        status=assign_status(len(supporting)),
        confidence=rule_based_confidence(supporting),
        supporting=supporting,
        assumptions=("Fingerprint and exploratory observations passed Observation Engine validation.",),
        limitations=_collect_limitations(supporting),
        rule_id=RULE_FINGERPRINT_STRUCTURE,
        software_version=software_version,
        schema_version=schema_version,
        created_at=created_at,
        tags=("fingerprint", "exploratory"),
    )


def _build_classification(
    grouped: dict[ObservationCategory, tuple[Observation, ...]],
    *,
    software_version: str,
    schema_version: str | None,
    created_at: str,
) -> Interpretation | None:
    supporting = grouped.get(ObservationCategory.CLASSIFICATION, ())
    if not supporting:
        return None
    return _make_interpretation(
        interpretation_id="INT-CHEMICAL_CLASSIFICATION-0001",
        category=InterpretationCategory.CHEMICAL_CLASSIFICATION,
        title="Chemical-class discrimination evidence",
        claim=(
            "The classification observations suggest that the derived biosensor fingerprints contain "
            "information associated with chemical-class discrimination."
        ),
        status=assign_status(len(supporting)),
        confidence=rule_based_confidence(supporting, external_validation_absent=True),
        supporting=supporting,
        assumptions=("Classification interpretation depends only on validated classification observations.",),
        limitations=_collect_limitations(supporting)
        or ("No claim is made about external blind-validation performance.",),
        rule_id=RULE_CHEMICAL_CLASSIFICATION,
        software_version=software_version,
        schema_version=schema_version,
        created_at=created_at,
        tags=("classification", "chemical-class"),
    )


def _build_regression(
    grouped: dict[ObservationCategory, tuple[Observation, ...]],
    *,
    software_version: str,
    schema_version: str | None,
    created_at: str,
) -> Interpretation | None:
    supporting = grouped.get(ObservationCategory.REGRESSION, ())
    if not supporting:
        return None
    regression_metric = _first_numeric_metric(supporting, ("r2_mean", "explained_variance_mean"))
    if regression_metric is None:
        claim = (
            "The regression observations are insufficient to assess concentration-related interpretation "
            "because R-squared and explained-variance evidence were not available."
        )
        status = InterpretationStatus.INSUFFICIENT_EVIDENCE
        confidence = rule_based_confidence(supporting, evidence_is_indirect=True)
    elif 0 < regression_metric < 0.5:
        claim = (
            "The regression observations indicate that concentration-related information is present, "
            "while a substantial proportion of target variance remains unaccounted for."
        )
        status = assign_status(len(supporting))
        confidence = rule_based_confidence(supporting, external_validation_absent=True)
    elif regression_metric > 0:
        claim = (
            "The regression observations indicate that concentration-related information is present in the "
            "reported benchmark."
        )
        status = assign_status(len(supporting))
        confidence = rule_based_confidence(supporting, external_validation_absent=True)
    else:
        claim = (
            "The regression observations do not meet the current rule threshold for concentration-related "
            "interpretation because the reported R-squared or explained-variance value was not above zero."
        )
        status = InterpretationStatus.INSUFFICIENT_EVIDENCE
        confidence = rule_based_confidence(supporting, evidence_is_indirect=True)
    return _make_interpretation(
        interpretation_id="INT-CONCENTRATION_REGRESSION-0001",
        category=InterpretationCategory.CONCENTRATION_REGRESSION,
        title="Concentration-regression evidence",
        claim=claim,
        status=status,
        confidence=confidence,
        supporting=supporting,
        assumptions=("Regression interpretation depends only on validated regression observations.",),
        limitations=_collect_limitations(supporting),
        rule_id=RULE_CONCENTRATION_REGRESSION,
        software_version=software_version,
        schema_version=schema_version,
        created_at=created_at,
        tags=("regression", "concentration"),
    )


def _build_feature_engineering(
    grouped: dict[ObservationCategory, tuple[Observation, ...]],
    *,
    software_version: str,
    schema_version: str | None,
    created_at: str,
) -> Interpretation | None:
    supporting = grouped.get(ObservationCategory.FEATURE_ENGINEERING, ())
    if not supporting:
        return None
    return _make_interpretation(
        interpretation_id="INT-FEATURE_ENGINEERING-0001",
        category=InterpretationCategory.FEATURE_ENGINEERING,
        title="Feature-engineering benchmark association",
        claim=(
            "The feature-engineering observations indicate that the selected temporal feature family was "
            "associated with higher reported benchmark values than the reference feature configuration."
        ),
        status=assign_status(len(supporting)),
        confidence=rule_based_confidence(supporting, external_validation_absent=True),
        supporting=supporting,
        assumptions=("Feature-engineering interpretation is limited to reported benchmark associations.",),
        limitations=_collect_limitations(supporting)
        or ("No causal claim is made about feature-family effects.",),
        rule_id=RULE_FEATURE_ENGINEERING,
        software_version=software_version,
        schema_version=schema_version,
        created_at=created_at,
        tags=("feature-engineering",),
    )


def _build_feature_selection(
    grouped: dict[ObservationCategory, tuple[Observation, ...]],
    *,
    software_version: str,
    schema_version: str | None,
    created_at: str,
) -> Interpretation | None:
    supporting = grouped.get(ObservationCategory.FEATURE_SELECTION, ())
    if not supporting:
        return None
    return _make_interpretation(
        interpretation_id="INT-FEATURE_SELECTION-0001",
        category=InterpretationCategory.FEATURE_SELECTION,
        title="Feature-selection evaluation",
        claim="The feature-selection outputs document that reduced or selected feature sets were evaluated.",
        status=assign_status(len(supporting)),
        confidence=rule_based_confidence(supporting),
        supporting=supporting,
        assumptions=("Feature-selection interpretation depends only on validated feature-selection observations.",),
        limitations=_collect_limitations(supporting)
        or ("No performance-preservation claim is made unless explicit observation evidence is present.",),
        rule_id=RULE_FEATURE_SELECTION,
        software_version=software_version,
        schema_version=schema_version,
        created_at=created_at,
        tags=("feature-selection",),
    )


def _build_strain_contribution(
    grouped: dict[ObservationCategory, tuple[Observation, ...]],
    *,
    software_version: str,
    schema_version: str | None,
    created_at: str,
) -> Interpretation | None:
    supporting = grouped.get(ObservationCategory.STRAIN_CONTRIBUTION, ())
    if not supporting:
        return None
    return _make_interpretation(
        interpretation_id="INT-STRAIN_CONTRIBUTION-0001",
        category=InterpretationCategory.STRAIN_CONTRIBUTION,
        title="Strain-contribution evaluation",
        claim=(
            "The strain-contribution outputs indicate that differential strain contribution was evaluated "
            "across the biosensor array."
        ),
        status=assign_status(len(supporting)),
        confidence=rule_based_confidence(supporting),
        supporting=supporting,
        assumptions=("Strain-contribution interpretation depends only on validated strain-contribution observations.",),
        limitations=_collect_limitations(supporting)
        or ("No biological importance claim is made for any individual strain.",),
        rule_id=RULE_STRAIN_CONTRIBUTION,
        software_version=software_version,
        schema_version=schema_version,
        created_at=created_at,
        tags=("strain-contribution",),
    )


def _build_blind_validation(
    grouped: dict[ObservationCategory, tuple[Observation, ...]],
    *,
    software_version: str,
    schema_version: str | None,
    created_at: str,
) -> Interpretation | None:
    supporting = grouped.get(ObservationCategory.BLIND_PREDICTION, ())
    if not supporting:
        return None
    labels_available = _blind_labels_available(supporting)
    if labels_available:
        claim = (
            "The blind-prediction observations include true-label availability, but external-validation "
            "performance must remain limited to observed validation metrics."
        )
        status = assign_status(len(supporting))
    else:
        claim = (
            "The available blind-prediction observations do not establish external validation performance "
            "because true labels were absent."
        )
        status = InterpretationStatus.PARTIALLY_SUPPORTED
    return _make_interpretation(
        interpretation_id="INT-BLIND_VALIDATION-0001",
        category=InterpretationCategory.BLIND_VALIDATION,
        title="Blind-validation boundary",
        claim=claim,
        status=status,
        confidence=rule_based_confidence(supporting, external_validation_absent=not labels_available),
        supporting=supporting,
        assumptions=("Blind-validation interpretation uses only validated blind-prediction observations.",),
        limitations=_collect_limitations(supporting)
        or ("No external validation performance claim is made when true labels are absent.",),
        rule_id=RULE_BLIND_VALIDATION,
        software_version=software_version,
        schema_version=schema_version,
        created_at=created_at,
        tags=("blind-validation",),
    )


def _build_overall_evidence(
    interpretations: Iterable[Interpretation],
    *,
    software_version: str,
    schema_version: str | None,
    created_at: str,
) -> Interpretation | None:
    component_categories = {
        InterpretationCategory.CHEMICAL_CLASSIFICATION,
        InterpretationCategory.CONCENTRATION_REGRESSION,
        InterpretationCategory.DATA_QUALITY,
        InterpretationCategory.BLIND_VALIDATION,
    }
    components = tuple(
        interpretation
        for interpretation in interpretations
        if interpretation.category in component_categories
    )
    if not components:
        return None
    observation_ids = tuple(sorted({item for component in components for item in component.supporting_observation_ids}))
    if not observation_ids:
        return None
    evidence_links = tuple(
        InterpretationEvidenceLink(
            observation_id=observation_id,
            direction=EvidenceDirection.SUPPORTING,
            rationale="Observation dependency is inherited from component interpretation rules.",
            metric_names=tuple(
                sorted(
                    {
                        metric
                        for component in components
                        for link in component.evidence_summary
                        if link.observation_id == observation_id
                        for metric in link.metric_names
                    }
                )
            ),
            provenance_ids=tuple(
                sorted(
                    {
                        provenance
                        for component in components
                        for link in component.evidence_summary
                        if link.observation_id == observation_id
                        for provenance in link.provenance_ids
                    }
                )
            ),
            source_files=tuple(
                sorted(
                    {
                        source
                        for component in components
                        for link in component.evidence_summary
                        if link.observation_id == observation_id
                        for source in link.source_files
                    }
                )
            ),
        )
        for observation_id in observation_ids
    )
    limitations = tuple(
        sorted({limitation for component in components for limitation in component.limitations})
    ) or ("Overall interpretation is limited to the currently validated observation set.",)
    return Interpretation(
        interpretation_id="INT-OVERALL_EVIDENCE-0001",
        category=InterpretationCategory.OVERALL_EVIDENCE,
        title="Overall evidence boundary",
        claim=(
            "The available evidence supports further evaluation of biosensor fingerprint classification, "
            "while quantitative concentration prediction and external validation remain limited by the current "
            "evidence base."
        ),
        status=assign_status(len(observation_ids)),
        confidence=rule_based_confidence((), evidence_is_indirect=True)
        if not observation_ids
        else _overall_confidence(components),
        supporting_observation_ids=observation_ids,
        contradicting_observation_ids=tuple(),
        assumptions=("Overall interpretation combines existing observation dependencies from component rules.",),
        limitations=limitations,
        evidence_summary=evidence_links,
        reasoning_rule_ids=(RULE_OVERALL_EVIDENCE,),
        created_at=created_at,
        software_version=software_version,
        source_observation_schema_version=schema_version,
        tags=("overall-evidence",),
        metadata={
            "component_interpretation_ids": tuple(sorted(component.interpretation_id for component in components)),
        },
    )


def _make_interpretation(
    *,
    interpretation_id: str,
    category: InterpretationCategory,
    title: str,
    claim: str,
    status: InterpretationStatus,
    confidence,
    supporting: tuple[Observation, ...],
    assumptions: tuple[str, ...],
    limitations: tuple[str, ...],
    rule_id: str,
    software_version: str,
    schema_version: str | None,
    created_at: str,
    tags: tuple[str, ...],
    metadata: dict[str, Any] | None = None,
) -> Interpretation:
    supporting = tuple(sorted(supporting, key=lambda observation: observation.observation_id))
    return Interpretation(
        interpretation_id=interpretation_id,
        category=category,
        title=title,
        claim=claim,
        status=status,
        confidence=confidence,
        supporting_observation_ids=tuple(observation.observation_id for observation in supporting),
        contradicting_observation_ids=tuple(),
        assumptions=assumptions,
        limitations=limitations,
        evidence_summary=tuple(_evidence_link(observation, rule_id) for observation in supporting),
        reasoning_rule_ids=(rule_id,),
        created_at=created_at,
        software_version=software_version,
        source_observation_schema_version=schema_version,
        tags=tags,
        metadata=metadata or {},
    )


def _evidence_link(observation: Observation, rule_id: str) -> InterpretationEvidenceLink:
    return InterpretationEvidenceLink(
        observation_id=observation.observation_id,
        direction=EvidenceDirection.SUPPORTING,
        rationale=f"Observation supports deterministic reasoning rule {rule_id}.",
        metric_names=tuple(sorted(metric.metric_name for metric in observation.supporting_metrics)),
        provenance_ids=tuple(sorted(record.provenance_id for record in observation.provenance_records)),
        source_files=tuple(sorted(observation.supporting_files)),
    )


def _group_by_observation_category(
    observations: Iterable[Observation],
) -> dict[ObservationCategory, tuple[Observation, ...]]:
    grouped: dict[ObservationCategory, list[Observation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.category].append(observation)
    return {
        category: tuple(sorted(items, key=lambda observation: observation.observation_id))
        for category, items in grouped.items()
    }


def _has_active_data_quality_limitation(observations: Iterable[Observation]) -> bool:
    metric_tokens = ("error", "warning", "failed", "excluded", "missing_required", "validation_issue")
    for observation in observations:
        if observation.limitations:
            return True
        metadata = dict(observation.metadata)
        if metadata.get("package_validation_passed") is False:
            return True
        for metric in observation.supporting_metrics:
            if any(token in metric.metric_name for token in metric_tokens):
                value = _numeric_or_none(metric.metric_value)
                if value is not None and value > 0:
                    return True
    return False


def _blind_labels_available(observations: Iterable[Observation]) -> bool:
    for observation in observations:
        for metric in observation.supporting_metrics:
            if metric.metric_name == "true_labels_included":
                return bool(metric.metric_value)
        if "true_labels_included" in observation.metadata:
            return bool(observation.metadata["true_labels_included"])
    return False


def _first_numeric_metric(observations: Iterable[Observation], metric_names: tuple[str, ...]) -> float | None:
    for metric_name in metric_names:
        for observation in observations:
            for metric in observation.supporting_metrics:
                if metric.metric_name == metric_name:
                    value = _numeric_or_none(metric.metric_value)
                    if value is not None:
                        return value
    return None


def _numeric_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        return None


def _collect_limitations(observations: Iterable[Observation]) -> tuple[str, ...]:
    limitations = sorted({limitation for observation in observations for limitation in observation.limitations})
    return tuple(limitations)


def _dedupe_observations(observations: Iterable[Observation]) -> tuple[Observation, ...]:
    by_id = {observation.observation_id: observation for observation in observations}
    return tuple(by_id[key] for key in sorted(by_id))


def _overall_confidence(components: tuple[Interpretation, ...]):
    confidences = {component.confidence.value for component in components}
    if "NOT_ASSESSABLE" in confidences:
        return "LOW"
    if "LOW" in confidences:
        return "LOW"
    return "MODERATE"
