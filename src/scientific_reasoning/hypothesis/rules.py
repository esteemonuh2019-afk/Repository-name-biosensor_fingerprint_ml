"""Deterministic BSIP v2.2.0 hypothesis generation rules."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from src.scientific_reasoning.interpretation import Interpretation

from .enums import HypothesisCategory, HypothesisConfidence, HypothesisStatus
from .models import Hypothesis
from .policies import assign_confidence, priority_from_score, priority_score


DEFAULT_SOURCE_INTERPRETATION_SCHEMA_VERSION = "BSIP-2.1.0"

RULE_TEMPORAL_INFORMATION = "RULE-TEMPORAL-INFORMATION-001"
RULE_CHEMICAL_DISCRIMINATION = "RULE-CHEMICAL-DISCRIMINATION-001"
RULE_CONCENTRATION_ENCODING = "RULE-CONCENTRATION-ENCODING-001"
RULE_FEATURE_REPRESENTATION = "RULE-FEATURE-REPRESENTATION-001"
RULE_STRAIN_CONTRIBUTION = "RULE-STRAIN-CONTRIBUTION-001"
RULE_DATA_QUALITY_EFFECT = "RULE-DATA-QUALITY-EFFECT-001"
RULE_GENERALIZATION = "RULE-GENERALIZATION-001"
RULE_OVERALL_SYSTEM_BEHAVIOR = "RULE-OVERALL-SYSTEM-BEHAVIOR-001"


def build_hypotheses(
    interpretations: Iterable[Interpretation],
    *,
    software_version: str,
    source_interpretation_schema_version: str | None = DEFAULT_SOURCE_INTERPRETATION_SCHEMA_VERSION,
    created_at: str,
    metadata: dict[str, Any] | None = None,
) -> tuple[Hypothesis, ...]:
    by_id = {interpretation.interpretation_id: interpretation for interpretation in interpretations}
    hypotheses: list[Hypothesis] = []

    hypotheses.extend(
        _temporal_information(
            by_id,
            software_version=software_version,
            schema_version=source_interpretation_schema_version,
            created_at=created_at,
        )
    )
    hypotheses.extend(
        _chemical_discrimination(
            by_id,
            software_version=software_version,
            schema_version=source_interpretation_schema_version,
            created_at=created_at,
        )
    )
    hypotheses.extend(
        _concentration_encoding(
            by_id,
            software_version=software_version,
            schema_version=source_interpretation_schema_version,
            created_at=created_at,
        )
    )
    hypotheses.extend(
        _feature_representation(
            by_id,
            software_version=software_version,
            schema_version=source_interpretation_schema_version,
            created_at=created_at,
        )
    )
    hypotheses.extend(
        _strain_contribution(
            by_id,
            software_version=software_version,
            schema_version=source_interpretation_schema_version,
            created_at=created_at,
        )
    )
    hypotheses.extend(
        _data_quality_effect(
            by_id,
            software_version=software_version,
            schema_version=source_interpretation_schema_version,
            created_at=created_at,
        )
    )
    hypotheses.extend(
        _generalization(
            by_id,
            software_version=software_version,
            schema_version=source_interpretation_schema_version,
            created_at=created_at,
        )
    )
    hypotheses.extend(
        _overall_system_behavior(
            by_id,
            software_version=software_version,
            schema_version=source_interpretation_schema_version,
            created_at=created_at,
            metadata=metadata or {},
        )
    )

    return tuple(sorted(hypotheses, key=lambda hypothesis: hypothesis.hypothesis_id))


def _temporal_information(by_id: dict[str, Interpretation], **context) -> tuple[Hypothesis, ...]:
    required = (
        "INT-FINGERPRINT_STRUCTURE-0001",
        "INT-FEATURE_ENGINEERING-0001",
        "INT-CHEMICAL_CLASSIFICATION-0001",
    )
    supporting = _available(by_id, required)
    if len(supporting) != len(required):
        return tuple()
    return (
        _make_hypothesis(
            hypothesis_id="HYP-TEMPORAL_INFORMATION-0001",
            category=HypothesisCategory.TEMPORAL_INFORMATION,
            title="Temporal information contribution",
            statement=(
                "Temporal characteristics of the biosensor response profiles may contribute discriminatory "
                "information beyond static summary measurements."
            ),
            status=HypothesisStatus.PLAUSIBLE,
            supporting=supporting,
            evidence_gaps=(
                "No direct causal test is available.",
                "No independent temporal-feature ablation is documented in the interpretation package.",
                "No external validation is available.",
            ),
            falsifiability_statement=(
                "This hypothesis would be weakened if models using temporally resolved features do not "
                "reproducibly outperform equivalent models restricted to static or endpoint features."
            ),
            rationale=(
                "Fingerprint-structure, feature-engineering, and classification interpretations jointly support "
                "a testable temporal-information explanation without establishing causation."
            ),
            rule_id=RULE_TEMPORAL_INFORMATION,
            tags=("temporal-information",),
            **context,
        ),
    )


def _chemical_discrimination(by_id: dict[str, Interpretation], **context) -> tuple[Hypothesis, ...]:
    required = ("INT-CHEMICAL_CLASSIFICATION-0001", "INT-FINGERPRINT_STRUCTURE-0001")
    supporting = _available(by_id, required)
    if len(supporting) != len(required):
        return tuple()
    primary = _make_hypothesis(
        hypothesis_id="HYP-CHEMICAL_DISCRIMINATION-0001",
        category=HypothesisCategory.CHEMICAL_DISCRIMINATION,
        title="Chemical-class response-pattern distinction",
        statement=(
            "Different chemical classes may produce partially distinct multistrain response patterns that "
            "contribute to classification."
        ),
        status=HypothesisStatus.PLAUSIBLE,
        supporting=supporting,
        evidence_gaps=(
            "The current interpretation package does not establish chemical identity as the only explanatory factor.",
            "No external validation is available.",
        ),
        falsifiability_statement=(
            "This hypothesis would be weakened if chemical-class labels cannot be distinguished after accounting "
            "for concentration, batch, and other correlated experimental structure."
        ),
        rationale=(
            "Classification and fingerprint-structure interpretations support a testable chemical-discrimination "
            "hypothesis while leaving correlated-structure alternatives unresolved."
        ),
        rule_id=RULE_CHEMICAL_DISCRIMINATION,
        alternative_hypothesis_ids=("HYP-CHEMICAL_DISCRIMINATION-0002",),
        tags=("chemical-discrimination",),
        **context,
    )
    competing = _make_hypothesis(
        hypothesis_id="HYP-CHEMICAL_DISCRIMINATION-0002",
        category=HypothesisCategory.CHEMICAL_DISCRIMINATION,
        title="Correlated-structure alternative",
        statement=(
            "Observed classification may depend partly on concentration, batch, or other correlated experimental "
            "structure rather than chemical identity alone."
        ),
        status=HypothesisStatus.COMPETING,
        supporting=supporting,
        evidence_gaps=(
            "The interpretation package does not isolate concentration, batch, or correlated structure effects.",
            "No external validation is available.",
        ),
        falsifiability_statement=(
            "This competing hypothesis would be weakened if classification remains reproducible after correlated "
            "concentration, batch, and experimental-structure effects are accounted for."
        ),
        rationale=(
            "The classification interpretation permits a competing explanation because internal classification "
            "performance alone cannot establish chemical identity as the sole source of separation."
        ),
        rule_id=RULE_CHEMICAL_DISCRIMINATION,
        alternative_hypothesis_ids=("HYP-CHEMICAL_DISCRIMINATION-0001",),
        tags=("chemical-discrimination", "competing"),
        **context,
    )
    return (primary, competing)


def _concentration_encoding(by_id: dict[str, Interpretation], **context) -> tuple[Hypothesis, ...]:
    required = ("INT-CONCENTRATION_REGRESSION-0001",)
    supporting = _available(by_id, required)
    if len(supporting) != len(required):
        return tuple()
    primary = _make_hypothesis(
        hypothesis_id="HYP-CONCENTRATION_ENCODING-0001",
        category=HypothesisCategory.CONCENTRATION_ENCODING,
        title="Concentration-related response encoding",
        statement=(
            "Biosensor response profiles may contain concentration-related information, but the current feature "
            "representation does not capture all concentration-dependent variation."
        ),
        status=HypothesisStatus.WEAKLY_SUPPORTED,
        supporting=supporting,
        evidence_gaps=(
            "Only one concentration-regression interpretation directly supports this hypothesis.",
            "No external validation is available.",
        ),
        falsifiability_statement=(
            "This hypothesis would be weakened if concentration-related prediction does not reproducibly exceed "
            "uninformative or label-permuted baselines under comparable internal evaluation."
        ),
        rationale=(
            "The concentration-regression interpretation supports possible concentration encoding while preserving "
            "the limitation that target variance remains incompletely accounted for."
        ),
        rule_id=RULE_CONCENTRATION_ENCODING,
        alternative_hypothesis_ids=("HYP-CONCENTRATION_ENCODING-0002",),
        tags=("concentration",),
        **context,
    )
    alternative = _make_hypothesis(
        hypothesis_id="HYP-CONCENTRATION_ENCODING-0002",
        category=HypothesisCategory.CONCENTRATION_ENCODING,
        title="Chemical-specific heterogeneity alternative",
        statement=(
            "Concentration prediction may be limited by chemical-specific response heterogeneity rather than "
            "insufficient temporal information."
        ),
        status=HypothesisStatus.COMPETING,
        supporting=supporting,
        evidence_gaps=(
            "The interpretation package does not separate chemical-specific heterogeneity from feature representation.",
            "No external validation is available.",
        ),
        falsifiability_statement=(
            "This alternative hypothesis would be weakened if concentration-prediction limitations persist after "
            "chemical-specific response heterogeneity is accounted for."
        ),
        rationale=(
            "The regression interpretation leaves more than one plausible explanation for limited concentration "
            "prediction performance."
        ),
        rule_id=RULE_CONCENTRATION_ENCODING,
        alternative_hypothesis_ids=("HYP-CONCENTRATION_ENCODING-0001",),
        tags=("concentration", "competing"),
        **context,
    )
    return (primary, alternative)


def _feature_representation(by_id: dict[str, Interpretation], **context) -> tuple[Hypothesis, ...]:
    required = ("INT-FEATURE_ENGINEERING-0001", "INT-FEATURE_SELECTION-0001")
    supporting = _available(by_id, required)
    if len(supporting) != len(required):
        return tuple()
    primary = _make_hypothesis(
        hypothesis_id="HYP-FEATURE_REPRESENTATION-0001",
        category=HypothesisCategory.FEATURE_REPRESENTATION,
        title="Window-based temporal feature representation",
        statement=(
            "Window-based temporal features may capture response information not fully represented by the "
            "reference feature configuration."
        ),
        status=HypothesisStatus.PLAUSIBLE,
        supporting=supporting,
        evidence_gaps=(
            "No direct causal test is available.",
            "No external validation is available.",
        ),
        falsifiability_statement=(
            "This hypothesis would be weakened if window-based temporal features do not reproducibly improve "
            "internal benchmarks after model capacity and feature count are accounted for."
        ),
        rationale=(
            "Feature-engineering and feature-selection interpretations support a feature-representation hypothesis "
            "without establishing a causal feature mechanism."
        ),
        rule_id=RULE_FEATURE_REPRESENTATION,
        alternative_hypothesis_ids=("HYP-FEATURE_REPRESENTATION-0002",),
        tags=("feature-representation",),
        **context,
    )
    competing = _make_hypothesis(
        hypothesis_id="HYP-FEATURE_REPRESENTATION-0002",
        category=HypothesisCategory.FEATURE_REPRESENTATION,
        title="Dimensionality or flexibility alternative",
        statement=(
            "The reported benchmark improvement may partly reflect increased feature dimensionality or model "
            "flexibility rather than uniquely informative temporal biology."
        ),
        status=HypothesisStatus.COMPETING,
        supporting=supporting,
        evidence_gaps=(
            "The interpretation package does not isolate dimensionality from temporal information content.",
            "No external validation is available.",
        ),
        falsifiability_statement=(
            "This competing hypothesis would be weakened if benchmark improvements remain reproducible after "
            "feature dimensionality and model flexibility are accounted for."
        ),
        rationale=(
            "The feature-engineering interpretation reports benchmark association but cannot distinguish feature "
            "information content from dimensionality or model-flexibility alternatives."
        ),
        rule_id=RULE_FEATURE_REPRESENTATION,
        alternative_hypothesis_ids=("HYP-FEATURE_REPRESENTATION-0001",),
        tags=("feature-representation", "competing"),
        **context,
    )
    return (primary, competing)


def _strain_contribution(by_id: dict[str, Interpretation], **context) -> tuple[Hypothesis, ...]:
    required = ("INT-STRAIN_CONTRIBUTION-0001", "INT-CHEMICAL_CLASSIFICATION-0001")
    supporting = _available(by_id, required)
    if len(supporting) != len(required):
        return tuple()
    primary = _make_hypothesis(
        hypothesis_id="HYP-STRAIN_CONTRIBUTION-0001",
        category=HypothesisCategory.STRAIN_CONTRIBUTION,
        title="Nonredundant strain contribution",
        statement=(
            "Individual strains may contribute nonredundant information to the multistrain classification "
            "fingerprint."
        ),
        status=HypothesisStatus.PLAUSIBLE,
        supporting=supporting,
        evidence_gaps=(
            "No specific strain is identified by the interpretation package as biologically important.",
            "No external validation is available.",
        ),
        falsifiability_statement=(
            "This hypothesis would be weakened if strain-removal or strain-subset interpretations do not "
            "reproducibly change classification-related evidence."
        ),
        rationale=(
            "Strain-contribution and classification interpretations support a testable nonredundancy hypothesis "
            "without assigning importance to a named strain."
        ),
        rule_id=RULE_STRAIN_CONTRIBUTION,
        alternative_hypothesis_ids=("HYP-STRAIN_CONTRIBUTION-0002",),
        tags=("strain-contribution",),
        **context,
    )
    alternative = _make_hypothesis(
        hypothesis_id="HYP-STRAIN_CONTRIBUTION-0002",
        category=HypothesisCategory.STRAIN_CONTRIBUTION,
        title="Sampling-variability strain alternative",
        statement=(
            "Observed strain contribution differences may reflect sampling variability or uneven "
            "chemical-response coverage."
        ),
        status=HypothesisStatus.COMPETING,
        supporting=supporting,
        evidence_gaps=(
            "The interpretation package does not distinguish nonredundant strain information from sampling variability.",
            "No external validation is available.",
        ),
        falsifiability_statement=(
            "This alternative hypothesis would be weakened if differential strain contribution remains "
            "reproducible under balanced chemical-response coverage."
        ),
        rationale=(
            "The strain-contribution interpretation supports evaluation of differential contribution while leaving "
            "sampling variability as a competing explanation."
        ),
        rule_id=RULE_STRAIN_CONTRIBUTION,
        alternative_hypothesis_ids=("HYP-STRAIN_CONTRIBUTION-0001",),
        tags=("strain-contribution", "competing"),
        **context,
    )
    return (primary, alternative)


def _data_quality_effect(by_id: dict[str, Interpretation], **context) -> tuple[Hypothesis, ...]:
    required = ("INT-DATA_QUALITY-0001", "INT-OVERALL_EVIDENCE-0001")
    supporting = _available(by_id, required)
    if len(supporting) != len(required):
        return tuple()
    return (
        _make_hypothesis(
            hypothesis_id="HYP-DATA_QUALITY_EFFECT-0001",
            category=HypothesisCategory.DATA_QUALITY_EFFECT,
            title="Data-quality contribution to uncertainty",
            statement=(
                "Active QC limitations may contribute to uncertainty in downstream classification and regression "
                "estimates."
            ),
            status=HypothesisStatus.PLAUSIBLE,
            supporting=supporting,
            evidence_gaps=(
                "The interpretation package does not establish that QC limitations caused any specific performance result.",
                "No external validation is available.",
            ),
            falsifiability_statement=(
                "This hypothesis would be weakened if downstream estimates remain reproducible in interpretation "
                "packages without active QC limitations."
            ),
            rationale=(
                "Data-quality and overall-evidence interpretations support a QC-uncertainty hypothesis without "
                "claiming that QC limitations caused a specific estimate."
            ),
            rule_id=RULE_DATA_QUALITY_EFFECT,
            tags=("data-quality",),
            **context,
        ),
    )


def _generalization(by_id: dict[str, Interpretation], **context) -> tuple[Hypothesis, ...]:
    required = (
        "INT-BLIND_VALIDATION-0001",
        "INT-CHEMICAL_CLASSIFICATION-0001",
        "INT-CONCENTRATION_REGRESSION-0001",
    )
    supporting = _available(by_id, required)
    if len(supporting) != len(required):
        return tuple()
    return (
        _make_hypothesis(
            hypothesis_id="HYP-GENERALIZATION-0001",
            category=HypothesisCategory.GENERALIZATION,
            title="Internal-to-external generalization boundary",
            statement=(
                "Performance observed during internal evaluation may not fully generalize to independently "
                "labelled unknown samples."
            ),
            status=HypothesisStatus.WEAKLY_SUPPORTED,
            supporting=supporting,
            evidence_gaps=(
                "True blind labels are absent.",
                "No external validation is available.",
            ),
            falsifiability_statement=(
                "This hypothesis would be weakened if independently labelled unknown samples show reproducible "
                "performance patterns consistent with the internal evaluation."
            ),
            rationale=(
                "Blind-validation, classification, and regression interpretations support a generalization-boundary "
                "hypothesis because the available blind-prediction interpretation does not establish external "
                "validation performance."
            ),
            rule_id=RULE_GENERALIZATION,
            tags=("generalization",),
            force_low_confidence=True,
            **context,
        ),
    )


def _overall_system_behavior(by_id: dict[str, Interpretation], **context) -> tuple[Hypothesis, ...]:
    required = ("INT-CHEMICAL_CLASSIFICATION-0001", "INT-CONCENTRATION_REGRESSION-0001")
    supporting = _available(by_id, tuple(sorted(set(required) | set(by_id))))
    if not all(item in by_id for item in required):
        return tuple()
    return (
        _make_hypothesis(
            hypothesis_id="HYP-OVERALL_SYSTEM_BEHAVIOR-0001",
            category=HypothesisCategory.OVERALL_SYSTEM_BEHAVIOR,
            title="Classification-versus-concentration information balance",
            statement=(
                "The multistrain biosensor array may be more informative for chemical identity discrimination "
                "than for precise concentration estimation under the current dataset and feature representation."
            ),
            status=HypothesisStatus.PLAUSIBLE,
            supporting=supporting,
            evidence_gaps=(
                "The evidence is based on internal evaluation.",
                "The evidence lacks real external validation.",
            ),
            falsifiability_statement=(
                "This hypothesis would be weakened if concentration-estimation interpretations become "
                "reproducibly comparable to or more informative than chemical-discrimination interpretations under "
                "the same evidence boundary."
            ),
            rationale=(
                "Classification, regression, blind-validation, and overall-evidence interpretations support a "
                "system-level hypothesis about relative information content under the current evidence boundary."
            ),
            rule_id=RULE_OVERALL_SYSTEM_BEHAVIOR,
            tags=("overall-system-behavior",),
            **context,
        ),
    )


def _make_hypothesis(
    *,
    hypothesis_id: str,
    category: HypothesisCategory,
    title: str,
    statement: str,
    status: HypothesisStatus,
    supporting: tuple[Interpretation, ...],
    evidence_gaps: tuple[str, ...],
    falsifiability_statement: str,
    rationale: str,
    rule_id: str,
    software_version: str,
    schema_version: str | None,
    created_at: str,
    tags: tuple[str, ...],
    alternative_hypothesis_ids: tuple[str, ...] = tuple(),
    force_low_confidence: bool = False,
    metadata: dict[str, Any] | None = None,
) -> Hypothesis:
    supporting = tuple(sorted(supporting, key=lambda interpretation: interpretation.interpretation_id))
    confidence = assign_confidence(
        supporting,
        evidence_gap_count=len(evidence_gaps),
        external_validation_gap=any("external validation" in gap.lower() for gap in evidence_gaps),
    )
    if force_low_confidence:
        confidence = HypothesisConfidence.LOW
    score = priority_score(
        category,
        supporting,
        confidence=confidence,
        evidence_gap_count=len(evidence_gaps),
    )
    return Hypothesis(
        hypothesis_id=hypothesis_id,
        category=category,
        title=title,
        statement=statement,
        status=status,
        confidence=confidence,
        supporting_interpretation_ids=tuple(interpretation.interpretation_id for interpretation in supporting),
        contradicting_interpretation_ids=tuple(),
        supporting_observation_ids=_supporting_observation_ids(supporting),
        assumptions=(
            "Validated Interpretation Engine outputs are the authoritative source for this hypothesis.",
            "The hypothesis is not presented as an established fact.",
        ),
        alternative_hypothesis_ids=tuple(sorted(alternative_hypothesis_ids)),
        evidence_gaps=evidence_gaps,
        falsifiability_statement=falsifiability_statement,
        rationale=rationale,
        reasoning_rule_ids=(rule_id,),
        priority_score=score,
        priority=priority_from_score(score),
        created_at=created_at,
        software_version=software_version,
        source_interpretation_schema_version=schema_version,
        tags=tags,
        metadata=metadata or {},
    )


def _available(by_id: dict[str, Interpretation], interpretation_ids: tuple[str, ...]) -> tuple[Interpretation, ...]:
    return tuple(by_id[item] for item in interpretation_ids if item in by_id)


def _supporting_observation_ids(interpretations: Iterable[Interpretation]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                observation_id
                for interpretation in interpretations
                for observation_id in interpretation.supporting_observation_ids
            }
        )
    )
