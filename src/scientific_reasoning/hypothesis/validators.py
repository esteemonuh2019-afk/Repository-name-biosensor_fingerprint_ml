"""Validation contracts for BSIP scientific hypotheses."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from datetime import datetime

from src.scientific_reasoning.interpretation import Interpretation

from .enums import (
    HypothesisCategory,
    HypothesisConfidence,
    HypothesisPriority,
    HypothesisSeverity,
    HypothesisStatus,
    category_id_token,
)
from .models import Hypothesis, HypothesisValidationIssue
from .policies import (
    assign_confidence,
    falsifiability_is_valid,
    find_forbidden_hypothesis_terms,
    find_literature_comparison_terms,
    find_protocol_terms,
    find_recommendation_terms,
    has_allowed_hypothesis_modality,
    priority_from_score,
    supports_confidence_assignment,
)


HYPOTHESIS_ID_PATTERN = re.compile(r"^HYP-([A-Z_]+)-([0-9]{4})$")


def validate_hypothesis(
    hypothesis: Hypothesis,
    interpretations: Iterable[Interpretation] | None = None,
    hypotheses: Iterable[Hypothesis] | None = None,
) -> tuple[HypothesisValidationIssue, ...]:
    issues: list[HypothesisValidationIssue] = []
    issues.extend(validate_required_fields(hypothesis))
    issues.extend(validate_hypothesis_id_format(hypothesis))
    issues.extend(validate_dependency_presence(hypothesis))
    issues.extend(validate_statement_language(hypothesis))
    issues.extend(validate_falsifiability(hypothesis))
    issues.extend(validate_priority_policy(hypothesis))
    issues.extend(validate_serializability(hypothesis))
    issues.extend(validate_hypothesis_deterministic_fields(hypothesis))
    if interpretations is not None:
        known = {interpretation.interpretation_id: interpretation for interpretation in interpretations}
        issues.extend(validate_interpretation_dependencies(hypothesis, known))
        issues.extend(validate_confidence_assignment(hypothesis, known))
    if hypotheses is not None:
        known_hypotheses = {item.hypothesis_id: item for item in hypotheses}
        issues.extend(validate_competing_hypothesis_links(hypothesis, known_hypotheses))
    return tuple(issues)


def validate_hypotheses(
    hypotheses: Iterable[Hypothesis],
    interpretations: Iterable[Interpretation] | None = None,
) -> tuple[HypothesisValidationIssue, ...]:
    ordered = tuple(hypotheses)
    interpretations_tuple = None if interpretations is None else tuple(interpretations)
    issues: list[HypothesisValidationIssue] = []
    for hypothesis in ordered:
        issues.extend(validate_hypothesis(hypothesis, interpretations_tuple, ordered))
    issues.extend(validate_unique_ids(ordered))
    issues.extend(validate_deterministic_ordering(ordered))
    return tuple(issues)


def validate_required_fields(hypothesis: Hypothesis) -> tuple[HypothesisValidationIssue, ...]:
    required_fields = (
        "hypothesis_id",
        "category",
        "title",
        "statement",
        "status",
        "confidence",
        "rationale",
        "created_at",
        "software_version",
        "source_interpretation_schema_version",
    )
    issues = []
    for field_name in required_fields:
        value = getattr(hypothesis, field_name)
        if value is None or value == "":
            issues.append(
                _issue(
                    "REQUIRED_FIELD_MISSING",
                    f"Required field is missing: {field_name}",
                    hypothesis,
                    field_name,
                )
            )
    return tuple(issues)


def validate_hypothesis_id_format(hypothesis: Hypothesis) -> tuple[HypothesisValidationIssue, ...]:
    match = HYPOTHESIS_ID_PATTERN.match(hypothesis.hypothesis_id)
    if not match:
        return (
            _issue(
                "INVALID_HYPOTHESIS_ID",
                "Hypothesis ID must match HYP-{CATEGORY}-{NUMBER} with a four-digit number.",
                hypothesis,
                "hypothesis_id",
            ),
        )
    category_token, _number = match.groups()
    expected = category_id_token(hypothesis.category)
    if category_token != expected:
        return (
            _issue(
                "HYPOTHESIS_ID_CATEGORY_MISMATCH",
                f"Hypothesis ID category token {category_token} does not match {expected}.",
                hypothesis,
                "hypothesis_id",
            ),
        )
    return tuple()


def validate_unique_ids(hypotheses: Iterable[Hypothesis]) -> tuple[HypothesisValidationIssue, ...]:
    seen: set[str] = set()
    issues = []
    for hypothesis in hypotheses:
        if hypothesis.hypothesis_id in seen:
            issues.append(
                _issue(
                    "DUPLICATE_HYPOTHESIS_ID",
                    f"Duplicate hypothesis ID: {hypothesis.hypothesis_id}",
                    hypothesis,
                    "hypothesis_id",
                )
            )
        seen.add(hypothesis.hypothesis_id)
    return tuple(issues)


def validate_dependency_presence(hypothesis: Hypothesis) -> tuple[HypothesisValidationIssue, ...]:
    if hypothesis.supporting_interpretation_ids:
        return tuple()
    return (
        _issue(
            "UNSUPPORTED_HYPOTHESIS",
            "Hypothesis must cite at least one supporting interpretation.",
            hypothesis,
            "supporting_interpretation_ids",
        ),
    )


def validate_interpretation_dependencies(
    hypothesis: Hypothesis,
    interpretations_by_id: dict[str, Interpretation],
) -> tuple[HypothesisValidationIssue, ...]:
    referenced = set(hypothesis.supporting_interpretation_ids)
    referenced.update(hypothesis.contradicting_interpretation_ids)
    issues = []
    for interpretation_id in sorted(referenced):
        if interpretation_id not in interpretations_by_id:
            issues.append(
                HypothesisValidationIssue(
                    code="MISSING_INTERPRETATION_DEPENDENCY",
                    severity=HypothesisSeverity.CRITICAL,
                    message=f"Referenced interpretation does not exist: {interpretation_id}",
                    hypothesis_id=hypothesis.hypothesis_id,
                    field="supporting_interpretation_ids",
                    interpretation_id=interpretation_id,
                    rule_id=None,
                )
            )
    return tuple(issues)


def validate_statement_language(hypothesis: Hypothesis) -> tuple[HypothesisValidationIssue, ...]:
    issues = []
    for term in find_forbidden_hypothesis_terms(hypothesis.statement):
        issues.append(_issue("CAUSAL_OVERCLAIM", f"Hypothesis statement contains forbidden term: {term}", hypothesis, "statement"))
    for term in find_recommendation_terms(hypothesis.statement):
        issues.append(
            _issue(
                "RECOMMENDATION_LANGUAGE",
                f"Hypothesis statement contains recommendation language: {term}",
                hypothesis,
                "statement",
            )
        )
    for term in find_literature_comparison_terms(hypothesis.statement):
        issues.append(
            _issue(
                "LITERATURE_COMPARISON_LANGUAGE",
                f"Hypothesis statement contains literature-comparison language: {term}",
                hypothesis,
                "statement",
            )
        )
    if hypothesis.status in (
        HypothesisStatus.PLAUSIBLE,
        HypothesisStatus.COMPETING,
        HypothesisStatus.WEAKLY_SUPPORTED,
    ) and not has_allowed_hypothesis_modality(hypothesis.statement):
        issues.append(
            _issue(
                "UNSUPPORTED_HYPOTHESIS",
                "Active hypothesis statements must use explicit uncertain hypothesis language.",
                hypothesis,
                "statement",
            )
        )
    return tuple(issues)


def validate_falsifiability(hypothesis: Hypothesis) -> tuple[HypothesisValidationIssue, ...]:
    if hypothesis.status not in (
        HypothesisStatus.PLAUSIBLE,
        HypothesisStatus.COMPETING,
        HypothesisStatus.WEAKLY_SUPPORTED,
    ):
        return tuple()
    if not falsifiability_is_valid(hypothesis.falsifiability_statement):
        return (
            _issue(
                "MISSING_FALSIFIABILITY",
                "Active hypotheses must include a falsifiability statement that describes weakening evidence.",
                hypothesis,
                "falsifiability_statement",
            ),
        )
    issues = []
    for term in find_recommendation_terms(hypothesis.falsifiability_statement or "") + find_protocol_terms(
        hypothesis.falsifiability_statement or ""
    ):
        issues.append(
            _issue(
                "RECOMMENDATION_LANGUAGE",
                f"Falsifiability statement contains protocol or recommendation wording: {term}",
                hypothesis,
                "falsifiability_statement",
            )
        )
    return tuple(issues)


def validate_confidence_assignment(
    hypothesis: Hypothesis,
    interpretations_by_id: dict[str, Interpretation],
) -> tuple[HypothesisValidationIssue, ...]:
    if any(
        interpretation_id not in interpretations_by_id
        for interpretation_id in hypothesis.supporting_interpretation_ids + hypothesis.contradicting_interpretation_ids
    ):
        return tuple()
    supporting = tuple(interpretations_by_id[item] for item in hypothesis.supporting_interpretation_ids)
    contradicting = tuple(interpretations_by_id[item] for item in hypothesis.contradicting_interpretation_ids)
    expected = assign_confidence(
        supporting,
        contradicting,
        evidence_gap_count=len(hypothesis.evidence_gaps),
        external_validation_gap=any("external validation" in gap.lower() for gap in hypothesis.evidence_gaps),
    )
    if supports_confidence_assignment(hypothesis.confidence, expected):
        return tuple()
    return (
        _issue(
            "CONFIDENCE_POLICY_ISSUE",
            f"Assigned confidence {hypothesis.confidence.value} exceeds policy-supported {expected.value}.",
            hypothesis,
            "confidence",
        ),
    )


def validate_priority_policy(hypothesis: Hypothesis) -> tuple[HypothesisValidationIssue, ...]:
    if not (0 <= hypothesis.priority_score <= 100):
        return (
            _issue(
                "PRIORITY_SCORE_OUT_OF_RANGE",
                "priority_score must be between 0 and 100.",
                hypothesis,
                "priority_score",
            ),
        )
    expected = priority_from_score(hypothesis.priority_score)
    if hypothesis.priority != expected:
        return (
            _issue(
                "PRIORITY_POLICY_ISSUE",
                f"Priority {hypothesis.priority.value} does not match score-derived priority {expected.value}.",
                hypothesis,
                "priority",
            ),
        )
    return tuple()


def validate_competing_hypothesis_links(
    hypothesis: Hypothesis,
    hypotheses_by_id: dict[str, Hypothesis],
) -> tuple[HypothesisValidationIssue, ...]:
    issues = []
    for alternative_id in hypothesis.alternative_hypothesis_ids:
        alternative = hypotheses_by_id.get(alternative_id)
        if alternative is None:
            issues.append(
                HypothesisValidationIssue(
                    code="COMPETING_HYPOTHESIS_LINK_ISSUE",
                    severity=HypothesisSeverity.CRITICAL,
                    message=f"Alternative hypothesis does not exist: {alternative_id}",
                    hypothesis_id=hypothesis.hypothesis_id,
                    field="alternative_hypothesis_ids",
                    interpretation_id=None,
                    rule_id=None,
                )
            )
        elif hypothesis.hypothesis_id not in alternative.alternative_hypothesis_ids:
            issues.append(
                HypothesisValidationIssue(
                    code="COMPETING_HYPOTHESIS_LINK_ISSUE",
                    severity=HypothesisSeverity.WARNING,
                    message=f"Alternative hypothesis link is not reciprocal: {alternative_id}",
                    hypothesis_id=hypothesis.hypothesis_id,
                    field="alternative_hypothesis_ids",
                    interpretation_id=None,
                    rule_id=None,
                )
            )
    if hypothesis.status == HypothesisStatus.COMPETING and not hypothesis.alternative_hypothesis_ids:
        issues.append(
            _issue(
                "COMPETING_HYPOTHESIS_LINK_ISSUE",
                "COMPETING hypotheses must link to at least one alternative hypothesis.",
                hypothesis,
                "alternative_hypothesis_ids",
            )
        )
    return tuple(issues)


def validate_serializability(hypothesis: Hypothesis) -> tuple[HypothesisValidationIssue, ...]:
    try:
        json.dumps(hypothesis.to_record(), sort_keys=True)
    except (TypeError, ValueError) as exc:
        return (
            _issue(
                "NON_SERIALIZABLE_METADATA",
                f"Hypothesis is not JSON serializable: {exc}",
                hypothesis,
                "metadata",
            ),
        )
    try:
        datetime.fromisoformat(hypothesis.created_at)
    except ValueError:
        return (
            _issue(
                "INVALID_TIMESTAMP",
                "created_at must be ISO 8601 parseable.",
                hypothesis,
                "created_at",
            ),
        )
    return tuple()


def validate_hypothesis_deterministic_fields(hypothesis: Hypothesis) -> tuple[HypothesisValidationIssue, ...]:
    checks = (
        ("supporting_interpretation_ids", hypothesis.supporting_interpretation_ids),
        ("contradicting_interpretation_ids", hypothesis.contradicting_interpretation_ids),
        ("supporting_observation_ids", hypothesis.supporting_observation_ids),
        ("alternative_hypothesis_ids", hypothesis.alternative_hypothesis_ids),
    )
    issues = []
    for field_name, values in checks:
        if tuple(sorted(values)) != values:
            issues.append(
                _issue(
                    "DETERMINISTIC_ORDERING_ISSUE",
                    f"{field_name} must be sorted deterministically.",
                    hypothesis,
                    field_name,
                )
            )
    return tuple(issues)


def validate_deterministic_ordering(hypotheses: Iterable[Hypothesis]) -> tuple[HypothesisValidationIssue, ...]:
    ordered = tuple(hypotheses)
    sorted_ids = sorted(hypothesis.hypothesis_id for hypothesis in ordered)
    actual_ids = [hypothesis.hypothesis_id for hypothesis in ordered]
    if actual_ids != sorted_ids:
        return (
            HypothesisValidationIssue(
                code="DETERMINISTIC_ORDERING_ISSUE",
                severity=HypothesisSeverity.WARNING,
                message="Hypotheses are not ordered deterministically by hypothesis_id.",
                hypothesis_id=None,
                field="hypotheses",
                interpretation_id=None,
                rule_id=None,
            ),
        )
    return tuple()


def _issue(code: str, message: str, hypothesis: Hypothesis, field: str) -> HypothesisValidationIssue:
    return HypothesisValidationIssue(
        code=code,
        severity=HypothesisSeverity.CRITICAL,
        message=message,
        hypothesis_id=hypothesis.hypothesis_id,
        field=field,
        interpretation_id=None,
        rule_id=None,
    )
