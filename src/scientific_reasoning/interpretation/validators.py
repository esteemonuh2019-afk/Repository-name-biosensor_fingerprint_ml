"""Validation contracts for BSIP scientific interpretations."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from datetime import datetime

from src.scientific_reasoning.observation import Observation

from .enums import (
    EvidenceDirection,
    InterpretationCategory,
    InterpretationConfidence,
    InterpretationStatus,
    ReasoningSeverity,
    category_id_token,
)
from .models import Interpretation, InterpretationValidationIssue
from .policies import (
    assign_confidence,
    find_blind_validation_overclaim_terms,
    find_forbidden_causal_terms,
    find_hypothesis_terms,
    find_literature_comparison_terms,
    find_recommendation_terms,
    supports_confidence_assignment,
)


INTERPRETATION_ID_PATTERN = re.compile(r"^INT-([A-Z_]+)-([0-9]{4})$")


def validate_interpretation(
    interpretation: Interpretation,
    observations: Iterable[Observation] | None = None,
) -> tuple[InterpretationValidationIssue, ...]:
    issues: list[InterpretationValidationIssue] = []
    issues.extend(validate_required_fields(interpretation))
    issues.extend(validate_interpretation_id_format(interpretation))
    issues.extend(validate_evidence_presence(interpretation))
    issues.extend(validate_evidence_links_are_recorded(interpretation))
    issues.extend(validate_claim_language(interpretation))
    issues.extend(validate_serializability(interpretation))
    issues.extend(validate_interpretation_deterministic_fields(interpretation))
    if observations is not None:
        known = {observation.observation_id: observation for observation in observations}
        issues.extend(validate_observation_dependencies(interpretation, known))
        issues.extend(validate_confidence_assignment(interpretation, known))
    return tuple(issues)


def validate_interpretations(
    interpretations: Iterable[Interpretation],
    observations: Iterable[Observation] | None = None,
) -> tuple[InterpretationValidationIssue, ...]:
    ordered = tuple(interpretations)
    observations_tuple = None if observations is None else tuple(observations)
    issues: list[InterpretationValidationIssue] = []
    for interpretation in ordered:
        issues.extend(validate_interpretation(interpretation, observations_tuple))
    issues.extend(validate_unique_ids(ordered))
    issues.extend(validate_deterministic_ordering(ordered))
    return tuple(issues)


def validate_required_fields(interpretation: Interpretation) -> tuple[InterpretationValidationIssue, ...]:
    required_fields = (
        "interpretation_id",
        "category",
        "title",
        "claim",
        "status",
        "confidence",
        "created_at",
        "software_version",
        "source_observation_schema_version",
    )
    issues = []
    for field_name in required_fields:
        value = getattr(interpretation, field_name)
        if value is None or value == "":
            issues.append(
                _issue(
                    "REQUIRED_FIELD_MISSING",
                    f"Required field is missing: {field_name}",
                    interpretation,
                    field_name,
                )
            )
    return tuple(issues)


def validate_interpretation_id_format(
    interpretation: Interpretation,
) -> tuple[InterpretationValidationIssue, ...]:
    match = INTERPRETATION_ID_PATTERN.match(interpretation.interpretation_id)
    if not match:
        return (
            _issue(
                "INVALID_INTERPRETATION_ID",
                "Interpretation ID must match INT-{CATEGORY}-{NUMBER} with a four-digit number.",
                interpretation,
                "interpretation_id",
            ),
        )
    category_token, _number = match.groups()
    expected = category_id_token(interpretation.category)
    if category_token != expected:
        return (
            _issue(
                "INTERPRETATION_ID_CATEGORY_MISMATCH",
                f"Interpretation ID category token {category_token} does not match {expected}.",
                interpretation,
                "interpretation_id",
            ),
        )
    return tuple()


def validate_unique_ids(interpretations: Iterable[Interpretation]) -> tuple[InterpretationValidationIssue, ...]:
    seen: set[str] = set()
    issues = []
    for interpretation in interpretations:
        if interpretation.interpretation_id in seen:
            issues.append(
                _issue(
                    "DUPLICATE_INTERPRETATION_ID",
                    f"Duplicate interpretation ID: {interpretation.interpretation_id}",
                    interpretation,
                    "interpretation_id",
                )
            )
        seen.add(interpretation.interpretation_id)
    return tuple(issues)


def validate_evidence_presence(interpretation: Interpretation) -> tuple[InterpretationValidationIssue, ...]:
    has_supporting_link = any(link.direction == EvidenceDirection.SUPPORTING for link in interpretation.evidence_summary)
    if interpretation.supporting_observation_ids or has_supporting_link:
        return tuple()
    return (
        _issue(
            "INTERPRETATION_WITHOUT_EVIDENCE",
            "Interpretation must depend on at least one supporting Observation ID.",
            interpretation,
            "supporting_observation_ids",
        ),
        _issue(
            "MISSING_SUPPORTING_OBSERVATION",
            "No supporting observation dependency is recorded.",
            interpretation,
            "supporting_observation_ids",
        ),
    )


def validate_observation_dependencies(
    interpretation: Interpretation,
    observations_by_id: dict[str, Observation],
) -> tuple[InterpretationValidationIssue, ...]:
    referenced_ids = set(interpretation.supporting_observation_ids)
    referenced_ids.update(interpretation.contradicting_observation_ids)
    referenced_ids.update(link.observation_id for link in interpretation.evidence_summary if link.observation_id)
    issues = []
    for observation_id in sorted(referenced_ids):
        if observation_id not in observations_by_id:
            issues.append(
                InterpretationValidationIssue(
                    code="NONEXISTENT_OBSERVATION_DEPENDENCY",
                    severity=ReasoningSeverity.CRITICAL,
                    message=f"Referenced observation does not exist: {observation_id}",
                    interpretation_id=interpretation.interpretation_id,
                    field="supporting_observation_ids",
                    observation_id=observation_id,
                    rule_id=None,
                )
            )
    return tuple(issues)


def validate_evidence_links_are_recorded(
    interpretation: Interpretation,
) -> tuple[InterpretationValidationIssue, ...]:
    supporting_ids = set(interpretation.supporting_observation_ids)
    contradicting_ids = set(interpretation.contradicting_observation_ids)
    issues = []
    for link in interpretation.evidence_summary:
        if link.direction == EvidenceDirection.SUPPORTING and link.observation_id not in supporting_ids:
            issues.append(
                InterpretationValidationIssue(
                    code="MISSING_SUPPORTING_OBSERVATION",
                    severity=ReasoningSeverity.CRITICAL,
                    message=f"Supporting evidence link is not listed as a supporting observation: {link.observation_id}",
                    interpretation_id=interpretation.interpretation_id,
                    field="supporting_observation_ids",
                    observation_id=link.observation_id,
                    rule_id=None,
                )
            )
        if link.direction == EvidenceDirection.CONTRADICTING and link.observation_id not in contradicting_ids:
            issues.append(
                InterpretationValidationIssue(
                    code="CONTRADICTION_NOT_RECORDED",
                    severity=ReasoningSeverity.CRITICAL,
                    message=f"Contradicting evidence link is not listed as a contradiction: {link.observation_id}",
                    interpretation_id=interpretation.interpretation_id,
                    field="contradicting_observation_ids",
                    observation_id=link.observation_id,
                    rule_id=None,
                )
            )
    if interpretation.status == InterpretationStatus.CONFLICTED and not interpretation.contradicting_observation_ids:
        issues.append(
            _issue(
                "CONTRADICTION_NOT_RECORDED",
                "CONFLICTED interpretations must record contradicting observation IDs.",
                interpretation,
                "contradicting_observation_ids",
            )
        )
    return tuple(issues)


def validate_confidence_assignment(
    interpretation: Interpretation,
    observations_by_id: dict[str, Observation],
) -> tuple[InterpretationValidationIssue, ...]:
    if any(
        observation_id not in observations_by_id
        for observation_id in interpretation.supporting_observation_ids + interpretation.contradicting_observation_ids
    ):
        return tuple()
    supporting = tuple(observations_by_id[observation_id] for observation_id in interpretation.supporting_observation_ids)
    contradicting = tuple(
        observations_by_id[observation_id] for observation_id in interpretation.contradicting_observation_ids
    )
    expected = assign_confidence(supporting, contradicting)
    if supports_confidence_assignment(interpretation.confidence, expected):
        return tuple()
    return (
        _issue(
            "UNSUPPORTED_CONFIDENCE_ASSIGNMENT",
            f"Assigned confidence {interpretation.confidence.value} exceeds policy-supported {expected.value}.",
            interpretation,
            "confidence",
        ),
    )


def validate_claim_language(interpretation: Interpretation) -> tuple[InterpretationValidationIssue, ...]:
    claim = interpretation.claim
    checks = (
        ("FORBIDDEN_CAUSAL_LANGUAGE", "claim", find_forbidden_causal_terms(claim)),
        ("RECOMMENDATION_LANGUAGE", "claim", find_recommendation_terms(claim)),
        ("HYPOTHESIS_LANGUAGE", "claim", find_hypothesis_terms(claim)),
        ("LITERATURE_COMPARISON_LANGUAGE", "claim", find_literature_comparison_terms(claim)),
    )
    issues = []
    for code, field_name, terms in checks:
        for term in terms:
            issues.append(
                _issue(
                    code,
                    f"Claim contains restricted wording: {term}",
                    interpretation,
                    field_name,
                )
            )
    blind_terms = find_blind_validation_overclaim_terms(claim)
    if interpretation.category == InterpretationCategory.BLIND_VALIDATION:
        for term in blind_terms:
            issues.append(
                _issue(
                    "BLIND_VALIDATION_OVERCLAIM",
                    f"Blind-validation interpretation overclaims unavailable validation evidence: {term}",
                    interpretation,
                    "claim",
                )
            )
    return tuple(issues)


def validate_serializability(interpretation: Interpretation) -> tuple[InterpretationValidationIssue, ...]:
    try:
        json.dumps(interpretation.to_record(), sort_keys=True)
    except (TypeError, ValueError) as exc:
        return (
            _issue(
                "NON_SERIALIZABLE_METADATA",
                f"Interpretation is not JSON serializable: {exc}",
                interpretation,
                "metadata",
            ),
        )
    try:
        datetime.fromisoformat(interpretation.created_at)
    except ValueError:
        return (
            _issue(
                "INVALID_TIMESTAMP",
                "created_at must be ISO 8601 parseable.",
                interpretation,
                "created_at",
            ),
        )
    return tuple()


def validate_interpretation_deterministic_fields(
    interpretation: Interpretation,
) -> tuple[InterpretationValidationIssue, ...]:
    issues = []
    if tuple(sorted(interpretation.supporting_observation_ids)) != interpretation.supporting_observation_ids:
        issues.append(
            _issue(
                "NON_DETERMINISTIC_ORDER",
                "supporting_observation_ids must be sorted deterministically.",
                interpretation,
                "supporting_observation_ids",
            )
        )
    if tuple(sorted(interpretation.contradicting_observation_ids)) != interpretation.contradicting_observation_ids:
        issues.append(
            _issue(
                "NON_DETERMINISTIC_ORDER",
                "contradicting_observation_ids must be sorted deterministically.",
                interpretation,
                "contradicting_observation_ids",
            )
        )
    return tuple(issues)


def validate_deterministic_ordering(
    interpretations: Iterable[Interpretation],
) -> tuple[InterpretationValidationIssue, ...]:
    ordered = tuple(interpretations)
    sorted_ids = sorted(interpretation.interpretation_id for interpretation in ordered)
    actual_ids = [interpretation.interpretation_id for interpretation in ordered]
    if actual_ids != sorted_ids:
        return (
            InterpretationValidationIssue(
                code="NON_DETERMINISTIC_ORDER",
                severity=ReasoningSeverity.WARNING,
                message="Interpretations are not ordered deterministically by interpretation_id.",
                interpretation_id=None,
                field="interpretations",
                observation_id=None,
                rule_id=None,
            ),
        )
    return tuple()


def _issue(
    code: str,
    message: str,
    interpretation: Interpretation,
    field: str,
) -> InterpretationValidationIssue:
    return InterpretationValidationIssue(
        code=code,
        severity=ReasoningSeverity.CRITICAL,
        message=message,
        interpretation_id=interpretation.interpretation_id,
        field=field,
        observation_id=None,
        rule_id=None,
    )
