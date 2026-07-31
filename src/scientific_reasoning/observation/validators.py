"""Validation contracts for factual observations."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Iterable

from .enums import (
    ConfidenceLevel,
    ObservationCategory,
    ObservationStatus,
    category_id_token,
)
from .models import Observation, ValidationIssue


OBSERVATION_ID_PATTERN = re.compile(r"^OBS-([A-Z_]+)-([0-9]{4})$")
RISKY_BLIND_WORDING = (
    "real blind validation",
    "blind validation performance",
    "validation accuracy",
    "correctly predicted",
    "confirmed",
    "true positive",
    "publication ready",
)


def validate_observation(observation: Observation) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []
    issues.extend(validate_required_fields(observation))
    issues.extend(validate_observation_id_format(observation))
    issues.extend(validate_allowed_enum_values(observation))
    issues.extend(validate_quantitative_provenance(observation))
    issues.extend(validate_model_metric_coherence(observation))
    issues.extend(validate_blind_validation_wording(observation))
    issues.extend(validate_serializability(observation))
    return tuple(issues)


def validate_observations(observations: Iterable[Observation]) -> tuple[ValidationIssue, ...]:
    ordered = tuple(observations)
    issues: list[ValidationIssue] = []
    for observation in ordered:
        issues.extend(validate_observation(observation))
    issues.extend(validate_unique_ids(ordered))
    issues.extend(validate_deterministic_ordering(ordered))
    return tuple(issues)


def validate_required_fields(observation: Observation) -> tuple[ValidationIssue, ...]:
    required_fields = (
        "observation_id",
        "category",
        "title",
        "statement",
        "status",
        "analysis_stage",
        "confidence",
        "created_at",
        "software_version",
    )
    issues = []
    for field_name in required_fields:
        value = getattr(observation, field_name)
        if value is None or value == "":
            issues.append(
                _issue(
                    "REQUIRED_FIELD_MISSING",
                    f"Required field is missing: {field_name}",
                    observation,
                    field_name,
                )
            )
    return tuple(issues)


def validate_observation_id_format(observation: Observation) -> tuple[ValidationIssue, ...]:
    match = OBSERVATION_ID_PATTERN.match(observation.observation_id)
    if not match:
        return (
            _issue(
                "INVALID_OBSERVATION_ID",
                "Observation ID must match OBS-{CATEGORY}-{NUMBER} with a four-digit number.",
                observation,
                "observation_id",
            ),
        )
    category_token, _number = match.groups()
    expected = category_id_token(observation.category)
    if category_token != expected:
        return (
            _issue(
                "OBSERVATION_ID_CATEGORY_MISMATCH",
                f"Observation ID category token {category_token} does not match {expected}.",
                observation,
                "observation_id",
            ),
        )
    return tuple()


def validate_allowed_enum_values(observation: Observation) -> tuple[ValidationIssue, ...]:
    issues = []
    if observation.category not in ObservationCategory:
        issues.append(_issue("INVALID_CATEGORY", "Category is not part of the stable enum.", observation, "category"))
    if observation.status not in ObservationStatus:
        issues.append(_issue("INVALID_STATUS", "Status is not part of the stable enum.", observation, "status"))
    if observation.confidence not in ConfidenceLevel:
        issues.append(
            _issue(
                "INVALID_CONFIDENCE",
                "Confidence is not part of the stable enum.",
                observation,
                "confidence",
            )
        )
    return tuple(issues)


def validate_unique_ids(observations: Iterable[Observation]) -> tuple[ValidationIssue, ...]:
    seen: set[str] = set()
    issues = []
    for observation in observations:
        if observation.observation_id in seen:
            issues.append(
                _issue(
                    "DUPLICATE_OBSERVATION_ID",
                    f"Duplicate observation ID: {observation.observation_id}",
                    observation,
                    "observation_id",
                )
            )
        seen.add(observation.observation_id)
    return tuple(issues)


def validate_quantitative_provenance(observation: Observation) -> tuple[ValidationIssue, ...]:
    provenance_ids = {record.provenance_id for record in observation.provenance_records}
    issues = []
    for metric in observation.supporting_metrics:
        if metric.metric_value is None:
            continue
        if not metric.provenance_id or metric.provenance_id not in provenance_ids:
            issues.append(
                ValidationIssue(
                    code="MISSING_PROVENANCE",
                    severity="ERROR",
                    message=f"Metric {metric.metric_name} has a value but no matching provenance record.",
                    observation_id=observation.observation_id,
                    field="supporting_metrics",
                    source_file=metric.source_file,
                )
            )
    return tuple(issues)


def validate_model_metric_coherence(observation: Observation) -> tuple[ValidationIssue, ...]:
    provenance_by_id = {record.provenance_id: record for record in observation.provenance_records}
    issues = []
    for metric in observation.supporting_metrics:
        if not metric.model_name or not metric.provenance_id:
            continue
        provenance = provenance_by_id.get(metric.provenance_id)
        if provenance and provenance.model_name and provenance.model_name != metric.model_name:
            issues.append(
                ValidationIssue(
                    code="MODEL_METRIC_MISMATCH",
                    severity="ERROR",
                    message=(
                        f"Metric model {metric.model_name} does not match provenance model "
                        f"{provenance.model_name}."
                    ),
                    observation_id=observation.observation_id,
                    field="supporting_metrics.model_name",
                    source_file=metric.source_file or provenance.source_file,
                )
            )
    return tuple(issues)


def validate_blind_validation_wording(observation: Observation) -> tuple[ValidationIssue, ...]:
    if observation.category != ObservationCategory.BLIND_PREDICTION:
        return tuple()
    lowered = observation.statement.lower()
    issues = []
    for phrase in RISKY_BLIND_WORDING:
        if phrase in lowered:
            issues.append(
                _issue(
                    "BLIND_VALIDATION_WORDING",
                    f"Blind-prediction observation uses interpretation or validation wording: {phrase}.",
                    observation,
                    "statement",
                )
            )
    return tuple(issues)


def validate_serializability(observation: Observation) -> tuple[ValidationIssue, ...]:
    try:
        json.dumps(observation.to_record(), sort_keys=True)
    except (TypeError, ValueError) as exc:
        return (
            _issue(
                "NOT_JSON_SERIALIZABLE",
                f"Observation is not JSON serializable: {exc}",
                observation,
                "metadata",
            ),
        )
    try:
        datetime.fromisoformat(observation.created_at)
    except ValueError:
        return (
            _issue(
                "INVALID_TIMESTAMP",
                "created_at must be ISO 8601 parseable.",
                observation,
                "created_at",
            ),
        )
    return tuple()


def validate_deterministic_ordering(observations: Iterable[Observation]) -> tuple[ValidationIssue, ...]:
    ordered = tuple(observations)
    sorted_ids = sorted(observation.observation_id for observation in ordered)
    actual_ids = [observation.observation_id for observation in ordered]
    if actual_ids != sorted_ids:
        return (
            ValidationIssue(
                code="NON_DETERMINISTIC_ORDER",
                severity="WARNING",
                message="Observations are not ordered deterministically by observation_id.",
                observation_id=None,
                field="observations",
                source_file=None,
            ),
        )
    return tuple()


def _issue(code: str, message: str, observation: Observation, field: str) -> ValidationIssue:
    return ValidationIssue(
        code=code,
        severity="ERROR",
        message=message,
        observation_id=observation.observation_id,
        field=field,
        source_file=None,
    )
