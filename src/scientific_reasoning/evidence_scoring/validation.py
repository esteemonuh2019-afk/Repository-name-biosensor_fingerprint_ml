"""Validation for BSIP evidence scoring sources and outputs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from .enums import EvidenceDimension, EvidenceScoringIssueSeverity, PublicationReadiness, UncertaintyLevel
from .models import EvidenceScoreRecord, EvidenceScoringValidationIssue
from .publication_policy import PUBLICATION_CEILINGS, READINESS_RANK
from .rules import (
    DIMENSION_WEIGHTS,
    SUPPORTED_CLAIM_SCHEMA_VERSIONS,
    SUPPORTED_GRAPH_SCHEMA_VERSIONS,
    WEIGHTED_DIMENSIONS,
    evidence_level_from_score,
)


def validate_source_documents(
    *,
    claims_document: dict[str, Any],
    claim_validation_document: dict[str, Any],
    graph_document: dict[str, Any],
    graph_validation_document: dict[str, Any],
) -> tuple[EvidenceScoringValidationIssue, ...]:
    issues: list[EvidenceScoringValidationIssue] = []
    if claims_document.get("schema_version") not in SUPPORTED_CLAIM_SCHEMA_VERSIONS:
        issues.append(_source_issue("UNSUPPORTED_SCHEMA_VERSION", f"Unsupported claim schema version: {claims_document.get('schema_version')}", field="claims.json"))
    if graph_document.get("schema_version") not in SUPPORTED_GRAPH_SCHEMA_VERSIONS:
        issues.append(_source_issue("UNSUPPORTED_SCHEMA_VERSION", f"Unsupported graph schema version: {graph_document.get('schema_version')}", field="reasoning_graph.json"))
    if claim_validation_document.get("validation_passed") is not True:
        issues.append(_source_issue("SOURCE_VALIDATION_FAILURE", "Claim validation did not pass.", field="claim_validation.json"))
    if int(claim_validation_document.get("critical_issue_count") or 0) > 0:
        issues.append(_source_issue("SOURCE_VALIDATION_FAILURE", "Claim validation reports critical issues.", field="claim_validation.json"))
    if graph_validation_document.get("validation_passed") is not True:
        issues.append(_source_issue("SOURCE_VALIDATION_FAILURE", "Reasoning graph validation did not pass.", field="reasoning_graph_validation.json"))
    if int(graph_validation_document.get("critical_issue_count") or 0) > 0:
        issues.append(_source_issue("SOURCE_VALIDATION_FAILURE", "Reasoning graph validation reports critical issues.", field="reasoning_graph_validation.json"))
    claims = tuple(claims_document.get("claims", ()) or ())
    claim_ids = [str(claim.get("claim_id")) for claim in claims]
    for claim_id, count in sorted(Counter(claim_ids).items()):
        if count > 1:
            issues.append(
                EvidenceScoringValidationIssue(
                    code="DUPLICATE_CLAIM_ID",
                    severity=EvidenceScoringIssueSeverity.CRITICAL,
                    message=f"Duplicate claim ID in source claims: {claim_id}",
                    claim_id=claim_id,
                    field="claim_id",
                )
            )
    node_ids = {str(node.get("node_id")) for node in graph_document.get("nodes", ())}
    for claim in claims:
        claim_id = str(claim.get("claim_id"))
        if not claim.get("supporting_hypothesis_ids") and claim.get("claim_type") != "WITHHELD":
            issues.append(_record_issue("UNSCORED_CLAIM", "Active source claim has no supporting hypothesis.", claim_id, "supporting_hypothesis_ids"))
        for node_id in sorted(str(item) for item in claim.get("reasoning_graph_node_ids", ()) or ()):
            if node_id not in node_ids:
                issues.append(
                    EvidenceScoringValidationIssue(
                        code="MISSING_TRACEABILITY",
                        severity=EvidenceScoringIssueSeverity.CRITICAL,
                        message=f"Source claim references a missing graph node: {node_id}",
                        claim_id=claim_id,
                        field="reasoning_graph_node_ids",
                        graph_node_id=node_id,
                        rule_id="EVIDENCE-TRACEABILITY-001",
                    )
                )
    return tuple(issues)


def validate_evidence_score_records(
    records: Iterable[EvidenceScoreRecord],
    *,
    source_claim_ids: Iterable[str] = (),
) -> tuple[EvidenceScoringValidationIssue, ...]:
    ordered = tuple(records)
    issues: list[EvidenceScoringValidationIssue] = []
    issues.extend(validate_weight_contract())
    issues.extend(validate_record_uniqueness(ordered))
    issues.extend(validate_record_completeness(ordered, source_claim_ids=tuple(source_claim_ids)))
    for record in ordered:
        issues.extend(validate_record(record))
    issues.extend(validate_deterministic_ordering(ordered))
    return tuple(issues)


def validate_weight_contract() -> tuple[EvidenceScoringValidationIssue, ...]:
    total = round(sum(DIMENSION_WEIGHTS.values()), 10)
    if total == 1.0 and all(weight >= 0 for weight in DIMENSION_WEIGHTS.values()):
        return tuple()
    return (
        EvidenceScoringValidationIssue(
            code="INVALID_WEIGHT",
            severity=EvidenceScoringIssueSeverity.CRITICAL,
            message=f"Dimension weights must be non-negative and sum to exactly 1.0; found {total}.",
            field="DIMENSION_WEIGHTS",
            rule_id="EVIDENCE-WEIGHT-CONTRACT-001",
        ),
    )


def validate_record_uniqueness(records: tuple[EvidenceScoreRecord, ...]) -> tuple[EvidenceScoringValidationIssue, ...]:
    issues = []
    for claim_id, count in sorted(Counter(record.claim_id for record in records).items()):
        if count > 1:
            issues.append(_record_issue("DUPLICATE_CLAIM_ID", f"Duplicate scored claim ID: {claim_id}", claim_id, "claim_id"))
    return tuple(issues)


def validate_record_completeness(
    records: tuple[EvidenceScoreRecord, ...],
    *,
    source_claim_ids: tuple[str, ...],
) -> tuple[EvidenceScoringValidationIssue, ...]:
    scored = {record.claim_id for record in records}
    return tuple(
        _record_issue("UNSCORED_CLAIM", f"Source claim was not scored or withheld: {claim_id}", claim_id, "claim_id")
        for claim_id in sorted(set(source_claim_ids) - scored)
    )


def validate_record(record: EvidenceScoreRecord) -> tuple[EvidenceScoringValidationIssue, ...]:
    issues: list[EvidenceScoringValidationIssue] = []
    actual_dimensions = set(record.dimension_scores)
    expected_dimensions = set(WEIGHTED_DIMENSIONS)
    missing_dimensions = expected_dimensions - actual_dimensions
    extra_dimensions = actual_dimensions - expected_dimensions
    for dimension in sorted(missing_dimensions, key=lambda item: item.value):
        issues.append(_record_issue("INVALID_DIMENSION", f"Missing evidence dimension: {dimension.value}", record.claim_id, "dimension_scores"))
    for dimension in sorted(extra_dimensions, key=lambda item: item.value):
        issues.append(_record_issue("INVALID_DIMENSION", f"Unexpected weighted evidence dimension: {dimension.value}", record.claim_id, "dimension_scores"))
    for dimension, score in sorted(record.dimension_scores.items(), key=lambda item: item[0].value):
        if not (0 <= score.raw_score <= 100):
            issues.append(_record_issue("INVALID_DIMENSION", f"Dimension score out of range: {dimension.value}", record.claim_id, "dimension_scores"))
        if score.weight != DIMENSION_WEIGHTS[EvidenceDimension(dimension)]:
            issues.append(_record_issue("INVALID_WEIGHT", f"Dimension weight mismatch: {dimension.value}", record.claim_id, "dimension_scores"))
    if not (0 <= record.normalized_score <= 100):
        issues.append(_record_issue("INVALID_DIMENSION", "Normalized score must be between 0 and 100.", record.claim_id, "normalized_score"))
    expected_level = evidence_level_from_score(record.normalized_score)
    if record.evidence_level != expected_level and not record.is_withheld:
        issues.append(
            _record_issue(
                "INVALID_EVIDENCE_LEVEL",
                f"Evidence level {record.evidence_level.value} does not match score-derived {expected_level.value}.",
                record.claim_id,
                "evidence_level",
            )
        )
    issues.extend(_validate_publication_policy(record))
    issues.extend(_validate_uncertainty_policy(record))
    issues.extend(_validate_withholding_policy(record))
    issues.extend(_validate_traceability(record))
    issues.extend(_validate_language(record))
    return tuple(issues)


def _validate_publication_policy(record: EvidenceScoreRecord) -> tuple[EvidenceScoringValidationIssue, ...]:
    issues = []
    ceiling = PUBLICATION_CEILINGS.get(record.claim_publication_use, PublicationReadiness.NOT_READY)
    if READINESS_RANK[record.publication_readiness] > READINESS_RANK[ceiling]:
        issues.append(
            _record_issue(
                "PUBLICATION_POLICY_ISSUE",
                f"Publication readiness exceeds Claim Engine publication_use ceiling: {record.claim_publication_use}.",
                record.claim_id,
                "publication_readiness",
            )
        )
    if record.claim_type == "LIMITATION" and record.publication_readiness not in (
        PublicationReadiness.LIMITATION_ONLY,
        PublicationReadiness.NOT_READY,
    ):
        issues.append(_record_issue("PUBLICATION_POLICY_ISSUE", "Limitation claims cannot become results-ready.", record.claim_id, "publication_readiness"))
    if record.claim_status == "TENTATIVE" and record.reviewer_confidence.value == "HIGH":
        issues.append(_record_issue("PUBLICATION_POLICY_ISSUE", "Tentative claims cannot receive HIGH reviewer confidence.", record.claim_id, "reviewer_confidence"))
    return tuple(issues)


def _validate_uncertainty_policy(record: EvidenceScoreRecord) -> tuple[EvidenceScoringValidationIssue, ...]:
    if record.claim_status == "CONFLICTED" and record.uncertainty_level is UncertaintyLevel.VERY_LOW:
        return (_record_issue("INVALID_UNCERTAINTY_MAPPING", "Conflicted claims cannot receive VERY_LOW uncertainty.", record.claim_id, "uncertainty_level"),)
    return tuple()


def _validate_withholding_policy(record: EvidenceScoreRecord) -> tuple[EvidenceScoringValidationIssue, ...]:
    if not record.is_withheld:
        return tuple()
    issues = []
    if record.publication_readiness is not PublicationReadiness.NOT_READY:
        issues.append(_record_issue("WITHHELD_POLICY_ISSUE", "Withheld evidence records must be NOT_READY.", record.claim_id, "publication_readiness"))
    if record.normalized_score != 0:
        issues.append(_record_issue("WITHHELD_POLICY_ISSUE", "Withheld evidence records must have normalized_score 0.", record.claim_id, "normalized_score"))
    if not record.withholding_reasons:
        issues.append(_record_issue("WITHHELD_POLICY_ISSUE", "Withheld evidence records must include reasons.", record.claim_id, "withholding_reasons"))
    return tuple(issues)


def _validate_traceability(record: EvidenceScoreRecord) -> tuple[EvidenceScoringValidationIssue, ...]:
    if record.is_withheld:
        return tuple()
    if not record.supporting_hypothesis_ids or not record.supporting_interpretation_ids or not record.supporting_observation_ids:
        return (_record_issue("MISSING_TRACEABILITY", "Active evidence records require hypothesis, interpretation, and observation links.", record.claim_id, "reasoning_graph_node_ids"),)
    if not record.reasoning_graph_node_ids:
        return (_record_issue("MISSING_TRACEABILITY", "Active evidence records require graph-node traceability.", record.claim_id, "reasoning_graph_node_ids"),)
    return tuple()


def _validate_language(record: EvidenceScoreRecord) -> tuple[EvidenceScoringValidationIssue, ...]:
    text = " ".join(
        [
            record.score_explanation,
            record.uncertainty_explanation,
            record.reviewer_confidence_explanation,
            record.publication_readiness_explanation,
        ]
    ).lower()
    issues = []
    probability_terms = ("probability that", "posterior probability", "p-value")
    if any(term in text for term in probability_terms) and "not probabilities" not in text and "not a probability" not in text:
        issues.append(_record_issue("PROBABILITY_LANGUAGE_ISSUE", "Score explanation may imply a probabilistic interpretation.", record.claim_id, "score_explanation"))
    for term in ("proves", "establishes causation", "caused"):
        if term in text:
            issues.append(_record_issue("CAUSAL_OVERCLAIM", f"Evidence scoring output contains causal overclaim: {term}", record.claim_id, "score_explanation"))
    for term in ("mechanism proof", "mechanism is"):
        if term in text and "not" not in text:
            issues.append(_record_issue("MECHANISM_OVERCLAIM", f"Evidence scoring output contains mechanistic overclaim: {term}", record.claim_id, "score_explanation"))
    for term in ("novel", "novelty evidence"):
        if term in text and "not" not in text:
            issues.append(_record_issue("NOVELTY_CLAIM_ISSUE", f"Evidence scoring output contains novelty overclaim: {term}", record.claim_id, "score_explanation"))
    if record.publication_readiness is PublicationReadiness.HIGH_CONFIDENCE_RESULTS_READY and any(
        factor == "no genuine external validation is traceable" for factor in record.negative_factors
    ):
        issues.append(_record_issue("EXTERNAL_VALIDATION_OVERCLAIM", "High-confidence results readiness requires genuine external validation.", record.claim_id, "publication_readiness"))
    return tuple(issues)


def validate_deterministic_ordering(records: tuple[EvidenceScoreRecord, ...]) -> tuple[EvidenceScoringValidationIssue, ...]:
    actual = [record.claim_id for record in records]
    if actual == sorted(actual):
        return tuple()
    return (
        EvidenceScoringValidationIssue(
            code="DETERMINISTIC_ORDERING_ISSUE",
            severity=EvidenceScoringIssueSeverity.WARNING,
            message="Evidence score records are not ordered deterministically by claim_id.",
            field="records",
        ),
    )


def validation_summary(
    records: tuple[EvidenceScoreRecord, ...],
    issues: tuple[EvidenceScoringValidationIssue, ...],
    *,
    output_readability_checks: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    code_counts = Counter(issue.code for issue in issues)
    critical_count = sum(1 for issue in issues if issue.severity is EvidenceScoringIssueSeverity.CRITICAL)
    warning_count = sum(1 for issue in issues if issue.severity is EvidenceScoringIssueSeverity.WARNING)
    readability_failed = sum(1 for check in output_readability_checks.values() if not check["readable"])
    return {
        "validation_passed": critical_count == 0 and readability_failed == 0,
        "critical_issue_count": critical_count,
        "warning_count": warning_count,
        "duplicate_claim_id_count": code_counts["DUPLICATE_CLAIM_ID"],
        "unscored_claim_count": code_counts["UNSCORED_CLAIM"],
        "invalid_dimension_count": code_counts["INVALID_DIMENSION"],
        "invalid_weight_count": code_counts["INVALID_WEIGHT"],
        "missing_traceability_count": code_counts["MISSING_TRACEABILITY"],
        "source_validation_failure_count": code_counts["SOURCE_VALIDATION_FAILURE"] + code_counts["UNSUPPORTED_SCHEMA_VERSION"],
        "publication_policy_issue_count": code_counts["PUBLICATION_POLICY_ISSUE"],
        "external_validation_overclaim_count": code_counts["EXTERNAL_VALIDATION_OVERCLAIM"],
        "probability_language_issue_count": code_counts["PROBABILITY_LANGUAGE_ISSUE"],
        "causal_overclaim_count": code_counts["CAUSAL_OVERCLAIM"],
        "mechanism_overclaim_count": code_counts["MECHANISM_OVERCLAIM"],
        "novelty_claim_issue_count": code_counts["NOVELTY_CLAIM_ISSUE"],
        "deterministic_ordering_issue_count": code_counts["DETERMINISTIC_ORDERING_ISSUE"],
        "withheld_claim_count": sum(1 for record in records if record.is_withheld),
        "structured_validation_issues": [issue.to_record() for issue in issues],
        "output_readability_checks": output_readability_checks,
    }


def _source_issue(code: str, message: str, *, field: str) -> EvidenceScoringValidationIssue:
    return EvidenceScoringValidationIssue(
        code=code,
        severity=EvidenceScoringIssueSeverity.CRITICAL,
        message=message,
        field=field,
    )


def _record_issue(code: str, message: str, claim_id: str, field: str) -> EvidenceScoringValidationIssue:
    return EvidenceScoringValidationIssue(
        code=code,
        severity=EvidenceScoringIssueSeverity.CRITICAL,
        message=message,
        claim_id=claim_id,
        field=field,
    )
