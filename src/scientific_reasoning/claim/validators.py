"""Validation contracts for BSIP scientific claims."""

from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from collections.abc import Iterable
from datetime import datetime
from typing import Any

from .enums import (
    ClaimCategory,
    ClaimIssueSeverity,
    ClaimStatus,
    ClaimType,
    EvidenceStrength,
    PublicationUse,
    category_id_token,
)
from .models import ScientificClaim, ClaimValidationIssue
from .policies import (
    causal_overclaim_terms,
    evidence_strength_from_score,
    external_validation_overclaim_terms,
    mechanistic_overclaim_terms,
    novelty_overclaim_terms,
    publication_use_for_claim,
    recommendation_terms,
)


CLAIM_ID_PATTERN = re.compile(r"^CLM-([A-Z_]+)-([0-9]{4})$")


def validate_claim(
    claim: ScientificClaim,
    *,
    hypotheses_by_id: dict[str, dict[str, Any]] | None = None,
    graph_document: dict[str, Any] | None = None,
) -> tuple[ClaimValidationIssue, ...]:
    issues: list[ClaimValidationIssue] = []
    issues.extend(validate_required_fields(claim))
    issues.extend(validate_claim_id_format(claim))
    issues.extend(validate_claim_language(claim))
    issues.extend(validate_claim_score_policy(claim))
    issues.extend(validate_publication_use_policy(claim))
    issues.extend(validate_claim_serializability(claim))
    issues.extend(validate_claim_deterministic_fields(claim))
    if hypotheses_by_id is not None:
        issues.extend(validate_hypothesis_dependencies(claim, hypotheses_by_id))
    if graph_document is not None:
        issues.extend(validate_graph_dependencies(claim, graph_document))
        issues.extend(validate_active_claim_traceability(claim, graph_document))
    issues.extend(validate_limitations(claim))
    issues.extend(validate_withheld_claim_policy(claim))
    return tuple(issues)


def validate_claims(
    claims: Iterable[ScientificClaim],
    *,
    hypotheses_by_id: dict[str, dict[str, Any]] | None = None,
    graph_document: dict[str, Any] | None = None,
) -> tuple[ClaimValidationIssue, ...]:
    ordered = tuple(claims)
    issues: list[ClaimValidationIssue] = []
    for claim in ordered:
        issues.extend(validate_claim(claim, hypotheses_by_id=hypotheses_by_id, graph_document=graph_document))
    issues.extend(validate_unique_ids(ordered))
    issues.extend(validate_deterministic_ordering(ordered))
    return tuple(issues)


def validate_required_fields(claim: ScientificClaim) -> tuple[ClaimValidationIssue, ...]:
    required = (
        "claim_id",
        "category",
        "title",
        "claim_text",
        "claim_type",
        "claim_status",
        "evidence_strength",
        "publication_use",
        "rationale",
        "created_at",
        "software_version",
    )
    issues = []
    for field_name in required:
        value = getattr(claim, field_name)
        if value is None or value == "":
            issues.append(_issue("REQUIRED_FIELD_MISSING", f"Required field is missing: {field_name}", claim, field_name))
    return tuple(issues)


def validate_claim_id_format(claim: ScientificClaim) -> tuple[ClaimValidationIssue, ...]:
    match = CLAIM_ID_PATTERN.match(claim.claim_id)
    if not match:
        return (
            _issue(
                "INVALID_CLAIM_ID",
                "Claim ID must match CLM-{CATEGORY}-{NUMBER} with a four-digit number.",
                claim,
                "claim_id",
            ),
        )
    category_token, _number = match.groups()
    expected = category_id_token(claim.category)
    if category_token != expected:
        return (
            _issue(
                "CLAIM_ID_CATEGORY_MISMATCH",
                f"Claim ID category token {category_token} does not match {expected}.",
                claim,
                "claim_id",
            ),
        )
    return tuple()


def validate_unique_ids(claims: Iterable[ScientificClaim]) -> tuple[ClaimValidationIssue, ...]:
    seen: set[str] = set()
    issues = []
    for claim in claims:
        if claim.claim_id in seen:
            issues.append(_issue("DUPLICATE_CLAIM_ID", f"Duplicate claim ID: {claim.claim_id}", claim, "claim_id"))
        seen.add(claim.claim_id)
    return tuple(issues)


def validate_hypothesis_dependencies(
    claim: ScientificClaim,
    hypotheses_by_id: dict[str, dict[str, Any]],
) -> tuple[ClaimValidationIssue, ...]:
    issues = []
    for hypothesis_id in sorted(set(claim.supporting_hypothesis_ids) | set(claim.competing_hypothesis_ids)):
        if hypothesis_id not in hypotheses_by_id:
            issues.append(
                ClaimValidationIssue(
                    code="MISSING_HYPOTHESIS_DEPENDENCY",
                    severity=ClaimIssueSeverity.CRITICAL,
                    message=f"Referenced hypothesis does not exist: {hypothesis_id}",
                    claim_id=claim.claim_id,
                    field="supporting_hypothesis_ids",
                    hypothesis_id=hypothesis_id,
                    graph_node_id=None,
                    rule_id=None,
                )
            )
    if claim.claim_type is not ClaimType.WITHHELD and not claim.supporting_hypothesis_ids:
        issues.append(_issue("UNSUPPORTED_CLAIM", "Active claims must cite at least one supporting hypothesis.", claim, "supporting_hypothesis_ids"))
    return tuple(issues)


def validate_graph_dependencies(
    claim: ScientificClaim,
    graph_document: dict[str, Any],
) -> tuple[ClaimValidationIssue, ...]:
    node_ids = {str(node.get("node_id")) for node in graph_document.get("nodes", ())}
    issues = []
    for node_id in sorted(claim.reasoning_graph_node_ids):
        if node_id not in node_ids:
            issues.append(
                ClaimValidationIssue(
                    code="MISSING_GRAPH_DEPENDENCY",
                    severity=ClaimIssueSeverity.CRITICAL,
                    message=f"Referenced graph node does not exist: {node_id}",
                    claim_id=claim.claim_id,
                    field="reasoning_graph_node_ids",
                    hypothesis_id=None,
                    graph_node_id=node_id,
                    rule_id=None,
                )
            )
    for node_id in sorted(claim.validation_summary_ids):
        if node_id not in node_ids:
            issues.append(
                ClaimValidationIssue(
                    code="MISSING_GRAPH_DEPENDENCY",
                    severity=ClaimIssueSeverity.CRITICAL,
                    message=f"Referenced validation-summary node does not exist: {node_id}",
                    claim_id=claim.claim_id,
                    field="validation_summary_ids",
                    hypothesis_id=None,
                    graph_node_id=node_id,
                    rule_id=None,
                )
            )
    return tuple(issues)


def validate_active_claim_traceability(
    claim: ScientificClaim,
    graph_document: dict[str, Any],
) -> tuple[ClaimValidationIssue, ...]:
    if claim.claim_type is ClaimType.WITHHELD:
        return tuple()
    node_types = {str(node.get("node_id")): str(node.get("node_type")) for node in graph_document.get("nodes", ())}
    support_parents: dict[str, set[str]] = defaultdict(set)
    for edge in graph_document.get("edges", ()):
        if edge.get("edge_type") == "supports":
            support_parents[str(edge.get("target_id"))].add(str(edge.get("source_id")))
    issues = []
    if not claim.validation_summary_ids:
        issues.append(_issue("MISSING_TRACEABILITY", "Active claims must cite at least one validation-summary node.", claim, "validation_summary_ids"))
    for hypothesis_id in claim.supporting_hypothesis_ids:
        ancestors = _support_ancestors(hypothesis_id, support_parents)
        has_interpretation = any(node_types.get(node_id) == "Interpretation" for node_id in ancestors)
        has_observation = any(node_types.get(node_id) == "Observation" for node_id in ancestors)
        if not (has_interpretation and has_observation):
            issues.append(
                ClaimValidationIssue(
                    code="MISSING_TRACEABILITY",
                    severity=ClaimIssueSeverity.CRITICAL,
                    message=(
                        "Active claims must have complete observation-to-interpretation-to-hypothesis "
                        f"traceability: {hypothesis_id}"
                    ),
                    claim_id=claim.claim_id,
                    field="supporting_hypothesis_ids",
                    hypothesis_id=hypothesis_id,
                    graph_node_id=hypothesis_id,
                    rule_id="CLAIM-TRACEABILITY-001",
                )
            )
    return tuple(issues)


def validate_claim_language(claim: ScientificClaim) -> tuple[ClaimValidationIssue, ...]:
    issues = []
    for term in causal_overclaim_terms(claim.claim_text):
        issues.append(_issue("CAUSAL_OVERCLAIM", f"Claim text contains causal overclaim language: {term}", claim, "claim_text"))
    for term in mechanistic_overclaim_terms(claim.claim_text):
        issues.append(_issue("MECHANISM_OVERCLAIM", f"Claim text contains mechanistic overclaim language: {term}", claim, "claim_text"))
    for term in novelty_overclaim_terms(claim.claim_text):
        issues.append(_issue("NOVELTY_CLAIM_ISSUE", f"Claim text contains novelty language: {term}", claim, "claim_text"))
    for term in external_validation_overclaim_terms(claim.claim_text):
        issues.append(
            _issue(
                "EXTERNAL_VALIDATION_OVERCLAIM",
                f"Claim text contains external-validation, deployment, clinical, or regulatory overclaim language: {term}",
                claim,
                "claim_text",
            )
        )
    for term in recommendation_terms(claim.claim_text):
        issues.append(_issue("UNSUPPORTED_CLAIM", f"Claim text contains recommendation language: {term}", claim, "claim_text"))
    return tuple(issues)


def validate_limitations(claim: ScientificClaim) -> tuple[ClaimValidationIssue, ...]:
    if claim.claim_type is ClaimType.WITHHELD:
        return tuple()
    if claim.limitations:
        return tuple()
    return (_issue("MISSING_LIMITATION", "Active claims must include explicit limitations.", claim, "limitations"),)


def validate_withheld_claim_policy(claim: ScientificClaim) -> tuple[ClaimValidationIssue, ...]:
    if claim.claim_type is not ClaimType.WITHHELD:
        return tuple()
    issues = []
    if claim.claim_status is not ClaimStatus.WITHHELD:
        issues.append(_issue("WITHHELD_POLICY_ISSUE", "WITHHELD claims must use ClaimStatus.WITHHELD.", claim, "claim_status"))
    if claim.publication_use is not PublicationUse.NOT_ELIGIBLE:
        issues.append(_issue("WITHHELD_POLICY_ISSUE", "WITHHELD claims must use PublicationUse.NOT_ELIGIBLE.", claim, "publication_use"))
    if claim.evidence_score != 0:
        issues.append(_issue("EVIDENCE_SCORE_POLICY_ISSUE", "WITHHELD claims must use evidence_score 0.", claim, "evidence_score"))
    if not claim.rationale:
        issues.append(_issue("WITHHELD_POLICY_ISSUE", "WITHHELD claims must include withholding rationale.", claim, "rationale"))
    return tuple(issues)


def validate_claim_score_policy(claim: ScientificClaim) -> tuple[ClaimValidationIssue, ...]:
    issues = []
    if not (0 <= claim.evidence_score <= 100):
        issues.append(_issue("EVIDENCE_SCORE_POLICY_ISSUE", "evidence_score must be between 0 and 100.", claim, "evidence_score"))
        return tuple(issues)
    expected = evidence_strength_from_score(claim.evidence_score)
    if claim.evidence_strength is not expected:
        issues.append(
            _issue(
                "EVIDENCE_SCORE_POLICY_ISSUE",
                f"Evidence strength {claim.evidence_strength.value} does not match score-derived {expected.value}.",
                claim,
                "evidence_strength",
            )
        )
    if claim.claim_type is not ClaimType.WITHHELD and claim.evidence_strength in (
        EvidenceStrength.INSUFFICIENT,
        EvidenceStrength.NOT_ASSESSABLE,
    ):
        issues.append(_issue("UNSUPPORTED_CLAIM", "Active claims must not have insufficient or not-assessable evidence.", claim, "evidence_strength"))
    return tuple(issues)


def validate_publication_use_policy(claim: ScientificClaim) -> tuple[ClaimValidationIssue, ...]:
    if claim.claim_type is ClaimType.WITHHELD:
        expected = PublicationUse.NOT_ELIGIBLE
    else:
        expected = publication_use_for_claim(
            claim_type=claim.claim_type,
            claim_status=claim.claim_status,
            evidence_strength=claim.evidence_strength,
            has_critical_issue=False,
            category=claim.category.value,
        )
    if claim.publication_use is expected:
        return tuple()
    return (
        _issue(
            "PUBLICATION_USE_POLICY_ISSUE",
            f"Publication use {claim.publication_use.value} does not match policy-derived {expected.value}.",
            claim,
            "publication_use",
        ),
    )


def validate_claim_serializability(claim: ScientificClaim) -> tuple[ClaimValidationIssue, ...]:
    try:
        json.dumps(claim.to_record(), sort_keys=True)
    except (TypeError, ValueError) as exc:
        return (_issue("NON_SERIALIZABLE_METADATA", f"Claim is not JSON serializable: {exc}", claim, "metadata"),)
    try:
        datetime.fromisoformat(claim.created_at)
    except ValueError:
        return (_issue("INVALID_TIMESTAMP", "created_at must be ISO 8601 parseable.", claim, "created_at"),)
    return tuple()


def validate_claim_deterministic_fields(claim: ScientificClaim) -> tuple[ClaimValidationIssue, ...]:
    issues = []
    for field_name in (
        "supporting_hypothesis_ids",
        "competing_hypothesis_ids",
        "supporting_interpretation_ids",
        "supporting_observation_ids",
        "evidence_gap_ids",
        "validation_summary_ids",
        "reasoning_graph_node_ids",
        "language_policy_rule_ids",
        "reasoning_rule_ids",
        "tags",
    ):
        values = getattr(claim, field_name)
        if tuple(sorted(values)) != values:
            issues.append(
                _issue(
                    "DETERMINISTIC_ORDERING_ISSUE",
                    f"{field_name} must be sorted deterministically.",
                    claim,
                    field_name,
                    severity=ClaimIssueSeverity.WARNING,
                )
            )
    return tuple(issues)


def validate_deterministic_ordering(claims: Iterable[ScientificClaim]) -> tuple[ClaimValidationIssue, ...]:
    ordered = tuple(claims)
    actual_ids = [claim.claim_id for claim in ordered]
    if actual_ids == sorted(actual_ids):
        return tuple()
    return (
        ClaimValidationIssue(
            code="DETERMINISTIC_ORDERING_ISSUE",
            severity=ClaimIssueSeverity.WARNING,
            message="Claims are not ordered deterministically by claim_id.",
            claim_id=None,
            field="claims",
            hypothesis_id=None,
            graph_node_id=None,
            rule_id=None,
        ),
    )


def _support_ancestors(node_id: str, support_parents: dict[str, set[str]]) -> set[str]:
    seen = set()
    queue = deque(sorted(support_parents.get(node_id, ())))
    while queue:
        current = queue.popleft()
        if current in seen:
            continue
        seen.add(current)
        queue.extend(parent for parent in sorted(support_parents.get(current, ())) if parent not in seen)
    return seen


def _issue(
    code: str,
    message: str,
    claim: ScientificClaim,
    field: str,
    *,
    severity: ClaimIssueSeverity = ClaimIssueSeverity.CRITICAL,
) -> ClaimValidationIssue:
    return ClaimValidationIssue(
        code=code,
        severity=severity,
        message=message,
        claim_id=claim.claim_id,
        field=field,
        hypothesis_id=None,
        graph_node_id=None,
        rule_id=None,
    )
