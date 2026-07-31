"""Deterministic review policies for the BSIP v4.1.0 Reviewer Engine."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Iterable

from .enums import (
    OverallRecommendation,
    PublicationRisk,
    ReviewCategory,
    ReviewerConfidence,
    ReviewerType,
    Severity,
)
from .models import REVIEW_SOFTWARE_VERSION, ReviewFinding


SUPPORTED_CLAIM_SCHEMA_VERSIONS: frozenset[str] = frozenset({"BSIP-3.2.0"})
SUPPORTED_EVIDENCE_SCHEMA_VERSIONS: frozenset[str] = frozenset({"BSIP-4.0.0"})
SUPPORTED_GRAPH_SCHEMA_VERSIONS: frozenset[str] = frozenset({"BSIP-3.1.0"})

SEVERITY_RANK: dict[Severity, int] = {
    Severity.INFORMATION: 0,
    Severity.MINOR: 1,
    Severity.MODERATE: 2,
    Severity.MAJOR: 3,
    Severity.CRITICAL: 4,
}

REVISION_REQUIREMENTS: dict[str, str] = {
    "REVIEW-SCI-CONFOUNDING-001": "State unresolved competing explanations alongside the affected claim.",
    "REVIEW-SCI-LIMITATION-001": "Keep claim wording aligned with the recorded limitations and gaps.",
    "REVIEW-STAT-EXTERNAL-001": "Restrict definitive generalization language unless independently labelled external-validation evidence is traceable.",
    "REVIEW-STAT-REGRESSION-001": "Report regression support as guarded where uncertainty remains high.",
    "REVIEW-STAT-METRIC-COMPAT-001": "Keep task-specific performance statements separated by metric family.",
    "REVIEW-EVIDENCE-UNCERTAINTY-001": "Qualify strong evidence scores with the recorded uncertainty level.",
    "REVIEW-EVIDENCE-READINESS-001": "Use downstream publication-readiness labels when assigning claim placement.",
    "REVIEW-EVIDENCE-GAP-001": "Retain missing-evidence statements for claims with recorded evidence gaps.",
    "REVIEW-VALIDATION-SOURCE-001": "Resolve upstream validation failures before publication-facing review use.",
    "REVIEW-VALIDATION-EXTERNAL-001": "Restrict definitive generalization language unless independently labelled external-validation evidence is traceable.",
    "REVIEW-REPRO-WORKFLOW-001": "Restore workflow or graph metadata required for reproducibility traceability.",
    "REVIEW-REPRO-REPLICATE-001": "Identify biological reproducibility as not established by computational traceability alone.",
    "REVIEW-FIGURE-METADATA-001": "Record claim-level figure and table links before treating visual support as reviewed.",
    "REVIEW-FIGURE-CLAIM-LINK-001": "Provide claim-level figure or table links for each affected publication-facing claim.",
    "REVIEW-WRITING-LANGUAGE-001": "Replace definitive wording with bounded language consistent with recorded uncertainty.",
    "REVIEW-WRITING-FORBIDDEN-001": "Remove unsupported causal, mechanistic, or external-validation wording.",
    "REVIEW-PUBLICATION-OVERALL-001": "Address blocking reviewer findings before draft-level publication use.",
}

PROHIBITED_PROTOCOL_TERMS: tuple[str, ...] = (
    "run a new experiment",
    "perform new experiments",
    "collect additional samples",
    "increase the sample size",
    "add more samples",
    "new assay",
    "wet-lab protocol",
)

NOVELTY_TERMS: tuple[str, ...] = (
    "novel",
    "first demonstration",
    "first-in-class",
    "unprecedented",
)

JOURNAL_PREDICTION_TERMS: tuple[str, ...] = (
    "acceptance probability",
    "will be accepted",
    "journal impact",
    "target journal",
    "reviewer will accept",
)

NEW_CLAIM_PATTERNS: tuple[str, ...] = (
    "we discovered",
    "we demonstrate that",
    "this proves",
    "proves that",
    "causes",
    "establishes causation",
    "mechanism is",
)

WRITING_OVERCLAIM_PATTERNS: tuple[tuple[str, str], ...] = (
    ("causal", r"\b(causes|caused|causation|causal effect|drives)\b"),
    ("mechanism", r"\b(mechanism|mechanistic|pathway)\b"),
    ("novelty", r"\b(novel|first demonstration|first-in-class|unprecedented)\b"),
    ("proof", r"\b(proves|proven|definitively|establishes)\b"),
    ("external validation", r"\b(externally validated|external validation established|generalizes to external)\b"),
)


def finding_id(reviewer_type: ReviewerType | str, index: int) -> str:
    return f"REV-{ReviewerType(reviewer_type).value}-{index:04d}"


def publication_risk_for(severity: Severity | str, *, blocking: bool) -> PublicationRisk:
    severity = Severity(severity)
    if blocking or severity is Severity.CRITICAL:
        return PublicationRisk.BLOCKING
    if severity is Severity.MAJOR:
        return PublicationRisk.HIGH
    if severity is Severity.MODERATE:
        return PublicationRisk.MODERATE
    if severity is Severity.MINOR:
        return PublicationRisk.LOW
    return PublicationRisk.NONE


def default_blocking(category: ReviewCategory | str, severity: Severity | str, rule_ids: Iterable[str] = ()) -> bool:
    category = ReviewCategory(category)
    severity = Severity(severity)
    rules = set(rule_ids)
    if severity is Severity.CRITICAL:
        return True
    if severity is not Severity.MAJOR:
        return False
    blocking_rules = {
        "REVIEW-SCI-CONFOUNDING-001",
        "REVIEW-STAT-EXTERNAL-001",
        "REVIEW-VALIDATION-EXTERNAL-001",
        "REVIEW-FIGURE-CLAIM-LINK-001",
    }
    blocking_categories = {
        ReviewCategory.GENERALIZATION,
        ReviewCategory.EXTERNAL_VALIDATION,
        ReviewCategory.TRACEABILITY,
        ReviewCategory.COMPETING_EXPLANATIONS,
    }
    return bool(rules & blocking_rules) or category in blocking_categories


def revision_requirement_for(rule_ids: Iterable[str], severity: Severity | str) -> str:
    if Severity(severity) is Severity.INFORMATION:
        return ""
    for rule_id in sorted(rule_ids):
        requirement = REVISION_REQUIREMENTS.get(rule_id)
        if requirement:
            return requirement
    return "Resolve the reviewer finding while preserving source traceability."


def build_finding(
    *,
    reviewer_type: ReviewerType | str,
    index: int,
    category: ReviewCategory | str,
    title: str,
    finding_text: str,
    severity: Severity | str,
    confidence: ReviewerConfidence | str = ReviewerConfidence.MODERATE,
    blocking: bool | None = None,
    affected_claim_ids: Iterable[str] = (),
    affected_hypothesis_ids: Iterable[str] = (),
    affected_interpretation_ids: Iterable[str] = (),
    affected_observation_ids: Iterable[str] = (),
    affected_figure_ids: Iterable[str] = (),
    affected_table_ids: Iterable[str] = (),
    evidence_score_ids: Iterable[str] = (),
    reasoning_graph_node_ids: Iterable[str] = (),
    source_validation_ids: Iterable[str] = (),
    rationale: str = "",
    evidence_summary: str = "",
    publication_risk: PublicationRisk | str | None = None,
    revision_requirement: str | None = None,
    limitations: Iterable[str] = (),
    rule_ids: Iterable[str] = (),
    created_at: str,
    software_version: str = REVIEW_SOFTWARE_VERSION,
    tags: Iterable[str] = (),
    metadata: dict[str, Any] | None = None,
) -> ReviewFinding:
    reviewer_type = ReviewerType(reviewer_type)
    severity = Severity(severity)
    category = ReviewCategory(category)
    rules = tuple(sorted(str(rule_id) for rule_id in rule_ids))
    if blocking is None:
        blocking = default_blocking(category, severity, rules)
    if publication_risk is None:
        publication_risk = publication_risk_for(severity, blocking=blocking)
    if revision_requirement is None:
        revision_requirement = revision_requirement_for(rules, severity)
    return ReviewFinding(
        finding_id=finding_id(reviewer_type, index),
        reviewer_type=reviewer_type,
        category=category,
        title=title,
        finding_text=finding_text,
        severity=severity,
        blocking=blocking,
        confidence=confidence,
        affected_claim_ids=tuple(affected_claim_ids),
        affected_hypothesis_ids=tuple(affected_hypothesis_ids),
        affected_interpretation_ids=tuple(affected_interpretation_ids),
        affected_observation_ids=tuple(affected_observation_ids),
        affected_figure_ids=tuple(affected_figure_ids),
        affected_table_ids=tuple(affected_table_ids),
        evidence_score_ids=tuple(evidence_score_ids),
        reasoning_graph_node_ids=tuple(reasoning_graph_node_ids),
        source_validation_ids=tuple(source_validation_ids),
        rationale=rationale,
        evidence_summary=evidence_summary,
        publication_risk=publication_risk,
        revision_requirement=revision_requirement,
        limitations=tuple(limitations),
        rule_ids=rules,
        created_at=created_at,
        software_version=software_version,
        tags=tuple(tags),
        metadata=metadata or {},
    )


def sort_findings(findings: Iterable[ReviewFinding]) -> tuple[ReviewFinding, ...]:
    return tuple(
        sorted(
            findings,
            key=lambda finding: (
                finding.reviewer_type.value,
                finding.finding_id,
                finding.category.value,
                finding.title,
            ),
        )
    )


def severity_counts(findings: Iterable[ReviewFinding]) -> dict[str, int]:
    counts = Counter(finding.severity.value for finding in findings)
    return {severity.value: counts[severity.value] for severity in Severity}


def determine_recommendation(
    findings: Iterable[ReviewFinding],
    *,
    has_results_ready_claim: bool,
) -> OverallRecommendation:
    material = [finding for finding in findings if finding.severity is not Severity.INFORMATION]
    if any(finding.severity is Severity.CRITICAL for finding in material):
        return OverallRecommendation.INTERNAL_REVIEW_ONLY
    blocking_major = [finding for finding in material if finding.severity is Severity.MAJOR and finding.blocking]
    if len(blocking_major) >= 2:
        return OverallRecommendation.NEEDS_MAJOR_REVISION
    if len(blocking_major) == 1:
        return OverallRecommendation.NEEDS_MAJOR_REVISION
    if any(finding.severity is Severity.MAJOR for finding in material):
        return OverallRecommendation.NEEDS_MAJOR_REVISION
    if any(finding.severity is Severity.MODERATE for finding in material):
        return OverallRecommendation.NEEDS_MODERATE_REVISION
    if any(finding.severity is Severity.MINOR for finding in material):
        return OverallRecommendation.NEEDS_MINOR_REVISION
    if has_results_ready_claim:
        return OverallRecommendation.READY_FOR_DRAFT_MANUSCRIPT
    return OverallRecommendation.NEEDS_MINOR_REVISION


def extract_claim_links(row: dict[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for key in ("claim_id", "claim_ids", "affected_claim_ids", "linked_claim_ids", "supporting_claim_ids"):
        raw = row.get(key)
        if raw is None or raw == "":
            continue
        values.extend(_split_identifier_value(str(raw)))
    return tuple(sorted({value for value in values if value.startswith("CLM-")}))


def extract_row_identifier(row: dict[str, Any], *preferred: str) -> str:
    for key in preferred:
        value = row.get(key)
        if value:
            return str(value)
    for key in sorted(row):
        if key.endswith("_id") and row.get(key):
            return str(row[key])
    return ""


def text_contains_any(text: str, terms: Iterable[str]) -> bool:
    lower = text.lower()
    return any(term.lower() in lower for term in terms)


def writing_overclaim_labels(text: str) -> tuple[str, ...]:
    lower = text.lower()
    labels = []
    for label, pattern in WRITING_OVERCLAIM_PATTERNS:
        if re.search(pattern, lower):
            labels.append(label)
    return tuple(sorted(labels))


def json_compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _split_identifier_value(value: str) -> list[str]:
    stripped = value.strip()
    if not stripped:
        return []
    if stripped[0] in "[{":
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
        if isinstance(parsed, dict):
            return [str(item).strip() for item in parsed.values() if str(item).strip()]
    parts = re.split(r"[;,|]", stripped)
    return [part.strip() for part in parts if part.strip()]
