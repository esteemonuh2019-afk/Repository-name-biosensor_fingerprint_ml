"""Policy helpers for conservative BSIP manuscript generation."""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

from .enums import DocumentStatus, PublicationBoundary, SectionType, SentenceStatus, TraceabilityStatus


REQUIRED_VALIDATION_FILENAMES: tuple[str, ...] = (
    "observation_validation.json",
    "interpretation_validation.json",
    "hypothesis_validation.json",
    "claim_validation.json",
    "evidence_scoring_validation.json",
    "reviewer_validation.json",
    "reasoning_graph_validation.json",
    "report_validation.json",
)

PLACEHOLDER_TEXT = {
    SectionType.ABSTRACT_PLACEHOLDER: "ABSTRACT PLACEHOLDER: Requires final manuscript scope, Methods summary, verified numerical results, and author-approved conclusions.",
    SectionType.INTRODUCTION_PLACEHOLDER: "INTRODUCTION PLACEHOLDER: Requires a verified literature package and author-approved research rationale.",
    SectionType.METHODS_PLACEHOLDER: "METHODS PLACEHOLDER: Requires validated experimental-design metadata, complete preprocessing details, model-training protocol, and author review.",
}

FORBIDDEN_LANGUAGE_PATTERNS: dict[str, tuple[str, ...]] = {
    "CAUSAL_LANGUAGE_ISSUE": ("proves", "confirms conclusively", "demonstrates definitively", "caused"),
    "MECHANISM_LANGUAGE_ISSUE": ("mechanism", "mechanistic"),
    "NOVELTY_LANGUAGE_ISSUE": ("novel", "groundbreaking"),
    "EXTERNAL_VALIDATION_OVERCLAIM": ("externally validated", "generalizes", "generalize to independently"),
    "STATISTICAL_SIGNIFICANCE_ISSUE": ("statistically significant", "significant difference", "p-value", "p value"),
    "PUBLICATION_BOUNDARY_ISSUE": ("publication-ready", "submission-ready", "field-ready", "deployment-ready", "clinically useful", "regulatory compliant"),
    "LANGUAGE_STRENGTH_ISSUE": ("excellent", "superior", "robust"),
}

NEGATION_TERMS: tuple[str, ...] = (
    "not ",
    "no ",
    "cannot ",
    "cannot yet ",
    "does not ",
    "do not ",
    "is not ",
    "are not ",
    "without ",
    "remains outstanding",
    "remain outstanding",
)

SECTION_SENTENCE_PREFIX = {
    SectionType.RESULTS: "RESULTS",
    SectionType.DISCUSSION: "DISCUSSION",
    SectionType.LIMITATIONS: "LIMITATIONS",
    SectionType.CONCLUSION: "CONCLUSION",
    SectionType.FIGURE_CAPTIONS: "FIGURE",
    SectionType.TABLE_CAPTIONS: "TABLE",
    SectionType.ABSTRACT_PLACEHOLDER: "ABSTRACT",
    SectionType.INTRODUCTION_PLACEHOLDER: "INTRODUCTION",
    SectionType.METHODS_PLACEHOLDER: "METHODS",
    SectionType.REVISION_NOTES: "REVISION",
    SectionType.TITLE: "TITLE",
}

RESULT_CATEGORY_ORDER: tuple[tuple[str, str], ...] = (
    ("DATASET", "Dataset and Analysis Scope"),
    ("QUALITY_CONTROL", "Data-quality Assessment"),
    ("FINGERPRINT", "Biosensor Fingerprint Structure"),
    ("EXPLORATORY_ANALYSIS", "Exploratory Multivariate Structure"),
    ("CLASSIFICATION", "Chemical Classification"),
    ("REGRESSION", "Concentration Regression"),
    ("FEATURE_ENGINEERING", "Temporal Feature Engineering"),
    ("FEATURE_SELECTION", "Feature Selection"),
    ("STRAIN_CONTRIBUTION", "Strain Contribution"),
    ("BLIND_PREDICTION", "Blind-prediction Boundary"),
    ("VALIDATION", "Validation Context"),
)

DISCUSSION_CATEGORY_ORDER: tuple[str, ...] = (
    "CHEMICAL_DISCRIMINATION",
    "CONCENTRATION_INFORMATION",
    "TEMPORAL_INFORMATION",
    "FEATURE_REPRESENTATION",
    "STRAIN_CONTRIBUTION",
    "DATA_QUALITY",
    "GENERALIZATION",
    "SYSTEM_LEVEL_PERFORMANCE",
)


def sentence_id(section_type: SectionType | str, index: int) -> str:
    section_type = SectionType(section_type)
    return f"SENT-{SECTION_SENTENCE_PREFIX[section_type]}-{index:04d}"


def paragraph_id(section_type: SectionType | str, index: int) -> str:
    section_type = SectionType(section_type)
    return f"PARA-{SECTION_SENTENCE_PREFIX[section_type]}-{index:04d}"


def section_id(section_type: SectionType | str, index: int = 1) -> str:
    section_type = SectionType(section_type)
    return f"SEC-{SECTION_SENTENCE_PREFIX[section_type]}-{index:04d}"


def boundary_for_claim(claim: dict[str, Any], evidence_score: dict[str, Any] | None = None) -> PublicationBoundary:
    if claim.get("claim_type") == "WITHHELD":
        return PublicationBoundary.WITHHELD
    readiness = None if evidence_score is None else evidence_score.get("publication_readiness")
    publication_use = claim.get("publication_use")
    if readiness in {"RESULTS_READY", "HIGH_CONFIDENCE_RESULTS_READY"}:
        return PublicationBoundary.RESULTS_ALLOWED
    if readiness == "DISCUSSION_READY" or publication_use == "DISCUSSION_ELIGIBLE":
        return PublicationBoundary.DISCUSSION_ONLY
    if readiness == "LIMITATION_ONLY" or publication_use == "LIMITATION_ONLY" or claim.get("claim_type") == "LIMITATION":
        return PublicationBoundary.LIMITATION_ONLY
    if readiness == "NOT_READY" or publication_use in {"INTERNAL_REVIEW_ONLY", "NOT_ELIGIBLE"}:
        return PublicationBoundary.INTERNAL_ONLY
    if publication_use == "RESULTS_ELIGIBLE":
        return PublicationBoundary.RESULTS_ALLOWED
    return PublicationBoundary.INTERNAL_ONLY


def document_status_from_review(assessment: dict[str, Any], reviewer_summary: dict[str, Any]) -> DocumentStatus:
    if assessment.get("manuscript_drafting_allowed") is not True:
        return DocumentStatus.NOT_GENERATED
    if int(reviewer_summary.get("blocking_finding_count") or 0) > 0 or assessment.get("overall_recommendation") == "NEEDS_MAJOR_REVISION":
        return DocumentStatus.REVISION_REQUIRED
    if assessment.get("overall_recommendation") == "READY_FOR_DRAFT_MANUSCRIPT":
        return DocumentStatus.SUPERVISOR_REVIEW_DRAFT
    return DocumentStatus.INTERNAL_DRAFT


def sentence_status_for_boundary(boundary: PublicationBoundary, *, section_type: SectionType) -> SentenceStatus:
    if boundary is PublicationBoundary.WITHHELD:
        return SentenceStatus.WITHHELD
    if section_type is SectionType.RESULTS and boundary is not PublicationBoundary.RESULTS_ALLOWED:
        return SentenceStatus.RESTRICTED
    if boundary in {PublicationBoundary.DISCUSSION_ONLY, PublicationBoundary.LIMITATION_ONLY, PublicationBoundary.INTERNAL_ONLY}:
        return SentenceStatus.QUALIFIED
    return SentenceStatus.ALLOWED


def traceability_status_for(
    *,
    source_ids: Iterable[str],
    placeholder: bool = False,
) -> TraceabilityStatus:
    if placeholder:
        return TraceabilityStatus.NOT_APPLICABLE
    ids = tuple(source_ids)
    if ids:
        return TraceabilityStatus.COMPLETE
    return TraceabilityStatus.MISSING


def normalize_sentence(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text.strip())
    if not collapsed:
        return collapsed
    if collapsed[-1] not in ".!?":
        collapsed += "."
    return collapsed


def qualified_claim_sentence(claim: dict[str, Any], evidence_score: dict[str, Any] | None = None) -> str:
    claim_text = str(claim.get("claim_text") or "").strip()
    if not claim_text:
        claim_text = f"Claim {claim.get('claim_id')} is recorded in the validated claim package."
    readiness = None if evidence_score is None else evidence_score.get("publication_readiness")
    uncertainty = None if evidence_score is None else evidence_score.get("uncertainty_level")
    prefix = "As a qualified Discussion claim under the current dataset, "
    body = claim_text[0].lower() + claim_text[1:] if claim_text else claim_text
    qualifiers = []
    if readiness:
        qualifiers.append(f"downstream readiness is {readiness}")
    if uncertainty:
        qualifiers.append(f"uncertainty is {uncertainty}")
    if qualifiers:
        return normalize_sentence(f"{prefix}{body}; {', '.join(qualifiers)}")
    return normalize_sentence(f"{prefix}{body}")


def limitation_sentence_from_finding(finding: dict[str, Any]) -> str:
    text = str(finding.get("finding_text") or finding.get("title") or "").strip()
    title = str(finding.get("title") or "reviewer finding").strip()
    if text:
        return normalize_sentence(f"Reviewer finding {finding.get('finding_id')} records this limitation: {text}")
    return normalize_sentence(f"Reviewer finding {finding.get('finding_id')} records a limitation titled {title}")


def evidence_gap_sentence(node: dict[str, Any]) -> str:
    text = str((node.get("attributes") or {}).get("text") or node.get("label") or "An evidence gap is recorded.")
    return normalize_sentence(f"Evidence gap {node.get('node_id')} states that {text[0].lower() + text[1:] if text else text}")


def conclusion_sentences(assessment: dict[str, Any]) -> tuple[str, ...]:
    allowed = assessment.get("definitive_generalization_allowed") is True
    generalization_clause = (
        "external validation remains outstanding for independently labelled unknown samples"
        if not allowed
        else "external-validation restrictions are recorded in the reviewer package"
    )
    return (
        "Under the current dataset and analysis design, the multistrain response profiles contain information associated with chemical-discrimination tasks.",
        "Concentration-related information remains more limited under the current representation than chemical-discrimination evidence.",
        "Temporal and strain-related summaries are suitable for qualified Discussion use under internal evaluation.",
        normalize_sentence(f"The manuscript conclusions remain restricted to the reviewed artifacts, and {generalization_clause}"),
    )


def language_issue_codes(text: str) -> tuple[str, ...]:
    lower = text.lower()
    issues = []
    for code, terms in FORBIDDEN_LANGUAGE_PATTERNS.items():
        for term in terms:
            index = lower.find(term)
            if index == -1:
                continue
            window = lower[max(0, index - 80) : index + len(term) + 30]
            if _is_negated(window):
                continue
            issues.append(code)
            break
    return tuple(sorted(set(issues)))


def number_tokens(text: str) -> tuple[str, ...]:
    return tuple(match.group(0) for match in re.finditer(r"(?<![A-Za-z])-?\d+(?:\.\d+)?(?![A-Za-z])", text))


def normalize_number_token(token: str) -> str:
    try:
        decimal = Decimal(token)
    except InvalidOperation:
        return token
    normalized = decimal.normalize()
    return format(normalized, "f").rstrip("0").rstrip(".") if "." in format(normalized, "f") else format(normalized, "f")


def source_number_tokens(records: Iterable[dict[str, Any]]) -> set[str]:
    values: set[str] = set()
    for record in records:
        text = json.dumps(record, sort_keys=True, default=str)
        for token in number_tokens(text):
            values.add(token)
            values.add(normalize_number_token(token))
    return values


def json_compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def extract_json_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    raw = str(value).strip()
    if not raw:
        return tuple()
    if raw[0] in "[{":
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return tuple(str(item) for item in parsed)
    return tuple(item.strip() for item in re.split(r"[;,|]", raw) if item.strip())


def _is_negated(window: str) -> bool:
    return any(term in window for term in NEGATION_TERMS)
